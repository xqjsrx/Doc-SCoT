import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, HfArgumentParser
from training.trainer import QwenTrainer, UnfreezeLoRACallback, StepSyncCallback
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib

from training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from training.constants import *
from deepspeed import zero

# torch 2.5.1 + transformers 5.x: check_torch_load_is_safe 会拒绝一切 torch.load
# （要求 torch>=2.6）。续训只加载本地自产的 optimizer/scheduler/rng checkpoint，
# 属可信文件，这里将该检查置空
import transformers.utils.import_utils as _tf_import_utils
import transformers.trainer as _tf_trainer
_tf_import_utils.check_torch_load_is_safe = lambda: None
if hasattr(_tf_trainer, "check_torch_load_is_safe"):
    _tf_trainer.check_torch_load_is_safe = lambda: None

# rng_state.pth 内含 numpy/dill 序列化对象，weights_only=True 需显式允许
def _allow_rng_state_globals():
    import numpy as _np
    import torch.serialization as _ts
    _cand = [_np.ndarray, _np.dtype, _np.dtypes.UInt32DType]
    try:
        from numpy.core.multiarray import _reconstruct
        _cand.append(_reconstruct)
    except ImportError:
        pass
    try:
        import dill._dill as _dd
        _cand.append(_dd._create_array)
    except ImportError:
        pass
    _ts.add_safe_globals(_cand)
_allow_rng_state_globals()

local_rank = None

# set seed 42
torch.manual_seed(42)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad
        
def set_anchor_requires_grad(model, anchor_model_id):
    if "det" in anchor_model_id:
        set_requires_grad(model.det_projection.parameters(), True)
        set_requires_grad(model.det_cross_attention.parameters(), True)
        model.det_query_vectors.requires_grad = True
    if "layout" in anchor_model_id:
        set_requires_grad(model.layout_projection.parameters(), True)
        set_requires_grad(model.layout_cross_attention.parameters(), True)
        model.layout_query_vectors.requires_grad = True
    if "flow" in anchor_model_id:
        set_requires_grad(model.flow_projection.parameters(), True)
        set_requires_grad(model.flow_cross_attention.parameters(), True)
        model.flow_query_vectors.requires_grad = True



def configure_vision_tower(model, training_args, compute_dtype, device):
    # Qwen3-VL: 视觉塔挂在 Qwen3VLModel 内部（model.model.visual），Qwen2.5-VL 时代是 model.visual
    vision_tower = model.model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)

    # Handle merger specifically
    set_requires_grad(vision_tower.merger.parameters(), training_args.tune_merger)
    
def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    anchor_model_id = ast.literal_eval(model_args.anchor_model_id)
    
    if model_args.model_path is None:
        # 单阶段训练的正常路径：权重与 processor 同源
        model_args.model_path = model_args.model_id
        rank0_print(f"model_path not provided, loading weights from model_id: {model_args.model_id}")
    
    # Qwen3-VL 路线：liger 0.5.5 不兼容 transformers 5.x 且无 qwen3_vl 支持，
    # 不做 monkey patch，text loss 走普通 CE（use_flce_text_loss 恒 False）
    if training_args.use_liger:
        rank0_print("use_liger=True ignored: liger kernel is unavailable for Qwen3-VL, falling back to plain CE")
        training_args.use_liger = False

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path,
        dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
        **bnb_model_from_pretrained_args
    )
    
    model.get_anchor_model_ids(anchor_model_id)
    # teacher 离线预计算缓存（命中则跳过 DBNet/YOLO/LayoutReader 在线推理）
    model.anchor_models.teacher_cache_dir = training_args.teacher_cache_dir
    if training_args.teacher_cache_dir:
        print(f'Teacher cache: {training_args.teacher_cache_dir}')
    # === 监督信号增强 ===
    model.prefix_supervision = training_args.prefix_supervision
    model.prefix_fracs = tuple(float(x) for x in training_args.prefix_fracs.split(',') if x.strip())
    model.prefix_weight = training_args.prefix_weight
    model.charge_absent_branches = training_args.charge_absent_branches
    if training_args.prefix_supervision:
        print(f'Prefix supervision: fracs={model.prefix_fracs}, weight={model.prefix_weight}')
    if training_args.charge_absent_branches:
        print('Absent-branch charging: omitted branches pay the prior-probe residual')
    # === 阶段阈值：统一以优化器步数为单位 ===
    # global_steps 由 StepSyncCallback 每步写入 TrainerState.global_step（resume 安全），
    # 因此阈值直接用 stage_1_steps / max_steps，无需再按 batch/accum 换算
    model.external_step_control = True
    # SFT 用 Liger FLCE 计算 text loss，不物化全词表 logits（RL/推理路径不受影响）
    model.use_flce_text_loss = training_args.use_liger
    model.align_anchor_task_only_stage = training_args.stage_1_steps
    print(f'Stage 1 ends at step: {training_args.stage_1_steps}')

    model.total_training_steps = training_args.max_steps
    print(f'Total training steps: {training_args.max_steps}')

    # 视觉权重衰减：Stage 2 的 1/5 处开始，4/5 处结束
    model.linear_visual_weight_decay = training_args.linear_visual_weight_decay
    model.visual_weight_start = training_args.visual_weight_start
    model.visual_weight_end = training_args.visual_weight_end
    print(f'linear_visual_weight_decay: {training_args.linear_visual_weight_decay}')
    if training_args.linear_visual_weight_decay:
        stage2_duration = training_args.max_steps - training_args.stage_1_steps
        decay_start = training_args.stage_1_steps + int(stage2_duration * 0.2)
        decay_end = training_args.stage_1_steps + int(stage2_duration * 0.8)
        print(f'Decay: {training_args.visual_weight_start} -> {training_args.visual_weight_end}, '
              f'step {decay_start} -> {decay_end}')

    model.config.use_cache = False
    model_to_configure = model
    # 保留 PEFT 包装前的 CoVT 模型引用，供 StepSyncCallback 直接写 global_steps
    covt_model = model
    configure_llm(model_to_configure, training_args)
    if "Qwen" in model_args.model_id:
        configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    
    # Set requires_grad for the Anchor projection layers
    set_anchor_requires_grad(model, anchor_model_id)
    
    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # 必须用非重入 checkpoint：reentrant 实现在段内输入均不带梯度时会跳过整段反向，
        # 导致 ViT blocks（输入 pixel_values 无梯度）上的 vision LoRA 永远收不到梯度
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)
        
        for name, param in model.named_parameters():
            if '_projection' in name:
                param.requires_grad = True
            if 'cross_attention' in name:
                param.requires_grad = True
            if '_query_vectors' in name:
                param.requires_grad = True
        # model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                            # The default setting is padding_side="left"
                                            # When training using the right-side padding is more efficient.
                                              padding_side="right")

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr
    
    old_processor_len = len(processor.tokenizer)
        
    # add special tokens - only add the tokens specified in anchor_model_id
    add_tokens = []

    _indexed = os.environ.get("ANCHOR_INDEXED_TOKENS", "0") == "1"
    if 'det' in anchor_model_id:
        add_tokens.append(DET_PAD_TOKEN)
        if _indexed:
            add_tokens.extend(DET_SLOT_TOKENS)
    if 'layout' in anchor_model_id:
        add_tokens.append(LAYOUT_PAD_TOKEN)
        if _indexed:
            add_tokens.extend(LAYOUT_SLOT_TOKENS)
    if 'flow' in anchor_model_id:
        add_tokens.append(FLOW_PAD_TOKEN)
        if _indexed:
            add_tokens.extend(FLOW_SLOT_TOKENS)

    # Add other special tokens
    add_tokens.extend(["<think>", "</think>", "<answer>", "</answer>"])

    processor.tokenizer.add_special_tokens({"additional_special_tokens": [ANCHOR_START_TOKEN, ANCHOR_END_TOKEN]})
    processor.tokenizer.add_tokens(add_tokens)
    # Get token indices only for tokens that were actually added to the vocabulary
    det_token_idx = processor.tokenizer(DET_PAD_TOKEN, add_special_tokens=False).input_ids[0] if 'det' in anchor_model_id else None
    layout_token_idx = processor.tokenizer(LAYOUT_PAD_TOKEN, add_special_tokens=False).input_ids[0] if 'layout' in anchor_model_id else None
    flow_token_idx = processor.tokenizer(FLOW_PAD_TOKEN, add_special_tokens=False).input_ids[0] if 'flow' in anchor_model_id else None

    think_idx = processor.tokenizer("<think>", add_special_tokens=False).input_ids[0]
    splash_think_idx = processor.tokenizer("</think>", add_special_tokens=False).input_ids[0]
    answer_idx = processor.tokenizer("<answer>", add_special_tokens=False).input_ids[0]
    splash_answer_idx = processor.tokenizer("</answer>", add_special_tokens=False).input_ids[0]

    qwen_embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    p = qwen_embed.weight
    if hasattr(p, 'ds_id'):
        with zero.GatheredParameters([p]):
            old_len = p.data.shape[0]
    else:
        old_len = p.data.shape[0]
    new_len = len(processor.tokenizer)

    model.get_anchor_token_idx(det_token_idx, layout_token_idx, flow_token_idx)

    # 索引槽位模式：注入各分支 8 个槽位 token 的 id。
    # 必须设在未包装的 covt_model 上——PeftModel 的属性赋值不会转发到内层模型，
    # 只有方法调用（如 get_anchor_token_idx）才经 __getattr__ 转发。
    if _indexed:
        for _br, _toks in ANCHOR_SLOT_TOKENS.items():
            if _br in anchor_model_id:
                _ids = [processor.tokenizer(t, add_special_tokens=False).input_ids[0] for t in _toks]
                setattr(covt_model, f"{_br}_slot_token_ids", torch.tensor(_ids, dtype=torch.long))
                print(f"[Train] {_br} 索引槽位已启用: {len(_ids)} 个 token, id {min(_ids)}~{max(_ids)}")
    
    for n, p in model.named_parameters():
        if any(
            [
                "embed_tokens" in n
            ]
        ):
            p.requires_grad = True
            
        if any(
            [
                "lm_head" in n
            ]
        ):
            p.requires_grad = True
    
    _mask = torch.ones(old_len, device=model.device, dtype=torch.bool)
    _mask[:] = False
    _mask[old_processor_len:new_len] = True

    def row_mask_hook(grad):
        if grad is None:
            return grad
        return grad * _mask.to(grad.device).view(-1, 1)
    
    model.get_input_embeddings().weight.register_hook(row_mask_hook)
    model.get_output_embeddings().weight.register_hook(row_mask_hook)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              anchor_model_id=anchor_model_id,
                                              stage_1_steps=training_args.stage_1_steps,
                                              per_device_batch_size=training_args.per_device_train_batch_size,
                                              grad_accum_steps=training_args.gradient_accumulation_steps)
    
    # Stage 1 冻结 LoRA（只训锚点适配器与新 token 行），stage_1_steps 后解冻进入联合训练
    callbacks = [StepSyncCallback(covt_model, data_module["train_dataset"])]
    if training_args.lora_enable and training_args.stage_1_steps > 0:
        callbacks.append(UnfreezeLoRACallback(unfreeze_step=training_args.stage_1_steps))

    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=training_args,
        callbacks=callbacks,
        **data_module
    )
    # model.print_trainable_parameters()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
            
    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        #     model.named_parameters(), require_grad_only=False
        # )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )


        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()