import os
import math
import torch
import torch.nn as nn
from deepspeed.utils import safe_get_full_grad

from transformers import Trainer, TrainerCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenTrainer(Trainer):

    def __init__(self, processor, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self.processor = processor
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        print("I got it! Maybe for future usage")
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            projection_parameters = [name for name in decay_parameters if ("_projection" in name or "query_vectors" in name or "cross_attention" in name)]
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in projection_parameters)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in projection_parameters)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.projection_layer_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in projection_parameters)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in projection_parameters)],
                        "weight_decay": 0.0,
                        "lr": self.args.projection_layer_lr,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
        # === 修复：增加对 DeepSpeed Offload 的判断 ===
        # DeepSpeed 在初始化 Engine 时才会真正创建优化器
        # 在这里如果是 DummyOptim，我们就跳过打印，或者是打印我们构建的 groups
        
        # 只要 self.optimizer 有 param_groups 属性才打印
        if hasattr(self.optimizer, "param_groups"):
            print(f"[Optimizer] {self.optimizer.__class__.__name__}")
            for i, param_group in enumerate(self.optimizer.param_groups):
                print(f"  - group {i}: lr={param_group['lr']}, "
                          f"betas={param_group.get('betas', 'N/A')}, "
                          f"eps={param_group.get('eps', 'N/A')}, "
                          f"weight_decay={param_group.get('weight_decay', 'N/A')}")
        else:
            print(f"[Optimizer] DeepSpeed Offload Enabled. Optimizer is {self.optimizer.__class__.__name__}. Skipping param_groups print.")
            # 如果你想确认分组是否正确，可以打印 optimizer_grouped_parameters
            # for i, group in enumerate(optimizer_grouped_parameters):
            #     print(f"  - group {i} (Config): lr={group.get('lr', 'default')}, weight_decay={group.get('weight_decay', 'default')}")
        
        # self.optimizer.param_groups
        # opt = self.optimizer
        # print(f"[Optimizer] {opt.__class__.__name__}")
        # for i, param_group in enumerate(self.optimizer.param_groups):
        #     print(f"  - group {i}: lr={param_group['lr']}, "
        #               f"betas={param_group.get('betas', 'N/A')}, "
        #               f"eps={param_group.get('eps', 'N/A')}, "
        #               f"weight_decay={param_group.get('weight_decay', 'N/A')}")
            
            # print(f"Param group {i}:")
            # for param in param_group["params"]:
            #     for name, p in opt_model.named_parameters():
            #         if p is param:
            #             print(f"  - {name}")
            #             break
        
        # self.optimizer.add_param_group({
        #                 "params": [p for n, p in opt_model.named_parameters() if ("query_vectors" in n)],
        #                 "weight_decay": self.args.weight_decay,
        #             })
        # import ipdb; ipdb.set_trace()
        
        return self.optimizer
    

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            # non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=True)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # transformers 5.x: _rotate_checkpoints 移除，改用模块级 rotate_checkpoints
                from transformers.trainer_utils import rotate_checkpoints
                rotate_checkpoints(
                    output_dir=run_dir,
                    save_total_limit=self.args.save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                    use_mtime=False,
                )

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=getattr(self.args, 'save_safetensors', True)
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if getattr(self.args, 'save_safetensors', True):
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=getattr(self.args, 'save_safetensors', True)
                )

            # transformers 5.x 移除了 Trainer.tokenizer（改为 processing_class）
            tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    # 
    #     return super().training_step(model, inputs)
    
    
class UnfreezeLoRACallback(TrainerCallback):
    """
    Stage 1 冻结 LoRA，只训练锚点适配器（projection/cross_attention/query_vectors）
    与新 token 的 embed/lm_head 行；到 unfreeze_step 后解冻 LoRA 进入联合训练。
    避免 Stage 1 初期随机 projector 产生的高噪声梯度冲刷基座 LoRA。
    注意：冻结必须发生在优化器/DeepSpeed engine 创建之后（on_train_begin 时机满足），
    否则 LoRA 参数会被排除在优化器外，后续解冻也无法更新。
    基于 global_step 与阈值的关系声明式对齐状态，resume 安全。
    """
    def __init__(self, unfreeze_step):
        self.unfreeze_step = unfreeze_step
        self._lora_trainable = None

    def _apply(self, model, step):
        trainable = step >= self.unfreeze_step
        if trainable == self._lora_trainable:
            return
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = trainable
        self._lora_trainable = trainable
        print(f"[UnfreezeLoRA] step={step}: LoRA {'unfrozen (joint training)' if trainable else 'frozen (anchor adapters only)'}")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._apply(model, state.global_step)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._apply(model, state.global_step)


class StepSyncCallback(TrainerCallback):
    """
    以 TrainerState.global_step（优化器步数，随 checkpoint 持久化）为唯一步数源：
    - 模型侧：每个 step 将 global_step 写入 model.global_steps（控制 loss 策略与视觉权重衰减），
      forward 与 step 同步执行，这里是精确的。
    - 数据侧：只在 epoch 开始时写入 epoch_start_step，样本的阶段由它在数据流中的
      消费位置确定性计算（见 SupervisedDataset.cur_step），与 worker 预取时序无关。
    resume_from_checkpoint 时 state.global_step / state.epoch 自动恢复，两侧随之对齐
    （mid-epoch resume 时 Trainer 的 skip_first_batches 仍会迭代数据集，位置计数保持一致）。
    """
    def __init__(self, covt_model, train_dataset):
        # 注意：这里持有的是 PEFT 包装前的 CoVT 模型引用，
        # 直接设属性避免写到 PeftModel 包装层上
        self.covt_model = covt_model
        self.train_dataset = train_dataset

    def on_train_begin(self, args, state, control, **kwargs):
        self.covt_model.global_steps = state.global_step

    def on_step_begin(self, args, state, control, **kwargs):
        self.covt_model.global_steps = state.global_step

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        # 本 epoch 起始的 optimizer step = 已完成的 epoch 数 × 每 epoch 步数
        # （on_epoch_begin 先于 dataloader iterator 创建/worker fork 触发，无竞争）
        epoch_index = int(state.epoch or 0)
        if train_dataloader is not None:
            steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            self.train_dataset.set_epoch_start_step(epoch_index * steps_per_epoch)
