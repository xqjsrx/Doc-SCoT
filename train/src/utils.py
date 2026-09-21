import os
import sys

# sys.path 自举：本文件在 train/src/ 下。队列内联命令直接跑 merge_lora_weights.py 时
# 不带 PYTHONPATH，这里必须自给自足，否则 'from src.training...' 直接 ModuleNotFoundError
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, AutoConfig
import warnings
import json

# covt_qwen3_vl 顶部已处理 doctr/huggingface_hub 兼容 shim
from src.training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from src.training.constants import *

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, anchor_model_id=None, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    # RL checkpoint 目录名为 checkpoint-N，按名字判会漏；有 adapter_config.json 即 LoRA 分支
    is_lora = 'lora' in model_name.lower() or os.path.exists(
        os.path.join(model_path, 'adapter_config.json'))
    if is_lora and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if is_lora and model_base is not None:
        try:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        except Exception:
            # RL checkpoint 不含 config.json，架构配置与底座一致
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_base)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            print('Loading Processor from model path...')
        except:
            # find the sub-dir of /checkpoint-xxxx, use the max number of the sub-dir
            checkpoint_dir = max([d for d in os.listdir(model_path) if d.startswith('checkpoint-')], key=lambda x: int(x.split('-')[1]))
            processor = AutoProcessor.from_pretrained(model_path + "/" + checkpoint_dir)
            print(f'Loading Processor from model path + {checkpoint_dir}...')
        print('Loading Qwen3-VL from base model...')
        # query bank 行数随训练配置变化（4/8）；非 LoRA 权重随后会覆盖这些参数，
        # 底座加载阶段的形状不一致不应直接崩
        model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
            model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
            ignore_mismatched_sizes=True, **kwargs)
        
        if anchor_model_id is not None:
            print(f"Loading anchor model ids: {anchor_model_id}")
            model.get_anchor_model_ids(anchor_model_id)
        
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.get_input_embeddings().weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Qwen3-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        # strict=False 只容忍 key 缺失/多余，不容忍形状冲突（如 4 行 vs 8 行 query bank）；
        # 形状对不上的跳过并保留底座权重
        own_state = model.state_dict()
        for k in [k for k, v in non_lora_trainables.items()
                  if k in own_state and own_state[k].shape != v.shape]:
            print(f'  skip mismatched {k}: ckpt {tuple(non_lora_trainables[k].shape)} '
                  f'vs model {tuple(own_state[k].shape)}')
            del non_lora_trainables[k]
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = CoVTQwen3VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]