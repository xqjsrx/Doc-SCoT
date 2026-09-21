import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default=os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"))
    # 权重加载路径，缺省时回退到 model_id。
    # 仅多阶段训练时需单独指定（如指向上一阶段的 lora_merged 产物，
    # 而 model_id 仍指向原始基座以加载 processor 与家族分支判断）
    model_path: Optional[str] = field(default=None)
    anchor_model_id: str = field(default=None, metadata={"help": "List of anchor model ids"})

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True
    
    # Anchor Model New Parameters
    training_stage: str = field(default="full", metadata={"help": "Training stage, should be one of `start` or `full`."})
    projection_layer_lr: Optional[float] = None
    # 唯一的阶段参数：以优化器步数 (TrainerState.global_step) 为单位，
    # 由 StepSyncCallback 同步到模型与数据集，无需再按 batch/accum 换算
    stage_1_steps: int = field(default=2000, metadata={"help": "Stage 1 end (optimizer steps). Stage 2 runs until max_steps."})
    # teacher 离线预计算缓存目录（tool/build_dataset_cache.py 产出）；
    # 空则全程在线推理，部分命中时未命中样本自动回退在线
    teacher_cache_dir: Optional[str] = field(default=None, metadata={"help": "Precomputed teacher feature cache dir."})
    # === 监督信号增强（详见 covt_qwen3_vl._branch_visual_loss）===
    # 前缀嵌套监督：额外用前 ceil(frac*k) 个 token 再解码一次。原监督把 k 个 token
    # 池化后重建整图，token 之间无分工压力（诊断已观测到分支可互换），且 loss 与 k 几乎
    # 无关，无法从模型本身导出最优预算。
    prefix_supervision: bool = field(default=False, metadata={"help": "Nested prefix supervision: also decode with the first ceil(frac*k) anchor tokens."})
    prefix_fracs: str = field(default="0.5", metadata={"help": "Comma-separated prefix fractions for nested supervision."})
    prefix_weight: float = field(default=0.5, metadata={"help": "Weight of each prefix variant relative to the full-length term."})
    # 缺席分支先验计价：k=0 时用固定 query bank 当解码权重，其残差即整步省略的信息代价。
    # 不开时被省略分支 loss 恒为 0，"全部省掉"是免费最优解，RL 侧也无法给省略定价。
    charge_absent_branches: bool = field(default=False, metadata={"help": "Charge omitted branches the residual of an image-independent prior probe."})

    # === 视觉权重衰减 ===
    # 衰减区间: Stage 2 的 1/5 处开始，4/5 处结束
    linear_visual_weight_decay: bool = field(
        default=True,
        metadata={"help": "Enable linear visual weight decay. If False, use constant visual_weight_start."}
    )
    visual_weight_start: float = field(
        default=0.2,
        metadata={"help": "Visual loss weight before decay starts."}
    )
    visual_weight_end: float = field(
        default=0.0,
        metadata={"help": "Visual loss weight after decay ends."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    fps: float = 1.0
    