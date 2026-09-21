"""
Doc-SCoT: GRPO 强化学习训练入口

加载 SFT 训练好的模型，用 GRPO 进行 RL 微调
"""

import os
import torch
# 在导入 heavy 模块(doctr/ultralytics 等)之前先初始化 CUDA，
# 否则沙箱环境下这些库的隐式 CUDA 初始化会触发 Error 304
torch.zeros(1).cuda()
import copy
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoProcessor, HfArgumentParser
from torch.utils.data import DataLoader

from training.rl_trainer import GRPOTrainer, RLDataset
from training.rl_reward import compute_reward
from training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from training.constants import *


@dataclass
class RLArguments:
    # 模型路径
    model_path: str = field(metadata={"help": "SFT trained model path (lora_merged)"})
    # 数据路径
    data_path: str = field(metadata={"help": "Training data JSON path"})
    image_folder: str = field(metadata={"help": "Image folder path"})
    output_dir: str = field(default="./rl_output", metadata={"help": "Output directory"})

    # GRPO 超参
    G: int = field(default=8, metadata={"help": "Number of trajectories per sample"})
    lr: float = field(default=1e-6, metadata={"help": "Learning rate"})
    beta: float = field(default=0.04, metadata={"help": "KL divergence coefficient"})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature"})
    max_new_tokens: int = field(default=128, metadata={"help": "Max new tokens for generation"})

    # 训练控制
    max_steps: int = field(default=100, metadata={"help": "Max training steps"})
    batch_size: int = field(default=1, metadata={"help": "Batch size (samples per step)"})
    save_steps: int = field(default=50, metadata={"help": "Save checkpoint every N steps"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

    # Anchor 配置
    anchor_model_id: str = field(default="['det', 'layout', 'flow']")

    # CoT 前缀开关（固定 False：不注入前缀，模型自己决定发多少视觉 token）
    force_cot_prefix: bool = field(default=False, metadata={"help": "True=注入CoT前缀+跳过R_format, False=模型自生成CoT+计算R_format"})

    # === 预算模式固定为"自己发现视觉 token 预算"（原 Arm B 配置）===
    # 'answer'=仅答案段算策略梯度（预算被冻结）; 'full'=含 <think> 段（预算可被优化）
    grad_scope: str = field(default="full", metadata={"help": "Policy-gradient span: answer | full"})
    # 视觉 token 开销权重；需远小于答案项，否则会把预算直接压到 0
    w_token: float = field(default=0.05, metadata={"help": "Weight of visual-token cost in reward."})
    token_cap: int = field(default=18, metadata={"help": "Normalizer for the token-cost term."})
    # 'mixed'=仅训跨界组; 'drop_allwrong'=保留全对组(学"该省"的唯一来源); 'off'=不门控
    gate_mode: str = field(default="drop_allwrong", metadata={"help": "Group gating: mixed | drop_allwrong | off"})
    # 分支省略会跳过该分支 anchor loss，使"少发 token"成为降低 va 的捷径；
    # 研究预算时应置 0，把 va 退为监控量，避免污染结论
    visual_align_weight: float = field(default=0.0, metadata={"help": "Weight of visual alignment aux loss."})
    # 变长 CoT 的语法约束权重（标签顺序 / token 归位 / 分支容量）
    w_cot_fmt: float = field(default=0.2, metadata={"help": "Weight of adaptive-CoT grammar reward."})
    # 对齐残差入奖励（与 w_token 反向，均衡点即真实预算）；>0 时每步多一轮 no_grad forward
    w_align: float = field(default=0.05, metadata={"help": "Weight of visual alignment residual in the reward."})
    # 省略分支计价；w_align>0 时必开，否则整步省略残差恒为 0，"全省"在奖励里免费
    charge_absent_branches: bool = field(default=True, metadata={"help": "Charge omitted branches the prior-probe residual."})


def main():
    parser = HfArgumentParser(RLArguments)
    args = parser.parse_args_into_dataclasses()[0]

    torch.manual_seed(args.seed)

    print("=" * 50)
    print("Doc-SCoT GRPO Reinforcement Learning")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data:  {args.data_path}")
    print(f"G={args.G}, lr={args.lr}, beta={args.beta}, temp={args.temperature}")
    print(f"force_cot_prefix={args.force_cot_prefix}")
    print("=" * 50)

    # === 1. 加载 Processor ===
    print("\n[1/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # === 2. 加载策略模型 (SFT checkpoint) ===
    print("[2/5] Loading policy model...")
    import ast
    anchor_model_id = ast.literal_eval(args.anchor_model_id)

    policy_model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # 历史 checkpoint 与当前代码的训练专用模块维度可能不一致;
        # 忽略尺寸不匹配, 相应 projection/query 重新初始化后随训练适应
        ignore_mismatched_sizes=True,
    )
    policy_model.get_anchor_model_ids(anchor_model_id)

    # 注入 token idx
    det_id = processor.tokenizer.convert_tokens_to_ids(DET_PAD_TOKEN)
    layout_id = processor.tokenizer.convert_tokens_to_ids(LAYOUT_PAD_TOKEN)
    flow_id = processor.tokenizer.convert_tokens_to_ids(FLOW_PAD_TOKEN)
    policy_model.get_anchor_token_idx(
        det_token_idx=det_id, layout_token_idx=layout_id, flow_token_idx=flow_id,
    )

    # === 设置训练阶段参数 (RL 阶段) ===
    # from_pretrained 会将 global_steps 重置为 0，需配置阶段参数匹配 RL 训练规模
    # visual_weight 暂时保持不变：禁用线性衰减，恒定使用 visual_weight_start
    policy_model.align_anchor_task_only_stage = 0       # RL 直接进入 Stage 2 联合推理
    policy_model.total_training_steps = 999999           # 足够大，确保 visual loss 始终计算
    policy_model.linear_visual_weight_decay = False      # 禁用衰减，visual_weight 恒定
    policy_model.visual_weight_start = 0.2
    policy_model.visual_weight_end = 0.2                  # start=end，即使启用衰减也恒定
    print(f"  RL stage config: anchor_task_only=0, total_steps=999999")
    print(f"  visual_weight=0.2 (fixed, decay disabled)")

    # charge_absent 是 python 属性，from_pretrained 不恢复，必须显式重设
    policy_model.charge_absent_branches = args.charge_absent_branches or bool(args.w_align)
    if args.w_align and not args.charge_absent_branches:
        print("  [注意] w_align>0 自动开启 charge_absent_branches，否则省略分支残差为 0")
    print(f"  decode: charge_absent={policy_model.charge_absent_branches} "
          f"(query bank={policy_model.det_query_vectors.shape[0]})")

    policy_device = 'cuda:0'
    policy_model = policy_model.to(policy_device)

    # 开启 gradient checkpointing (use_reentrant=False 与 RL 梯度图兼容)
    if hasattr(policy_model, 'gradient_checkpointing_enable'):
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Gradient checkpointing enabled (use_reentrant=False)")

    # === 3. 参考模型: 使用 LoRA disable 技巧,无需加载单独的 ref 模型 ===
    # 通过 policy.disable_adapter() 即可得到 ref 模型的输出,节省 ~14GB 显存
    print("[3/5] Using LoRA disable trick for reference model (no separate ref model)")

    # === 4. 设置可训练参数 ===
    # RL 需要梯度通过 answer token 反向传播到视觉 token 的隐状态
    # 因此需要 LLM 的部分参数可训练（用 LoRA 减少显存）
    print("[4/5] Setting trainable parameters...")
    from peft import LoraConfig, get_peft_model

    # 先冻结所有参数
    for param in policy_model.parameters():
        param.requires_grad = False

    # 对 LLM 的 attention 和 MLP 层添加 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
        bias="none",
    )
    policy_model = get_peft_model(policy_model, lora_config)

    # 同时保持 projection/cross_attention/query_vectors 可训练（视觉对齐）
    # 以及 embed_tokens/lm_head 可训练（log_prob 计算需要梯度）
    trainable_keywords = ['_projection', 'cross_attention', '_query_vectors',
                          'embed_tokens', 'lm_head']
    for name, param in policy_model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True

    trainable_count = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_count / 1e6:.1f}M (LoRA + embed + lm_head + anchor)")

    # === 5. 加载数据 & 初始化 GRPO ===
    print("[5/5] Loading data and initializing GRPO trainer...")
    dataset = RLDataset(args.data_path, args.image_folder)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    print(f"  Dataset size: {len(dataset)}")

    config = {
        'G': args.G,
        'lr': args.lr,
        'beta': args.beta,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        # 作用于 forward 返回的纯视觉对齐 loss (不含 text_loss)。
        # 0.02 = 旧版有效视觉系数 (va_weight 0.1 × forward 内部 visual_weight 0.2)，
        # 保持修复前后视觉梯度尺度一致
        'visual_align_weight': args.visual_align_weight,
        'force_cot_prefix': args.force_cot_prefix,
        'grad_scope': args.grad_scope,
        'w_token': args.w_token,
        'token_cap': args.token_cap,
        'gate_mode': args.gate_mode,
        'w_cot_fmt': args.w_cot_fmt,
        'w_align': args.w_align,
        # 周期性 checkpoint 由 trainer 在 train_epoch 内按有效步触发
        'save_steps': args.save_steps,
        'output_dir': args.output_dir,
    }
    trainer = GRPOTrainer(policy_model, None, processor, config)

    # === 训练循环 ===
    print("\n" + "=" * 50)
    print("Starting GRPO training...")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    all_logs = []

    for epoch in range(999):  # 由 max_steps 控制停止
        logs = trainer.train_epoch(dataloader, max_steps=args.max_steps - trainer.step_count)
        all_logs.extend(logs)

        if trainer.step_count >= args.max_steps:
            break
        # 周期性 checkpoint 已移入 train_epoch (旧版在此处检查,
        # 单 epoch 内永远不触发, 导致 checkpoint-50 从未保存)

    # 保存最终模型 (含非 LoRA 可训练参数)
    print(f"\nSaving final model...")
    trainer.save_checkpoint("final")

    # 打印统计
    if all_logs:
        rewards = [l['mean_reward'] for l in all_logs]
        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"  Total steps: {trainer.step_count}")
        print(f"  Final reward: {rewards[-1]:.3f}")
        print(f"  Best reward:  {max(rewards):.3f}")
        print(f"  Mean reward:  {sum(rewards)/len(rewards):.3f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
