#!/bin/bash

# Doc-SCoT: GRPO 强化学习训练（自发现视觉 token 预算）
export MASTER_PORT=${MASTER_PORT:-22811}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-DocSCoT_RL}"
export WANDB_MODE=offline

# === 基础路径 ===
# 默认取脚本所在仓库根目录，可用 BASE_DIR 覆盖
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${BASE_DIR}/train:${BASE_DIR}/train/src:${PYTHONPATH:-}"
# 解释器：默认用当前环境的 python，可设 PY 指向绝对路径
PY="${PY:-python}"

# === 模型路径 (SFT 训练好的合并模型, 用环境变量覆盖: SFT_MODEL=<run>/lora_merged) ===
SFT_MODEL="${SFT_MODEL:-${BASE_DIR}/output/lora_merged}"
if [ ! -d "$SFT_MODEL" ]; then
    echo "Error: SFT model not found at $SFT_MODEL"
    echo "       请传入 SFT_MODEL=<run_dir>/lora_merged，或把合并模型放到该路径"
    exit 1
fi
echo "SFT Model: $SFT_MODEL"

# === 数据集 (全量 train 混合集; 自发现预算模式下全对组要学"省 token", 无需难例筛选) ===
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset/CORD_SROIE_FUNSD_POIE_DocVQA_InfoVQA_VisualMRC}"
DATA_PATH="${DATA_PATH:-${DATASET_DIR}/data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATASET_DIR}/images}"

# 自适应预算表（与 SFT 同一份表）
export ANCHOR_BUDGET_FILE="${ANCHOR_BUDGET_OVERRIDE:-${DATASET_DIR}/anchor_budget.json}"

# === 输出目录 ===
RUN_DIR="${BASE_DIR}/output/rl_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/rl_log.txt"

# === GRPO 超参 ===
G=8                    # 每个样本采样轨迹数
LR=1e-6                # 学习率 (比 SFT 低 10x)
BETA=0.04              # KL 散度系数
TEMPERATURE=1.0        # 采样温度 (提高至1.0增加轨迹多样性,解决reward_std=0)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}     # 最大生成长度
MAX_STEPS=${MAX_STEPS:-500}     # 有效步数 (零方差跳过不计入)
SAVE_STEPS=${SAVE_STEPS:-100}   # 每 N 有效步存 checkpoint (含 extra_trainable.bin)

# === 预算模式（固定为"自己发现视觉 token 预算"，原 ArmB 配置，不可覆盖）===
# 模型自生成完整 CoT，梯度覆盖 think 段；token 开销惩罚 + CoT 语法奖励 +
# 对齐残差奖励共同塑形预算决策；门控保留全对组（学"该省"的唯一来源）
FORCE_COT_PREFIX=False   # 不注入 CoT 前缀，模型自己决定发多少视觉 token
GRAD_SCOPE=full          # 策略梯度覆盖 think 段（预算本身可优化）
W_TOKEN=0.05             # 视觉 token 开销惩罚权重
GATE_MODE=drop_allwrong  # 丢弃全错组，保留全对组
VA_WEIGHT=0.0            # 视觉对齐辅助 loss 关闭（对齐信号走奖励通道）
W_COT_FMT=0.2            # 变长 CoT 语法奖励权重
W_ALIGN=0.05             # 对齐残差入奖励（经 advantage 约束预算决策）
CHARGE_ABSENT=True       # 省略分支计先验残差（w_align>0 必需）

# === 启动 ===
mkdir -p "$RUN_DIR"
exec > >(tee "$LOG_FILE") 2>&1

echo "========================================"
echo "GRPO RL"
echo "========================================"
echo "SFT Model: $SFT_MODEL"
echo "Output:    $RUN_DIR"
echo "Dataset:   $DATA_PATH"
echo "G=$G, lr=$LR, beta=$BETA, temp=$TEMPERATURE, force_cot=$FORCE_COT_PREFIX"
echo "========================================"

$PY ${BASE_DIR}/train/src/training/train_rl.py \
    --model_path "$SFT_MODEL" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$RUN_DIR" \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --temperature $TEMPERATURE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --batch_size 1 \
    --seed 42 \
    --force_cot_prefix $FORCE_COT_PREFIX \
    --grad_scope $GRAD_SCOPE \
    --w_token $W_TOKEN \
    --gate_mode $GATE_MODE \
    --visual_align_weight $VA_WEIGHT \
    --w_cot_fmt $W_COT_FMT \
    --w_align $W_ALIGN \
    --charge_absent_branches $CHARGE_ABSENT

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "==== RL 训练失败 ===="
    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo "RL 训练完成！"
echo "========================================"
echo "输出: $RUN_DIR"
echo "日志: $LOG_FILE"
