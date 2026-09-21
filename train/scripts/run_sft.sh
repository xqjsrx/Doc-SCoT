#!/bin/bash
# SFT 训练 + 合并 LoRA（自适应视觉 token 预算版）
# 环境变量（均有默认值，可手动指定覆盖）:
#   RUN_TAG(输出标识, 默认 sft) GPUS(默认 0,1) PORT(默认 22850)
#   EXTRA_ARGS(追加训练参数) ANCHOR_BUDGET_OVERRIDE(换预算表)
set -u
export MASTER_PORT=${PORT:-22850}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
RUN_TAG=${RUN_TAG:-sft}
# 减少显存碎片化（可回收 reserved-but-unallocated 的残留块）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PYTHONPATH 在 BASE_DIR 定义后设置
export WANDB_PROJECT="${WANDB_PROJECT:-DocSCoT_Phase2}"
export WANDB_MODE=offline

# === 基础路径 ===
# 默认取脚本所在仓库根目录，可用 BASE_DIR 覆盖
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${BASE_DIR}/train:${BASE_DIR}/train/src:${PYTHONPATH:-}"
MODEL_NAME="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_PY="${BASE_DIR}/train/src/training/train.py"
MERGE_PY="${BASE_DIR}/train/src/merge_lora_weights.py"
DS_CONFIG="${BASE_DIR}/train/scripts/zero2.json"
# 解释器：默认用当前环境的 python，可设 PY 指向绝对路径
PY="${PY:-python}"
PY_BIN="$(dirname "$(command -v "$PY")")"

# === 数据集 (train 混合集, 四件套同目录) ===
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset/CORD_SROIE_FUNSD_POIE_DocVQA_InfoVQA_VisualMRC}"
DATA_PATH="${DATA_PATH:-${DATASET_DIR}/data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATASET_DIR}/images}"
# teacher 离线预计算缓存（tool/build_dataset_cache.py 产出；缺失文件自动回退在线推理）
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-${DATASET_DIR}/teacher_cache}"

# === 自适应视觉 token 预算 ===
# 数量随文档复杂度变（k∈{0,2,4,6}），k=0 的步骤整步省略（分支级选择）
# 预算表由 tool/build_anchor_budget.py 生成，可被 ANCHOR_BUDGET_OVERRIDE 覆盖
export ANCHOR_BUDGET_FILE="${ANCHOR_BUDGET_OVERRIDE:-${DATASET_DIR}/anchor_budget.json}"

# === 输出目录 (带时间戳) ===
RUN_DIR="${BASE_DIR}/output/$(date +%Y%m%d_%H%M%S)_${RUN_TAG}"
OUTPUT_DIR="${RUN_DIR}/lora_vision_test"
SAVE_MODEL_PATH="${RUN_DIR}/lora_merged"
LOG_FILE="${RUN_DIR}/train_log.txt"

# === 训练配置 ===
# 全局 batch = BATCH × ACCUM × 卡数 = 8
BATCH_PER_DEVICE=4
GRAD_ACCUM_STEPS=1

# === 阶段步数 (均为优化器步数，无需任何换算) ===
MAX_STEPS=${MAX_STEPS:-16000}
STAGE_1_STEPS=${STAGE_1_STEPS:-2000}

# === 视觉权重衰减 (Stage 2 的 1/5~4/5 线性衰减) ===
LINEAR_VISUAL_WEIGHT_DECAY=True
VISUAL_WEIGHT_START=0.2
VISUAL_WEIGHT_END=0.05

VISUAL_MODEL_ID="['det', 'layout', 'flow']"

# === 启动 ===
mkdir -p "$RUN_DIR"
exec > >(tee "$LOG_FILE") 2>&1

echo "========================================"
echo "Doc-SCoT SFT: $RUN_TAG"
echo "========================================"
echo "输出目录: $RUN_DIR"
echo "数据集:   $DATASET_DIR"
echo "预算表:   $ANCHOR_BUDGET_FILE"
echo "Batch:    $((BATCH_PER_DEVICE * GRAD_ACCUM_STEPS * $(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")))  (GPU $CUDA_VISIBLE_DEVICES)"
echo "Steps:    $MAX_STEPS (Stage1=$STAGE_1_STEPS)"
echo "========================================"

$PY_BIN/deepspeed \
    --master_port $MASTER_PORT \
    $TRAIN_PY \
    --deepspeed $DS_CONFIG \
    --use_liger False \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['embed_tokens', 'lm_head', 'det', 'layout', 'flow']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((1024 * 32 * 32)) \
    --image_resized_width 512 \
    --image_resized_height 512 \
    --learning_rate 5e-5 \
    --projection_layer_lr 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name "doc_covt3_${RUN_TAG}" \
    --anchor_model_id "$VISUAL_MODEL_ID" \
    --training_stage "full" \
    --stage_1_steps $STAGE_1_STEPS \
    --teacher_cache_dir "$TEACHER_CACHE_DIR" \
    --linear_visual_weight_decay $LINEAR_VISUAL_WEIGHT_DECAY \
    --visual_weight_start $VISUAL_WEIGHT_START \
    --visual_weight_end $VISUAL_WEIGHT_END \
    ${EXTRA_ARGS:-}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "==== 训练失败 ===="
    exit $TRAIN_EXIT_CODE
fi

echo "==== 训练完成，开始合并 ===="

$PY $MERGE_PY \
    --model-path "$OUTPUT_DIR" \
    --model-base "$MODEL_NAME" \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization \
    --anchor-model-id "$VISUAL_MODEL_ID"

MERGE_EXIT_CODE=$?
if [ $MERGE_EXIT_CODE -ne 0 ]; then
    echo "==== LoRA合并失败 (exit=$MERGE_EXIT_CODE) ===="
    exit $MERGE_EXIT_CODE
fi

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo "训练: $OUTPUT_DIR"
echo "模型: $SAVE_MODEL_PATH"
echo "日志: $LOG_FILE"
