#!/usr/bin/env bash
# TRL + unsloth + MiniCPM5 全量训练（不使用 LoRA）
#
# 依赖通过 uv dependency group 管理，无需手动安装：
#   uv sync --group unsloth
#
# 使用：
#   bash scripts/trl_unsloth_full_train_queryner.sh
#   LOAD_IN_4BIT=1 bash scripts/trl_unsloth_full_train_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0     指定 GPU
#   BASE_MODEL=openbmb/MiniCPM5-1B  模型名或本地路径
#   LOAD_IN_4BIT=1             启用 4-bit 量化加载（显存更省）

set -euo pipefail

FOURBIT_FLAG=""
if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    FOURBIT_FLAG="--load_in_4bit"
fi

uv run --group unsloth main.py train queryner \
    --backend trl \
    --use_unsloth \
    --full_finetune \
    ${FOURBIT_FLAG} \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --epochs 2 \
    --batch_size 2 \
    --accumulative_counts 8 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --max_length 2048 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run trl_unsloth_full_minicpm5_train_queryner
