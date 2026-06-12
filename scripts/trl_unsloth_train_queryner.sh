#!/usr/bin/env bash
# TRL + unsloth + MiniCPM5 LoRA/QLoRA 微调 QueryNER
#
# 依赖通过 uv dependency group 管理，无需手动安装：
#   uv sync --group unsloth
#
# 使用：
#   bash scripts/trl_unsloth_train_queryner.sh           # LoRA bf16
#   LOAD_IN_4BIT=1 bash scripts/trl_unsloth_train_queryner.sh  # QLoRA 4-bit
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0     指定 GPU
#   BASE_MODEL=openbmb/MiniCPM5-1B  模型名或本地路径
#   LOAD_IN_4BIT=1             启用 QLoRA 4-bit（适合 ≤24GB 显卡）

set -euo pipefail

FOURBIT_FLAG=""
if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    FOURBIT_FLAG="--load_in_4bit"
fi

uv run --group unsloth main.py train queryner \
    --backend trl \
    --use_unsloth \
    ${FOURBIT_FLAG} \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --epochs 2 \
    --batch_size 4 \
    --accumulative_counts 4 \
    --lr 2e-4 \
    --warmup_ratio 0.03 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run trl_unsloth_minicpm5_train_queryner
