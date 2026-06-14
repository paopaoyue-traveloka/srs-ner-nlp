#!/usr/bin/env bash
# TRL + MiniCPM5 全量训练（不使用 LoRA）
#
# 依赖通过 uv dependency group 管理，无需手动安装：
#   uv sync --group trl
#
# 使用：
#   bash scripts/trl_full_train_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0     指定 GPU
#   BASE_MODEL=openbmb/MiniCPM5-1B  模型名或本地路径

set -euo pipefail

uv run --group trl main.py train queryner \
    --backend trl \
    --full_finetune \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --epochs 2 \
    --batch_size 2 \
    --accumulative_counts 8 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --max_length 2048 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run trl_full_minicpm5_train_queryner
