#!/usr/bin/env bash
# TRL + MiniCPM5 GRPO 训练 QueryNER（自定义 reward）
#
# 依赖通过 uv dependency group 管理：
#   uv sync --group trl
#
# 使用：
#   bash scripts/trl_grpo_train_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0
#   BASE_MODEL=openbmb/MiniCPM5-1B

set -euo pipefail

uv run --group trl main.py train queryner \
    --backend trl \
    --trl_mode grpo \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --epochs 1 \
    --batch_size 8 \
    --accumulative_counts 2 \
    --lr 1e-5 \
    --warmup_ratio 0.03 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --grpo_num_generations 4 \
    --grpo_max_prompt_length 1024 \
    --grpo_max_completion_length 128 \
    --grpo_beta 0.0 \
    --max_steps 80 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run trl_grpo_minicpm5_train_queryner
