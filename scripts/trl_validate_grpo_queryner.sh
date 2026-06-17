#!/usr/bin/env bash
# TRL + GRPO 评估脚本（在 test 集上评估已训练的完整 checkpoint）
#
# 依赖通过 uv dependency group 管理：
#   uv sync --group trl
#
# 使用：
#   bash scripts/trl_validate_grpo_queryner.sh
#   MODEL_DIR=/path/to/checkpoint bash scripts/trl_validate_grpo_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0
#   BASE_MODEL=openbmb/MiniCPM5-1B
#   MODEL_DIR=                 模型目录（默认 .model/queryner/trl_grpo/best）

set -euo pipefail

uv run --group trl main.py validate queryner \
    --backend trl \
    --trl_mode grpo \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --split test \
    --model_dir "${MODEL_DIR:-.model/queryner/trl_grpo/best}" \
    --wandb_run trl_grpo_minicpm5_validate_queryner
