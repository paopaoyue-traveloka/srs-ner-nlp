#!/usr/bin/env bash
# TRL + MiniCPM5 评估脚本（在 test 集上评估已训练的完整 checkpoint）
#
# 依赖通过 uv dependency group 管理，无需手动安装：
#   uv sync --group trl
#
# 使用：
#   bash scripts/trl_validate_queryner.sh
#   MODEL_DIR=/path/to/checkpoint bash scripts/trl_validate_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0     指定 GPU
#   BASE_MODEL=openbmb/MiniCPM5-1B  模型名或本地路径
#   MODEL_DIR=                 模型目录（默认 .model/queryner/trl_standard/best）

set -euo pipefail

uv run --group trl main.py validate queryner \
    --backend trl \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --split test \
    --model_dir "${MODEL_DIR:-.model/queryner/trl_standard/best}" \
    --wandb_run trl_minicpm5_validate_queryner
