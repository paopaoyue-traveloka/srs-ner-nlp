#!/usr/bin/env bash
# TRL + unsloth 评估脚本（在 test 集上评估已训练的完整 checkpoint）
#
# 依赖通过 uv dependency group 管理：
#   uv sync --group unsloth
#
# 使用：
#   bash scripts/trl_validate_unsloth_queryner.sh
#   MODEL_DIR=/path/to/checkpoint bash scripts/trl_validate_unsloth_queryner.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES=0
#   BASE_MODEL=openbmb/MiniCPM5-1B
#   MODEL_DIR=                 模型目录（默认 .model/queryner/trl_unsloth/best）
#   LOAD_IN_4BIT=1             使用 4-bit 加载

set -euo pipefail

FOURBIT_FLAG=""
if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    FOURBIT_FLAG="--load_in_4bit"
fi

uv run --group unsloth main.py validate queryner \
    --backend trl \
    --use_unsloth \
    ${FOURBIT_FLAG} \
    --base_model "${BASE_MODEL:-openbmb/MiniCPM5-1B}" \
    --split test \
    --model_dir "${MODEL_DIR:-.model/queryner/trl_unsloth/best}" \
    --wandb_run trl_unsloth_minicpm5_validate_queryner
