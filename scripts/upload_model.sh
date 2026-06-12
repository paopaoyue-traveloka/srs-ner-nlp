#!/usr/bin/env bash
# 上传模型目录到 WandB Artifacts
#
# 使用：
#   bash scripts/upload_model.sh .model/queryner/trl/best
#   bash scripts/upload_model.sh .model/queryner/trl/best --artifact_name queryner-trl-v1
#
# 环境变量（可选）：
#   WANDB_PROJECT=ner-finetune     WandB 项目名
#   WANDB_ENTITY=my-team           WandB 团队/用户名

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "用法: bash scripts/upload_model.sh <model_dir> [--artifact_name NAME] [--wandb_project PROJECT]"
    exit 1
fi

uv run main.py upload-model "$@"
