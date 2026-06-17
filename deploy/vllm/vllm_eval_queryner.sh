#!/usr/bin/env bash
# 评估 vLLM 服务上的模型（可用于 BF16/FP8 对比）
#
# 用法：
#   MODEL_NAME=.model/queryner/trl_standard/best bash deploy/vllm/vllm_eval_queryner.sh

set -euo pipefail

if [ -z "${MODEL_NAME:-}" ]; then
    echo "错误: 请设置 MODEL_NAME"
  echo "示例: MODEL_NAME=.model/queryner/trl_standard/best bash deploy/vllm/vllm_eval_queryner.sh"
    exit 1
fi

uv run --group trl python deploy/vllm/vllm_eval_queryner.py \
  --endpoint "${ENDPOINT:-http://127.0.0.1:8000}" \
  --model "$MODEL_NAME" \
  --split "${SPLIT:-test}" \
  --n "${N:-0}" \
  --max_in_flight "${MAX_IN_FLIGHT:-128}" \
  --max_tokens "${MAX_TOKENS:-20}" \
  --temperature "${TEMPERATURE:-0}" \
  --out_csv "${OUT_CSV:-.model/eval/vllm_queryner_eval.csv}" \
  --summary_json "${SUMMARY_JSON:-.model/eval/vllm_queryner_eval_summary.json}"
