#!/usr/bin/env bash
# QueryNER 采样请求压测 vLLM（固定 QPS，记录完整返回 E2E 延迟）
#
# 用法：
#   MODEL_NAME=.model/queryner/trl_standard/best bash scripts/vllm_benchmark_queryner.sh
#
# 可选环境变量：
#   ENDPOINT=http://127.0.0.1:8000
#   SPLIT=test
#   REQUESTS=200
#   QPS=50
#   MAX_IN_FLIGHT=2000
#   MAX_TOKENS=20
#   TEMPERATURE=0
#   OUT_CSV=.model/benchmark/vllm_queryner_latency.csv

set -euo pipefail

if [ -z "${MODEL_NAME:-}" ]; then
    echo "错误: 请设置 MODEL_NAME（vLLM /v1/chat/completions 的 model 字段）"
  echo "示例: MODEL_NAME=.model/queryner/trl_standard/best bash scripts/vllm_benchmark_queryner.sh"
    exit 1
fi

uv run --group trl python deploy/vllm/vllm_benchmark_queryner.py \
  --endpoint "${ENDPOINT:-http://127.0.0.1:8000}" \
  --model "$MODEL_NAME" \
  --split "${SPLIT:-test}" \
  --requests "${REQUESTS:-200}" \
  --qps "${QPS:-50}" \
  --max_in_flight "${MAX_IN_FLIGHT:-2000}" \
  --max_tokens "${MAX_TOKENS:-20}" \
  --temperature "${TEMPERATURE:-0}" \
  --out_csv "${OUT_CSV:-.model/benchmark/vllm_queryner_latency.csv}"
