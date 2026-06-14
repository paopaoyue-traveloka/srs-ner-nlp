#!/usr/bin/env bash
# vLLM 部署脚本（针对 QueryNER 20-in/20-out 轻量请求优化）
#
# 用法：
#   MODEL_PATH=/path/to/model bash scripts/vllm_serve_queryner.sh
#
# 可选环境变量：
#   HOST=0.0.0.0
#   PORT=8000
#   DTYPE=bfloat16
#   QUANTIZATION= (如 fp8 / fbgemm_fp8)
#   KV_CACHE_DTYPE=auto (如 fp8)
#   MAX_MODEL_LEN=1024
#   MAX_NUM_SEQS=1024
#   GPU_MEMORY_UTILIZATION=0.92
#   EXTRA_ARGS="..."

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
    echo "错误: 请设置 MODEL_PATH，例如: MODEL_PATH=.model/queryner/trl/best"
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

QUANT_ARGS=""
if [ -n "$QUANTIZATION" ]; then
  QUANT_ARGS="$QUANT_ARGS --quantization $QUANTIZATION"
fi
if [ "$KV_CACHE_DTYPE" != "auto" ]; then
  QUANT_ARGS="$QUANT_ARGS --kv-cache-dtype $KV_CACHE_DTYPE"
fi

echo "启动 vLLM 服务:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  HOST=$HOST PORT=$PORT"
echo "  DTYPE=$DTYPE"
echo "  QUANTIZATION=${QUANTIZATION:-none}"
echo "  KV_CACHE_DTYPE=$KV_CACHE_DTYPE"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  enable_chunked_prefill=false"

vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill false \
  --disable-log-requests \
  $QUANT_ARGS \
  ${EXTRA_ARGS:-}
