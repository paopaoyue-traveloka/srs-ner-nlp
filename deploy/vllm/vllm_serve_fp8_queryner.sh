#!/usr/bin/env bash
# vLLM 原生 FP8 部署（QueryNER 场景）
#
# 用法：
#   MODEL_PATH=.model/queryner/trl_standard/best bash deploy/vllm/vllm_serve_fp8_queryner.sh
#
# 可选环境变量：
#   QUANTIZATION=fp8
#   KV_CACHE_DTYPE=fp8

set -euo pipefail

export QUANTIZATION="${QUANTIZATION:-fp8}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

bash deploy/vllm/vllm_serve_queryner.sh
