#!/usr/bin/env bash
# 一键对比 BF16 vs FP8：
# 1) 启动 BF16 服务 -> evaluate + benchmark
# 2) 启动 FP8 服务 -> evaluate + benchmark
# 3) 输出 summary 表
#
# 依赖：jq（用于解析 benchmark/eval 输出文件可选，不强依赖）

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
  echo "错误: 请设置 MODEL_PATH"
  echo "示例: MODEL_PATH=.model/queryner/trl/best MODEL_NAME=.model/queryner/trl/best bash deploy/vllm/vllm_compare_fp8_bf16.sh"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-$MODEL_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT_BF16="${PORT_BF16:-8000}"
PORT_FP8="${PORT_FP8:-8001}"

REQUESTS="${REQUESTS:-200}"
QPS="${QPS:-50}"
SPLIT="${SPLIT:-test}"

OUT_DIR="${OUT_DIR:-.model/benchmark/fp8_vs_bf16}"
mkdir -p "$OUT_DIR"

wait_port() {
  local host="$1"
  local port="$2"
  local n=0
  until nc -z "$host" "$port" >/dev/null 2>&1; do
    n=$((n+1))
    if [ "$n" -gt 120 ]; then
      echo "等待端口超时: $host:$port"
      return 1
    fi
    sleep 1
  done
}

run_eval_and_bench() {
  local mode="$1"
  local endpoint="$2"

  local eval_csv="$OUT_DIR/${mode}_eval.csv"
  local bench_csv="$OUT_DIR/${mode}_bench.csv"

  echo "[$mode] evaluate..."
  ENDPOINT="$endpoint" MODEL_NAME="$MODEL_NAME" SPLIT="$SPLIT" OUT_CSV="$eval_csv" \
    bash deploy/vllm/vllm_eval_queryner.sh

  echo "[$mode] benchmark..."
  ENDPOINT="$endpoint" MODEL_NAME="$MODEL_NAME" REQUESTS="$REQUESTS" QPS="$QPS" OUT_CSV="$bench_csv" \
    bash deploy/vllm/vllm_benchmark_queryner.sh
}

summarize_python() {
python - "$OUT_DIR" <<'PY'
import csv, statistics, sys, pathlib

out_dir = pathlib.Path(sys.argv[1])

def read_eval(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        return {'case_accuracy': 0.0}
    case_acc = sum(int(r['is_case_correct']) for r in rows) / len(rows)
    return {'case_accuracy': case_acc, 'n': len(rows)}

def read_bench(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    ok = [float(r['latency_ms']) for r in rows if int(r['ok']) == 1]
    if not ok:
        return {'mean': 0.0, 'p95': 0.0, 'n_ok': 0, 'n': len(rows)}
    ok_sorted = sorted(ok)
    def pct(p):
        k = (len(ok_sorted)-1)*p
        f = int(k)
        c = min(f+1, len(ok_sorted)-1)
        if f == c:
            return ok_sorted[f]
        return ok_sorted[f]*(c-k)+ok_sorted[c]*(k-f)
    return {'mean': statistics.mean(ok), 'p95': pct(0.95), 'n_ok': len(ok), 'n': len(rows)}

bf16_eval = read_eval(out_dir / 'bf16_eval.csv')
fp8_eval = read_eval(out_dir / 'fp8_eval.csv')
bf16_bench = read_bench(out_dir / 'bf16_bench.csv')
fp8_bench = read_bench(out_dir / 'fp8_bench.csv')

print('\n=== FP8 vs BF16 Summary ===')
print('mode,case_accuracy,bench_mean_ms,bench_p95_ms,bench_ok/total')
print(f"bf16,{bf16_eval['case_accuracy']:.4f},{bf16_bench['mean']:.2f},{bf16_bench['p95']:.2f},{bf16_bench['n_ok']}/{bf16_bench['n']}")
print(f"fp8,{fp8_eval['case_accuracy']:.4f},{fp8_bench['mean']:.2f},{fp8_bench['p95']:.2f},{fp8_bench['n_ok']}/{fp8_bench['n']}")
PY
}

echo "启动 BF16 服务..."
MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT_BF16" DTYPE="bfloat16" QUANTIZATION="" \
  bash deploy/vllm/vllm_serve_queryner.sh >"$OUT_DIR/bf16_server.log" 2>&1 &
PID_BF16=$!
trap 'kill $PID_BF16 >/dev/null 2>&1 || true; kill ${PID_FP8:-0} >/dev/null 2>&1 || true' EXIT
wait_port "$HOST" "$PORT_BF16"
run_eval_and_bench "bf16" "http://$HOST:$PORT_BF16"
kill $PID_BF16 >/dev/null 2>&1 || true
sleep 2

echo "启动 FP8 服务..."
MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT_FP8" \
  bash deploy/vllm/vllm_serve_fp8_queryner.sh >"$OUT_DIR/fp8_server.log" 2>&1 &
PID_FP8=$!
wait_port "$HOST" "$PORT_FP8"
run_eval_and_bench "fp8" "http://$HOST:$PORT_FP8"
kill $PID_FP8 >/dev/null 2>&1 || true

summarize_python
echo "明细输出目录: $OUT_DIR"
