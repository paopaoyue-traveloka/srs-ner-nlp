# vLLM Deploy Module (QueryNER)

本模块提供训练后模型的 vLLM 部署与压测入口。

## 1) 启动服务（针对 20 in / 20 out 轻量请求优化）

```bash
MODEL_PATH=.model/queryner/trl_standard/best bash deploy/vllm/serve.sh
```

默认优化：
- `--enable-chunked-prefill false`
- `--max-num-seqs 1024`
- `--max-model-len 1024`

可通过环境变量覆盖：`HOST` / `PORT` / `DTYPE` / `MAX_MODEL_LEN` / `MAX_NUM_SEQS` / `GPU_MEMORY_UTILIZATION` / `EXTRA_ARGS`。

### FP8 原生量化部署（vLLM）

```bash
MODEL_PATH=.model/queryner/trl_standard/best bash deploy/vllm/vllm_serve_fp8_queryner.sh
```

默认会设置：
- `QUANTIZATION=fp8`
- `KV_CACHE_DTYPE=fp8`

你也可以覆盖：`QUANTIZATION` / `KV_CACHE_DTYPE`。

## 2) 固定 QPS Benchmark（采样 QueryNER）

```bash
MODEL_NAME=.model/queryner/trl_standard/best bash deploy/vllm/benchmark.sh
```

主要记录：完整返回端到端延迟（从请求发起到完整响应返回）。

关键参数（环境变量）：
- `QPS`（固定发压速率）
- `REQUESTS`（请求总数）
- `MAX_IN_FLIGHT`（客户端并发上限）
- `MAX_TOKENS`（默认 20，贴合 QueryNER）
- `OUT_CSV`（明细输出）

## 3) Evaluate（接入已有评估字段）

```bash
MODEL_NAME=.model/queryner/trl_standard/best bash deploy/vllm/vllm_eval_queryner.sh
```

输出 CSV 字段与项目 evaluate 明细一致：
- `input_case`
- `expected_output`
- `actual_output`
- `is_case_correct`
- `tp_entity_count`
- `expected_entity_count`
- `actual_entity_count`

同时输出 summary JSON（包含聚合指标）：
- `precision`
- `recall`
- `f1`
- `case_accuracy`

## 4) 一键 FP8 vs BF16 对比

```bash
MODEL_PATH=.model/queryner/trl_standard/best MODEL_NAME=.model/queryner/trl_standard/best \
bash deploy/vllm/vllm_compare_fp8_bf16.sh
```

若使用其他训练模式，可替换为对应目录：
- `trl_unsloth/best`
- `trl_grpo/best`

会自动：
1. 启 BF16 服务，跑 evaluate + benchmark
2. 启 FP8 服务，跑 evaluate + benchmark
3. 输出对比表（case accuracy + mean/p95 latency）
