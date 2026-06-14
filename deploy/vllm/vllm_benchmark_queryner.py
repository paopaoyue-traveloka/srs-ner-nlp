#!/usr/bin/env python3
"""
基于 QueryNER 随机采样请求，对 vLLM OpenAI 接口做固定 QPS 压测。

主要统计：端到端完整返回延迟（从请求发起到拿到完整响应）。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from datasets import load_dataset


DEFAULT_SYSTEM_PROMPT = (
    "NER for e-commerce queries. Label each token with BIO tags. "
    "Output one tag per token, comma-separated. Count MUST match input tokens."
)


@dataclass
class BenchResult:
    request_id: int
    query: str
    started_at: float
    ended_at: float
    latency_ms: float
    ok: bool
    status: int
    response_text: str
    error: str


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def sample_queries(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("bltlab/queryner", split=split)
    idxs = list(range(len(ds)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    chosen = idxs[:n]
    queries: list[str] = []
    for i in chosen:
        tokens = ds[i]["tokens"]
        queries.append(" ".join(tokens))
    return queries


async def send_one(
    session: aiohttp.ClientSession,
    *,
    request_id: int,
    query: str,
    endpoint: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> BenchResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    started_at = time.perf_counter()
    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            text = await resp.text()
            ended_at = time.perf_counter()
            latency_ms = (ended_at - started_at) * 1000.0
            if resp.status != 200:
                return BenchResult(
                    request_id=request_id,
                    query=query,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    ok=False,
                    status=resp.status,
                    response_text="",
                    error=text[:1000],
                )

            try:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                return BenchResult(
                    request_id=request_id,
                    query=query,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    ok=False,
                    status=resp.status,
                    response_text="",
                    error=f"invalid_json: {e}; raw={text[:500]}",
                )

            return BenchResult(
                request_id=request_id,
                query=query,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                ok=True,
                status=resp.status,
                response_text=content,
                error="",
            )
    except Exception as e:  # noqa: BLE001
        ended_at = time.perf_counter()
        return BenchResult(
            request_id=request_id,
            query=query,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(ended_at - started_at) * 1000.0,
            ok=False,
            status=0,
            response_text="",
            error=str(e),
        )


async def run_bench(args: argparse.Namespace) -> list[BenchResult]:
    endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"
    queries = sample_queries(args.split, args.requests, args.seed)

    connector = aiohttp.TCPConnector(limit=args.max_in_flight)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks: list[asyncio.Task[BenchResult]] = []
        start = time.perf_counter()
        interval = 1.0 / args.qps

        for i, q in enumerate(queries):
            target = start + i * interval
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)
            task = asyncio.create_task(
                send_one(
                    session,
                    request_id=i,
                    query=q,
                    endpoint=endpoint,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_sec=args.timeout,
                )
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "query",
                "latency_ms",
                "ok",
                "status",
                "response_text",
                "error",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "request_id": r.request_id,
                    "query": r.query,
                    "latency_ms": f"{r.latency_ms:.3f}",
                    "ok": int(r.ok),
                    "status": r.status,
                    "response_text": r.response_text,
                    "error": r.error,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vLLM endpoint with QueryNER sampled requests")
    p.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--qps", type=float, default=50.0)
    p.add_argument("--max_in_flight", type=int, default=2000)
    p.add_argument("--max_tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--out_csv", type=str, default=".model/benchmark/vllm_queryner_latency.csv")
    args = p.parse_args()

    rows = asyncio.run(run_bench(args))
    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)

    ok_rows = [r for r in rows if r.ok]
    lat = [r.latency_ms for r in ok_rows]

    print(f"总请求: {len(rows)}")
    print(f"成功请求: {len(ok_rows)}")
    print(f"失败请求: {len(rows) - len(ok_rows)}")
    if lat:
        print(f"端到端延迟(ms): mean={statistics.mean(lat):.2f} p50={percentile(lat, 0.50):.2f} p90={percentile(lat, 0.90):.2f} p95={percentile(lat, 0.95):.2f} p99={percentile(lat, 0.99):.2f} max={max(lat):.2f}")
    print(f"明细已写入: {out_csv}")


if __name__ == "__main__":
    main()
