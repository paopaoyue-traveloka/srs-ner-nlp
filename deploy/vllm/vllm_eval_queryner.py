#!/usr/bin/env python3
"""
对 vLLM OpenAI 接口执行 QueryNER 评估。

输出：
- 聚合指标（precision/recall/f1/case_accuracy）
- 每条样本明细 CSV（与项目 evaluate CSV 字段一致）
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import aiohttp
from datasets import load_dataset

from ner_trainer.gen_utils import abbrev_to_label, bio_to_entity_spans, parse_bio_output


DEFAULT_SYSTEM_PROMPT = (
    "NER for e-commerce queries. Label each token with BIO tags. "
    "Output one tag per token, comma-separated. Count MUST match input tokens."
)


async def infer_one(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    query: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> tuple[bool, str]:
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
    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            if resp.status != 200:
                return False, await resp.text()
            data = await resp.json()
            return True, data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


async def main_async(args: argparse.Namespace) -> None:
    ds = load_dataset("bltlab/queryner", split=args.split)
    if args.n > 0:
        ds = ds.select(range(min(args.n, len(ds))))

    endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"

    tp = 0
    fp = 0
    fn = 0
    case_correct = 0
    case_total = 0

    rows: list[dict[str, object]] = []

    connector = aiohttp.TCPConnector(limit=args.max_in_flight)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(args.max_in_flight)

        async def run_case(idx: int):
            nonlocal tp, fp, fn, case_correct, case_total
            ex = ds[idx]
            tokens = ex["tokens"]
            query = " ".join(tokens)
            gold_labels = [ds.features["ner_tags"].feature.names[t] for t in ex["ner_tags"]]

            async with sem:
                ok, out = await infer_one(
                    session,
                    endpoint=endpoint,
                    model=args.model,
                    query=query,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_sec=args.timeout,
                )

            if ok:
                pred_raw = parse_bio_output(out, len(tokens))
                pred_labels = [abbrev_to_label(t) for t in pred_raw]
            else:
                pred_labels = ["O"] * len(tokens)

            pred_spans = bio_to_entity_spans(pred_labels)
            gold_spans = bio_to_entity_spans(gold_labels)

            tp_case = len(pred_spans & gold_spans)
            fp_case = len(pred_spans - gold_spans)
            fn_case = len(gold_spans - pred_spans)

            tp += tp_case
            fp += fp_case
            fn += fn_case

            is_case_correct = pred_labels == gold_labels
            if is_case_correct:
                case_correct += 1
            case_total += 1

            rows.append(
                {
                    "input_case": query,
                    "expected_output": ",".join(gold_labels),
                    "actual_output": ",".join(pred_labels),
                    "is_case_correct": int(is_case_correct),
                    "tp_entity_count": tp_case,
                    "expected_entity_count": len(gold_spans),
                    "actual_entity_count": len(pred_spans),
                }
            )

        await asyncio.gather(*(run_case(i) for i in range(len(ds))))

    nb_pred = tp + fp
    nb_true = tp + fn
    precision = tp / nb_pred if nb_pred else 0.0
    recall = tp / nb_true if nb_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    case_acc = case_correct / case_total if case_total else 0.0

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_case",
                "expected_output",
                "actual_output",
                "is_case_correct",
                "tp_entity_count",
                "expected_entity_count",
                "actual_entity_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "n": case_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "case_accuracy": case_acc,
        "nb_correct": tp,
        "nb_pred": nb_pred,
        "nb_true": nb_true,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"split={args.split} n={case_total}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} case_accuracy={case_acc:.4f}")
    print(f"detail_csv={out_csv}")
    print(f"summary_json={summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate vLLM endpoint on QueryNER")
    p.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--n", type=int, default=0, help="0 表示全量")
    p.add_argument("--max_in_flight", type=int, default=128)
    p.add_argument("--max_tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--out_csv", type=str, default=".model/eval/vllm_queryner_eval.csv")
    p.add_argument("--summary_json", type=str, default=".model/eval/vllm_queryner_eval_summary.json")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
