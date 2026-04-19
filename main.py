"""
CLI 入口 — NER 数据集探索工具

用法:
    uv run main.py list
    uv run main.py stats queryner
    uv run main.py show  queryner [--split test] [--n 5]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from ner_datasets import registry
from ner_datasets.base import DatasetStats, NERDataset, NERExample


# ── 展示函数 ──────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> None:
    """列出所有已注册数据集。"""
    entries = registry.list()
    if not entries:
        print("(没有已注册的数据集)")
        return

    print(f"\n{'已注册数据集':=^52}")
    for e in entries:
        print(f"\n  名称      : {e['name']}")
        print(f"  说明      : {e['description']}")
        print(f"  HF 仓库   : {e['hf_repo']}")
    print()


def _print_stats(stats: DatasetStats) -> None:
    """渲染 DatasetStats。"""
    print(f"\n{'数据集统计':=^52}")
    print(f"  数据集: {stats.name}\n")

    # Split 规模
    print(f"  {'Split':<14} {'样本数':>8}  {'平均词数':>8}")
    print(f"  {'-'*34}")

    # 需要从外部传入 avg_lengths，这里直接从 stats 取
    for split, count in stats.splits.items():
        avg = stats.avg_lengths.get(split, 0)
        print(f"  {split:<14} {count:>8}  {avg:>8.1f}")

    # 标签列表
    print(f"\n  标签列表 ({len(stats.label_names)} 个):")
    for i, name in enumerate(stats.label_names):
        print(f"    [{i:>2}] {name}")

    # 标签分布
    total = sum(stats.label_counts.values())
    print(f"\n  训练集标签分布 (共 {total:,} 个 token):")
    print(f"  {'标签':<30} {'数量':>8}  {'占比':>6}")
    print(f"  {'-'*48}")
    for label, count in stats.label_counts.most_common():
        pct = count / total * 100 if total else 0
        print(f"  {label:<30} {count:>8,}  {pct:>5.1f}%")
    print()


def cmd_stats(args: argparse.Namespace) -> None:
    """加载指定数据集并展示统计信息。"""
    try:
        ds = registry.get(args.dataset)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据集 '{ds.name}'...")
    ds.load()
    _print_stats(ds.stats())


def _print_examples(examples: list[NERExample], split: str) -> None:
    """渲染样本列表（含 BIO 标注和实体提取）。"""
    print(f"\n{'样本展示':=^52}")
    print(f"  Split: {split}  共展示 {len(examples)} 条\n")

    for i, ex in enumerate(examples, 1):
        print(f"  [{i}] 查询: {ex.query}")

        # BIO 逐词标注
        print(f"       {'词':<22} {'标签'}")
        print(f"       {'-'*40}")
        for token, label in zip(ex.tokens, ex.labels):
            print(f"       {token:<22} {label}")

        # 实体提取
        entities = ex.entities()
        if entities:
            print(f"       实体:")
            for span, etype in entities:
                print(f"         [{etype}] {span}")
        else:
            print(f"       实体: (无)")
        print()


def cmd_show(args: argparse.Namespace) -> None:
    """加载指定数据集并展示样本及实体提取。"""
    try:
        ds = registry.get(args.dataset)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据集 '{ds.name}'...")
    ds.load()

    split = args.split
    if split not in ds.splits():
        print(
            f"错误: split '{split}' 不存在，可用: {', '.join(ds.splits())}",
            file=sys.stderr,
        )
        sys.exit(1)

    examples = ds.sample(split=split, n=args.n)
    _print_examples(examples, split)


# ── CLI 解析 ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NER 数据集探索工具",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # list
    list_p = sub.add_parser("list", help="列出所有已注册的数据集")
    list_p.set_defaults(func=cmd_list)

    # stats
    stats_p = sub.add_parser("stats", help="展示数据集基本统计和标签分布")
    stats_p.add_argument("dataset", help="数据集名称，如 queryner")
    stats_p.set_defaults(func=cmd_stats)

    # show
    show_p = sub.add_parser("show", help="展示数据集样本和实体提取")
    show_p.add_argument("dataset", help="数据集名称，如 queryner")
    show_p.add_argument("--split", default="train", help="split 名称（默认 train）")
    show_p.add_argument("--n", type=int, default=5, help="展示条数（默认 5）")
    show_p.set_defaults(func=cmd_show)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
