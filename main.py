"""
CLI 入口 — NER 数据集探索 & 微调工具

用法:
    uv run main.py list
    uv run main.py stats queryner
    uv run main.py show  queryner [--split test] [--n 5]

    uv run main.py train queryner --backend hanlp
    uv run main.py train queryner --backend gliner2
    uv run main.py train queryner --backend trl --base_model openbmb/MiniCPM5-1B

    uv run main.py validate queryner --backend trl --split test
    uv run main.py validate queryner --split test --model_dir .model/queryner/trl_standard/best

    uv run main.py upload-model .model/queryner/trl_standard/best
    uv run main.py upload-model .model/queryner/trl_standard/best --artifact_name my-ner-model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from ner_datasets import registry
from ner_datasets.base import DatasetStats, NERDataset, NERExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


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
    print(f"\n{'数据集统计':=^52}")
    print(f"  数据集: {stats.name}\n")

    print(f"  {'Split':<14} {'样本数':>8}  {'平均词数':>8}")
    print(f"  {'-'*34}")
    for split, count in stats.splits.items():
        avg = stats.avg_lengths.get(split, 0)
        print(f"  {split:<14} {count:>8}  {avg:>8.1f}")

    print(f"\n  标签列表 ({len(stats.label_names)} 个):")
    for i, name in enumerate(stats.label_names):
        print(f"    [{i:>2}] {name}")

    total = sum(stats.label_counts.values())
    print(f"\n  训练集标签分布 (共 {total:,} 个 token):")
    print(f"  {'标签':<30} {'数量':>8}  {'占比':>6}")
    print(f"  {'-'*48}")
    for label, count in stats.label_counts.most_common():
        pct = count / total * 100 if total else 0
        print(f"  {label:<30} {count:>8,}  {pct:>5.1f}%")
    print()


def cmd_stats(args: argparse.Namespace) -> None:
    try:
        ds = registry.get(args.dataset)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据集 '{ds.name}'...")
    ds.load()
    _print_stats(ds.stats())


def _print_examples(examples: list[NERExample], split: str) -> None:
    print(f"\n{'样本展示':=^52}")
    print(f"  Split: {split}  共展示 {len(examples)} 条\n")

    for i, ex in enumerate(examples, 1):
        print(f"  [{i}] 查询: {ex.query}")
        print(f"       {'词':<22} {'标签'}")
        print(f"       {'-'*40}")
        for token, label in zip(ex.tokens, ex.labels):
            print(f"       {token:<22} {label}")

        entities = ex.entities()
        if entities:
            print(f"       实体:")
            for span, etype in entities:
                print(f"         [{etype}] {span}")
        else:
            print(f"       实体: (无)")
        print()


def cmd_show(args: argparse.Namespace) -> None:
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


# ── 训练 & 评估命令 ───────────────────────────────────────────────────

def _build_train_config(args: argparse.Namespace):
    """从 argparse Namespace 构建后端对应的 TrainConfig。"""
    backend = getattr(args, "backend", "hanlp")

    if backend == "hanlp":
        from ner_trainer.hanlp import HanLPTrainConfig

        kwargs: dict = {"dataset_name": args.dataset}
        for f in (
            # 通用
            "epochs", "train_split", "dev_split", "test_split",
            "data_dir", "save_dir", "best_metric", "early_stopping_patience",
            # HanLP MTL 专属
            "transformer", "average_subwords", "word_dropout", "max_sequence_length",
            "batch_size", "lr", "encoder_lr", "grad_norm",
            "gradient_accumulation", "warmup_steps",
            "tagging_scheme", "crf", "eval_trn",
        ):
            val = getattr(args, f, None)
            if val is not None:
                kwargs[f] = val
        return HanLPTrainConfig(**kwargs)

    if backend == "gliner2":
        from ner_trainer.gliner2 import GLiNER2TrainConfig

        kwargs = {"dataset_name": args.dataset}
        for f in (
            # 通用
            "epochs", "train_split", "dev_split", "test_split",
            "data_dir", "save_dir", "best_metric", "early_stopping_patience",
            # GLiNER2 专属
            "pretrained_model", "batch_size", "eval_batch_size", "encoder_lr", "task_lr",
            "warmup_ratio", "scheduler_type", "threshold", "fp16",
            "use_lora", "lora_r", "lora_alpha", "lora_dropout", "save_adapter_only",
        ):
            val = getattr(args, f, None)
            if val is not None:
                kwargs[f] = val
        return GLiNER2TrainConfig(**kwargs)

    if backend == "trl":
        from ner_trainer.trl import TRLTrainConfig

        kwargs = {"dataset_name": args.dataset}
        for f in (
            # 通用
            "epochs", "train_split", "dev_split", "test_split",
            "data_dir", "save_dir", "best_metric", "early_stopping_patience",
            # TRL 专属
            "base_model", "lora_r", "lora_alpha", "lora_dropout",
            "batch_size", "lr", "warmup_ratio",
            "max_length", "max_new_tokens", "temperature", "system_prompt",
            "use_unsloth", "load_in_4bit", "full_finetune", "eval_csv_path",
            "trl_mode", "grpo_num_generations", "grpo_max_prompt_length",
            "grpo_max_completion_length", "grpo_beta", "grpo_temperature",
            "max_steps",
        ):
            val = getattr(args, f, None)
            if val is not None:
                kwargs[f] = val
        # accumulative_counts → gradient_accumulation_steps
        val = getattr(args, "accumulative_counts", None)
        if val is not None:
            kwargs["gradient_accumulation_steps"] = val
        return TRLTrainConfig(**kwargs)

    raise ValueError(f"不支持的 backend: {backend}")


def _resolve_trainer_cls(backend: str):
    if backend == "hanlp":
        from ner_trainer.hanlp import HanLPTrainer

        return HanLPTrainer
    if backend == "gliner2":
        from ner_trainer.gliner2 import GLiNER2Trainer

        return GLiNER2Trainer
    if backend == "trl":
        from ner_trainer.trl import TRLTrainer

        return TRLTrainer
    raise ValueError(f"不支持的 backend: {backend}")


def _build_wandb_logger(args: argparse.Namespace, run_name_default: str | None = None):
    from metrics import WandbConfig, WandbLogger
    from metrics.wandb import _load_dotenv

    # 先加载 .env，使 WANDB_PROJECT / WANDB_ENTITY 可从 os.environ 读取
    _load_dotenv()

    wconfig = WandbConfig(
        project=(
            getattr(args, "wandb_project", None)
            or os.environ.get("WANDB_PROJECT")
            or "ner-finetune"
        ),
        entity=(
            getattr(args, "wandb_entity", None)
            or os.environ.get("WANDB_ENTITY")
            or None
        ),
        run_name=getattr(args, "wandb_run", None) or run_name_default,
        enabled=not getattr(args, "no_wandb", False),
    )
    return WandbLogger(wconfig)


def cmd_train(args: argparse.Namespace) -> None:
    """微调 NER 模型（支持 HanLP / GLiNER2 / TRL 后端）。"""

    config = _build_train_config(args)
    wb = _build_wandb_logger(args, run_name_default=f"{args.dataset}-train")
    trainer_cls = _resolve_trainer_cls(getattr(args, "backend", "hanlp"))

    try:
        ds = registry.get(config.dataset_name)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据集 '{ds.name}'...")
    ds.load()

    wb.init(config)
    trainer = trainer_cls(config)

    try:
        best_dir, dev_history, test_metrics = trainer.train(dataset=ds, wb=wb)

        print(f"\n训练完成，best checkpoint: {best_dir}")
        print(f"Dev 历史 ({len(dev_history)} epochs):")
        for m in dev_history:
            print(f"  Epoch {m.epoch:>3}: {config.best_metric}={getattr(m, config.best_metric):.4f}  F1={m.f1:.4f}")

        if test_metrics:
            print(f"\n{test_metrics}")
    finally:
        wb.finish()


def cmd_validate(args: argparse.Namespace) -> None:
    """在指定 split 上评估已训练模型（支持 HanLP / GLiNER2 / TRL）。"""

    config = _build_train_config(args)
    wb = _build_wandb_logger(args, run_name_default=f"{args.dataset}-validate")
    trainer_cls = _resolve_trainer_cls(getattr(args, "backend", "hanlp"))

    try:
        ds = registry.get(config.dataset_name)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据集 '{ds.name}'...")
    ds.load()

    wb.init(config)
    trainer = trainer_cls(config)

    try:
        metrics = trainer.validate(
            split=args.split,
            dataset=ds,
            model_dir=getattr(args, "model_dir", None),
        )
        print(f"\n{metrics}")
        wb.log_metrics(metrics)
    finally:
        wb.finish()


def cmd_upload_model(args: argparse.Namespace) -> None:
    """将本地模型目录上传到 WandB Artifacts。"""
    from metrics import WandbConfig, WandbLogger
    from metrics.wandb import _load_dotenv

    _load_dotenv()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    wconfig = WandbConfig(
        project=(
            getattr(args, "wandb_project", None)
            or os.environ.get("WANDB_PROJECT")
            or "ner-finetune"
        ),
        entity=(
            getattr(args, "wandb_entity", None)
            or os.environ.get("WANDB_ENTITY")
            or None
        ),
        run_name=getattr(args, "wandb_run", None) or f"upload-{model_dir.name}",
        enabled=True,
        log_model_artifact=True,
    )
    wb = WandbLogger(wconfig)

    # 用空 config 初始化 run（仅用于上传 artifact）
    from dataclasses import dataclass

    @dataclass
    class _MinimalConfig:
        model_dir: str = str(model_dir)
        artifact_name: str = getattr(args, "artifact_name", None) or model_dir.name

    wb.init(_MinimalConfig())

    try:
        artifact_name = getattr(args, "artifact_name", None) or None
        wb.log_model(model_dir, artifact_name=artifact_name)
        print(f"模型已上传: {model_dir} → WandB artifact")
    finally:
        wb.finish()


# ── CLI 解析 ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NER 数据集探索 & HanLP 微调工具",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # list
    list_p = sub.add_parser("list", help="列出所有已注册的数据集")
    list_p.set_defaults(func=cmd_list)

    # stats
    stats_p = sub.add_parser("stats", help="展示数据集统计和标签分布")
    stats_p.add_argument("dataset", help="数据集名称，如 queryner")
    stats_p.set_defaults(func=cmd_stats)

    # show
    show_p = sub.add_parser("show", help="展示数据集样本")
    show_p.add_argument("dataset", help="数据集名称，如 queryner")
    show_p.add_argument("--split", default="train")
    show_p.add_argument("--n", type=int, default=5)
    show_p.set_defaults(func=cmd_show)

    # train
    train_p = sub.add_parser("train", help="微调 NER 模型（HanLP / GLiNER2 / TRL）")
    train_p.add_argument("dataset", help="数据集名称，如 queryner")
    train_p.add_argument("--backend", choices=["hanlp", "gliner2", "trl"], default="hanlp")
    # Transformer 编码器
    train_p.add_argument("--transformer", type=str, default=None,
                         help="HuggingFace transformer 名称（默认 bert-base-cased）")
    train_p.add_argument("--average_subwords", action="store_true", default=None)
    train_p.add_argument("--word_dropout", type=float, default=None)
    train_p.add_argument("--max_sequence_length", type=int, default=None)
    # 训练超参
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--batch_size", type=int, default=None)
    train_p.add_argument("--lr", type=float, default=None, help="任务头学习率")
    train_p.add_argument("--encoder_lr", type=float, default=None, help="Transformer 编码器学习率")
    train_p.add_argument("--warmup_steps", type=float, default=None)
    train_p.add_argument("--grad_norm", type=float, default=None)
    train_p.add_argument("--gradient_accumulation", type=int, default=None)
    train_p.add_argument("--tagging_scheme", type=str, default=None)
    train_p.add_argument("--crf", action="store_true", default=None)
    train_p.add_argument("--eval_trn", action="store_true", default=None)
    # GLiNER2 训练参数
    train_p.add_argument("--pretrained_model", type=str, default=None,
                         help="GLiNER2 预训练模型名或本地路径（如 fastino/gliner2-base-v1）")
    train_p.add_argument("--task_lr", type=float, default=None)
    train_p.add_argument("--warmup_ratio", type=float, default=None)
    train_p.add_argument("--scheduler_type", type=str, default=None)
    train_p.add_argument("--eval_batch_size", type=int, default=None)
    train_p.add_argument("--threshold", type=float, default=None)
    train_p.add_argument("--fp16", action="store_true", default=None)
    train_p.add_argument("--use_lora", action="store_true", default=None)
    train_p.add_argument("--lora_r", type=int, default=None)
    train_p.add_argument("--lora_alpha", type=float, default=None)
    train_p.add_argument("--lora_dropout", type=float, default=None)
    train_p.add_argument("--save_adapter_only", action="store_true", default=None)
    # TRL 训练参数
    train_p.add_argument("--base_model", type=str, default=None,
                         help="TRL base model（如 openbmb/MiniCPM5-1B）")
    train_p.add_argument("--accumulative_counts", type=int, default=None,
                         help="TRL 梯度累积步数")
    train_p.add_argument("--max_length", type=int, default=None,
                         help="TRL tokenizer 最大序列长度")
    train_p.add_argument("--max_new_tokens", type=int, default=None,
                         help="TRL 推理时最大生成 token 数")
    train_p.add_argument("--temperature", type=float, default=None,
                         help="TRL 推理温度（0=greedy）")
    train_p.add_argument("--system_prompt", type=str, default=None,
                         help="TRL 自定义 system prompt（留空用内置）")
    train_p.add_argument("--trl_mode", choices=["sft", "grpo"], default=None,
                         help="TRL 训练模式：sft（默认）或 grpo")
    train_p.add_argument("--grpo_num_generations", type=int, default=None,
                         help="GRPO 每个 prompt 采样候选数")
    train_p.add_argument("--grpo_max_prompt_length", type=int, default=None,
                         help="GRPO prompt 最大长度")
    train_p.add_argument("--grpo_max_completion_length", type=int, default=None,
                         help="GRPO completion 最大长度")
    train_p.add_argument("--grpo_beta", type=float, default=None,
                         help="GRPO KL 惩罚系数")
    train_p.add_argument("--grpo_temperature", type=float, default=None,
                         help="GRPO 生成采样温度（默认 0.5）")
    train_p.add_argument("--max_steps", type=int, default=None,
                         help="最大训练步数（设置后忽略 epochs）")
    # TRL + unsloth 选项
    train_p.add_argument("--use_unsloth", action="store_true", default=None,
                         help="使用 unsloth 加速（FastLanguageModel），需 pip install unsloth")
    train_p.add_argument("--load_in_4bit", action="store_true", default=None,
                         help="QLoRA 4-bit 量化加载（仅 --use_unsloth 时有效）")
    train_p.add_argument("--full_finetune", action="store_true", default=None,
                         help="TRL 全量训练（不使用 LoRA，显存占用更高）")
    train_p.add_argument("--eval_csv_path", type=str, default=None,
                         help="评估明细 CSV 输出路径（默认 .model/<dataset>/trl/eval_<split>.csv）")
    train_p.add_argument("--train_split", type=str, default=None)
    train_p.add_argument("--dev_split", type=str, default=None)
    train_p.add_argument("--test_split", type=str, default=None)
    train_p.add_argument("--data_dir", type=str, default=None)
    train_p.add_argument("--save_dir", type=str, default=None)
    # 选优 & 早停
    train_p.add_argument(
        "--best_metric",
        choices=["f1", "precision", "recall", "case_accuracy"],
        default=None,
        help="best checkpoint 选优指标（默认 f1）",
    )
    train_p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="早停耐心值，0 表示禁用（默认 5）",
    )
    # WandB
    train_p.add_argument("--wandb_project", type=str, default=None)
    train_p.add_argument("--wandb_entity", type=str, default=None)
    train_p.add_argument("--wandb_run", type=str, default=None)
    train_p.add_argument("--no_wandb", action="store_true", help="禁用 WandB 上传")
    train_p.set_defaults(func=cmd_train)

    # validate
    val_p = sub.add_parser("validate", help="在指定 split 上评估已训练模型")
    val_p.add_argument("dataset", help="数据集名称，如 queryner")
    val_p.add_argument("--backend", choices=["hanlp", "gliner2", "trl"], default="hanlp")
    val_p.add_argument("--split", type=str, default="test")
    val_p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="模型路径（TRL 默认 .model/<dataset>/trl_standard|trl_unsloth|trl_grpo/best）",
    )
    val_p.add_argument("--data_dir", type=str, default=None)
    val_p.add_argument("--save_dir", type=str, default=None)
    val_p.add_argument("--pretrained_model", type=str, default=None,
                       help="GLiNER2 初始模型（当 model_dir 未指定时使用）")
    val_p.add_argument("--base_model", type=str, default=None,
                       help="TRL base model（当 model_dir 未指定时使用）")
    val_p.add_argument("--use_unsloth", action="store_true", default=None,
                       help="使用 unsloth 推理（FastLanguageModel）")
    val_p.add_argument("--load_in_4bit", action="store_true", default=None,
                       help="QLoRA 4-bit 量化加载（仅 --use_unsloth 时有效）")
    val_p.add_argument("--full_finetune", action="store_true", default=None,
                       help="加载全量模型 checkpoint（不走 LoRA adapter）")
    val_p.add_argument("--eval_csv_path", type=str, default=None,
                       help="评估明细 CSV 输出路径（默认 .model/<dataset>/trl/eval_<split>.csv）")
    val_p.add_argument("--threshold", type=float, default=None)
    val_p.add_argument("--wandb_project", type=str, default=None)
    val_p.add_argument("--wandb_entity", type=str, default=None)
    val_p.add_argument("--wandb_run", type=str, default=None)
    val_p.add_argument("--no_wandb", action="store_true")
    val_p.set_defaults(func=cmd_validate)

    # upload-model
    up_p = sub.add_parser("upload-model", help="将模型目录上传到 WandB Artifacts")
    up_p.add_argument("model_dir", help="模型目录路径（如 .model/queryner/trl_standard/best）")
    up_p.add_argument("--artifact_name", type=str, default=None,
                       help="WandB Artifact 名称（默认使用目录名）")
    up_p.add_argument("--wandb_project", type=str, default=None)
    up_p.add_argument("--wandb_entity", type=str, default=None)
    up_p.add_argument("--wandb_run", type=str, default=None)
    up_p.set_defaults(func=cmd_upload_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
