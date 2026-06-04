"""
ner_trainer/gliner2/trainer.py

GLiNER2Trainer — NERTrainer 的 GLiNER2 后端实现。

实现策略：
- 覆盖 train()：调用 GLiNER2 官方训练器一次性完成训练
- evaluate()：按 token-level BIO 聚合计算实体级 P/R/F1，并额外统计 case_accuracy
- load_from_checkpoint()：从目录加载 GLiNER2 本地模型
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from metrics.base import NERMetrics
from ner_datasets.base import NERExample
from ner_trainer.base import NERTrainer
from ner_trainer.gliner2.config import GLiNER2TrainConfig

if TYPE_CHECKING:
    from metrics.wandb import WandbLogger
    from ner_datasets.base import NERDataset

logger = logging.getLogger(__name__)

# torch 在导入时可能触发 pynvml 弃用告警；该告警不影响训练逻辑。
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)


def _bio_to_entity_spans(labels: list[str]) -> set[tuple[str, int, int]]:
    """将 BIO 标签序列转换为实体集合：(type, start, end_exclusive)。"""
    spans: set[tuple[str, int, int]] = set()
    start = -1
    ent_type = ""
    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            if start >= 0:
                spans.add((ent_type, start, i))
            start = i
            ent_type = tag[2:]
        elif tag.startswith("I-"):
            if start < 0 or tag[2:] != ent_type:
                if start >= 0:
                    spans.add((ent_type, start, i))
                start = -1
                ent_type = ""
        else:
            if start >= 0:
                spans.add((ent_type, start, i))
                start = -1
                ent_type = ""
    if start >= 0:
        spans.add((ent_type, start, len(labels)))
    return spans


def _spans_to_bio(labels_len: int, spans: list[dict], tokens: list[str]) -> list[str]:
    """
    将 GLiNER2 抽取结果转回 BIO。

    采用 token 文本精确匹配的贪心映射：
    - 先按 span 文本分词
    - 在 tokens 中寻找第一个未占用且完全匹配的位置
    """
    bio = ["O"] * labels_len
    occupied = [False] * labels_len

    for s in spans:
        ent_type = s.get("label", "")
        text = s.get("text", "")
        if not ent_type or not text:
            continue
        phrase_tokens = text.split()
        n = len(phrase_tokens)
        if n == 0 or n > labels_len:
            continue

        found = -1
        for i in range(0, labels_len - n + 1):
            if any(occupied[i:i + n]):
                continue
            if tokens[i:i + n] == phrase_tokens:
                found = i
                break
        if found < 0:
            continue

        bio[found] = f"B-{ent_type}"
        occupied[found] = True
        for j in range(found + 1, found + n):
            bio[j] = f"I-{ent_type}"
            occupied[j] = True

    return bio


class GLiNER2Trainer(NERTrainer):
    def __init__(self, config: GLiNER2TrainConfig) -> None:
        super().__init__(config)
        self.config: GLiNER2TrainConfig = config

    def load_model(self) -> None:
        from gliner2 import GLiNER2  # type: ignore

        logger.info("加载 GLiNER2 模型: %s", self.config.pretrained_model)
        self.model = GLiNER2.from_pretrained(self.config.pretrained_model)

    def train_one_epoch(
        self,
        trn_path: Path,
        dev_path: Path,
        epoch_ckpt_dir: Path,
        epoch: int,
    ) -> None:
        raise NotImplementedError("GLiNER2Trainer 不使用 train_one_epoch()，请调用 train()")

    def _to_gliner2_jsonl(self, examples: list[NERExample], out_path: Path) -> None:
        """将 NERExample 转为 GLiNER2 训练 JSONL（entities 任务）。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                entities: dict[str, list[str]] = {}
                current_type = None
                current_tokens: list[str] = []
                for tok, tag in zip(ex.tokens, ex.labels):
                    if tag.startswith("B-"):
                        if current_type is not None and current_tokens:
                            entities.setdefault(current_type, []).append(" ".join(current_tokens))
                        current_type = tag[2:]
                        current_tokens = [tok]
                    elif tag.startswith("I-") and current_type is not None and tag[2:] == current_type:
                        current_tokens.append(tok)
                    else:
                        if current_type is not None and current_tokens:
                            entities.setdefault(current_type, []).append(" ".join(current_tokens))
                        current_type = None
                        current_tokens = []
                if current_type is not None and current_tokens:
                    entities.setdefault(current_type, []).append(" ".join(current_tokens))

                row = {
                    "input": ex.query,
                    "output": {
                        "entities": entities,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def train(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        from gliner2.training.trainer import GLiNER2Trainer as RawTrainer  # type: ignore
        from gliner2.training.trainer import TrainingConfig  # type: ignore

        cfg = self.config
        # 避免 tokenizer 多进程 fork 警告
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.dataset = self._ensure_dataset(dataset)

        train_examples = list(self.dataset.iter_split(cfg.train_split))
        dev_examples = list(self.dataset.iter_split(cfg.dev_split))

        run_dir = Path(cfg.save_dir) / cfg.dataset_name / "gliner2"
        best_dir = run_dir / "best"
        train_jsonl = run_dir / "train.jsonl"
        dev_jsonl = run_dir / "dev.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

        self._to_gliner2_jsonl(train_examples, train_jsonl)
        self._to_gliner2_jsonl(dev_examples, dev_jsonl)

        self.load_model()

        training_cfg = TrainingConfig(
            output_dir=str(run_dir),
            num_epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size or cfg.batch_size,
            encoder_lr=cfg.encoder_lr,
            task_lr=cfg.task_lr,
            warmup_ratio=cfg.warmup_ratio,
            scheduler_type=cfg.scheduler_type,
            eval_strategy="epoch",
            metric_for_best="eval_loss",
            greater_is_better=False,
            fp16=cfg.fp16,
            num_workers=0,
            pin_memory=False,
            save_best=True,
            early_stopping=cfg.early_stopping_patience > 0,
            early_stopping_patience=max(cfg.early_stopping_patience, 1),
            use_lora=cfg.use_lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            save_adapter_only=cfg.save_adapter_only,
        )

        logger.info(
            "开始 GLiNER2 训练（pretrained=%s, epochs=%d, batch_size=%d）",
            cfg.pretrained_model,
            cfg.epochs,
            cfg.batch_size,
        )

        trainer = RawTrainer(self.model, training_cfg)
        trainer.train(train_data=str(train_jsonl), eval_data=str(dev_jsonl))

        # 统一 best 路径
        native_best = run_dir / "best"
        if native_best.exists():
            best_dir = native_best
        else:
            final_dir = run_dir / "final"
            if final_dir.exists():
                best_dir = final_dir

        self.load_from_checkpoint(best_dir)

        test_metrics: NERMetrics | None = None
        if cfg.test_split and cfg.test_split in self.dataset.splits():
            test_path = Path(cfg.data_dir) / f"{self.dataset.name}_{cfg.test_split}.tsv"
            if not test_path.exists():
                self.dataset.export_tsv(output_dir=cfg.data_dir, splits=[cfg.test_split])
            test_metrics = self.evaluate(test_path, cfg.test_split, epoch=None)
            test_metrics.model_dir = str(best_dir)
            if wb is not None:
                wb.log_metrics(test_metrics)

        # GLiNER2 官方 trainer 不直接暴露每 epoch dev 指标
        return best_dir, [], test_metrics

    def evaluate(
        self,
        data_path: Path,
        split: str,
        epoch: int | None = None,
    ) -> NERMetrics:
        assert self.model is not None, "请先调用 load_model() 并完成训练或加载 checkpoint"

        if self.dataset is None:
            raise RuntimeError("dataset 未初始化")

        examples = list(self.dataset.iter_split(split))

        entity_types = list({sorted(
            {
                label[2:]
                for label in self.dataset.label_names()
                if label != "O" and "-" in label
            }
        )})

        tp = 0
        fp = 0
        fn = 0
        correct_cases = 0

        for ex in examples:
            pred = self.model.extract_entities(  # type: ignore[attr-defined]
                ex.query,
                entity_types,
                threshold=self.config.threshold,
                include_spans=False,
            )
            pred_dict = pred.get("entities", {}) if isinstance(pred, dict) else {}

            spans: list[dict] = []
            for et, vals in pred_dict.items():
                if not isinstance(vals, list):
                    continue
                for v in vals:
                    if isinstance(v, dict):
                        text = v.get("text", "")
                    else:
                        text = str(v)
                    spans.append({"label": et, "text": text})

            pred_labels = _spans_to_bio(len(ex.tokens), spans, ex.tokens)
            gold_labels = ex.labels

            pred_spans = _bio_to_entity_spans(pred_labels)
            gold_spans = _bio_to_entity_spans(gold_labels)

            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)

            if pred_labels == gold_labels:
                correct_cases += 1

        nb_pred = tp + fp
        nb_true = tp + fn
        precision = tp / nb_pred if nb_pred else 0.0
        recall = tp / nb_true if nb_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        case_total = len(examples)
        case_acc = correct_cases / case_total if case_total else 0.0

        return NERMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            nb_correct=tp,
            nb_pred=nb_pred,
            nb_true=nb_true,
            case_accuracy=case_acc,
            nb_cases_correct=correct_cases,
            nb_cases_total=case_total,
            loss=None,
            split=split,
            dataset_name=self.config.dataset_name,
            model_dir=self.config.save_dir,
            epoch=epoch,
        )

    def validate(
        self,
        split: str = "test",
        dataset: NERDataset | None = None,
        model_dir: str | None = None,
    ) -> NERMetrics:
        """独立评估：默认从 .model/<dataset>/gliner2/best 加载。"""
        cfg = self.config
        resolved = Path(model_dir) if model_dir else (
            Path(cfg.save_dir) / cfg.dataset_name / "gliner2" / "best"
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"模型目录不存在: '{resolved}'，请先运行 train()。"
            )

        self.dataset = self._ensure_dataset(dataset)
        if split not in self.dataset.splits():
            raise ValueError(
                f"split '{split}' 不存在，可用: {', '.join(self.dataset.splits())}"
            )

        self.load_model()
        self.load_from_checkpoint(resolved)

        tsv = self.dataset.export_tsv(output_dir=cfg.data_dir, splits=[split])
        metrics = self.evaluate(tsv[split], split, epoch=None)
        metrics.model_dir = str(resolved)
        return metrics

    def load_from_checkpoint(self, ckpt_dir: Path) -> None:
        from gliner2 import GLiNER2  # type: ignore

        logger.info("从 checkpoint '%s' 加载 GLiNER2 模型...", ckpt_dir)
        self.model = GLiNER2.from_pretrained(str(ckpt_dir))
