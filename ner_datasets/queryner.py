"""
QueryNER 数据集实现。

来源: https://huggingface.co/datasets/bltlab/queryner
论文: QueryNER: Segmentation of E-commerce Queries (arXiv:2405.09507)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from .base import NERDataset, NERExample

if TYPE_CHECKING:
    from datasets import DatasetDict  # HuggingFace datasets


class QueryNERDataset(NERDataset):
    """
    QueryNER — 电商搜索查询 NER 数据集。

    Splits: train / validation / test
    标签数: 35（17 种实体类型 × B-/I- + O）
    """

    name = "queryner"
    description = "电商搜索查询 NER 数据集，来自 Amazon ESCI，含 17 种实体类型"
    hf_repo = "bltlab/queryner"

    def __init__(self) -> None:
        self._dataset: DatasetDict | None = None
        self._loaded: bool = False
        self._label_names: list[str] = []

    # ── 加载 ─────────────────────────────────────────────────────

    def load(self) -> None:
        from datasets import load_dataset  # type: ignore

        self._dataset = load_dataset(self.hf_repo)
        self._label_names = (
            self._dataset["train"]
            .features["ner_tags"]
            .feature
            .names
        )
        self._loaded = True

    # ── 基本接口 ──────────────────────────────────────────────────

    def splits(self) -> list[str]:
        self._require_loaded()
        return list(self._dataset.keys())  # type: ignore[union-attr]

    def label_names(self) -> list[str]:
        self._require_loaded()
        return self._label_names

    def split_len(self, split: str) -> int:
        self._require_loaded()
        return len(self._dataset[split])  # type: ignore[index]

    def __len__(self) -> int:
        return self.split_len("train")

    def __iter__(self) -> Iterator[NERExample]:
        return self.iter_split("train")

    def iter_split(self, split: str) -> Iterator[NERExample]:
        self._require_loaded()
        ds = self._dataset[split]  # type: ignore[index]
        for row in ds:
            yield NERExample(
                tokens=row["tokens"],
                labels=[self._label_names[t] for t in row["ner_tags"]],
            )
