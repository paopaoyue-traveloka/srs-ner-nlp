"""
NERDataset 抽象基类 — 所有 NER 数据集需继承此类并实现接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class NERExample:
    """单条 NER 样本。"""
    tokens: list[str]
    labels: list[str]  # BIO 字符串标签，与 tokens 等长

    def __post_init__(self):
        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"tokens({len(self.tokens)}) 与 labels({len(self.labels)}) 长度不匹配"
            )

    @property
    def query(self) -> str:
        return " ".join(self.tokens)

    def entities(self) -> list[tuple[str, str]]:
        """返回 [(span_text, entity_type), ...] 列表。"""
        result: list[tuple[str, str]] = []
        current_type: str | None = None
        current_tokens: list[str] = []

        for token, label in zip(self.tokens, self.labels):
            if label.startswith("B-"):
                if current_type:
                    result.append((" ".join(current_tokens), current_type))
                current_type = label[2:]
                current_tokens = [token]
            elif label.startswith("I-") and current_type:
                current_tokens.append(token)
            else:  # O 或不合法的 I-
                if current_type:
                    result.append((" ".join(current_tokens), current_type))
                current_type = None
                current_tokens = []

        if current_type:
            result.append((" ".join(current_tokens), current_type))

        return result


@dataclass
class DatasetStats:
    """数据集基本统计信息。"""
    name: str
    splits: dict[str, int] = field(default_factory=dict)        # split -> 样本数
    avg_lengths: dict[str, float] = field(default_factory=dict) # split -> 平均词数
    label_names: list[str] = field(default_factory=list)
    label_counts: Counter = field(default_factory=Counter)      # train split 标签分布


class NERDataset(ABC):
    """所有 NER 数据集的抽象基类。"""

    # 子类必须设置的类属性
    name: str = ""
    description: str = ""
    hf_repo: str = ""

    # ── 子类必须实现 ──────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """加载（或下载）数据集。调用后 is_loaded 应为 True。"""

    @abstractmethod
    def splits(self) -> list[str]:
        """返回可用 split 名称列表，如 ['train', 'validation', 'test']。"""

    @abstractmethod
    def label_names(self) -> list[str]:
        """返回所有 BIO 标签字符串列表（含 O）。"""

    @abstractmethod
    def __iter__(self) -> Iterator[NERExample]:
        """迭代 train split 的所有样本。"""

    @abstractmethod
    def iter_split(self, split: str) -> Iterator[NERExample]:
        """迭代指定 split 的所有样本。"""

    @abstractmethod
    def __len__(self) -> int:
        """返回 train split 样本数。"""

    @abstractmethod
    def split_len(self, split: str) -> int:
        """返回指定 split 的样本数。"""

    # ── 已实现的通用方法 ──────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                f"数据集 '{self.name}' 尚未加载，请先调用 .load()"
            )

    def stats(self, count_split: str = "train") -> DatasetStats:
        """计算并返回数据集统计信息。"""
        self._require_loaded()
        split_sizes = {s: self.split_len(s) for s in self.splits()}
        avg_lengths = {
            s: sum(len(ex.tokens) for ex in self.iter_split(s)) / max(self.split_len(s), 1)
            for s in self.splits()
        }
        counter: Counter = Counter()
        for ex in self.iter_split(count_split):
            counter.update(ex.labels)

        return DatasetStats(
            name=self.name,
            splits=split_sizes,
            avg_lengths=avg_lengths,
            label_names=self.label_names(),
            label_counts=counter,
        )

    def sample(self, split: str = "train", n: int = 5) -> list[NERExample]:
        """返回指定 split 的前 n 条样本。"""
        self._require_loaded()
        result = []
        for i, ex in enumerate(self.iter_split(split)):
            if i >= n:
                break
            result.append(ex)
        return result

    def export_tsv(
        self,
        output_dir: str | Path = ".data",
        splits: list[str] | None = None,
    ) -> dict[str, Path]:
        """
        将指定 split 导出为 HanLP 兼容的两列 BIO TSV 文件。

        格式：每行 `token\\tlabel`，空行分隔句子。
        文件命名：<output_dir>/<dataset_name>_<split>.tsv

        Args:
            output_dir: 输出目录，默认 .data/（已在 .gitignore 中）
            splits:     要导出的 split 列表，默认导出全部 split

        Returns:
            dict[split_name -> 文件路径]
        """
        self._require_loaded()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        target_splits = splits if splits is not None else self.splits()
        exported: dict[str, Path] = {}

        for split in target_splits:
            if split not in self.splits():
                raise ValueError(
                    f"split '{split}' 不存在，可用: {', '.join(self.splits())}"
                )
            path = out / f"{self.name}_{split}.tsv"
            with path.open("w", encoding="utf-8") as f:
                for ex in self.iter_split(split):
                    for token, label in zip(ex.tokens, ex.labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")  # 空行分隔句子
            exported[split] = path

        return exported
