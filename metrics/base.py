"""
metrics/base.py

NERMetrics — 统一的 NER 评估结果数据类。

涵盖：
  - Entity-level span 精确匹配指标（precision / recall / f1）
  - 原始计数（TP / FP / FN）
  - 平均 loss（cross-entropy，来自模型 evaluate()）
  - Case-level 准确率：整条查询的所有标签预测完全正确才算对
  - 元信息（split / dataset / model / epoch）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NERMetrics:
    """
    NER 模型单次评估的完整结果。

    指标分三层：
      1. Entity-level F1 —— 主要排名指标，严格 span 精确匹配
      2. Case-level accuracy —— 整条查询完全正确率
      3. Loss —— 训练过程监控，用于判断是否过拟合
    """

    # ── 1. Entity-level 指标（主指标）─────────────────────────────
    # 聚合方式：micro-average（跨所有句子累加 TP/FP/FN 后再计算）
    # 匹配条件：(entity_type, start_idx, end_idx) 三者全部一致才算 TP

    precision: float
    """
    实体精确率 = TP / (TP + FP)
    模型预测出的实体中，有多少是完全正确的（类型+边界全匹配）。
    """

    recall: float
    """
    实体召回率 = TP / (TP + FN)
    gold 标注的实体中，有多少被模型成功预测出来。
    """

    f1: float
    """
    实体 F1 = 2 * P * R / (P + R)
    精确率与召回率的调和平均，综合衡量实体识别质量。
    通常作为模型选优（best checkpoint）和早停的主指标。
    """

    # ── 2. 原始计数（用于 debug 和 per-type 分析）──────────────────

    nb_correct: int
    """TP：预测集与 gold 集的 set 交集大小（完全匹配的实体数）。"""

    nb_pred: int
    """TP + FP：模型在本 split 上预测出的实体总数。"""

    nb_true: int
    """TP + FN：gold 标注中的实体总数。"""

    # ── 3. Case-level 准确率（辅助指标）──────────────────────────

    case_accuracy: float = 0.0
    """
    Case-level（查询级别）完全正确率。

    一条查询（sentence/case）当且仅当其所有 token 的预测标签
    与 gold 标签完全一致时，才算预测正确。

    case_accuracy = 完全正确的查询数 / 总查询数

    与 entity F1 的区别：
      - entity F1 在 span 层面局部衡量，一条查询里对了部分实体也得分
      - case_accuracy 是全有或全无（all-or-nothing），更严格
      - 适合用于衡量"能否完美处理一条用户查询"这类场景
    """

    nb_cases_correct: int = 0
    """预测完全正确的查询数（分子）。"""

    nb_cases_total: int = 0
    """参与评估的查询总数（分母）。"""

    # ── 4. Loss（训练过程监控）────────────────────────────────────

    loss: float | None = None
    """
    平均 cross-entropy loss（来自 HanLP evaluate() 返回的第二个元素）。
    训练集 loss 持续下降而 dev loss 上升，通常意味着过拟合。
    None 表示该次评估未返回 loss（如独立 validate() 调用时）。
    """

    # ── 5. 元信息 ────────────────────────────────────────────────

    split: str = "test"
    """评估所用的数据集 split 名称（train / validation / test）。"""

    dataset_name: str = ""
    """评估所用的数据集名称，如 'queryner'。"""

    model_dir: str = ""
    """被评估模型的保存路径。"""

    epoch: int | None = None
    """对应的训练 epoch（从 1 开始），None 表示独立评估而非训练中间状态。"""

    extra: dict[str, Any] = field(default_factory=dict)
    """额外自定义字段，用于记录实验特定信息。"""

    # ── 派生属性 ────────────────────────────────────────────────

    @property
    def fp(self) -> int:
        """FP = nb_pred - nb_correct：预测出但与 gold 不匹配的实体数。"""
        return self.nb_pred - self.nb_correct

    @property
    def fn(self) -> int:
        """FN = nb_true - nb_correct：gold 中未被预测出的实体数。"""
        return self.nb_true - self.nb_correct

    # ── 格式化输出 ───────────────────────────────────────────────

    def __str__(self) -> str:
        epoch_str = f"  Epoch       : {self.epoch}\n" if self.epoch is not None else ""
        loss_str = f"  Loss        : {self.loss:.4f}\n" if self.loss is not None else ""
        return (
            f"NERMetrics [{self.dataset_name} / {self.split}]\n"
            f"{epoch_str}"
            f"  Precision   : {self.precision:.4f}  ({self.nb_correct}/{self.nb_pred})\n"
            f"  Recall      : {self.recall:.4f}  ({self.nb_correct}/{self.nb_true})\n"
            f"  F1          : {self.f1:.4f}\n"
            f"  TP={self.nb_correct}  FP={self.fp}  FN={self.fn}\n"
            f"  Case Acc    : {self.case_accuracy:.4f}"
            f"  ({self.nb_cases_correct}/{self.nb_cases_total})\n"
            f"{loss_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        转为 flat dict，便于写入 WandB / CSV / JSON。
        所有键均为小写下划线风格。
        """
        d: dict[str, Any] = {
            "split": self.split,
            "dataset_name": self.dataset_name,
            "model_dir": self.model_dir,
            # entity-level
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "nb_correct": self.nb_correct,
            "nb_pred": self.nb_pred,
            "nb_true": self.nb_true,
            "fp": self.fp,
            "fn": self.fn,
            # case-level
            "case_accuracy": self.case_accuracy,
            "nb_cases_correct": self.nb_cases_correct,
            "nb_cases_total": self.nb_cases_total,
        }
        if self.loss is not None:
            d["loss"] = self.loss
        if self.epoch is not None:
            d["epoch"] = self.epoch
        d.update(self.extra)
        return d
