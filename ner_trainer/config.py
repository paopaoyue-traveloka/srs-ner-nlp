"""
ner_trainer/config.py

BaseTrainConfig — 所有 NER 训练后端共用的超参基类。

只包含与具体框架无关的通用字段：
  数据路径、split 命名、保存目录、epoch 数、选优指标、早停。

各后端子类（如 HanLPTrainConfig）在此基础上添加框架专属字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# 支持的 best-checkpoint 选优指标
BestMetric = Literal["f1", "precision", "recall", "case_accuracy"]


@dataclass
class BaseTrainConfig:
    """
    框架无关的通用训练配置。

    子类只需添加自己后端专属的字段（如 pretrained_model、lr 等），
    通用训练循环（NERTrainer）读取此基类字段驱动 epoch 控制和早停。
    """

    # ── 数据 ─────────────────────────────────────────────────────
    dataset_name: str = "queryner"
    """要微调的数据集名称（必须已在 ner_datasets.registry 注册）。"""

    train_split: str = "train"
    """用于训练的 split 名称。"""

    dev_split: str = "validation"
    """用于每 epoch 验证的 split 名称。"""

    test_split: str = "test"
    """训练结束后最终评估的 split 名称（空字符串表示跳过）。"""

    data_dir: str = ".data"
    """TSV 文件临时导出目录（已在 .gitignore，不提交 git）。"""

    # ── 保存 ─────────────────────────────────────────────────────
    save_dir: str = ".model"
    """模型保存根目录（不提交 git）。子目录按 dataset_name 区分。"""

    # ── 训练控制 ──────────────────────────────────────────────────
    epochs: int = 30
    """总训练轮数。每轮结束后在 dev 集上评估一次。"""

    # ── 选优 & 早停 ───────────────────────────────────────────────
    best_metric: BestMetric = "f1"
    """
    保存 best checkpoint 和早停所依据的指标（在 dev 集上评估）。
    可选：f1 / precision / recall / case_accuracy
    """

    early_stopping_patience: int = 5
    """
    早停耐心值：连续多少个 epoch dev 上的 best_metric 没有改善就停止训练。
    设为 0 或负数表示禁用早停，跑满 epochs。
    """

    # ── 扩展 ──────────────────────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict)
    """
    任意额外键值对，子类或后端实现可自由解读。
    通用训练循环不读取此字段。
    """
