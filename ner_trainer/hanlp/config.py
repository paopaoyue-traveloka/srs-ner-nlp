"""
ner_trainer/hanlp/config.py

HanLPTrainConfig — HanLP MTL 后端专属训练配置。

继承 BaseTrainConfig，添加 HanLP MultiTaskLearning 特有字段。
训练流程参考：
  https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train/open_base.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ner_trainer.config import BaseTrainConfig


@dataclass
class HanLPTrainConfig(BaseTrainConfig):
    """
    HanLP MultiTaskLearning + TaggingNamedEntityRecognition 训练配置。

    通用字段继承自 BaseTrainConfig（dataset_name / epochs / best_metric 等）。
    HanLP MTL 特有字段在此声明，由 HanLPTrainer 读取后传入 mtl.fit()。
    """

    # ── Transformer 编码器 ────────────────────────────────────────────
    transformer: str = "bert-base-cased"
    """
    HuggingFace transformer 模型名称，作为 ContextualWordEmbedding 的底座。
    例如：
      "bert-base-cased"                     （英文，默认）
      "hfl/chinese-electra-180g-small-discriminator"  （中文 ELECTRA-small）
      "answerdotai/ModernBERT-base"          （英文 ModernBERT）
      "xlm-roberta-base"                    （多语种 XLM-R）
    """

    average_subwords: bool = True
    """是否对子词做平均池化（英文 WordPiece 场景建议 True）。"""

    word_dropout: float = 0.1
    """Embedding dropout 概率。"""

    max_sequence_length: int = 512
    """Transformer 最大输入序列长度。"""

    # ── 任务超参 ─────────────────────────────────────────────────────
    batch_size: int = 32
    """每个 batch 的样本数。"""

    lr: float = 1e-3
    """任务头（decoder）学习率。"""

    encoder_lr: float = 5e-5
    """Transformer 编码器学习率（通常远小于任务头）。"""

    grad_norm: float = 5.0
    """梯度裁剪 max norm。"""

    gradient_accumulation: int = 1
    """梯度累积步数，等效于将 batch_size 乘以此倍数。"""

    warmup_steps: float = 0.1
    """
    Warmup 步数或比例。
    float < 1.0 表示占总步数的比例（如 0.1 = 前 10% 步做 warmup）；
    int >= 1 表示绝对步数。
    """

    tagging_scheme: str | None = None
    """
    BIO 标注方案。None 由 HanLP 自动推断。
    可选：'BIO' / 'BIOES' / 'BMES'。
    """

    crf: bool = False
    """是否使用 CRF 解码层（增加序列约束，通常提升精度但速度慢）。"""

    eval_trn: bool = False
    """每 epoch 是否同时在训练集上评估（通常关闭以节省时间）。"""

    # ── 扩展 ──────────────────────────────────────────────────────────
    task_extra: dict[str, Any] = field(default_factory=dict)
    """
    传给 TaggingNamedEntityRecognition 的额外参数。
    通用训练循环不读取此字段，由 HanLPTrainer 透传给任务构造器。
    """
