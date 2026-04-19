"""
ner_trainer/hanlp/config.py

HanLPTrainConfig — HanLP 后端专属训练配置。

继承 BaseTrainConfig，添加 HanLP/Transformer 特有字段：
  pretrained_model、batch_size、lr、warmup_steps、grad_norm。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ner_trainer.config import BaseTrainConfig


@dataclass
class HanLPTrainConfig(BaseTrainConfig):
    """
    HanLP TransformerNamedEntityRecognizer 专属训练配置。

    通用字段继承自 BaseTrainConfig（dataset_name / epochs / best_metric 等）。
    HanLP 特有字段在此声明，由 HanLPTrainer 读取并注入 ner.config。
    """

    pretrained_model: str = "MSRA_NER_ELECTRA_SMALL_ZH"
    """
    HanLP 预训练 NER 模型名称（hanlp.pretrained.ner 下的属性名）。
    例如：MSRA_NER_ELECTRA_SMALL_ZH / MSRA_NER_BERT_BASE_ZH
    """

    batch_size: int | None = None
    """批大小。None 表示沿用预训练默认值（通常 32）。"""

    lr: float | None = None
    """学习率。None 表示沿用预训练默认值。微调建议 1e-5 ~ 5e-5。"""

    warmup_steps: int | None = None
    """AdamW warmup 步数。None 表示沿用预训练默认值。"""

    grad_norm: float | None = None
    """梯度裁剪 max norm。None 表示沿用预训练默认值（通常 5.0）。"""

    def to_hanlp_overrides(self) -> dict[str, Any]:
        """
        生成可注入 ner.config 的覆盖字典。
        epochs 固定为 1（由 NERTrainer 基类的 epoch 循环控制）。
        其余只写入非 None 字段及 extra 内容。
        """
        overrides: dict[str, Any] = {"epochs": 1}
        for key in ("batch_size", "lr", "warmup_steps", "grad_norm"):
            val = getattr(self, key)
            if val is not None:
                overrides[key] = val
        overrides.update(self.extra)
        return overrides
