"""
ner_trainer.trl — TRL + LoRA NER 微调子模块。

公共接口：
    TRLTrainConfig   TRL 专属训练配置
    TRLTrainer       NERTrainer 的 TRL 实现
"""

from ner_trainer.trl.config import TRLTrainConfig
from ner_trainer.trl.trainer import TRLTrainer

__all__ = ["TRLTrainConfig", "TRLTrainer"]
