"""
ner_trainer.hanlp — HanLP NER 微调子模块。

公共接口：
    HanLPTrainConfig   HanLP 专属训练配置
    HanLPTrainer       NERTrainer 的 HanLP 实现
"""

from ner_trainer.hanlp.config import HanLPTrainConfig
from ner_trainer.hanlp.trainer import HanLPTrainer

__all__ = ["HanLPTrainConfig", "HanLPTrainer"]
