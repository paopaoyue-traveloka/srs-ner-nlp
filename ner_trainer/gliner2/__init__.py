"""
ner_trainer.gliner2 — GLiNER2 NER 微调子模块。

公共接口：
    GLiNER2TrainConfig   GLiNER2 专属训练配置
    GLiNER2Trainer       NERTrainer 的 GLiNER2 实现
"""

from ner_trainer.gliner2.config import GLiNER2TrainConfig
from ner_trainer.gliner2.trainer import GLiNER2Trainer

__all__ = ["GLiNER2TrainConfig", "GLiNER2Trainer"]
