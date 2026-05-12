"""
ner_trainer — NER 训练框架。

架构：
  NERTrainer（抽象基类）        ner_trainer/base.py
    └── HanLPTrainer           ner_trainer/hanlp/trainer.py

  BaseTrainConfig（通用配置）   ner_trainer/config.py
    └── HanLPTrainConfig       ner_trainer/hanlp/config.py

快速上手（HanLP 后端）：
    from ner_trainer.hanlp import HanLPTrainer, HanLPTrainConfig

    config = HanLPTrainConfig(
        dataset_name="queryner",
        epochs=30,
        lr=2e-5,
        best_metric="f1",
        early_stopping_patience=5,
    )
    trainer = HanLPTrainer(config)
    best_dir, dev_history, test_metrics = trainer.train()

新增后端步骤：
  1. 在 ner_trainer/<backend>/ 下新建 config.py 和 trainer.py
  2. Config 继承 BaseTrainConfig，Trainer 继承 NERTrainer
  3. 实现 load_model / train_one_epoch / evaluate / load_from_checkpoint
  4. 在此 __init__.py 中可选导出
"""

from ner_trainer.config import BaseTrainConfig, BestMetric
from ner_trainer.base import NERTrainer
from ner_trainer.hanlp import HanLPTrainConfig, HanLPTrainer
from ner_trainer.gliner2 import GLiNER2TrainConfig, GLiNER2Trainer

# 便捷别名：保持旧调用 ner_trainer.TrainConfig 兼容
TrainConfig = HanLPTrainConfig

__all__ = [
    # 抽象层
    "BaseTrainConfig",
    "BestMetric",
    "NERTrainer",
    # HanLP 后端
    "HanLPTrainConfig",
    "HanLPTrainer",
    # GLiNER2 后端
    "GLiNER2TrainConfig",
    "GLiNER2Trainer",
    # 兼容别名
    "TrainConfig",
]
