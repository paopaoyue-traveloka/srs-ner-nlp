"""
ner_trainer/gliner2/config.py

GLiNER2TrainConfig — GLiNER2 后端专属训练配置。
"""

from __future__ import annotations

from dataclasses import dataclass

from ner_trainer.config import BaseTrainConfig


@dataclass
class GLiNER2TrainConfig(BaseTrainConfig):
    """
    GLiNER2 训练配置。

    说明：
    - 使用 gliner2.training.trainer 的 TrainingConfig 执行训练
    - 保留 BaseTrainConfig 的通用字段（split/save_dir/epochs/early_stop 等）
    """

    pretrained_model: str = "fastino/gliner2-base-v1"
    """GLiNER2 预训练模型名称或本地路径。"""

    batch_size: int = 8
    """训练 batch 大小。"""

    eval_batch_size: int | None = None
    """评估 batch 大小；None 时沿用 batch_size。"""

    encoder_lr: float = 1e-5
    """编码器学习率。"""

    task_lr: float = 5e-4
    """任务头学习率。"""

    warmup_ratio: float = 0.1
    """warmup 比例。"""

    scheduler_type: str = "cosine"
    """学习率调度器类型。"""

    threshold: float = 0.5
    """推理时实体置信度阈值。"""

    fp16: bool = False
    """是否启用 fp16（仅 GPU 场景有效）。"""

    use_lora: bool = False
    """是否启用 LoRA 参数高效微调。"""

    lora_r: int = 8
    """LoRA rank。"""

    lora_alpha: float = 16.0
    """LoRA alpha。"""

    lora_dropout: float = 0.0
    """LoRA dropout。"""

    save_adapter_only: bool = False
    """启用 LoRA 时是否仅保存 adapter。"""
