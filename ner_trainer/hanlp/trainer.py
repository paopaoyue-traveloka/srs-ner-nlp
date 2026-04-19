"""
ner_trainer/hanlp/trainer.py

HanLPTrainer — NERTrainer 的 HanLP 后端实现。

实现 NERTrainer 的四个抽象方法：
  load_model()          — 从 hanlp.pretrained.ner 加载预训练模型
  train_one_epoch()     — 调用 ner.fit(epochs=1) 跑一轮
  evaluate()            — 调用 ner.evaluate() 解析为 NERMetrics
  load_from_checkpoint()— 从 checkpoint 目录重新加载模型权重
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ner_trainer.base import NERTrainer
from ner_trainer.hanlp.config import HanLPTrainConfig
from metrics.base import NERMetrics

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HanLPTrainer(NERTrainer):
    """
    HanLP TransformerNamedEntityRecognizer 训练器。

    self.model 类型为 TransformerNamedEntityRecognizer。
    config 类型为 HanLPTrainConfig（含 pretrained_model / lr 等字段）。
    """

    def __init__(self, config: HanLPTrainConfig) -> None:
        super().__init__(config)
        self.config: HanLPTrainConfig = config  # 明确类型，方便 IDE 提示

    # ── 抽象方法实现 ───────────────────────────────────────────────

    def load_model(self) -> None:
        """
        从 hanlp.pretrained.ner 加载预训练 TransformerNamedEntityRecognizer，
        注入 HanLP 超参覆盖（batch_size / lr / warmup_steps / grad_norm）。
        结果赋值给 self.model。
        """
        from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer  # type: ignore
        import hanlp  # type: ignore

        model_name = self.config.pretrained_model
        url = getattr(hanlp.pretrained.ner, model_name, None)
        if url is None:
            raise ValueError(
                f"找不到预训练模型 '{model_name}'，"
                f"请查看 hanlp.pretrained.ner 中的可用名称。"
            )

        logger.info("正在加载预训练模型 '%s'...", model_name)
        ner = TransformerNamedEntityRecognizer()
        ner.load(url)

        overrides = self.config.to_hanlp_overrides()
        logger.info("超参覆盖: %s", overrides)
        for k, v in overrides.items():
            ner.config[k] = v

        self.model = ner

    def train_one_epoch(
        self,
        trn_path: Path,
        dev_path: Path,
        epoch_ckpt_dir: Path,
        epoch: int,
    ) -> None:
        """
        调用 ner.fit(epochs=1) 跑一个训练 epoch。
        HanLP fit() 在内部保存 checkpoint 到 save_dir。

        注意：HanLP fit() 每次调用都从头开始按 epochs 参数跑，
        因此这里强制设 epochs=1，配合基类的 epoch 循环实现逐轮训练。
        """
        assert self.model is not None, "请先调用 load_model()"
        self.model.fit(
            trn_data=str(trn_path),
            dev_data=str(dev_path),
            save_dir=str(epoch_ckpt_dir),
            **{**self.model.config, "epochs": 1},
        )

    def evaluate(
        self,
        data_path: Path,
        split: str,
        epoch: int | None = None,
    ) -> NERMetrics:
        """
        调用 ner.evaluate()，解析返回值为 NERMetrics。

        HanLP evaluate() 兼容两种返回格式：
          (F1_metric, avg_loss)  — 标准情况
          F1_metric              — 部分版本只返回 metric
        """
        assert self.model is not None, "请先调用 load_model()"

        result = self.model.evaluate(str(data_path), save_dir=None)

        if isinstance(result, tuple):
            metric, loss_val = result[0], float(result[1])
        else:
            metric, loss_val = result, None

        p, r, f = metric.prf

        # Case accuracy：整条查询预测完全正确才算对
        # HanLP evaluate() 不直接暴露 case-level 统计，
        # 尝试读取 nb_correct_sentences / nb_sentences 属性（若存在），
        # 否则置 0 并在注释中说明需子类/后续版本补充。
        nb_cases_correct = getattr(metric, "nb_correct_sentences", None)
        nb_cases_total = getattr(metric, "nb_sentences", None)
        if nb_cases_correct is not None and nb_cases_total:
            case_acc = nb_cases_correct / nb_cases_total
        else:
            nb_cases_correct = 0
            nb_cases_total = 0
            case_acc = 0.0

        return NERMetrics(
            precision=float(p),
            recall=float(r),
            f1=float(f),
            nb_correct=int(metric.nb_correct),
            nb_pred=int(metric.nb_pred),
            nb_true=int(metric.nb_true),
            case_accuracy=case_acc,
            nb_cases_correct=int(nb_cases_correct),
            nb_cases_total=int(nb_cases_total),
            loss=loss_val,
            split=split,
            dataset_name=self.config.dataset_name,
            model_dir=self.config.save_dir,
            epoch=epoch,
        )

    def load_from_checkpoint(self, ckpt_dir: Path) -> None:
        """
        从 checkpoint 目录重新加载模型权重到 self.model。
        用于 test 集评估前加载 best checkpoint。
        """
        from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer  # type: ignore

        logger.info("从 checkpoint '%s' 加载模型...", ckpt_dir)
        ner = TransformerNamedEntityRecognizer()
        ner.load(str(ckpt_dir))
        self.model = ner
