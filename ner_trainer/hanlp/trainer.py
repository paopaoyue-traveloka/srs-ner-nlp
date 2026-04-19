"""
ner_trainer/hanlp/trainer.py

HanLPTrainer — NERTrainer 的 HanLP MTL 后端实现。

训练流程参考官方 demo：
  https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train/open_base.py

核心结构：
  MultiTaskLearning（mtl）
    └── encoder: ContextualWordEmbedding（HuggingFace transformer）
    └── tasks:   {'ner': TaggingNamedEntityRecognition(...)}

设计说明：
  HanLP 的 MultiTaskLearning.fit() 内置 epoch 循环、dev 评估和 patience 早停，
  与基类 NERTrainer 的外层 epoch 循环存在根本性冲突（多次 fit() 会导致
  self.vocabs 未初始化的 AttributeError）。

  因此 HanLPTrainer 选择 **覆盖基类的 train()** 方法，将整个训练生命周期
  交给 fit() 一次性完成，而不是每 epoch 调用一次 train_one_epoch()。

  覆盖后流程：
    1. 准备数据集，导出 TSV
    2. load_model()
    3. fit(epochs=cfg.epochs, save_dir=run_dir/best, ...)  ← HanLP 内置 epoch 循环
    4. evaluate(test_path) → NERMetrics
    5. 返回 (best_dir, [], test_metrics)
       （dev_history 为空列表，因为 fit() 不暴露逐 epoch 指标）

  train_one_epoch() 保留抽象方法签名但不会被调用（供 validate() 等复用基类骨架）。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from metrics.base import NERMetrics
from ner_trainer.base import NERTrainer
from ner_trainer.hanlp.config import HanLPTrainConfig

if TYPE_CHECKING:
    from ner_datasets.base import NERDataset
    from metrics.wandb import WandbLogger

logger = logging.getLogger(__name__)


class HanLPTrainer(NERTrainer):
    """
    HanLP MultiTaskLearning + TaggingNamedEntityRecognition 训练器。

    self.model 类型为 MultiTaskLearning。
    config 类型为 HanLPTrainConfig。

    训练数据通过基类的 _export_splits() 导出为 BIO TSV，
    再传入 TaggingNamedEntityRecognition 作为 trn/dev/tst 路径。

    整个训练生命周期由 fit() 一次性完成（覆盖了基类的 train()），
    train_one_epoch() 实现为 pass（不会被调用）。
    """

    def __init__(self, config: HanLPTrainConfig) -> None:
        super().__init__(config)
        self.config: HanLPTrainConfig = config

    # ── 抽象方法实现 ───────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        构建 MultiTaskLearning 实例。
        模型权重在调用 train() → fit() 时由 HanLP 初始化。
        """
        from hanlp.components.mtl.multi_task_learning import MultiTaskLearning  # type: ignore

        logger.info("构建 MultiTaskLearning 实例（transformer=%s）", self.config.transformer)
        self.model = MultiTaskLearning()

    def train_one_epoch(
        self,
        trn_path: Path,
        dev_path: Path,
        epoch_ckpt_dir: Path,
        epoch: int,
    ) -> None:
        """
        不使用：HanLPTrainer 覆盖了基类 train()，不走逐 epoch 循环。
        此方法保留以满足抽象基类要求。
        """
        raise NotImplementedError(
            "HanLPTrainer 不使用 train_one_epoch()，训练由 train() → fit() 一次完成。"
        )

    def evaluate(
        self,
        data_path: Path,
        split: str,
        epoch: int | None = None,
    ) -> NERMetrics:
        """
        在给定数据路径上评估当前 self.model，返回 NERMetrics。

        直接调用 TorchComponent.evaluate(tst_data=str(data_path))，
        绕过 MultiTaskLearning.evaluate() 的硬编码 'tst' 数据源。
        """
        assert self.model is not None, "请先调用 load_model() 并完成训练或加载 checkpoint"

        from hanlp.common.torch_component import TorchComponent  # type: ignore

        logger.info("评估 split='%s'（data_path=%s）", split, data_path)

        # 调用父类 evaluate 以支持任意路径
        rets = TorchComponent.evaluate(self.model, str(data_path))

        # rets = (MetricDict, (total_loss, MetricDict, dataloader))
        metric_dict = rets[0]
        loss_val: float | None = None
        if isinstance(rets[1], tuple) and len(rets[1]) >= 1:
            try:
                loss_val = float(rets[1][0])
            except (TypeError, ValueError):
                loss_val = None

        # 提取 ner 任务 metric
        if hasattr(metric_dict, "__getitem__"):
            metric = metric_dict["ner"]
        else:
            metric = metric_dict

        p, r, f = metric.prf

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
        从 checkpoint 目录重新加载 MultiTaskLearning 模型。
        """
        from hanlp.components.mtl.multi_task_learning import MultiTaskLearning  # type: ignore

        logger.info("从 checkpoint '%s' 加载模型...", ckpt_dir)
        mtl = MultiTaskLearning()
        mtl.load(str(ckpt_dir))
        self.model = mtl

    # ── 覆盖基类 train()，将完整训练交给 fit() 一次完成 ──────────────

    def train(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """
        完整训练流程（覆盖基类，由 HanLP fit() 一次完成所有 epoch）。

        流程：
          1. 准备数据集，导出 TSV
          2. load_model()
          3. fit(epochs=cfg.epochs, save_dir=best_dir)  ← HanLP 内置 epoch 循环 + patience 早停
          4. evaluate(test_path) → NERMetrics
          5. 返回 (best_dir, [], test_metrics)

        Args:
            dataset: NERDataset（可不传，自动从 registry 加载）
            wb:      WandbLogger（可不传，不上报 WandB）

        Returns:
            (best_checkpoint_dir, dev_history=[], test_metrics_or_None)
        """
        from hanlp.common.dataset import SortingSamplerBuilder  # type: ignore
        from hanlp.components.mtl.tasks.ner.tag_ner import TaggingNamedEntityRecognition  # type: ignore
        from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding  # type: ignore

        cfg = self.config

        # 1. 数据集
        self.dataset = self._ensure_dataset(dataset)
        tsv = self._export_splits(self.dataset)
        trn_path = tsv[cfg.train_split]
        dev_path = tsv[cfg.dev_split]
        test_path = tsv.get(cfg.test_split) if cfg.test_split else None

        # 2. 加载模型
        logger.info("正在初始化模型...")
        self.load_model()

        # checkpoint 目录：HanLP fit() 将 best checkpoint 保存到 save_dir
        run_dir = Path(cfg.save_dir) / cfg.dataset_name
        best_dir = run_dir / "best"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 3. 构建 encoder 和 tasks
        encoder = ContextualWordEmbedding(
            field="token",
            transformer=cfg.transformer,
            average_subwords=cfg.average_subwords,
            word_dropout=cfg.word_dropout,
            max_sequence_length=cfg.max_sequence_length,
        )

        # patience 参数：HanLP 接受 float（占 epochs 的比例）或 int（绝对轮数）
        # early_stopping_patience <= 0 表示禁用早停，传 epochs 本身使条件永远不触发
        patience = cfg.early_stopping_patience if cfg.early_stopping_patience > 0 else cfg.epochs

        tasks = {
            "ner": TaggingNamedEntityRecognition(
                trn=str(trn_path),
                dev=str(dev_path),
                tst=str(test_path) if test_path else str(dev_path),
                sampler_builder=SortingSamplerBuilder(batch_size=cfg.batch_size),
                lr=cfg.lr,
                tagging_scheme=cfg.tagging_scheme,
                crf=cfg.crf,
                **cfg.task_extra,
            )
        }

        logger.info(
            "开始训练（transformer=%s, epochs=%d, lr=%s, encoder_lr=%s, patience=%s）",
            cfg.transformer, cfg.epochs, cfg.lr, cfg.encoder_lr, patience,
        )

        # 3. fit()：HanLP 内置 epoch 循环 + dev 评估 + patience 早停 + best 保存
        self.model.fit(
            encoder=encoder,
            tasks=tasks,
            save_dir=str(best_dir),
            epochs=cfg.epochs,
            patience=patience,
            lr=cfg.lr,
            encoder_lr=cfg.encoder_lr,
            grad_norm=cfg.grad_norm,
            gradient_accumulation=cfg.gradient_accumulation,
            warmup_steps=cfg.warmup_steps,
            eval_trn=cfg.eval_trn,
        )

        logger.info("训练完成，best checkpoint 已保存至 '%s'", best_dir)

        # 4. Test 评估
        test_metrics: NERMetrics | None = None
        if test_path and test_path.exists():
            logger.info("在 test 集上评估 best checkpoint...")
            test_metrics = self.evaluate(test_path, cfg.test_split, epoch=None)
            test_metrics.model_dir = str(best_dir)
            logger.info("[Test]\n%s", test_metrics)

            if wb is not None:
                wb.log_metrics(test_metrics)

        # dev_history 为空列表（fit() 不暴露逐 epoch 指标）
        return best_dir, [], test_metrics
