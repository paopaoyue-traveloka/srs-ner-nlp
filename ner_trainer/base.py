"""
ner_trainer/base.py

NERTrainer — NER 训练器抽象基类。

封装了与具体框架无关的通用训练骨架：
  - epoch 循环
  - 每 epoch 在 dev 集评估
  - best checkpoint 管理（按指定指标选优）
  - 早停
  - 训练结束后在 test 集评估 best checkpoint
  - WandB metrics 上报（可选）

子类只需实现三个抽象方法：
  load_model()      加载/初始化底层模型
  train_one_epoch() 跑一个训练 epoch
  evaluate()        在给定数据路径上评估，返回 NERMetrics

用法示例（子类）：
    class HanLPTrainer(NERTrainer):
        def load_model(self): ...
        def train_one_epoch(self, epoch): ...
        def evaluate(self, data_path, split, epoch=None): ...
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from metrics.base import NERMetrics
from ner_trainer.config import BaseTrainConfig

if TYPE_CHECKING:
    from ner_datasets.base import NERDataset
    from metrics.wandb import WandbLogger

logger = logging.getLogger(__name__)


class NERTrainer(ABC):
    """
    NER 训练器抽象基类。

    通用训练骨架在 train() 中实现，子类通过实现以下三个方法接入具体框架：

      load_model()        — 初始化并加载预训练模型到 self.model
      train_one_epoch()   — 在训练集上跑一个 epoch（原地更新 self.model）
      evaluate()          — 在给定数据路径上评估，返回 NERMetrics

    checkpoint 管理由基类统一处理：
      每 epoch 快照保存至  <run_dir>/epoch_NNN/
      best checkpoint 保存至 <run_dir>/best/（按 config.best_metric 选优）

    Attributes:
        config:  BaseTrainConfig（或其子类）训练配置
        dataset: NERDataset 实例，由 train()/validate() 注入
        model:   底层模型对象，由 load_model() 初始化，子类自行定义类型
    """

    def __init__(self, config: BaseTrainConfig) -> None:
        self.config = config
        self.dataset: NERDataset | None = None
        self.model = None  # 子类在 load_model() 中赋值

    # ── 子类必须实现的三个钩子 ─────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """
        初始化并加载预训练模型。
        结果赋值给 self.model，后续 train_one_epoch / evaluate 均通过它操作。
        调用时机：train() / validate() 开始时。
        """

    @abstractmethod
    def train_one_epoch(
        self,
        trn_path: Path,
        dev_path: Path,
        epoch_ckpt_dir: Path,
        epoch: int,
    ) -> None:
        """
        在训练集上跑一个 epoch，并将本轮 checkpoint 保存到 epoch_ckpt_dir。

        Args:
            trn_path:       训练集 TSV 路径
            dev_path:       开发集 TSV 路径（部分框架在 fit 内部同时评估）
            epoch_ckpt_dir: 本 epoch checkpoint 保存目录
            epoch:          当前 epoch 编号（从 1 开始）
        """

    @abstractmethod
    def evaluate(
        self,
        data_path: Path,
        split: str,
        epoch: int | None = None,
    ) -> NERMetrics:
        """
        在给定数据路径上评估当前 self.model，返回 NERMetrics。

        Args:
            data_path: 数据集 TSV 路径
            split:     split 名称（用于填充 NERMetrics.split 字段）
            epoch:     对应的 epoch 编号（None 表示非训练中间状态）

        Returns:
            NERMetrics（precision / recall / f1 / loss / case_accuracy 等）
        """

    @abstractmethod
    def load_from_checkpoint(self, ckpt_dir: Path) -> None:
        """
        从 checkpoint 目录加载模型权重到 self.model。
        用于训练结束后加载 best checkpoint 做 test 集评估。

        Args:
            ckpt_dir: checkpoint 目录路径
        """

    # ── 通用训练骨架（子类不应覆盖）──────────────────────────────────

    def train(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """
        完整训练流程（epoch 循环 + 早停 + test 评估）。

        流程：
          1. 准备数据集，导出 TSV
          2. load_model()
          3. for epoch in 1..epochs:
               a. train_one_epoch() → 保存 epoch checkpoint
               b. evaluate(dev) → NERMetrics
               c. 若 best_metric 改善 → 复制到 best/
               d. 检查早停
          4. 从 best/ 加载，evaluate(test)
          5. 返回 (best_dir, dev_history, test_metrics)

        Args:
            dataset: NERDataset（可不传，自动从 registry 加载）
            wb:      WandbLogger（可不传，不上报 WandB）

        Returns:
            (best_checkpoint_dir, dev_metrics_per_epoch, test_metrics_or_None)
        """
        cfg = self.config

        # 1. 数据集
        self.dataset = self._ensure_dataset(dataset)
        tsv = self._export_splits(self.dataset)
        trn_path = tsv[cfg.train_split]
        dev_path = tsv[cfg.dev_split]

        # 2. 加载模型
        logger.info("正在初始化模型...")
        self.load_model()

        # checkpoint 目录
        run_dir = Path(cfg.save_dir) / cfg.dataset_name
        best_dir = run_dir / "best"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 3. Epoch 循环
        best_score: float = -1.0
        no_improve_count: int = 0
        dev_history: list[NERMetrics] = []
        patience = cfg.early_stopping_patience

        for epoch in range(1, cfg.epochs + 1):
            logger.info("── Epoch %d / %d ──────────────────", epoch, cfg.epochs)

            # 3a. 训练一轮
            epoch_ckpt = run_dir / f"epoch_{epoch:03d}"
            self.train_one_epoch(trn_path, dev_path, epoch_ckpt, epoch)

            # 3b. dev 评估
            dev_metrics = self.evaluate(dev_path, cfg.dev_split, epoch=epoch)
            dev_history.append(dev_metrics)
            logger.info("[Epoch %d] dev\n%s", epoch, dev_metrics)

            if wb is not None:
                wb.log_metrics(dev_metrics, step=epoch)

            # 3c. 选优
            score = float(getattr(dev_metrics, cfg.best_metric, 0.0))
            if score > best_score:
                best_score = score
                no_improve_count = 0
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(str(epoch_ckpt), str(best_dir))
                logger.info(
                    "[Epoch %d] 新 best checkpoint（%s=%.4f）→ %s",
                    epoch, cfg.best_metric, best_score, best_dir,
                )
            else:
                no_improve_count += 1
                logger.info(
                    "[Epoch %d] 无改善（%s=%.4f，best=%.4f，patience %d/%d）",
                    epoch, cfg.best_metric, score, best_score,
                    no_improve_count, patience,
                )

            # 3d. 早停
            if patience > 0 and no_improve_count >= patience:
                logger.info(
                    "早停触发：连续 %d 轮 dev %s 无改善。",
                    patience, cfg.best_metric,
                )
                break

        # 4. Test 评估（加载 best checkpoint）
        test_metrics: NERMetrics | None = None
        test_split = cfg.test_split
        if test_split and test_split in self.dataset.splits() and best_dir.exists():
            test_path = tsv.get(test_split)
            if test_path is None:
                exported = self.dataset.export_tsv(
                    output_dir=cfg.data_dir, splits=[test_split]
                )
                test_path = exported[test_split]

            logger.info("加载 best checkpoint，在 test 集评估...")
            self.load_from_checkpoint(best_dir)
            test_metrics = self.evaluate(test_path, test_split, epoch=None)
            test_metrics.model_dir = str(best_dir)
            logger.info("[Test]\n%s", test_metrics)

            if wb is not None:
                wb.log_metrics(test_metrics)

        return best_dir, dev_history, test_metrics

    def validate(
        self,
        split: str = "test",
        dataset: NERDataset | None = None,
        model_dir: str | None = None,
    ) -> NERMetrics:
        """
        独立评估：加载指定 checkpoint，在给定 split 上输出 NERMetrics。

        Args:
            split:     评估 split，默认 "test"
            dataset:   NERDataset（可不传，自动加载）
            model_dir: checkpoint 路径，None 时使用 save_dir/<dataset_name>/best

        Returns:
            NERMetrics
        """
        cfg = self.config
        resolved = Path(model_dir) if model_dir else (
            Path(cfg.save_dir) / cfg.dataset_name / "best"
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"模型目录不存在: '{resolved}'，请先运行 train()。"
            )

        self.dataset = self._ensure_dataset(dataset)

        if split not in self.dataset.splits():
            raise ValueError(
                f"split '{split}' 不存在，可用: {', '.join(self.dataset.splits())}"
            )

        exported = self.dataset.export_tsv(
            output_dir=cfg.data_dir, splits=[split]
        )
        tsv_path = exported[split]

        logger.info("正在加载模型 '%s'...", resolved)
        self.load_model()
        self.load_from_checkpoint(resolved)

        logger.info("正在评估 split='%s'...", split)
        metrics = self.evaluate(tsv_path, split, epoch=None)
        metrics.model_dir = str(resolved)
        logger.info("评估完成:\n%s", metrics)
        return metrics

    # ── 内部工具 ──────────────────────────────────────────────────────

    def _ensure_dataset(self, dataset: NERDataset | None) -> NERDataset:
        if dataset is not None:
            if not dataset.is_loaded:
                dataset.load()
            return dataset
        from ner_datasets import registry
        ds = registry.get(self.config.dataset_name)
        if not ds.is_loaded:
            logger.info("正在加载数据集 '%s'...", self.config.dataset_name)
            ds.load()
        return ds

    def _export_splits(self, dataset: NERDataset) -> dict[str, Path]:
        cfg = self.config
        needed = [cfg.train_split, cfg.dev_split]
        if cfg.test_split and cfg.test_split in dataset.splits():
            needed.append(cfg.test_split)
        needed = list(dict.fromkeys(needed))  # 去重保序
        return dataset.export_tsv(output_dir=cfg.data_dir, splits=needed)
