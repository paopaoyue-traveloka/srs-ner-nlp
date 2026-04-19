"""
WandbLogger — WandB 训练日志与 artifact 上传。

密钥加载顺序（优先级由高到低）：
  1. 环境变量 WANDB_API_KEY（CI/CD 场景）
  2. 项目根目录 .env 文件中的 WANDB_API_KEY（本地开发，不提交 git）
  3. wandb 自身的 ~/.netrc 登录缓存（wandb login 后会写入）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ner_trainer.config import BaseTrainConfig as TrainConfig
    from metrics.base import NERMetrics

logger = logging.getLogger(__name__)


# ── .env 加载 ─────────────────────────────────────────────────────────

def _load_dotenv(env_path: Path | None = None) -> None:
    """
    从 .env 文件读取环境变量并注入到当前进程。
    不依赖 python-dotenv，手动解析 KEY=VALUE 格式。
    只写入当前进程中未设置的变量（不覆盖 CI 中的环境变量）。
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ── 配置 ──────────────────────────────────────────────────────────────

@dataclass
class WandbConfig:
    project: str = "ner-finetune"
    """WandB 项目名称。"""

    entity: str | None = None
    """WandB 团队/用户名，None 时使用账号默认值。"""

    run_name: str | None = None
    """本次运行的显示名称，None 时由 WandB 自动生成。"""

    tags: list[str] = field(default_factory=list)
    """运行标签，便于在 WandB UI 中过滤。"""

    notes: str = ""
    """运行备注，自由文本。"""

    enabled: bool = True
    """
    是否启用 WandB 上传。设为 False 时所有方法变为 no-op，
    便于在没有网络或不需要记录时运行。
    """

    log_model_artifact: bool = True
    """是否将模型目录作为 WandB Artifact 上传。"""


# ── 日志记录器 ────────────────────────────────────────────────────────

class WandbLogger:
    """
    封装 WandB run 的生命周期：init → log_metrics → log_model → finish。

    所有方法在 config.enabled=False 时均为 no-op。
    """

    def __init__(self, config: WandbConfig) -> None:
        self.config = config
        self._run = None
        _load_dotenv()

    def init(self, train_config: TrainConfig) -> None:
        """
        初始化 WandB run，将 TrainConfig 的所有字段作为超参上传。
        """
        if not self.config.enabled:
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning("wandb 未安装，跳过日志上传。运行 `uv add wandb` 安装。")
            self.config.enabled = False
            return

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)

        hparams: dict[str, Any] = {
            k: v for k, v in vars(train_config).items() if not k.startswith("_")
        }

        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            tags=self.config.tags or None,
            notes=self.config.notes or None,
            config=hparams,
        )
        logger.info("WandB run 已初始化: %s", self._run.url if self._run else "")

    def log_metrics(self, metrics: NERMetrics, step: int | None = None) -> None:
        """
        上传 NERMetrics 到 WandB。

        指标 key 格式：`{split}/{metric_name}`，便于在 UI 中按 split 分组。
        step 建议传入 epoch 编号（从 1 开始）以对齐训练曲线。
        """
        if not self.config.enabled or self._run is None:
            return

        import wandb  # type: ignore

        prefix = metrics.split
        log_dict: dict[str, Any] = {
            f"{prefix}/precision": metrics.precision,
            f"{prefix}/recall": metrics.recall,
            f"{prefix}/f1": metrics.f1,
            f"{prefix}/nb_correct": metrics.nb_correct,
            f"{prefix}/nb_pred": metrics.nb_pred,
            f"{prefix}/nb_true": metrics.nb_true,
            f"{prefix}/fp": metrics.fp,
            f"{prefix}/fn": metrics.fn,
            f"{prefix}/case_accuracy": metrics.case_accuracy,
        }
        if metrics.loss is not None:
            log_dict[f"{prefix}/loss"] = metrics.loss

        wandb.log(log_dict, step=step)
        logger.info("WandB metrics 已上传 step=%s: %s", step, log_dict)

    def log_model(self, model_dir: str | Path, artifact_name: str | None = None) -> None:
        """
        将模型目录作为 WandB Artifact 上传（type="model"）。
        版本由 WandB 自动管理（v0, v1, ...）。
        """
        if not self.config.enabled or not self.config.log_model_artifact:
            return
        if self._run is None:
            logger.warning("WandB run 未初始化，跳过模型上传。")
            return

        import wandb  # type: ignore

        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning("模型目录不存在: %s，跳过上传。", model_path)
            return

        name = artifact_name or model_path.name
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_dir(str(model_path))
        self._run.log_artifact(artifact)
        logger.info("模型 Artifact '%s' 上传完成。", name)

    def log_config(self, extra: dict[str, Any]) -> None:
        """补充上传额外的 config 字段（在 init 之后调用）。"""
        if not self.config.enabled or self._run is None:
            return
        import wandb  # type: ignore
        wandb.config.update(extra)

    def finish(self) -> None:
        """结束 WandB run，确保所有数据上传完毕再退出。"""
        if not self.config.enabled or self._run is None:
            return
        import wandb  # type: ignore
        wandb.finish()
        logger.info("WandB run 已结束。")
