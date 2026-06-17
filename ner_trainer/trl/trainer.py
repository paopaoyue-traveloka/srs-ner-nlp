"""
ner_trainer/trl/trainer.py

TRLTrainer — NERTrainer 的 TRL + LoRA 后端实现。

策略：
- 将 NER 转为生成式任务（query → 缩写 BIO 标签序列）
- 使用 TRL SFTTrainer + PEFT LoRA 微调
- 标准模式：通过 patched chat template ({% generation %}) 实现 assistant_only_loss
- unsloth 模式：使用 FastLanguageModel + dataset_text_field="text" 预格式化
- 评估时加载完整 checkpoint 做推理，解析输出为 BIO 标签并计算 entity-level F1

训练数据格式（OpenAI messages）：
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "nike air force 1 running shoes men"},
        {"role": "assistant", "content": "B-C,B-N,I-N,I-N,B-M,B-P,B-D"}
    ]}

依赖（标准模式）：
    pip install "trl>=0.21" "peft>=0.13" "transformers>=5.6,<6" datasets accelerate

依赖（unsloth 模式）：
    pip install "unsloth>=2026.5"
    pip install --force-reinstall "transformers==4.57.3"
"""

from __future__ import annotations

# ------ compatibility shim: llm_blender uses removed TRANSFORMERS_CACHE ------
# Must run before any TRL import (TRL -> judges -> llm_blender)
try:
    import transformers.utils.hub as _tfm_hub
    if not hasattr(_tfm_hub, "TRANSFORMERS_CACHE"):
        from huggingface_hub.constants import HF_HUB_CACHE
        _tfm_hub.TRANSFORMERS_CACHE = HF_HUB_CACHE
except ImportError:
    pass  # transformers not installed yet
# ------------------------------------------------------------------------------

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from metrics.base import NERMetrics
from ner_datasets.base import NERExample
from ner_trainer.base import NERTrainer
from ner_trainer.gen_utils import (
    abbrev_to_label,
    bio_to_entity_spans,
    examples_to_messages,
    parse_bio_output,
)
from ner_trainer.trl.config import TRAIN_CHAT_TEMPLATE, TRLTrainConfig

if TYPE_CHECKING:
    from metrics.wandb import WandbLogger
    from ner_datasets.base import NERDataset

logger = logging.getLogger(__name__)


# ── TRLTrainer ────────────────────────────────────────────────────────

class TRLTrainer(NERTrainer):
    """
    TRL + LoRA NER 微调训练器。

    与 HanLP / GLiNER2 不同，TRL 使用生成式方法：
    - 训练：query → 缩写 BIO 标签序列（作为文本生成目标）
    - 评估：生成 BIO 标签 → 还原缩写 → 计算 entity-level 指标

    由于 TRL SFTTrainer 自行管理训练循环，
    此类覆盖 train() 方法，不使用基类的 epoch 循环。

    可选 unsloth 加速（config.use_unsloth=True）：
    - 使用 FastLanguageModel 替代标准 HF 加载
    - 支持 QLoRA 4-bit 量化（config.load_in_4bit=True）
    - 推理时 FastLanguageModel.for_inference() 提供 2× 加速
    """

    def __init__(self, config: TRLTrainConfig) -> None:
        super().__init__(config)
        self.config: TRLTrainConfig = config
        self._tokenizer = None
        self._model = None
        self._is_unsloth_loaded = False  # 标记当前 model 是否通过 unsloth 加载

    # ── 抽象方法实现 ────────────────────────────────────────────────

    def load_model(self) -> None:
        """加载 base model + tokenizer（用于推理评估）。"""
        cfg = self.config
        if cfg.use_unsloth:
            self._load_model_unsloth()
        else:
            self._load_model_standard()

    def _load_model_standard(self) -> None:
        """标准模式：AutoModelForCausalLM + AutoTokenizer。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        logger.info("加载 base model: %s", cfg.base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = self._model
        self._is_unsloth_loaded = False

    def _load_model_unsloth(self) -> None:
        """unsloth 模式：FastLanguageModel。"""
        import torch
        from unsloth import FastLanguageModel

        cfg = self.config
        logger.info(
            "加载 base model (unsloth, 4bit=%s): %s",
            cfg.load_in_4bit, cfg.base_model,
        )
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=cfg.load_in_4bit,
            full_finetuning=False,
        )
        self.model = self._model
        self._is_unsloth_loaded = True

    def train_one_epoch(
        self,
        trn_path: Path,
        dev_path: Path,
        epoch_ckpt_dir: Path,
        epoch: int,
    ) -> None:
        raise NotImplementedError(
            "TRLTrainer 不使用 train_one_epoch()，训练由 SFTTrainer 管理。"
            "请调用 train() 执行完整训练流程。"
        )

    def evaluate(
        self,
        data_path: Path,
        split: str,
        epoch: int | None = None,
    ) -> NERMetrics:
        """
        使用当前加载的 model 在给定 split 上做生成式评估。

        流程：
        1. 遍历 split 中的每条样本
        2. 构建 ChatML prompt（system + user）
        3. model.generate() 生成缩写 BIO 标签序列
        4. 还原缩写 → 完整标签，计算 entity-level P/R/F1 + case accuracy
        """
        import torch

        assert self._model is not None, "请先调用 load_model() 或 load_from_checkpoint()"
        assert self._tokenizer is not None
        assert self.dataset is not None, "dataset 未初始化"

        cfg = self.config
        system_prompt = cfg.get_system_prompt()
        examples = list(self.dataset.iter_split(split))

        tp = 0
        fp = 0
        fn = 0
        correct_cases = 0
        per_case_rows: list[dict[str, object]] = []

        # unsloth 推理加速
        if self._is_unsloth_loaded:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self._model)

        self._model.eval()
        with torch.no_grad():
            for ex in examples:
                pred_labels = self._predict_single(ex, system_prompt)

                gold_labels = ex.labels
                pred_spans = bio_to_entity_spans(pred_labels)
                gold_spans = bio_to_entity_spans(gold_labels)

                tp_case = len(pred_spans & gold_spans)
                pred_entity_count = len(pred_spans)
                gold_entity_count = len(gold_spans)

                tp += tp_case
                fp += len(pred_spans - gold_spans)
                fn += len(gold_spans - pred_spans)

                is_case_correct = pred_labels == gold_labels
                if is_case_correct:
                    correct_cases += 1

                per_case_rows.append(
                    {
                        "input_case": ex.query,
                        "expected_output": ",".join(gold_labels),
                        "actual_output": ",".join(pred_labels),
                        "is_case_correct": int(is_case_correct),
                        "tp_entity_count": tp_case,
                        "expected_entity_count": gold_entity_count,
                        "actual_entity_count": pred_entity_count,
                    }
                )

        nb_pred = tp + fp
        nb_true = tp + fn
        precision = tp / nb_pred if nb_pred else 0.0
        recall = tp / nb_true if nb_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        case_total = len(examples)
        case_acc = correct_cases / case_total if case_total else 0.0

        csv_path = self._resolve_eval_csv_path(split)
        self._write_eval_csv(csv_path, per_case_rows)
        logger.info("评估明细已写入 CSV: %s", csv_path)

        return NERMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            nb_correct=tp,
            nb_pred=nb_pred,
            nb_true=nb_true,
            case_accuracy=case_acc,
            nb_cases_correct=correct_cases,
            nb_cases_total=case_total,
            loss=None,
            split=split,
            dataset_name=cfg.dataset_name,
            model_dir=cfg.save_dir,
            epoch=epoch,
        )

    def load_from_checkpoint(self, ckpt_dir: Path) -> None:
        """
        从 checkpoint 目录加载完整模型。

        当前训练流程只保存 merged/full checkpoint（包含 config.json）。
        """
        ckpt_dir = Path(ckpt_dir)

        if not (ckpt_dir / "config.json").exists():
            raise FileNotFoundError(
                f"模型目录不存在或不完整: '{ckpt_dir}'\n"
                "应包含 config.json（完整模型 checkpoint）。"
            )

        logger.info("检测到 full checkpoint: %s", ckpt_dir)
        loader = (
            self._load_full_checkpoint_unsloth
            if self._use_unsloth_loader()
            else self._load_full_checkpoint_standard
        )
        loader(ckpt_dir)

    def _run_subdir(self) -> str:
        """根据配置解析当前 TRL 运行目录名。"""
        cfg = self.config
        if cfg.trl_mode == "grpo":
            return "trl_grpo"
        if cfg.use_unsloth:
            return "trl_unsloth"
        return "trl_standard"

    def _use_unsloth_loader(self) -> bool:
        """解析 checkpoint 加载时是否使用 unsloth loader。"""
        cfg = self.config
        if cfg.trl_mode == "grpo" and cfg.use_unsloth:
            logger.warning("GRPO checkpoint 使用标准 HF loader（忽略 use_unsloth=True）")
            return False
        return bool(cfg.use_unsloth)

    def _load_full_checkpoint_standard(self, ckpt_dir: Path) -> None:
        """全量模式：从 checkpoint 目录直接加载完整模型。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("加载全量模型 checkpoint: %s", ckpt_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_dir), trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        self.model = self._model
        self._is_unsloth_loaded = False

    def _load_full_checkpoint_unsloth(self, ckpt_dir: Path) -> None:
        """unsloth 全量模式：从 checkpoint 目录直接加载完整模型。"""
        import torch
        from unsloth import FastLanguageModel

        cfg = self.config
        logger.info(
            "加载全量模型 checkpoint (unsloth, 4bit=%s): %s",
            cfg.load_in_4bit,
            ckpt_dir,
        )
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(ckpt_dir),
            max_seq_length=cfg.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=cfg.load_in_4bit,
        )
        self._model.eval()
        self.model = self._model
        self._is_unsloth_loaded = True

    # ── 覆盖 train()（TRL SFTTrainer 自行管理训练循环）──────────────

    def train(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """
        完整 TRL 训练流程：

        1. 准备 messages 训练数据（缩写标签）
        2. 加载 base model + tokenizer + LoRA
        3. 标准模式：patch chat template + assistant_only_loss
           unsloth 模式：pre-format text + dataset_text_field
        4. SFTTrainer 训练
        5. 保存完整 checkpoint → best/
        6. 在 test 集评估
        """
        cfg = self.config

        if cfg.trl_mode == "grpo":
            return self._train_grpo(dataset, wb)

        if cfg.use_unsloth:
            return self._train_unsloth(dataset, wb)
        else:
            return self._train_standard(dataset, wb)

    def _train_grpo(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """GRPO 训练路径（TRL GRPOTrainer + 自定义 reward）。"""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        try:
            from trl import GRPOConfig, GRPOTrainer
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise RuntimeError(
                "导入 GRPOTrainer 失败。请安装/同步 GRPO 依赖后重试：\n"
                "  uv sync --group trl\n"
                "若仍失败，请确认环境中已安装 mergekit 和 llm-blender（TRL GRPO 依赖）。"
            ) from exc

        cfg = self.config
        self.dataset = self._ensure_dataset(dataset)

        if cfg.use_unsloth:
            logger.warning("GRPO 模式暂不支持 unsloth，自动回退标准 HF 路径。")

        run_dir = Path(cfg.save_dir) / cfg.dataset_name / "trl_grpo"
        best_dir = run_dir / "best"
        run_dir.mkdir(parents=True, exist_ok=True)

        system_prompt = cfg.get_system_prompt()
        train_examples = list(self.dataset.iter_split(cfg.train_split))
        train_rows = []
        for ex in train_examples:
            train_rows.append(
                {
                    "query": ex.query,
                    "gold_labels": ex.labels,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": ex.query},
                    ],
                }
            )

        train_jsonl = run_dir / "train_grpo.jsonl"
        with train_jsonl.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        ds = Dataset.from_list(train_rows)
        logger.info("GRPO 训练数据已准备: %d 条 → %s", len(ds), train_jsonl)

        set_seed(42)
        tok = AutoTokenizer.from_pretrained(
            cfg.base_model, use_fast=True, trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        if cfg.full_finetune:
            logger.info("GRPO 使用全量训练模式（不应用 LoRA）")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = 100.0 * trainable_params / total_params if total_params else 0.0
            logger.info(
                "trainable params: %s / %s (%.2f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                pct,
            )
        else:
            logger.info("GRPO 使用 LoRA 训练模式")
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg.lora_target_modules,
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()

        # TRL GRPOTrainer 会访问 model.warnings_issued，但 PeftModel 的
        # __getattr__ 无法透传到底层模型的该属性（transformers Trainer 动态设置）。
        # 预先设置避免 AttributeError。
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        logger.info(
            "开始 TRL GRPO 训练（model=%s, epochs=%d, batch=%d×%d, lr=%s, generations=%d）",
            cfg.base_model,
            cfg.epochs,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.lr,
            cfg.grpo_num_generations,
        )

        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tok,
            reward_funcs=self._grpo_reward,
            args=GRPOConfig(
                output_dir=str(run_dir),
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.lr,
                warmup_ratio=cfg.warmup_ratio,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=cfg.logging_steps,
                save_steps=cfg.save_steps,
                save_total_limit=cfg.save_total_limit,
                report_to="wandb" if wb is not None else "none",
                dataloader_num_workers=2,
                max_prompt_length=cfg.grpo_max_prompt_length,
                max_completion_length=cfg.grpo_max_completion_length,
                num_generations=cfg.grpo_num_generations,
                beta=cfg.grpo_beta,
                seed=42,
            ),
            train_dataset=ds,
        )

        grpo_trainer.train()

        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        if cfg.full_finetune:
            grpo_trainer.model.save_pretrained(str(best_dir))
            tok.save_pretrained(str(best_dir))
            logger.info("GRPO 全量模型已保存: %s", best_dir)
        else:
            logger.info("GRPO LoRA 训练完成，开始合并并保存独立模型: %s", best_dir)
            merged_model = grpo_trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(best_dir), safe_serialization=True)
            tok.save_pretrained(str(best_dir))
            logger.info("GRPO merged 全量模型已保存: %s", best_dir)

        return self._post_train_eval(best_dir, wb)

    def _train_standard(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """标准 TRL 训练路径（{% generation %} + assistant_only_loss）。"""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer

        cfg = self.config
        self.dataset = self._ensure_dataset(dataset)

        # ── 1. 准备目录 ──────────────────────────────────────────
        run_dir = Path(cfg.save_dir) / cfg.dataset_name / "trl_standard"
        best_dir = run_dir / "best"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── 2. 导出训练数据为 messages ────────────────────────────
        system_prompt = cfg.get_system_prompt()
        train_examples = list(self.dataset.iter_split(cfg.train_split))
        messages_data = examples_to_messages(train_examples, system_prompt)

        train_jsonl = run_dir / "train_messages.jsonl"
        with train_jsonl.open("w", encoding="utf-8") as f:
            for row in messages_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        ds = Dataset.from_list(messages_data)
        logger.info("训练数据已准备: %d 条 → %s", len(ds), train_jsonl)

        # ── 3. 加载 model + tokenizer ────────────────────────────
        set_seed(42)
        tok = AutoTokenizer.from_pretrained(
            cfg.base_model, use_fast=True, trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        tok.chat_template = TRAIN_CHAT_TEMPLATE

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── 4. LoRA / 全量训练 ───────────────────────────────────
        if cfg.full_finetune:
            logger.info("使用全量训练模式（不应用 LoRA）")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = 100.0 * trainable_params / total_params if total_params else 0.0
            logger.info(
                "trainable params: %s / %s (%.2f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                pct,
            )
        else:
            logger.info("使用 LoRA 训练模式")
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg.lora_target_modules,
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()

        # ── 5. 创建 SFTTrainer 并训练 ────────────────────────────
        logger.info(
            "开始 TRL 训练（model=%s, epochs=%d, batch=%d×%d, lr=%s）",
            cfg.base_model, cfg.epochs, cfg.batch_size,
            cfg.gradient_accumulation_steps, cfg.lr,
        )

        sft_trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(run_dir),
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.lr,
                warmup_ratio=cfg.warmup_ratio,
                lr_scheduler_type="cosine",
                bf16=True,
                max_length=cfg.max_length,
                packing=cfg.packing,
                assistant_only_loss=True,
                logging_steps=cfg.logging_steps,
                save_steps=cfg.save_steps,
                save_total_limit=cfg.save_total_limit,
                report_to="wandb" if wb is not None else "none",
                dataloader_num_workers=2,
                remove_unused_columns=False,
                seed=42,
            ),
            train_dataset=ds,
            processing_class=tok,
        )

        sft_trainer.train()

        # ── 6. 保存 checkpoint（merged LoRA / 全量模型）──────────
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        if cfg.full_finetune:
            sft_trainer.model.save_pretrained(str(best_dir))
            tok.save_pretrained(str(best_dir))
            logger.info("全量模型已保存: %s", best_dir)
        else:
            logger.info("LoRA 训练完成，开始合并并保存独立模型: %s", best_dir)
            merged_model = sft_trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(best_dir), safe_serialization=True)
            tok.save_pretrained(str(best_dir))
            logger.info("merged 全量模型已保存: %s", best_dir)

        # ── 7. 评估 ──────────────────────────────────────────────
        return self._post_train_eval(best_dir, wb)

    def _train_unsloth(
        self,
        dataset: NERDataset | None = None,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """unsloth 训练路径（FastLanguageModel + dataset_text_field）。"""
        import torch
        from datasets import Dataset
        from transformers import set_seed
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        cfg = self.config
        self.dataset = self._ensure_dataset(dataset)

        # ── 1. 准备目录 ──────────────────────────────────────────
        run_dir = Path(cfg.save_dir) / cfg.dataset_name / "trl_unsloth"
        best_dir = run_dir / "best"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── 2. 导出训练数据为 messages ────────────────────────────
        system_prompt = cfg.get_system_prompt()
        train_examples = list(self.dataset.iter_split(cfg.train_split))
        messages_data = examples_to_messages(train_examples, system_prompt)

        train_jsonl = run_dir / "train_messages.jsonl"
        with train_jsonl.open("w", encoding="utf-8") as f:
            for row in messages_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("训练数据已准备: %d 条 → %s", len(messages_data), train_jsonl)

        # ── 3. 加载 model + tokenizer（unsloth）────────────────
        set_seed(42)
        logger.info(
            "加载 model (unsloth, 4bit=%s): %s",
            cfg.load_in_4bit, cfg.base_model,
        )
        model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=cfg.load_in_4bit,
            full_finetuning=False,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ── 4. LoRA / 全量训练（unsloth 方式）────────────────────
        if cfg.full_finetune:
            logger.info("unsloth 使用全量训练模式（不应用 LoRA）")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = 100.0 * trainable_params / total_params if total_params else 0.0
            logger.info(
                "trainable params: %s / %s (%.2f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                pct,
            )
        else:
            logger.info("unsloth 使用 LoRA 训练模式")
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                target_modules=cfg.lora_target_modules,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

        # ── 5. 预格式化训练数据为 text 列 ────────────────────────
        # unsloth 使用 dataset_text_field="text"，
        # 需要预先将 messages 通过 chat template 转为纯文本。
        ds = Dataset.from_list([
            {"text": tok.apply_chat_template(
                row["messages"], tokenize=False,
            )}
            for row in messages_data
        ])

        # ── 6. 创建 SFTTrainer 并训练 ────────────────────────────
        logger.info(
            "开始 TRL+unsloth 训练（model=%s, 4bit=%s, epochs=%d, batch=%d×%d, lr=%s）",
            cfg.base_model, cfg.load_in_4bit, cfg.epochs, cfg.batch_size,
            cfg.gradient_accumulation_steps, cfg.lr,
        )

        sft_trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(run_dir),
                dataset_text_field="text",
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.lr,
                warmup_ratio=cfg.warmup_ratio,
                lr_scheduler_type="cosine",
                bf16=True,
                max_length=cfg.max_length,
                packing=cfg.packing,
                logging_steps=cfg.logging_steps,
                save_steps=cfg.save_steps,
                save_total_limit=cfg.save_total_limit,
                report_to="wandb" if wb is not None else "none",
                dataloader_num_workers=2,
                seed=42,
            ),
            train_dataset=ds,
            processing_class=tok,
        )

        sft_trainer.train()

        # ── 7. 保存 checkpoint（merged LoRA / 全量模型）──────────
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        if cfg.full_finetune:
            sft_trainer.model.save_pretrained(str(best_dir))
            tok.save_pretrained(str(best_dir))
            logger.info("全量模型已保存: %s", best_dir)
        else:
            logger.info("LoRA 训练完成，开始合并并保存独立模型: %s", best_dir)
            merged_model = sft_trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(best_dir), safe_serialization=True)
            tok.save_pretrained(str(best_dir))
            logger.info("merged 全量模型已保存: %s", best_dir)

        # ── 8. 评估 ──────────────────────────────────────────────
        return self._post_train_eval(best_dir, wb)

    def _post_train_eval(
        self,
        best_dir: Path,
        wb: WandbLogger | None = None,
    ) -> tuple[Path, list[NERMetrics], NERMetrics | None]:
        """训练后 test 集评估（标准/unsloth 共用）。"""
        cfg = self.config
        test_metrics: NERMetrics | None = None

        if cfg.test_split and cfg.test_split in self.dataset.splits():
            logger.info("加载 checkpoint 进行 test 集评估...")
            self.load_from_checkpoint(best_dir)

            test_metrics = self.evaluate(
                data_path=Path(cfg.data_dir) / f"{self.dataset.name}_{cfg.test_split}.tsv",
                split=cfg.test_split,
                epoch=None,
            )
            test_metrics.model_dir = str(best_dir)
            logger.info("[Test]\n%s", test_metrics)

            if wb is not None:
                wb.log_metrics(test_metrics)

        return best_dir, [], test_metrics

    def validate(
        self,
        split: str = "test",
        dataset: NERDataset | None = None,
        model_dir: str | None = None,
    ) -> NERMetrics:
        """独立评估：加载 checkpoint，在指定 split 上生成并计算指标。"""
        cfg = self.config
        run_subdir = self._run_subdir()
        resolved = Path(model_dir) if model_dir else (
            Path(cfg.save_dir) / cfg.dataset_name / run_subdir / "best"
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

        self.load_from_checkpoint(resolved)

        tsv_path = Path(cfg.data_dir) / f"{self.dataset.name}_{split}.tsv"
        if not tsv_path.exists():
            self.dataset.export_tsv(output_dir=cfg.data_dir, splits=[split])

        metrics = self.evaluate(tsv_path, split, epoch=None)
        metrics.model_dir = str(resolved)
        return metrics

    # ── 内部工具 ──────────────────────────────────────────────────────

    def _predict_single(self, ex: NERExample, system_prompt: str) -> list[str]:
        """对单条样本做推理，返回完整形式的 BIO 标签列表。"""
        import torch

        tokenizer = self._tokenizer
        model = self._model
        cfg = self.config

        # 构建 ChatML 格式的 messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.query},
        ]

        # 使用 tokenizer 的 chat template 编码
        # 推理时 tokenizer 使用的是原始 chat template（不含 {% generation %}）
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # 生成
        outputs = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            do_sample=cfg.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 解码生成部分（去除 input 部分）
        generated_ids = outputs[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 解析缩写标签并还原为完整形式（与 gold labels 对齐）
        raw_tags = parse_bio_output(output_text, len(ex.tokens))
        return [abbrev_to_label(t) for t in raw_tags]

    def _grpo_reward(
        self,
        prompts,
        completions,
        gold_labels,
        query,
        **kwargs,
    ):
        """
        GRPO reward 规则：
        1) 输出格式错误：-1
        2) O 标签逐 token match：每个 +0.1
        3) 非 O 标签逐 token match（仅要求 entity type matched）：
           - 预测 B-* 且 gold 为同类型 B-*：每个 +0.2
           - 预测 I-* 且 gold 为同类型 I-* 或 B-*：每个 +0.1
        """
        rewards: list[float] = []
        for completion, labels, q in zip(completions, gold_labels, query):
            pred_text = self._extract_completion_text(completion)
            score = self._score_single_grpo_output(pred_text, labels, q)
            rewards.append(float(score))
        return rewards

    def _extract_completion_text(self, completion) -> str:
        """从 GRPO completion 对象中提取文本。"""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            last = completion[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
            if isinstance(last, str):
                return last
        if isinstance(completion, dict):
            if "content" in completion:
                return str(completion["content"])
            if "text" in completion:
                return str(completion["text"])
        return str(completion)

    def _score_single_grpo_output(
        self,
        pred_text: str,
        gold_labels: list[str],
        query: str,
    ) -> float:
        num_tokens = len(query.split())

        parsed_raw = parse_bio_output(pred_text, num_tokens)

        # 格式校验：必须是逗号分隔且 token 数对齐
        raw_parts = [p.strip() for p in pred_text.replace("\n", ",").split(",") if p.strip()]
        if len(raw_parts) != num_tokens:
            return -1.0

        valid_prefix = {"O", "B", "I"}
        for tag in raw_parts:
            if tag == "O":
                continue
            if "-" not in tag:
                return -1.0
            prefix, _ = tag.split("-", 1)
            if prefix not in valid_prefix:
                return -1.0

        pred_labels = [abbrev_to_label(t) for t in parsed_raw]

        o_matches = 0
        non_o_reward = 0.0
        for p, g in zip(pred_labels, gold_labels):
            if p == "O" and g == "O":
                o_matches += 1

            if p == "O" or g == "O":
                continue

            if "-" not in p or "-" not in g:
                continue

            p_prefix, p_type = p.split("-", 1)
            g_prefix, g_type = g.split("-", 1)
            if p_type != g_type:
                continue

            if p_prefix == "B" and g_prefix == "B":
                non_o_reward += 0.2
            elif p_prefix == "I" and g_prefix in {"I", "B"}:
                non_o_reward += 0.1

        score = 0.0
        score += 0.1 * o_matches
        score += non_o_reward
        return score

    def _resolve_eval_csv_path(self, split: str) -> Path:
        """解析 evaluate 明细 CSV 输出路径。"""
        cfg = self.config
        if cfg.eval_csv_path:
            return Path(cfg.eval_csv_path)
        return Path(cfg.save_dir) / cfg.dataset_name / self._run_subdir() / f"eval_{split}.csv"

    def _write_eval_csv(self, csv_path: Path, rows: list[dict[str, object]]) -> None:
        """写出 evaluate 明细 CSV。"""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "input_case",
            "expected_output",
            "actual_output",
            "is_case_correct",
            "tp_entity_count",
            "expected_entity_count",
            "actual_entity_count",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
