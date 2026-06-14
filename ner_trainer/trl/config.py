"""
ner_trainer/trl/config.py

TRLTrainConfig — TRL 后端专属训练配置（支持 LoRA / 全量训练）。

策略：将 NER 任务转换为生成式任务，使用 MiniCPM5（ChatML 模板）
输入 query 文本，输出逗号分隔的缩写 BIO 标签序列。

训练方式：使用 TRL 的 SFTTrainer + PEFT LoRA，配合 assistant_only_loss
只对 assistant 回复计算 loss，通过 chat template 中的 {% generation %} 块标记。

可选 unsloth 加速：使用 FastLanguageModel 替代标准 HF 加载，
支持 QLoRA 4-bit 量化，显存降低约 2 倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ner_trainer.config import BaseTrainConfig
from ner_trainer.gen_utils import build_system_prompt


# ── TRL 训练专用 chat template ────────────────────────────────────
# 在标准 ChatML 模板基础上加入 {% generation %} 块，
# 使 SFTConfig(assistant_only_loss=True) 能正确 mask 非 assistant token。
# token 序列与模型原始 chat template 完全一致，训练后的 adapter 推理时兼容。
# 注意：unsloth 模式下不使用此模板，改用 dataset_text_field="text" 方式。

TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


@dataclass
class TRLTrainConfig(BaseTrainConfig):
    """
    TRL 微调配置。

    训练流程：
    1. 将 QueryNER 转为 OpenAI messages JSONL（缩写标签）
    2. 加载 base model（可选 LoRA）
    3. patch chat template 加入 {% generation %} 块
    4. 使用 SFTTrainer(assistant_only_loss=True) 训练
    5. 保存 LoRA adapter
    6. 加载 base + adapter 做推理评估

    依赖（标准模式，与现有 transformers<4.47 冲突，需独立安装）：
        pip install "trl>=0.21" "peft>=0.13" "transformers>=5.6,<6" datasets accelerate

    依赖（unsloth 模式）：
        pip install "unsloth>=2026.5"
        pip install --force-reinstall "transformers==4.57.3"
    """

    # ── 模型 ──────────────────────────────────────────────────────
    base_model: str = "openbmb/MiniCPM5-1B"
    """HuggingFace 模型名称或本地路径。"""

    # ── unsloth 选项 ──────────────────────────────────────────────
    use_unsloth: bool = False
    """
    是否使用 unsloth 加速。开启后：
    - 使用 FastLanguageModel 替代 AutoModelForCausalLM
    - 使用 FastLanguageModel.get_peft_model() 替代 PEFT LoraConfig
    - 推理时调用 FastLanguageModel.for_inference() 获得 2× 加速
    - 训练数据使用 dataset_text_field="text"（预格式化）替代 {% generation %} 模板
    - 支持 load_in_4bit QLoRA
    """

    load_in_4bit: bool = False
    """
    是否使用 4-bit 量化加载（QLoRA）。仅在 use_unsloth=True 时有效。
    开启后显存需求大幅降低（约 2×），适合 24GB 及以下显卡。
    """

    # ── 训练模式 ──────────────────────────────────────────────────
    trl_mode: str = "sft"
    """
    TRL 训练模式：
    - "sft": 监督微调（SFTTrainer）
    - "grpo": 强化学习（GRPOTrainer）
    """

    full_finetune: bool = False
    """
    是否全量训练（不使用 LoRA）。
    - False: LoRA/QLoRA（默认，显存占用更低）
    - True: 全量训练（参数全部可训练，显存占用更高）
    """

    # ── LoRA 参数（仅 full_finetune=False 时生效）─────────────────
    lora_r: int = 16
    """LoRA rank。"""

    lora_alpha: int = 32
    """LoRA alpha scaling。"""

    lora_dropout: float = 0.05
    """LoRA dropout 概率。"""

    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    """LoRA 目标模块。"""

    # ── 训练超参 ──────────────────────────────────────────────────
    batch_size: int = 4
    """每设备 batch 大小。"""

    gradient_accumulation_steps: int = 4
    """梯度累积步数（有效 batch = batch_size x gradient_accumulation_steps）。"""

    lr: float = 2e-4
    """学习率。"""

    warmup_ratio: float = 0.03
    """Warmup 比例。"""

    max_length: int = 2048
    """Tokenizer 最大序列长度。"""

    packing: bool = False
    """是否启用 sequence packing（短序列任务通常关闭）。"""

    # ── 保存 & 日志 ───────────────────────────────────────────────
    save_steps: int = 200
    """每多少步保存一次 checkpoint。"""

    save_total_limit: int = 2
    """最多保留几个中间 checkpoint。"""

    logging_steps: int = 10
    """每多少步输出一次 training loss。"""

    # ── GRPO 专属超参 ────────────────────────────────────────────
    grpo_num_generations: int = 4
    """每个 prompt 采样的候选 completion 数。"""

    grpo_max_prompt_length: int = 1024
    """GRPO prompt 最大长度。"""

    grpo_max_completion_length: int = 128
    """GRPO completion 最大长度。"""

    grpo_beta: float = 0.0
    """GRPO KL 惩罚系数。"""

    # ── 评估输出 ──────────────────────────────────────────────────
    eval_csv_path: str = ""
    """
    evaluate() 明细 CSV 输出路径。
    为空时自动写入: <save_dir>/<dataset_name>/trl/eval_<split>.csv
    """

    # ── 生成参数（评估用）────────────────────────────────────────
    max_new_tokens: int = 128
    """推理时最大生成 token 数。"""

    temperature: float = 0.0
    """推理温度（0 = greedy）。"""

    # ── 系统提示词 ────────────────────────────────────────────────
    system_prompt: str = ""
    """
    自定义 system prompt。为空时使用内置的 QueryNER 默认 prompt。
    """

    def get_system_prompt(self) -> str:
        """获取最终使用的 system prompt。"""
        if self.system_prompt:
            return self.system_prompt
        return build_system_prompt()
