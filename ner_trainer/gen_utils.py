"""
ner_trainer/gen_utils.py

生成式 NER 共享工具 — 供 trl 等生成式后端共用。

包含：
- 实体类型缩写映射（17 种 → 17 个单字母）
- BIO 标签与缩写的相互转换
- 系统提示词（system prompt）构建
- BIO 标签序列 ↔ 实体 span 集合
- 模型文本输出 → BIO 标签解析
- NERExample → OpenAI messages 格式转换
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ner_datasets.base import NERExample


# ── 实体类型缩写映射 ──────────────────────────────────────────────
# 完整类型名 → 单字母缩写（推理时模型只输出缩写，大幅减少 token 数）
# 注意：不使用 O，因为 BIO 中 O 表示 Outside

FULL_TO_ABBREV: dict[str, str] = {
    "core_product_type": "P",   # Product type
    "creator":           "C",   # Creator
    "product_name":      "N",   # product Name
    "product_number":    "R",   # product numbeR
    "modifier":          "M",   # Modifier
    "color":             "L",   # coLor
    "material":          "T",   # maTerial
    "department":        "D",   # Department
    "occasion":          "A",   # occAsion
    "shape":             "S",   # Shape
    "content":           "X",   # conteXt
    "UoM":               "U",   # UoM
    "quantity":          "Q",   # Quantity
    "price":             "V",   # Value
    "condition":         "K",   # condition (K)
    "time":              "W",   # When
    "origin":            "G",   # oriGin
}

ABBREV_TO_FULL: dict[str, str] = {v: k for k, v in FULL_TO_ABBREV.items()}


def label_to_abbrev(label: str) -> str:
    """将完整 BIO 标签转为缩写形式：B-creator → B-C, I-UoM → I-U, O → O"""
    if label == "O":
        return "O"
    prefix, etype = label.split("-", 1)
    abbrev = FULL_TO_ABBREV.get(etype)
    if abbrev is None:
        return label  # 未知类型，保持原样
    return f"{prefix}-{abbrev}"


def abbrev_to_label(tag: str) -> str:
    """将缩写 BIO 标签还原为完整形式：B-C → B-creator, I-U → I-UoM, O → O"""
    if tag == "O":
        return "O"
    if "-" not in tag:
        return "O"
    prefix, code = tag.split("-", 1)
    full = ABBREV_TO_FULL.get(code)
    if full is None:
        return tag  # 未知缩写，保持原样（容错）
    return f"{prefix}-{full}"


# ── QueryNER 实体类型描述（用于 system prompt）──────────────────────
QUERYNER_ENTITY_TYPES: dict[str, str] = {
    "core_product_type": "the main product being searched (e.g., shirt, headphones, notebook)",
    "creator": "brand or author name (e.g., Nike, Apple, Samsung)",
    "product_name": "specific product model or series (e.g., AirPods Pro, Galaxy S23)",
    "product_number": "product model/version number (e.g., 6s, XR, 2080ti)",
    "modifier": "descriptive adjectives about the product (e.g., waterproof, portable, heavy duty)",
    "color": "color descriptors (e.g., red, navy blue, black)",
    "material": "material composition (e.g., cotton, stainless steel, leather)",
    "department": "target audience or demographic (e.g., women, kids, mens)",
    "occasion": "usage context or purpose (e.g., wedding, camping, office)",
    "shape": "physical form (e.g., round, rectangular, oval)",
    "content": "topic or subject matter (e.g., math, cooking, Harry Potter)",
    "UoM": "unit of measure (e.g., 16 oz, 1 gallon, pack of 2)",
    "quantity": "amount or count (e.g., 3 pack, set of 4)",
    "price": "price-related terms (e.g., under 50, cheap)",
    "condition": "product state or quality (e.g., used, refurbished, new)",
    "time": "time-related terms (e.g., 2023, vintage, retro)",
    "origin": "place of origin (e.g., Japanese, Italian, organic)",
}

# ── 系统提示词模板（使用缩写标签）─────────────────────────────────
QUERYNER_SYSTEM_PROMPT = """NER for e-commerce queries. Label each token with BIO tags.

## Entity Codes
{entity_types}

## Rules
- B-X: first token of entity X
- I-X: continuation of entity X
- O: not an entity
Output one tag per token, comma-separated. Count MUST match input tokens.

## Examples

Input: nike air force 1 running shoes men
Output: B-C,B-N,I-N,I-N,B-M,B-P,B-D

Input: 16 oz whey protein powder used
Output: B-U,I-U,B-X,B-P,I-P,B-K

Input: cotton waterproof t-shirt men
Output: B-T,B-M,B-P,B-D

Input: lego star wars set
Output: B-C,B-N,I-N,B-P

Input: japanese stainless steel chef knife set of 3
Output: B-G,B-T,I-T,B-M,B-P,B-Q,I-Q,I-Q"""


def build_system_prompt(entity_types: dict[str, str] | None = None) -> str:
    """构建系统提示词，使用缩写标签。"""
    types = entity_types or QUERYNER_ENTITY_TYPES
    # 格式：P=core_product_type (shirt, headphones)
    type_lines = []
    for full_name, desc in types.items():
        abbrev = FULL_TO_ABBREV.get(full_name, full_name)
        type_lines.append(f"- {abbrev}={full_name}: {desc}")
    return QUERYNER_SYSTEM_PROMPT.format(entity_types="\n".join(type_lines))


# ── BIO 解析工具 ──────────────────────────────────────────────────

def bio_to_entity_spans(labels: list[str]) -> set[tuple[str, int, int]]:
    """将 BIO 标签序列转换为实体集合：{(type, start, end_exclusive)}。"""
    spans: set[tuple[str, int, int]] = set()
    start = -1
    ent_type = ""
    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            if start >= 0:
                spans.add((ent_type, start, i))
            start = i
            ent_type = tag[2:]
        elif tag.startswith("I-"):
            if start < 0 or tag[2:] != ent_type:
                if start >= 0:
                    spans.add((ent_type, start, i))
                start = -1
                ent_type = ""
        else:
            if start >= 0:
                spans.add((ent_type, start, i))
                start = -1
                ent_type = ""
    if start >= 0:
        spans.add((ent_type, start, len(labels)))
    return spans


def parse_bio_output(output: str, num_tokens: int) -> list[str]:
    """
    解析模型生成的 BIO 标签输出。

    输出格式预期：逗号分隔的 BIO 标签（如 "B-C,B-N,I-N"）。
    对不合法或长度不匹配的输出做容错处理：
      - 输出标签多于 num_tokens：截断
      - 输出标签少于 num_tokens：后续填充 O
      - 无法解析：全部填 O

    返回的标签保持原样（可能是缩写形式如 B-C，也可能是完整形式如 B-creator），
    调用方需自行通过 abbrev_to_label() 转回完整形式再与 gold 对比。
    """
    output = output.strip()

    # 移除可能的前缀（如 "Output: "）
    for prefix in ("Output:", "output:", "Output :", "output :"):
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
            break

    if not output:
        return ["O"] * num_tokens

    tags = [t.strip() for t in output.split(",")]

    # 验证每个 tag 的合法性
    valid_tags = []
    for t in tags:
        if t == "O" or t.startswith("B-") or t.startswith("I-"):
            valid_tags.append(t)
        else:
            valid_tags.append("O")

    # 对齐长度
    if len(valid_tags) > num_tokens:
        valid_tags = valid_tags[:num_tokens]
    elif len(valid_tags) < num_tokens:
        valid_tags.extend(["O"] * (num_tokens - len(valid_tags)))

    return valid_tags


def examples_to_messages(
    examples: list[NERExample],
    system_prompt: str,
) -> list[dict]:
    """
    将 NERExample 列表转为 OpenAI messages 格式字典列表。

    labels 自动转为缩写形式（与 system prompt 中的示例一致）。
    返回 [{"messages": [...]}, ...] 格式，可直接用于 Dataset.from_list()。
    """
    rows = []
    for ex in examples:
        # ✅ 使用缩写标签（与 system prompt 示例一致）
        assistant_content = ",".join(label_to_abbrev(l) for l in ex.labels)
        row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex.query},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        rows.append(row)
    return rows
