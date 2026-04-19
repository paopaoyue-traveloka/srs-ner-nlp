"""
ner_datasets/ — NER 数据集封装模块

结构:
    base.py       — 抽象基类 NERDataset
    queryner.py   — QueryNER 实现
    registry.py   — 数据集注册表
    __init__.py   — 公共导出
"""

from .registry import registry
from .base import NERDataset
from .queryner import QueryNERDataset

__all__ = ["registry", "NERDataset", "QueryNERDataset"]
