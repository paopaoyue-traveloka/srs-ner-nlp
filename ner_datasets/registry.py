"""
数据集注册表 — 统一管理所有可用 NERDataset。

用法:
    from datasets import registry

    registry.list()           # 列出所有已注册数据集
    ds = registry.get("queryner")
    ds.load()
"""

from __future__ import annotations

from typing import Type

from .base import NERDataset


class DatasetRegistry:
    """轻量级数据集注册表，支持按名称查找和列举。"""

    def __init__(self) -> None:
        self._registry: dict[str, Type[NERDataset]] = {}

    def register(self, cls: Type[NERDataset]) -> Type[NERDataset]:
        """注册一个 NERDataset 子类（可用作装饰器）。"""
        if not cls.name:
            raise ValueError(f"{cls.__name__} 必须设置 name 类属性")
        self._registry[cls.name] = cls
        return cls

    def list(self) -> list[dict[str, str]]:
        """返回所有已注册数据集的简要信息列表。"""
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "hf_repo": cls.hf_repo,
            }
            for cls in self._registry.values()
        ]

    def get(self, name: str) -> NERDataset:
        """按名称实例化并返回数据集（未加载状态）。"""
        if name not in self._registry:
            available = ", ".join(self._registry) or "(无)"
            raise KeyError(
                f"未知数据集 '{name}'，可用: {available}"
            )
        return self._registry[name]()

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# 全局单例注册表
registry = DatasetRegistry()

# ── 注册内置数据集 ────────────────────────────────────────────────
from .queryner import QueryNERDataset  # noqa: E402

registry.register(QueryNERDataset)
