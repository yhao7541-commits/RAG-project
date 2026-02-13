"""Splitter 抽象层对外导出。

为了让上层代码导入路径稳定，这里统一导出基础抽象和工厂类。
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

__all__ = ["BaseSplitter", "SplitterFactory"]
