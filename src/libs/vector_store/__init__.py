"""VectorStore 抽象层统一导出。

上层代码推荐从这里导入基础抽象与工厂，
避免直接依赖内部文件路径。
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.chroma_store import ChromaStore, ChromaStoreError
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# 模块加载时注册默认 provider，确保上层开箱即用。
if "chroma" not in VectorStoreFactory._PROVIDERS:
    VectorStoreFactory.register_provider("chroma", ChromaStore)

__all__ = [
    "BaseVectorStore",
    "VectorStoreFactory",
    "ChromaStore",
    "ChromaStoreError",
]
