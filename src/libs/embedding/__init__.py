"""Embedding 抽象层对外导出。

该文件统一导出最常用的基础类型，
让上层调用方用稳定路径导入。
"""

from src.libs.embedding.azure_embedding import AzureEmbedding, AzureEmbeddingError
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError
from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError

# 模块加载时注册默认 provider，减少上层样板代码。
if "openai" not in EmbeddingFactory._PROVIDERS:
    EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
if "azure" not in EmbeddingFactory._PROVIDERS:
    EmbeddingFactory.register_provider("azure", AzureEmbedding)
if "ollama" not in EmbeddingFactory._PROVIDERS:
    EmbeddingFactory.register_provider("ollama", OllamaEmbedding)

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "OpenAIEmbeddingError",
    "AzureEmbedding",
    "AzureEmbeddingError",
    "OllamaEmbedding",
    "OllamaEmbeddingError",
]
