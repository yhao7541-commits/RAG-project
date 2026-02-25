"""Reranker 抽象层统一导出。

为什么建议从这里导入：
1. 导入路径稳定，不依赖内部文件结构。
2. 业务代码更容易读（看到 `from src.libs.reranker import ...` 就知道是重排层）。
"""

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.cross_encoder_reranker import (
    CrossEncoderRerankError,
    CrossEncoderReranker,
)
from src.libs.reranker.llm_reranker import LLMReranker, LLMRerankError
from src.libs.reranker.reranker_factory import RerankerFactory

# 模块加载时注册默认 provider，减少上层样板代码。
if "llm" not in RerankerFactory._PROVIDERS:
    RerankerFactory.register_provider("llm", LLMReranker)
if "cross_encoder" not in RerankerFactory._PROVIDERS:
    RerankerFactory.register_provider("cross_encoder", CrossEncoderReranker)

__all__ = [
    "BaseReranker",
    "NoneReranker",
    "RerankerFactory",
    "CrossEncoderReranker",
    "CrossEncoderRerankError",
    "LLMReranker",
    "LLMRerankError",
]
