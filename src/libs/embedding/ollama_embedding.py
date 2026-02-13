"""Ollama Embedding 提供者实现（基于 LlamaIndex）。

这个模块只做两件事：
1. 把项目统一的 `BaseEmbedding` 接口，映射到 LlamaIndex 的 OllamaEmbedding。
2. 统一错误包装，确保上层拿到稳定且可读的异常。
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """Ollama Embedding 调用失败时抛出的统一异常。"""


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider。"""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_DIMENSION = 768

    @staticmethod
    def _load_llamaindex_class() -> Any:
        """延迟加载 LlamaIndex 的 OllamaEmbedding 类。"""

        try:
            module = import_module("llama_index.embeddings.ollama")
        except Exception as error:  # noqa: BLE001 - 统一包装导入失败
            raise OllamaEmbeddingError(
                "llama-index-embeddings-ollama is required for OllamaEmbedding"
            ) from error

        provider_class = getattr(module, "OllamaEmbedding", None)
        if provider_class is None:
            raise OllamaEmbeddingError(
                "Failed to load LlamaIndex OllamaEmbedding class"
            )
        return provider_class

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        """把配置值规范为可选字符串。"""

        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return None

    @staticmethod
    def _as_optional_int(value: Any) -> int | None:
        """把配置值规范为可选整数。"""

        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def __init__(
        self,
        settings: Any,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Ollama Embedding provider。"""

        self.settings = settings

        embedding_settings = getattr(settings, "embedding", None)

        model_value = self._as_optional_str(getattr(embedding_settings, "model", None))
        self.model = model_value or self.DEFAULT_MODEL

        dimensions_value = getattr(embedding_settings, "dimensions", None)
        configured_dimension = self._as_optional_int(dimensions_value)
        self.dimension = configured_dimension or self.DEFAULT_DIMENSION

        settings_base_url = self._as_optional_str(getattr(embedding_settings, "base_url", None))
        configured_base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or settings_base_url
        self.base_url = str(configured_base_url or self.DEFAULT_BASE_URL).rstrip("/")

        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """把文本列表转换成向量列表。"""

        self.validate_texts(texts)

        model_name = str(kwargs.get("model", self.model))
        base_url = str(kwargs.get("base_url", self.base_url)).rstrip("/")
        timeout_value = float(kwargs.get("timeout", self.timeout))

        client_kwargs: dict[str, Any] = {"timeout": timeout_value}
        override_client_kwargs = kwargs.get("client_kwargs")
        if isinstance(override_client_kwargs, dict):
            client_kwargs.update(override_client_kwargs)

        try:
            llamaindex_ollama_embedding = self._load_llamaindex_class()

            embedder = llamaindex_ollama_embedding(
                model_name=model_name,
                base_url=base_url,
                client_kwargs=client_kwargs,
            )
            raw_vectors = embedder.get_text_embedding_batch(texts)
        except OllamaEmbeddingError:
            raise
        except Exception as error:  # noqa: BLE001 - 统一包装 SDK 异常
            raise OllamaEmbeddingError(
                f"Ollama Embeddings API call failed: {error}"
            ) from error

        vectors = [list(vector) for vector in raw_vectors]
        if len(vectors) != len(texts):
            raise OllamaEmbeddingError(
                "Output length mismatch: embedding result count does not match input count"
            )

        return vectors

    def get_dimension(self) -> int | None:
        """返回当前 provider 的向量维度。"""

        return self.dimension
