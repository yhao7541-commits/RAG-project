"""OpenAI Embedding 提供者实现（基于 LlamaIndex）。

这个模块负责把“文本列表”转换成“向量列表”。
它遵循项目统一的 `BaseEmbedding` 接口，让上层不需要关心底层供应商细节。

为什么要单独封装：
1. 统一输入校验和错误信息（可读、可定位）。
2. 统一返回结构（list[list[float]]），便于后续检索流程直接消费。
3. 支持配置化切换模型与维度。
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """OpenAI Embedding 调用失败时抛出的统一异常。"""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding provider。"""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT = 60.0

    # 常见模型维度映射（用于 get_dimension 的合理推断）。
    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @staticmethod
    def _load_llamaindex_class() -> Any:
        """延迟加载 LlamaIndex OpenAIEmbedding，避免硬依赖导致导入期失败。"""

        try:
            module = import_module("llama_index.embeddings.openai")
        except Exception as error:  # noqa: BLE001 - 统一包装导入失败
            raise OpenAIEmbeddingError(
                "llama-index-embeddings-openai is required for OpenAIEmbedding"
            ) from error

        provider_class = getattr(module, "OpenAIEmbedding", None)
        if provider_class is None:
            raise OpenAIEmbeddingError(
                "Failed to load LlamaIndex OpenAIEmbedding class"
            )
        return provider_class

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        """把可能来自配置对象的值规范为可选字符串。"""

        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return None

    @staticmethod
    def _as_optional_int(value: Any) -> int | None:
        """把可能来自配置对象的值规范为可选整数。"""

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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI Embedding provider。

        参数说明：
        - settings: 全局配置对象，读取 `settings.embedding.*`。
        - api_key: 显式 API Key（优先级最高）。
        - base_url: OpenAI 兼容网关地址（可选）。
        - timeout: 请求超时秒数。
        - **kwargs: 预留扩展参数位。
        """

        self.settings = settings

        embedding_settings = getattr(settings, "embedding", None)
        model_value = self._as_optional_str(getattr(embedding_settings, "model", None))
        self.model = model_value or "text-embedding-3-small"
        dimensions_value = getattr(embedding_settings, "dimensions", None)
        self.dimensions = self._as_optional_int(dimensions_value)

        # API Key 优先级：参数 > settings.embedding.api_key > 环境变量
        settings_api_key = self._as_optional_str(getattr(embedding_settings, "api_key", None))
        api_key_value = (
            api_key
            or settings_api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key_value:
            raise ValueError("OpenAI API key not provided")
        self.api_key = str(api_key_value)

        settings_base_url = self._as_optional_str(getattr(embedding_settings, "base_url", None))
        configured_base_url = base_url or settings_base_url
        self.base_url = str(configured_base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """把文本列表转换成向量列表。

        参数说明：
        - texts: 待向量化文本列表。
        - trace: 追踪上下文（可选，当前实现未直接使用）。
        - **kwargs: 单次覆盖参数（如 model、dimensions）。
        """

        # 步骤 1：统一输入校验。
        self.validate_texts(texts)

        # 步骤 2：解析单次覆盖参数。
        model_name = str(kwargs.get("model", self.model))
        dimensions_override = kwargs.get("dimensions", self.dimensions)
        dimensions = int(dimensions_override) if dimensions_override is not None else None

        try:
            llamaindex_openai_embedding = self._load_llamaindex_class()

            embedder_kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": self.api_key,
                "api_base": self.base_url,
                "timeout": self.timeout,
            }
            # dimensions 只有在非 None 时才传，避免与某些模型不兼容。
            if dimensions is not None:
                embedder_kwargs["dimensions"] = dimensions

            embedder = llamaindex_openai_embedding(**embedder_kwargs)
            raw_vectors = embedder.get_text_embedding_batch(texts)
        except OpenAIEmbeddingError:
            raise
        except Exception as error:  # noqa: BLE001 - 统一包装 SDK 异常
            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed: {error}"
            ) from error

        # 步骤 3：提取向量并校验输出长度与输入一致。
        vectors = [list(vector) for vector in raw_vectors]
        if len(vectors) != len(texts):
            raise OpenAIEmbeddingError(
                "Output length mismatch: embedding result count does not match input count"
            )

        return vectors

    def get_dimension(self) -> int | None:
        """返回当前 embedding 维度。

        优先级：
        1) 显式配置的 dimensions
        2) 已知模型默认维度
        3) 无法推断则返回 None
        """

        if self.dimensions is not None:
            return self.dimensions
        return self.MODEL_DIMENSIONS.get(self.model)
