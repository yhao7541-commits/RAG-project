"""Azure OpenAI Embedding 提供者实现。

实现目标：
1. 复用 OpenAI Embedding 的核心流程（输入校验、输出校验、维度策略）。
2. 增加 Azure 特有配置（endpoint/api_version/deployment_name）。
3. 对外保持与 OpenAIEmbedding 相同的接口行为。
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


class AzureEmbeddingError(OpenAIEmbeddingError):
    """Azure Embedding 调用失败时抛出的统一异常。"""


class AzureEmbedding(OpenAIEmbedding):
    """Azure OpenAI Embedding provider。"""

    DEFAULT_API_VERSION = "2024-02-01"

    @staticmethod
    def _load_llamaindex_class() -> Any:
        """延迟加载 LlamaIndex AzureOpenAIEmbedding。"""

        try:
            module = import_module("llama_index.embeddings.azure_openai")
        except Exception as error:  # noqa: BLE001 - 统一包装导入失败
            raise AzureEmbeddingError(
                "llama-index-embeddings-azure-openai is required for AzureEmbedding"
            ) from error

        provider_class = getattr(module, "AzureOpenAIEmbedding", None)
        if provider_class is None:
            raise AzureEmbeddingError(
                "Failed to load LlamaIndex AzureOpenAIEmbedding class"
            )
        return provider_class

    def __init__(
        self,
        settings: Any,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure Embedding provider。

        参数说明：
        - api_key: 优先使用显式参数，其次环境变量。
        - azure_endpoint: Azure 资源地址。
        - api_version: Azure API 版本。
        - timeout: 请求超时秒数。
        """

        # 先复用父类逻辑，建立 model/dimensions 与基础字段。
        super().__init__(
            settings,
            api_key="_placeholder_",  # 先占位，后面按 Azure 规则覆盖
            base_url="https://placeholder.local",
            timeout=timeout,
            **kwargs,
        )

        embedding_settings = getattr(settings, "embedding", None)

        # Azure Key 优先级：参数 > settings.embedding.api_key > AZURE_OPENAI_API_KEY > OPENAI_API_KEY
        settings_api_key = self._as_optional_str(getattr(embedding_settings, "api_key", None))
        api_key_value = (
            api_key
            or settings_api_key
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key_value:
            raise ValueError("Azure OpenAI API key not provided")
        self.api_key = str(api_key_value)

        settings_endpoint = self._as_optional_str(getattr(embedding_settings, "azure_endpoint", None))
        endpoint_value = (
            azure_endpoint
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or settings_endpoint
        )
        if not endpoint_value:
            raise ValueError("Azure OpenAI endpoint not provided")
        self.azure_endpoint = str(endpoint_value)

        settings_version = self._as_optional_str(getattr(embedding_settings, "api_version", None))
        version_value = api_version or settings_version
        self.api_version = str(version_value or self.DEFAULT_API_VERSION)

        deployment_value = self._as_optional_str(
            getattr(embedding_settings, "deployment_name", None)
        )
        self.deployment_name = str(deployment_value or self.model)

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """调用 Azure OpenAI Embeddings API。"""

        self.validate_texts(texts)

        model_name = str(kwargs.get("model", self.model))
        deployment_name = str(kwargs.get("deployment_name", self.deployment_name))
        dimensions_override = kwargs.get("dimensions", self.dimensions)
        dimensions = int(dimensions_override) if dimensions_override is not None else None

        try:
            llamaindex_azure_embedding = self._load_llamaindex_class()

            embedder_kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": self.api_key,
                "azure_endpoint": self.azure_endpoint,
                "api_version": self.api_version,
                "azure_deployment": deployment_name,
            }
            if dimensions is not None:
                embedder_kwargs["dimensions"] = dimensions

            embedder = llamaindex_azure_embedding(**embedder_kwargs)
            raw_vectors = embedder.get_text_embedding_batch(texts)
        except Exception as error:  # noqa: BLE001
            raise AzureEmbeddingError(
                f"Azure OpenAI Embeddings API call failed: {error}"
            ) from error

        vectors = [list(vector) for vector in raw_vectors]
        if len(vectors) != len(texts):
            raise AzureEmbeddingError(
                "Output length mismatch: embedding result count does not match input count"
            )

        return vectors

    def get_dimension(self) -> int | None:
        """返回 Azure embedding 维度。

        优先级：
        1) 配置 dimensions
        2) deployment_name 精确匹配模型名
        3) deployment_name 包含已知模型关键字
        4) 无法识别则返回 None
        """

        if self.dimensions is not None:
            return self.dimensions

        deployment = self.deployment_name
        if deployment in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[deployment]

        deployment_lower = deployment.lower()
        for model_name, dimension in self.MODEL_DIMENSIONS.items():
            if model_name in deployment_lower:
                return dimension

        return None
