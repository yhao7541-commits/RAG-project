"""VectorStore 提供者工厂。

职责：
1) 管理 provider 注册表（provider 名称 -> provider 类）。
2) 按配置 `settings.vector_store.provider` 创建具体向量库实例。
3) 在配置错误或实例化失败时，输出清晰的错误信息。
"""

from __future__ import annotations

from typing import Any

from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """基于注册表的向量存储工厂。"""

    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_class: type[BaseVectorStore],
    ) -> None:
        """注册一个向量存储 provider。

        参数说明：
        - provider_name: provider 名称（例如 `chroma`）。
        - provider_class: 对应实现类，必须继承 BaseVectorStore。
        """

        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError("Provider class must inherit from BaseVectorStore")

        cls._PROVIDERS[normalized_name] = provider_class

    @classmethod
    def create(cls, settings: Any, **overrides: Any) -> BaseVectorStore:
        """根据配置创建向量存储实例。

        参数说明：
        - settings: 全局配置对象，要求包含 `settings.vector_store.provider`。
        - **overrides: 单次覆盖构造参数（例如测试注入 mock client）。
        """

        # 步骤 1：读取 provider 配置。
        vector_store_settings = getattr(settings, "vector_store", None)
        provider_raw = getattr(vector_store_settings, "provider", None)

        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please set it in settings.yaml"
            )

        provider_name = provider_raw.strip().lower()

        # 步骤 2：查找注册表并在缺失时输出可用列表。
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = cls.list_providers()
            available_text = ", ".join(available) if available else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_raw}'. "
                f"Available providers: {available_text}"
            )

        # 步骤 3：实例化 provider，必要时包装底层错误，便于上层定位。
        try:
            provider_constructor: Any = provider_class
            return provider_constructor(settings=settings, **overrides)
        except Exception as error:  # noqa: BLE001 - 统一包装底层初始化错误
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {error}"
            ) from error

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 名称列表（字母序）。

        排序后的列表适合直接展示在错误信息里，便于快速定位配置问题。
        """

        return sorted(cls._PROVIDERS.keys())
