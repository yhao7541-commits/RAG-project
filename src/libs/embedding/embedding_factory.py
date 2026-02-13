"""Embedding 提供者工厂。

核心作用：
1) 维护“provider 名称 -> provider 类”的注册表。
2) 根据 `settings.embedding.provider` 动态创建具体实例。
3) 给出可读的错误信息，帮助快速排查配置问题。
"""

from __future__ import annotations

from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """基于注册表的 Embedding 工厂。

    使用方式（示意）：
    1. 在启动时注册 provider：
       `EmbeddingFactory.register_provider("openai", OpenAIEmbedding)`
    2. 在运行时按配置创建实例：
       `embedding = EmbeddingFactory.create(settings)`
    """

    # 注册表：key 为 provider 名称（统一小写），value 为 provider 类。
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_class: type[BaseEmbedding],
    ) -> None:
        """注册一个 Embedding 提供者。

        参数:
        - provider_name: 提供者名称，例如 openai / azure / ollama。
        - provider_class: 对应的实现类，必须继承 BaseEmbedding。

        参数设计说明：
        - 名称统一小写存储，避免配置文件里大小写不一致导致无法匹配。
        """

        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError("Provider class must inherit from BaseEmbedding")

        cls._PROVIDERS[normalized_name] = provider_class

    @classmethod
    def create(cls, settings: Any, **overrides: Any) -> BaseEmbedding:
        """根据配置创建 Embedding 实例。

        参数:
        - settings: 全局配置对象，要求包含 `settings.embedding.provider`。
        - **overrides: 运行时覆盖参数，会透传给 provider 构造函数。

        参数设计说明：
        - `overrides` 用于临时覆盖（例如测试注入 mock client），
          不需要改全局配置就能控制单次行为。

        返回:
        - BaseEmbedding 的具体实现实例。

        异常策略:
        - 配置缺失: 抛 ValueError（可读提示 + 指向 settings.yaml）。
        - provider 未注册: 抛 ValueError，并附带可用 provider 列表。
        - provider 初始化失败: 抛 RuntimeError，并保留原始错误信息。
        """

        # 步骤 1：读取 provider 配置；如果配置缺失，给出可执行的修复提示。
        embedding_settings = getattr(settings, "embedding", None)
        provider_raw = getattr(embedding_settings, "provider", None)

        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.embedding.provider. "
                "Please set it in settings.yaml"
            )

        provider_name = provider_raw.strip().lower()

        # 步骤 2：查找注册表，若不存在则提示可用选项。
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available_providers = cls.list_providers()
            available_text = ", ".join(available_providers) if available_providers else "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_raw}'. "
                f"Available providers: {available_text}"
            )

        # 步骤 3：实例化具体 provider，失败时包装成统一错误，便于调用方处理。
        try:
            # 说明：各 provider 的构造参数可能不同，这里采用动态分发。
            provider_constructor: Any = provider_class
            return provider_constructor(settings, **overrides)
        except Exception as error:  # noqa: BLE001 - 这里需要统一包装底层初始化错误
            raise RuntimeError(
                f"Failed to instantiate Embedding provider '{provider_name}': {error}"
            ) from error

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 列表（按字母排序，便于稳定展示）。"""

        return sorted(cls._PROVIDERS.keys())
