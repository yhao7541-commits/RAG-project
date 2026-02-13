"""Reranker 工厂。

这个工厂负责三件事：
1. 注册不同的重排器实现。
2. 根据配置创建对应实例。
3. 在关闭重排或 provider=none 时自动回退到 NoneReranker。
"""

from __future__ import annotations

from typing import Any

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """基于注册表的重排器工厂。"""

    _PROVIDERS: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_class: type[BaseReranker],
    ) -> None:
        """注册重排器实现类。

        参数说明：
        - provider_name: 重排器名称（如 `llm`、`cross_encoder`）。
        - provider_class: 实现类，必须继承 BaseReranker。
        """

        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, BaseReranker):
            raise ValueError("Provider class must inherit from BaseReranker")

        cls._PROVIDERS[normalized_name] = provider_class

    @classmethod
    def create(cls, settings: Any, **overrides: Any) -> BaseReranker:
        """按 `settings.rerank` 配置创建重排器。

        回退规则：
        - `enabled = false` -> NoneReranker
        - `provider = none` -> NoneReranker

        参数说明：
        - settings: 全局配置对象。
        - **overrides: 运行时覆盖参数，会透传给具体 provider。
        """

        rerank_settings = getattr(settings, "rerank", None)
        if rerank_settings is None:
            raise ValueError(
                "Missing required configuration: settings.rerank.provider"
            )

        enabled_value = getattr(rerank_settings, "enabled", None)
        if enabled_value is False:
            return NoneReranker()

        provider_raw = getattr(rerank_settings, "provider", None)
        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.rerank.provider"
            )

        provider_name = provider_raw.strip().lower()
        if provider_name == "none":
            return NoneReranker()

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = cls.list_providers()
            available_text = ", ".join(available) if available else "none"
            raise ValueError(
                f"Unsupported Reranker provider: '{provider_raw}'. "
                f"Available providers: {available_text}"
            )

        provider_constructor: Any = provider_class
        return provider_constructor(settings, **overrides)

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回当前已注册 provider 名称（字母序）。"""

        return sorted(cls._PROVIDERS.keys())
