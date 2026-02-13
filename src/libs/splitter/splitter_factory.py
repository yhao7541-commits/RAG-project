"""Splitter 提供者工厂。

职责：
1) 维护 provider 注册表（名称 -> 类）。
2) 按 `settings.ingestion.splitter` 创建具体切分器实例。
3) 在配置错误时给出清晰、可执行的报错信息。
"""

from __future__ import annotations

from typing import Any

from src.libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """基于注册表的 Splitter 工厂。"""

    _PROVIDERS: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_class: type[BaseSplitter],
    ) -> None:
        """注册切分器实现类。

        参数说明：
        - provider_name: 切分器名称（如 `recursive`）。
        - provider_class: 切分器类，必须继承 BaseSplitter。
        """

        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, BaseSplitter):
            raise ValueError("Provider class must inherit from BaseSplitter")

        cls._PROVIDERS[normalized_name] = provider_class

    @classmethod
    def create(cls, settings: Any, **overrides: Any) -> BaseSplitter:
        """根据配置创建切分器实例。

        参数说明：
        - settings: 全局配置对象，要求包含 `settings.ingestion.splitter`。
        - **overrides: 单次覆盖构造参数（常用于测试或实验）。
        """

        # 步骤 1：读取配置。这里要求 settings.ingestion.splitter 必须存在。
        ingestion_settings = getattr(settings, "ingestion", None)
        splitter_raw = getattr(ingestion_settings, "splitter", None)

        if not isinstance(splitter_raw, str) or not splitter_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.ingestion.splitter. "
                "Please set splitter provider in settings.yaml"
            )

        splitter_name = splitter_raw.strip().lower()

        # 步骤 2：按名称查找注册表；若未注册则提示可用项与后续任务编号。
        splitter_class = cls._PROVIDERS.get(splitter_name)
        if splitter_class is None:
            available_providers = cls.list_providers()
            available_text = ", ".join(available_providers) if available_providers else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{splitter_raw}'. "
                f"Available providers: {available_text}. "
                "If you need default recursive splitter support, complete B7.5 first."
            )

        # 步骤 3：实例化并返回。这里允许传入 overrides 覆盖构造参数。
        splitter_constructor: Any = splitter_class
        return splitter_constructor(settings=settings, **overrides)

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 名称列表（字母序）。

        使用字母序可避免“注册顺序不同导致日志顺序变化”。
        """

        return sorted(cls._PROVIDERS.keys())
