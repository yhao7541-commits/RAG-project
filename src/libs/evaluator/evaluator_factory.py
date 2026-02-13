"""Evaluator 工厂。

职责：
1. 注册评估器实现。
2. 按 `settings.evaluation` 动态创建对应评估器。
3. 在关闭评估时回退到 NoneEvaluator，保证调用方无需分支判断。
"""

from __future__ import annotations

from typing import Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """基于注册表的评估器工厂。"""

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {"custom": CustomEvaluator}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        provider_class: type[BaseEvaluator],
    ) -> None:
        """注册评估器实现类。

        参数说明：
        - provider_name: 评估器名称（例如 `custom`）。
        - provider_class: 实现类，必须继承 BaseEvaluator。
        """

        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError("Provider class must inherit from BaseEvaluator")

        cls._PROVIDERS[normalized_name] = provider_class

    @classmethod
    def create(cls, settings: Any, **overrides: Any) -> BaseEvaluator:
        """根据配置创建评估器实例。

        参数说明：
        - settings: 全局配置对象，读取 `settings.evaluation.*`。
        - **overrides: 运行时覆盖参数，透传给具体评估器实现。
        """

        evaluation_settings = getattr(settings, "evaluation", None)
        if evaluation_settings is None:
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider"
            )

        enabled_value = getattr(evaluation_settings, "enabled", None)
        if enabled_value is False:
            return NoneEvaluator()

        provider_raw = getattr(evaluation_settings, "provider", None)
        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider"
            )

        provider_name = provider_raw.strip().lower()
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = cls.list_providers()
            available_text = ", ".join(available) if available else "none"
            raise ValueError(
                f"Unsupported Evaluator provider: '{provider_raw}'. "
                f"Available providers: {available_text}"
            )

        provider_constructor: Any = provider_class
        provider_metrics = getattr(evaluation_settings, "metrics", None)
        return provider_constructor(metrics=provider_metrics, **overrides)

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 列表（字母序）。

        排序输出方便日志展示和错误提示比对。
        """

        return sorted(cls._PROVIDERS.keys())
