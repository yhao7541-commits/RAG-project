"""根据配置创建 LLM 提供者实例的工厂模块。

把它理解为“模型调度中心”：
- 注册阶段：告诉系统有哪些 provider 可用。
- 创建阶段：根据 `settings.llm.provider` 选择对应实现并实例化。

这样上层业务不需要 `if provider == "openai" ...` 这类分支逻辑。
"""

from __future__ import annotations

from typing import Any

from src.libs.llm.base_llm import BaseLLM


class LLMFactory:
    """基于注册表的 LLM 提供者工厂。"""

    _PROVIDERS: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseLLM]) -> None:
        """注册一个 LLM 提供者。

        参数说明：
        - name: provider 名称（例如 `openai`、`azure`）。
          设计上会自动转小写，避免大小写导致的配置失败。
        - provider_cls: provider 对应的类，必须继承 BaseLLM。
        """

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_cls, BaseLLM):
            raise ValueError("Provider class must inherit from BaseLLM")

        cls._PROVIDERS[normalized_name] = provider_cls
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    @classmethod
    def create(cls, settings: Any, **kwargs: Any) -> BaseLLM:
        """根据 `settings.llm.provider` 创建提供者实例。

        参数说明：
        - settings: 全局配置对象，要求包含 `settings.llm.provider`。
        - **kwargs: 额外构造参数，会透传给 provider 构造函数。

        失败场景：
        - provider 未配置：抛 ValueError（提示缺少 llm.provider）。
        - provider 未注册：抛 ValueError（附带可用 provider 列表）。
        """

        provider_name = getattr(getattr(settings, "llm", None), "provider", None)
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("Missing required setting: llm.provider")

        normalized_name = provider_name.strip().lower()
        provider_cls = cls._PROVIDERS.get(normalized_name)
        if provider_cls is None:
            available = ", ".join(cls.list_providers()) or "(none)"
            raise ValueError(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Available providers: {available}"
            )

        return provider_cls(settings, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 名称列表（字母序）。

        为什么排序：
        - 让日志、错误信息、测试断言更稳定，避免注册顺序影响输出。
        """

        return sorted(cls._PROVIDERS)
