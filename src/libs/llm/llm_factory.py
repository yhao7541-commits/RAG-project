"""根据配置创建 LLM 提供者实例的工厂模块。

把它理解为“模型调度中心”：
- 注册阶段：告诉系统有哪些 provider 可用。
- 创建阶段：根据 `settings.llm.provider` 选择对应实现并实例化。

这样上层业务不需要 `if provider == "openai" ...` 这类分支逻辑。
"""

from __future__ import annotations

from typing import Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM


class LLMFactory:
    """基于注册表的 LLM 提供者工厂。
    两个注册表（普通 LLM / Vision LLM）
    """

    _PROVIDERS: dict[str, type[BaseLLM]] = {}
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}
#egister_provider()，把字符串 provider 映射到类。
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
   #create()，读 settings.llm.provider，找类，实例化；找不到就抛“可用 provider 列表”。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
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

        return provider_cls(settings, **kwargs)  # 调用 provider 构造函数

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册 provider 名称列表（字母序）。

        为什么排序：
        - 让日志、错误信息、测试断言更稳定，避免注册顺序影响输出。
        """

        return sorted(cls._PROVIDERS)
#vision 版同理，支持 settings.vision_llm.provider 回退到 settings.llm.provider。
    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_cls: type[BaseVisionLLM],
    ) -> None:
        """注册一个 Vision LLM provider。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_cls, BaseVisionLLM):
            raise ValueError("Vision provider class must inherit from BaseVisionLLM")

        cls._VISION_PROVIDERS[normalized_name] = provider_cls  
        

    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """返回已注册 Vision provider 名称列表（字母序）。"""
        return sorted(cls._VISION_PROVIDERS)

    @classmethod
    def create_vision_llm(cls, settings: Any, **kwargs: Any) -> BaseVisionLLM:
        """根据配置创建 Vision LLM provider。"""

        vision_settings = getattr(settings, "vision_llm", None)

        provider_raw = getattr(vision_settings, "provider", None)
        if not isinstance(provider_raw, str) or not provider_raw.strip():
            provider_raw = getattr(getattr(settings, "llm", None), "provider", None)

        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider "
                "(or fallback settings.llm.provider)"
            )

        normalized_name = provider_raw.strip().lower()
        provider_cls = cls._VISION_PROVIDERS.get(normalized_name)
        if provider_cls is None:
            available = ", ".join(cls.list_vision_providers()) or "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_raw}'. "
                f"Available Vision LLM providers: {available}"
            )

        try:
            provider_constructor: Any = provider_cls
            return provider_constructor(settings, **kwargs)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{normalized_name}': {error}"
            ) from error
