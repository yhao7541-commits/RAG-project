"""LLM 抽象层对外导出与默认 provider 注册。

这个文件做两件事：
1. 统一导出 LLM 相关类型与实现，供上层稳定导入。
2. 在模块加载时注册默认 provider，避免调用方手动注册。

对新手来说，可以直接记住：
`from src.libs.llm import LLMFactory, Message` 然后通过工厂创建实例。
"""

from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError

# 模块导入时注册默认 provider，方便测试和运行时直接使用工厂。
# 这样做的意义：
# - 少写样板代码（不用每个入口都手动 register）
# - 降低“忘记注册导致 provider not found”的概率
if "openai" not in LLMFactory._PROVIDERS:
    LLMFactory.register_provider("openai", OpenAILLM)
if "azure" not in LLMFactory._PROVIDERS:
    LLMFactory.register_provider("azure", AzureLLM)
if "deepseek" not in LLMFactory._PROVIDERS:
    LLMFactory.register_provider("deepseek", DeepSeekLLM)
if "ollama" not in LLMFactory._PROVIDERS:
    LLMFactory.register_provider("ollama", OllamaLLM)
if "azure" not in LLMFactory._VISION_PROVIDERS:
    LLMFactory.register_vision_provider("azure", AzureVisionLLM)

__all__ = [
    "BaseLLM",
    "ChatResponse",
    "Message",
    "BaseVisionLLM",
    "ImageInput",
    "LLMFactory",
    "OpenAILLM",
    "OpenAILLMError",
    "AzureLLM",
    "AzureLLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "OllamaLLM",
    "OllamaLLMError",
    "AzureVisionLLM",
    "AzureVisionLLMError",
]
