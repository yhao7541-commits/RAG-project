"""Azure OpenAI LLM 提供者实现（默认 HTTP，支持可选 LangChain 路径）。

实现原则：
1. 默认走 Azure OpenAI 的 HTTP 接口（满足原始需求：mock HTTP 可测试）。
2. 保留 `use_langchain=True` 的可选路径，满足后续 agent 统一接入需求。
3. 对外统一返回 ChatResponse，保持工厂与上层调用稳定。
"""

from __future__ import annotations

import os
import importlib
from typing import Any, Sequence

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class AzureLLMError(RuntimeError):
    """Azure OpenAI 调用失败时抛出的统一异常。"""


class AzureLLM(BaseLLM):
    """Azure OpenAI provider。"""

    DEFAULT_TIMEOUT = 60.0
    DEFAULT_API_VERSION = "2024-02-15-preview"

    def __init__(
        self,
        settings: Any,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        use_langchain: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure provider 配置。"""

        super().__init__(settings, **kwargs)

        llm_settings = getattr(settings, "llm", None)
        self.model = str(getattr(llm_settings, "model", "gpt-4o-mini"))
        self.deployment_name = str(getattr(llm_settings, "deployment_name", self.model))
        self.default_temperature = float(getattr(llm_settings, "temperature", 0.0))
        self.default_max_tokens = int(getattr(llm_settings, "max_tokens", 1024))

        api_key_value = (
            api_key
            or getattr(llm_settings, "api_key", None)
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if not api_key_value:
            raise ValueError("Azure OpenAI API key not provided")
        self.api_key = str(api_key_value)

        endpoint_value = (
            endpoint
            or getattr(llm_settings, "azure_endpoint", None)
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not endpoint_value:
            raise ValueError("Azure OpenAI endpoint not provided")
        self.endpoint = str(endpoint_value).rstrip("/")

        version_value = api_version or getattr(llm_settings, "api_version", None)
        self.api_version = str(version_value or self.DEFAULT_API_VERSION)
        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

        if use_langchain is None:
            self.use_langchain = bool(getattr(llm_settings, "use_langchain", False))
        else:
            self.use_langchain = bool(use_langchain)

    def chat(
        self,
        messages: Sequence[Message],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """发送 Azure OpenAI 对话请求。"""

        self.validate_messages(messages)

        if self.use_langchain:
            return self._chat_via_langchain(messages, **kwargs)
        return self._chat_via_http(messages, **kwargs)

    def _chat_via_http(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """默认路径：HTTP 直连 Azure OpenAI Chat Completions。"""

        model_name = str(kwargs.get("model", self.deployment_name))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        payload = {
            "model": model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        endpoint = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
        except httpx.RequestError as error:
            raise AzureLLMError(f"[Azure OpenAI] API error: {error}") from error

        try:
            data = response.json()
        except Exception as error:  # noqa: BLE001
            raise AzureLLMError("[Azure OpenAI] API error: Unexpected response format") from error

        if response.status_code >= 400:
            error_message = (
                data.get("error", {}).get("message")
                if isinstance(data, dict)
                else None
            )
            readable_message = error_message or response.text or "Unknown error"
            raise AzureLLMError(
                f"[Azure OpenAI] API error (HTTP {response.status_code}): {readable_message}"
            )

        try:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            output_model = str(data.get("model", self.model))
        except Exception as error:  # noqa: BLE001
            raise AzureLLMError("[Azure OpenAI] API error: Unexpected response format") from error

        return ChatResponse(
            content=str(content),
            model=output_model,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            raw_response=data if isinstance(data, dict) else None,
        )

    def _chat_via_langchain(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """可选路径：LangChain AzureChatOpenAI 调用。"""

        try:
            langchain_openai = importlib.import_module("langchain_openai")
            llm_utils = importlib.import_module("src.libs.llm.langchain_utils")
            AzureChatOpenAI = getattr(langchain_openai, "AzureChatOpenAI")
            messages_to_langchain = getattr(llm_utils, "messages_to_langchain")
            extract_usage_from_ai_message = getattr(
                llm_utils,
                "extract_usage_from_ai_message",
            )
        except Exception as error:  # noqa: BLE001
            raise AzureLLMError(f"[Azure OpenAI] LangChain adapter unavailable: {error}") from error

        model_name = str(kwargs.get("model", self.deployment_name))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        try:
            model = AzureChatOpenAI(
                model=model_name,
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                azure_deployment=self.deployment_name,
                temperature=temperature,
                timeout=self.timeout,
                max_completion_tokens=max_tokens,
            )
            ai_message = model.invoke(messages_to_langchain(messages))
        except Exception as error:  # noqa: BLE001
            raise AzureLLMError(f"[Azure OpenAI] API error: {error}") from error

        return ChatResponse(
            content=self._extract_content(ai_message),
            model=model_name,
            usage=extract_usage_from_ai_message(ai_message),
        )

    def _extract_content(self, ai_message: Any) -> str:
        """从模型响应中提取文本内容。"""

        raw_content = getattr(ai_message, "content", "")
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        return str(raw_content)
