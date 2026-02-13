"""OpenAI LLM 提供者实现（默认 HTTP，支持可选 LangChain 适配路径）。

实现策略：
1. 默认走 OpenAI 兼容 HTTP 接口（更稳定、排障更直接）。
2. 保留 `use_langchain=True` 的可选路径，满足后续 agent 统一接入 LangChain 的需求。
3. 对外统一返回 `ChatResponse`，避免上层感知底层差异。
"""

from __future__ import annotations

import os
import importlib
from typing import Any, Sequence

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """OpenAI 调用失败时抛出的统一异常。"""


class OpenAILLM(BaseLLM):
    """OpenAI provider。

    参数优先级（从高到低）：
    1) 构造函数显式参数
    2) settings.llm.*
    3) 环境变量（OPENAI_API_KEY）
    4) 默认值
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        settings: Any,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        use_langchain: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI provider 配置。

        参数说明：
        - settings: 全局配置对象。
        - api_key/base_url/timeout: 显式覆盖项。
        - use_langchain: 是否启用可选 LangChain 调用路径。
          - False（默认）：HTTP 直连
          - True：LangChain 适配层
        """

        super().__init__(settings, **kwargs)

        llm_settings = getattr(settings, "llm", None)
        self.model = str(getattr(llm_settings, "model", "gpt-4o-mini"))
        self.default_temperature = float(getattr(llm_settings, "temperature", 0.0))
        self.default_max_tokens = int(getattr(llm_settings, "max_tokens", 1024))

        api_key_value = api_key or getattr(llm_settings, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key_value:
            raise ValueError("OpenAI API key not provided")
        self.api_key = str(api_key_value)

        configured_base_url = base_url or getattr(llm_settings, "base_url", None)
        self.base_url = str(configured_base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

        # 可选 LangChain 开关：默认关闭，保持“原要求”的 HTTP 实现。
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
        """发送对话请求。

        处理流程：
        1) 先做输入校验。
        2) 根据开关选择 HTTP 路径或 LangChain 路径。
        """

        self.validate_messages(messages)

        if self.use_langchain:
            return self._chat_via_langchain(messages, **kwargs)
        return self._chat_via_http(messages, **kwargs)

    def _chat_via_http(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """默认路径：通过 HTTP 调 OpenAI `/chat/completions`。"""

        model_name = str(kwargs.get("model", self.model))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        payload = {
            "model": model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
        except httpx.RequestError as error:
            raise OpenAILLMError(f"[OpenAI] API error: {error}") from error

        try:
            data = response.json()
        except Exception as error:  # noqa: BLE001 - 统一包装响应格式异常
            raise OpenAILLMError("[OpenAI] API error: Unexpected response format") from error

        if response.status_code >= 400:
            error_message = (
                data.get("error", {}).get("message")
                if isinstance(data, dict)
                else None
            )
            readable_message = error_message or response.text or "Unknown error"
            raise OpenAILLMError(
                f"[OpenAI] API error (HTTP {response.status_code}): {readable_message}"
            )

        try:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            output_model = str(data.get("model", model_name))
        except Exception as error:  # noqa: BLE001 - 统一包装响应结构异常
            raise OpenAILLMError("[OpenAI] API error: Unexpected response format") from error

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
        """可选路径：通过 LangChain ChatOpenAI 调用。

        说明：
        - 该路径主要用于后续 agent 统一接入场景。
        - 仍然复用项目统一输出结构 ChatResponse。
        """

        try:
            langchain_openai = importlib.import_module("langchain_openai")
            llm_utils = importlib.import_module("src.libs.llm.langchain_utils")
            ChatOpenAI = getattr(langchain_openai, "ChatOpenAI")
            messages_to_langchain = getattr(llm_utils, "messages_to_langchain")
            extract_usage_from_ai_message = getattr(
                llm_utils,
                "extract_usage_from_ai_message",
            )
        except Exception as error:  # noqa: BLE001 - 统一包装可选依赖异常
            raise OpenAILLMError(f"[OpenAI] LangChain adapter unavailable: {error}") from error

        model_name = str(kwargs.get("model", self.model))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        try:
            model = ChatOpenAI(
                model=model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
            )
            ai_message = model.invoke(messages_to_langchain(messages))
        except Exception as error:  # noqa: BLE001
            raise OpenAILLMError(f"[OpenAI] API error: {error}") from error

        return ChatResponse(
            content=self._extract_content(ai_message),
            model=model_name,
            usage=extract_usage_from_ai_message(ai_message),
        )

    def _extract_content(self, ai_message: Any) -> str:
        """从模型响应中提取文本内容，兼容多种结构。"""

        raw_content = getattr(ai_message, "content", "")
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(raw_content)
