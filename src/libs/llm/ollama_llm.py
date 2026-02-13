"""Ollama LLM 提供者实现（本地/私有部署优先）。

这个实现的目标是让项目可以直接调用本地 Ollama 服务，
并继续沿用项目统一的 `BaseLLM -> ChatResponse` 契约。

为什么单独做一个 Ollama provider：
1. Ollama 常用于本地推理，默认无需云端 API Key。
2. Ollama 的接口字段与 OpenAI 不完全一致（例如 `num_predict`）。
3. 需要给出更贴近本地排障的错误提示（例如提醒 `ollama serve`）。
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Sequence

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OllamaLLMError(RuntimeError):
    """Ollama 调用失败时抛出的统一异常。"""


class OllamaLLM(BaseLLM):
    """Ollama provider 实现。

    参数优先级（从高到低）：
    1) 构造函数显式参数
    2) settings.llm.*
    3) 环境变量 `OLLAMA_BASE_URL`
    4) 默认值
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        settings: Any,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        use_langchain: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Ollama provider。

        参数说明：
        - settings: 全局配置对象。
        - base_url: Ollama 服务地址（例如 `http://localhost:11434`）。
        - timeout: 单次请求超时秒数。
        - **kwargs: 预留扩展参数。
        """

        super().__init__(settings, **kwargs)

        llm_settings = getattr(settings, "llm", None)

        # 步骤 1：读取模型默认参数。
        self.model = str(getattr(llm_settings, "model", "llama3"))
        self.default_temperature = float(getattr(llm_settings, "temperature", 0.7))
        self.default_max_tokens = int(getattr(llm_settings, "max_tokens", 2048))

        # 步骤 2：读取 base_url（显式参数优先，其次环境变量）。
        configured_base_url = base_url or os.environ.get("OLLAMA_BASE_URL")
        self.base_url = str(configured_base_url or self.DEFAULT_BASE_URL).rstrip("/")

        # 步骤 3：读取超时配置。Ollama 本地推理常偏慢，默认给更长超时。
        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

        # 可选 LangChain 开关：默认关闭，保持原始 HTTP 行为。
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
        """调用 Ollama `/api/chat` 并返回统一响应。

        参数说明：
        - messages: 对话消息列表。
        - trace: 追踪上下文（可选，当前实现未直接使用）。
        - **kwargs: 单次覆盖参数，常用：
          - model: 覆盖模型名
          - temperature: 覆盖采样温度
          - max_tokens: 覆盖最大生成 token（会映射为 Ollama 的 `num_predict`）
        """

        # 步骤 1：先做统一输入校验，避免无效请求打到服务端。
        self.validate_messages(messages)

        # 步骤 2：根据开关选择调用路径。
        if self.use_langchain:
            return self._chat_via_langchain(messages, **kwargs)
        return self._chat_via_http(messages, **kwargs)

    def _chat_via_http(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """默认路径：HTTP 直连 Ollama `/api/chat`。"""

        # 步骤 1：合并单次调用参数。
        model_name = str(kwargs.get("model", self.model))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        # 步骤 2：组装 Ollama 请求体。
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        endpoint = f"{self.base_url}/api/chat"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
        except httpx.TimeoutException as error:
            raise OllamaLLMError(
                f"[Ollama] Request timed out after {self.timeout:.0f} seconds"
            ) from error
        except httpx.ConnectError as error:
            raise OllamaLLMError(
                "[Ollama] Connection failed. Please make sure Ollama is running (`ollama serve`)"
            ) from error
        except httpx.RequestError as error:
            raise OllamaLLMError(f"[Ollama] API request failed: {error}") from error

        # 步骤 3：解析响应 JSON。若解析失败，统一按格式错误处理。
        try:
            data = response.json()
        except Exception as error:  # noqa: BLE001 - 统一包装响应格式异常
            raise OllamaLLMError("[Ollama] Unexpected response format") from error

        if response.status_code >= 400:
            error_message = "Unknown error"
            if isinstance(data, dict) and isinstance(data.get("error"), str):
                error_message = data["error"]
            elif response.text:
                error_message = response.text

            # 注意：不拼接 endpoint，避免把内部地址暴露到错误信息中。
            raise OllamaLLMError(
                f"[Ollama] API error (HTTP {response.status_code}): {error_message}"
            )

        if not isinstance(data, dict):
            raise OllamaLLMError("[Ollama] Unexpected response format")

        message_obj = data.get("message")
        if not isinstance(message_obj, dict):
            raise OllamaLLMError("[Ollama] Unexpected response format")

        content = message_obj.get("content")
        if not isinstance(content, str):
            raise OllamaLLMError("[Ollama] Unexpected response format")

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            content=content,
            model=str(data.get("model", model_name)),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            raw_response=data,
        )

    def _chat_via_langchain(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """可选路径：LangChain 适配调用。

        说明：
        - 该路径用于“统一 agent 走 LangChain”场景。
        - 仍保持与项目统一的 ChatResponse 输出契约。
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
        except Exception as error:  # noqa: BLE001
            raise OllamaLLMError(f"[Ollama] LangChain adapter unavailable: {error}") from error

        model_name = str(kwargs.get("model", self.model))
        temperature = float(kwargs.get("temperature", self.default_temperature))
        max_tokens = int(kwargs.get("max_tokens", self.default_max_tokens))

        try:
            # Ollama 的 OpenAI-compatible 路径通常是 /v1。
            openai_compatible_url = f"{self.base_url}/v1"
            model = ChatOpenAI(
                model=model_name,
                api_key="ollama",  # 本地服务一般不校验 key，这里填占位值
                base_url=openai_compatible_url,
                timeout=self.timeout,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
            )
            ai_message = model.invoke(messages_to_langchain(messages))
        except Exception as error:  # noqa: BLE001
            raise OllamaLLMError(f"[Ollama] API error: {error}") from error

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
