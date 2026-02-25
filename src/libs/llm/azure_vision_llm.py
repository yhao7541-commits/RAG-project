"""Azure Vision LLM 默认实现。"""

from __future__ import annotations

import base64
import io
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory


class AzureVisionLLMError(RuntimeError):
    """Azure Vision LLM 统一异常。"""


class AzureVisionLLM(BaseVisionLLM):
    """基于 Azure OpenAI Vision 接口的实现。"""

    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_MAX_IMAGE_SIZE = 2048

    def __init__(
        self,
        settings: Any,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
        max_image_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings

        vision_settings = getattr(settings, "vision_llm", None)
        llm_settings = getattr(settings, "llm", None)

        settings_api_key = getattr(vision_settings, "api_key", None)
        key_value = api_key or settings_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not key_value:
            raise ValueError("Azure OpenAI API key not provided")
        self.api_key = str(key_value)

        settings_endpoint = getattr(vision_settings, "azure_endpoint", None)
        endpoint_value = endpoint or settings_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint_value:
            raise ValueError("Azure OpenAI endpoint not provided")
        self.endpoint = str(endpoint_value)

        settings_deployment = getattr(vision_settings, "deployment_name", None)
        fallback_model = getattr(vision_settings, "model", None) or getattr(llm_settings, "model", None)
        self.deployment_name = str(deployment_name or settings_deployment or fallback_model or "gpt-4o")

        settings_version = getattr(vision_settings, "api_version", None)
        self.api_version = str(api_version or settings_version or self.DEFAULT_API_VERSION)

        settings_max_image_size = getattr(vision_settings, "max_image_size", None)
        max_size_value = max_image_size or settings_max_image_size or self.DEFAULT_MAX_IMAGE_SIZE
        self.max_image_size = int(max_size_value)

    def _get_image_base64(self, image: ImageInput) -> str:
        """把图片输入转换成 base64 字符串。"""

        if image.base64 is not None:
            return image.base64

        if image.data is not None:
            return base64.b64encode(image.data).decode("utf-8")

        if image.path is not None:
            raw_bytes = Path(image.path).read_bytes()
            return base64.b64encode(raw_bytes).decode("utf-8")

        raise AzureVisionLLMError("Invalid image input: no usable source found")

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: tuple[int, int] | None = None,
    ) -> ImageInput:
        """压缩过大的图片，保持长宽比。"""

        self.validate_image(image)

        if image.base64 is not None:
            return image

        if image.data is None:
            return image

        target_size = max_size or (self.max_image_size, self.max_image_size)

        try:
            image_module = import_module("PIL.Image")
            open_image = getattr(image_module, "open")
        except Exception:
            return image

        try:
            with open_image(io.BytesIO(image.data)) as img:
                width, height = img.size
                if width <= target_size[0] and height <= target_size[1]:
                    return image

                resized = img.copy()
                resized.thumbnail(target_size)

                buffer = io.BytesIO()
                format_name = img.format or "PNG"
                resized.save(buffer, format=format_name)
                return ImageInput(data=buffer.getvalue(), mime_type=image.mime_type)
        except Exception:
            return image

    def _call_api(
        self,
        *,
        messages: list[dict[str, Any]],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """调用 Azure OpenAI Chat Completions API。"""

        try:
            import httpx
        except Exception as error:  # noqa: BLE001
            raise AzureVisionLLMError("httpx is required for AzureVisionLLM") from error

        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, params=params, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """发送图文请求并返回标准响应。"""

        self.validate_text(text)
        self.validate_image(image)

        processed_image = self.preprocess_image(image)
        image_base64 = self._get_image_base64(processed_image)

        history: list[dict[str, Any]] = []
        if messages:
            for msg in messages:
                history.append({"role": msg.role, "content": msg.content})

        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{processed_image.mime_type};base64,{image_base64}"},
            },
        ]
        history.append({"role": "user", "content": content})

        temperature_value = float(kwargs.get("temperature", getattr(getattr(self.settings, "llm", None), "temperature", 0.0)))
        max_tokens_value = int(kwargs.get("max_tokens", getattr(getattr(self.settings, "llm", None), "max_tokens", 1024)))
        deployment_value = str(kwargs.get("deployment_name", self.deployment_name))

        try:
            raw_response = self._call_api(
                messages=history,
                deployment=deployment_value,
                temperature=temperature_value,
                max_tokens=max_tokens_value,
            )

            content_text = raw_response["choices"][0]["message"]["content"]
            model_name = str(raw_response.get("model", deployment_value))
            usage = raw_response.get("usage", {})

            return ChatResponse(
                content=content_text,
                model=model_name,
                usage=usage if isinstance(usage, dict) else {},
                raw_response=raw_response,
            )
        except Exception as error:  # noqa: BLE001
            error_type = error.__class__.__name__
            raise AzureVisionLLMError(f"API call failed: {error_type}: {error}") from error


# 模块导入时注册默认 Vision provider，保证工厂可直接创建。
if "azure" not in LLMFactory._VISION_PROVIDERS:
    LLMFactory.register_vision_provider("azure", AzureVisionLLM)
