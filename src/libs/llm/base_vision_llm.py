"""Vision LLM 抽象契约。

这个模块定义了图像理解类模型的统一接口：
1. 统一图片输入结构（`ImageInput`）。
2. 统一文本/图片输入校验。
3. 统一 `chat_with_image()` 调用签名。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """图像输入结构。

    约束：`path` / `data` / `base64` 三者必须且只能提供一个。
    """

    path: str | Path | None = None
    data: bytes | None = None
    base64: str | None = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        provided_count = sum(
            value is not None for value in (self.path, self.data, self.base64)
        )
        if provided_count == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if provided_count > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """Vision LLM 抽象基类。"""

    def validate_text(self, text: str) -> None:
        """校验文本提示词。"""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if not text.strip():
            raise ValueError("Text prompt cannot be empty")

    def validate_image(self, image: ImageInput) -> None:
        """校验图像输入。"""

        if not isinstance(image, ImageInput):
            raise ValueError("Image must be an ImageInput instance")

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: tuple[int, int] | None = None,
    ) -> ImageInput:
        """默认不做预处理，直接返回原图像。"""

        return image

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """发送图文请求并返回标准化 ChatResponse。"""
