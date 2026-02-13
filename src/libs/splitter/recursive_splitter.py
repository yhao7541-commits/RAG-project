"""Recursive Splitter 默认实现（基于 langchain-text-splitters）。

实现思路：
1. 读取 `settings.ingestion` 里的 chunk 参数。
2. 用 LangChain 的 `RecursiveCharacterTextSplitter` 做递归切分。
3. 保持项目统一接口：输入校验 -> 切分 -> 输出校验。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """默认递归文本切分器。"""

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        settings: Any,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化切分器并构建底层 LangChain splitter。"""

        self.settings = settings

        ingestion_settings = getattr(settings, "ingestion", None)
        if ingestion_settings is None:
            raise ValueError("Missing ingestion configuration: settings.ingestion")

        configured_chunk_size = (
            chunk_size
            if chunk_size is not None
            else getattr(ingestion_settings, "chunk_size", self.DEFAULT_CHUNK_SIZE)
        )
        configured_chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else getattr(ingestion_settings, "chunk_overlap", self.DEFAULT_CHUNK_OVERLAP)
        )

        if isinstance(configured_chunk_size, bool) or not isinstance(configured_chunk_size, int):
            raise ValueError("chunk_size must be a positive integer")
        if configured_chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        if isinstance(configured_chunk_overlap, bool) or not isinstance(configured_chunk_overlap, int):
            raise ValueError("chunk_overlap must be a non-negative integer")
        if configured_chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")

        if configured_chunk_overlap >= configured_chunk_size:
            raise ValueError(
                f"chunk_overlap ({configured_chunk_overlap}) must be less than chunk_size ({configured_chunk_size})"
            )

        self.chunk_size = configured_chunk_size
        self.chunk_overlap = configured_chunk_overlap
        self.separators = list(separators) if separators is not None else list(self.DEFAULT_SEPARATORS)

        self._splitter = self._build_splitter()

    def _build_splitter(self) -> Any:
        """按当前配置构建 LangChain 递归切分器。"""

        try:
            module = import_module("langchain_text_splitters")
            splitter_class = getattr(module, "RecursiveCharacterTextSplitter")
        except Exception as error:  # noqa: BLE001 - 统一转换依赖缺失错误
            raise ImportError(
                "langchain-text-splitters is not installed. Please install it with: "
                "pip install langchain-text-splitters"
            ) from error

        return splitter_class(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_text(
        self,
        text: str,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """把输入文本切分为多个 chunk。"""

        self.validate_text(text)

        raw_chunks = self._splitter.split_text(text)
        chunks = [chunk for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]

        self.validate_chunks(chunks)
        return chunks
