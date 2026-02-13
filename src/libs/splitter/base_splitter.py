"""文本切分器的基础抽象层。

这个模块的目标是统一“如何把一段长文本切成多个短文本块（chunks）”的接口。
上层调用方只需要依赖 BaseSplitter，而不关心底层具体是递归切分、语义切分，
还是其他策略。

设计收益：
1. 方便替换切分实现（可插拔）。
2. 方便测试（可快速构造 FakeSplitter）。
3. 统一输入与输出校验，减少重复 bug。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """所有切分器实现都应继承的抽象基类。"""

    def validate_text(self, text: str) -> None:
        """校验待切分文本。

        校验规则：
        - 必须是字符串类型。
        - 不能是空字符串或纯空白字符串。

        参数设计说明：
        - 输入只接收单条 `text`，是因为切分器核心职责是
          “把一段文本切块”；批处理由上层流程控制更清晰。
        """

        # 步骤 1：先校验类型，避免对非法类型调用字符串方法。
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # 步骤 2：校验内容有效性，空白文本没有切分意义。
        if not text.strip():
            raise ValueError("Input text cannot be empty")

    def validate_chunks(self, chunks: list[str]) -> None:
        """校验切分结果。

        校验规则：
        - 结果列表不能为空。
        - 每个 chunk 必须是字符串。
        - 每个 chunk 不能是空串或纯空白。

        参数设计说明：
        - 输出使用 `list[str]`，便于后续模块直接遍历并附加元数据。
        """

        # 步骤 1：切分结果必须至少有一个块。
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # 步骤 2：逐个检查每个块，保证返回数据质量一致。
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(f"Chunk at index {index} is not a string")

            if not chunk.strip():
                raise ValueError(
                    f"Chunk at index {index} is empty or whitespace-only"
                )

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """把一段输入文本切分成多个文本块。

        参数说明：
        - text: 原始长文本。
        - trace: 追踪上下文（可选），用于记录每次切分行为。
        - **kwargs: 预留给不同切分器实现的扩展参数。
        """
