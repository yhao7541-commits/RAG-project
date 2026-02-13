"""重排器（Reranker）抽象层。

这个模块定义了统一的重排接口，让上层检索流程不关心具体用的是
LLM 重排、Cross-Encoder 重排，还是“不开启重排”的回退实现。

为什么需要它：
1. 统一调用方式，降低业务层耦合。
2. 统一输入校验，避免不同实现重复写校验逻辑。
3. 提供 NoneReranker，保证在关闭重排时链路仍可稳定运行。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """所有重排器实现都需要继承的抽象基类。"""

    def validate_query(self, query: str) -> None:
        """校验查询文本。

        参数设计说明：
        - query 设计为 `str`，保持与大多数检索入口一致，
          避免上层在进入重排前再做格式转换。
        """

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

    def validate_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """校验候选列表。

        参数设计说明：
        - 使用 `list[dict]` 是为了兼容不同检索器输出（dense/sparse/fusion）。
        """

        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list")

        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError(f"Candidate at index {index} is not a dict")

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """按相关性对候选结果进行重排。

        参数说明：
        - query: 用户查询。
        - candidates: 待重排候选集合。
        - trace: 可选追踪上下文。
        - **kwargs: 子类扩展参数（如 top_k、score_threshold）。
        """


class NoneReranker(BaseReranker):
    """空重排器：保持原有顺序不变。

    用于以下场景：
    - 显式关闭重排（enabled = false）
    - provider 配置为 none
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """不做重排，原样返回候选列表的浅拷贝。"""

        self.validate_query(query)
        self.validate_candidates(candidates)

        # 返回浅拷贝，既保持顺序，也避免调用方误以为是同一个列表对象。
        return list(candidates)
