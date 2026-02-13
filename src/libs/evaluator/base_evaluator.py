"""评估器（Evaluator）抽象层。

本模块定义统一的评估接口，用于计算检索/生成质量指标。
这样上层只需要调用 `evaluate()`，不关心具体是 custom、ragas 还是其他后端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """所有评估器实现都应继承的抽象基类。"""

    def validate_query(self, query: str) -> None:
        """校验查询文本。

        参数设计说明：
        - query 为字符串，和检索/生成入口保持一致，便于流水线串联。
        """

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

    def validate_retrieved_chunks(self, retrieved_chunks: list[dict[str, Any]]) -> None:
        """校验检索结果列表。

        参数设计说明：
        - 采用 `list[dict]` 结构，能兼容不同检索器输出格式，
          由具体评估器自行读取需要的键。
        """

        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")

        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")

        for index, item in enumerate(retrieved_chunks):
            if not isinstance(item, dict):
                raise ValueError(f"retrieved_chunks[{index}] must be a dict")

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, float]:
        """计算并返回评估指标。

        参数说明：
        - query: 用户问题。
        - retrieved_chunks: 检索返回的候选块。
        - **kwargs: 评估所需扩展输入（如 ground_truth）。
        """


class NoneEvaluator(BaseEvaluator):
    """空评估器：用于关闭评估时的稳定回退。"""

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, float]:
        """关闭评估时返回空指标字典。"""

        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
