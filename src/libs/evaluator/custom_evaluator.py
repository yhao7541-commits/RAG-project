"""自定义轻量评估器。

当前仅实现两个基础指标：
1. hit_rate: 检索结果中是否命中过任一 ground_truth。
2. mrr: 首个命中项的倒数排名（Mean Reciprocal Rank for single query）。

说明：
- 这是 B6 的最小可用实现，目标是稳定、可测试、易理解。
"""

from __future__ import annotations

from typing import Any

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """支持 hit_rate / mrr 的轻量评估器。"""

    SUPPORTED_METRICS = {"hit_rate", "mrr"}

    def __init__(self, metrics: list[str] | None = None, **_: Any) -> None:
        """初始化自定义评估器。

        参数说明：
        - metrics: 需要计算的指标列表。
          - 传 `None` 时默认启用 `hit_rate` 和 `mrr`。
          - 这样设计是为了“开箱即用”，减少新手配置负担。
        """

        selected_metrics = metrics or ["hit_rate", "mrr"]

        unsupported_metrics = sorted(
            metric for metric in selected_metrics if metric not in self.SUPPORTED_METRICS
        )
        if unsupported_metrics:
            unsupported_text = ", ".join(unsupported_metrics)
            raise ValueError(f"Unsupported custom metrics: {unsupported_text}")

        self.metrics = selected_metrics

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        *,
        ground_truth: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """根据请求指标计算评估结果。

        参数说明：
        - query: 用户查询。
        - retrieved_chunks: 检索结果列表。
        - ground_truth: 正确答案对应的 chunk id 列表（可选）。
          - 如果为空，hit_rate/mrr 都会返回 0.0。
        """

        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        ground_truth_ids = set(ground_truth or [])

        results: dict[str, float] = {}
        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_chunks, ground_truth_ids)

        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_chunks, ground_truth_ids)

        return results

    def _compute_hit_rate(
        self,
        retrieved_chunks: list[dict[str, Any]],
        ground_truth_ids: set[str],
    ) -> float:
        """计算命中率。

        当前定义（单查询场景）：
        - 只要候选里出现任一 ground_truth id，结果记 1.0。
        - 否则记 0.0。
        """

        if not ground_truth_ids:
            return 0.0

        for chunk in retrieved_chunks:
            chunk_id = chunk.get("id")
            if isinstance(chunk_id, str) and chunk_id in ground_truth_ids:
                return 1.0

        return 0.0

    def _compute_mrr(
        self,
        retrieved_chunks: list[dict[str, Any]],
        ground_truth_ids: set[str],
    ) -> float:
        """计算首个命中的倒数排名（MRR）。

        例子：
        - 第 1 位命中 -> 1.0
        - 第 2 位命中 -> 0.5
        - 没命中 -> 0.0
        """

        if not ground_truth_ids:
            return 0.0

        for index, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = chunk.get("id")
            if isinstance(chunk_id, str) and chunk_id in ground_truth_ids:
                return 1.0 / float(index)

        return 0.0
