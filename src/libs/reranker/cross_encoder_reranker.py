"""Cross-Encoder Reranker 默认实现。

实现思路：
1. 使用 sentence-transformers 的 CrossEncoder 对 (query, passage) 对打分。
2. 将分数附加到候选集合后按降序排序。
3. 统一异常信息，便于在上层快速定位问题。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.libs.reranker.base_reranker import BaseReranker


class CrossEncoderRerankError(RuntimeError):
    """Cross-Encoder 重排统一异常。"""


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排器。"""

    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        settings: Any,
        *,
        model: Any | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 CrossEncoderReranker。"""

        self.settings = settings
        self.timeout = float(timeout if timeout is not None else self.DEFAULT_TIMEOUT)

        try:
            if model is not None:
                self.model = model
            else:
                model_name = self._get_model_name_from_settings(settings)
                self.model = self._load_cross_encoder_model(model_name)
        except Exception as error:  # noqa: BLE001
            raise CrossEncoderRerankError(f"Failed to initialize CrossEncoderReranker: {error}") from error

    @staticmethod
    def _get_model_name_from_settings(settings: Any) -> str:
        """从配置读取 cross-encoder 模型名。"""

        rerank_settings = getattr(settings, "rerank", None)
        model_raw = getattr(rerank_settings, "model", None)
        if not isinstance(model_raw, str) or not model_raw.strip():
            raise ValueError("Missing required configuration: settings.rerank.model")
        return model_raw.strip()

    @staticmethod
    def _load_cross_encoder_model(model_name: str) -> Any:
        """延迟加载并创建 sentence-transformers CrossEncoder。"""

        try:
            module = import_module("sentence_transformers")
            cross_encoder_class = getattr(module, "CrossEncoder", None)
            if cross_encoder_class is None:
                raise RuntimeError("CrossEncoder class not found in sentence_transformers")
            return cross_encoder_class(model_name)
        except Exception as error:  # noqa: BLE001
            raise CrossEncoderRerankError(
                f"Failed to load CrossEncoder model '{model_name}': {error}"
            ) from error

    @staticmethod
    def _prepare_pairs(
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """把候选转成 (query, passage) 对。"""

        pairs: list[tuple[str, str]] = []
        for candidate in candidates:
            text_value = candidate.get("text", candidate.get("content", ""))
            passage_text = text_value if isinstance(text_value, str) else ""
            pairs.append((query, passage_text))
        return pairs

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """调用模型对 query/passage 对打分。"""

        raw_scores = self.model.predict(pairs)
        return [float(score) for score in raw_scores]

    @staticmethod
    def _attach_scores_and_sort(
        candidates: list[dict[str, Any]],
        scores: list[float],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """附加分数并按分数降序返回前 top_k 条。"""

        merged: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            merged.append(item)

        merged.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return merged[:top_k]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """对候选集合执行 Cross-Encoder 重排。"""

        self.validate_query(query)
        self.validate_candidates(candidates)

        top_k_raw = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k_raw, int) or top_k_raw <= 0:
            raise ValueError("top_k must be a positive integer")

        pairs = self._prepare_pairs(query, candidates)
        scores = self._score_pairs(pairs)
        return self._attach_scores_and_sort(candidates, scores, top_k=top_k_raw)
