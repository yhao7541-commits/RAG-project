"""LLM Reranker 默认实现。

设计目标：
1. 复用项目统一的 `BaseReranker` 契约。
2. 使用 LLM 对候选段落做相关性打分并重排。
3. 对解析失败、调用失败等场景给出可读错误信息。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """LLM Reranker 统一异常。"""


class LLMReranker(BaseReranker):
    """基于 LLM 的重排器实现。"""

    DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"

    def __init__(
        self,
        settings: Any,
        *,
        llm: BaseLLM | None = None,
        prompt_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 LLMReranker。"""

        self.settings = settings
        self.llm = llm or LLMFactory.create(settings)

        resolved_prompt_path = str(prompt_path or self.DEFAULT_PROMPT_PATH)
        self.prompt_template = self._load_prompt_template(resolved_prompt_path)

    def _load_prompt_template(self, prompt_path: str) -> str:
        """从文件加载重排 Prompt 模板。"""

        try:
            return Path(prompt_path).read_text(encoding="utf-8")
        except Exception as error:  # noqa: BLE001
            raise LLMRerankError(f"Failed to load rerank prompt: {error}") from error

    def _build_rerank_prompt(self, query: str, candidates: list[dict[str, Any]]) -> str:
        """把 query 与候选段落拼装成可发送给 LLM 的 prompt。"""

        candidate_lines: list[str] = []
        for candidate in candidates:
            passage_id = str(candidate.get("id", ""))
            text_value = candidate.get("text", candidate.get("content", ""))
            passage_text = text_value if isinstance(text_value, str) else ""
            candidate_lines.append(f"- passage_id: {passage_id}\n  passage: {passage_text}")

        candidates_block = "\n".join(candidate_lines)
        return (
            f"{self.prompt_template.strip()}\n\n"
            f"Query:\n{query}\n\n"
            f"Passages:\n{candidates_block}\n\n"
            "Output JSON array only."
        )

    def _parse_llm_response(self, response_content: str) -> list[dict[str, Any]]:
        """解析并校验 LLM 返回的 JSON 重排结果。"""

        cleaned = response_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed: Any = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise LLMRerankError(f"LLM response is not valid JSON: {error}") from error

        if not isinstance(parsed, list):
            raise LLMRerankError("Expected JSON array in LLM response")

        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMRerankError(f"Item at index {index} is not an object")

            if "passage_id" not in item:
                raise LLMRerankError(
                    f"Item at index {index} missing required field 'passage_id'"
                )
            if "score" not in item:
                raise LLMRerankError(
                    f"Item at index {index} missing required field 'score'"
                )
            if isinstance(item["score"], bool) or not isinstance(item["score"], (int, float)):
                raise LLMRerankError(f"Item at index {index} score must be numeric")

        return parsed

    @staticmethod
    def _attach_and_sort_by_rerank_score(
        candidates: list[dict[str, Any]],
        parsed_scores: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """把 LLM 分数附加到候选并按分数降序返回。"""

        candidate_map = {str(item.get("id", "")): item for item in candidates}
        used_ids: set[str] = set()
        reranked: list[dict[str, Any]] = []

        sorted_scores = sorted(
            parsed_scores,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )

        for scored_item in sorted_scores:
            passage_id = str(scored_item.get("passage_id", ""))
            original = candidate_map.get(passage_id)
            if original is None:
                continue

            merged = dict(original)
            merged["rerank_score"] = float(scored_item["score"])
            if "reasoning" in scored_item:
                merged["rerank_reasoning"] = scored_item["reasoning"]
            reranked.append(merged)
            used_ids.add(passage_id)

        # 若 LLM 漏掉部分候选，追加到末尾并给默认分数，避免结果丢失。
        for candidate in candidates:
            candidate_id = str(candidate.get("id", ""))
            if candidate_id in used_ids:
                continue
            fallback_item = dict(candidate)
            fallback_item["rerank_score"] = float(fallback_item.get("score", 0.0))
            reranked.append(fallback_item)

        return reranked

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """调用 LLM 对候选段落重排。"""

        self.validate_query(query)
        self.validate_candidates(candidates)

        if len(candidates) == 1:
            return list(candidates)

        prompt = self._build_rerank_prompt(query, candidates)
        messages = [
            Message(role="system", content="You are a reranking assistant."),
            Message(role="user", content=prompt),
        ]

        try:
            llm_response = self.llm.chat(messages, trace=trace)
        except Exception as error:  # noqa: BLE001
            raise LLMRerankError(f"LLM call failed: {error}") from error

        parsed_scores = self._parse_llm_response(llm_response.content)
        reranked = self._attach_and_sort_by_rerank_score(candidates, parsed_scores)

        top_k_value = kwargs.get("top_k")
        if top_k_value is None:
            return reranked

        if not isinstance(top_k_value, int) or top_k_value <= 0:
            raise ValueError("top_k must be a positive integer")

        return reranked[:top_k_value]
