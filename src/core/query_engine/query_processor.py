"""Query preprocessing for retrieval pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedQuery:
    original_query: str
    normalized_query: str
    keywords: list[str]
    filters: dict[str, str]


class QueryProcessor:
    """Extracts keywords and structured filters from user query."""

    FILTER_KEYS = {"collection", "doc_type", "language", "source_path"}
    TOKEN_RE = re.compile(r"[\w]+(?:[-_][\w]+)*", flags=re.UNICODE)

    def process(self, query: str) -> ProcessedQuery:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        normalized = re.sub(r"\s+", " ", query).strip()
        filters = self._extract_filters(normalized)
        keywords = self._extract_keywords(normalized)
        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized,
            keywords=keywords,
            filters=filters,
        )

    def _extract_filters(self, query: str) -> dict[str, str]:
        pairs = re.findall(r"\b([a-zA-Z_]+)\s*:\s*([\w./-]+)", query)
        out: dict[str, str] = {}
        for key, value in pairs:
            k = key.lower()
            if k in self.FILTER_KEYS:
                out[k] = value
        return out

    def _extract_keywords(self, query: str) -> list[str]:
        terms = [token.lower() for token in self.TOKEN_RE.findall(query)]
        # Remove key:value filter keys from keyword list
        return [term for term in terms if term not in self.FILTER_KEYS]
