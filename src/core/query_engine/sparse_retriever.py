"""Sparse retriever implementation using in-memory BM25-like scores."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from src.core.types import ChunkRecord


class SparseRetriever:
    """Performs lightweight sparse retrieval over chunk records."""

    TOKEN_RE = re.compile(r"[\w]+(?:[-_][\w]+)*", flags=re.UNICODE)

    def __init__(self, records: list[ChunkRecord] | None = None) -> None:
        self.records = records or []

    def build_index(self, records: list[ChunkRecord]) -> None:
        self.records = records

    def retrieve(
        self, query: str, top_k: int = 20, filters: dict[str, Any] | None = None, trace=None
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        df = self._document_frequency()
        n_docs = max(len(self.records), 1)
        scores: list[tuple[float, ChunkRecord]] = []

        for record in self.records:
            if not self._passes_filters(record, filters):
                continue
            tf = Counter(self._tokenize(record.text))
            if not tf:
                continue
            score = 0.0
            for term in query_terms:
                term_df = df.get(term, 0)
                if term_df == 0:
                    continue
                idf = math.log((n_docs + 1) / (term_df + 1)) + 1.0
                score += tf.get(term, 0) * idf
            if score > 0:
                scores.append((score, record))

        scores.sort(key=lambda item: item[0], reverse=True)
        output: list[dict[str, Any]] = []
        for score, record in scores[:top_k]:
            output.append(
                {
                    "id": record.id,
                    "score": score,
                    "metadata": dict(record.metadata),
                    "document": record.text,
                    "route": "sparse",
                }
            )
        return output

    def _document_frequency(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for record in self.records:
            for term in set(self._tokenize(record.text)):
                df[term] = df.get(term, 0) + 1
        return df

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in self.TOKEN_RE.findall(text)]

    @staticmethod
    def _passes_filters(record: ChunkRecord, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if record.metadata.get(key) != value:
                return False
        return True
