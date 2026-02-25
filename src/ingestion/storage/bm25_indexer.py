"""In-memory BM25 indexer for sparse retrieval bootstrap."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.core.types import ChunkRecord


class BM25Indexer:
    TOKEN_RE = re.compile(r"[\w]+(?:[-_][\w]+)*", flags=re.UNICODE)

    def __init__(self) -> None:
        self._records: list[ChunkRecord] = []
        self._doc_tokens: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()

    def build(self, records: list[ChunkRecord]) -> None:
        self._records = records
        self._doc_tokens.clear()
        self._doc_lengths.clear()
        self._df.clear()

        for record in records:
            terms = self._tokenize(record.text)
            tf = Counter(terms)
            self._doc_tokens[record.id] = tf
            self._doc_lengths[record.id] = len(terms)
            for term in tf.keys():
                self._df[term] += 1

    def query(
        self, query_text: str, top_k: int = 20, k1: float = 1.5, b: float = 0.75
    ) -> list[dict]:
        terms = self._tokenize(query_text)
        if not terms:
            return []
        n_docs = max(len(self._records), 1)
        avg_dl = sum(self._doc_lengths.values()) / n_docs if self._doc_lengths else 1.0

        scored: list[tuple[float, ChunkRecord]] = []
        for record in self._records:
            tf = self._doc_tokens.get(record.id, Counter())
            dl = self._doc_lengths.get(record.id, 0)
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = f + k1 * (1 - b + b * (dl / max(avg_dl, 1e-9)))
                score += idf * ((f * (k1 + 1)) / max(denom, 1e-9))
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": record.id,
                "score": score,
                "metadata": dict(record.metadata),
                "document": record.text,
                "route": "sparse",
            }
            for score, record in scored[:top_k]
        ]

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in self.TOKEN_RE.findall(text)]
