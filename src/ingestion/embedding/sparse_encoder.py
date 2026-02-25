"""Sparse term-frequency encoder."""

from __future__ import annotations

import re
from collections import Counter

from src.core.types import Chunk


class SparseEncoder:
    """Converts chunk text into sparse term-frequency stats."""

    TOKEN_RE = re.compile(r"[\w]+(?:[-_][\w]+)*", flags=re.UNICODE)

    def __init__(self, min_term_length: int = 2, lowercase: bool = True):
        if min_term_length < 1:
            raise ValueError("min_term_length must be >= 1")
        self.min_term_length = min_term_length
        self.lowercase = lowercase

    def _tokenize(self, text: str) -> list[str]:
        tokens = self.TOKEN_RE.findall(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        return [t for t in tokens if len(t) >= self.min_term_length]

    def encode(self, chunks: list[Chunk], trace=None) -> list[dict[str, object]]:
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")

        output: list[dict[str, object]] = []
        for idx, chunk in enumerate(chunks):
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk at index {idx} has empty or whitespace-only text")

            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            output.append(
                {
                    "chunk_id": chunk.id,
                    "term_frequencies": dict(tf),
                    "doc_length": len(tokens),
                    "unique_terms": len(tf),
                }
            )

        return output

    def get_corpus_stats(self, encoded_chunks: list[dict[str, object]]) -> dict[str, object]:
        if not encoded_chunks:
            return {"num_docs": 0, "avg_doc_length": 0.0, "document_frequency": {}}

        num_docs = len(encoded_chunks)
        total_length = 0
        df_counter: Counter[str] = Counter()

        for item in encoded_chunks:
            raw_doc_length = item.get("doc_length", 0)
            doc_length = raw_doc_length if isinstance(raw_doc_length, int) else 0
            total_length += doc_length
            term_frequencies = item.get("term_frequencies", {})
            if isinstance(term_frequencies, dict):
                for term in term_frequencies.keys():
                    if isinstance(term, str):
                        df_counter[term] += 1

        return {
            "num_docs": num_docs,
            "avg_doc_length": total_length / num_docs,
            "document_frequency": dict(df_counter),
        }
