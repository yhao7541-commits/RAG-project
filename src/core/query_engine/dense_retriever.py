"""Dense retriever implementation."""

from __future__ import annotations

from typing import Any

from src.libs.embedding import EmbeddingFactory
from src.libs.vector_store import VectorStoreFactory


class DenseRetriever:
    """Runs embedding-based retrieval against vector store."""

    def __init__(
        self, settings: Any, vector_store: Any | None = None, embedding: Any | None = None
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreFactory.create(settings)
        self.embedding = embedding or EmbeddingFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        effective_top_k = top_k or self.settings.retrieval.dense_top_k
        query_vector = self.embedding.embed([query], trace=trace)[0]
        results = self.vector_store.query(
            vector=query_vector,
            top_k=effective_top_k,
            filters=filters,
            trace=trace,
        )
        for item in results:
            item["route"] = "dense"
        return results
