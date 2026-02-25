"""Retrieval pipeline orchestration."""

from __future__ import annotations

from typing import Any

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import rrf_fusion
from src.core.query_engine.metadata_filter import MetadataFilter
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.libs.reranker import NoneReranker, RerankerFactory


class RetrievalPipeline:
    """Implements Query -> Hybrid retrieve -> Fusion -> Filter -> Rerank."""

    def __init__(
        self,
        settings: Any,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.reranker = reranker or RerankerFactory.create(settings)

    def retrieve(self, query: str, top_k: int | None = None, trace=None) -> list[dict[str, Any]]:
        processed = self.query_processor.process(query)
        if trace is not None:
            trace.record_stage(
                "query_processing",
                {
                    "query": processed.normalized_query,
                    "keywords": processed.keywords,
                    "filters": processed.filters,
                },
            )

        dense_results = self.dense_retriever.retrieve(
            processed.normalized_query,
            top_k=self.settings.retrieval.dense_top_k,
            filters=processed.filters,
            trace=trace,
        )
        if trace is not None:
            trace.record_stage("dense_retrieval", {"count": len(dense_results)})
        sparse_results = self.sparse_retriever.retrieve(
            processed.normalized_query,
            top_k=self.settings.retrieval.sparse_top_k,
            filters=processed.filters,
            trace=trace,
        )
        if trace is not None:
            trace.record_stage("sparse_retrieval", {"count": len(sparse_results)})

        fused = rrf_fusion([dense_results, sparse_results], k=self.settings.retrieval.rrf_k)
        if trace is not None:
            trace.record_stage(
                "fusion", {"count": len(fused), "rrf_k": self.settings.retrieval.rrf_k}
            )
        filtered = MetadataFilter.apply(fused, processed.filters)
        if trace is not None:
            trace.record_stage("metadata_filter", {"count": len(filtered)})

        final_top_k = top_k or self.settings.retrieval.fusion_top_k
        candidates = filtered[: max(final_top_k, self.settings.rerank.top_k)]

        try:
            if isinstance(self.reranker, NoneReranker) or not getattr(
                self.settings.rerank, "enabled", False
            ):
                ranked = candidates
            else:
                ranked = self.reranker.rerank(processed.normalized_query, candidates, trace=trace)
        except Exception:  # noqa: BLE001
            ranked = candidates
            if trace is not None:
                trace.record_stage("rerank", {"fallback": True, "count": len(ranked)})
        else:
            if trace is not None:
                trace.record_stage("rerank", {"fallback": False, "count": len(ranked)})

        return ranked[:final_top_k]
