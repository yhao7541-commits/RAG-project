"""
Query Engine Module.

This package contains the hybrid search engine components:
- Query preprocessing
- Dense retrieval (embedding-based)
- Sparse retrieval (BM25)
- Result fusion (RRF)
- Reranking
"""

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import rrf_fusion
from src.core.query_engine.metadata_filter import MetadataFilter
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.query_engine.sparse_retriever import SparseRetriever

__all__ = [
    "ProcessedQuery",
    "QueryProcessor",
    "DenseRetriever",
    "SparseRetriever",
    "rrf_fusion",
    "MetadataFilter",
    "RetrievalPipeline",
]
