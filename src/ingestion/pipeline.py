"""Ingestion pipeline orchestration (MVP)."""

from __future__ import annotations

from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord, Document
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from src.ingestion.storage import BM25Indexer, VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.embedding import EmbeddingFactory
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.vector_store import VectorStoreFactory


class IngestionPipeline:
    """Coordinates loader -> chunk -> transform -> encode -> upsert."""

    def __init__(self, settings):
        self.settings = settings
        self.integrity = SQLiteIntegrityChecker()
        self.loader = PdfLoader(extract_images=True)
        self.chunker = DocumentChunker(settings)
        self.refiner = ChunkRefiner(settings)
        self.enricher = MetadataEnricher(settings)
        self.captioner = ImageCaptioner(settings)

        embedding = EmbeddingFactory.create(settings)
        self.dense_encoder = DenseEncoder(embedding, batch_size=settings.ingestion.batch_size)
        self.sparse_encoder = SparseEncoder()
        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=settings.ingestion.batch_size,
        )

        self.vector_store = VectorStoreFactory.create(settings)
        self.vector_upserter = VectorUpserter(self.vector_store)
        self.bm25_indexer = BM25Indexer()

    def ingest_pdf(
        self, file_path: str, collection: str | None = None, trace: TraceContext | None = None
    ) -> dict[str, object]:
        trace_ctx = trace or TraceContext(log_file=self.settings.observability.trace_file)
        file_hash = self.integrity.compute_sha256(file_path)
        if self.integrity.should_skip(file_hash):
            trace_ctx.record_stage("integrity", {"skipped": True})
            return {"status": "skipped", "reason": "unchanged_file"}

        try:
            doc: Document = self.loader.load(file_path)
            chunks = self.chunker.split_document(doc)
            chunks = self.refiner.transform(chunks, trace=trace_ctx)
            chunks = self.enricher.transform(chunks, trace=trace_ctx)
            chunks = self.captioner.transform(chunks, trace=trace_ctx)

            batch_result = self.batch_processor.process(chunks, trace=trace_ctx)

            records: list[ChunkRecord] = []
            for idx, chunk in enumerate(chunks):
                dense = (
                    batch_result.dense_vectors[idx]
                    if idx < len(batch_result.dense_vectors)
                    else None
                )
                sparse_stat = (
                    batch_result.sparse_stats[idx] if idx < len(batch_result.sparse_stats) else None
                )
                sparse_tf = (
                    sparse_stat.get("term_frequencies", {}) if isinstance(sparse_stat, dict) else {}
                )
                records.append(
                    ChunkRecord.from_chunk(chunk, dense_vector=dense, sparse_vector=sparse_tf)
                )

            self.vector_upserter.upsert_records(records, trace=trace_ctx)
            self.bm25_indexer.build(records)

            self.integrity.mark_success(file_hash, file_path, collection=collection)
            trace_ctx.record_stage("ingestion", {"chunks": len(chunks), "upserted": len(records)})
            trace_ctx.finish()
            return {"status": "success", "chunks": len(chunks)}
        except Exception as e:  # noqa: BLE001
            self.integrity.mark_failed(file_hash, file_path, str(e))
            trace_ctx.record_stage("ingestion_error", {"error": str(e)})
            trace_ctx.finish()
            raise
