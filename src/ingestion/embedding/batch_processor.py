"""Batch processor for dense + sparse encoding orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.core.types import Chunk


@dataclass(frozen=True)
class BatchResult:
    dense_vectors: list[list[float]]
    sparse_stats: list[dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """Runs dense + sparse encoding in chunk batches with partial-failure tolerance."""

    def __init__(self, dense_encoder: Any, sparse_encoder: Any, batch_size: int = 100):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def _create_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

    def get_batch_count(self, chunk_count: int) -> int:
        if chunk_count <= 0:
            return 0
        return (chunk_count + self.batch_size - 1) // self.batch_size

    def process(self, chunks: list[Chunk], trace=None) -> BatchResult:
        if not chunks:
            raise ValueError("Cannot process empty chunks list")

        started = perf_counter()
        batches = self._create_batches(chunks)

        dense_vectors: list[list[float]] = []
        sparse_stats: list[dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        for batch_index, batch in enumerate(batches):
            batch_started = perf_counter()
            try:
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                dense_vectors.extend(batch_dense)
                sparse_stats.extend(batch_sparse)
                success_count += len(batch)

                if trace is not None:
                    trace.record_stage(
                        f"batch_{batch_index}",
                        {
                            "chunks_processed": len(batch),
                            "duration_seconds": perf_counter() - batch_started,
                        },
                    )
            except Exception as e:  # noqa: BLE001
                failed_count += len(batch)
                if trace is not None:
                    trace.record_stage(
                        f"batch_{batch_index}_error",
                        {
                            "error": str(e),
                            "chunks_failed": len(batch),
                            "duration_seconds": perf_counter() - batch_started,
                        },
                    )

        total_time = perf_counter() - started
        if trace is not None:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": len(batches),
                    "batch_size": self.batch_size,
                    "successful_chunks": success_count,
                    "failed_chunks": failed_count,
                    "total_time": total_time,
                },
            )

        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=len(batches),
            total_time=total_time,
            successful_chunks=success_count,
            failed_chunks=failed_count,
        )
