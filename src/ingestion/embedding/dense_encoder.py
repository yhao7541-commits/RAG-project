"""Dense vector encoder wrapper."""

from __future__ import annotations

from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """Batches chunk text and delegates embedding generation."""

    def __init__(self, embedding: BaseEmbedding, batch_size: int = 100):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.embedding = embedding
        self.batch_size = batch_size

    def get_batch_count(self, chunk_count: int) -> int:
        if chunk_count <= 0:
            return 0
        return (chunk_count + self.batch_size - 1) // self.batch_size

    def encode(self, chunks: list[Chunk], trace=None) -> list[list[float]]:
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")

        for idx, chunk in enumerate(chunks):
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk at index {idx} has empty or whitespace-only text")

        vectors: list[list[float]] = []
        expected_dim: int | None = None

        for start in range(0, len(chunks), self.batch_size):
            end = min(start + self.batch_size, len(chunks))
            batch = chunks[start:end]
            texts = [c.text for c in batch]
            try:
                batch_vectors = self.embedding.embed(texts, trace=trace)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Failed to encode batch {start}-{end}: {e}") from e

            if len(batch_vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding provider returned {len(batch_vectors)} vectors for {len(texts)} texts"
                )

            for vec in batch_vectors:
                if expected_dim is None:
                    expected_dim = len(vec)
                elif len(vec) != expected_dim:
                    raise RuntimeError("Inconsistent vector dimensions")

            vectors.extend(batch_vectors)

        return vectors
