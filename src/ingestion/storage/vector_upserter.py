"""Idempotent upsert adapter for vector store writes."""

from __future__ import annotations

from hashlib import sha256

from src.core.types import ChunkRecord


class VectorUpserter:
    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store

    @staticmethod
    def make_chunk_id(source_path: str, section_path: str, content: str) -> str:
        key = f"{source_path}|{section_path}|{content}"
        return sha256(key.encode("utf-8")).hexdigest()

    def upsert_records(self, records: list[ChunkRecord], trace=None) -> None:
        payload = []
        for record in records:
            if record.dense_vector is None:
                continue
            payload.append(
                {
                    "id": record.id,
                    "vector": record.dense_vector,
                    "metadata": record.metadata,
                    "text": record.text,
                }
            )
        if payload:
            self.vector_store.upsert(payload, trace=trace)
