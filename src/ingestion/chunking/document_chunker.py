"""Document -> Chunk adapter with deterministic IDs and metadata propagation."""

from __future__ import annotations

import hashlib

from src.core.settings import Settings
from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """Converts normalized documents into chunk objects."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)

    @staticmethod
    def _chunk_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    def split_document(self, document: Document) -> list[Chunk]:
        if not document.text or not document.text.strip():
            raise ValueError(f"Document '{document.id}' has no text content")

        pieces = self._splitter.split_text(document.text)
        if not pieces:
            raise ValueError("Splitter returned no chunks")

        chunks: list[Chunk] = []
        for index, text in enumerate(pieces):
            chunk_id = f"{document.id}_{index:04d}_{self._chunk_hash(text)}"
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["source_ref"] = document.id

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    source_ref=document.id,
                )
            )

        return chunks
