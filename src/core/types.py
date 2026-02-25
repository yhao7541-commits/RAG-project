"""Core data types and contracts for the pipeline.

These types are shared across ingestion, retrieval, and MCP tooling.

Rules:
- metadata must include `source_path` for traceability
- types are JSON-serializable via to_dict()/from_dict()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _validate_source_path(metadata: Mapping[str, Any]) -> None:
    if "source_path" not in metadata:
        raise ValueError("metadata must contain 'source_path'")


def _validate_images_metadata(metadata: Mapping[str, Any]) -> None:
    images_value = metadata.get("images")
    if images_value is None:
        return

    if not isinstance(images_value, list):
        raise ValueError("metadata.images must be a list")

    required_fields = ("id", "path", "text_offset", "text_length")

    for index, image_ref in enumerate(images_value):
        if not isinstance(image_ref, dict):
            raise ValueError(f"metadata.images[{index}] must be a dict with image reference fields")

        for field_name in required_fields:
            if field_name not in image_ref:
                raise ValueError(f"metadata.images[{index}] missing required field: '{field_name}'")

        if not isinstance(image_ref["id"], str) or not image_ref["id"].strip():
            raise ValueError(f"metadata.images[{index}].id must be a non-empty string")

        if not isinstance(image_ref["path"], str) or not image_ref["path"].strip():
            raise ValueError(f"metadata.images[{index}].path must be a non-empty string")

        if not isinstance(image_ref["text_offset"], int) or image_ref["text_offset"] < 0:
            raise ValueError(f"metadata.images[{index}].text_offset must be a non-negative integer")

        if not isinstance(image_ref["text_length"], int) or image_ref["text_length"] < 0:
            raise ValueError(f"metadata.images[{index}].text_length must be a non-negative integer")

        page_value = image_ref.get("page")
        if page_value is not None and not isinstance(page_value, int):
            raise ValueError(f"metadata.images[{index}].page must be an integer")

        position_value = image_ref.get("position")
        if position_value is not None and not isinstance(position_value, dict):
            raise ValueError(f"metadata.images[{index}].position must be a dict")


@dataclass
class Document:
    """A loaded document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_source_path(self.metadata)
        _validate_images_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Chunk:
    """A chunk of a Document, with inherited and enriched metadata."""

    id: str
    text: str
    metadata: dict[str, Any]
    start_offset: int | None = None
    end_offset: int | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_source_path(self.metadata)
        _validate_images_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=dict(data.get("metadata", {})),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            source_ref=data.get("source_ref"),
        )


@dataclass
class ChunkRecord:
    """A stored/retrievable chunk record with optional vectors."""

    id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        _validate_source_path(self.metadata)
        _validate_images_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "dense_vector": list(self.dense_vector) if self.dense_vector is not None else None,
            "sparse_vector": dict(self.sparse_vector) if self.sparse_vector is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkRecord":
        dense = data.get("dense_vector")
        sparse = data.get("sparse_vector")
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=dict(data.get("metadata", {})),
            dense_vector=list(dense) if dense is not None else None,
            sparse_vector=dict(sparse) if sparse is not None else None,
        )

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[str, float] | None = None,
    ) -> "ChunkRecord":
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata),
            dense_vector=list(dense_vector) if dense_vector is not None else None,
            sparse_vector=dict(sparse_vector) if sparse_vector is not None else None,
        )
