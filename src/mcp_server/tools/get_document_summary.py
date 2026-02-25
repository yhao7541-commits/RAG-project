"""get_document_summary tool implementation."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


def get_document_summary(vector_store: Any, doc_id: str) -> dict[str, Any]:
    # Query by metadata source_ref/doc_id fallback on id exact search
    summary = {
        "doc_id": doc_id,
        "title": None,
        "summary": None,
        "source_path": None,
        "tags": [],
        "images": [],
    }

    collection = getattr(vector_store, "_collection", None)
    if collection is not None and hasattr(collection, "get"):
        try:
            raw = collection.get(
                where={"source_ref": doc_id}, include=["metadatas", "documents", "ids"]
            )
            ids = (raw.get("ids") or []) if isinstance(raw, dict) else []
            metadatas = (raw.get("metadatas") or []) if isinstance(raw, dict) else []
            documents = (raw.get("documents") or []) if isinstance(raw, dict) else []

            if ids:
                metadata = metadatas[0] if metadatas else {}
                doc_text = documents[0] if documents else ""
                if isinstance(metadata, dict):
                    summary["title"] = metadata.get("title")
                    summary["source_path"] = metadata.get("source_path")
                    tags = metadata.get("tags")
                    if isinstance(tags, list):
                        summary["tags"] = tags
                    images = metadata.get("images")
                    if isinstance(images, list):
                        summary["images"] = images
                summary["summary"] = doc_text[:500]
        except Exception:  # noqa: BLE001
            pass

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Document: {summary['doc_id']}\n"
                f"Title: {summary['title'] or 'N/A'}\n"
                f"Source: {summary['source_path'] or 'N/A'}\n"
                f"Summary: {summary['summary'] or 'N/A'}"
            ),
        }
    ]

    # Multimodal return: include first image as base64 if available
    for image in summary["images"][:1]:
        if not isinstance(image, dict):
            continue
        image_path = image.get("path")
        if not isinstance(image_path, str):
            continue
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = "image/png"
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            content.append({"type": "image", "data": data, "mimeType": mime})
            break
        except Exception:  # noqa: BLE001
            continue

    return {
        "content": content,
        "structuredContent": summary,
    }
