"""list_collections tool implementation."""

from __future__ import annotations

from typing import Any


def list_collections(vector_store: Any) -> dict[str, Any]:
    collections: list[dict[str, Any]] = []

    client = getattr(vector_store, "_client", None)
    if client is not None and hasattr(client, "list_collections"):
        try:
            raw = client.list_collections()
            for item in raw:
                if isinstance(item, str):
                    collections.append({"name": item, "description": "", "document_count": None})
                else:
                    name = getattr(item, "name", None) or str(item)
                    collections.append({"name": name, "description": "", "document_count": None})
        except Exception:  # noqa: BLE001
            pass

    if not collections:
        default_name = getattr(vector_store, "collection_name", "knowledge_hub")
        collections.append(
            {"name": default_name, "description": "Default collection", "document_count": None}
        )

    lines = [f"- {c['name']}" for c in collections]
    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "structuredContent": {"collections": collections},
    }
