"""query_knowledge_hub tool implementation."""

from __future__ import annotations

from typing import Any

from src.core.query_engine import RetrievalPipeline


def query_knowledge_hub(
    pipeline: RetrievalPipeline,
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    trace=None,
) -> dict[str, Any]:
    if collection:
        query = f"{query} collection:{collection}"

    hits = pipeline.retrieve(query=query, top_k=top_k, trace=trace)
    citations = []
    lines = []
    for idx, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {})
        source = metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown"
        page = metadata.get("page") if isinstance(metadata, dict) else None
        snippet = (hit.get("document") or "")[:300]
        citations.append(
            {
                "id": idx,
                "source": source,
                "page": page,
                "text": snippet,
                "score": hit.get("score"),
                "chunk_id": hit.get("id"),
            }
        )
        page_text = f", p.{page}" if page is not None else ""
        lines.append(f"[{idx}] {source}{page_text}: {snippet}")

    return {
        "content": [
            {
                "type": "text",
                "text": "\n".join(lines) if lines else "No relevant context found.",
            }
        ],
        "structuredContent": {
            "answer": "Retrieved context snippets for downstream answer generation.",
            "citations": citations,
        },
    }
