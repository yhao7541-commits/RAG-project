"""Fusion algorithms for dense+sparse retrieval."""

from __future__ import annotations

from typing import Any


def rrf_fusion(result_sets: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked result sets."""
    bucket: dict[str, dict[str, Any]] = {}

    for result_set in result_sets:
        for rank, item in enumerate(result_set, start=1):
            item_id = str(item.get("id"))
            if item_id not in bucket:
                bucket[item_id] = {
                    "id": item_id,
                    "metadata": item.get("metadata", {}),
                    "document": item.get("document"),
                    "score": 0.0,
                }
            bucket[item_id]["score"] += 1.0 / (k + rank)

    fused = list(bucket.values())
    fused.sort(key=lambda entry: float(entry["score"]), reverse=True)
    return fused
