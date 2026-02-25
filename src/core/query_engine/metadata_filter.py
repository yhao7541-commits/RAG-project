"""Metadata filtering utilities."""

from __future__ import annotations

from typing import Any


class MetadataFilter:
    """Applies post-retrieval metadata filters as a safety net."""

    @staticmethod
    def apply(
        results: list[dict[str, Any]], filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not filters:
            return results
        output: list[dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                output.append(item)
                continue
            matched = True
            for key, value in filters.items():
                # missing->include to avoid aggressive false negatives
                if key in metadata and metadata.get(key) != value:
                    matched = False
                    break
            if matched:
                output.append(item)
        return output
