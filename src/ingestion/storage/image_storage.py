"""Image persistence helper for ingestion pipeline."""

from __future__ import annotations

from pathlib import Path


class ImageStorage:
    def __init__(self, base_dir: str = "data/images") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image_id: str, content: bytes, suffix: str = ".bin") -> str:
        path = self.base_dir / f"{image_id}{suffix}"
        path.write_bytes(content)
        return str(path)
