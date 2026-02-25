"""Base document loader contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.types import Document


class BaseLoader(ABC):
    """Abstract loader for source documents."""

    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    @abstractmethod
    def load(self, file_path: str | Path, **kwargs: Any) -> Document:
        """Load a source file into a normalized `Document`."""
