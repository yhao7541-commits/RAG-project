"""CLI entry for ingestion pipeline."""

from __future__ import annotations

import argparse

from src.core.settings import load_settings
from src.ingestion import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF files into vector store")
    parser.add_argument("file", help="Path to PDF file")
    parser.add_argument("--collection", default=None, help="Optional collection name")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    pipeline = IngestionPipeline(settings)
    result = pipeline.ingest_pdf(args.file, collection=args.collection)
    print(result)


if __name__ == "__main__":
    main()
