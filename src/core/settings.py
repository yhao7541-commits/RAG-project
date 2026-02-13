"""Settings loading and validation.

This module provides a minimal, type-safe configuration loader for the project.

Design principles:
- Fail-fast: missing required fields raise a readable error that includes field path
- No side effects: this module only parses/validates configuration; no network/IO init
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class SettingsError(ValueError):
    """Raised when settings are missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: list[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings


def _require_section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None or not isinstance(value, Mapping):
        raise SettingsError(f"Missing required section: {key}")
    return value


def _optional_section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SettingsError(f"Invalid section type: {key}")
    return value


def _require(raw: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in raw:
        raise SettingsError(f"Missing required field: {path}")
    return raw[key]


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Invalid value for {path}: expected non-empty string")
    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(f"Invalid value for {path}: expected int")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SettingsError(f"Invalid value for {path}: expected bool")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"Invalid value for {path}: expected float")
    return float(value)


def _as_str_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SettingsError(f"Invalid value for {path}: expected list[str]")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SettingsError(f"Invalid value for {path}[{i}]: expected str")
        out.append(item)
    return out


def validate_settings(settings: Settings) -> None:
    """Validate required fields and basic invariants."""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")


def load_settings(path: str | Path) -> Settings:
    """Load settings from a YAML file."""

    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    try:
        raw_obj = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SettingsError(f"Invalid YAML in settings file: {settings_path}") from e

    if raw_obj is None or not isinstance(raw_obj, Mapping):
        raise SettingsError(f"Invalid settings root: expected mapping in {settings_path}")

    llm_raw = _require_section(raw_obj, "llm")
    embedding_raw = _require_section(raw_obj, "embedding")
    vector_store_raw = _require_section(raw_obj, "vector_store")
    retrieval_raw = _require_section(raw_obj, "retrieval")
    rerank_raw = _require_section(raw_obj, "rerank")
    evaluation_raw = _require_section(raw_obj, "evaluation")
    observability_raw = _require_section(raw_obj, "observability")
    ingestion_raw = _optional_section(raw_obj, "ingestion")

    llm = LLMSettings(
        provider=_as_str(_require(llm_raw, "provider", "llm.provider"), "llm.provider"),
        model=_as_str(_require(llm_raw, "model", "llm.model"), "llm.model"),
        temperature=_as_float(llm_raw.get("temperature", 0.0), "llm.temperature"),
        max_tokens=_as_int(llm_raw.get("max_tokens", 1024), "llm.max_tokens"),
    )

    embedding = EmbeddingSettings(
        provider=_as_str(
            _require(embedding_raw, "provider", "embedding.provider"),
            "embedding.provider",
        ),
        model=_as_str(_require(embedding_raw, "model", "embedding.model"), "embedding.model"),
        dimensions=_as_int(
            _require(embedding_raw, "dimensions", "embedding.dimensions"),
            "embedding.dimensions",
        ),
    )

    vector_store = VectorStoreSettings(
        provider=_as_str(
            _require(vector_store_raw, "provider", "vector_store.provider"),
            "vector_store.provider",
        ),
        persist_directory=_as_str(
            _require(vector_store_raw, "persist_directory", "vector_store.persist_directory"),
            "vector_store.persist_directory",
        ),
        collection_name=_as_str(
            _require(vector_store_raw, "collection_name", "vector_store.collection_name"),
            "vector_store.collection_name",
        ),
    )

    retrieval = RetrievalSettings(
        dense_top_k=_as_int(
            _require(retrieval_raw, "dense_top_k", "retrieval.dense_top_k"),
            "retrieval.dense_top_k",
        ),
        sparse_top_k=_as_int(
            _require(retrieval_raw, "sparse_top_k", "retrieval.sparse_top_k"),
            "retrieval.sparse_top_k",
        ),
        fusion_top_k=_as_int(
            _require(retrieval_raw, "fusion_top_k", "retrieval.fusion_top_k"),
            "retrieval.fusion_top_k",
        ),
        rrf_k=_as_int(_require(retrieval_raw, "rrf_k", "retrieval.rrf_k"), "retrieval.rrf_k"),
    )

    rerank = RerankSettings(
        enabled=_as_bool(_require(rerank_raw, "enabled", "rerank.enabled"), "rerank.enabled"),
        provider=_as_str(_require(rerank_raw, "provider", "rerank.provider"), "rerank.provider"),
        model=_as_str(_require(rerank_raw, "model", "rerank.model"), "rerank.model"),
        top_k=_as_int(_require(rerank_raw, "top_k", "rerank.top_k"), "rerank.top_k"),
    )

    evaluation = EvaluationSettings(
        enabled=_as_bool(
            _require(evaluation_raw, "enabled", "evaluation.enabled"),
            "evaluation.enabled",
        ),
        provider=_as_str(
            _require(evaluation_raw, "provider", "evaluation.provider"),
            "evaluation.provider",
        ),
        metrics=_as_str_list(
            evaluation_raw.get("metrics", []),
            "evaluation.metrics",
        ),
    )

    observability = ObservabilitySettings(
        log_level=_as_str(
            _require(observability_raw, "log_level", "observability.log_level"),
            "observability.log_level",
        ),
        trace_enabled=_as_bool(
            _require(observability_raw, "trace_enabled", "observability.trace_enabled"),
            "observability.trace_enabled",
        ),
        trace_file=_as_str(
            _require(observability_raw, "trace_file", "observability.trace_file"),
            "observability.trace_file",
        ),
        structured_logging=_as_bool(
            _require(
                observability_raw,
                "structured_logging",
                "observability.structured_logging",
            ),
            "observability.structured_logging",
        ),
    )

    ingestion = IngestionSettings(
        chunk_size=_as_int(ingestion_raw.get("chunk_size", 1000), "ingestion.chunk_size"),
        chunk_overlap=_as_int(
            ingestion_raw.get("chunk_overlap", 200),
            "ingestion.chunk_overlap",
        ),
        splitter=_as_str(ingestion_raw.get("splitter", "recursive"), "ingestion.splitter"),
        batch_size=_as_int(ingestion_raw.get("batch_size", 100), "ingestion.batch_size"),
    )

    settings = Settings(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        retrieval=retrieval,
        rerank=rerank,
        evaluation=evaluation,
        observability=observability,
        ingestion=ingestion,
    )

    validate_settings(settings)
    return settings
