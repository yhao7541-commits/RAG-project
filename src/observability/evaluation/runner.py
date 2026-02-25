"""Evaluation runner for golden datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluationRunner:
    def __init__(self, settings: Any, retrieval_pipeline: Any):
        self.settings = settings
        self.retrieval_pipeline = retrieval_pipeline

    def evaluate_golden_set(self, golden_set_path: str) -> dict[str, Any]:
        entries = json.loads(Path(golden_set_path).read_text(encoding="utf-8"))
        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])

        scores: list[dict[str, float]] = []
        for entry in entries:
            query = entry["query"]
            ground_truth_ids = entry.get("ground_truth_ids", [])
            retrieved = self.retrieval_pipeline.retrieve(query)
            scores.append(evaluator.evaluate(query, retrieved, ground_truth=ground_truth_ids))

        aggregate = {
            "count": len(scores),
            "hit_rate": sum(item["hit_rate"] for item in scores) / max(len(scores), 1),
            "mrr": sum(item["mrr"] for item in scores) / max(len(scores), 1),
        }
        return {"aggregate": aggregate, "samples": scores}
