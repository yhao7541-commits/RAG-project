"""Trace context for observability across pipeline stages."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    """Trace context for recording and persisting pipeline stages.

    Attributes:
        trace_id: Unique identifier for this trace
        started_at: Timestamp when trace was created
        stages: Dictionary storing data from each pipeline stage
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    stages: Dict[str, Any] = field(default_factory=dict)
    user_query: str | None = None
    collection: str | None = None
    log_file: str | None = None

    def record_stage(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Record data from a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage (e.g., "chunk_refiner", "embedding")
            data: Stage-specific data to record
        """
        self.stages[stage_name] = {"timestamp": datetime.now().isoformat(), "data": data}

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recorded data for a specific stage.

        Args:
            stage_name: Name of the stage to retrieve

        Returns:
            Stage data dict or None if stage not found
        """
        return self.stages.get(stage_name)

    def to_dict(self) -> Dict[str, Any]:
        ended_at = datetime.now()
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_latency": (ended_at - self.started_at).total_seconds(),
            "user_query": self.user_query,
            "collection": self.collection,
            "stages": self.stages,
        }

    def finish(self) -> Dict[str, Any]:
        payload = self.to_dict()
        if self.log_file:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
