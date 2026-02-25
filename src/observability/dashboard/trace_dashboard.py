"""Streamlit dashboard for trace inspection."""

from __future__ import annotations

import json
from pathlib import Path


def load_traces(log_file: str) -> list[dict]:
    path = Path(log_file)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_dashboard(log_file: str = "logs/traces.jsonl") -> None:
    try:
        import streamlit as st
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("streamlit is required for dashboard") from e

    st.set_page_config(page_title="RAG Trace Dashboard", layout="wide")
    st.title("RAG Trace Dashboard")

    traces = load_traces(log_file)
    st.caption(f"Loaded {len(traces)} traces from {log_file}")
    if not traces:
        st.info("No traces found.")
        return

    trace_ids = [t.get("trace_id", "unknown") for t in traces]
    selected_id = st.selectbox("Trace ID", options=trace_ids)
    selected = next((t for t in traces if t.get("trace_id") == selected_id), traces[0])

    st.subheader("Overview")
    st.json(
        {
            "trace_id": selected.get("trace_id"),
            "started_at": selected.get("started_at"),
            "ended_at": selected.get("ended_at"),
            "total_latency": selected.get("total_latency"),
            "user_query": selected.get("user_query"),
        }
    )

    st.subheader("Stages")
    stages = selected.get("stages", {})
    if isinstance(stages, dict):
        for stage_name, stage_payload in stages.items():
            with st.expander(stage_name, expanded=False):
                st.json(stage_payload)
