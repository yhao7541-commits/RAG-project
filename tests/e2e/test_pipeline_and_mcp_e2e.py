"""E2E smoke tests for retrieval/mcp/evaluation flow."""

from __future__ import annotations

from pathlib import Path

from src.core.settings import load_settings
from src.mcp_server.server import MCPServer
from src.observability.evaluation import EvaluationRunner


class _FakePipeline:
    def retrieve(self, query: str, top_k: int | None = None, trace=None):
        return [
            {
                "id": "doc_sample_001_0000_aaaaaaaa",
                "score": 0.9,
                "metadata": {"source_path": "tests/fixtures/sample_documents/simple.pdf"},
                "document": "Sample Document snippet",
            }
        ]


class _FakeCollection:
    def get(self, where=None, include=None):
        return {
            "ids": ["doc_sample_001_0000_aaaaaaaa"],
            "metadatas": [
                {
                    "source_path": "tests/fixtures/sample_documents/simple.pdf",
                    "title": "Sample Document",
                    "tags": ["sample"],
                    "images": [],
                }
            ],
            "documents": ["Sample Document summary text"],
        }


class _FakeVectorStore:
    collection_name = "knowledge_hub"

    def __init__(self):
        self._collection = _FakeCollection()
        self._client = self

    def list_collections(self):
        return ["knowledge_hub"]


def test_mcp_tools_smoke(monkeypatch):
    settings = load_settings("config/settings.yaml")

    monkeypatch.setattr(
        "src.mcp_server.server.RetrievalPipeline", lambda _settings: _FakePipeline()
    )
    monkeypatch.setattr(
        "src.mcp_server.server.VectorStoreFactory.create", lambda _settings: _FakeVectorStore()
    )

    server = MCPServer(settings)

    tools = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert "tools" in tools
    assert any(item["name"] == "query_knowledge_hub" for item in tools["tools"])

    query_result = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "query_knowledge_hub", "arguments": {"query": "sample"}},
        }
    )
    assert "content" in query_result
    assert "structuredContent" in query_result


def test_evaluation_runner_with_golden_set():
    settings = load_settings("config/settings.yaml")
    runner = EvaluationRunner(settings, retrieval_pipeline=_FakePipeline())
    golden = Path("tests/fixtures/golden_test_set.json")
    report = runner.evaluate_golden_set(str(golden))

    assert "aggregate" in report
    assert "hit_rate" in report["aggregate"]
    assert "mrr" in report["aggregate"]
