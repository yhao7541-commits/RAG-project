"""ChromaStore 单元测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.libs.vector_store.chroma_store import ChromaStore, ChromaStoreError


@pytest.fixture
def mock_settings() -> Any:
    """构造最小可用 settings。"""

    settings = Mock()
    settings.vector_store = Mock()
    settings.vector_store.provider = "chroma"
    settings.vector_store.persist_directory = "./tmp/chroma"
    settings.vector_store.collection_name = "test_collection"
    return settings


@pytest.fixture
def mock_collection() -> Mock:
    """构造集合 mock。"""

    collection = Mock()
    collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "distances": [[0.1, 0.5]],
        "metadatas": [[{"source": "a"}, {"source": "b"}]],
        "documents": [["text-a", "text-b"]],
    }
    return collection


@pytest.fixture
def mock_client(mock_collection: Mock) -> Mock:
    """构造客户端 mock。"""

    client = Mock()
    client.get_or_create_collection.return_value = mock_collection
    return client


class TestChromaStore:
    """验证 ChromaStore 的初始化与核心行为。"""

    def test_initialization_reads_settings(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)

        assert store.persist_directory == "./tmp/chroma"
        assert store.collection_name == "test_collection"
        mock_client.get_or_create_collection.assert_called_once_with(name="test_collection")

    def test_initialization_with_overrides(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(
            settings=mock_settings,
            client=mock_client,
            persist_directory="./override/chroma",
            collection_name="override_collection",
        )

        assert store.persist_directory == "./override/chroma"
        assert store.collection_name == "override_collection"
        mock_client.get_or_create_collection.assert_called_once_with(name="override_collection")

    @patch("src.libs.vector_store.chroma_store.import_module")
    def test_initialization_creates_persistent_client(
        self,
        mock_import_module: Mock,
        mock_settings: Any,
    ) -> None:
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_client_class = Mock(return_value=mock_client)
        mock_chromadb_module = Mock(PersistentClient=mock_client_class)
        mock_import_module.return_value = mock_chromadb_module

        store = ChromaStore(settings=mock_settings)

        assert store.collection_name == "test_collection"
        mock_client_class.assert_called_once_with(path="./tmp/chroma")

    @patch("src.libs.vector_store.chroma_store.import_module")
    def test_missing_chromadb_dependency(
        self,
        mock_import_module: Mock,
    ) -> None:
        mock_import_module.side_effect = ModuleNotFoundError("missing chromadb")

        with pytest.raises(ChromaStoreError, match="chromadb is required"):
            ChromaStore._load_chromadb_module()

    def test_upsert_success(self, mock_settings: Any, mock_client: Mock, mock_collection: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)

        records = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source": "a.pdf"},
                "text": "hello",
            },
            {
                "id": "doc2",
                "vector": [0.4, 0.5, 0.6],
                "metadata": {"source": "b.pdf"},
                "document": "world",
            },
        ]
        store.upsert(records)

        called_kwargs = mock_collection.upsert.call_args.kwargs
        assert called_kwargs["ids"] == ["doc1", "doc2"]
        assert called_kwargs["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert called_kwargs["metadatas"] == [{"source": "a.pdf"}, {"source": "b.pdf"}]
        assert called_kwargs["documents"] == ["hello", "world"]

    def test_upsert_omits_documents_when_empty(
        self,
        mock_settings: Any,
        mock_client: Mock,
        mock_collection: Mock,
    ) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)

        records = [{"id": "doc1", "vector": [1.0, 2.0], "metadata": {"k": "v"}}]
        store.upsert(records)

        called_kwargs = mock_collection.upsert.call_args.kwargs
        assert "documents" not in called_kwargs

    def test_upsert_non_numeric_vector_raises(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)

        records = [{"id": "doc1", "vector": [0.1, "bad", 0.3]}]
        with pytest.raises(ValueError, match="non-numeric vector values"):
            store.upsert(records)

    def test_query_success(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)
        results = store.query([0.1, 0.2], top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["metadata"] == {"source": "a"}
        assert results[0]["document"] == "text-a"
        assert isinstance(results[0]["score"], float)

    def test_query_passes_filters(self, mock_settings: Any, mock_client: Mock, mock_collection: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)
        store.query([0.3, 0.4], top_k=3, filters={"source": "a"})

        called_kwargs = mock_collection.query.call_args.kwargs
        assert called_kwargs["where"] == {"source": "a"}
        assert called_kwargs["n_results"] == 3

    def test_query_wraps_errors(self, mock_settings: Any, mock_client: Mock, mock_collection: Mock) -> None:
        mock_collection.query.side_effect = RuntimeError("boom")
        store = ChromaStore(settings=mock_settings, client=mock_client)

        with pytest.raises(ChromaStoreError, match="Chroma query failed"):
            store.query([0.1, 0.2], top_k=1)

    def test_delete_success(self, mock_settings: Any, mock_client: Mock, mock_collection: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)
        store.delete(["doc1", "doc2"])

        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])

    def test_delete_empty_ids_raises(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)

        with pytest.raises(ValueError, match="ids cannot be empty"):
            store.delete([])

    def test_clear_success(self, mock_settings: Any, mock_client: Mock) -> None:
        store = ChromaStore(settings=mock_settings, client=mock_client)
        store.clear()

        mock_client.delete_collection.assert_called_once_with(name="test_collection")
        assert mock_client.get_or_create_collection.call_count == 2


class TestChromaStoreFormatting:
    """验证 query 响应格式转换。"""

    def test_format_query_response_with_missing_fields(self) -> None:
        raw_response: dict[str, Any] = {
            "ids": [["doc1"]],
            "distances": [[]],
        }

        results = ChromaStore._format_query_response(raw_response)
        assert results == [
            {
                "id": "doc1",
                "score": None,
                "distance": None,
                "metadata": {},
                "document": None,
            }
        ]
