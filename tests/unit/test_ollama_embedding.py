"""Ollama Embedding（LlamaIndex 适配）单元测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError


@pytest.fixture
def mock_settings_ollama() -> Any:
    """构造最小可用的 Ollama embedding 配置。"""

    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "ollama"
    settings.embedding.model = "nomic-embed-text"
    settings.embedding.dimensions = 768
    settings.embedding.base_url = None
    return settings


@pytest.fixture
def mock_batch_vectors() -> list[list[float]]:
    """构造稳定可断言的向量结果。"""

    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


class TestOllamaEmbedding:
    """验证 OllamaEmbedding 的初始化、调用与错误处理。"""

    def test_initialization_default(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama)

        assert embedding.model == "nomic-embed-text"
        assert embedding.dimension == 768
        assert embedding.base_url == OllamaEmbedding.DEFAULT_BASE_URL
        assert embedding.timeout == OllamaEmbedding.DEFAULT_TIMEOUT

    def test_initialization_with_custom_base_url(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(
            mock_settings_ollama,
            base_url="http://custom-ollama:11434",
        )

        assert embedding.base_url == "http://custom-ollama:11434"

    def test_initialization_with_env_var(
        self,
        mock_settings_ollama: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama:11434")

        embedding = OllamaEmbedding(mock_settings_ollama)

        assert embedding.base_url == "http://env-ollama:11434"

    def test_initialization_with_custom_timeout(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama, timeout=60.0)
        assert embedding.timeout == 60.0

    def test_initialization_without_dimensions_setting(self, mock_settings_ollama: Any) -> None:
        delattr(mock_settings_ollama.embedding, "dimensions")

        embedding = OllamaEmbedding(mock_settings_ollama)

        assert embedding.dimension == OllamaEmbedding.DEFAULT_DIMENSION

    @patch.object(OllamaEmbedding, "_load_llamaindex_class")
    def test_embed_single_text(
        self,
        mock_loader: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        mock_embedder = Mock()
        mock_embedder.get_text_embedding_batch.return_value = [[0.1, 0.2, 0.3]]
        mock_provider_cls = Mock(return_value=mock_embedder)
        mock_loader.return_value = mock_provider_cls

        embedding = OllamaEmbedding(mock_settings_ollama)
        result = embedding.embed(["hello world"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_provider_cls.assert_called_once_with(
            model_name="nomic-embed-text",
            base_url=OllamaEmbedding.DEFAULT_BASE_URL,
            client_kwargs={"timeout": OllamaEmbedding.DEFAULT_TIMEOUT},
        )
        mock_embedder.get_text_embedding_batch.assert_called_once_with(["hello world"])

    @patch.object(OllamaEmbedding, "_load_llamaindex_class")
    def test_embed_multiple_texts(
        self,
        mock_loader: Mock,
        mock_settings_ollama: Any,
        mock_batch_vectors: list[list[float]],
    ) -> None:
        mock_embedder = Mock()
        mock_embedder.get_text_embedding_batch.return_value = mock_batch_vectors
        mock_provider_cls = Mock(return_value=mock_embedder)
        mock_loader.return_value = mock_provider_cls

        embedding = OllamaEmbedding(mock_settings_ollama)
        result = embedding.embed(["hello world", "test"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch.object(OllamaEmbedding, "_load_llamaindex_class")
    def test_embed_with_override_kwargs(
        self,
        mock_loader: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        mock_embedder = Mock()
        mock_embedder.get_text_embedding_batch.return_value = [[0.7, 0.8, 0.9]]
        mock_provider_cls = Mock(return_value=mock_embedder)
        mock_loader.return_value = mock_provider_cls

        embedding = OllamaEmbedding(mock_settings_ollama)
        embedding.embed(
            ["override"],
            model="bge-m3",
            base_url="http://another-host:11434",
            timeout=30.0,
            client_kwargs={"verify": False},
        )

        mock_provider_cls.assert_called_once_with(
            model_name="bge-m3",
            base_url="http://another-host:11434",
            client_kwargs={"timeout": 30.0, "verify": False},
        )

    @patch.object(OllamaEmbedding, "_load_llamaindex_class")
    def test_embed_api_error(self, mock_loader: Mock, mock_settings_ollama: Any) -> None:
        mock_embedder = Mock()
        mock_embedder.get_text_embedding_batch.side_effect = Exception("API Error")
        mock_provider_cls = Mock(return_value=mock_embedder)
        mock_loader.return_value = mock_provider_cls

        embedding = OllamaEmbedding(mock_settings_ollama)

        with pytest.raises(OllamaEmbeddingError, match="Ollama Embeddings API call failed"):
            embedding.embed(["test"])

    @patch.object(OllamaEmbedding, "_load_llamaindex_class")
    def test_embed_length_mismatch(self, mock_loader: Mock, mock_settings_ollama: Any) -> None:
        mock_embedder = Mock()
        mock_embedder.get_text_embedding_batch.return_value = [[0.1, 0.2]]
        mock_provider_cls = Mock(return_value=mock_embedder)
        mock_loader.return_value = mock_provider_cls

        embedding = OllamaEmbedding(mock_settings_ollama)

        with pytest.raises(OllamaEmbeddingError, match="Output length mismatch"):
            embedding.embed(["test1", "test2"])

    def test_embed_empty_list(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama)
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embedding.embed([])

    def test_embed_with_empty_string(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama)
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.embed([""])

    def test_embed_with_non_string(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama)
        invalid_texts: list[Any] = [123]
        with pytest.raises(ValueError, match="not a string"):
            embedding.embed(invalid_texts)

    @patch("src.libs.embedding.ollama_embedding.import_module")
    def test_load_llamaindex_class_missing_dependency(self, mock_import: Mock) -> None:
        mock_import.side_effect = ModuleNotFoundError("missing")

        with pytest.raises(OllamaEmbeddingError, match="llama-index-embeddings-ollama"):
            OllamaEmbedding._load_llamaindex_class()

    @patch("src.libs.embedding.ollama_embedding.import_module")
    def test_load_llamaindex_class_missing_symbol(self, mock_import: Mock) -> None:
        mock_import.return_value = Mock(OllamaEmbedding=None)

        with pytest.raises(OllamaEmbeddingError, match="Failed to load LlamaIndex"):
            OllamaEmbedding._load_llamaindex_class()

    def test_get_dimension(self, mock_settings_ollama: Any) -> None:
        embedding = OllamaEmbedding(mock_settings_ollama)
        assert embedding.get_dimension() == 768

    def test_get_dimension_default(self, mock_settings_ollama: Any) -> None:
        delattr(mock_settings_ollama.embedding, "dimensions")
        embedding = OllamaEmbedding(mock_settings_ollama)
        assert embedding.get_dimension() == OllamaEmbedding.DEFAULT_DIMENSION


class TestOllamaEmbeddingFactoryIntegration:
    """验证 Ollama provider 与工厂集成是否正常。"""

    def setup_method(self) -> None:
        EmbeddingFactory._PROVIDERS.clear()

    def test_factory_creates_ollama_embedding(self, mock_settings_ollama: Any) -> None:
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)

        embedding = EmbeddingFactory.create(mock_settings_ollama)

        assert isinstance(embedding, OllamaEmbedding)
        assert embedding.model == "nomic-embed-text"

    def test_factory_with_override_kwargs(self, mock_settings_ollama: Any) -> None:
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)

        embedding = EmbeddingFactory.create(
            mock_settings_ollama,
            base_url="http://override:11434",
            timeout=30.0,
        )

        assert isinstance(embedding, OllamaEmbedding)
        assert embedding.base_url == "http://override:11434"
        assert embedding.timeout == 30.0
