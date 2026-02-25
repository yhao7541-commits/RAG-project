"""Chroma 向量库默认实现。

这个实现把项目统一的 `BaseVectorStore` 接口映射到 ChromaDB：
1. `upsert`：写入/更新向量记录。
2. `query`：按查询向量检索 Top-K 相似结果。
3. `delete/clear`：提供常用运维能力。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStoreError(RuntimeError):
    """ChromaStore 运行时统一异常。"""


class ChromaStore(BaseVectorStore):
    """基于 ChromaDB 的向量存储实现。"""

    DEFAULT_PERSIST_DIRECTORY = "./data/db/chroma"
    DEFAULT_COLLECTION_NAME = "knowledge_hub"

    @staticmethod
    def _load_chromadb_module() -> Any:
        """延迟加载 chromadb，避免在模块导入阶段就强依赖。"""

        try:
            return import_module("chromadb")
        except Exception as error:  # noqa: BLE001 - 统一包装导入失败
            raise ChromaStoreError(
                "chromadb is required for ChromaStore. Install it with: pip install chromadb"
            ) from error

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        """把配置值规范为可选字符串。"""

        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return None

    def __init__(
        self,
        settings: Any,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Chroma 客户端与集合。"""

        self.settings = settings

        vector_store_settings = getattr(settings, "vector_store", None)
        settings_persist_directory = self._as_optional_str(
            getattr(vector_store_settings, "persist_directory", None)
        )
        settings_collection_name = self._as_optional_str(
            getattr(vector_store_settings, "collection_name", None)
        )

        self.persist_directory = (
            persist_directory
            or settings_persist_directory
            or self.DEFAULT_PERSIST_DIRECTORY
        )
        self.collection_name = (
            collection_name
            or settings_collection_name
            or self.DEFAULT_COLLECTION_NAME
        )

        # 支持测试注入 client，避免依赖真实 Chroma 服务。
        self._client = client or self._create_client()
        self._collection_metadata = kwargs.get("collection_metadata")
        self._collection = self._get_or_create_collection()

    def _create_client(self) -> Any:
        """创建 Chroma PersistentClient。"""

        chromadb_module = self._load_chromadb_module()
        client_class = getattr(chromadb_module, "PersistentClient", None)
        if client_class is None:
            raise ChromaStoreError("Failed to load chromadb.PersistentClient")

        try:
            return client_class(path=self.persist_directory)
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(f"Failed to create Chroma client: {error}") from error

    def _get_or_create_collection(self) -> Any:
        """获取或创建目标集合。"""

        try:
            if isinstance(self._collection_metadata, dict):
                return self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata=self._collection_metadata,
                )
            return self._client.get_or_create_collection(name=self.collection_name)
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(
                f"Failed to get or create Chroma collection '{self.collection_name}': {error}"
            ) from error

    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """写入或更新记录到 Chroma。"""

        self.validate_records(records)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for index, record in enumerate(records):
            ids.append(str(record["id"]))

            try:
                vector_values = [float(value) for value in record["vector"]]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Record at index {index} contains non-numeric vector values"
                ) from error
            embeddings.append(vector_values)

            metadata = record.get("metadata")
            metadatas.append(metadata if isinstance(metadata, dict) else {})

            document_value = record.get("text", record.get("document", ""))
            documents.append(document_value if isinstance(document_value, str) else "")

        payload: dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
        if any(documents):
            payload["documents"] = documents

        try:
            self._collection.upsert(**payload)
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(f"Chroma upsert failed: {error}") from error

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """按查询向量检索最相似记录。"""

        self.validate_query_vector(vector, top_k)

        include = kwargs.get("include", ["metadatas", "documents", "distances"])
        where_filter = filters if isinstance(filters, dict) and filters else None

        try:
            response = self._collection.query(
                query_embeddings=[[float(value) for value in vector]],
                n_results=top_k,
                where=where_filter,
                include=include,
            )
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(f"Chroma query failed: {error}") from error

        return self._format_query_response(response)

    @staticmethod
    def _format_query_response(response: dict[str, Any]) -> list[dict[str, Any]]:
        """把 Chroma 原始响应格式转换为项目统一结果格式。"""

        ids_nested = response.get("ids") or [[]]
        distances_nested = response.get("distances") or [[]]
        metadatas_nested = response.get("metadatas") or [[]]
        documents_nested = response.get("documents") or [[]]

        ids = ids_nested[0] if isinstance(ids_nested, list) and ids_nested else []
        distances = (
            distances_nested[0]
            if isinstance(distances_nested, list) and distances_nested
            else []
        )
        metadatas = (
            metadatas_nested[0]
            if isinstance(metadatas_nested, list) and metadatas_nested
            else []
        )
        documents = (
            documents_nested[0]
            if isinstance(documents_nested, list) and documents_nested
            else []
        )

        results: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else None

            score = None
            if distance is not None:
                score = 1.0 / (1.0 + float(distance))

            results.append(
                {
                    "id": item_id,
                    "score": score,
                    "distance": distance,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "document": document,
                }
            )

        return results

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """删除指定 id 记录。"""

        if not ids:
            raise ValueError("ids cannot be empty")

        try:
            self._collection.delete(ids=[str(item_id) for item_id in ids])
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(f"Chroma delete failed: {error}") from error

    def clear(self, **kwargs: Any) -> None:
        """清空当前集合（删除后重建空集合）。"""

        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._get_or_create_collection()
        except Exception as error:  # noqa: BLE001
            raise ChromaStoreError(f"Chroma clear failed: {error}") from error
