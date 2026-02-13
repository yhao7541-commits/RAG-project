"""向量存储层的基础抽象定义。

本模块负责定义“向量库该怎么被调用”的统一契约。
上层代码只依赖 BaseVectorStore，就可以在不同后端之间切换
（例如 Chroma / Qdrant / Milvus），而不需要改业务逻辑。

核心设计点：
1. 统一 upsert/query 接口，避免上层感知后端差异。
2. 提供可复用的输入校验，减少重复代码。
3. 对可选能力（delete/clear）给出默认实现与明确报错。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """所有向量存储实现都应继承的抽象基类。"""

    def validate_records(self, records: list[dict[str, Any]]) -> None:
        """校验 upsert 输入记录。

        记录结构要求：
        - 每条记录必须是 dict。
        - 必须包含 `id` 字段。
        - 必须包含 `vector` 字段。
        - `vector` 必须是非空 list/tuple。

        参数设计说明：
        - 使用 `list[dict]` 而不是自定义类，
          是为了兼容多种上游数据来源（文件、数据库、API）并降低接入门槛。
        """

        # 步骤 1：至少要有一条记录。
        if not records:
            raise ValueError("Records list cannot be empty")

        # 步骤 2：逐条校验结构与向量字段。
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {index} is not a dict")

            if "id" not in record:
                raise ValueError(f"Record at index {index} missing required field: 'id'")

            if "vector" not in record:
                raise ValueError(
                    f"Record at index {index} missing required field: 'vector'"
                )

            vector_value = record["vector"]
            if not isinstance(vector_value, (list, tuple)):
                raise ValueError(f"Record at index {index} has invalid vector type")

            if len(vector_value) == 0:
                raise ValueError(f"Record at index {index} has empty vector")

    def validate_query_vector(self, vector: list[float], top_k: int) -> None:
        """校验 query 输入参数。

        参数说明：
        - vector: 查询向量。
        - top_k: 希望返回的最相似结果数量。
        """

        # 规则 1：查询向量必须是 list 或 tuple。
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Query vector must be a list or tuple")

        # 规则 2：向量不能为空。
        if len(vector) == 0:
            raise ValueError("Query vector cannot be empty")

        # 规则 3：top_k 必须是正整数。
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    @abstractmethod
    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """将记录写入向量库（已存在则更新，不存在则新增）。

        参数说明：
        - records: 批量写入记录。
        - trace: 追踪上下文（可选）。
        - **kwargs: 后端私有参数扩展位。
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """按查询向量检索最相似的前 top_k 条结果。

        参数说明：
        - vector: 查询向量。
        - top_k: 返回数量。
        - filters: 元数据过滤条件（可选）。
        - trace: 追踪上下文（可选）。
        - **kwargs: 后端私有参数扩展位。
        """

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """可选能力：删除指定 id 的记录。

        默认不实现，要求具体后端按需覆盖。
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete"
        )

    def clear(self, **kwargs: Any) -> None:
        """可选能力：清空当前集合/索引。

        默认不实现，要求具体后端按需覆盖。
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear"
        )
