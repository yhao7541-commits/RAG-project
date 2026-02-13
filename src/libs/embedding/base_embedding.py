"""Embedding 提供者的基础抽象层。

这个文件的目标是把不同厂商/不同后端的向量化能力统一成同一套接口，
让上层业务代码只关心“给文本 -> 拿到向量”，而不需要关心底层细节。

为什么要有这个抽象层：
1. 方便替换实现（OpenAI / Azure / Ollama / 本地模型）。
2. 方便测试（可以很容易写 FakeEmbedding）。
3. 统一输入校验规则，避免各实现各写一套重复逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """所有 Embedding 提供者都需要继承的抽象基类。

    约定：
    - `embed()` 负责把一批文本转换为一批向量。
    - `get_dimension()` 返回向量维度（如果实现可知）。
    - `validate_texts()` 提供统一的输入校验逻辑，子类可直接复用。
    """

    def validate_texts(self, texts: list[str]) -> None:
        """校验 embedding 输入文本列表。

        参数:
        - texts: 待向量化的文本列表。

        校验规则:
        1) 列表不能为空。
        2) 每个元素必须是字符串。
        3) 字符串不能是空串或纯空白。

        抛出:
        - ValueError: 任一规则不满足时抛出，错误信息尽量可读可定位。

        参数设计说明：
        - 为什么是 `list[str]`：embedding 常见是“批量输入换吞吐”，
          用列表能一次处理多条文本，减少远程调用次数。
        """

        # 步骤 1：先保证“有输入”，否则后续流程没有意义。
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # 步骤 2：逐个元素检查类型和内容质量。
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Item at index {index} is not a string")

            if not text.strip():
                raise ValueError(
                    f"Item at index {index} is empty or whitespace-only"
                )

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """将一批文本转换为向量。

        参数:
        - texts: 输入文本列表。
        - trace: 可选追踪上下文（便于可观测性埋点）。
        - **kwargs: 子类实现需要的额外参数。

        返回:
        - list[list[float]]: 与输入文本一一对应的向量列表。

        参数设计说明：
        - `trace`: 预留给观测系统（例如记录本次调用耗时、请求ID）。
        - `**kwargs`: 允许子类扩展厂商私有参数，
          避免每加一个新参数就修改基类签名。
        """

    def get_dimension(self) -> int | None:
        """返回向量维度。

        默认行为:
        - 基类不做猜测，强制提醒子类实现自己的维度逻辑。
        - 如果某些模型维度不固定，可在子类中返回 None。

        参数设计说明：
        - 返回 `int | None` 而不是仅 `int`，
          是为了兼容“运行时才能确定维度”的模型实现。
        """

        raise NotImplementedError("Embedding provider must implement get_dimension")
