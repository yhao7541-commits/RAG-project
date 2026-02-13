"""LLM 提供者的基础契约定义。

这个模块是“所有 LLM 实现都要遵守的公共规则”。
你可以把它理解成“统一插座”：
- 不管底层是 OpenAI、Azure、DeepSeek 还是其他后端，
  上层都通过同一套 `Message -> chat() -> ChatResponse` 协议调用。

这样做的好处：
1. 上层代码不需要知道每个厂商的细节，便于替换模型供应商。
2. 校验规则只写一次，减少重复 bug。
3. 单测更容易写（可以用 FakeLLM 快速验证流程）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence


@dataclass(frozen=True)
class Message:
    """单条对话消息（兼容 OpenAI 风格）。

    参数说明：
    - role: 说话角色。
      - `system`: 系统指令（例如“你是一个助手”）
      - `user`: 用户提问
      - `assistant`: 历史助手回复
    - content: 消息正文。
    """

    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """LLM 提供者返回的标准化响应结构。

    参数说明：
    - content: 模型最终回复的文本。
    - model: 实际使用的模型标识（便于日志与排查）。
    - usage: token 统计信息，常见键：
      - `prompt_tokens`: 输入 token 数
      - `completion_tokens`: 输出 token 数
      - `total_tokens`: 总 token 数
    - raw_response: 可选，保留底层原始响应（便于调试复杂问题）。
    """

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


class BaseLLM(ABC):
    """所有 LLM 提供者实现的抽象基类。

    子类最少要做两件事：
    1) 在 `__init__` 中读取配置并完成必要校验。
    2) 实现 `chat()`，把输入消息发给具体模型服务并返回 ChatResponse。
    """

    VALID_ROLES: ClassVar[set[str]] = {"system", "user", "assistant"}

    def __init__(self, settings: Any, **_: Any) -> None:
        # 保留 settings 到实例上，方便子类在任意方法读取配置。
        self.settings = settings

    def validate_messages(self, messages: Sequence[Message]) -> None:
        """在调用具体提供者前，统一校验消息列表。

        为什么要先校验：
        - 让错误尽量在“本地参数层”就被发现，
          而不是把非法数据发到远端 API 才失败。

        校验内容：
        1. 列表不能为空。
        2. 每个元素都必须是 `Message` 实例。
        3. role 必须在允许范围（system/user/assistant）内。
        4. content 不能为空字符串或纯空白。
        """

        if not messages:
            raise ValueError("Messages list cannot be empty")

        for index, message in enumerate(messages):
            if not isinstance(message, Message):
                raise ValueError(f"Item at index {index} is not a Message instance")

            if message.role not in self.VALID_ROLES:
                raise ValueError(
                    f"Message at index {index} has invalid role '{message.role}'"
                )

            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError(f"Message at index {index} has empty content")

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """向提供者发送消息并返回标准化响应。

        参数说明：
        - messages: 对话消息列表，通常包含 system + user + 历史 assistant。
        - trace: 追踪上下文（可用于日志、链路追踪；可选）。
        - **kwargs: 单次调用覆盖参数（如 temperature、max_tokens、model）。

        返回说明：
        - 统一返回 ChatResponse，确保上层不依赖厂商原始格式。
        """
