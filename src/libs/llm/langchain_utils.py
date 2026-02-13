"""LangChain 相关的通用小工具函数（给 LLM provider 复用）。

为什么需要这个文件：
1) OpenAI / DeepSeek 等 provider 都会把项目内 `Message` 转成 LangChain 消息对象。
2) 这些 provider 也都需要从 LangChain 的返回对象里提取 token 统计。
3) 如果每个 provider 都自己写一遍，很容易出现“某个文件悄悄改了规则”导致行为不一致。

这个文件只做两件事：
- messages_to_langchain: 统一消息角色映射。
- extract_usage_from_ai_message: 统一 token 统计提取逻辑。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.libs.llm.base_llm import Message


def messages_to_langchain(messages: Sequence[Message]) -> list[BaseMessage]:
    """把项目内部 Message 列表转换为 LangChain 消息对象列表。

    参数说明：
    - messages: 项目统一的消息结构（role/content）。

    角色映射规则：
    - system -> SystemMessage（系统指令/上下文）
    - user -> HumanMessage（用户输入）
    - assistant -> AIMessage（历史助手回复）

    为什么要显式映射：
    - LangChain 内部会根据消息类型决定如何组织请求。
    - 统一映射能保证不同 provider 的行为一致。
    """

    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        else:
            converted.append(HumanMessage(content=message.content))
    return converted


def extract_usage_from_ai_message(ai_message: Any) -> dict[str, int]:
    """从 LangChain 的 AIMessage 响应对象中提取 token 统计。

    返回结构（统一口径）：
    - prompt_tokens: 输入 token 数
    - completion_tokens: 输出 token 数
    - total_tokens: 总 token 数

    兼容策略：
    1) 优先读取 `usage_metadata`（LangChain 常见字段）。
    2) 再尝试读取 `response_metadata.token_usage`（部分模型/版本使用）。
    3) 都没有就返回 0，保证上层结构稳定。
    """

    usage_metadata = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return {
            "prompt_tokens": int(usage_metadata.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage_metadata.get("output_tokens", 0) or 0),
            "total_tokens": int(usage_metadata.get("total_tokens", 0) or 0),
        }

    response_metadata = getattr(ai_message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {})
        if isinstance(token_usage, dict):
            return {
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
            }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
