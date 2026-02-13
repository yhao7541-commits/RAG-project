"""B7.1：OpenAI-Compatible LLM provider 冒烟测试（mock HTTP）。

测试目标：
1. 工厂可创建 openai/azure/deepseek provider。
2. provider 默认走 HTTP 路径，不走真实网络（全部 mock）。
3. 成功响应、API 错误、输入校验都符合统一契约。
4. 保留可选 LangChain 路径（这里只做最小可调用验证，不做真实外部调用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import (
    AzureLLM,
    AzureLLMError,
    DeepSeekLLM,
    DeepSeekLLMError,
    LLMFactory,
    Message,
    OpenAILLM,
    OpenAILLMError,
)


@dataclass
class MockLLMSettings:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class MockSettings:
    llm: MockLLMSettings


def _make_settings(provider: str) -> MockSettings:
    """生成最小 settings。"""

    return MockSettings(llm=MockLLMSettings(provider=provider))


def _make_openai_compatible_response(content: str, model: str = "gpt-4o-mini") -> MagicMock:
    """构造 OpenAI-compatible 成功响应。"""

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    return response


def _make_error_response(status_code: int, message: str) -> MagicMock:
    """构造错误响应。"""

    response = MagicMock()
    response.status_code = status_code
    response.text = message
    response.json.return_value = {"error": {"message": message}}
    return response


@pytest.fixture(autouse=True)
def ensure_providers_registered() -> None:
    """确保默认 provider 注册存在。"""

    if "openai" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("openai", OpenAILLM)
    if "azure" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("azure", AzureLLM)
    if "deepseek" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("deepseek", DeepSeekLLM)


class TestLLMFactoryRegistration:
    def test_openai_registered(self) -> None:
        assert "openai" in LLMFactory.list_providers()

    def test_azure_registered(self) -> None:
        assert "azure" in LLMFactory.list_providers()

    def test_deepseek_registered(self) -> None:
        assert "deepseek" in LLMFactory.list_providers()

    def test_factory_creates_openai(self) -> None:
        settings = _make_settings("openai")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, OpenAILLM)

    def test_factory_creates_azure(self) -> None:
        settings = _make_settings("azure")
        env = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, AzureLLM)

    def test_factory_creates_deepseek(self) -> None:
        settings = _make_settings("deepseek")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, DeepSeekLLM)


class TestOpenAILLM:
    def test_chat_http_success(self) -> None:
        settings = _make_settings("openai")
        llm = OpenAILLM(settings, api_key="test-key")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_openai_compatible_response("OpenAI response")
            )

            response = llm.chat([Message(role="user", content="Hello")])

            assert response.content == "OpenAI response"
            assert response.model == "gpt-4o-mini"
            assert response.usage["total_tokens"] == 30

    def test_chat_http_error(self) -> None:
        settings = _make_settings("openai")
        llm = OpenAILLM(settings, api_key="test-key")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_error_response(400, "bad request")
            )

            with pytest.raises(OpenAILLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])

    def test_optional_langchain_path(self) -> None:
        settings = _make_settings("openai")
        llm = OpenAILLM(settings, api_key="test-key", use_langchain=True)

        with patch("importlib.import_module") as mock_import:
            mock_chat = MagicMock()
            mock_chat.invoke.return_value = MagicMock(content="LC response", usage_metadata={})

            mock_langchain_module = MagicMock()
            mock_langchain_module.ChatOpenAI.return_value = mock_chat

            mock_utils_module = MagicMock()
            mock_utils_module.messages_to_langchain.return_value = []
            mock_utils_module.extract_usage_from_ai_message.return_value = {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }

            def _import_side_effect(name: str):
                if name == "langchain_openai":
                    return mock_langchain_module
                if name == "src.libs.llm.langchain_utils":
                    return mock_utils_module
                raise ImportError(name)

            mock_import.side_effect = _import_side_effect

            response = llm.chat([Message(role="user", content="Hello")])
            assert response.content == "LC response"


class TestAzureLLM:
    def test_chat_http_success(self) -> None:
        settings = _make_settings("azure")
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_openai_compatible_response("Azure response")
            )

            response = llm.chat([Message(role="user", content="Hello")])
            assert response.content == "Azure response"
            assert response.usage["total_tokens"] == 30

    def test_chat_http_error(self) -> None:
        settings = _make_settings("azure")
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_error_response(401, "unauthorized")
            )

            with pytest.raises(AzureLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


class TestDeepSeekLLM:
    def test_chat_http_success(self) -> None:
        settings = _make_settings("deepseek")
        llm = DeepSeekLLM(settings, api_key="test-key")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_openai_compatible_response("DeepSeek response", model="deepseek-chat")
            )

            response = llm.chat([Message(role="user", content="Hello")])
            assert response.content == "DeepSeek response"

    def test_chat_http_error(self) -> None:
        settings = _make_settings("deepseek")
        llm = DeepSeekLLM(settings, api_key="test-key")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                _make_error_response(500, "internal error")
            )

            with pytest.raises(DeepSeekLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])

    def test_optional_langchain_path(self) -> None:
        settings = _make_settings("deepseek")
        llm = DeepSeekLLM(settings, api_key="test-key", use_langchain=True)

        with patch("importlib.import_module") as mock_import:
            mock_chat = MagicMock()
            mock_chat.invoke.return_value = MagicMock(content="LC deepseek response", usage_metadata={})

            mock_langchain_module = MagicMock()
            mock_langchain_module.ChatOpenAI.return_value = mock_chat

            mock_utils_module = MagicMock()
            mock_utils_module.messages_to_langchain.return_value = []
            mock_utils_module.extract_usage_from_ai_message.return_value = {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }

            def _import_side_effect(name: str):
                if name == "langchain_openai":
                    return mock_langchain_module
                if name == "src.libs.llm.langchain_utils":
                    return mock_utils_module
                raise ImportError(name)

            mock_import.side_effect = _import_side_effect

            response = llm.chat([Message(role="user", content="Hello")])
            assert response.content == "LC deepseek response"


class TestMessageValidation:
    @pytest.mark.parametrize(
        "llm_class,env_key",
        [(OpenAILLM, "OPENAI_API_KEY"), (DeepSeekLLM, "DEEPSEEK_API_KEY")],
    )
    def test_empty_content_validation(self, llm_class: type[OpenAILLM], env_key: str) -> None:
        settings = _make_settings("openai")
        with patch.dict("os.environ", {env_key: "test-key"}):
            llm = llm_class(settings)
            with pytest.raises(ValueError, match="empty content"):
                llm.chat([Message(role="user", content="")])
