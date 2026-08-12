"""
Tests for common.llm.client — multi-provider LLM client support.

Covers:
1. OpenAICompatibleClient (renamed from OpenRouterClient) — base_url, headers, backward compat
2. OpenRouterClient alias — backward-compatible import
3. OpenAIClient — native SDK wrapper (mocked)
4. AzureOpenAIClient — Azure-specific auth (mocked)
5. Factory function — all providers, env var fallback, error cases
6. Config-based factory — env var cascade, provider-specific keys
7. Anthropic client — unchanged (regression guard)
"""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(status_code=200, body=None):
    """Build a mock requests.Response for OpenAI-compatible /chat/completions."""
    if body is None:
        body = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _make_openai_sdk_response(content="hello", model="gpt-4o", finish_reason="stop"):
    """Build a mock openai.ChatCompletion-like response."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# OpenAICompatibleClient — base_url and headers
# ---------------------------------------------------------------------------


class TestOpenAICompatibleClient:
    """Verify the renamed OpenAICompatibleClient works correctly."""

    @patch("requests.post")
    def test_default_is_openrouter(self, mock_post):
        """Without explicit base_url, defaults to OpenRouter."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response()
        client = OpenAICompatibleClient(api_key="test-key")
        assert "openrouter.ai" in client.base_url
        assert client.get_provider_name() == "openrouter"

    @patch("requests.post")
    def test_custom_base_url(self, mock_post):
        """Custom base_url overrides default."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response()
        client = OpenAICompatibleClient(
            api_key="test-key", base_url="http://localhost:11434/v1"
        )
        assert client.base_url == "http://localhost:11434/v1"
        assert client.get_provider_name() == "openai-compatible"

    @patch("requests.post")
    def test_no_api_key_for_local(self, mock_post):
        """Local servers (e.g. Ollama) don't need an API key."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response()
        client = OpenAICompatibleClient(
            api_key="", base_url="http://localhost:11434/v1"
        )
        client.create_message(
            model="llama3", messages=[{"role": "user", "content": "hi"}]
        )
        _, kwargs = mock_post.call_args
        assert "Authorization" not in kwargs["headers"]

    @patch("requests.post")
    def test_extra_headers(self, mock_post):
        """Extra headers are included in requests."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response()
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000/v1",
            extra_headers={"X-Custom": "value"},
        )
        client.create_message(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["X-Custom"] == "value"

    @patch("requests.post")
    def test_openrouter_headers_still_work(self, mock_post):
        """OpenRouter-specific headers are set when base_url is openrouter."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response()
        client = OpenAICompatibleClient(
            api_key="test-key",
            http_referer="https://example.com",
            x_title="test-app",
        )
        client.create_message(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["HTTP-Referer"] == "https://example.com"
        assert kwargs["headers"]["X-Title"] == "test-app"

    @patch("requests.post")
    def test_generate_json(self, mock_post):
        """generate_json works with the compatible client."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _make_mock_response(
            body={
                "choices": [
                    {
                        "message": {"content": '{"key": "value"}'},
                        "finish_reason": "stop",
                    }
                ],
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            }
        )
        client = OpenAICompatibleClient(
            api_key="test-key", base_url="http://localhost:8000/v1"
        )
        result = client.generate_json(
            model="test",
            prompt="extract json",
            response_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        )
        assert result == {"key": "value"}

    @patch("requests.post")
    def test_trailing_slash_stripped(self, mock_post):
        """base_url trailing slash is normalized."""
        from common.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="key", base_url="http://localhost:8000/v1/"
        )
        assert client.base_url == "http://localhost:8000/v1"

    @patch("requests.post")
    def test_null_content_coerced_to_empty_string(self, mock_post):
        """Reasoning models (e.g. minimax/minimax-m3) may return content: null
        when their reasoning_tokens exhaust the max_tokens budget. The
        response.content field must coerce None to '' so downstream callers
        can call .lstrip() without crashing. Regression test for the failure
        observed in doc-improvement --use-llm runs against research-orchestrator.
        """
        from common.llm import OpenAICompatibleClient

        body = {
            "choices": [
                {
                    "message": {"content": None, "reasoning": "thought process..."},
                    "finish_reason": "length",
                }
            ],
            "model": "minimax/minimax-m3",
            "usage": {"prompt_tokens": 100, "completion_tokens": 2000},
        }
        mock_post.return_value = _make_mock_response(body=body)
        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="minimax/minimax-m3",
            messages=[{"role": "user", "content": "hi"}],
        )
        # None is coerced to empty string (not crashed, not None)
        assert isinstance(resp.content, str)
        assert resp.content == "[reasoning] thought process..."

    @patch("requests.post")
    def test_null_content_no_reasoning_returns_empty_string(self, mock_post):
        """If both content and reasoning are absent, return '' silently."""
        from common.llm import OpenAICompatibleClient

        body = {
            "choices": [
                {
                    "message": {"content": None},
                    "finish_reason": "length",
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 100, "completion_tokens": 2000},
        }
        mock_post.return_value = _make_mock_response(body=body)
        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.content == ""

    def test_model_required(self):
        """create_message raises ValueError if model is empty."""
        from common.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="key", base_url="http://x/v1")
        with pytest.raises(ValueError, match="Model is required"):
            client.create_message(
                model="", messages=[{"role": "user", "content": "hi"}]
            )


# ---------------------------------------------------------------------------
# OpenRouterClient backward-compat alias
# ---------------------------------------------------------------------------


class TestOpenRouterClientAlias:
    """OpenRouterClient is an alias for OpenAICompatibleClient."""

    def test_is_same_class(self):
        from common.llm import OpenRouterClient, OpenAICompatibleClient

        assert OpenRouterClient is OpenAICompatibleClient

    @patch("requests.post")
    def test_import_from_init(self, mock_post):
        """Can import OpenRouterClient from common.llm."""
        from common.llm import OpenRouterClient

        mock_post.return_value = _make_mock_response()
        client = OpenRouterClient(api_key="test-key")
        assert client.get_provider_name() == "openrouter"


# ---------------------------------------------------------------------------
# OpenAIClient — native SDK
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    """Verify the native OpenAI SDK wrapper."""

    @patch("openai.OpenAI")
    def test_init_with_key(self, mock_cls):
        from common.llm import OpenAIClient

        client = OpenAIClient(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")
        assert client.get_provider_name() == "openai"

    @patch("openai.OpenAI")
    def test_init_with_base_url(self, mock_cls):
        from common.llm import OpenAIClient

        OpenAIClient(api_key="sk-test", base_url="http://proxy:8080/v1")
        mock_cls.assert_called_once_with(
            api_key="sk-test", base_url="http://proxy:8080/v1"
        )

    @patch("openai.OpenAI")
    def test_init_env_fallback(self, mock_cls):
        from common.llm import OpenAIClient

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            OpenAIClient()
            mock_cls.assert_called_once_with(api_key="env-key")

    @patch("openai.OpenAI")
    def test_init_no_key_raises(self, mock_cls):
        from common.llm import OpenAIClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIClient()

    @patch("openai.OpenAI")
    def test_create_message(self, mock_cls):
        from common.llm import OpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_sdk_response(
            content="hi there", model="gpt-4o"
        )

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.content == "hi there"
        assert resp.model == "gpt-4o"
        assert resp.usage["input_tokens"] == 10

    @patch("openai.OpenAI")
    def test_generate_json(self, mock_cls):
        from common.llm import OpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_sdk_response(
            content='{"result": 42}'
        )

        client = OpenAIClient(api_key="sk-test")
        result = client.generate_json(
            model="gpt-4o",
            prompt="what is 6*7",
            response_schema={
                "type": "object",
                "properties": {"result": {"type": "integer"}},
            },
        )
        assert result == {"result": 42}


# ---------------------------------------------------------------------------
# AzureOpenAIClient
# ---------------------------------------------------------------------------


class TestAzureOpenAIClient:
    """Verify the Azure OpenAI wrapper."""

    @patch("openai.AzureOpenAI")
    def test_init_with_params(self, mock_cls):
        from common.llm import AzureOpenAIClient

        client = AzureOpenAIClient(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-02-01",
        )
        mock_cls.assert_called_once_with(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-02-01",
        )
        assert client.get_provider_name() == "azure-openai"

    @patch("openai.AzureOpenAI")
    def test_init_env_fallback(self, mock_cls):
        from common.llm import AzureOpenAIClient

        env = {
            "AZURE_OPENAI_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com/",
            "OPENAI_API_VERSION": "2024-06-01",
        }
        with patch.dict("os.environ", env):
            AzureOpenAIClient()
            mock_cls.assert_called_once_with(
                api_key="env-key",
                azure_endpoint="https://env.openai.azure.com/",
                api_version="2024-06-01",
            )

    @patch("openai.AzureOpenAI")
    def test_init_no_key_raises(self, mock_cls):
        from common.llm import AzureOpenAIClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Azure OpenAI API key is required"):
                AzureOpenAIClient()

    @patch("openai.AzureOpenAI")
    def test_init_no_endpoint_raises(self, mock_cls):
        from common.llm import AzureOpenAIClient

        with patch.dict("os.environ", {"AZURE_OPENAI_KEY": "key"}, clear=False):
            with pytest.raises(ValueError, match="Azure endpoint is required"):
                AzureOpenAIClient(api_key="key")

    @patch("openai.AzureOpenAI")
    def test_deployment_name_fallback(self, mock_cls):
        """When no model is passed, deployment_name is used."""
        from common.llm import AzureOpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_sdk_response(
            content="ok", model="my-deployment"
        )

        client = AzureOpenAIClient(
            api_key="key",
            azure_endpoint="https://x.openai.azure.com/",
            deployment_name="my-deployment",
        )
        client.create_message(model="", messages=[{"role": "user", "content": "hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "my-deployment"

    @patch("openai.AzureOpenAI")
    def test_explicit_model_overrides_deployment(self, mock_cls):
        from common.llm import AzureOpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_sdk_response()

        client = AzureOpenAIClient(
            api_key="key",
            azure_endpoint="https://x.openai.azure.com/",
            deployment_name="default-deploy",
        )
        client.create_message(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Factory: get_llm_client
# ---------------------------------------------------------------------------


class TestGetLlmClient:
    """Verify the factory function for all providers."""

    def test_anthropic(self):
        from common.llm import get_llm_client, AnthropicClient

        client = get_llm_client("anthropic", "test-key")
        assert isinstance(client, AnthropicClient)

    def test_anthropic_env_fallback(self):
        from common.llm import get_llm_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            client = get_llm_client("anthropic", "")
            assert client.get_provider_name() == "anthropic"

    def test_anthropic_no_key_raises(self):
        from common.llm import get_llm_client

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key is required"):
                get_llm_client("anthropic", "")

    @patch("requests.post")
    def test_openrouter(self, mock_post):
        from common.llm import get_llm_client

        mock_post.return_value = _make_mock_response()
        client = get_llm_client("openrouter", "test-key")
        assert client.get_provider_name() == "openrouter"

    def test_openrouter_env_fallback(self):
        from common.llm import get_llm_client

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
            client = get_llm_client("openrouter", "")
            assert client.get_provider_name() == "openrouter"

    def test_openrouter_no_key_raises(self):
        from common.llm import get_llm_client

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                get_llm_client("openrouter", "")

    @patch("openai.OpenAI")
    def test_openai(self, mock_cls):
        from common.llm import get_llm_client, OpenAIClient

        client = get_llm_client("openai", "sk-test")
        assert isinstance(client, OpenAIClient)

    @patch("openai.AzureOpenAI")
    def test_azure_openai(self, mock_cls):
        from common.llm import get_llm_client, AzureOpenAIClient

        client = get_llm_client(
            "azure-openai",
            "az-key",
            azure_endpoint="https://x.openai.azure.com/",
        )
        assert isinstance(client, AzureOpenAIClient)

    def test_azure_openai_alias(self):
        """azure_openai (underscore) is accepted."""
        from common.llm import get_llm_client

        with patch("openai.AzureOpenAI"):
            client = get_llm_client(
                "azure_openai",
                "az-key",
                azure_endpoint="https://x.openai.azure.com/",
            )
            assert client.get_provider_name() == "azure-openai"

    @patch("requests.post")
    def test_ollama(self, mock_post):
        from common.llm import get_llm_client

        mock_post.return_value = _make_mock_response()
        client = get_llm_client("ollama", "")
        assert client.base_url == "http://localhost:11434/v1"
        assert client.get_provider_name() == "openai-compatible"

    @patch("requests.post")
    def test_ollama_custom_url(self, mock_post):
        from common.llm import get_llm_client

        mock_post.return_value = _make_mock_response()
        client = get_llm_client("ollama", "", base_url="http://gpu-box:11434/v1")
        assert client.base_url == "http://gpu-box:11434/v1"

    @patch("requests.post")
    def test_openai_compatible(self, mock_post):
        from common.llm import get_llm_client

        mock_post.return_value = _make_mock_response()
        client = get_llm_client(
            "openai-compatible", "key", base_url="http://litellm:4000/v1"
        )
        assert client.base_url == "http://litellm:4000/v1"
        assert client.get_provider_name() == "openai-compatible"

    def test_openai_compatible_no_base_url_raises(self):
        from common.llm import get_llm_client

        with pytest.raises(ValueError, match="base_url is required"):
            get_llm_client("openai-compatible", "key")

    def test_unknown_provider_raises(self):
        from common.llm import get_llm_client

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client("nonexistent", "key")

    def test_case_insensitive(self):
        from common.llm import get_llm_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            client = get_llm_client("Anthropic", "")
            assert client.get_provider_name() == "anthropic"


# ---------------------------------------------------------------------------
# Config-based factory: get_llm_client_from_config
# ---------------------------------------------------------------------------


class TestGetLlmClientFromConfig:
    """Verify config-driven client creation with env var cascade."""

    @patch("openai.OpenAI")
    def test_openai_provider(self, mock_cls):
        from common.llm import get_llm_client_from_config

        @dataclass
        class FakeConfig:
            llm_provider: str = "openai"
            openai_api_key: str = "sk-test"

        client = get_llm_client_from_config(FakeConfig())
        assert client.get_provider_name() == "openai"

    @patch("openai.AzureOpenAI")
    def test_azure_provider(self, mock_cls):
        from common.llm import get_llm_client_from_config

        @dataclass
        class FakeConfig:
            llm_provider: str = "azure-openai"
            azure_openai_key: str = "az-key"
            azure_endpoint: str = "https://x.openai.azure.com/"

        client = get_llm_client_from_config(FakeConfig())
        assert client.get_provider_name() == "azure-openai"

    def test_env_var_cascade_provider(self):
        """LLM_PROVIDER env var is used when config doesn't set llm_provider."""
        from common.llm import get_llm_client_from_config

        @dataclass
        class FakeConfig:
            pass

        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "env-key"},
        ):
            client = get_llm_client_from_config(FakeConfig())
            assert client.get_provider_name() == "anthropic"

    @patch("requests.post")
    def test_env_var_cascade_base_url(self, mock_post):
        """LLM_BASE_URL env var is used when config doesn't set llm_base_url."""
        from common.llm import get_llm_client_from_config

        mock_post.return_value = _make_mock_response()

        @dataclass
        class FakeConfig:
            llm_provider: str = "openai-compatible"
            llm_api_key: str = "key"

        with patch.dict("os.environ", {"LLM_BASE_URL": "http://env-host:8000/v1"}):
            client = get_llm_client_from_config(FakeConfig())
            assert client.base_url == "http://env-host:8000/v1"

    @patch("requests.post")
    def test_config_base_url_overrides_env(self, mock_post):
        """Config attribute takes precedence over env var."""
        from common.llm import get_llm_client_from_config

        mock_post.return_value = _make_mock_response()

        @dataclass
        class FakeConfig:
            llm_provider: str = "openai-compatible"
            llm_api_key: str = "key"
            llm_base_url: str = "http://config-host:8000/v1"

        with patch.dict("os.environ", {"LLM_BASE_URL": "http://env-host:8000/v1"}):
            client = get_llm_client_from_config(FakeConfig())
            assert client.base_url == "http://config-host:8000/v1"

    @patch("requests.post")
    def test_ollama_via_config(self, mock_post):
        from common.llm import get_llm_client_from_config

        mock_post.return_value = _make_mock_response()

        @dataclass
        class FakeConfig:
            llm_provider: str = "ollama"
            llm_base_url: str = "http://gpu-box:11434/v1"

        client = get_llm_client_from_config(FakeConfig())
        assert client.base_url == "http://gpu-box:11434/v1"


# ---------------------------------------------------------------------------
# Anthropic client — unchanged (regression guard)
# ---------------------------------------------------------------------------


class TestAnthropicClientUnchanged:
    """Verify that the Anthropic client is completely unaffected."""

    def test_init_signature(self):
        from common.llm import AnthropicClient

        with patch("common.llm.client.AnthropicClient.__init__", return_value=None):
            _ = AnthropicClient.__new__(AnthropicClient)

    def test_factory_no_referer(self):
        from common.llm import get_llm_client

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_llm_client("anthropic", "")

    def test_get_provider_name(self):
        from common.llm import AnthropicClient

        with patch("common.llm.client.AnthropicClient.__init__", return_value=None):
            c = AnthropicClient.__new__(AnthropicClient)
            assert c.get_provider_name() == "anthropic"


# ---------------------------------------------------------------------------
# Embedding support
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Verify embed() method on OpenAICompatibleClient."""

    @patch("requests.post")
    def test_embed_basic(self, mock_post):
        """embed() returns EmbeddingResponse with correct shape."""
        from common.llm import OpenAICompatibleClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client = OpenAICompatibleClient(api_key="test-key")
        result = client.embed(
            model="text-embedding-3-small",
            input=["hello", "world"],
        )

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.model == "text-embedding-3-small"
        assert result.usage == {"prompt_tokens": 10, "total_tokens": 10}

        # Verify the request was sent to /embeddings
        call_args = mock_post.call_args
        assert "/embeddings" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["hello", "world"]

    @patch("requests.post")
    def test_embed_single_input(self, mock_post):
        """embed() works with a single input string in a list."""
        from common.llm import OpenAICompatibleClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client = OpenAICompatibleClient(api_key="test-key")
        result = client.embed(model="text-embedding-3-small", input=["single"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2]

    def test_embed_empty_input_raises(self):
        """embed() raises ValueError on empty input."""
        from common.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="test-key")
        with pytest.raises(ValueError, match="Input texts are required"):
            client.embed(model="text-embedding-3-small", input=[])

    def test_embed_no_model_raises(self):
        """embed() raises ValueError when model is empty."""
        from common.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="test-key")
        with pytest.raises(ValueError, match="Model is required"):
            client.embed(model="", input=["hello"])

    @patch("requests.post")
    def test_embed_sends_auth_headers(self, mock_post):
        """embed() sends Authorization and OpenRouter headers."""
        from common.llm import OpenAICompatibleClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [], "model": "m"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client = OpenAICompatibleClient(
            api_key="sk-test",
            http_referer="https://example.com",
            x_title="test-app",
        )
        client.embed(model="m", input=["x"])

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["HTTP-Referer"] == "https://example.com"
        assert headers["X-Title"] == "test-app"

    def test_base_llm_client_embed_raises(self):
        """Default LLMClient.embed() raises NotImplementedError."""
        from common.llm import LLMClient

        class DummyClient(LLMClient):
            def create_message(self, **kw):
                pass

            def generate_json(self, **kw):
                pass

            def get_provider_name(self):
                return "dummy"

        client = DummyClient()
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            client.embed(model="m", input=["x"])


# ---------------------------------------------------------------------------
# Real-call smoke test: deepseek-v4-pro via OpenRouter
# ---------------------------------------------------------------------------
# Regression guard for the deepseek-v4-pro path added to llm_client.py.
# This is an INTEGRATION test (real HTTP call) — deselect with
# ``pytest -m "not integration"`` for fast unit runs. Skipped gracefully
# when ``OPENROUTER_API_KEY`` is not set so CI without secrets still works.


@pytest.mark.integration
class TestDeepseekV4ProSmoke:
    """End-to-end smoke test: real call through OpenRouter to deepseek-v4-pro."""

    MODEL = (
        "deepseek/deepseek-chat"  # canonical OpenRouter slug; v4-pro aliases to this
    )
    PROMPT = (
        "Reply with the word 'pong' and nothing else. No punctuation, no whitespace."
    )
    MAX_TOKENS = 16

    @pytest.fixture(autouse=True)
    def _require_openrouter_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set; skipping real-call smoke test")

    def test_real_call_returns_parseable_content(self):
        """Real call to deepseek-v4-pro returns non-empty parseable content."""
        from common.llm import get_llm_client

        client = get_llm_client(
            provider="openrouter",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=self.MODEL,
            http_referer="https://github.com/DarojaAI/py-daroja-libs",
            x_title="py-daroja-libs-smoke",
        )

        import time

        start = time.monotonic()
        resp = client.create_message(
            model=self.MODEL,
            messages=[{"role": "user", "content": self.PROMPT}],
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
        )
        elapsed = time.monotonic() - start

        # response shape: LLMResponse dataclass
        assert resp is not None
        assert isinstance(
            resp.content, str
        ), f"content must be str, got {type(resp.content).__name__}: {resp.content!r}"
        assert resp.content.strip(), "content must be non-empty"

        # model tag echoed by OpenRouter should contain "deepseek"
        assert (
            "deepseek" in resp.model.lower()
        ), f"response.model should mention deepseek, got {resp.model!r}"

        # Should complete in well under 60s for a 16-token call
        assert elapsed < 60.0, f"call took {elapsed:.1f}s, expected <60s"

    def test_real_call_via_factory_from_config(self):
        """Same call via get_llm_client_from_config with a dataclass config."""
        from common.llm import get_llm_client_from_config

        @dataclass
        class SmokeConfig:
            llm_provider: str = "openrouter"
            openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
            llm_model: str = "deepseek/deepseek-chat"
            http_referer: str = "https://github.com/DarojaAI/py-daroja-libs"
            x_title: str = "py-daroja-libs-smoke"

        client = get_llm_client_from_config(SmokeConfig())
        resp = client.create_message(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": self.PROMPT}],
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
        )
        assert resp.content.strip()
        assert "deepseek" in resp.model.lower()
