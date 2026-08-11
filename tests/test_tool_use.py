"""
Tests for common.llm.tool_use — tool-use (function-calling) support.

Covers:
1. Anthropic tool-use request round-trip (mocked SDK)
2. OpenRouter/OpenAI-style tool-use request round-trip (mocked HTTP)
3. Backward compatibility: no tools → tool_calls is None
4. Mixed text + tool_use response → both fields populated
5. ToolCall dataclass construction
6. Invalid JSON in OpenAI tool_calls.arguments → fallback to {}
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use_response():
    """Build a mock Anthropic Messages API response with tool_use blocks."""
    # Simulate: text block + tool_use block
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Let me check the weather."

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_01A09q90qw90lq917835lq9"
    tool_block.name = "get_weather"
    tool_block.input = {"city": "San Francisco"}

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50

    response = MagicMock()
    response.content = [text_block, tool_block]
    response.model = "claude-3-5-sonnet-20241022"
    response.stop_reason = "tool_use"
    response.usage = usage
    return response


def _make_text_only_response():
    """Build a mock Anthropic Messages API response with text only."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "The weather is sunny."

    usage = MagicMock()
    usage.input_tokens = 80
    usage.output_tokens = 30

    response = MagicMock()
    response.content = [text_block]
    response.model = "claude-3-5-sonnet-20241022"
    response.stop_reason = "end_turn"
    response.usage = usage
    return response


def _make_openai_tool_calls_response():
    """Build a mock OpenAI-compatible /chat/completions response with tool_calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "San Francisco"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _make_openai_mixed_response():
    """Build a mock OpenAI response with both content and tool_calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": '{"query": "weather today"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _mock_requests_post(response_body):
    """Return a mock for requests.post that yields the given JSON body."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = response_body
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Anthropic tool-use
# ---------------------------------------------------------------------------


class TestAnthropicToolUse:
    """Verify Anthropic tool-use block normalization."""

    @patch("anthropic.Anthropic")
    def test_tool_use_round_trip(self, mock_anthropic_cls):
        """Anthropic tool_use blocks are normalized into ToolCall objects."""
        from common.llm import AnthropicClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_tool_use_response()

        client = AnthropicClient(api_key="test-key")
        resp = client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

        # Verify text content extracted
        assert "Let me check the weather." in resp.content
        # Verify tool_calls normalized
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "toolu_01A09q90qw90lq917835lq9"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "San Francisco"}
        # Verify stop_reason
        assert resp.stop_reason == "tool_use"

    @patch("anthropic.Anthropic")
    def test_text_only_no_tool_calls(self, mock_anthropic_cls):
        """Text-only response has tool_calls=None."""
        from common.llm import AnthropicClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_text_only_response()

        client = AnthropicClient(api_key="test-key")
        resp = client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert resp.content == "The weather is sunny."
        assert resp.tool_calls is None
        assert resp.stop_reason == "end_turn"

    @patch("anthropic.Anthropic")
    def test_multiple_tool_calls(self, mock_anthropic_cls):
        """Multiple tool_use blocks in one response are all captured."""
        from common.llm import AnthropicClient

        block1 = MagicMock()
        block1.type = "tool_use"
        block1.id = "toolu_001"
        block1.name = "get_weather"
        block1.input = {"city": "NYC"}

        block2 = MagicMock()
        block2.type = "tool_use"
        block2.id = "toolu_002"
        block2.name = "get_time"
        block2.input = {"timezone": "EST"}

        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50

        response = MagicMock()
        response.content = [block1, block2]
        response.model = "claude-3-5-sonnet-20241022"
        response.stop_reason = "tool_use"
        response.usage = usage

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = response

        client = AnthropicClient(api_key="test-key")
        resp = client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "What's the weather and time in NYC?"}
            ],
            tools=[],
        )

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[1].name == "get_time"


# ---------------------------------------------------------------------------
# OpenRouter/OpenAI-style tool-use
# ---------------------------------------------------------------------------


class TestOpenRouterToolUse:
    """Verify OpenAI-style tool_calls normalization."""

    @patch("requests.post")
    def test_tool_calls_round_trip(self, mock_post):
        """OpenAI-style tool_calls are normalized into ToolCall objects."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _mock_requests_post(_make_openai_tool_calls_response())

        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        # Verify content
        assert resp.content == ""
        # Verify tool_calls normalized
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "San Francisco"}
        # Verify stop_reason
        assert resp.stop_reason == "tool_calls"

    @patch("requests.post")
    def test_mixed_text_and_tool_calls(self, mock_post):
        """Response with both content and tool_calls populates both fields."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _mock_requests_post(_make_openai_mixed_response())

        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for today's weather"}],
        )

        assert resp.content == "Let me check that for you."
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search_web"
        assert resp.tool_calls[0].arguments == {"query": "weather today"}

    @patch("requests.post")
    def test_no_tool_calls_backward_compat(self, mock_post):
        """Standard text response has tool_calls=None (backward compat)."""
        from common.llm import OpenAICompatibleClient

        body = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        mock_post.return_value = _mock_requests_post(body)

        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert resp.content == "hello"
        assert resp.tool_calls is None

    @patch("requests.post")
    def test_invalid_json_in_arguments(self, mock_post):
        """Malformed JSON in function.arguments falls back to {}."""
        from common.llm import OpenAICompatibleClient

        body = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "broken_tool",
                                    "arguments": "not valid json {{{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        mock_post.return_value = _mock_requests_post(body)

        client = OpenAICompatibleClient(api_key="test-key")
        resp = client.create_message(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert resp.tool_calls is not None
        assert resp.tool_calls[0].arguments == {}

    @patch("requests.post")
    def test_tool_choice_forwarded(self, mock_post):
        """tool_choice kwarg is forwarded in the request payload."""
        from common.llm import OpenAICompatibleClient

        mock_post.return_value = _mock_requests_post(_make_openai_tool_calls_response())

        client = OpenAICompatibleClient(api_key="test-key")
        client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "auto"},
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["tool_choice"] == {"type": "auto"}


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------


class TestToolCallDataclass:
    """Verify ToolCall construction and fields."""

    def test_basic_construction(self):
        from common.llm import ToolCall

        tc = ToolCall(id="tc_1", name="my_tool", arguments={"key": "value"})
        assert tc.id == "tc_1"
        assert tc.name == "my_tool"
        assert tc.arguments == {"key": "value"}

    def test_empty_arguments(self):
        from common.llm import ToolCall

        tc = ToolCall(id="tc_2", name="no_args_tool", arguments={})
        assert tc.arguments == {}


# ---------------------------------------------------------------------------
# LLMResponse backward compat
# ---------------------------------------------------------------------------


class TestLLMResponseBackwardCompat:
    """Verify LLMResponse is backward-compatible with existing callers."""

    def test_tool_calls_defaults_to_none(self):
        """Existing code that creates LLMResponse without tool_calls still works."""
        from common.llm import LLMResponse

        resp = LLMResponse(content="hi", model="test")
        assert resp.tool_calls is None
        assert resp.content == "hi"
        assert resp.model == "test"

    def test_explicit_tool_calls(self):
        from common.llm import LLMResponse, ToolCall

        tc = ToolCall(id="tc_1", name="tool", arguments={"a": 1})
        resp = LLMResponse(
            content="",
            model="test",
            stop_reason="tool_use",
            tool_calls=[tc],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "tool"


# ---------------------------------------------------------------------------
# OpenAI SDK tool-use (native client)
# ---------------------------------------------------------------------------


def _make_openai_sdk_tool_calls_response():
    """Build a mock openai SDK response with tool_calls (SDK object shape)."""
    # The openai SDK returns ChatCompletionMessageToolCall objects,
    # not raw dicts. We simulate them with MagicMock.
    function = MagicMock()
    function.name = "get_weather"
    function.arguments = '{"city": "San Francisco"}'

    sdk_tool_call = MagicMock()
    sdk_tool_call.id = "call_sdk_001"
    sdk_tool_call.function = function

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [sdk_tool_call]
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50

    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-4o"
    response.usage = usage
    return response


def _make_openai_sdk_mixed_response():
    """Build a mock openai SDK response with content + tool_calls."""
    function = MagicMock()
    function.name = "search_web"
    function.arguments = '{"query": "weather today"}'

    sdk_tool_call = MagicMock()
    sdk_tool_call.id = "call_sdk_002"
    sdk_tool_call.function = function

    choice = MagicMock()
    choice.message.content = "Let me check that for you."
    choice.message.tool_calls = [sdk_tool_call]
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.usage = 50

    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-4o"
    response.usage = usage
    return response


def _make_openai_sdk_text_only_response():
    """Build a mock openai SDK response with text only (no tool_calls)."""
    choice = MagicMock()
    choice.message.content = "The weather is sunny."
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 80
    usage.completion_tokens = 30

    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-4o"
    response.usage = usage
    return response


class TestOpenAISDKToolUse:
    """Verify OpenAIClient tool_use normalization from SDK objects."""

    @patch("openai.OpenAI")
    def test_tool_calls_round_trip(self, mock_cls):
        from common.llm import OpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_tool_calls_response()
        )

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert resp.content == ""
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_sdk_001"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "San Francisco"}
        assert resp.stop_reason == "tool_calls"

    @patch("openai.OpenAI")
    def test_mixed_text_and_tool_calls(self, mock_cls):
        from common.llm import OpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_mixed_response()
        )

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for weather"}],
        )

        assert resp.content == "Let me check that for you."
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search_web"
        assert resp.tool_calls[0].arguments == {"query": "weather today"}

    @patch("openai.OpenAI")
    def test_text_only_no_tool_calls(self, mock_cls):
        from common.llm import OpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_text_only_response()
        )

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert resp.content == "The weather is sunny."
        assert resp.tool_calls is None
        assert resp.stop_reason == "stop"

    @patch("openai.OpenAI")
    def test_invalid_json_in_arguments(self, mock_cls):
        from common.llm import OpenAIClient

        function = MagicMock()
        function.name = "broken_tool"
        function.arguments = "not valid json {{{"

        sdk_tc = MagicMock()
        sdk_tc.id = "call_bad"
        sdk_tc.function = function

        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = [sdk_tc]
        choice.finish_reason = "tool_calls"

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 10

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4o"
        response.usage = usage

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = response

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert resp.tool_calls is not None
        assert resp.tool_calls[0].arguments == {}

    @patch("openai.OpenAI")
    def test_multiple_tool_calls(self, mock_cls):
        from common.llm import OpenAIClient

        fn1 = MagicMock()
        fn1.name = "get_weather"
        fn1.arguments = '{"city": "NYC"}'
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function = fn1

        fn2 = MagicMock()
        fn2.name = "get_time"
        fn2.arguments = '{"timezone": "EST"}'
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function = fn2

        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = [tc1, tc2]
        choice.finish_reason = "tool_calls"

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4o"
        response.usage = usage

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = response

        client = OpenAIClient(api_key="sk-test")
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "weather and time in NYC"}],
        )

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[1].name == "get_time"


# ---------------------------------------------------------------------------
# Azure OpenAI SDK tool-use
# ---------------------------------------------------------------------------


class TestAzureOpenAISDKToolUse:
    """Verify AzureOpenAIClient tool_use normalization from SDK objects."""

    @patch("openai.AzureOpenAI")
    def test_tool_calls_round_trip(self, mock_cls):
        from common.llm import AzureOpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_tool_calls_response()
        )

        client = AzureOpenAIClient(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com/",
        )
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_sdk_001"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "San Francisco"}
        assert resp.stop_reason == "tool_calls"

    @patch("openai.AzureOpenAI")
    def test_text_only_no_tool_calls(self, mock_cls):
        from common.llm import AzureOpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_text_only_response()
        )

        client = AzureOpenAIClient(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com/",
        )
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert resp.content == "The weather is sunny."
        assert resp.tool_calls is None

    @patch("openai.AzureOpenAI")
    def test_mixed_text_and_tool_calls(self, mock_cls):
        from common.llm import AzureOpenAIClient

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_sdk_mixed_response()
        )

        client = AzureOpenAIClient(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com/",
        )
        resp = client.create_message(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for weather"}],
        )

        assert resp.content == "Let me check that for you."
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].name == "search_web"
        assert resp.tool_calls[0].arguments == {"query": "weather today"}


# ---------------------------------------------------------------------------
# build_tool_result_messages
# ---------------------------------------------------------------------------


class TestBuildToolResultMessages:
    """Verify provider-specific message formatting for tool results."""

    def test_anthropic_single_result(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="toolu_001", content='{"temp": 72}')]
        msgs = build_tool_result_messages(results, provider="anthropic")

        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], list)
        assert len(msgs[0]["content"]) == 1
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_001"
        assert block["content"] == '{"temp": 72}'

    def test_anthropic_multiple_results(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [
            ToolResult(tool_call_id="toolu_001", content="sunny"),
            ToolResult(tool_call_id="toolu_002", content="65F"),
        ]
        msgs = build_tool_result_messages(results, provider="anthropic")

        assert len(msgs) == 1  # Single message with multiple blocks
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["tool_use_id"] == "toolu_001"
        assert content[1]["tool_use_id"] == "toolu_002"

    def test_openai_single_result(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="call_abc", content='{"temp": 72}')]
        msgs = build_tool_result_messages(results, provider="openai")

        assert len(msgs) == 1
        assert msgs[0] == {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": '{"temp": 72}',
        }

    def test_openai_multiple_results(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [
            ToolResult(tool_call_id="call_1", content="sunny"),
            ToolResult(tool_call_id="call_2", content="65F"),
        ]
        msgs = build_tool_result_messages(results, provider="openai")

        assert len(msgs) == 2
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "call_2"

    def test_openrouter_same_as_openai(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="call_x", content="ok")]
        msgs_openai = build_tool_result_messages(results, provider="openai")
        msgs_openrouter = build_tool_result_messages(results, provider="openrouter")
        assert msgs_openai == msgs_openrouter

    def test_azure_same_as_openai(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="call_y", content="ok")]
        msgs_openai = build_tool_result_messages(results, provider="openai")
        msgs_azure = build_tool_result_messages(results, provider="azure")
        assert msgs_openai == msgs_azure

    def test_azure_openai_alias(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="call_z", content="ok")]
        msgs1 = build_tool_result_messages(results, provider="azure-openai")
        msgs2 = build_tool_result_messages(results, provider="azure_openai")
        assert msgs1 == msgs2

    def test_unknown_provider_raises(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="x", content="y")]
        with pytest.raises(ValueError, match="Unknown provider"):
            build_tool_result_messages(results, provider="bedrock")

    def test_case_insensitive_provider(self):
        from common.llm import ToolResult, build_tool_result_messages

        results = [ToolResult(tool_call_id="c", content="d")]
        msgs = build_tool_result_messages(results, provider="Anthropic")
        assert msgs[0]["role"] == "user"

    def test_empty_list(self):
        from common.llm import build_tool_result_messages

        msgs = build_tool_result_messages([], provider="openai")
        assert msgs == []


class TestToolResultDataclass:
    """Verify ToolResult construction."""

    def test_basic(self):
        from common.llm import ToolResult

        tr = ToolResult(tool_call_id="tc_1", content="result data")
        assert tr.tool_call_id == "tc_1"
        assert tr.content == "result data"

    def test_empty_content(self):
        from common.llm import ToolResult

        tr = ToolResult(tool_call_id="tc_2", content="")
        assert tr.content == ""


# ---------------------------------------------------------------------------
# Round-trip: tool_calls → tool_results → follow-up messages
# ---------------------------------------------------------------------------


class TestToolUseRoundTrip:
    """End-to-end round-trip: model calls tool → we build results → send back."""

    @patch("openai.OpenAI")
    def test_openai_round_trip(self, mock_cls):
        """Simulate: model requests tool → we send result → model responds."""
        from common.llm import OpenAIClient, ToolResult, build_tool_result_messages

        # Turn 1: model requests a tool call
        fn = MagicMock()
        fn.name = "get_weather"
        fn.arguments = '{"city": "SF"}'
        tc = MagicMock()
        tc.id = "call_001"
        tc.function = fn

        choice1 = MagicMock()
        choice1.message.content = ""
        choice1.message.tool_calls = [tc]
        choice1.finish_reason = "tool_calls"

        usage1 = MagicMock()
        usage1.prompt_tokens = 100
        usage1.completion_tokens = 20

        resp1_sdk = MagicMock()
        resp1_sdk.choices = [choice1]
        resp1_sdk.model = "gpt-4o"
        resp1_sdk.usage = usage1

        # Turn 2: model responds after receiving tool result
        choice2 = MagicMock()
        choice2.message.content = "The weather in SF is 72F and sunny."
        choice2.message.tool_calls = None
        choice2.finish_reason = "stop"

        usage2 = MagicMock()
        usage2.prompt_tokens = 120
        usage2.completion_tokens = 30

        resp2_sdk = MagicMock()
        resp2_sdk.choices = [choice2]
        resp2_sdk.model = "gpt-4o"
        resp2_sdk.usage = usage2

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [resp1_sdk, resp2_sdk]

        client = OpenAIClient(api_key="sk-test")
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Turn 1
        resp1 = client.create_message(model="gpt-4o", messages=messages)
        assert resp1.tool_calls is not None
        assert resp1.tool_calls[0].name == "get_weather"

        # Build tool results
        results = [
            ToolResult(tool_call_id=resp1.tool_calls[0].id, content='{"temp": 72}')
        ]
        result_msgs = build_tool_result_messages(results, provider="openai")

        # Append assistant message + tool results to conversation
        messages.append(
            {
                "role": "assistant",
                "content": resp1.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": fn.name, "arguments": fn.arguments},
                    }
                ],
            }
        )
        messages.extend(result_msgs)

        # Turn 2
        resp2 = client.create_message(model="gpt-4o", messages=messages)
        assert resp2.tool_calls is None
        assert "72F" in resp2.content

    @patch("anthropic.Anthropic")
    def test_anthropic_round_trip(self, mock_cls):
        """Simulate: Anthropic model requests tool → we send result → model responds."""
        from common.llm import AnthropicClient, ToolResult, build_tool_result_messages

        # Turn 1: model requests tool
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_abc"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "SF"}

        usage1 = MagicMock()
        usage1.input_tokens = 100
        usage1.output_tokens = 20

        resp1_sdk = MagicMock()
        resp1_sdk.content = [tool_block]
        resp1_sdk.model = "claude-3-5-sonnet-20241022"
        resp1_sdk.stop_reason = "tool_use"
        resp1_sdk.usage = usage1

        # Turn 2: model responds
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The weather in SF is 72F and sunny."

        usage2 = MagicMock()
        usage2.input_tokens = 120
        usage2.output_tokens = 30

        resp2_sdk = MagicMock()
        resp2_sdk.content = [text_block]
        resp2_sdk.model = "claude-3-5-sonnet-20241022"
        resp2_sdk.stop_reason = "end_turn"
        resp2_sdk.usage = usage2

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [resp1_sdk, resp2_sdk]

        client = AnthropicClient(api_key="test-key")
        messages = [{"role": "user", "content": "What's the weather?"}]

        # Turn 1
        resp1 = client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        )
        assert resp1.tool_calls is not None
        assert resp1.tool_calls[0].name == "get_weather"

        # Build tool results
        results = [
            ToolResult(tool_call_id=resp1.tool_calls[0].id, content='{"temp": 72}')
        ]
        result_msgs = build_tool_result_messages(results, provider="anthropic")

        # Append assistant + tool results
        messages.append({"role": "assistant", "content": resp1.content})
        messages.extend(result_msgs)

        # Turn 2
        resp2 = client.create_message(
            model="claude-3-5-sonnet-20241022", messages=messages
        )
        assert resp2.tool_calls is None
        assert "72F" in resp2.content
