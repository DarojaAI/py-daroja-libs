"""Tests for common.llm.reasoning — reasoning output extraction."""

from common.llm.reasoning import (
    extract_response_from_reasoning,
    extract_json_from_reasoning,
)


class TestExtractResponseFromReasoning:
    def test_already_clean_json(self):
        assert extract_response_from_reasoning('{"score": 42}') == '{"score": 42}'

    def test_think_tags_stripped(self):
        raw = '<think>Let me analyze...</think>\n{"score": 42}'
        assert extract_response_from_reasoning(raw) == '{"score": 42}'

    def test_multiple_think_tags(self):
        raw = (
            '<think>First thought</think>\n<think>Second thought</think>\n{"score": 99}'
        )
        result = extract_response_from_reasoning(raw)
        assert '"score": 99' in result
        assert "<think>" not in result

    def test_empty_string(self):
        assert extract_response_from_reasoning("") == ""

    def test_none_input(self):
        assert extract_response_from_reasoning(None) is None

    def test_plain_text_no_json(self):
        text = "Just some plain text without any JSON."
        assert extract_response_from_reasoning(text) == text

    def test_json_after_marker(self):
        raw = 'Here is the JSON response:\n{"score": 7, "analysis": "good"}'
        result = extract_response_from_reasoning(raw)
        assert '"score"' in result
        assert result.startswith("{")

    def test_json_in_code_block(self):
        raw = '```json\n{"score": 5}\n```'
        result = extract_response_from_reasoning(raw)
        assert '"score"' in result

    def test_deepseek_r1_format(self):
        raw = '<think>\nI need to evaluate this resume.\nThe score should reflect...\n</think>\n\n{"score": 85, "analysis": "strong candidate"}'
        result = extract_response_from_reasoning(raw)
        assert result == '{"score": 85, "analysis": "strong candidate"}'


class TestExtractJsonFromReasoning:
    def test_valid_json(self):
        raw = '<think>thinking...</think>{"score": 42}'
        result = extract_json_from_reasoning(raw)
        assert result == {"score": 42}

    def test_no_json(self):
        assert extract_json_from_reasoning("just text") is None

    def test_empty(self):
        assert extract_json_from_reasoning("") is None

    def test_list_json(self):
        raw = "<think>analysis</think>[1, 2, 3]"
        # extract_response_from_reasoning returns the list as-is since it starts with [
        # but extract_json_from_reasoning will try json.loads on it
        result = extract_json_from_reasoning(raw)
        assert result == [1, 2, 3]
