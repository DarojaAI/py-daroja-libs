"""Tests for common.llm.template_adapter — schema-validated LLM template filling."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock
from common.llm.template_adapter import adapt_template, AdaptationResult


@dataclass
class FakeStyleGuide:
    prose: str = "Be concise."


@dataclass
class FakePlaceholder:
    hint: str = ""


@dataclass
class FakeTemplate:
    body: str = "# Report\n\n## {{section1}}\n\n{{section2}}"
    name: str = "test-template"
    version: str = "1.0"
    placeholders: dict = None
    style_guide: FakeStyleGuide = None

    def __post_init__(self):
        if self.placeholders is None:
            self.placeholders = {
                "section1": FakePlaceholder(hint="Main section"),
                "section2": FakePlaceholder(hint="Details"),
            }
        if self.style_guide is None:
            self.style_guide = FakeStyleGuide()


class TestAdaptTemplate:
    def test_successful_adaptation(self):
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {
            "section1": "Analysis",
            "section2": "Details here",
        }

        result = adapt_template(template, client, model="test-model")

        assert isinstance(result, AdaptationResult)
        assert result.template_name == "test-template"
        assert result.template_version == "1.0"
        assert result.parsed["section1"] == "Analysis"
        assert "Analysis" in result.filled_body
        assert "Details here" in result.filled_body
        assert not result.has_blocking_errors

    def test_with_source_doc(self):
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        adapt_template(template, client, model="m", source_doc="Source text here")

        call_kwargs = client.generate_json.call_args
        assert "Source text here" in call_kwargs.kwargs["prompt"]

    def test_schema_violation_json_parse_error(self):
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.side_effect = ValueError("bad json")

        result = adapt_template(template, client, model="m")

        assert len(result.schema_violations) == 1
        assert result.has_blocking_errors

    def test_create_message_fallback(self):
        template = FakeTemplate()
        client = MagicMock()
        response = MagicMock()
        response.content = '{"section1": "A", "section2": "B"}'
        client.create_message.return_value = response

        result = adapt_template(template, client, model="m")

        assert result.parsed["section1"] == "A"
        assert not result.has_blocking_errors

    def test_no_llm_methods_raises(self):
        template = FakeTemplate()
        client = MagicMock()
        del client.generate_json
        del client.create_message

        result = adapt_template(template, client, model="m")
        assert result.has_blocking_errors

    def test_rules_fn_called(self):
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        rules_fn = MagicMock(return_value=[])
        adapt_template(template, client, model="m", run_rules=True, rules_fn=rules_fn)

        rules_fn.assert_called_once()

    def test_filled_body_replaces_placeholders(self):
        template = FakeTemplate()
        template.body = "Hello {{name}}, welcome to {{place}}."
        template.placeholders = {
            "name": FakePlaceholder(hint=""),
            "place": FakePlaceholder(hint=""),
        }
        client = MagicMock()
        client.generate_json.return_value = {"name": "Alice", "place": "Wonderland"}

        result = adapt_template(template, client, model="m")

        assert result.filled_body == "Hello Alice, welcome to Wonderland."

    def test_list_values_in_filled_body(self):
        template = FakeTemplate()
        template.body = "Items:\n{{items}}"
        template.placeholders = {"items": FakePlaceholder(hint="")}
        client = MagicMock()
        client.generate_json.return_value = {"items": ["one", "two", "three"]}

        result = adapt_template(template, client, model="m")

        assert "- one" in result.filled_body
        assert "- two" in result.filled_body
        assert "- three" in result.filled_body
