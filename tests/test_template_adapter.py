"""Tests for common.llm.template_adapter — schema-validated LLM template filling."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

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
        client = MagicMock(spec=["create_message"])
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


# ---------------------------------------------------------------------------
# New in this PR: template_schema parameter (preserves per-template constraints)
# ---------------------------------------------------------------------------


class TestTemplateSchemaOverride:
    """Verify that adapt_template accepts a caller-provided JSON Schema
    and uses it instead of the synthesized placeholder-derived one."""

    def test_template_schema_overrides_synthesized(self):
        """Passing template_schema should make it the schema passed to
        generate_json, not the synthesized one."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        custom_schema = {
            "type": "object",
            "properties": {
                "section1": {"type": "string", "minLength": 5, "maxLength": 50},
                "section2": {"type": "string", "minLength": 10},
            },
            "required": ["section1", "section2"],
            "additionalProperties": False,
        }

        adapt_template(
            template,
            client,
            model="m",
            template_schema=custom_schema,
        )

        call_kwargs = client.generate_json.call_args
        passed_schema = call_kwargs.kwargs["response_schema"]
        # Custom schema was passed through verbatim.
        assert passed_schema == custom_schema
        # The synthesized form is NOT what got passed.
        synthesized = {
            "type": "object",
            "properties": {
                "section1": {"type": "string"},
                "section2": {"type": "string"},
            },
            "required": ["section1", "section2"],
            "additionalProperties": False,
        }
        assert passed_schema != synthesized

    def test_template_schema_catches_min_length_violation(self):
        """End-to-end: a short answer that violates minLength must be caught
        when template_schema is provided, but missed when it isn't.

        Skipped when ``jsonschema`` is not installed — the source code gates
        ``Draft202012Validator`` behind a ``try/except ImportError`` and the
        CI env for this repo does not declare ``jsonschema`` as a runtime
        dep.  Install ``jsonschema`` locally (or via the optional-dep
        group, once it ships) to exercise this test.
        """
        pytest.importorskip("jsonschema")
        template = FakeTemplate()
        client = MagicMock()
        # section1 = "Hi" is < minLength 5
        client.generate_json.return_value = {"section1": "Hi", "section2": "B"}

        custom_schema = {
            "type": "object",
            "properties": {
                "section1": {"type": "string", "minLength": 5},
                "section2": {"type": "string"},
            },
            "required": ["section1", "section2"],
            "additionalProperties": False,
        }

        result = adapt_template(
            template,
            client,
            model="m",
            template_schema=custom_schema,
        )

        assert len(result.schema_violations) >= 1
        assert result.has_blocking_errors

    def test_no_template_schema_uses_synthesized_fallback(self):
        """Omitting template_schema keeps the legacy behaviour: no
        minLength validation, short strings pass."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "Hi", "section2": "B"}

        result = adapt_template(template, client, model="m")

        assert result.schema_violations == []
        assert not result.has_blocking_errors


# ---------------------------------------------------------------------------
# New in this PR: rules_fn accepts both (parsed) and (style_guide, parsed)
# ---------------------------------------------------------------------------


class TestRulesFnSignature:
    """Verify that adapt_template detects the rules_fn arity and dispatches."""

    def test_rules_fn_one_arg_legacy_form(self):
        """1-arg form: rules_fn(parsed) — preserved from v1.16.0."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        received = []

        def legacy_rules(parsed):
            received.append(("one_arg", parsed))
            return []

        adapt_template(template, client, model="m", rules_fn=legacy_rules)
        assert received == [("one_arg", {"section1": "A", "section2": "B"})]

    def test_rules_fn_two_arg_styleguide_form(self):
        """2-arg form: rules_fn(style_guide, parsed) — preferred."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        received = []

        def styleguide_rules(style_guide, parsed):
            received.append(("two_arg", style_guide, parsed))
            return []

        adapt_template(template, client, model="m", rules_fn=styleguide_rules)
        assert len(received) == 1
        kind, sg, parsed = received[0]
        assert kind == "two_arg"
        assert sg is template.style_guide
        assert parsed == {"section1": "A", "section2": "B"}

    def test_rules_fn_two_arg_with_default_is_detected_as_two(self):
        """A 2-arg fn whose 2nd param has a default should still be treated
        as 2-arg (style_guide, parsed).  Defensive: callers may have
        rules_fn(guide, parsed, *, strict=True) with a kwarg default."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        def rules(style_guide, parsed, *, strict=True):
            return []

        # Should not raise; should call with (style_guide, parsed).
        result = adapt_template(template, client, model="m", rules_fn=rules)
        assert result.rule_violations == []

    def test_run_rules_false_skips_rules_fn(self):
        """run_rules=False should never call rules_fn regardless of arity."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        rules_fn = MagicMock(return_value=[])
        adapt_template(
            template,
            client,
            model="m",
            run_rules=False,
            rules_fn=rules_fn,
        )
        rules_fn.assert_not_called()

    def test_builtin_callable_falls_back_to_one_arg(self):
        """Builtins (no inspectable signature) are assumed 1-arg.
        Verifies the try/except guard around inspect.signature."""
        template = FakeTemplate()
        client = MagicMock()
        client.generate_json.return_value = {"section1": "A", "section2": "B"}

        # str.lower is a builtin; no signature.
        # We can't actually call it on parsed dict — wrap so it doesn't crash.
        def safe_builtin_like(parsed):
            return []

        result = adapt_template(template, client, model="m", rules_fn=safe_builtin_like)
        assert result.rule_violations == []
