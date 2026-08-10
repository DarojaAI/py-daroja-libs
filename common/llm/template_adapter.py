"""Schema-validated LLM template adapter.

Wraps an LLM call with a template's schema and style guide.  Builds
the system prompt from the template body + style guide, constrains
the LLM output to the template's JSON Schema, validates the result,
and runs machine rules.

This is the shared version of resume-customizer's ``templates/adapter.py``.
It does not depend on a specific LLM client — any object with a
``create_message`` or ``generate_json`` method works.

Usage::

    from common.llm.template_adapter import adapt_template, TemplateLike

    result = adapt_template(
        template=my_template,
        llm_client=client,
        model="anthropic/claude-3-5-sonnet",
        template_schema=my_template.schema,  # optional; preserves minLength/etc.
        rules_fn=run_rules,  # optional; accepts (parsed) or (style_guide, parsed)
    )
    if result.has_blocking_errors:
        ...
    else:
        print(result.filled_body)

Dependencies: ``jsonschema`` (Draft 2020-12).
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


class TemplateLike(Protocol):
    """Minimal protocol for a template object.

    Any object with these attributes works — resume-customizer's
    ``Template`` dataclass, document-pipeline's report templates, etc.
    """

    body: str
    name: str
    version: str
    placeholders: dict
    style_guide: Any  # must have a .prose attribute


@dataclass
class AdaptationResult:
    """Result of :func:`adapt_template`.

    Attributes:
        template_name: Name of the template used.
        template_version: Version string.
        parsed: Parsed JSON dict from the LLM.
        filled_body: Template body with placeholders replaced.
        schema_violations: List of JSON Schema validation error strings.
        rule_violations: List of RuleViolation objects from machine rules.
    """

    template_name: str
    template_version: str
    parsed: dict[str, Any]
    filled_body: str
    schema_violations: list[str] = field(default_factory=list)
    rule_violations: list = field(default_factory=list)

    @property
    def has_blocking_errors(self) -> bool:
        return bool(self.schema_violations) or any(
            getattr(v, "severity", None) == "error" for v in self.rule_violations
        )


def _build_system_prompt(template: TemplateLike) -> str:
    """Build the system prompt from the template's body + style guide prose."""
    parts: list[str] = []
    parts.append(
        "You fill a document template according to the spec below. "
        "Return only JSON matching the provided JSON Schema. "
        "Do not include markdown fences, explanations, or extra keys.\n"
    )
    parts.append("## Template body\n")
    parts.append(template.body)
    parts.append("\n\n## Placeholder hints\n")
    for name, ph in template.placeholders.items():
        hint = getattr(ph, "hint", None) if not isinstance(ph, str) else None
        if hint:
            parts.append(f"- `{name}`: {hint}")
    parts.append("\n\n## Style guide\n")
    parts.append(getattr(template.style_guide, "prose", ""))
    return "\n".join(parts)


def _fill_body(template: TemplateLike, parsed: dict[str, Any]) -> str:
    """Replace {{name}} placeholders in the template body with parsed values."""
    body = template.body
    for name in template.placeholders:
        value = parsed.get(name)
        if value is None:
            continue
        if isinstance(value, list):
            value = "\n".join(f"- {item}" for item in value)
        body = re.sub(
            rf"\{{\{{\s*{re.escape(name)}\b[^}}]*\}}\}}",
            str(value),
            body,
        )
    return body


def _synthesized_schema(template: TemplateLike) -> dict[str, Any]:
    """Fallback schema when caller does not provide one.

    Used only when ``template_schema`` is omitted.  Preserves the
    pre-#76 behaviour so existing callers (e.g. simple projects without
    per-template constraints) keep working unchanged.
    """
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in template.placeholders},
        "required": list(template.placeholders),
        "additionalProperties": False,
    }


def adapt_template(
    template: TemplateLike,
    llm_client: Any,
    model: str,
    *,
    source_doc: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    run_rules: bool = True,
    rules_fn: Any = None,
    template_schema: dict[str, Any] | None = None,
) -> AdaptationResult:
    """Adapt a template using an LLM call.

    Builds the system prompt from the template, calls the LLM with
    JSON Schema constraint, validates the output, fills the template
    body, and optionally runs machine rules.

    Args:
        template: Template object (see :class:`TemplateLike`).
        llm_client: Any client with a ``generate_json`` method.
            Typically an ``LLMClient`` from ``common.llm``.
        model: Model identifier for the LLM call.
        source_doc: Source document text to include in the user prompt.
        temperature: Sampling temperature (default 0.0 for structured output).
        max_tokens: Max output tokens.
        run_rules: If True, call ``rules_fn`` on the parsed output.
        rules_fn: Optional callable for rule checking.  Two signatures
            are supported, detected via :func:`inspect.signature`::

                rules_fn(parsed) -> list                  # legacy (1-arg)
                rules_fn(style_guide, parsed) -> list     # preferred (2-arg)

            The 2-arg form receives ``template.style_guide`` so callers
            like resume-customizer's ``run_rules(StyleGuide, parsed)``
            work without adapter glue.  If None, rule checking is
            skipped even when ``run_rules=True``.
        template_schema: Optional JSON Schema to use for validation.
            If provided, overrides the synthesized schema (which only
            encodes ``required`` + ``additionalProperties:false``).
            Pass this when your template declares per-field constraints
            (``minLength``, ``maxLength``, ``minItems``, ``maxItems``,
            ``additionalProperties:false`` on nested objects, etc.).
            If None, a synthesized schema is used — see
            :func:`_synthesized_schema`.

    Returns:
        AdaptationResult with parsed JSON, filled body, and any violations.
    """
    system_prompt = _build_system_prompt(template)
    user_prompt = source_doc or "Fill the template based on the provided context."

    schema = (
        template_schema
        if template_schema is not None
        else _synthesized_schema(template)
    )

    schema_violations: list[str] = []
    parsed: dict[str, Any] = {}

    try:
        if hasattr(llm_client, "generate_json"):
            parsed = llm_client.generate_json(
                model=model,
                prompt=f"{system_prompt}\n\n{user_prompt}",
                response_schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif hasattr(llm_client, "create_message"):
            response = llm_client.create_message(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = getattr(response, "content", str(response))
            parsed = json.loads(content)
        else:
            raise ValueError(
                "llm_client must have generate_json or create_message method"
            )
    except (json.JSONDecodeError, ValueError) as exc:
        schema_violations.append(f"JSON parse error: {exc}")
        parsed = {}

    # Validate against schema if jsonschema is available
    if Draft202012Validator is not None and parsed:
        validator = Draft202012Validator(schema)
        for error in validator.iter_errors(parsed):
            schema_violations.append(error.message)

    filled = _fill_body(template, parsed) if parsed else template.body

    rule_violations = []
    if run_rules and rules_fn is not None and parsed:
        try:
            sig = inspect.signature(rules_fn)
            nparams = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    or p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
            )
        except (TypeError, ValueError):
            nparams = 1  # builtins / no signature; assume legacy 1-arg form
        if nparams >= 2:
            rule_violations = rules_fn(template.style_guide, parsed)
        else:
            rule_violations = rules_fn(parsed)

    return AdaptationResult(
        template_name=getattr(template, "name", "unknown"),
        template_version=getattr(template, "version", "0"),
        parsed=parsed,
        filled_body=filled,
        schema_violations=schema_violations,
        rule_violations=rule_violations,
    )
