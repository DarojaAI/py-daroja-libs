"""Reasoning model output extraction.

Strips thinking tags (DeepSeek R1, o1-style models) and extracts
JSON from mixed reasoning/text output.  Pure string processing -- no
network, no LLM calls, no dependencies beyond stdlib.

Usage::

    from common.llm.reasoning import extract_response_from_reasoning

    raw = '<think>Let me analyze this...</think>\\n{"score": 42}'
    clean = extract_response_from_reasoning(raw)
    # '{"score": 42}'

This is a post-processor: call it on the ``content`` field of an
``LLMResponse`` after ``create_message()`` returns.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Think tag constants (built programmatically to avoid tool mangling)
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


def extract_response_from_reasoning(content: str) -> str:
    """Extract the actual response from a reasoning model's output.

    Handles common reasoning model formats:
    - DeepSeek/o1-style output containing think-tag blocks
    - Already-clean JSON payloads
    - Mixed prose + JSON responses where JSON must be extracted

    Args:
        content: Raw model output string.

    Returns:
        Best-effort extracted response text (typically JSON if present).
    """
    if not content:
        return content

    # Method 1: Explicit thinking tags (DeepSeek R1, etc.)
    if _THINK_OPEN in content and _THINK_CLOSE in content:
        parts = content.split(_THINK_CLOSE)
        if len(parts) > 1:
            extracted = parts[-1].strip()
            logger.debug(
                "Stripped think tags, response length: %d chars", len(extracted)
            )
            return extracted

    # Method 2: Already clean JSON
    content_stripped = content.strip()
    if content_stripped.startswith("{") or content_stripped.startswith("["):
        return content

    # Method 3: Extract JSON from mixed content via heuristic markers
    json_start = content.find("{")
    json_markers = [
        "json\n{",
        "JSON:\n{",
        "```json",
        "Here is the JSON",
        "Here's the JSON",
        "The JSON response",
        "Final response:",
        '"score":',
        "'score':",
    ]
    for marker in json_markers:
        marker_pos = content.lower().find(marker.lower())
        if marker_pos != -1:
            temp_start = content.find("{", marker_pos)
            if temp_start != -1 and (json_start == -1 or temp_start < json_start):
                json_start = temp_start

    if json_start != -1:
        json_end = content.rfind("}")
        if json_end != -1 and json_end > json_start:
            potential_json = content[json_start : json_end + 1]
            if any(
                key in potential_json
                for key in ('"score"', '"analysis"', '"suggestions"')
            ):
                return potential_json

    # Method 4: Regex fallback -- find last top-level JSON object
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        potential_json = json_match.group(0)
        if '"score"' in potential_json or '"analysis"' in potential_json:
            return potential_json

    return content


def extract_json_from_reasoning(content: str) -> dict | list | None:
    """Extract and parse JSON from a reasoning model's output.

    Convenience wrapper around :func:`extract_response_from_reasoning`
    that also runs ``json.loads`` on the result.  Returns ``None`` if
    the content doesn't contain parseable JSON.

    Args:
        content: Raw model output string.

    Returns:
        Parsed JSON (dict or list), or ``None``.
    """
    cleaned = extract_response_from_reasoning(content)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
