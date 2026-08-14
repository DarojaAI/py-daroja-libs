# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Tool-use (function calling) support** in `common.llm.LLMClient.create_message()`. All four providers (Anthropic, OpenAI, Azure OpenAI, OpenRouter/OpenAI-compatible) now normalize native tool-call shapes into a unified `ToolCall` dataclass (`id`, `name`, `arguments`). Pass `tools=` and optionally `tool_choice=` via `**kwargs`; the response `LLMResponse.tool_calls` field contains normalized `ToolCall` objects or `None`. Fully backward-compatible: existing callers that don't pass `tools` see no change. Issue #1204 Phase 2.
- **`ToolResult` dataclass and `build_tool_result_messages()` helper.** Provider-agnostic `ToolResult(tool_call_id, content)` objects can be converted to provider-specific message dicts via `build_tool_result_messages(results, provider)`. Anthropic returns a single `role: "user"` message with `tool_result` content blocks; OpenAI-style providers return individual `role: "tool"` messages. Issue #1204 Phase 2.

## [1.7.0] - 2026-06-11

### Added

- **psycopg 3 backend** (`DatabaseManager(backend="psycopg3", ...)`). Pure-Python wrapper around libpq. Opt-in via `pip install "py-daroja-libs[psycopg3]"`. Implements the same `DatabaseBackend` Protocol as the asyncpg backend, shares the same public API, same dedicated event loop, same cancellation-safety machinery. Issue #28, #29.
- **`psycopg3` and `pgvector` optional-dependency groups** in `pyproject.toml`. Install the psycopg 3 backend with `pip install "py-daroja-libs[psycopg3]"`; full pgvector support with `pip install "py-daroja-libs[psycopg3,pgvector]"`.
- **Real-PG stress test parametrized against both backends.** The concurrent-cancellation stress test now runs against both `asyncpg` and `psycopg3` via two jobs in `.github/workflows/py-daroja-libs-stress.yml`. Issue #29.

### Test coverage

- 6 existing basic-query-helper tests parametrized against both backends (issue #29)
- 9 new dispatcher-specific tests in `TestBackendDispatch` (issue #29)
- 2 new tests verifying the translator is applied for psycopg 3 only (issue #29)
- 1 new stress test parametrized against both backends (issue #29)
- 1 new CI workflow job for the psycopg 3 stress test (issue #29)

### Total tests

- 128 (pre-parametrization) → 165+ (post-parametrization, since basic-query tests run 2x)

### Notes

- **asyncpg remains the default.** Existing deployments are unaffected.
- The psycopg 3 backend is **opt-in** for v1.7.0. It will become the default in v2.0 after 1 quarter of production soak.
- Both backends can be installed side-by-side if needed: `pip install "py-daroja-libs[psycopg3]"` adds psycopg 3 to an asyncpg-using environment.
