# `common.llm` — Shared LLM Client

Unified Python interface for Anthropic, OpenRouter, OpenAI, Azure OpenAI,
Ollama, and any other OpenAI-compatible endpoint. One import, one call
shape, one error model.

This module is the shared LLM client for `dev-nexus`, `rag_research_tool`,
and future DarojaAI consumers. See [py-daroja-libs `README.md`](../../README.md)
for the wider library context.

---

## Why this exists

Without `common.llm`, every consumer rewrites the same five things:

- HTTP client construction (httpx, requests, openai SDK, anthropic SDK)
- Provider-specific model-name conventions
- OpenRouter attribution headers (`HTTP-Referer`, `X-Title`)
- Error-handling retries and narrow exception types
- Token-usage extraction

The shared client collapses all of that into a single factory call:

```python
from common.llm import get_llm_client

client = get_llm_client(
    provider="openrouter",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="anthropic/claude-3-5-sonnet",
    http_referer="https://github.com/DarojaAI/rag_research_tool",
    x_title="rag-research-tool",
)
```

---

## Quick start

### Chat completion

```python
from common.llm import get_llm_client, LLMResponse

client = get_llm_client(
    provider="openrouter",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="anthropic/claude-3-5-sonnet",
    http_referer="https://github.com/DarojaAI/my_app",
    x_title="my-app",
)

response: LLMResponse = client.create_message(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Summarize the F3 incident."}],
    max_tokens=512,
    temperature=0.7,
)
print(response.content)
print(response.usage)  # {"input_tokens": ..., "output_tokens": ...}
```

### Structured output

```python
schema = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["domain", "confidence"],
    "additionalProperties": False,
}

result = client.generate_json(
    model="anthropic/claude-3-5-sonnet",
    prompt="Classify the domain of: 'quantum chromodynamics'",
    response_schema=schema,
    temperature=0.0,  # deterministic for structured output
)
# result == {"domain": "physics", "confidence": 0.97}
```

### Embeddings

```python
result = client.embed(
    model="openai/text-embedding-3-small",
    input=["hello world", "goodbye world"],
)
vectors = result.embeddings  # [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

### Embeddings — async batch

For high-throughput embedding pipelines (e.g. vector-store ingestion),
batch all texts in one call. Most providers charge per-request, not per-token,
for small batches.

```python
chunks = [chunk.text for chunk in document_chunks]
result = client.embed(model="openai/text-embedding-3-small", input=chunks)
# result.embeddings[i] corresponds to chunks[i]
```

---

## Providers

| `provider=` | Reads env var | Notes |
|---|---|---|
| `"anthropic"` | `ANTHROPIC_API_KEY` | Native Anthropic SDK. Model IDs: `claude-3-5-sonnet-20241022`, `claude-haiku-4-5`, … |
| `"openrouter"` | `OPENROUTER_API_KEY` | Default `base_url` = `https://openrouter.ai/api/v1`. Sets OpenRouter attribution headers. |
| `"openai"` | `OPENAI_API_KEY` | Native OpenAI SDK. Model IDs: `gpt-4o`, `gpt-4o-mini`, `text-embedding-3-small`, … |
| `"azure-openai"` (or `"azure_openai"`) | `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` | Requires `azure_endpoint`, `api_version` (default `2024-12-01-preview`). |
| `"ollama"` | — | `base_url` defaults to `http://localhost:11434/v1`. No auth. |
| `"openai-compatible"` (or `"openai_compatible"`) | — | Any `/v1/chat/completions` endpoint (vLLM, LiteLLM, LM Studio, …). Requires `base_url`. |

### OpenRouter model naming

OpenRouter uses `provider/model` slugs. Examples:

- `anthropic/claude-3-5-sonnet`
- `openai/gpt-4o-mini`
- `minimax/minimax-m3`
- `nomic-ai/nomic-embed-text-v1.5`

See <https://openrouter.ai/models> for the full catalogue.

### OpenRouter attribution headers

OpenRouter requires `HTTP-Referer` and `X-Title` on every request and shows
them on the dashboard / leaderboard. **Always pass them explicitly** to
attribute traffic to the right project:

```python
get_llm_client(
    provider="openrouter",
    api_key=OPENROUTER_API_KEY,
    model="anthropic/claude-3-5-sonnet",
    http_referer="https://github.com/DarojaAI/rag_research_tool",  # your project's URL
    x_title="rag-research-tool",                                   # your short project name
)
```

The client only sends these headers when `base_url` points at OpenRouter
(detected by the `openrouter.ai` substring). For other OpenAI-compatible
endpoints they're silently skipped.

---

## Error handling

The shared client raises:

- **`requests.exceptions.RequestException`** — HTTP / network failures. Includes:
  - `HTTPError` (4xx / 5xx — `raise_for_status()`)
  - `ConnectionError`
  - `Timeout`
- **`ValueError`** — `generate_json` couldn't parse the model's output as
  JSON. The original `json.JSONDecodeError` is chained via `raise ... from e`.
- **`ImportError`** — required SDK not installed (e.g. `pip install anthropic`
  for the Anthropic client).
- **`ValueError`** from the factory — missing API key, unknown provider,
  missing required `base_url`.

### Narrowing error types in consumer code

Consumers migrating from raw `openai.OpenAI` may notice the previous code
caught six `openai_errors.*` types (`AuthenticationError`,
`PermissionDeniedError`, `RateLimitError`, `APIConnectionError`,
`APITimeoutError`, `APIError`). After moving to `common.llm`, these collapse
to `requests.exceptions.RequestException` + `ValueError`.

**This is intentional.** The shared client is transport-agnostic. Consumer
code that needs to distinguish auth from rate-limit from network should
branch on `response.status_code` (or wrap a richer client) rather than
re-introducing OpenAI-SDK-specific imports.

Example from `rag_research_tool` (`tools/llm/hallucination_verifier.py`,
`EvidenceScorer.score_with_llm`):

```python
api_error_types = (requests.exceptions.RequestException,)
try:
    resp = client.create_message(...)
except api_error_types as exc:
    logger.error("LLM call failed: %s", exc)
    if exc.response is not None:
        status = exc.response.status_code
        if status in (401, 403):
            raise TokenTrackerAuthError(...) from exc
        if status == 429:
            raise TokenTrackerRateLimitError(...) from exc
    raise
```

This preserves the F3-incident error-narrowing rationale without coupling
the consumer to a specific SDK.

---

## Factory functions

### `get_llm_client(provider, api_key="", model=None, base_url=None, http_referer=None, x_title=None, ...)`

Direct factory. Use when you have all the credentials as explicit values
(env vars, vault secrets, settings file).

```python
client = get_llm_client(
    provider="openrouter",
    api_key=os.environ["OPENROUTER_API_KEY"],
    http_referer="https://github.com/DarojaAI/my_app",
    x_title="my-app",
)
```

### `get_llm_client_from_config(config)`

Reads attributes from a config object (Pydantic `BaseSettings`, dataclass,
plain object with `getattr`). Use when you have a project-wide settings
class.

```python
from pydantic_settings import BaseSettings
from common.llm import get_llm_client_from_config

class Settings(BaseSettings):
    llm_provider: str = "openrouter"
    openrouter_api_key: str
    llm_model: str = "anthropic/claude-3-5-sonnet"
    http_referer: str = "https://github.com/DarojaAI/my_app"
    x_title: str = "my-app"

settings = Settings()
client = get_llm_client_from_config(settings)
```

The attribute mapping is documented in `get_llm_client_from_config`'s
docstring. The short version: provider-specific keys
(`anthropic_api_key`, `openrouter_api_key`, `openai_api_key`,
`azure_openai_key`) win over the generic `llm_api_key` / `api_key` /
`LLM_API_KEY` fallback.

---

## Public API reference

| Symbol | Purpose |
|---|---|
| `LLMClient` | Abstract base class. All providers inherit from it. |
| `LLMResponse` | Dataclass with `content`, `model`, `stop_reason`, `usage`. |
| `Message` | Dataclass with `role`, `content`. Optional convenience wrapper. |
| `EmbeddingResponse` | Dataclass with `embeddings`, `model`, `usage`. |
| `AnthropicClient` | Anthropic SDK wrapper. |
| `OpenAICompatibleClient` | Generic `/v1` client. Aliased as `OpenRouterClient`. |
| `OpenRouterClient` | **Backward-compat alias** for `OpenAICompatibleClient`. New code should use the canonical name. |
| `OpenAIClient` | OpenAI SDK wrapper. |
| `AzureOpenAIClient` | Azure OpenAI SDK wrapper. |
| `get_llm_client(...)` | Direct factory. |
| `get_llm_client_from_config(config)` | Config-object factory. |

For the full method signatures, see `client.py` — every public method has
an Args / Returns / Raises docstring.

---

## Adding a new provider

If you need a provider that isn't covered (e.g. AWS Bedrock, Cohere,
Google Vertex), open an issue in `DarojaAI/py-daroja-libs` rather than
forking the client. The migration usually fits one of two patterns:

1. **OpenAI-compatible** — your provider exposes `/v1/chat/completions`.
   Use `OpenAICompatibleClient` directly with `base_url`. No new class needed.
2. **Native SDK** — your provider has a Python SDK. Subclass `LLMClient`,
   implement `create_message`, `generate_json`, `embed`, `get_provider_name`,
   then add a branch to `get_llm_client`.

---

## Cross-references

- [`rag_research_tool` consumer guide](https://github.com/DarojaAI/rag_research_tool/blob/main/docs/llm-client.md)
  — the canonical pattern for new LLM call sites in `rag_research_tool`.
- `py-daroja-libs` `README.md` — the wider shared library.
