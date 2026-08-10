"""
LLM Client Factory — unified interface for multiple LLM providers.

Providers supported
-------------------

=============== ================================================== ==================
Provider        ``provider=`` argument                             Notes
=============== ================================================== ==================
Anthropic       ``"anthropic"``                                    Native SDK.
OpenRouter      ``"openrouter"``                                   Default ``base_url`` is OpenRouter. Sets ``HTTP-Referer`` and ``X-Title`` headers for OpenRouter attribution.
OpenAI          ``"openai"``                                       Native SDK; reads ``OPENAI_API_KEY``.
Azure OpenAI    ``"azure-openai"`` (or ``"azure_openai"``)          Native SDK with Azure auth; requires endpoint + API version.
Ollama          ``"ollama"``                                       ``http://localhost:11434/v1`` default; no auth required.
OpenAI-compat.  ``"openai-compatible"``                            Any ``/v1/chat/completions`` endpoint (vLLM, LiteLLM, LM Studio, …). Requires ``base_url``.
=============== ================================================== ==================

Model naming
------------

- **Anthropic** — Anthropic model IDs (``claude-3-5-sonnet-20241022``, ``claude-haiku-4-5``, …).
- **OpenRouter** — OpenRouter ``provider/model`` slugs (``anthropic/claude-3-5-sonnet``, ``openai/gpt-4o-mini``, ``minimax/minimax-m3``). See https://openrouter.ai/models.
- **OpenAI** — OpenAI model IDs (``gpt-4o``, ``gpt-4o-mini``, ``text-embedding-3-small``, …).
- **Azure OpenAI** — your deployment name (``my-gpt4-deployment``), not the underlying model.

OpenRouter headers
------------------

When the resolved ``base_url`` points at OpenRouter, ``OpenAICompatibleClient`` automatically sends ``HTTP-Referer`` and ``X-Title`` headers. **OpenRouter requires these on every request** (they show up in the OpenRouter dashboard attribution). Pass them explicitly per consumer to attribute traffic correctly:

.. code-block:: python

    get_llm_client(
        provider="openrouter",
        api_key=OPENROUTER_API_KEY,
        model="anthropic/claude-3-5-sonnet",
        http_referer="https://github.com/DarojaAI/rag_research_tool",
        x_title="rag-research-tool",
    )

Error model
-----------

- ``create_message`` — raises ``requests.exceptions.RequestException`` (HTTP / network). All HTTP error responses (``raise_for_status()``) and connection failures surface as ``RequestException`` or its subclasses (``HTTPError``, ``ConnectionError``, ``Timeout``).
- ``generate_json`` — same as ``create_message`` plus ``ValueError`` if the model's output fails ``json.loads`` or doesn't conform to ``response_schema``.
- ``embed`` — same as ``create_message``.
- The factory ``get_llm_client(...)`` raises ``ValueError`` for missing API keys, unknown providers, or ``base_url`` requirements (openai-compatible).

**Why narrow to ``RequestException + ValueError``?** A consumer migrating from raw ``openai.OpenAI`` may notice the previous code caught six ``openai_errors.*`` types (AuthenticationError, PermissionDeniedError, RateLimitError, APIConnectionError, APITimeoutError, APIError). After moving to ``common.llm``, these collapse to ``requests.exceptions.RequestException`` + ``ValueError``. This is intentional — the shared client is transport-agnostic, and consumer-side code that needs to distinguish auth from rate-limit from network should branch on ``response.status_code`` (or wrap a richer client) rather than re-introducing OpenAI-SDK-specific imports. See ``rag_research_tool``'s ``tools/llm/hallucination_verifier.py`` EvidenceScorer for the F3-incident-preserving pattern.

Quick start
-----------

.. code-block:: python

    from common.llm import get_llm_client, LLMResponse

    client = get_llm_client(
        provider="openrouter",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="anthropic/claude-3-5-sonnet",
        http_referer="https://github.com/DarojaAI/rag_research_tool",
        x_title="rag-research-tool",
    )

    response: LLMResponse = client.create_message(
        model="anthropic/claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        temperature=0.7,
    )
    print(response.content)           # str
    print(response.usage)             # {"input_tokens": ..., "output_tokens": ...} or None

Structured output:

.. code-block:: python

    schema = {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["domain", "confidence"],
    }
    result = client.generate_json(
        model="anthropic/claude-3-5-sonnet",
        prompt="Classify the domain of: 'quantum chromodynamics'",
        response_schema=schema,
        temperature=0.0,  # deterministic for structured output
    )
    # result == {"domain": "physics", "confidence": 0.97}
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message wrapper for chat-style LLM calls.

    Attributes:
        role: One of ``"user"`` or ``"assistant"``. (Provider-specific
            roles like ``"system"`` or ``"tool"`` are passed as plain
            ``Dict[str, str]`` entries in ``messages=`` rather than
            wrapped in ``Message``.)
        content: Text content of the message. Multimodal content blocks
            are not supported by this dataclass; pass ``{"role": ..., "content": [...]}``
            dicts directly.
    """

    role: str  # "user", "assistant"
    content: str


@dataclass
class EmbeddingResponse:
    """Response wrapper for embedding API calls.

    Attributes:
        embeddings: List of embedding vectors (one per input string).
            Each vector is a list of floats whose length depends on
            the model (e.g. 1536 for ``text-embedding-3-small``,
            3072 for ``text-embedding-3-large``, 4096 for many
            OpenRouter embedding models).
        model: The model identifier actually used by the provider
            (may differ from the request if the provider aliases).
        usage: Token usage dict with provider-specific keys, or
            ``None`` if the provider did not return usage. Typical
            shape: ``{"prompt_tokens": int, "total_tokens": int}``.
    """

    embeddings: List[List[float]]
    model: str
    usage: Optional[Dict[str, int]] = None


@dataclass
class LLMResponse:
    """Response wrapper for ``create_message``.

    Attributes:
        content: The model's text response. Empty string if the model
            returned no content (e.g. ``finish_reason="length"`` with
            nothing generated yet).
        model: The model identifier echoed back by the provider
            (may include provider-side suffixes or aliases).
        stop_reason: Why the model stopped generating. Common values:
            ``"end_turn"`` / ``"stop"`` (natural completion),
            ``"max_tokens"`` / ``"length"`` (truncated),
            ``"tool_use"`` / ``"tool_calls"`` (function-call requested).
            ``None`` if the provider did not return one.
        usage: Token usage dict with keys ``"input_tokens"`` and
            ``"output_tokens"``, or ``None`` if the provider did not
            return usage. Consumers wiring up a token tracker should
            read both fields and sum them for total cost tracking.
    """

    content: str
    model: str
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class LLMClient(ABC):
    """Abstract LLM client interface"""

    @abstractmethod
    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Create a chat completion and return the model's reply.

        Args:
            model: Provider-specific model identifier. See module-level
                docstring for the per-provider naming conventions
                (Anthropic model IDs, OpenRouter ``provider/model`` slugs,
                OpenAI model IDs, Azure deployment names).
            messages: Chat messages in OpenAI-style ``{"role": ..., "content": ...}``
                shape. ``role`` is typically ``"user"`` or ``"assistant"``.
                Some providers accept ``"system"`` as a leading message.
                Content may be a string or a list of content blocks
                (provider-dependent).
            max_tokens: Upper bound on output tokens. The shared client
                does **not** default this aggressively; pick a value
                appropriate for the call (1024 for short classifications,
                4096+ for long generations).
            temperature: Sampling temperature. Use ``0.0`` for deterministic
                structured output, ``0.7`` for natural-language replies.
            **kwargs: Forwarded verbatim to the underlying SDK or HTTP call.
                Common keys: ``top_p``, ``stop``, ``presence_penalty``,
                ``frequency_penalty``, ``response_format``.

        Returns:
            LLMResponse with ``content``, ``model``, ``stop_reason``, ``usage``.

        Raises:
            requests.exceptions.RequestException: HTTP or network failure.
                Includes ``HTTPError`` (4xx/5xx), ``ConnectionError``,
                ``Timeout``. For OpenAI/Anthropic SDK clients, the SDK's
                own exception types may also surface.
            ValueError: Provider rejected the request (e.g. missing model).
            ImportError: Required SDK is not installed (Anthropic / OpenAI / Azure).
        """
        pass

    @abstractmethod
    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response conforming to *response_schema*.

        Sends the prompt as a single user message and asks the model to
        reply with JSON that matches the supplied JSON Schema. Uses the
        provider's ``response_format=json_schema`` enforcement where
        available (OpenAI, Anthropic, OpenAI-compatible endpoints).

        Args:
            model: Provider-specific model identifier.
            prompt: User prompt text. The model receives this as the
                single ``user`` message; ``system`` guidance should be
                prepended by the caller.
            response_schema: JSON Schema dict describing the expected
                response shape. Example for a domain classifier:

                .. code-block:: python

                    {
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["domain", "confidence"],
                        "additionalProperties": False,
                    }

            max_tokens: Upper bound on output tokens. JSON responses are
                usually short; 1024 is often plenty.
            temperature: **Set to ``0.0`` for deterministic structured output.**
                The default is ``0.0`` for this reason. Higher temperatures
                risk schema-conforming but semantically random output.
            **kwargs: Forwarded to the underlying call.

        Returns:
            Parsed JSON dict matching *response_schema*. **The shared client
            does not validate the response against *response_schema*** —
            it relies on the provider's ``response_format=json_schema``
            enforcement. If the model emits invalid JSON despite that,
            ``ValueError`` is raised.

        Raises:
            ValueError: Model returned non-JSON or JSON that doesn't
                parse. The original ``json.JSONDecodeError`` is chained
                (``raise ... from e``) for debugging.
            requests.exceptions.RequestException: HTTP or network failure.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name (e.g. 'anthropic', 'openrouter')"""
        pass

    def embed(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> "EmbeddingResponse":
        """Create embeddings for the given input texts.

        Args:
            model: Embedding model identifier (e.g. "text-embedding-3-small").
            input: List of strings to embed.

        Returns:
            EmbeddingResponse with embeddings, model, and usage.

        Raises:
            NotImplementedError: If the client does not support embeddings.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )

    def embed_normalized(
        self,
        model: str,
        input: List[str],
        dim: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = 50,
        **kwargs,
    ) -> List[List[float]]:
        """Embed texts with optional L2-normalization and dimension padding.

        Convenience wrapper around :meth:`embed` that adds batching,
        L2-normalization, and zero-padding / truncation to a target
        dimension.  This is the recommended entry point for embedding
        call sites that store vectors in pgvector or other vector stores.

        Args:
            model: Embedding model identifier.
            input: List of strings to embed.
            dim: Target embedding dimension.  ``None`` skips pad/truncate.
            normalize: L2-normalize each vector to unit length (default ``True``).
            batch_size: Number of texts per API call (default 50).
            **kwargs: Forwarded to :meth:`embed`.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            NotImplementedError: If the client does not support embeddings.
        """
        results: List[List[float]] = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            resp = self.embed(model=model, input=batch, **kwargs)
            for vector in resp.embeddings:
                if normalize:
                    vector = normalize_and_pad(vector, dim=dim)
                elif dim is not None:
                    vector = _pad_or_truncate(vector, dim)
                results.append(vector)
        return results


def normalize_and_pad(vector: List[float], dim: Optional[int] = None) -> List[float]:
    """L2-normalize *vector* to unit length, then optionally pad/truncate to *dim*.

    Args:
        vector: Raw embedding from the API.
        dim: Target dimension.  When provided the vector is truncated (if
            longer) or zero-padded (if shorter).  ``None`` skips the
            pad/truncate step and only normalizes.

    Returns:
        Normalized (and optionally padded/truncated) vector.
    """
    norm = sum(x * x for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    if dim is not None:
        vector = _pad_or_truncate(vector, dim)
    return vector


def _pad_or_truncate(vector: List[float], dim: int) -> List[float]:
    """Pad or truncate *vector* to *dim* dimensions."""
    if len(vector) >= dim:
        return vector[:dim]
    return vector + [0.0] * (dim - len(vector))


def batch_embed(
    client: "LLMClient",
    texts: List[str],
    model: str,
    dim: Optional[int] = None,
    normalize: bool = True,
    batch_size: int = 50,
) -> List[List[float]]:
    """Standalone helper: embed *texts* with batching, normalization, and dim-pad.

    This is a convenience wrapper that delegates to
    ``client.embed_normalized()``.  Use this when you have a client
    instance but prefer a functional API.

    Args:
        client: An ``LLMClient`` instance.
        texts: Input strings to embed.
        model: Embedding model identifier.
        dim: Target embedding dimension (``None`` = no pad/truncate).
        normalize: L2-normalize each vector (default ``True``).
        batch_size: Texts per API call (default 50).

    Returns:
        List of embedding vectors.
    """
    return client.embed_normalized(
        model=model,
        input=texts,
        dim=dim,
        normalize=normalize,
        batch_size=batch_size,
    )


class AnthropicClient(LLMClient):
    """Anthropic Claude client wrapper"""

    def __init__(self, api_key: str):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=api_key)
        self.api_key = api_key
        logger.info("✓ Anthropic LLM client initialized")

    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Create a message via Anthropic API"""
        try:
            response = self.client.messages.create(
                model=model or "claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract content from response
            content = ""
            if response.content:
                content = (
                    response.content[0].text
                    if hasattr(response.content[0], "text")
                    else str(response.content[0])
                )

            return LLMResponse(
                content=content,
                model=response.model,
                stop_reason=getattr(response, "stop_reason", None),
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if response.usage
                else None,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response via Anthropic with json_schema.
        Uses response_format for guaranteed schema adherence.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.messages.create(
                model=model or "claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_schema,
                    },
                },
                **kwargs,
            )
            content = ""
            if response.content:
                content = (
                    response.content[0].text
                    if hasattr(response.content[0], "text")
                    else str(response.content[0])
                )
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Anthropic generate_json: invalid JSON from model: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Anthropic generate_json error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "anthropic"


class OpenAICompatibleClient(LLMClient):
    """Generic OpenAI-compatible client wrapper.

    Works with any endpoint that implements the ``/v1/chat/completions``
    API: OpenRouter, Ollama, vLLM, LiteLLM, LM Studio, text-generation-
    webui, and other OpenAI-compatible servers.

    Args:
        api_key: API key for the endpoint.  For local servers that don't
            require auth (e.g. Ollama), pass an empty string.
        base_url: Base URL of the ``/v1`` endpoint.  Defaults to
            ``https://openrouter.ai/api/v1`` (preserves OpenRouter behaviour
            for callers that don't pass an explicit URL).
        http_referer: ``HTTP-Referer`` header sent to OpenRouter.  Ignored
            for non-OpenRouter endpoints.  Defaults to
            ``https://github.com/DarojaAI/devnexus-common``.
        x_title: ``X-Title`` header sent to OpenRouter.  Ignored for
            non-OpenRouter endpoints.  Defaults to ``devnexus-common``.
        extra_headers: Additional headers to include in every request.
            ``Authorization`` and ``Content-Type`` are always set by the
            client and should NOT appear here.
    """

    DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError("requests package required: pip install requests")

        self.api_key = api_key
        self.base_url = (
            base_url.rstrip("/") if base_url else self.DEFAULT_OPENROUTER_URL
        )
        self.extra_headers = extra_headers or {}

        # OpenRouter-specific headers (only sent when base_url is OpenRouter)
        is_openrouter = "openrouter.ai" in self.base_url
        self._openrouter_headers: Dict[str, str] = {}
        if is_openrouter and api_key:
            self._openrouter_headers = {
                "HTTP-Referer": http_referer
                or "https://github.com/DarojaAI/devnexus-common",
                "X-Title": x_title or "devnexus-common",
            }

        logger.info(
            "✓ OpenAI-compatible LLM client initialized (base_url=%s)",
            self.base_url,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._openrouter_headers)
        headers.update(self.extra_headers)
        return headers

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Create a chat completion via an OpenAI-compatible ``/v1/chat/completions`` endpoint.

        Works against OpenRouter, Ollama, vLLM, LiteLLM, LM Studio, and any
        other server that implements the OpenAI chat-completions schema.
        Sends OpenRouter attribution headers (``HTTP-Referer``, ``X-Title``)
        automatically when ``base_url`` points at OpenRouter.

        See :class:`LLMClient` for the full argument and exception reference;
        this override only adds OpenAI-compatible-specific behaviour.

        Example:

        .. code-block:: python

            client = get_llm_client(
                provider="openrouter",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model="anthropic/claude-3-5-sonnet",
                http_referer="https://github.com/DarojaAI/my_app",
                x_title="my-app",
            )
            response = client.create_message(
                model="anthropic/claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Summarize the F3 incident."}],
                max_tokens=512,
            )
        """
        import requests

        if not model:
            raise ValueError("Model is required")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if data.get("choices") and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                # Some OpenRouter providers (e.g. minimax/minimax-m3) return
                # `content: null` when their reasoning_tokens consumed the
                # full max_tokens budget. The .get(..., "") default doesn't
                # catch this because the key exists but is JSON null. Coerce
                # None to "" so callers that call .lstrip() don't crash.
                raw_content = message.get("content")
                content = raw_content if isinstance(raw_content, str) else ""
                # If content is empty but the model emitted a separate
                # `reasoning` field (reasoning models), fall back to it so
                # the work isn't lost. The reasoning text may not be the
                # final answer — callers that need strict separation should
                # inspect message["reasoning"] / message["reasoning_details"]
                # themselves; this fallback exists for consumer convenience.
                if not content and message.get("reasoning"):
                    content = "[reasoning] " + message["reasoning"]

            return LLMResponse(
                content=content,
                model=data.get("model", model),
                stop_reason=data["choices"][0].get("finish_reason")
                if data.get("choices")
                else None,
                usage={
                    "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                }
                if data.get("usage")
                else None,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI-compatible API error ({self.base_url}): {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a structured JSON response via an OpenAI-compatible endpoint.

        Sends ``response_format={"type": "json_schema", ...}`` to the upstream
        server. Most OpenAI-compatible providers (OpenRouter, vLLM, LiteLLM,
        LM Studio) honor this for schema-constrained generation. Providers
        that don't support ``response_format`` (some Ollama builds) will fall
        back to ``json.loads(content)`` and may raise ``ValueError`` if the
        model emits invalid JSON.

        See :meth:`LLMClient.generate_json` for the full argument reference.
        """
        import requests

        if not model:
            raise ValueError("Model is required")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                },
            },
            **kwargs,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI-compatible generate_json: invalid JSON: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(
                f"OpenAI-compatible generate_json error ({self.base_url}): {e}"
            )
            raise

    def embed(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> "EmbeddingResponse":
        """Create embeddings via an OpenAI-compatible ``/v1/embeddings`` endpoint.

        Works against OpenRouter's embedding routes, Ollama's embedding
        endpoint, vLLM, and any other OpenAI-compatible server that
        implements ``/v1/embeddings``. Sends OpenRouter attribution headers
        when ``base_url`` points at OpenRouter.

        Args:
            model: Embedding model identifier. Examples:
                ``"openai/text-embedding-3-small"`` (OpenRouter),
                ``"text-embedding-3-small"`` (raw OpenAI),
                ``"nomic-ai/nomic-embed-text-v1.5"`` (OpenRouter).
            input: List of strings to embed. Batch all texts in one call
                when possible — OpenRouter and most providers charge per
                request, not per token, for small batches.
            **kwargs: Forwarded to the underlying HTTP call. Common keys:
                ``encoding_format`` (``"float"`` or ``"base64"``),
                ``dimensions`` (for Matryoshka-capable models).

        Returns:
            EmbeddingResponse with ``embeddings`` (list of float vectors,
            one per input string), ``model``, ``usage``.

        Raises:
            ValueError: ``model`` or ``input`` is empty.
            requests.exceptions.RequestException: HTTP or network failure.

        Example:

        .. code-block:: python

            client = get_llm_client(provider="openrouter", api_key=...)
            result = client.embed(
                model="openai/text-embedding-3-small",
                input=["hello world", "goodbye world"],
            )
            vectors = result.embeddings  # [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        """
        import requests as _requests

        if not model:
            raise ValueError("Model is required for embeddings")
        if not input:
            raise ValueError("Input texts are required for embeddings")

        payload = {
            "model": model,
            "input": input,
            **kwargs,
        }

        try:
            response = _requests.post(
                f"{self.base_url}/embeddings",
                headers=self._build_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            embeddings = [item["embedding"] for item in data.get("data", [])]
            return EmbeddingResponse(
                embeddings=embeddings,
                model=data.get("model", model),
                usage=data.get("usage"),
            )
        except _requests.exceptions.RequestException as e:
            logger.error(f"OpenAI-compatible embed error ({self.base_url}): {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_provider_name(self) -> str:
        if "openrouter.ai" in self.base_url:
            return "openrouter"
        return "openai-compatible"


# Backward-compatible alias
OpenRouterClient = OpenAICompatibleClient


class OpenAIClient(LLMClient):
    """Native OpenAI SDK client wrapper.

    Uses the ``openai`` Python package for built-in retries, streaming,
    and structured outputs.  Connects to ``api.openai.com`` by default
    or a custom ``base_url`` for proxies.

    Args:
        api_key: OpenAI API key (or env ``OPENAI_API_KEY``).
        base_url: Optional custom base URL (for proxies / compatible APIs).
        organization: Optional OpenAI organization ID.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai package required: pip install devnexus-common[openai]"
            )

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Pass api_key or set OPENAI_API_KEY."
            )

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        self.client = openai.OpenAI(**kwargs)
        self.api_key = api_key
        logger.info("✓ OpenAI SDK client initialized")

    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Create a message via the OpenAI SDK."""
        try:
            response = self.client.chat.completions.create(
                model=model or "gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else ""
            stop_reason = choice.finish_reason if choice else None

            usage = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            return LLMResponse(
                content=content or "",
                model=response.model or model,
                stop_reason=stop_reason,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured JSON via the OpenAI SDK."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=model or "gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_schema,
                    },
                },
                **kwargs,
            )
            content = ""
            if response.choices:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI generate_json: invalid JSON from model: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI generate_json error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "openai"


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI client wrapper.

    Uses the ``openai`` Python SDK with Azure-specific authentication.
    Requires an Azure OpenAI endpoint and API key (or managed identity).

    Args:
        api_key: Azure OpenAI API key (or env ``AZURE_OPENAI_KEY``).
        azure_endpoint: Azure OpenAI resource endpoint URL (or env
            ``AZURE_OPENAI_ENDPOINT``).  Example:
            ``https://my-resource.openai.azure.com/``
        api_version: Azure API version (or env ``OPENAI_API_VERSION``).
            Defaults to ``2024-12-01-preview``.
        deployment_name: Optional default deployment name.  If provided,
            this is used as the model in ``create_message`` when no model
            is explicitly passed.
    """

    DEFAULT_API_VERSION = "2024-12-01-preview"

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai package required: pip install devnexus-common[openai]"
            )

        api_key = api_key or os.environ.get("AZURE_OPENAI_KEY", "")
        if not api_key:
            raise ValueError(
                "Azure OpenAI API key is required. "
                "Pass api_key or set AZURE_OPENAI_KEY."
            )

        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not azure_endpoint:
            raise ValueError(
                "Azure endpoint is required. "
                "Pass azure_endpoint or set AZURE_OPENAI_ENDPOINT."
            )

        api_version = api_version or os.environ.get(
            "OPENAI_API_VERSION", self.DEFAULT_API_VERSION
        )

        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        logger.info("✓ Azure OpenAI client initialized (endpoint=%s)", azure_endpoint)

    def _resolve_model(self, model: str) -> str:
        """Resolve model: explicit > deployment_name > fallback."""
        return model or self.deployment_name or "gpt-4o"

    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Create a message via Azure OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self._resolve_model(model),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else ""
            stop_reason = choice.finish_reason if choice else None

            usage = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            return LLMResponse(
                content=content or "",
                model=response.model or model,
                stop_reason=stop_reason,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured JSON via Azure OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self._resolve_model(model),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_schema,
                    },
                },
                **kwargs,
            )
            content = ""
            if response.choices:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Azure OpenAI generate_json: invalid JSON from model: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Azure OpenAI generate_json error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "azure-openai"


# ======================================================================
# Factory functions
# ======================================================================

# Supported provider aliases
_PROVIDER_ALIASES: Dict[str, str] = {
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "openai": "openai",
    "azure-openai": "azure-openai",
    "azure_openai": "azure-openai",
    "openai-compatible": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "ollama": "ollama",
}


def get_llm_client(
    provider: str,
    api_key: str = "",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    # Azure-specific
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    deployment_name: Optional[str] = None,
    # OpenAI-specific
    organization: Optional[str] = None,
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.

    Provider resolution
    -------------------

    Pass ``provider=`` as one of the canonical names or accepted aliases:

    ===================  ============================================  ======================================
    Canonical            Aliases                                       Required arguments
    ===================  ============================================  ======================================
    ``"anthropic"``      —                                             ``api_key`` or ``ANTHROPIC_API_KEY``
    ``"openrouter"``     —                                             ``api_key`` or ``OPENROUTER_API_KEY``
    ``"openai"``         —                                             ``api_key`` or ``OPENAI_API_KEY``
    ``"azure-openai"``   ``"azure_openai"``                            ``api_key``, ``azure_endpoint``, ``api_version``
    ``"ollama"``         —                                             ``base_url`` (default ``http://localhost:11434/v1``)
    ``"openai-compatible"``  ``"openai_compatible"``                   ``base_url``
    ===================  ============================================  ======================================

    Args:
        provider: Provider name (see table above).
        api_key: API key for the provider. Falls back to the provider's
            standard env var (``ANTHROPIC_API_KEY``, ``OPENROUTER_API_KEY``,
            ``OPENAI_API_KEY``, ``AZURE_OPENAI_KEY``). Pass an empty
            string explicitly for unauthenticated local servers (Ollama).
        model: Default model hint. **Not used by the factory itself** —
            store it on the caller side if you want a project-wide default.
            Each ``create_message(...)`` / ``generate_json(...)`` call
            passes the model explicitly.
        base_url: Base URL for ``openai-compatible`` / ``ollama`` providers.
            For ``openrouter`` defaults to ``https://openrouter.ai/api/v1``.
            For ``ollama`` defaults to ``http://localhost:11434/v1``.
        http_referer: ``HTTP-Referer`` header (OpenRouter only). OpenRouter
            uses this for traffic attribution on the dashboard. Pass your
            project's URL — e.g. ``"https://github.com/DarojaAI/my_app"``.
        x_title: ``X-Title`` header (OpenRouter only). OpenRouter shows
            this in the dashboard and rankings. Pass your short project
            name — e.g. ``"my-app"``.
        extra_headers: Additional HTTP headers for OpenAI-compatible
            requests. ``Authorization`` and ``Content-Type`` are always set
            by the client and should NOT appear here.
        azure_endpoint: Azure OpenAI endpoint URL (required for
            ``azure-openai``; also reads ``AZURE_OPENAI_ENDPOINT``).
        api_version: Azure API version (default ``2024-12-01-preview``;
            also reads ``OPENAI_API_VERSION``).
        deployment_name: Default Azure deployment name. Used as the
            ``model=`` argument in ``create_message`` when no model is
            explicitly passed.
        organization: OpenAI organization ID.

    Returns:
        LLMClient instance. Concrete type depends on ``provider``:
        ``AnthropicClient``, ``OpenAICompatibleClient`` (alias
        ``OpenRouterClient``), ``OpenAIClient``, or ``AzureOpenAIClient``.

    Raises:
        ValueError: Unknown provider, missing required credentials, or
            missing required URL (``base_url`` for ``openai-compatible``).

    Examples:

    .. code-block:: python

        # OpenRouter with attribution headers
        client = get_llm_client(
            provider="openrouter",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="anthropic/claude-3-5-sonnet",
            http_referer="https://github.com/DarojaAI/my_app",
            x_title="my-app",
        )

        # Local Ollama, no auth
        client = get_llm_client(
            provider="ollama",
            api_key="",
            model="llama3.1",
        )

        # Azure
        client = get_llm_client(
            provider="azure-openai",
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint="https://my-resource.openai.azure.com/",
            deployment_name="my-gpt4-deployment",
        )
    """
    provider_key = provider.lower().strip()
    resolved = _PROVIDER_ALIASES.get(provider_key, provider_key)

    if resolved == "anthropic":
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Pass api_key or set ANTHROPIC_API_KEY."
            )
        return AnthropicClient(api_key=api_key)

    elif resolved == "openrouter":
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Pass api_key or set OPENROUTER_API_KEY."
            )
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url or OpenAICompatibleClient.DEFAULT_OPENROUTER_URL,
            http_referer=http_referer,
            x_title=x_title,
            extra_headers=extra_headers,
        )

    elif resolved == "openai":
        return OpenAIClient(
            api_key=api_key or None,
            base_url=base_url,
            organization=organization,
        )

    elif resolved == "azure-openai":
        return AzureOpenAIClient(
            api_key=api_key or None,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
        )

    elif resolved == "ollama":
        return OpenAICompatibleClient(
            api_key=api_key or "",
            base_url=base_url or "http://localhost:11434/v1",
            extra_headers=extra_headers,
        )

    elif resolved == "openai-compatible":
        if not base_url:
            raise ValueError("base_url is required for openai-compatible provider")
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            extra_headers=extra_headers,
        )

    else:
        supported = ", ".join(sorted(set(_PROVIDER_ALIASES.values())))
        raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: {supported}")


def get_llm_client_from_config(config: Any) -> LLMClient:
    """
    Create an LLM client from a config object (typically a Pydantic
    ``BaseSettings`` or a dataclass with the relevant attributes).

    Attribute mapping
    ----------------

    The factory reads the following attributes from *config* in order
    (first non-empty wins):

    ============================  =====================================  ===============================================
    Config attribute              Env var fallback                       Used for
    ============================  =====================================  ===============================================
    ``llm_provider``              ``LLM_PROVIDER`` (default ``anthropic``)  Provider name.
    ``llm_base_url``              ``LLM_BASE_URL``                       Base URL (openai-compatible / ollama).
    ``llm_model``                 ``LLM_MODEL``                          Default model hint (informational).
    ``anthropic_api_key``         ``ANTHROPIC_API_KEY``                 Anthropic API key.
    ``openrouter_api_key``        ``OPENROUTER_API_KEY``                OpenRouter API key.
    ``openai_api_key``            ``OPENAI_API_KEY``                    OpenAI API key.
    ``azure_openai_key``          ``AZURE_OPENAI_KEY``                  Azure API key.
    ``llm_api_key`` / ``api_key`` ``LLM_API_KEY``                       Generic fallback for unknown providers.
    ``http_referer``              —                                     OpenRouter ``HTTP-Referer``.
    ``x_title``                   —                                     OpenRouter ``X-Title``.
    ``extra_headers``             —                                     Extra HTTP headers.
    ``azure_endpoint``            ``AZURE_OPENAI_ENDPOINT``             Azure endpoint URL.
    ``api_version``               ``OPENAI_API_VERSION``                Azure API version.
    ``deployment_name``           —                                     Azure deployment name.
    ``organization``              —                                     OpenAI org ID.
    ============================  =====================================  ===============================================

    Args:
        config: Config object with ``llm_provider`` and (per provider)
            API key attributes. Any object with ``getattr`` works —
            Pydantic v2 ``BaseSettings``, dataclasses, simple ``object``
            subclasses are all fine.

    Returns:
        LLMClient instance.

    Raises:
        ValueError: Provider is unknown, or required credentials are
            missing after all fallback lookups.

    Example:

    .. code-block:: python

        from pydantic_settings import BaseSettings
        from common.llm import get_llm_client_from_config

        class Settings(BaseSettings):
            llm_provider: str = "openrouter"
            openrouter_api_key: str
            llm_model: str = "anthropic/claude-3-5-sonnet"
            http_referer: str = "https://github.com/DarojaAI/my_app"
            x_title: str = "my-app"

        settings = Settings()  # reads env vars
        client = get_llm_client_from_config(settings)
    """
    provider = (
        getattr(config, "llm_provider", None)
        or os.environ.get("LLM_PROVIDER", "anthropic")
    ).lower()

    base_url = getattr(config, "llm_base_url", None) or os.environ.get("LLM_BASE_URL")

    # Provider-specific API key resolution
    api_key = ""
    if provider in ("anthropic",):
        api_key = getattr(config, "anthropic_api_key", "") or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
    elif provider in ("openrouter",):
        api_key = getattr(config, "openrouter_api_key", "") or os.environ.get(
            "OPENROUTER_API_KEY", ""
        )
    elif provider in ("openai",):
        api_key = getattr(config, "openai_api_key", "") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
    elif provider in ("azure-openai", "azure_openai"):
        api_key = getattr(config, "azure_openai_key", "") or os.environ.get(
            "AZURE_OPENAI_KEY", ""
        )
    else:
        # generic — try common key attributes
        api_key = (
            getattr(config, "llm_api_key", "")
            or getattr(config, "api_key", "")
            or os.environ.get("LLM_API_KEY", "")
        )

    return get_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        http_referer=getattr(config, "http_referer", None),
        x_title=getattr(config, "x_title", None),
        extra_headers=getattr(config, "extra_headers", None),
        azure_endpoint=getattr(config, "azure_endpoint", None),
        api_version=getattr(config, "api_version", None),
        deployment_name=getattr(config, "deployment_name", None),
        organization=getattr(config, "organization", None),
    )
