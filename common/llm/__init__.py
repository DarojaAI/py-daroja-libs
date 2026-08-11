from common.llm.client import (
    LLMClient,
    AnthropicClient,
    OpenAICompatibleClient,
    OpenRouterClient,  # backward-compatible alias
    OpenAIClient,
    AzureOpenAIClient,
    LLMResponse,
    ToolCall,
    ToolResult,
    build_tool_result_messages,
    EmbeddingResponse,
    Message,
    get_llm_client,
    get_llm_client_from_config,
    normalize_and_pad,
    batch_embed,
)
from common.llm.reasoning import (
    extract_response_from_reasoning,
    extract_json_from_reasoning,
)
from common.llm.retry import (
    retry_with_backoff,
)
from common.llm.template_adapter import (
    adapt_template,
    AdaptationResult,
    TemplateLike,
)
from common.llm.pdf_layout import (
    estimate_lines,
    estimate_pages,
    find_page_breaks,
    LayoutConfig,
)

__all__ = [
    # Client + factory
    "LLMClient",
    "AnthropicClient",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "build_tool_result_messages",
    "EmbeddingResponse",
    "Message",
    "get_llm_client",
    "get_llm_client_from_config",
    "normalize_and_pad",
    "batch_embed",
    # Reasoning extraction
    "extract_response_from_reasoning",
    "extract_json_from_reasoning",
    # Retry
    "retry_with_backoff",
    # Template adapter
    "adapt_template",
    "AdaptationResult",
    "TemplateLike",
    # PDF layout
    "estimate_lines",
    "estimate_pages",
    "find_page_breaks",
    "LayoutConfig",
]
