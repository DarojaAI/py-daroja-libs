"""Exponential backoff retry for LLM calls.

Generic retry utility with configurable retryable status codes and
exponential backoff.  Not LLM-specific — usable for any function that
may raise transient errors.

Usage::

    from common.llm.retry import retry_with_backoff

    result = retry_with_backoff(
        my_api_call,
        max_retries=5,
        initial_delay=5.0,
        retryable_status_codes=(503, 429),
    )

The retry checks for a ``status_code`` attribute on the raised
exception (common for HTTP errors).  If ``status_code`` is not in
``retryable_status_codes``, the exception is re-raised immediately
(non-retryable).
"""

import logging
import time
from typing import Callable, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 5,
    initial_delay: float = 5.0,
    retryable_status_codes: Tuple[int, ...] = (503,),
    description: str = "LLM call",
) -> T:
    """Execute *func* with exponential backoff on retryable errors.

    Args:
        func: Callable to execute.  Should raise on failure.
        max_retries: Maximum number of attempts (including the first).
        initial_delay: Initial delay in seconds before first retry.
            Subsequent delays double: ``initial_delay * 2**attempt``.
        retryable_status_codes: HTTP status codes that trigger a retry.
            The exception's ``status_code`` attribute is checked (if
            present).  If the attribute is missing, the exception is
            retried unconditionally (conservative — transient errors
            are common with network calls).
        description: Human-readable label for log messages.

    Returns:
        Result of *func*.

    Raises:
        The last exception if all retries are exhausted, or the
        original exception immediately if it's non-retryable.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code is not None and status_code not in retryable_status_codes:
                raise  # Non-retryable

            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s. Retrying in %.1fs …",
                    description,
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] All %d retries exhausted. Last error: %s",
                    description,
                    max_retries,
                    exc,
                )
                raise
    raise RuntimeError("retry_with_backoff reached an unexpected state")
