"""Tests for common.llm.retry — exponential backoff retry."""

import pytest
from unittest.mock import patch, MagicMock
from common.llm.retry import retry_with_backoff


class TestRetryWithBackoff:
    def test_success_first_try(self):
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, description="test")
        assert result == "ok"
        func.assert_called_once()

    def test_success_after_retries(self):
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "ok"])
        with patch("common.llm.retry.time.sleep") as mock_sleep:
            result = retry_with_backoff(
                func, max_retries=3, initial_delay=0.1, description="test"
            )
        assert result == "ok"
        assert func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_all_retries_exhausted(self):
        error = Exception("persistent failure")
        func = MagicMock(side_effect=error)
        with patch("common.llm.retry.time.sleep"):
            with pytest.raises(Exception, match="persistent failure"):
                retry_with_backoff(
                    func, max_retries=2, initial_delay=0.01, description="test"
                )
        assert func.call_count == 2

    def test_non_retryable_status_code(self):
        error = Exception("bad request")
        error.status_code = 400
        func = MagicMock(side_effect=error)
        with pytest.raises(Exception, match="bad request"):
            retry_with_backoff(func, retryable_status_codes=(503,), description="test")
        func.assert_called_once()

    def test_retryable_status_code(self):
        error = Exception("service unavailable")
        error.status_code = 503
        func = MagicMock(side_effect=[error, "ok"])
        with patch("common.llm.retry.time.sleep"):
            result = retry_with_backoff(
                func, retryable_status_codes=(503,), description="test"
            )
        assert result == "ok"
        assert func.call_count == 2

    def test_no_status_code_retries(self):
        """Exceptions without status_code are retried (conservative)."""
        func = MagicMock(side_effect=[ValueError("transient"), "ok"])
        with patch("common.llm.retry.time.sleep"):
            result = retry_with_backoff(func, max_retries=3, description="test")
        assert result == "ok"
        assert func.call_count == 2

    def test_exponential_backoff_timing(self):
        func = MagicMock(side_effect=[Exception("fail"), "ok"])
        with patch("common.llm.retry.time.sleep") as mock_sleep:
            retry_with_backoff(
                func, max_retries=2, initial_delay=5.0, description="test"
            )
        mock_sleep.assert_called_once_with(5.0)

    def test_exponential_backoff_second_retry(self):
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "ok"])
        with patch("common.llm.retry.time.sleep") as mock_sleep:
            retry_with_backoff(
                func, max_retries=3, initial_delay=2.0, description="test"
            )
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 2.0  # 2.0 * 2^0
        assert mock_sleep.call_args_list[1][0][0] == 4.0  # 2.0 * 2^1
