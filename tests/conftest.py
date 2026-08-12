"""Pytest configuration and shared fixtures for the devnexus-common test suite."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

repo_root = Path(__file__).parent.parent.resolve()
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers so ``-m integration`` filters cleanly."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (real network/API calls); "
        "deselect with -m 'not integration' for fast unit runs",
    )


# ---------------------------------------------------------------------------
# Mock optional SDKs for CI
# ---------------------------------------------------------------------------
# CI does not install ``openai`` or ``anthropic``.  We inject stub modules
# into ``sys.modules`` so that lazy imports inside client constructors
# succeed.  Tests that use ``@patch("openai.OpenAI")`` etc. will then
# correctly replace the mock class during each test.


def _ensure_mock_module(name: str):
    """Insert a stub module into sys.modules if absent."""
    import types as _types

    if name not in sys.modules:
        mod = _types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


_openai = _ensure_mock_module("openai")
_openai.OpenAI = MagicMock  # type: ignore[attr-defined]
_openai.AzureOpenAI = MagicMock  # type: ignore[attr-defined]

_anthropic = _ensure_mock_module("anthropic")
_anthropic.Anthropic = MagicMock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["asyncpg", "psycopg3"])
def backend(request):
    """Parametrized fixture: each test runs against both backends.

    Use this fixture to ensure a test exercises BOTH the asyncpg
    and psycopg 3 dispatch paths introduced in issue #28. The
    fixture value is a string: either ``"asyncpg"`` or
    ``"psycopg3"``.

    For backend-specific tests, gate execution with
    ``pytest.skip(...)`` so the test only runs against the relevant
    backend (otherwise you'd be asserting on the wrong path).

    See ``tests/test_postgres_sync.py::TestBackendDispatch`` for the
    canonical usage pattern.
    """
    return request.param
