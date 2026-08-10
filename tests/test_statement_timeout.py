"""Tests for the statement_timeout / command_timeout contract on DatabaseManager.

These tests codify the post-SIGSEGV-fix observability contract:
  - The asyncpg pool MUST be created with a `statement_timeout` in its
    `server_settings` dict (string value, asyncpg requires str).
  - The value MUST be at least 1000 ms (1 second) — sanity floor so a
    future refactor can't silently set it to 0 ("no timeout") and
    reintroduce the runaway-query class of bugs.
  - The pool MUST also set `command_timeout` to a positive number of seconds
    (asyncpg-level timeout, separate from the server-side `statement_timeout`).
  - The `statement_timeout` value SHOULD be configurable via the
    `POSTGRES_STATEMENT_TIMEOUT_MS` environment variable (opt-in contract
    test; skipped unless the env var is set in the test environment).

No real postgres required — `asyncpg.create_pool` is mocked at the
`common.db.postgres.asyncpg.create_pool` boundary so the kwargs can be
captured.
"""

from __future__ import annotations

import os
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.db.postgres import DatabaseManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager_with_mock_pool(
    *,
    enabled: bool = True,
    pool: Any = None,
    pool_state: str = "connected",
) -> DatabaseManager:
    """Build a DatabaseManager bypassing env-var validation.

    Mirrors the helper in tests/test_postgres_sync.py; copied here so this
    test module is self-contained (the contract test should not depend on
    the order tests are collected or whether test_postgres_sync.py is
    importable on every CI matrix).
    """
    mgr = DatabaseManager.__new__(DatabaseManager)
    mgr.enabled = enabled
    mgr.pool = pool
    mgr._connection_state = pool_state
    mgr._connection_error = None
    mgr.min_size = 0
    mgr.max_size = 5
    mgr.host = "fake-host"
    mgr.port = 5432
    mgr.database = "fake-db"
    mgr.user = "fake-user"
    mgr.password = "fake-pass"
    mgr._application_name = "test"
    mgr._search_path = "public"

    # Issue #686: env-var-driven statement_timeout. Tests that bypass
    # __init__ via __new__() must set this manually or connect() will
    # AttributeError on self.statement_timeout_ms.
    mgr.statement_timeout_ms = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "30000"))

    # Issue #7: cancellation-safety state.
    mgr._needs_pool_reset = False

    # Issues #7/#9: dedicated event loop state. Tests that call _run_sync
    # set these; tests that only call connect() directly never touch them.
    mgr._loop = None
    mgr._loop_thread = None
    mgr._loop_lock = threading.Lock()
    mgr._loop_started = threading.Event()
    mgr._loop_failed = None

    return mgr


def _fake_connected_pool(pgvector_row: Any = None) -> MagicMock:
    """A mock asyncpg pool that survives `mgr.connect()`'s post-create probe.

    `connect()` runs `SELECT extversion FROM pg_extension WHERE extname =
    'vector'` on a freshly-acquired connection. We return None (no row) so
    the probe logs a warning and returns — no further interaction with the
    pool is required.
    """
    pool = MagicMock(name="asyncpg_pool")

    conn = MagicMock(name="asyncpg_conn")
    conn.fetchrow = AsyncMock(return_value=pgvector_row)

    cm = MagicMock(name="acquire_cm")
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=cm)

    return pool


# ---------------------------------------------------------------------------
# Issue #686: statement_timeout is part of the pool contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_has_statement_timeout_in_server_settings():
    """`asyncpg.create_pool` MUST be called with a `server_settings` dict that
    contains a `statement_timeout` key whose value is a parseable integer of
    at least 1000 ms.

    This is the post-SIGSEGV-fix observability contract: a runaway query
    cannot hold a connection forever. The 1000 ms floor is intentionally
    loose (the actual default is 30000) so this test is a sanity check, not
    a pin on a specific value.
    """
    mgr = _make_manager_with_mock_pool()
    fake_pool = _fake_connected_pool()

    with patch(
        "common.db.postgres.asyncpg.create_pool",
        new_callable=AsyncMock,
    ) as mock_create_pool:
        mock_create_pool.return_value = fake_pool
        await mgr.connect()

    assert (
        mock_create_pool.await_count == 1
    ), "connect() must call asyncpg.create_pool exactly once on the happy path"

    kwargs = mock_create_pool.call_args.kwargs
    assert "server_settings" in kwargs, (
        f"asyncpg.create_pool must receive `server_settings`; got kwargs keys: "
        f"{sorted(kwargs.keys())}"
    )
    server_settings = kwargs["server_settings"]
    assert isinstance(
        server_settings, dict
    ), f"`server_settings` must be a dict (asyncpg requirement); got {type(server_settings).__name__}"
    assert "statement_timeout" in server_settings, (
        f"`server_settings` must contain 'statement_timeout'; got keys: "
        f"{sorted(server_settings.keys())}"
    )

    raw_value = server_settings["statement_timeout"]
    assert isinstance(raw_value, str), (
        f"`statement_timeout` must be a string (asyncpg passes server_settings "
        f"verbatim to the startup packet as text); got {type(raw_value).__name__}"
    )
    parsed = int(raw_value)
    assert parsed >= 1000, (
        f"`statement_timeout` must be at least 1000 ms (1s) to be a real "
        f"timeout; got {parsed} ms"
    )

    print(f"statement_timeout = {parsed} ms (raw={raw_value!r})")


@pytest.mark.asyncio
async def test_pool_has_command_timeout():
    """`asyncpg.create_pool` MUST set `command_timeout` to a positive number
    of seconds.

    `command_timeout` is asyncpg's per-operation timeout (separate from the
    server-side `statement_timeout`). asyncpg wants this as a float number
    of seconds. A 0 or missing value means "no timeout".
    """
    mgr = _make_manager_with_mock_pool()
    fake_pool = _fake_connected_pool()

    with patch(
        "common.db.postgres.asyncpg.create_pool",
        new_callable=AsyncMock,
    ) as mock_create_pool:
        mock_create_pool.return_value = fake_pool
        await mgr.connect()

    kwargs = mock_create_pool.call_args.kwargs
    assert "command_timeout" in kwargs, (
        f"asyncpg.create_pool must receive `command_timeout`; got kwargs keys: "
        f"{sorted(kwargs.keys())}"
    )
    command_timeout = kwargs["command_timeout"]
    assert command_timeout is not None, (
        "`command_timeout` must be set to a positive number; got None "
        "(asyncpg treats None as 'no timeout')"
    )
    assert command_timeout >= 1.0, (
        f"`command_timeout` must be at least 1.0 second to be a real timeout; "
        f"got {command_timeout!r}"
    )

    print(f"command_timeout = {command_timeout!r} seconds")


# ---------------------------------------------------------------------------
# Soft contract: configurability via POSTGRES_STATEMENT_TIMEOUT_MS.
# Skipped unless the env var is set in the test environment.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "POSTGRES_STATEMENT_TIMEOUT_MS" not in os.environ,
    reason=(
        "POSTGRES_STATEMENT_TIMEOUT_MS not set in the test environment — "
        "skipping the configurability contract test. Set the env var (any "
        "value) to opt in."
    ),
)
@pytest.mark.asyncio
async def test_statement_timeout_is_configurable(monkeypatch):
    """When `POSTGRES_STATEMENT_TIMEOUT_MS` is set, the captured
    `statement_timeout` MUST reflect that value (as a string).

    Forward-looking contract: when this test is run with the env var set,
    it asserts the implementation actually honors the override.
    """
    monkeypatch.setenv("POSTGRES_STATEMENT_TIMEOUT_MS", "5000")

    mgr = _make_manager_with_mock_pool()
    fake_pool = _fake_connected_pool()

    with patch(
        "common.db.postgres.asyncpg.create_pool",
        new_callable=AsyncMock,
    ) as mock_create_pool:
        mock_create_pool.return_value = fake_pool
        await mgr.connect()

    kwargs = mock_create_pool.call_args.kwargs
    server_settings = kwargs.get("server_settings", {})
    captured = server_settings.get("statement_timeout")

    assert captured == "5000", (
        f"`statement_timeout` must reflect POSTGRES_STATEMENT_TIMEOUT_MS=5000; "
        f"got {captured!r}. The implementation needs to read the env var "
        f"instead of hardcoding the value."
    )
