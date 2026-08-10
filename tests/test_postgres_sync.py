"""Tests for the sync facade and bulk_insert helper on DatabaseManager.

These tests cover:
  - _run_sync event-loop guard (must raise in a running loop, must work outside)
  - Sync wrappers bridge to the async methods correctly
  - bulk_insert builds the right SQL, chunks at page_size, and rejects
    mismatched column counts

No real postgres required — asyncpg is mocked at the .pool level.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import threading

import asyncpg.exceptions

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
    """Build a DatabaseManager with a mocked asyncpg pool.

    Bypasses the env-var validation by setting attributes directly.
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

    # Issue #7: cancellation-safety state (set by _with_retry on CancelledError,
    # consumed by ensure_connected).
    mgr._needs_pool_reset = False

    # Issues #7/#9: dedicated event loop state (set by _ensure_loop, used by
    # _run_sync). The test helper sets them to None so _ensure_loop knows
    # to start a fresh daemon thread if a test calls it.
    mgr._loop = None
    mgr._loop_thread = None
    mgr._loop_lock = threading.Lock()
    mgr._loop_started = threading.Event()
    mgr._loop_failed = None

    return mgr


def _fake_pool_with_execute(return_value: str = "INSERT 0 0") -> MagicMock:
    """AsyncMock pool whose acquire().execute(...) returns return_value."""
    pool = MagicMock()

    async def _execute(query, *args):
        return return_value

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=None)
    cm.execute = AsyncMock(side_effect=_execute)

    pool.acquire = MagicMock(return_value=cm)
    return pool, cm


def _make_psycopg_backend_mock(
    *,
    execute_return: str = "INSERT 0 0",
    fetch_return: Optional[list] = None,
    fetchrow_return: Any = None,
    fetchval_return: Any = None,
) -> MagicMock:
    """Build a MagicMock standing in for a psycopg 3 backend.

    The mock exposes the four basic-query methods the manager's
    dispatcher calls (execute / fetch / fetchrow / fetchval) plus
    ``executemany`` (the ``Protocol`` requires it). Each method
    defaults to a benign AsyncMock; tests that need a specific
    return value pass it via the corresponding keyword.
    """
    mock = MagicMock(name="psycopg3_backend")
    mock.name = "psycopg3"
    mock.execute = AsyncMock(return_value=execute_return)
    mock.fetch = AsyncMock(
        return_value=fetch_return if fetch_return is not None else []
    )
    mock.fetchrow = AsyncMock(return_value=fetchrow_return)
    mock.fetchval = AsyncMock(return_value=fetchval_return)
    mock.executemany = AsyncMock(return_value=None)
    return mock


def _make_manager_for_backend(
    backend: str,
    *,
    enabled: bool = True,
    pool: Any = None,
    backend_mock: Any = None,
    pool_state: str = "connected",
) -> DatabaseManager:
    """Build a DatabaseManager configured for the given backend.

    For ``backend='asyncpg'``: sets ``mgr.pool = pool`` and leaves
    ``mgr._backend`` unset so the dispatcher falls through to the
    asyncpg path via ``self.pool.acquire()``. The caller is
    expected to mock ``mgr.ensure_connected`` to a no-op
    AsyncMock so the dispatch actually runs.

    For ``backend='psycopg3'``: sets ``mgr._backend = backend_mock``
    (a MagicMock standing in for the psycopg 3 ``DatabaseBackend``)
    so the dispatcher takes the psycopg 3 path. ``backend_mock``
    defaults to a no-op MagicMock with ``name='psycopg3'`` and
    benign AsyncMocks for the four basic query methods. Pass a
    custom ``backend_mock`` to assert on call args / return values.
    """
    mgr = _make_manager_with_mock_pool(
        enabled=enabled,
        pool=pool,
        pool_state=pool_state,
    )
    if backend == "asyncpg":
        return mgr
    if backend == "psycopg3":
        if backend_mock is None:
            backend_mock = _make_psycopg_backend_mock()
        mgr._backend = backend_mock
        return mgr
    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# _run_sync event-loop guard
# ---------------------------------------------------------------------------


def test_run_sync_runs_coro_outside_loop():
    """Outside an event loop, _run_sync runs the coroutine via asyncio.run()."""
    mgr = _make_manager_with_mock_pool()

    async def _coro():
        return 42

    assert mgr._run_sync(_coro()) == 42


def test_run_sync_runs_coro_from_running_loop():
    """Inside a running event loop, _run_sync schedules the coroutine
    on the dedicated DB loop (a different thread) and waits for the
    result. It must not raise.

    Historical context: prior to the audit fix (2026-06-14), this
    function raised RuntimeError when called from a running event
    loop. The audit of rag_research_tool/eai found that FastAPI
    routers (which are async def endpoints) calling sync DB
    methods (db.fetch_sync, db.execute_sync, ...) hit that
    RuntimeError, and the routers' broad except handlers silently
    fell back to a disk cache — masking the persistence failure.
    The fix is to schedule on the dedicated DB loop instead of
    raising. The proper long-term fix is for the routers to
    ``await db.execute(...)`` directly.
    """
    mgr = _make_manager_with_mock_pool()
    result_box: dict = {}

    async def _coro():
        return "ok"

    async def _runner():
        # Must not raise. Should schedule on the dedicated loop
        # and return the coroutine's result.
        result_box["value"] = mgr._run_sync(_coro())

    asyncio.run(_runner())
    assert result_box.get("value") == "ok"


# ---------------------------------------------------------------------------
# Sync wrappers bridge to the async methods
# ---------------------------------------------------------------------------


def test_execute_sync_bridges_to_execute(backend):
    """execute_sync calls execute(query, *args) via _run_sync.

    Parametrized over both backends (issue #29). For asyncpg the
    dispatcher routes to ``self.pool.acquire().execute(...)``; for
    psycopg 3 it routes to ``self._backend.execute(...)`` with the
    SQL translated ``$1, $2`` → ``%s, %s``. Either way the sync
    facade must produce the backend's return value and forward
    the args unchanged.
    """
    if backend == "asyncpg":
        pool, cm = _fake_pool_with_execute(return_value="INSERT 0 7")
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(execute_return="INSERT 0 7")
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    # ensure_connected is a no-op when pool is set and state is "connected"
    mgr.ensure_connected = AsyncMock()

    # _execute_impl dispatches to either the pool's connection or
    # the backend's execute method, both of which return the status
    # string ("INSERT 0 7").
    result = mgr.execute_sync("INSERT INTO t VALUES ($1, $2)", "a", 1)
    assert result == "INSERT 0 7"
    if backend == "asyncpg":
        cm.execute.assert_awaited_once()
        # The original $1, $2 SQL passes through unchanged
        cm.execute.assert_called_once_with("INSERT INTO t VALUES ($1, $2)", "a", 1)
    else:
        mock.execute.assert_awaited_once()
        # The dispatcher translates $1, $2 → %s, %s for psycopg 3
        mock.execute.assert_called_once_with("INSERT INTO t VALUES (%s, %s)", "a", 1)


def test_fetch_sync_bridges_to_fetch(backend):
    """fetch_sync calls fetch(query, *args) via _run_sync.

    Parametrized over both backends (issue #29). The dispatcher
    routes to either ``self.pool.acquire().fetch(...)`` (asyncpg)
    or ``self._backend.fetch(...)`` (psycopg 3, with the SQL
    translated). The returned rows list must be the backend's
    return value verbatim.
    """
    if backend == "asyncpg":
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetch = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(
            fetch_return=[{"id": 1}, {"id": 2}],
        )
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    rows = mgr.fetch_sync("SELECT * FROM t WHERE id = $1", 1)
    assert len(rows) == 2
    if backend == "asyncpg":
        cm.fetch.assert_awaited_once()
        cm.fetch.assert_called_once_with("SELECT * FROM t WHERE id = $1", 1)
    else:
        mock.fetch.assert_awaited_once()
        mock.fetch.assert_called_once_with("SELECT * FROM t WHERE id = %s", 1)


def test_fetchrow_sync_returns_single_row(backend):
    """fetchrow_sync returns a single row (or ``None``) via _run_sync.

    Parametrized over both backends (issue #29). The dispatcher
    routes to either ``self.pool.acquire().fetchrow(...)`` (asyncpg)
    or ``self._backend.fetchrow(...)`` (psycopg 3, SQL translated).
    The returned row must be the backend's return value verbatim.
    """
    if backend == "asyncpg":
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetchrow = AsyncMock(return_value={"id": 1})
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(fetchrow_return={"id": 1})
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    row = mgr.fetchrow_sync("SELECT * FROM t WHERE id = $1", 1)
    assert row == {"id": 1}
    if backend == "asyncpg":
        cm.fetchrow.assert_awaited_once()
        cm.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id = $1", 1)
    else:
        mock.fetchrow.assert_awaited_once()
        mock.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id = %s", 1)


def test_fetchval_sync_returns_scalar(backend):
    """fetchval_sync returns a single scalar value via _run_sync.

    Parametrized over both backends (issue #29). The dispatcher
    routes to either ``self.pool.acquire().fetchval(...)`` (asyncpg)
    or ``self._backend.fetchval(...)`` (psycopg 3, SQL translated).
    The returned value must be the backend's return value verbatim.
    """
    if backend == "asyncpg":
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetchval = AsyncMock(return_value=42)
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(fetchval_return=42)
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    val = mgr.fetchval_sync("SELECT count(*) FROM t")
    assert val == 42
    if backend == "asyncpg":
        cm.fetchval.assert_awaited_once()
        cm.fetchval.assert_called_once_with("SELECT count(*) FROM t")
    else:
        mock.fetchval.assert_awaited_once()
        mock.fetchval.assert_called_once_with("SELECT count(*) FROM t")


def test_sync_wrappers_ensure_connected_each_call():
    """ensure_connected should be invoked through the bridge so the
    pool can lazily connect on first use."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=None)
    cm.fetchval = AsyncMock(return_value=1)
    pool.acquire = MagicMock(return_value=cm)
    mgr = _make_manager_with_mock_pool(pool=pool)
    mgr.ensure_connected = AsyncMock()

    mgr.fetchval_sync("SELECT 1")
    mgr.fetchval_sync("SELECT 2")

    assert mgr.ensure_connected.await_count == 2


# ---------------------------------------------------------------------------
# bulk_insert: SQL building, pagination, validation
# ---------------------------------------------------------------------------


def test_bulk_insert_empty_rows_returns_zero():
    """No rows means no SQL executed, returns 'INSERT 0'."""
    pool, cm = _fake_pool_with_execute()
    mgr = _make_manager_with_mock_pool(pool=pool)
    mgr.ensure_connected = AsyncMock()

    result = asyncio.run(mgr.bulk_insert("t", ["a", "b"], []))
    assert result == "INSERT 0"
    cm.execute.assert_not_called()
    mgr.ensure_connected.assert_not_called()


def test_bulk_insert_builds_correct_sql(backend):
    """Single-page insert: builds the parameterized VALUES clause.

    Parametrized over both backends (issue #29). The SQL builder
    itself is backend-agnostic (the manager produces the same
    ``$1, $2``-style placeholders regardless of backend), but
    parametrizing the test exercises the dispatch path too: the
    SQL is sent through the manager's execute_impl, which
    translates ``$N`` → ``%s`` for psycopg 3.
    """
    if backend == "asyncpg":
        pool, cm = _fake_pool_with_execute(return_value="INSERT 0 3")
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(execute_return="INSERT 0 3")
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    asyncio.run(
        mgr.bulk_insert("widgets", ["name", "qty"], [("a", 1), ("b", 2), ("c", 3)])
    )
    if backend == "asyncpg":
        cm.execute.assert_awaited_once()
        args, _ = cm.execute.call_args
        sql, *params = args
        # asyncpg path: $1, $2 placeholders pass through unchanged
        assert sql == "INSERT INTO widgets (name, qty) VALUES ($1, $2)"
    else:
        mock.execute.assert_awaited_once()
        args, _ = mock.execute.call_args
        sql, *params = args
        # psycopg 3 path: $1, $2 translated to %s, %s by the dispatcher
        assert sql == "INSERT INTO widgets (name, qty) VALUES (%s, %s)"
    # All six values (3 rows × 2 cols) flattened in order
    assert params == ["a", 1, "b", 2, "c", 3]


def test_bulk_insert_with_on_conflict_clause():
    """on_conflict is appended verbatim to the SQL."""
    pool, cm = _fake_pool_with_execute(return_value="INSERT 0 1")
    mgr = _make_manager_with_mock_pool(pool=pool)
    mgr.ensure_connected = AsyncMock()

    asyncio.run(
        mgr.bulk_insert(
            "patterns",
            ["repo_id", "name"],
            [(1, "x")],
            on_conflict="ON CONFLICT (repo_id, name) DO UPDATE SET description = EXCLUDED.description",
        )
    )
    args, _ = cm.execute.call_args
    sql = args[0]
    assert sql.startswith("INSERT INTO patterns (repo_id, name) VALUES ($1, $2) ")
    assert "ON CONFLICT (repo_id, name) DO UPDATE" in sql


def test_bulk_insert_rejects_mismatched_column_counts():
    pool, cm = _fake_pool_with_execute()
    mgr = _make_manager_with_mock_pool(pool=pool)
    mgr.ensure_connected = AsyncMock()

    with pytest.raises(ValueError, match="every row must have exactly 2 values"):
        asyncio.run(
            mgr.bulk_insert(
                "t", ["a", "b"], [("only-one",), ("two", "cols"), ("three", "cols")]
            )
        )
    # ensure_connected should NOT have been called since validation
    # fails before any SQL.
    mgr.ensure_connected.assert_not_called()
    cm.execute.assert_not_called()


def test_bulk_insert_paginates_at_page_size(backend):
    """5 rows with page_size=2 → 3 pages of 2, 2, 1.

    Parametrized over both backends (issue #29). Pagination logic
    is backend-agnostic (the manager chunks rows in the same way
    regardless of which backend is downstream), but parametrizing
    the test verifies the dispatch happens correctly on every
    page and the per-page result is parsed and summed.
    """
    if backend == "asyncpg":
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.execute = AsyncMock(side_effect=["INSERT 0 2", "INSERT 0 2", "INSERT 0 1"])
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(
            execute_return="INSERT 0 2",  # default; overridden by side_effect
        )
        mock.execute = AsyncMock(side_effect=["INSERT 0 2", "INSERT 0 2", "INSERT 0 1"])
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    rows = [(f"r{i}", i) for i in range(5)]
    result = asyncio.run(mgr.bulk_insert("t", ["a", "b"], rows, page_size=2))
    assert result == "INSERT 0 5"
    if backend == "asyncpg":
        assert cm.execute.await_count == 3
        # Each page gets the right number of params
        page_sizes = [
            len(call_args[0][1:]) // 2 for call_args in cm.execute.call_args_list
        ]
        assert page_sizes == [2, 2, 1]
    else:
        assert mock.execute.await_count == 3
        # Each page gets the right number of params; SQL is translated
        page_sizes = [
            len(call_args[0][1:]) // 2 for call_args in mock.execute.call_args_list
        ]
        assert page_sizes == [2, 2, 1]
        # The translated SQL on the first call should use %s, %s
        first_call_sql = mock.execute.call_args_list[0][0][0]
        assert first_call_sql == "INSERT INTO t (a, b) VALUES (%s, %s)"


def test_bulk_insert_total_handles_unparseable_result():
    """If the execute result is something we can't parse, return a
    best-effort total of 0 without raising."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=None)
    cm.execute = AsyncMock(side_effect=["weird response from server"])
    pool.acquire = MagicMock(return_value=cm)
    mgr = _make_manager_with_mock_pool(pool=pool)
    mgr.ensure_connected = AsyncMock()

    result = asyncio.run(mgr.bulk_insert("t", ["a"], [("x",)]))
    # Best-effort: still returns "INSERT 0 0"
    assert result == "INSERT 0 0"


def test_bulk_insert_sync_bridges(backend):
    """bulk_insert_sync is the sync wrapper around bulk_insert.

    Parametrized over both backends (issue #29). The sync facade
    must produce the same return value regardless of which backend
    the manager dispatches to.
    """
    if backend == "asyncpg":
        pool, cm = _fake_pool_with_execute(return_value="INSERT 0 2")
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
    else:
        mock = _make_psycopg_backend_mock(execute_return="INSERT 0 2")
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
    mgr.ensure_connected = AsyncMock()

    result = mgr.bulk_insert_sync("t", ["a", "b"], [("x", 1), ("y", 2)])
    assert result == "INSERT 0 2"
    if backend == "asyncpg":
        cm.execute.assert_awaited_once()
    else:
        mock.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Module-level sync helpers
# ---------------------------------------------------------------------------


def test_init_db_sync_returns_singleton():
    """init_db_sync returns the singleton, calling connect via _run_sync."""
    from common.db import postgres as pg_module

    with patch.object(pg_module, "_db_manager", None):
        mgr = DatabaseManager.__new__(DatabaseManager)
        mgr.enabled = True
        mgr.pool = MagicMock()
        mgr._connection_state = "connected"
        mgr._connection_error = None
        mgr.host = "fake"
        mgr.port = 5432
        mgr.database = "fake"
        mgr.user = "fake"
        mgr.password = "fake"
        mgr._application_name = "test"
        mgr._search_path = "public"
        mgr._run_sync = MagicMock(return_value=None)

        with patch.object(pg_module, "DatabaseManager", return_value=mgr):
            result = pg_module.init_db_sync()
            assert result is mgr
            mgr._run_sync.assert_called_once()


def test_close_db_sync_skips_when_pool_none():
    """close_db_sync is a no-op when the pool was never initialized."""
    from common.db import postgres as pg_module

    with patch.object(pg_module, "_db_manager", None):
        mgr = DatabaseManager.__new__(DatabaseManager)
        mgr.enabled = True
        mgr.pool = None
        mgr._connection_state = "disconnected"
        mgr._connection_error = None
        mgr.host = "fake"
        mgr.port = 5432
        mgr.database = "fake"
        mgr.user = "fake"
        mgr.password = "fake"
        mgr._application_name = "test"
        mgr._search_path = "public"
        mgr._run_sync = MagicMock()

        with patch.object(pg_module, "DatabaseManager", return_value=mgr):
            pg_module.close_db_sync()
            mgr._run_sync.assert_not_called()


def test_close_db_sync_closes_when_pool_present():
    from common.db import postgres as pg_module

    mgr = DatabaseManager.__new__(DatabaseManager)
    mgr.enabled = True
    mgr.pool = MagicMock()
    mgr._connection_state = "connected"
    mgr._connection_error = None
    mgr.host = "fake"
    mgr.port = 5432
    mgr.database = "fake"
    mgr.user = "fake"
    mgr.password = "fake"
    mgr._application_name = "test"
    mgr._search_path = "public"
    mgr._run_sync = MagicMock()

    with patch.object(pg_module, "_db_manager", mgr):
        pg_module.close_db_sync()
        mgr._run_sync.assert_called_once()


# ---------------------------------------------------------------------------
# __init__ port handling — regression for 'int' object has no attribute 'strip'
# ---------------------------------------------------------------------------
# The constructor's type hint is `port: Optional[int]`, but the previous
# implementation did `port_str.strip()` which raised AttributeError when a
# caller passed an int (the natural Python type). This was first noticed
# when rag_research_tool's wiki server, after migrating to the sync facade,
# hit `Connection failed: 'int' object has no attribute 'strip'` in /health.


def test_init_accepts_int_port():
    """port=5432 (int) must not raise; the natural Python type for a port."""
    env = {"POSTGRES_HOST": "fake-host"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=5432)
    assert mgr.port == 5432
    assert isinstance(mgr.port, int)


def test_init_accepts_str_port():
    """port='5432' (str, e.g. from os.environ) must be coerced to int."""
    env = {"POSTGRES_HOST": "fake-host"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port="5432")
    assert mgr.port == 5432
    assert isinstance(mgr.port, int)


def test_init_port_none_uses_env_var():
    """port=None falls back to POSTGRES_PORT env var."""
    env = {"POSTGRES_HOST": "fake-host", "POSTGRES_PORT": "6432"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=None)
    assert mgr.port == 6432


def test_init_port_none_default_when_env_unset():
    """port=None with POSTGRES_PORT unset uses the 5432 default."""
    env = {"POSTGRES_HOST": "fake-host"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("POSTGRES_PORT", None)
        mgr = DatabaseManager(host="fake-host", port=None)
    assert mgr.port == 5432


def test_init_default_use_postgresql_is_true():
    """The facade exists to provide PG access — defaulting to enabled.

    Regression: the previous default of 'false' silently disabled every
    downstream that didn't set USE_POSTGRESQL=true explicitly. First caught
    when rag_research_tool's wiki /db-status returned 'PostgreSQL is disabled'
    even though POSTGRES_HOST was set, because no caller set the flag.
    """
    env = {"POSTGRES_HOST": "fake-host"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("USE_POSTGRESQL", None)
        mgr = DatabaseManager(host="fake-host", port=5432)
    assert mgr.enabled is True
    assert mgr.port == 5432


def test_init_use_postgresql_false_explicit_opt_out():
    """Setting USE_POSTGRESQL=false still disables the client."""
    env = {"POSTGRES_HOST": "fake-host", "USE_POSTGRESQL": "false"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=5432)
    assert mgr.enabled is False
    assert mgr.port == 5432


def test_init_min_pool_size_defaults_to_zero_outside_prod():
    """min_size defaults to 0 in non-prod envs (lazy connect).

    A previous version tried min_size=2 to reduce cold-start races, but
    that amplified a concurrency bug: the DatabaseManager's disconnect()/
    connect() path is NOT safe for concurrent callers, and eager init
    increased the chance of two coroutines racing for the pool. Reverted.
    """
    env = {"POSTGRES_HOST": "fake-host", "ENVIRONMENT": "dev"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=5432)
    assert mgr.min_size == 0


def test_init_min_pool_size_two_in_prod():
    """In prod env, min_size=2 is the original default."""
    env = {"POSTGRES_HOST": "fake-host", "ENVIRONMENT": "prod"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=5432)
    assert mgr.min_size == 2


def test_init_min_pool_size_explicit_zero_still_honored():
    """Callers who genuinely want lazy connect can pass min_size=0."""
    env = {"POSTGRES_HOST": "fake-host", "ENVIRONMENT": "prod"}
    with patch.dict(os.environ, env, clear=False):
        mgr = DatabaseManager(host="fake-host", port=5432, min_size=0)
    assert mgr.min_size == 0


@pytest.mark.asyncio
async def test_ensure_connected_does_no_liveness_probe():
    """ensure_connected must not acquire a connection just to probe.

    A previous version ran a `SELECT 1` liveness probe and called
    disconnect()+connect() on failure. That pattern is NOT safe for
    concurrent callers (segfaults when one coroutine is in disconnect()
    while another tries to acquire from the closing pool).

    The natural asyncpg pattern is to skip the probe entirely; let
    `_with_retry` handle transient errors by retrying on a fresh
    connection from the pool.
    """
    mgr = _make_manager_with_mock_pool()
    # pool.acquire should NOT be called at all
    mgr.pool = MagicMock()
    mgr.pool.acquire = MagicMock()
    mgr.disconnect = AsyncMock()
    mgr.connect = AsyncMock()

    await mgr.ensure_connected()

    mgr.pool.acquire.assert_not_called()
    mgr.disconnect.assert_not_called()
    mgr.connect.assert_not_called()


@pytest.mark.asyncio
async def test_with_retry_does_not_disconnect_on_transient_error():
    """_with_retry must NOT call disconnect() on transient errors.

    Regression: rag_research_tool's wiki revision -00078-bsc segfaulted
    because _with_retry called disconnect() while another concurrent
    request was using the pool. Two SIGSEGVs in two minutes.

    Fix: _with_retry just sleeps and retries. asyncpg's pool naturally
    returns a different connection on the next acquire.
    """
    mgr = _make_manager_with_mock_pool()
    mgr.disconnect = AsyncMock()
    mgr.connect = AsyncMock()

    call_count = [0]

    async def _flaky():
        call_count[0] += 1
        if call_count[0] == 1:
            raise asyncpg.exceptions.ConnectionDoesNotExistError("simulated stale")
        return 42

    result = await mgr._with_retry(_flaky)
    assert result == 42
    assert call_count[0] == 2
    # CRITICAL: disconnect and connect must NOT have been called
    mgr.disconnect.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #7: cancellation safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_retry_marks_pool_for_reset_on_cancellation():
    """When the in-flight coro is cancelled, _with_retry sets
    _needs_pool_reset = True and re-raises CancelledError unchanged.

    This is the load-bearing assertion for issue #7: the flag is the
    handoff that lets the next non-cancelled ensure_connected() rebuild
    the pool cleanly, instead of us disconnecting mid-cancellation (which
    is the original segfault pattern).
    """
    mgr = _make_manager_with_mock_pool()
    assert mgr._needs_pool_reset is False

    async def _cancelled_coro():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await mgr._with_retry(_cancelled_coro)

    assert mgr._needs_pool_reset is True, (
        "CancelledError must flag the pool for deferred reset "
        "(see issue #7 / production SIGSEGV root cause)"
    )


@pytest.mark.asyncio
async def test_ensure_connected_resets_pool_when_flag_set():
    """If _needs_pool_reset is True and pool is alive, ensure_connected
    disconnects cleanly and reconnects, then clears the flag.

    The disconnect happens in a non-racing context (no CancelledError
    in flight), which is what makes this pattern safe.
    """
    mgr = _make_manager_with_mock_pool()
    mgr._needs_pool_reset = True
    old_pool = MagicMock(name="old_pool")
    mgr.pool = old_pool
    mgr.disconnect = AsyncMock()
    mgr.connect = AsyncMock()

    await mgr.ensure_connected()

    mgr.disconnect.assert_awaited_once()
    mgr.connect.assert_awaited_once()
    assert mgr._needs_pool_reset is False, "reset flag must be consumed"


@pytest.mark.asyncio
async def test_ensure_connected_no_reset_when_flag_unset():
    """Regression check: when _needs_pool_reset is False (the normal
    case), ensure_connected does NOT call disconnect. We rely on
    asyncpg's natural pool-dead-connection handling; calling disconnect
    on every call would be expensive and would re-introduce the
    segfault pattern (issue #7).
    """
    mgr = _make_manager_with_mock_pool()
    assert mgr._needs_pool_reset is False
    # pool is non-None so ensure_connected takes the "already connected" path
    mgr.pool = MagicMock(name="healthy_pool")
    mgr.disconnect = AsyncMock()
    mgr.connect = AsyncMock()

    await mgr.ensure_connected()

    mgr.disconnect.assert_not_called()
    mgr.connect.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_connected_resets_flag_even_if_disconnect_fails():
    """If disconnect() raises during the reset path, ensure_connected
    must still clear the flag and proceed (set pool to None, fall
    through to reconnect). Otherwise the flag stays set forever and
    every subsequent call tries (and fails) to reset."""
    mgr = _make_manager_with_mock_pool()
    mgr._needs_pool_reset = True
    mgr.pool = MagicMock(name="old_pool")
    mgr.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))
    mgr.connect = AsyncMock()

    await mgr.ensure_connected()

    assert mgr._needs_pool_reset is False
    mgr.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# Issues #7/#9: dedicated event loop in _run_sync
# ---------------------------------------------------------------------------


def test_run_sync_uses_dedicated_loop_not_per_call_asyncio_run():
    """Two _run_sync calls in a row must share the same loop. The whole
    point of the refactor is to bind the asyncpg pool to one stable loop
    for the lifetime of the DatabaseManager; a fresh loop per call would
    re-introduce the segfault (issue #7/#9)."""
    mgr = _make_manager_with_mock_pool()

    async def _coro():
        return 1

    mgr._run_sync(_coro())
    first_loop = mgr._loop
    first_thread = mgr._loop_thread
    assert first_loop is not None
    assert first_thread is not None
    assert first_thread.daemon is True, "loop thread must be daemon"
    assert first_thread.is_alive()

    mgr._run_sync(_coro())
    assert mgr._loop is first_loop, "second call must reuse the same loop"
    assert mgr._loop_thread is first_thread, "second call must reuse the same thread"


def test_ensure_loop_returns_same_loop_for_concurrent_threads():
    """Thread-safety: two threads racing on _ensure_loop() must both
    get the same loop (not start two threads). The lock serializes
    the start sequence."""
    mgr = _make_manager_with_mock_pool()
    loops: list = []

    def _worker():
        loop = mgr._ensure_loop()
        loops.append(loop)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(loops) == 2
    assert loops[0] is loops[1], "both threads must observe the same loop"
    assert mgr._loop is not None
    assert mgr._loop_thread is not None


def test_run_sync_propagates_cancelled_error_from_coro():
    """If the coroutine itself raises CancelledError, _run_sync re-raises
    it as-is. This is the cancellation contract: the caller knows the
    op was cancelled and can react. Combined with _with_retry's flag-set
    behavior, this is what makes the whole pipeline work."""
    mgr = _make_manager_with_mock_pool()

    async def _cancelled():
        raise asyncio.CancelledError()

    # concurrent.futures.Future.result() re-raises asyncio.CancelledError
    # as concurrent.futures.CancelledError (different class, NOT a subclass
    # of asyncio.CancelledError as of Python 3.10). The cancellation
    # contract from the caller's perspective is "a CancelledError-family
    # exception was raised" — accept either class.
    import concurrent.futures

    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        mgr._run_sync(_cancelled())

    assert mgr._loop is not None, "loop must be initialized even on cancellation"


# ---------------------------------------------------------------------------
# Issue #28/#29: backend dispatcher
# ---------------------------------------------------------------------------
# The manager's _execute_impl / _fetch_impl / _fetchrow_impl /
# _fetchval_impl route by backend type:
#   - asyncpg: use self.pool (existing behavior)
#   - psycopg3: call self._backend.{execute,fetch,fetchrow,fetchval}
#               with the SQL translated via translate_asyncpg_to_psycopg
#
# The tests in TestBackendDispatch verify the dispatch path itself:
# that the right method is called for each backend, that the
# translator is wired up for psycopg 3 only, and that the args
# (and return value) flow through unchanged.


class TestBackendDispatch:
    """Tests for the backend dispatcher introduced in #28.

    Each test uses the ``backend`` fixture from conftest.py so the
    full dispatcher behavior is exercised against BOTH backends.
    Tests that target a single backend explicitly call
    ``pytest.skip`` for the other so the assertions are correct.
    """

    def test_execute_dispatches_to_pool_for_asyncpg(self, backend):
        """For asyncpg, ``_execute_impl`` reads through ``self.pool``.

        The dispatcher condition is ``backend.name != "asyncpg"``;
        when ``self._backend`` is unset (the bypass-__init__ test
        path) the dispatcher falls through to the asyncpg path
        via ``getattr(self, "_backend", None) is None``.
        """
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool, cm = _fake_pool_with_execute(return_value="INSERT 0 7")
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
        mgr.ensure_connected = AsyncMock()

        result = asyncio.run(mgr.execute("INSERT INTO t VALUES ($1, $2)", "a", 1))

        assert result == "INSERT 0 7"
        # The pool's connection.execute was called with the ORIGINAL
        # $1, $2 SQL (no translation)
        cm.execute.assert_awaited_once()
        cm.execute.assert_called_once_with("INSERT INTO t VALUES ($1, $2)", "a", 1)

    def test_execute_dispatches_to_backend_for_psycopg3(self, backend):
        """For psycopg 3, ``_execute_impl`` delegates to
        ``self._backend.execute`` with the SQL translated.
        """
        if backend != "psycopg3":
            pytest.skip("psycopg 3-specific dispatch path")
        mock = _make_psycopg_backend_mock(execute_return="INSERT 0 7")
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
        mgr.ensure_connected = AsyncMock()

        result = asyncio.run(mgr.execute("INSERT INTO t VALUES ($1, $2)", "a", 1))

        assert result == "INSERT 0 7"
        # The backend's execute was called with the TRANSLATED SQL
        mock.execute.assert_awaited_once()
        mock.execute.assert_called_once_with("INSERT INTO t VALUES (%s, %s)", "a", 1)

    def test_fetch_dispatches_to_pool_for_asyncpg(self, backend):
        """For asyncpg, ``_fetch_impl`` reads through ``self.pool``."""
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetch = AsyncMock(return_value=[{"id": 1}])
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
        mgr.ensure_connected = AsyncMock()

        rows = asyncio.run(mgr.fetch("SELECT * FROM t WHERE id = $1", 1))

        assert len(rows) == 1
        cm.fetch.assert_awaited_once()
        cm.fetch.assert_called_once_with("SELECT * FROM t WHERE id = $1", 1)

    def test_fetch_dispatches_to_backend_for_psycopg3(self, backend):
        """For psycopg 3, ``_fetch_impl`` delegates to
        ``self._backend.fetch`` with the SQL translated.
        """
        if backend != "psycopg3":
            pytest.skip("psycopg 3-specific dispatch path")
        mock = _make_psycopg_backend_mock(fetch_return=[{"id": 1}])
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
        mgr.ensure_connected = AsyncMock()

        rows = asyncio.run(mgr.fetch("SELECT * FROM t WHERE id = $1", 1))

        assert len(rows) == 1
        mock.fetch.assert_awaited_once()
        mock.fetch.assert_called_once_with("SELECT * FROM t WHERE id = %s", 1)

    def test_fetchrow_dispatches_to_pool_for_asyncpg(self, backend):
        """For asyncpg, ``_fetchrow_impl`` reads through ``self.pool``."""
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetchrow = AsyncMock(return_value={"id": 1})
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
        mgr.ensure_connected = AsyncMock()

        row = asyncio.run(mgr.fetchrow("SELECT * FROM t WHERE id = $1", 1))

        assert row == {"id": 1}
        cm.fetchrow.assert_awaited_once()
        cm.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id = $1", 1)

    def test_fetchrow_dispatches_to_backend_for_psycopg3(self, backend):
        """For psycopg 3, ``_fetchrow_impl`` delegates to
        ``self._backend.fetchrow`` with the SQL translated.
        """
        if backend != "psycopg3":
            pytest.skip("psycopg 3-specific dispatch path")
        mock = _make_psycopg_backend_mock(fetchrow_return={"id": 1})
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
        mgr.ensure_connected = AsyncMock()

        row = asyncio.run(mgr.fetchrow("SELECT * FROM t WHERE id = $1", 1))

        assert row == {"id": 1}
        mock.fetchrow.assert_awaited_once()
        mock.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id = %s", 1)

    def test_fetchval_dispatches_to_pool_for_asyncpg(self, backend):
        """For asyncpg, ``_fetchval_impl`` reads through ``self.pool``."""
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=None)
        cm.fetchval = AsyncMock(return_value=42)
        pool.acquire = MagicMock(return_value=cm)
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
        mgr.ensure_connected = AsyncMock()

        val = asyncio.run(mgr.fetchval("SELECT count(*) FROM t"))

        assert val == 42
        cm.fetchval.assert_awaited_once()
        cm.fetchval.assert_called_once_with("SELECT count(*) FROM t")

    def test_fetchval_dispatches_to_backend_for_psycopg3(self, backend):
        """For psycopg 3, ``_fetchval_impl`` delegates to
        ``self._backend.fetchval`` (the SQL is unchanged because
        there are no ``$N`` placeholders, but the dispatcher
        still routes through the backend).
        """
        if backend != "psycopg3":
            pytest.skip("psycopg 3-specific dispatch path")
        mock = _make_psycopg_backend_mock(fetchval_return=42)
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
        mgr.ensure_connected = AsyncMock()

        val = asyncio.run(mgr.fetchval("SELECT count(*) FROM t"))

        assert val == 42
        mock.fetchval.assert_awaited_once()
        mock.fetchval.assert_called_once_with("SELECT count(*) FROM t")

    def test_translator_is_applied_for_psycopg3(self, backend):
        """``translate_asyncpg_to_psycopg()`` is invoked for psycopg 3.

        Verify the SQL passed to the backend has ``$N`` translated
        to ``%s`` and that no ``$`` characters remain in the
        placeholders.
        """
        if backend != "psycopg3":
            pytest.skip("psycopg 3-specific translator behavior")
        mock = _make_psycopg_backend_mock(execute_return="OK")
        mgr = _make_manager_for_backend("psycopg3", backend_mock=mock)
        mgr.ensure_connected = AsyncMock()

        asyncio.run(mgr.execute("SELECT $1, $2, $3", 1, 2, 3))

        called_sql = mock.execute.call_args[0][0]
        # The translated SQL must have %s, %s, %s — NOT $1, $2, $3
        assert "$" not in called_sql, (
            f"translator failed: $N placeholders remain in {called_sql!r}"
        )
        assert called_sql.count("%s") == 3, (
            f"expected 3 %s placeholders, got {called_sql!r}"
        )

    def test_translator_is_NOT_applied_for_asyncpg(self, backend):
        """For asyncpg, the SQL passes through unchanged.

        The ``$N`` placeholders must remain in the SQL because
        asyncpg natively supports them. The translator is only
        invoked for non-asyncpg backends.
        """
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool, cm = _fake_pool_with_execute(return_value="OK")
        mgr = _make_manager_for_backend("asyncpg", pool=pool)
        mgr.ensure_connected = AsyncMock()

        asyncio.run(mgr.execute("SELECT $1, $2", 1, 2))

        # pool.execute must be called with the original $1, $2 SQL
        call_args = cm.execute.call_args
        assert call_args[0][0] == "SELECT $1, $2", (
            f"translator must NOT be applied for asyncpg; got {call_args[0][0]!r}"
        )
        assert call_args[0][1] == 1
        assert call_args[0][2] == 2

    def test_dispatch_falls_through_when_backend_unset(self, backend):
        """When ``self._backend`` is unset (the bypass-__init__ path),
        the dispatcher falls through to the asyncpg path even for
        the psycopg 3 fixture value.

        The dispatcher uses ``getattr(self, "_backend", None)`` and
        treats ``None`` as "use the asyncpg path". This is the
        behavior the existing 82-test suite relies on, so a test
        that goes through ``__new__``-bypass without setting
        ``_backend`` must still take the asyncpg path.
        """
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific: unset _backend falls through")
        pool, cm = _fake_pool_with_execute(return_value="OK")
        # Bypass-__init__ path: do NOT set mgr._backend
        mgr = _make_manager_with_mock_pool(pool=pool)
        assert not hasattr(mgr, "_backend") or mgr._backend is None
        mgr.ensure_connected = AsyncMock()

        asyncio.run(mgr.execute("SELECT 1"))

        # Pool was used (not a backend) — the dispatcher fell through
        cm.execute.assert_awaited_once()

    def test_dispatch_falls_through_when_backend_name_is_asyncpg(self, backend):
        """When ``self._backend.name == "asyncpg"`` (e.g. the
        AsyncpgBackend instance is set as ``_backend``), the
        dispatcher takes the asyncpg path via the backend's
        pool — but for unit-test simplicity we use a pool mock
        and verify the dispatcher still goes through ``self.pool``
        when ``backend.name == "asyncpg"``.
        """
        if backend != "asyncpg":
            pytest.skip("asyncpg-specific dispatch path")
        pool, cm = _fake_pool_with_execute(return_value="OK")
        mgr = _make_manager_with_mock_pool(pool=pool)
        # Set _backend to something whose name is "asyncpg" — the
        # dispatcher's condition ``backend.name != "asyncpg"``
        # must be False, so the asyncpg pool path is taken.
        asyncpg_backend = MagicMock()
        asyncpg_backend.name = "asyncpg"
        mgr._backend = asyncpg_backend
        mgr.ensure_connected = AsyncMock()

        asyncio.run(mgr.execute("SELECT $1", 1))

        # Pool was used (not _backend.execute) — dispatcher chose
        # the asyncpg path because backend.name == "asyncpg"
        cm.execute.assert_awaited_once()
        asyncpg_backend.execute.assert_not_called()
