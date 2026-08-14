"""Unit tests for the psycopg 3 backend (issue #28).

These tests pin down the behavior of
:class:`common.db.backends.psycopg3.Psycopg3Backend` in isolation.
The dev env does NOT have a real Postgres server, so the tests use
``unittest.mock`` to stand in for the driver. Real-PG integration
testing is deferred to Phase 3 (issue #29).

What the tests cover
--------------------

* :class:`TestPsycopg3BackendBasics` — Protocol surface and identity
  (name, presence of every method the manager dispatches to).
* :class:`TestRowAdapter` — the ``_RowAdapter`` shim that lets psycopg 3
  rows behave like asyncpg's ``Record`` (positional + name indexing,
  ``.keys()`` / ``.values()`` / ``__iter__`` / ``__len__`` /
  ``__contains__``). Covers both the psycopg 3.1+ path (where
  ``Row.keys()`` exists) and the fallback for older 3.0 builds (where
  the adapter reads the private ``_index``).
* :class:`TestVectorEncoding` — the no-pgvector-package fallback that
  encodes ``List[float]`` as a Postgres ``vector`` literal the
  ``::vector`` cast can parse.
* :class:`TestPoolConfigTranslation` — verifies ``BackendConfig`` is
  mapped to the right ``psycopg_pool.AsyncConnectionPool`` kwargs
  (libpq naming, ``max_idle``, SSL params, ``application_name``,
  ``options='-c statement_timeout=NNNms'``).
* :class:`TestHealthCheck` — happy path returns ``{status, version, ...}``,
  error path returns ``{status: "unhealthy", error}``, and the
  disconnected path returns ``{status: "disconnected"}``.
* :class:`TestQueryExecution` — ``execute`` returns the statusmessage,
  ``fetch`` wraps rows in ``_RowAdapter``, ``fetchrow`` returns
  ``None`` when no rows match, ``fetchval`` returns ``row[0]``.
* :class:`TestPatterns` — the patterns helpers use ``%s``-style SQL
  and the string-encoded vector fallback, and shape their return
  values to match the asyncpg backend exactly.

The test contract for the manager (the existing 82 tests in
``test_postgres_sync.py``, ``test_pool_saturation.py``,
``test_statement_timeout.py``) is NOT exercised here — it is
unchanged and the parent agent will run it as part of the full
verification pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# psycopg / psycopg_pool are runtime deps of the backend under test.
# The CONTEXT guarantees they are installed in this env, but guard
# the imports so a stray bare ``pytest`` invocation on an asyncpg-only
# machine still produces a clean "module not available" skip rather
# than a hard import error at collection time.
psycopg = pytest.importorskip("psycopg")
psycopg_pool = pytest.importorskip("psycopg_pool")

from common.db.backends.base import BackendConfig  # noqa: E402
from common.db.backends.psycopg3 import (  # noqa: E402
    Psycopg3Backend,
    _RowAdapter,
    _encode_vector,
    _build_pool_kwargs,
    _INSERT_PATTERN_SQL,
    _FIND_SIMILAR_PATTERNS_SQL,
    _UPDATE_PATTERN_EMBEDDING_SQL,
    _GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> BackendConfig:
    """Build a BackendConfig with sensible defaults for tests.

    Each test passes overrides for the fields it cares about; the
    rest take the default below. Centralized so a future schema
    change (new field on ``BackendConfig``) only needs to update one
    place.
    """
    fields: Dict[str, Any] = dict(
        host="localhost",
        port=5432,
        database="testdb",
        user="u",
        password="p",
        min_size=0,
        max_size=5,
        application_name="test-app",
        search_path="public",
        statement_timeout_ms=30000,
        ssl_mode="disable",
        ssl_no_verify=False,
    )
    fields.update(overrides)
    return BackendConfig(**fields)


def _make_psycopg_row(data: Dict[str, Any]) -> MagicMock:
    """Build a mock psycopg ``Row`` that supports ``__getitem__`` and ``.keys()``.

    Mimics psycopg 3.1+ rows: ``row["col"]`` and ``row.keys()`` both
    work. The mock is reused by the row-adapter tests and the
    patterns-helper tests (which need a row that yields real values
    when the adapter does ``row["id"]``).
    """
    row = MagicMock(name="psycopg_row")
    row.__getitem__ = MagicMock(side_effect=lambda k: data[k])
    row.keys = MagicMock(return_value=list(data.keys()))
    return row


def _make_mock_pool(
    *,
    cursor_execute: Optional[AsyncMock] = None,
    cursor_fetchall: Optional[List[Any]] = None,
    cursor_fetchone: Any = None,
    cursor_statusmessage: str = "INSERT 0 1",
) -> MagicMock:
    """Build a mock ``psycopg_pool.AsyncConnectionPool``.

    The returned mock supports the chain the backend uses on every
    query method:

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(...)
                rows = await cur.fetchall()

    The individual cursor methods default to no-op / empty so a
    test only needs to override the pieces it asserts on.
    """
    mock_cursor = MagicMock(name="psycopg_cursor")
    mock_cursor.statusmessage = cursor_statusmessage
    mock_cursor.execute = cursor_execute or AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=cursor_fetchall or [])
    # ``cursor_fetchone`` may be (a) None, (b) a concrete value to
    # return every call, (c) an AsyncMock whose side_effect produces
    # different values per call, or (d) a list to use as side_effect.
    # The unconditional ``AsyncMock(return_value=cursor_fetchone)``
    # shape swallowed case (c) — the inner AsyncMock was returned
    # verbatim, never called, so its side_effect never fired.
    if cursor_fetchone is None:
        mock_cursor.fetchone = AsyncMock(return_value=None)
    elif isinstance(cursor_fetchone, (AsyncMock, list)):
        # list → side_effect (per-call distinct values); AsyncMock →
        # use the supplied mock as-is so its side_effect fires.
        if isinstance(cursor_fetchone, list):
            mock_cursor.fetchone = AsyncMock(side_effect=cursor_fetchone)
        else:
            mock_cursor.fetchone = cursor_fetchone
    else:
        mock_cursor.fetchone = AsyncMock(return_value=cursor_fetchone)
    mock_cursor.executemany = AsyncMock()

    cursor_cm = MagicMock(name="cursor_cm")
    cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)

    mock_conn = MagicMock(name="psycopg_conn")
    mock_conn.cursor = MagicMock(return_value=cursor_cm)

    conn_cm = MagicMock(name="conn_cm")
    conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock(name="psycopg_pool")
    mock_pool.connection = MagicMock(return_value=conn_cm)
    return mock_pool


# ---------------------------------------------------------------------------
# Basics: name + Protocol surface
# ---------------------------------------------------------------------------


class TestPsycopg3BackendBasics:
    """Identity and Protocol-conformance smoke tests."""

    def test_name_property(self):
        """``name`` returns ``"psycopg3"`` (the manager uses this for dispatch).

        ``name`` is a property on the Protocol; class-level access returns
        the property descriptor itself, not the value. The manager reads
        it from an instance, so that's what the test exercises.
        """
        assert Psycopg3Backend().name == "psycopg3"

    def test_protocol_conformance(self):
        """Every Protocol method exists on the backend.

        The manager's basic query dispatch (``_execute_impl`` /
        ``_fetch_impl`` / etc.) and the patterns helpers all reach
        into specific methods; if any is missing the manager would
        AttributeError on the first call. This test pins the
        surface so a future rename or accidental deletion is caught
        at the test layer.
        """
        backend = Psycopg3Backend()
        expected_methods = [
            "connect",
            "disconnect",
            "health_check",
            "acquire",
            "execute",
            "fetch",
            "fetchrow",
            "fetchval",
            "executemany",
            "insert_pattern_with_embedding",
            "find_similar_patterns",
            "update_pattern_embedding",
            "get_patterns_without_embeddings",
        ]
        for name in expected_methods:
            assert hasattr(backend, name), f"missing method: {name}"
            assert callable(getattr(backend, name)), f"not callable: {name}"

    def test_pool_attribute_is_public_for_manager_compatibility(self):
        """``pool`` is a public attribute (not ``_pool``).

        Regression test for the py-daroja-libs-stress.yml nightly
        job's psycopg 3 parametrization, which was failing with
        ``AttributeError: 'Psycopg3Backend' object has no attribute
        'pool'``. Root cause: ``DatabaseManager.connect`` does
        ``self.pool = self._backend.pool`` so the manager can read
        through ``self.pool`` uniformly across backends. The asyncpg
        backend exposes ``self.pool``; the psycopg 3 backend used
        ``self._pool`` (private), so the assignment raised
        ``AttributeError`` the first time the manager ran against
        a real PG with ``backend="psycopg3"``.

        Fix: rename ``_pool`` → ``pool`` in ``Psycopg3Backend`` so
        both backends expose the same public attribute name. This
        test pins down:

          1. The attribute exists and is publicly accessible.
          2. It starts as ``None`` (set by ``connect()``).
          3. It can be reassigned (the manager's connect path
             does ``self.pool = self._backend.pool``).

        If a future refactor reintroduces ``_pool`` (or any other
        private name), this test catches it before the stress
        CI does.
        """
        backend = Psycopg3Backend()
        # Public attribute must exist.
        assert hasattr(backend, "pool"), (
            "Psycopg3Backend must expose a public `pool` attribute "
            "for DatabaseManager.connect to sync it via "
            "`self.pool = self._backend.pool`. The previous private "
            "name `_pool` caused AttributeError in the nightly "
            "stress test's psycopg 3 parametrization."
        )
        assert backend.pool is None
        # Reassignable. The manager's connect path assigns to it.
        sentinel = object()
        backend.pool = sentinel
        assert backend.pool is sentinel


# ---------------------------------------------------------------------------
# Row adapter
# ---------------------------------------------------------------------------


class TestRowAdapter:
    """The ``_RowAdapter`` shim normalizes psycopg 3's ``Row`` shape."""

    def test_str_indexing_delegates_to_underlying_row(self):
        """``row["col"]`` returns the underlying row's value."""
        inner = _make_psycopg_row({"id": 42, "name": "alice"})
        adapter = _RowAdapter(inner)
        assert adapter["id"] == 42
        assert adapter["name"] == "alice"

    def test_int_indexing_delegates_to_underlying_row(self):
        """``row[0]`` works for rows that support positional access.

        psycopg 3.1+ rows accept ``row[0]`` (positional) the same way
        asyncpg's ``Record`` does; we just delegate. The mock row
        here supports it via its side_effect that maps the int to
        the value at the corresponding key.
        """
        data = {"id": 1, "name": "x"}
        inner = _make_psycopg_row(data)
        # Add positional access to the mock: row[0] -> list(data.values())[0]
        inner.__getitem__ = MagicMock(
            side_effect=lambda k: (
                list(data.values())[k] if isinstance(k, int) else data[k]
            )
        )
        adapter = _RowAdapter(inner)
        assert adapter[0] == 1
        assert adapter[1] == "x"

    def test_keys_returns_underlying_keys(self):
        """``keys()`` delegates to the wrapped row's ``keys()`` method."""
        inner = _make_psycopg_row({"a": 1, "b": 2, "c": 3})
        adapter = _RowAdapter(inner)
        assert adapter.keys() == ["a", "b", "c"]

    def test_keys_falls_back_to_index_for_old_psycopg(self):
        """Older psycopg 3 builds without ``Row.keys()`` use ``_index``."""
        inner = MagicMock(name="old_psycopg_row")
        # No ``keys()`` method.
        del inner.keys
        # ``_index`` is the private dict psycopg uses internally.
        inner._index = {"a": 0, "b": 1}
        # ``__getitem__`` still works.
        inner.__getitem__ = MagicMock(side_effect=lambda k: {"a": 1, "b": 2}[k])
        adapter = _RowAdapter(inner)
        # ``hasattr(self._row, "keys")`` is False (we deleted it);
        # the adapter should fall back to ``_index``.
        assert set(adapter.keys()) == {"a", "b"}

    def test_values_iterates_via_keys(self):
        """``values()`` returns the values in column order."""
        inner = _make_psycopg_row({"a": 1, "b": 2, "c": 3})
        adapter = _RowAdapter(inner)
        assert adapter.values() == [1, 2, 3]

    def test_iter_yields_column_names(self):
        """``iter(row)`` iterates over column names (asyncpg parity)."""
        inner = _make_psycopg_row({"a": 1, "b": 2})
        adapter = _RowAdapter(inner)
        assert list(adapter) == ["a", "b"]

    def test_len_returns_column_count(self):
        """``len(row)`` returns the number of columns."""
        inner = _make_psycopg_row({"a": 1, "b": 2, "c": 3, "d": 4})
        adapter = _RowAdapter(inner)
        assert len(adapter) == 4

    def test_contains_checks_membership(self):
        """``"col" in row`` works like asyncpg's ``Record``."""
        inner = _make_psycopg_row({"a": 1, "b": 2})
        adapter = _RowAdapter(inner)
        assert "a" in adapter
        assert "b" in adapter
        assert "c" not in adapter


# ---------------------------------------------------------------------------
# Vector encoding (the no-pgvector-package fallback)
# ---------------------------------------------------------------------------


class TestVectorEncoding:
    """The ``_encode_vector`` helper produces a Postgres ``vector`` literal."""

    def test_encode_vector_none(self):
        """``None`` input passes through as ``None`` (the SQL cast becomes NULL)."""
        assert _encode_vector(None) is None

    def test_encode_vector_empty(self):
        """An empty list encodes to ``"[]"`` (valid Postgres vector literal)."""
        assert _encode_vector([]) == "[]"

    def test_encode_vector_floats(self):
        """A list of floats encodes to the bracket form with no spaces."""
        assert _encode_vector([1.0, 2.5, 3.0]) == "[1.0,2.5,3.0]"

    def test_encode_vector_ints_become_floats(self):
        """Int elements are coerced to float so the ``::vector`` cast accepts them.

        Postgres' vector parser is strict: ``'[1,2,3]'`` is rejected;
        ``'[1.0,2.0,3.0]'`` is accepted. The ``str(float(x))`` form
        guarantees the latter.
        """
        assert _encode_vector([1, 2, 3]) == "[1.0,2.0,3.0]"

    def test_encode_vector_mixed_types(self):
        """Mixed int/float elements all get coerced to float."""
        assert _encode_vector([1, 2.5, 3]) == "[1.0,2.5,3.0]"


# ---------------------------------------------------------------------------
# Pool config translation
# ---------------------------------------------------------------------------


class TestPoolConfigTranslation:
    """``BackendConfig`` → ``psycopg_pool.AsyncConnectionPool`` kwargs."""

    def test_basic_config_translation(self):
        """Connection fields and pool sizing land in the right kwargs."""
        config = _make_config(
            host="myhost",
            port=6432,
            database="mydb",
            user="myuser",
            password="mypass",
            min_size=2,
            max_size=20,
            application_name="myapp",
        )
        kwargs = _build_pool_kwargs(config)
        # Connection params (libpq naming).
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 6432
        assert kwargs["dbname"] == "mydb"  # NOTE: renamed from ``database``
        assert kwargs["user"] == "myuser"
        assert kwargs["password"] == "mypass"
        # Pool sizing.
        assert kwargs["min_size"] == 2
        assert kwargs["max_size"] == 20
        # Idle lifetime — psycopg_pool uses ``max_idle`` (renamed from
        # asyncpg's ``max_inactive_connection_lifetime``).
        assert kwargs["max_idle"] == 300
        # ``open=False`` lets the caller control the connect-vs-open sequence.
        assert kwargs["open"] is False
        # ``configure`` is attached separately by the backend's connect();
        # ``_build_pool_kwargs`` does not include it.
        assert "configure" not in kwargs
        # SSL default for ``ssl_mode='disable'``.
        assert kwargs["sslmode"] == "disable"

    def test_statement_timeout_in_options(self):
        """The ``statement_timeout`` GUC is set server-side via libpq's ``options``."""
        config = _make_config(statement_timeout_ms=30000)
        kwargs = _build_pool_kwargs(config)
        assert kwargs["options"] == "-c statement_timeout=30000ms"

    def test_statement_timeout_zero_omits_options(self):
        """A zero / falsy ``statement_timeout_ms`` does not set ``options``."""
        config = _make_config(statement_timeout_ms=0)
        kwargs = _build_pool_kwargs(config)
        assert "options" not in kwargs

    def test_application_name_passed_through(self):
        """``application_name`` is forwarded as a connection kwarg."""
        config = _make_config(application_name="my-service")
        kwargs = _build_pool_kwargs(config)
        assert kwargs["application_name"] == "my-service"

    def test_ssl_disable(self):
        """``ssl_mode='disable'`` → ``sslmode='disable'`` in kwargs (no sslrootcert)."""
        config = _make_config(ssl_mode="disable", ssl_no_verify=False)
        kwargs = _build_pool_kwargs(config)
        assert kwargs["sslmode"] == "disable"
        assert "sslrootcert" not in kwargs

    def test_ssl_require_with_verify(self):
        """``ssl_mode='require'`` + ``ssl_no_verify=False`` → ``sslmode='require'``.

        No ``sslrootcert`` (full verification against the system trust
        store).
        """
        config = _make_config(ssl_mode="require", ssl_no_verify=False)
        kwargs = _build_pool_kwargs(config)
        assert kwargs["sslmode"] == "require"
        assert "sslrootcert" not in kwargs

    def test_ssl_require_no_verify_uses_empty_sslrootcert(self):
        """``ssl_mode='require'`` + ``ssl_no_verify=True`` → ``sslrootcert=''``.

        The libpq-documented way to say "encrypt-don't-verify" without
        building a Python ``ssl.SSLContext``.
        """
        config = _make_config(ssl_mode="require", ssl_no_verify=True)
        kwargs = _build_pool_kwargs(config)
        assert kwargs["sslmode"] == "require"
        assert kwargs["sslrootcert"] == ""

    @pytest.mark.asyncio
    async def test_psycopg3_backend_calls_asyncconnectionpool_with_kwargs(self):
        """``backend.connect()`` instantiates the pool with the translated kwargs.

        End-to-end check: the kwargs built by ``_build_pool_kwargs``
        are what reach ``psycopg_pool.AsyncConnectionPool(**kwargs)``.
        """
        config = _make_config(
            host="h",
            port=1,
            database="d",
            user="u",
            password="p",
            min_size=0,
            max_size=3,
            application_name="app",
            statement_timeout_ms=1000,
            ssl_mode="disable",
        )
        backend = Psycopg3Backend()
        with patch(
            "common.db.backends.psycopg3.psycopg_pool.AsyncConnectionPool"
        ) as MockPool:
            instance = MagicMock()
            instance.open = AsyncMock()
            # Set up a minimal connection for the pgvector probe.
            cur = MagicMock()
            cur.execute = AsyncMock()
            cur.fetchone = AsyncMock(return_value=None)
            cur_cm = MagicMock()
            cur_cm.__aenter__ = AsyncMock(return_value=cur)
            cur_cm.__aexit__ = AsyncMock(return_value=None)
            conn = MagicMock()
            conn.cursor = MagicMock(return_value=cur_cm)
            conn_cm = MagicMock()
            conn_cm.__aenter__ = AsyncMock(return_value=conn)
            conn_cm.__aexit__ = AsyncMock(return_value=None)
            instance.connection = MagicMock(return_value=conn_cm)
            MockPool.return_value = instance

            await backend.connect(config)

        # Inspect the kwargs the pool was constructed with.
        call_kwargs = MockPool.call_args.kwargs
        assert call_kwargs["host"] == "h"
        assert call_kwargs["port"] == 1
        assert call_kwargs["dbname"] == "d"
        assert call_kwargs["user"] == "u"
        assert call_kwargs["password"] == "p"
        assert call_kwargs["min_size"] == 0
        assert call_kwargs["max_size"] == 3
        assert call_kwargs["max_idle"] == 300
        assert call_kwargs["application_name"] == "app"
        assert call_kwargs["options"] == "-c statement_timeout=1000ms"
        assert call_kwargs["sslmode"] == "disable"
        # ``configure`` is attached (it's the per-connection callback).
        assert "configure" in call_kwargs
        assert callable(call_kwargs["configure"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """``health_check()`` returns a uniform status dict."""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self):
        """Happy path: status='healthy' with version + pgvector version + host/db."""
        backend = Psycopg3Backend()
        backend._config = _make_config(host="h", database="d")
        # ``side_effect`` on the AsyncMock itself makes each call return
        # the next tuple. The previous shape passed a MagicMock-with-
        # side_effect as ``return_value``, which made ``fetchone`` always
        # return that MagicMock (not the tuples) and the backend then
        # got ``MagicMock.__getitem__()`` instead of the string.
        backend.pool = _make_mock_pool(
            cursor_fetchone=AsyncMock(
                side_effect=[
                    ("PostgreSQL 16.0 on x86_64",),  # version()
                    ("0.7.0",),  # pgvector extversion
                ]
            )
        )
        result = await backend.health_check()
        assert result["status"] == "healthy"
        assert result["version"] == "PostgreSQL 16.0 on x86_64"
        assert result["pgvector_version"] == "0.7.0"
        assert result["host"] == "h"
        assert result["database"] == "d"

    @pytest.mark.asyncio
    async def test_health_check_returns_unhealthy_on_error(self):
        """If the probe raises, the result is ``{status: 'unhealthy', error}``."""
        backend = Psycopg3Backend()
        mock_pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
        cm.__aexit__ = AsyncMock(return_value=None)
        mock_pool.connection = MagicMock(return_value=cm)
        backend.pool = mock_pool
        backend._config = _make_config()

        result = await backend.health_check()
        assert result["status"] == "unhealthy"
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_health_check_returns_disconnected_when_pool_is_none(self):
        """Without an open pool, the result is ``{status: 'disconnected'}``."""
        backend = Psycopg3Backend()
        assert backend.pool is None
        result = await backend.health_check()
        assert result["status"] == "disconnected"
        assert "No active connection pool" in result["message"]


# ---------------------------------------------------------------------------
# Basic query methods
# ---------------------------------------------------------------------------


class TestQueryExecution:
    """``execute`` / ``fetch`` / ``fetchrow`` / ``fetchval`` / ``executemany``."""

    @pytest.mark.asyncio
    async def test_execute_returns_statusmessage(self):
        """``execute()`` returns ``cursor.statusmessage`` (e.g. ``"INSERT 0 1"``)."""
        backend = Psycopg3Backend()
        mock_cursor = MagicMock()
        mock_cursor.statusmessage = "INSERT 0 7"
        mock_cursor.execute = AsyncMock()
        # Reuse the helper but override the cursor mock.
        mock_pool = MagicMock()
        cursor_cm = MagicMock()
        cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_cm.__aexit__ = AsyncMock(return_value=None)
        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=cursor_cm)
        conn_cm = MagicMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pool.connection = MagicMock(return_value=conn_cm)
        backend.pool = mock_pool

        result = await backend.execute("INSERT INTO t VALUES (%s)", 1)
        assert result == "INSERT 0 7"
        mock_cursor.execute.assert_awaited_once_with("INSERT INTO t VALUES (%s)", (1,))

    @pytest.mark.asyncio
    async def test_fetch_returns_list_of_adapters(self):
        """``fetch()`` returns ``[_RowAdapter(...)]`` for each row."""
        backend = Psycopg3Backend()
        row1 = _make_psycopg_row({"id": 1, "name": "a"})
        row2 = _make_psycopg_row({"id": 2, "name": "b"})
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[row1, row2])
        _install_mock_pool_with_cursor(backend, mock_cursor)

        rows = await backend.fetch("SELECT * FROM t")
        assert len(rows) == 2
        assert all(isinstance(r, _RowAdapter) for r in rows)
        # The adapter delegates to the underlying row.
        assert rows[0]["id"] == 1
        assert rows[1]["name"] == "b"

    @pytest.mark.asyncio
    async def test_fetchrow_returns_adapter_when_row_present(self):
        """``fetchrow()`` wraps the row in ``_RowAdapter`` when one is returned."""
        backend = Psycopg3Backend()
        row = _make_psycopg_row({"id": 1, "name": "a"})
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=row)
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.fetchrow("SELECT * FROM t WHERE id = %s", 1)
        assert isinstance(result, _RowAdapter)
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_fetchrow_returns_none_when_no_row(self):
        """``fetchrow()`` returns ``None`` when the cursor's ``fetchone`` returns ``None``."""
        backend = Psycopg3Backend()
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.fetchrow("SELECT * FROM t WHERE id = %s", -1)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetchval_returns_scalar(self):
        """``fetchval()`` returns ``row[0]`` (the first column of the first row)."""
        backend = Psycopg3Backend()
        # Use a real indexable object. A MagicMock with ``row[0] = 42``
        # set as an attribute does NOT make ``row[0]`` return 42 in
        # Python's index protocol — MagicMock's __getitem__ is auto-
        # mocked to return a child MagicMock. A tuple works directly.
        row = (42,)
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=row)
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.fetchval("SELECT count(*) FROM t")
        assert result == 42

    @pytest.mark.asyncio
    async def test_fetchval_returns_none_when_no_row(self):
        """``fetchval()`` returns ``None`` when the cursor's ``fetchone`` returns ``None``."""
        backend = Psycopg3Backend()
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.fetchval("SELECT count(*) FROM t WHERE 0")
        assert result is None

    @pytest.mark.asyncio
    async def test_executemany_passes_args_to_cursor(self):
        """``executemany()`` forwards the query and args list to ``cur.executemany``."""
        backend = Psycopg3Backend()
        mock_cursor = MagicMock()
        mock_cursor.executemany = AsyncMock()
        _install_mock_pool_with_cursor(backend, mock_cursor)

        await backend.executemany(
            "INSERT INTO t VALUES (%s, %s)",
            [(1, "a"), (2, "b"), (3, "c")],
        )
        mock_cursor.executemany.assert_awaited_once_with(
            "INSERT INTO t VALUES (%s, %s)",
            [(1, "a"), (2, "b"), (3, "c")],
        )

    def test_acquire_raises_when_disconnected(self):
        """``acquire()`` raises ``RuntimeError`` when the pool is not connected."""
        backend = Psycopg3Backend()
        assert backend.pool is None
        with pytest.raises(RuntimeError, match="Database not connected"):
            backend.acquire()

    def test_acquire_returns_pool_connection_context(self):
        """``acquire()`` returns the result of ``self._pool.connection()`` (an ACM)."""
        backend = Psycopg3Backend()
        cm = MagicMock(name="acm")
        mock_pool = MagicMock()
        mock_pool.connection = MagicMock(return_value=cm)
        backend.pool = mock_pool
        result = backend.acquire()
        assert result is cm
        mock_pool.connection.assert_called_once_with()


def _install_mock_pool_with_cursor(
    backend: Psycopg3Backend, mock_cursor: MagicMock
) -> MagicMock:
    """Install a pool on ``backend`` that yields the given cursor on every query."""
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=cursor_cm)
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=conn_cm)
    backend.pool = mock_pool
    return mock_pool


# ---------------------------------------------------------------------------
# Patterns helpers
# ---------------------------------------------------------------------------


class TestPatterns:
    """The patterns helpers use ``%s``-style SQL and the string-encoded vector fallback."""

    @pytest.mark.asyncio
    async def test_insert_pattern_uses_s_style_sql(self):
        """The SQL constants used by the patterns helpers are ``%s``-style.

        This guards against an accidental revert to asyncpg's ``$1``
        style (which would break the manager's translator at the
        boundary — patterns helpers bypass the translator because
        they go through the backend's own ``fetchval``).
        """
        assert "%s" in _INSERT_PATTERN_SQL
        assert "$1" not in _INSERT_PATTERN_SQL
        assert "$2" not in _INSERT_PATTERN_SQL

    @pytest.mark.asyncio
    async def test_find_similar_patterns_uses_s_style_sql(self):
        """The ``_FIND_SIMILAR_PATTERNS_SQL`` constant is ``%s``-style."""
        assert "%s" in _FIND_SIMILAR_PATTERNS_SQL
        assert "$1" not in _FIND_SIMILAR_PATTERNS_SQL

    @pytest.mark.asyncio
    async def test_update_pattern_embedding_uses_s_style_sql(self):
        """The ``_UPDATE_PATTERN_EMBEDDING_SQL`` constant is ``%s``-style."""
        assert "%s" in _UPDATE_PATTERN_EMBEDDING_SQL
        assert "$1" not in _UPDATE_PATTERN_EMBEDDING_SQL

    @pytest.mark.asyncio
    async def test_get_patterns_without_embeddings_uses_s_style_sql(self):
        """The ``_GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL`` constant is ``%s``-style."""
        assert "%s" in _GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL
        assert "$1" not in _GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL

    @pytest.mark.asyncio
    async def test_insert_pattern_with_embedding_calls_fetchval(self):
        """``insert_pattern_with_embedding`` runs through the backend's ``fetchval``."""
        backend = Psycopg3Backend()
        # Make ``fetchval`` a spy that records its (sql, args) call.
        backend.fetchval = AsyncMock()  # type: ignore[assignment]

        await backend.insert_pattern_with_embedding(
            repo_id=1,
            name="my-pattern",
            description="desc",
            context="ctx",
            embedding=[0.1, 0.2, 0.3],
        )
        # fetchval was called once.
        backend.fetchval.assert_awaited_once()
        # The args include the string-encoded vector (no pgvector
        # package, so we fall back to the literal form).
        call_args = backend.fetchval.await_args
        sql, *args = call_args.args
        assert sql == _INSERT_PATTERN_SQL
        # The last positional arg is the encoded vector literal.
        assert args[-1] == "[0.1,0.2,0.3]"

    @pytest.mark.asyncio
    async def test_insert_pattern_with_embedding_handles_none_embedding(self):
        """A ``None`` embedding is passed through as ``None`` (NULL in SQL)."""
        backend = Psycopg3Backend()
        backend.fetchval = AsyncMock()  # type: ignore[assignment]

        await backend.insert_pattern_with_embedding(
            repo_id=1,
            name="p",
            description="d",
            context="c",
            embedding=None,
        )
        call_args = backend.fetchval.await_args
        args = call_args.args[1:]
        # The last positional arg is None.
        assert args[-1] is None

    @pytest.mark.asyncio
    async def test_find_similar_patterns_returns_dicts(self):
        """``find_similar_patterns`` returns a list of dicts with the asyncpg contract."""
        backend = Psycopg3Backend()
        # Build mock rows that match the SELECT projection.
        row1 = _make_psycopg_row(
            {
                "id": 1,
                "name": "a",
                "description": "da",
                "context": "ca",
                "repo_id": 10,
                "repo_name": "r1",
                "similarity": 0.9,
            }
        )
        row2 = _make_psycopg_row(
            {
                "id": 2,
                "name": "b",
                "description": "db",
                "context": "cb",
                "repo_id": 20,
                "repo_name": "r2",
                "similarity": 0.85,
            }
        )
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[row1, row2])
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.find_similar_patterns(
            embedding=[0.1, 0.2, 0.3], limit=5, threshold=0.8
        )
        assert len(result) == 2
        # Same shape as the asyncpg backend's return.
        for entry in result:
            assert set(entry.keys()) == {
                "id",
                "name",
                "description",
                "context",
                "repo_id",
                "repo_name",
                "similarity",
            }
            assert isinstance(entry["similarity"], float)
        assert result[0]["id"] == 1
        assert result[0]["name"] == "a"
        assert result[0]["similarity"] == 0.9
        assert result[1]["id"] == 2

    @pytest.mark.asyncio
    async def test_find_similar_patterns_encodes_vector(self):
        """The embedding is passed to the cursor as a string-encoded vector."""
        backend = Psycopg3Backend()
        # The vector should appear in the execute() call args.
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        _install_mock_pool_with_cursor(backend, mock_cursor)

        await backend.find_similar_patterns(embedding=[0.1, 0.2, 0.3])

        mock_cursor.execute.assert_awaited_once()
        call_args = mock_cursor.execute.await_args
        sql, params = call_args.args

        # SQL is the module-level constant.
        assert sql == _FIND_SIMILAR_PATTERNS_SQL
        # The first three params are the encoded vector (used in the
        # three <=> expressions in the SQL).
        assert params[0] == "[0.1,0.2,0.3]"
        assert params[1] == "[0.1,0.2,0.3]"
        # (params[2] is threshold = 0.8)
        assert params[2] == 0.8
        # (params[3] and params[4] are exclude_repo_id, repeated twice)
        assert params[3] is None
        assert params[4] is None
        # params[5] is the encoded vector for the ORDER BY <=>.
        assert params[5] == "[0.1,0.2,0.3]"
        # params[6] is the default limit (10).
        assert params[6] == 10

    @pytest.mark.asyncio
    async def test_update_pattern_embedding_passes_encoded_vector(self):
        """``update_pattern_embedding`` encodes the vector as a string."""
        backend = Psycopg3Backend()
        backend.execute = AsyncMock()  # type: ignore[assignment]

        await backend.update_pattern_embedding(pattern_id=5, embedding=[1.0, 2.0, 3.0])
        backend.execute.assert_awaited_once()
        call_args = backend.execute.await_args
        sql, *params = call_args.args
        assert sql == _UPDATE_PATTERN_EMBEDDING_SQL
        assert params[0] == "[1.0,2.0,3.0]"
        assert params[1] == 5

    @pytest.mark.asyncio
    async def test_get_patterns_without_embeddings_returns_dicts(self):
        """``get_patterns_without_embeddings`` returns a list of dicts (asyncpg shape)."""
        backend = Psycopg3Backend()
        row = _make_psycopg_row(
            {
                "id": 1,
                "name": "a",
                "description": "d",
                "context": "c",
                "repo_id": 10,
                "repo_name": "r1",
            }
        )
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[row])
        _install_mock_pool_with_cursor(backend, mock_cursor)

        result = await backend.get_patterns_without_embeddings(limit=50)
        assert len(result) == 1
        entry = result[0]
        # No ``similarity`` key in the no-embedding path (different SQL).
        assert "similarity" not in entry
        assert entry["id"] == 1
        assert entry["name"] == "a"
        assert entry["description"] == "d"
        assert entry["context"] == "c"
        assert entry["repo_id"] == 10
        assert entry["repo_name"] == "r1"
