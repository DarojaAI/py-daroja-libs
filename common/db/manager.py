"""DatabaseManager — backend-agnostic facade for PostgreSQL.

This is the public API every downstream repo uses. It owns:

  * The dedicated event loop for the sync facade (issues #7/#9). The
    asyncpg pool is bound to whichever loop is running when the pool
    is created; if we created a fresh loop per ``_run_sync`` call,
    asyncpg's internal primitives become stale relative to the new
    loop and ``CancelledError`` cleanup dereferences primitives bound
    to the OLD loop, leading to SIGSEGV in the C protocol code. The
    fix is one stable loop in a daemon thread for the lifetime of the
    manager.

  * Retry + cancellation safety (issue #7). Transient asyncpg errors
    (``ConnectionDoesNotExistError``, ``InterfaceError``) are retried
    on the same pool — we do NOT call ``disconnect()`` inside the
    retry loop, because that races with concurrent acquirers and
    re-introduces the original segfault. On ``CancelledError`` we set
    a deferred-reset flag; the next non-cancelled ``ensure_connected``
    observes it and rebuilds the pool cleanly.

  * Pool saturation metrics (issue #687). ``_PoolStats`` accumulates
    acquire wait times and failure counts; ``acquire()`` wraps the
    backend's pool acquire with timing.

  * The sync facade (``execute_sync``, ``fetch_sync``, etc.) and the
    singleton helpers (``get_db``, ``init_db``, ``close_db``,
    ``init_db_sync``, ``close_db_sync``).

What it does NOT own:

  * Driver-specific SQL and pool creation — those live on the
    ``DatabaseBackend`` (common.db.backends.base). The manager
    constructs the backend in ``__init__`` and delegates
    ``connect``/``disconnect`` to it. For Phase 1 the only backend
    is ``AsyncpgBackend``; ``Psycopg3Backend`` is issue #28.

Public API compatibility:

  The old ``common.db.postgres`` module re-exports everything from
  here, so existing ``from common.db.postgres import DatabaseManager``
  imports keep working without changes in any downstream repo.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from common.db.backends.asyncpg import AsyncpgBackend
from common.db.backends.base import BackendConfig

# Phase 1 stub: subagent A is writing the real translator in
# common.db.query_translator in parallel. If it's not importable yet
# (e.g. mid-refactor), fall back to the identity function so this
# module still loads. Phase 2 will wire ``_run_sync`` to actually
# invoke the translator when ``self._backend.name == "psycopg3"``.
try:
    from common.db.query_translator import translate_asyncpg_to_psycopg

    _TRANSLATOR_AVAILABLE = True
except ImportError:  # pragma: no cover - Phase 1 fallback

    def translate_asyncpg_to_psycopg(sql: str) -> str:  # type: ignore[no-redef]
        return sql

    _TRANSLATOR_AVAILABLE = False


# Phase 2 (issue #28): psycopg 3 backend. Lazily imported so a project
# that doesn't have psycopg / psycopg_pool installed (or runs an
# older build) can still import this module and use asyncpg. When
# the import fails the registry simply omits the "psycopg3" entry
# and ``DatabaseManager(backend="psycopg3", ...)`` raises a clear
# ValueError in ``__init__`` (line 313-317 below).
try:
    from common.db.backends.psycopg3 import Psycopg3Backend

    _PSYCOPG3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without psycopg
    Psycopg3Backend = None  # type: ignore[misc,assignment]
    _PSYCOPG3_AVAILABLE = False

logger = logging.getLogger(__name__)

# Backend name -> backend class. Phase 1 shipped asyncpg; Phase 2
# (issue #28) added the psycopg 3 backend. Importing the classes
# (not the modules) here would create a circular dependency with
# backends/__init__, so we import from the concrete modules.
#
# ``psycopg3`` is only registered when the lazy import above
# succeeded; otherwise constructing a manager with
# ``backend="psycopg3"`` raises the standard "Unknown backend"
# ValueError from ``__init__``.
_BACKEND_REGISTRY: Dict[str, type] = {
    "asyncpg": AsyncpgBackend,
}
if _PSYCOPG3_AVAILABLE:
    _BACKEND_REGISTRY["psycopg3"] = Psycopg3Backend

# ---------------------------------------------------------------------------
# Pool saturation metrics (issue #687)
# ---------------------------------------------------------------------------


class _PoolStats:
    """In-memory pool saturation metrics for observability (issue #687).

    Tracks acquire counts, wait times, and failures so callers can
    detect pool starvation before it manifests as user-visible slowness.
    The samples ring buffer is bounded to keep memory predictable in
    long-running processes.
    """

    # Number of recent wait-time samples kept for percentile estimation.
    # 1024 is enough for stable p99 over 5-minute windows at typical
    # QPS without unbounded growth.
    _SAMPLE_CAP = 1024

    def __init__(self) -> None:
        self.acquire_total: int = 0
        self.acquire_failed_total: int = 0
        self.acquire_wait_ms_total: float = 0.0
        # Bounded ring buffer of recent wait times (ms) for percentile
        # estimation. deque(maxlen=...) is O(1) append and discards old
        # entries automatically.
        self.recent_wait_ms: deque = deque(maxlen=self._SAMPLE_CAP)

    def record_acquire(self, wait_ms: float) -> None:
        self.acquire_total += 1
        self.acquire_wait_ms_total += wait_ms
        self.recent_wait_ms.append(wait_ms)

    def record_acquire_failure(self) -> None:
        self.acquire_failed_total += 1

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary suitable for health-check output."""
        waits = sorted(self.recent_wait_ms) if self.recent_wait_ms else []
        n = len(waits)
        if n == 0:
            p50 = p95 = p99 = 0.0
        else:
            p50 = waits[max(0, int(n * 0.50) - 1)]
            p95 = waits[max(0, int(n * 0.95) - 1)]
            p99 = waits[max(0, int(n * 0.99) - 1)]
        return {
            "acquire_total": self.acquire_total,
            "acquire_failed_total": self.acquire_failed_total,
            "acquire_wait_ms_avg": (
                self.acquire_wait_ms_total / self.acquire_total
                if self.acquire_total
                else 0.0
            ),
            "acquire_wait_ms_p50": p50,
            "acquire_wait_ms_p95": p95,
            "acquire_wait_ms_p99": p99,
            "sample_count": n,
        }


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages PostgreSQL database connections with pgvector support.

    The manager is a thin facade over a ``DatabaseBackend`` instance.
    It owns the event loop, retry/cancellation logic, and pool
    saturation metrics; the backend owns driver-specific code (pool
    creation, SQL, connection semantics).
    """

    def __init__(
        self,
        *,
        backend: str = "asyncpg",
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        application_name: Optional[str] = None,
        search_path: Optional[str] = None,
    ):
        """
        Initialize database manager.

        Args:
            backend: Backend driver name. ``"asyncpg"`` (default) or
                ``"psycopg3"`` (Phase 2). Unknown names raise
                ``ValueError`` so a typo fails loud at startup, not on
                the first query.
            host: PostgreSQL host (defaults to POSTGRES_HOST env var).
            port: PostgreSQL port (defaults to POSTGRES_PORT env var).
            database: Database name (defaults to POSTGRES_DB env var).
            user: Database user (defaults to POSTGRES_USER env var).
            password: Database password (defaults to POSTGRES_PASSWORD env var).
            min_size: Minimum pool size.
            max_size: Maximum pool size.
            application_name: Connection application_name (defaults to POSTGRES_APP_NAME env var).
            search_path: Schema search path (defaults to POSTGRES_SEARCH_PATH env var).
        """
        # ------------------------------------------------------------------
        # Connection params (read from kwargs or POSTGRES_* env vars).
        # ------------------------------------------------------------------
        self.host = (host or os.getenv("POSTGRES_HOST", "localhost")).strip()
        # port is typed `Optional[int]`; coerce str→int so callers can pass either.
        # The previous `port_str.strip()` raised AttributeError on int input.
        if port is not None:
            self.port = int(port)
        else:
            self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = (database or os.getenv("POSTGRES_DB", "devnexus")).strip()
        self.user = (user or os.getenv("POSTGRES_USER", "devnexus")).strip()
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

        # Validate host - reject values with invalid characters
        if "\n" in self.host or "\r" in self.host:
            raise ValueError(
                f"POSTGRES_HOST contains invalid characters (newline): "
                f"{repr(self.host)}"
            )
        if not self.host or self.host == "localhost":
            raise ValueError(f"POSTGRES_HOST is not configured: {self.host}")

        # Pool sizing: keep the original 0/2-by-env default. The previous
        # min_size=2 default amplified a concurrency bug: the
        # DatabaseManager's disconnect()/connect() path is NOT safe for
        # concurrent callers, and eager connection init increased the
        # chance of two coroutines racing for the pool at startup.
        # Reverted to the original sizing — the natural-asyncpg pattern
        # (lazy connect + let the pool recycle dead connections) is
        # the correct way to handle this.
        env = os.getenv("ENVIRONMENT", "dev").lower()
        # Use `is not None` (not `or`) so explicit min_size=0 is honored
        # as the lazy-connect opt-out.
        self.min_size = (
            min_size if min_size is not None else (2 if env == "prod" else 0)
        )
        self.max_size = max_size or (10 if env == "prod" else 5)

        # ``self.pool`` is the manager's authoritative reference to the
        # active connection pool. After ``connect()`` it is synced from
        # ``self._backend.pool``; tests that bypass ``__init__`` via
        # ``__new__`` set it directly. The manager's basic query
        # helpers (``acquire``, ``execute``, ``fetch``, ...) read
        # through this attribute, which is what the existing test
        # suite patches.
        self.pool: Optional[asyncpg.Pool] = None

        # Connection state tracking: "disconnected" | "initializing" | "connected" | "failed"
        self._connection_state = "disconnected"
        self._connection_error: Optional[str] = None

        # Pool saturation metrics (issue #687). Acquire wait times and
        # failure counts are accumulated here and exposed via
        # health_check() so the operator sees pool starvation before it
        # becomes user-visible. The data is per-process (not global);
        # multiple DatabaseManager instances have independent counters.
        self._stats = _PoolStats()

        # Check if PostgreSQL should be used
        # Default to enabled: the facade exists to provide PG access; callers
        # who need to opt out set USE_POSTGRESQL=false explicitly. The previous
        # default of "false" silently disabled every downstream that didn't
        # remember to set the flag — first caught when rag_research_tool's
        # wiki /db-status returned "PostgreSQL is disabled" after migrate.
        self.enabled = os.getenv("USE_POSTGRESQL", "true").lower() == "true"

        self._application_name = (
            application_name or os.getenv("POSTGRES_APP_NAME", "py-daroja-libs")
        ).strip()
        self._search_path = (
            search_path or os.getenv("POSTGRES_SEARCH_PATH", "public")
        ).strip()

        # Statement timeout (ms) — applied as a session setting
        # (statement_timeout) AND as asyncpg's per-operation
        # command_timeout (seconds). Both bound the runtime of any
        # single query. The value is POSTGRES_STATEMENT_TIMEOUT_MS
        # (default 30000 = 30s), tunable per-tool for batch jobs that
        # legitimately need longer. Issue #686. Stored as int ms
        # internally; converted to float seconds for command_timeout
        # at call time.
        self.statement_timeout_ms = int(
            os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "30000")
        )

        # SSL settings (used by AsyncpgBackend via the BackendConfig).
        # Read once at construction so the BackendConfig is fully
        # populated before connect() runs.
        self._ssl_mode = os.getenv("POSTGRES_SSLMODE", "disable").lower()
        self._ssl_no_verify = os.getenv("POSTGRES_SSL_NO_VERIFY", "false").lower() in (
            "1",
            "true",
            "yes",
        )

        # ------------------------------------------------------------------
        # Dedicated event loop for the sync facade (issues #7/#9).
        # See module docstring for the full rationale.
        # ------------------------------------------------------------------
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        self._loop_started = threading.Event()
        self._loop_failed: Optional[BaseException] = None

        # Cancellation safety (issue #7): when an in-flight asyncpg op
        # is cancelled, asyncpg's pool/connection can be left in a
        # half-state. We flag a deferred reset; the next non-cancelled
        # ensure_connected observes the flag and rebuilds the pool
        # cleanly (deferred = not racing with active acquirers).
        self._needs_pool_reset: bool = False

        # ------------------------------------------------------------------
        # Backend construction.
        # ------------------------------------------------------------------
        if backend not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Known backends: {sorted(_BACKEND_REGISTRY)}"
            )
        backend_cls = _BACKEND_REGISTRY[backend]
        self._backend = backend_cls()

        # Pre-build the BackendConfig. The SSL/tuning fields are
        # captured at construction time; env-var overrides after
        # __init__ are not honored (consistent with the rest of the
        # constructor's kwargs-or-env-var resolution).
        self._config = BackendConfig(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=self.min_size,
            max_size=self.max_size,
            application_name=self._application_name,
            search_path=self._search_path,
            statement_timeout_ms=self.statement_timeout_ms,
            ssl_mode=self._ssl_mode,
            ssl_no_verify=self._ssl_no_verify,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        """Lazily construct ``self._backend`` and ``self._config`` from
        ``self.*`` attributes when they have not been set yet.

        The normal production path goes through ``__init__`` and sets
        these up. The test path uses ``DatabaseManager.__new__()`` to
        bypass ``__init__`` and sets only the attributes the test
        cares about (host, port, ..., ``self.pool``). Without this
        lazy path, those tests would see ``AttributeError: object has
        no attribute '_backend'`` on the first ``connect()`` or
        ``health_check()`` call.

        The backend is constructed from the same ``self.*`` attrs the
        old code used directly. The test patches
        ``common.db.postgres.asyncpg.create_pool``; because both the
        shim and the backend import the same ``asyncpg`` module
        object, the patch propagates to the backend's pool creation
        call.
        """
        # Use getattr to handle the bypass-__init__ test path: those
        # tests create a manager via DatabaseManager.__new__() and
        # never set self._backend, so a direct attribute access
        # would raise AttributeError. getattr returns None, which
        # then falls through to the construction path.
        if getattr(self, "_backend", None) is not None:
            return
        # _BACKEND_REGISTRY is defined in this module (manager.py),
        # not in common.db.backends. Looking it up in the
        # backends package was a mistake in an earlier draft.
        from common.db.backends.base import BackendConfig

        # Build the config from self.* (test-friendly) rather than
        # reading from env. Tests that bypass __init__ set self.*
        # attrs directly; this picks them up.
        self._config = BackendConfig(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=self.min_size,
            max_size=self.max_size,
            application_name=self._application_name,
            search_path=self._search_path,
            statement_timeout_ms=self.statement_timeout_ms,
            ssl_mode=os.getenv("POSTGRES_SSLMODE", "disable").lower(),
            ssl_no_verify=os.getenv("POSTGRES_SSL_NO_VERIFY", "false").lower()
            in ("1", "true", "yes"),
        )
        self._backend = _BACKEND_REGISTRY["asyncpg"]()

    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL.

        Delegates pool creation to the configured backend, then syncs
        ``self.pool`` so the manager's own helpers (acquire, execute,
        fetch, ...) can use the same reference the backend owns.
        """
        if not self.enabled:
            logger.info("PostgreSQL is disabled (USE_POSTGRESQL=false)")
            self._connection_state = "disconnected"
            return

        if self.pool is not None:
            logger.warning("Database pool already exists")
            return

        self._connection_state = "initializing"
        self._connection_error = None

        # Lazy backend init for the test-bypass path. Production hits
        # this once at startup; the no-op fast path is on every
        # subsequent call.
        self._ensure_backend()

        try:
            await self._backend.connect(self._config)
        except Exception as e:
            self._connection_state = "failed"
            self._connection_error = str(e)
            raise

        # Sync the manager's pool reference to the backend's pool.
        # The manager's basic query helpers read through ``self.pool``,
        # so this keeps them working without an extra layer of
        # delegation for every execute/fetch.
        self.pool = self._backend.pool
        self._connection_state = "connected"
        self._connection_error = None
        logger.info("Database connection pool established")

    async def disconnect(self) -> None:
        """Close connection pool."""
        await self._backend.disconnect()
        # Always clear the manager's pool reference, even if the
        # backend didn't (mocks, partial failures). The manager's
        # helpers gate on ``self.pool is None``, so leaving a stale
        # reference here would cause confusing AttributeError later.
        self.pool = None
        if self._connection_state != "disconnected":
            self._connection_state = "disconnected"

    async def ensure_connected(self) -> None:
        """Ensure database is connected, connecting if needed.

        Consumes the deferred-reset flag set by _with_retry on
        cancellation (issue #7). The flag is processed in a
        non-racing context — never inside the cancellation path
        itself, where it would race with active acquirers. See
        _with_retry for the full rationale.
        """
        if not self.enabled:
            raise RuntimeError("PostgreSQL is disabled")

        # Issue #7: if a previous call was cancelled mid-flight, the
        # pool may be in a half-state. We do NOT disconnect+reconnect
        # inside the cancellation path (that races with active
        # acquirers — the original segfault pattern). Instead, the
        # next non-cancelled ensure_connected() observes the flag,
        # disconnects cleanly, and reconnects.
        if self._needs_pool_reset and self.pool is not None:
            logger.warning("Pool marked for reset (post-cancellation); rebuilding")
            self._needs_pool_reset = False
            try:
                await self.disconnect()
            except Exception as e:
                logger.warning(f"Pool reset disconnect failed (continuing): {e}")
            # Defensive: always clear pool after a reset attempt. The
            # real disconnect() sets pool=None on success, but mocks
            # in tests don't, and a buggy disconnect could leave it
            # set. Either way, fall through to the reconnect path
            # below.
            self.pool = None

        if self.pool is None:
            logger.debug("Database pool not initialized, connecting now...")
            await self.connect()
            return

    async def _with_retry(self, coro_factory, max_retries: int = 2):
        """Execute coroutine with transient error retry.

        IMPORTANT: do NOT call disconnect()/connect() inside this
        loop. A previous version did ``await self.disconnect()`` on
        transient error to force a fresh pool. That pattern is
        unsafe for concurrent callers: when request A is in the
        middle of disconnect() (closing the pool), request B's
        acquire() runs against a closing pool and asyncpg
        segfaults. Production hit this on rag_research_tool's wiki
        revision -00078-bsc.

        The natural asyncpg pattern is to retry the operation. When
        a connection dies, the pool marks it as broken and the next
        acquire() returns a different (or freshly created)
        connection.
        """
        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()
            except (
                asyncpg.ConnectionDoesNotExistError,
                asyncpg.InterfaceError,
            ) as e:
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"DB transient error, retrying ({attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(0.5 * (2**attempt))
            except asyncio.CancelledError:
                # Issue #7: cancellation safety. When the in-flight
                # op is cancelled, asyncpg's pool/connection can be
                # left in a half-state. We do NOT try to
                # disconnect/reconnect here (that races with active
                # acquirers — the original segfault pattern).
                # Instead, flag a deferred reset; the next
                # non-cancelled ensure_connected() will rebuild the
                # pool cleanly. We must re-raise CancelledError so
                # the caller's cancellation contract is preserved.
                self._needs_pool_reset = True
                logger.debug("DB op cancelled; pool marked for deferred reset")
                raise

    # ------------------------------------------------------------------
    # Pool acquisition (with timing for issue #687)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self):
        """Context manager for acquiring a connection from pool.

        Times the pool acquire wait and records it (and any failure)
        on the _PoolStats for issue #687 observability. The timing
        covers ONLY the time spent waiting for a connection in the
        pool's internal queue — not the time the user code holds
        the connection.

        Reads through ``self.pool`` (not ``self._backend.pool``) so
        the existing test suite — which sets ``mgr.pool`` directly
        after ``__new__``-bypass construction — continues to work.
        """
        if not self.enabled or self.pool is None:
            raise RuntimeError("Database not connected")

        start = time.monotonic()
        failed = True
        try:
            pool_ctx = self.pool.acquire()
            connection = await pool_ctx.__aenter__()
            failed = False
        finally:
            wait_ms = (time.monotonic() - start) * 1000.0
            if failed:
                self._stats.record_acquire_failure()
            else:
                self._stats.record_acquire(wait_ms)

        try:
            yield connection
        finally:
            await pool_ctx.__aexit__(None, None, None)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Check database health.

        Composes the backend's health report with the live asyncpg
        pool's size/free/min/max snapshot and the accumulated
        saturation metrics. The operator sees both at a glance: the
        driver-visible state + pressure over time.
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "PostgreSQL is not enabled",
            }

        if self.pool is None:
            # Lazy-init pattern (issue #787): ``self.pool`` is None
            # until the first query triggers ``ensure_connected()``.
            # Services that construct a DatabaseManager at startup
            # but never issue a query would otherwise see
            # ``disconnected`` here even though the pool would
            # connect fine on first use. Trigger the connect here
            # so /health reflects real state, not lazy state.
            try:
                await self.ensure_connected()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"connect failed: {e}",
                }

        if self.pool is None:
            # ``ensure_connected`` ran but did not initialize the
            # pool (e.g. disabled, or a backend that defers pool
            # creation). Treat as disconnected so the operator sees
            # the real state.
            return {
                "status": "disconnected",
                "message": "No active connection pool",
            }

        # Run the driver-visible probe (version + pgvector) directly
        # against ``self.pool`` rather than delegating to the backend.
        # The test path sets ``self.pool`` directly after a
        # ``__new__``-bypass; the backend would not see that pool.
        # The probe SQL is identical for both asyncpg and psycopg 3,
        # so there's no backend-specific work to delegate — but the
        # pool's acquire API differs: asyncpg exposes ``.acquire()``,
        # psycopg 3's ``AsyncConnectionPool`` exposes ``.connection()``
        # and has no ``.acquire()``. Dispatch on the backend when set.
        backend = getattr(self, "_backend", None)
        try:
            if backend is not None and getattr(backend, "name", "") != "asyncpg":
                # psycopg 3 path: backend.acquire() returns the pool's
                # native async context manager (.connection()).
                async with backend.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT version()")
                        v = await cur.fetchone()
                        version = v[0] if v else None
                        await cur.execute(
                            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                        )
                        e = await cur.fetchone()
                    ext = {"extversion": e[0]} if e else None
            else:
                async with self.pool.acquire() as conn:
                    version = await conn.fetchval("SELECT version()")
                    ext = await conn.fetchrow(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

        probe = {
            "status": "healthy",
            "version": version,
            "pgvector": ext["extversion"] if ext else None,
        }

        # asyncpg's pool exposes .get_size()/.get_idle_size()/.get_min_size()/
        # .get_max_size(); psycopg 3's AsyncConnectionPool exposes
        # .min_size / .max_size attributes and a .get_stats() method
        # returning a dict with pool stats. Dispatch the same way as
        # the acquire() path above so health_check_sync doesn't crash
        # on psycopg3 setups.
        backend = getattr(self, "_backend", None)
        if backend is not None and getattr(backend, "name", "") != "asyncpg":
            try:
                stats = self.pool.get_stats()
            except Exception:
                stats = {}
            live_pool = {
                "size": stats.get("pool_size"),
                "free": stats.get("pool_available"),
                "min": getattr(self.pool, "min_size", None),
                "max": getattr(self.pool, "max_size", None),
            }
        else:
            live_pool = {
                "size": self.pool.get_size(),
                "free": self.pool.get_idle_size(),
                "min": self.pool.get_min_size(),
                "max": self.pool.get_max_size(),
            }
        saturation = self._stats.summary()
        return {
            **probe,
            "pool": {**live_pool, **saturation},
        }

    # ------------------------------------------------------------------
    # Basic query helpers (async, used by the sync facade and by
    # callers who already have a running event loop)
    # ------------------------------------------------------------------

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL command (INSERT, UPDATE, DELETE)."""
        await self.ensure_connected()
        return await self._with_retry(lambda: self._execute_impl(query, *args))

    async def _execute_impl(self, query: str, *args: Any) -> str:
        # Phase 2 (issue #28): when the configured backend is psycopg 3,
        # hand the query to the backend after running it through the
        # asyncpg→psycopg placeholder translator. The asyncpg path is
        # preserved verbatim so the existing 82 tests (which set
        # ``self.pool`` directly via ``__new__``-bypass) keep working
        # without changes; ``getattr`` makes the dispatch tolerant of
        # the bypass-__init__ test path where ``self._backend`` may
        # not be set yet.
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.name != "asyncpg":
            translated = translate_asyncpg_to_psycopg(query)
            return await backend.execute(translated, *args)
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[Any]:
        """Fetch multiple rows.

        Return type is ``List[Any]`` (was ``List[asyncpg.Record]``
        pre-issue-#28) so the same signature covers both the asyncpg
        and psycopg 3 backends; the row objects differ in type but
        both support ``row[0]`` / ``row["col"]`` / ``row.keys()`` so
        callsites that used the asyncpg ``Record`` continue to work.
        """
        await self.ensure_connected()
        return await self._with_retry(lambda: self._fetch_impl(query, *args))

    async def _fetch_impl(self, query: str, *args: Any) -> List[Any]:
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.name != "asyncpg":
            translated = translate_asyncpg_to_psycopg(query)
            return await backend.fetch(translated, *args)
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[Any]:
        """Fetch single row.

        Return type is ``Optional[Any]`` (was ``Optional[asyncpg.Record]``
        pre-issue-#28) so the same signature covers both backends; the
        underlying row objects differ in type but support the same
        ``row[0]`` / ``row["col"]`` access pattern.
        """
        await self.ensure_connected()
        return await self._with_retry(lambda: self._fetchrow_impl(query, *args))

    async def _fetchrow_impl(self, query: str, *args: Any) -> Optional[Any]:
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.name != "asyncpg":
            translated = translate_asyncpg_to_psycopg(query)
            return await backend.fetchrow(translated, *args)
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch single value."""
        await self.ensure_connected()
        return await self._with_retry(lambda: self._fetchval_impl(query, *args))

    async def _fetchval_impl(self, query: str, *args: Any) -> Any:
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.name != "asyncpg":
            translated = translate_asyncpg_to_psycopg(query)
            return await backend.fetchval(translated, *args)
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # ------------------------------------------------------------------
    # Patterns helpers — thin delegators to the backend. The actual
    # SQL lives on the backend (asyncpg's $1::vector casts etc.); the
    # manager just forwards the kwargs.
    # ------------------------------------------------------------------

    async def insert_pattern_with_embedding(
        self,
        repo_id: int,
        name: str,
        description: str,
        context: str,
        embedding: Optional[List[float]] = None,
    ) -> Any:
        """Insert pattern with optional embedding. Backend-owned SQL."""
        await self.ensure_connected()
        return await self._backend.insert_pattern_with_embedding(
            repo_id=repo_id,
            name=name,
            description=description,
            context=context,
            embedding=embedding,
        )

    async def find_similar_patterns(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        exclude_repo_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar patterns using vector similarity. Backend-owned SQL."""
        await self.ensure_connected()
        return await self._backend.find_similar_patterns(
            embedding=embedding,
            limit=limit,
            threshold=threshold,
            exclude_repo_id=exclude_repo_id,
        )

    async def update_pattern_embedding(
        self, pattern_id: int, embedding: List[float]
    ) -> None:
        """Update embedding for existing pattern. Backend-owned SQL."""
        await self.ensure_connected()
        return await self._backend.update_pattern_embedding(
            pattern_id=pattern_id, embedding=embedding
        )

    async def get_patterns_without_embeddings(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get patterns that don't have embeddings yet. Backend-owned SQL."""
        await self.ensure_connected()
        return await self._backend.get_patterns_without_embeddings(limit=limit)

    # ------------------------------------------------------------------
    # Sync facade
    # ------------------------------------------------------------------
    # Use these from SYNC contexts only (CLI scripts, sync request
    # handlers, threadpool workers). The _run_sync guard raises if
    # called from inside a running event loop — in async contexts,
    # use the async methods directly.
    #
    # Implementation note (issues #7/#9): we previously used
    # asyncio.run(coro) per call, creating a fresh event loop on
    # every sync facade invocation. That pattern is the root cause
    # of the production SIGSEGV: asyncpg's pool is bound to
    # whichever loop was running when the pool was created. A new
    # loop on every call meant the pool's internal asyncio
    # primitives were stale relative to the new loop, and
    # CancelledError cleanup would deref primitives bound to the
    # OLD loop → SIGSEGV in the C protocol code.
    #
    # New pattern: all sync-facade calls run on a single dedicated
    # event loop in a daemon thread owned by this DatabaseManager.
    # The asyncpg pool is then bound to one stable loop for its
    # lifetime, which is what asyncpg expects.
    # concurrent.futures.Future.cancel() correctly propagates to
    # the underlying asyncio task on the dedicated loop, giving us
    # proper cancellation semantics (the missing primitive that
    # asyncio.run()-per-call couldn't provide).
    #
    # Phase 2 (issue #28) will wire ``translate_asyncpg_to_psycopg``
    # into the SQL path when ``self._backend.name == "psycopg3"``,
    # so the same ``$1``-style queries work against the new driver.

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start the dedicated event loop in a daemon thread.

        Idempotent: subsequent calls return the same loop.
        Thread-safe: the start sequence is serialized by
        ``_loop_lock``. The loop is bound for the lifetime of the
        DatabaseManager; the daemon thread is reaped when the
        process exits.

        Raises:
            RuntimeError: if the loop thread fails to start within
                5s or fails to initialize (e.g. asyncio.new_event_loop
                itself raises).
        """
        if self._loop is not None:
            return self._loop
        with self._loop_lock:
            if self._loop is not None:
                return self._loop
            self._loop_started.clear()
            self._loop_failed = None

            def _run_loop() -> None:
                loop: Optional[asyncio.AbstractEventLoop] = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._loop = loop
                    self._loop_started.set()
                    loop.run_forever()
                except BaseException as e:  # noqa: BLE001
                    self._loop_failed = e
                    self._loop_started.set()
                finally:
                    if loop is not None:
                        try:
                            loop.close()
                        except Exception:
                            pass
                    self._loop = None

            self._loop_thread = threading.Thread(
                target=_run_loop,
                daemon=True,
                name=f"db-loop-{id(self):x}",
            )
            self._loop_thread.start()
            if not self._loop_started.wait(timeout=5.0):
                raise RuntimeError(
                    "DatabaseManager loop thread did not signal ready within 5s"
                )
            if self._loop_failed is not None:
                raise RuntimeError(
                    f"DatabaseManager loop thread failed to "
                    f"initialize: {self._loop_failed}"
                )
            if self._loop is None:
                raise RuntimeError("DatabaseManager loop is None after start")
            return self._loop

    def _run_sync(self, coro, timeout: Optional[float] = None):
        """Run a coroutine to completion on the dedicated DB loop.

        Blocks the calling thread until the coroutine completes (or
        ``timeout`` elapses, in which case the underlying asyncio
        task is cancelled via
        ``concurrent.futures.Future.cancel()`` and
        ``concurrent.futures.TimeoutError`` is raised). The asyncpg
        pool is bound to the dedicated loop for the lifetime of
        this DatabaseManager, so all sync-facade calls share the
        same loop and asyncpg's internal primitives stay
        consistent.

        The function works from BOTH contexts:
        - a thread with no event loop (e.g. a worker thread
          spawned by ``asyncio.to_thread``), and
        - a thread that is currently inside a running event loop
          (e.g. the FastAPI request handler's main coroutine).

        In the second case we still schedule on the *dedicated*
        DB loop (a different loop running in a different thread),
        not the caller's loop, because the caller's loop is the
        one that is busy running the request handler and cannot
        itself execute the coroutine without yielding. This is
        technically a blocking call from the caller's event loop
        (the DB call's wall time is added to the caller's
        coroutine's elapsed time), but it is a small, bounded
        cost (a few ms for the typical PG round-trip in
        py-daroja-libs) and unblocks the FastAPI routers
        that previously raised ``RuntimeError`` and silently fell
        back to the disk cache (the source of the audit's
        "successful but not persisted" mystery).

        The proper long-term fix is for the FastAPI routers to
        ``await db.execute(...)`` directly. This is a pragmatic
        interim: schedule on the dedicated DB loop and wait,
        which is correct, just non-async.

        Raises:
            concurrent.futures.TimeoutError: if ``timeout`` is set
                and the coroutine did not complete in time.
            Anything the coroutine raises: re-raised here.
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def execute_sync(self, query: str, *args: Any) -> str:
        """Sync wrapper for execute(). See class docstring."""
        return self._run_sync(self.execute(query, *args))

    def fetch_sync(self, query: str, *args: Any) -> "List[asyncpg.Record]":
        """Sync wrapper for fetch(). See class docstring."""
        return self._run_sync(self.fetch(query, *args))

    def fetchrow_sync(self, query: str, *args: Any) -> "Optional[asyncpg.Record]":
        """Sync wrapper for fetchrow(). See class docstring."""
        return self._run_sync(self.fetchrow(query, *args))

    def fetchval_sync(self, query: str, *args: Any) -> Any:
        """Sync wrapper for fetchval(). See class docstring."""
        return self._run_sync(self.fetchval(query, *args))

    def health_check_sync(self) -> Dict[str, Any]:
        """Sync wrapper for health_check(). See class docstring."""
        return self._run_sync(self.health_check())

    # ------------------------------------------------------------------
    # Bulk insert (pgvector-aware, async — bridges via bulk_insert_sync)
    # ------------------------------------------------------------------

    async def bulk_insert(
        self,
        table: str,
        columns: List[str],
        rows: List[Tuple],
        *,
        on_conflict: Optional[str] = None,
        page_size: int = 1000,
    ) -> str:
        """Insert many rows in batches of page_size. Equivalent to
        psycopg2.extras.execute_values.

        Args:
            table: Target table name. Not quoted; caller is
                responsible for ensuring the table name is safe
                (e.g., not from untrusted user input).
            columns: List of column names. Same safety caveat as
                table.
            rows: Tuples of values, one per row. Each tuple's
                length must equal len(columns).
            on_conflict: Optional ON CONFLICT clause appended
                verbatim, e.g. "ON CONFLICT (id) DO UPDATE SET x =
                EXCLUDED.x".
            page_size: Rows per INSERT statement. asyncpg handles
                1000s of params in one call; 1000 is a safe default.

        Returns:
            The total rows-inserted string, e.g. "INSERT 0 1042".
        """
        if not rows:
            return "INSERT 0"
        n = len(columns)
        if any(len(r) != n for r in rows):
            raise ValueError(
                f"bulk_insert: every row must have exactly {n} "
                f"values; got row lengths {[len(r) for r in rows]}"
            )
        cols = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(n))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        if on_conflict:
            sql += f" {on_conflict}"
        await self.ensure_connected()
        total = 0

        for i in range(0, len(rows), page_size):
            page = rows[i : i + page_size]
            params = list(chain.from_iterable(page))
            result = await self._with_retry(lambda: self._execute_impl(sql, *params))
            # asyncpg returns "INSERT 0 42" for inserts; the trailing
            # number is rows-inserted. Parse defensively.
            try:
                total += int(result.rsplit(" ", 1)[-1])
            except (ValueError, IndexError):
                logger.debug(f"bulk_insert: could not parse row count from {result!r}")
        return f"INSERT 0 {total}"

    def bulk_insert_sync(
        self,
        table: str,
        columns: List[str],
        rows: List[Tuple],
        *,
        on_conflict: Optional[str] = None,
        page_size: int = 1000,
    ) -> str:
        """Sync wrapper for bulk_insert()."""
        return self._run_sync(
            self.bulk_insert(
                table,
                columns,
                rows,
                on_conflict=on_conflict,
                page_size=page_size,
            )
        )


# ---------------------------------------------------------------------------
# Singleton helpers (module-level)
# ---------------------------------------------------------------------------
# These are the canonical entry points for application startup. The
# tests patch ``common.db.postgres._db_manager`` (which re-exports the
# one here), so the singleton MUST be accessible as a module-level
# attribute on this module.

_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get database manager singleton.

    Uses late imports from ``common.db.postgres`` for both
    ``DatabaseManager`` and ``_db_manager`` so test patches
    (e.g. ``patch.object(pg_module, "DatabaseManager", return_value=stub)``
    and ``patch.object(pg_module, "_db_manager", mgr)``) propagate
    correctly. The class object is identical — we just look it up
    where the test is patching.
    """
    from common.db import postgres as _pg

    global _db_manager
    # Honor a test patch to pg_module._db_manager (set to None to
    # clear, or to a stub to inject). Production never patches this;
    # the patch is only set inside ``with patch.object(...)`` blocks.
    if _pg._db_manager is not None:
        _db_manager = _pg._db_manager
        return _db_manager
    if _db_manager is None:
        _db_manager = _pg.DatabaseManager()
    return _db_manager


async def init_db() -> DatabaseManager:
    """Initialize database connection."""
    db = get_db()
    await db.connect()
    return db


async def close_db() -> None:
    """Close database connection."""
    db = get_db()
    await db.disconnect()


def init_db_sync() -> DatabaseManager:
    """Sync wrapper for init_db().

    Use from sync contexts (CLI scripts, sync request handlers,
    threadpool workers). For async contexts, use init_db() directly.
    """
    db = get_db()
    db._run_sync(db.connect())
    return db


def close_db_sync() -> None:
    """Sync wrapper for close_db(). See init_db_sync for usage."""
    db = get_db()
    if db.pool is not None:
        db._run_sync(db.disconnect())
