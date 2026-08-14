"""psycopg 3 implementation of the ``DatabaseBackend`` Protocol.

This is the second backend for ``DatabaseManager`` (issue #28). It pairs
with the existing :class:`AsyncpgBackend` so callers can choose between
the two drivers at construction time:

    DatabaseManager(backend="asyncpg", ...)   # default, unchanged
    DatabaseManager(backend="psycopg3", ...)   # issue #28

Why a second backend
--------------------

asyncpg has a class of C-level concurrency bugs around cancellation /
future cleanup that the dedicated-loop fix in issue #9 only mitigates,
not eliminates (the upstream segfault in asyncpg 0.31.0 / issue #1292
is one example). psycopg 3 is a pure-Python wrapper around libpq and
does not keep asyncio primitives in C, so the bug class simply does
not exist there. Issue #11 is the long-form rationale.

How this file is organized
--------------------------

The shape mirrors :mod:`common.db.backends.asyncpg` so a reviewer can
diff method-by-method. The driver-specific differences are concentrated
in three places:

  1. Pool config translation. ``dbname=`` instead of ``database=``;
     ``max_idle=300`` instead of ``max_inactive_connection_lifetime=300``.
     psycopg 3 has no pool-level ``command_timeout``; the per-statement
     timeout is set server-side via ``options='-c statement_timeout=NNNms'``.

  2. Parameter placeholders. ``$1, $2`` → ``%s, %s`` is handled by the
     manager's translator at the boundary, NOT here. The SQL constants
     in this module are written in ``%s`` style and reach the driver
     already-translated.

  3. Row objects. psycopg 3's ``Row`` is dict-like but is missing a
     clean ``.keys()`` in some versions and does not support numeric
     indexing uniformly. The :class:`_RowAdapter` shim normalizes
     both so callsites that use ``row[0]`` or ``row["column_name"]``
     work the same way against both backends.

Optional dependencies
---------------------

psycopg and psycopg_pool are required at runtime when this backend
is selected; the manager's lazy import (``_PSYCOPG3_AVAILABLE``) keeps
them out of the critical path for asyncpg-only deployments.

The pgvector package is optional. If it is importable, the per-connection
``configure`` callback registers pgvector's vector adapter so callers
can pass ``List[float]`` directly. If it is not importable, the patterns
helpers fall back to encoding vectors as a ``'[1.0,2.0,3.0]'`` string,
which the SQL ``::vector`` cast parses without any driver help.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

# psycopg / psycopg_pool are runtime deps when this backend is used.
# The manager's lazy import keeps them out of the import-time critical
# path for asyncpg-only deployments. If the import fails here, the
# module still loads (so ``common.db.backends.psycopg3`` can be imported
# in environments without psycopg, e.g. lint or test environments that
# only exercise the asyncpg backend). The class itself raises a clear
# ImportError when ``connect()`` is called without the driver.
#
# ``import psycopg`` is the canonical "is psycopg 3 installed?" guard.
# It is referenced via the ``TYPE_CHECKING`` block below to give ruff's
# F401 (unused import) a reason to leave it alone; at runtime psycopg
# is only ever touched by psycopg_pool and ``psycopg.types.json`` so
# the bare import here exists purely as a presence probe.
try:
    import psycopg  # type: ignore  # noqa: F401
    import psycopg_pool  # type: ignore

    _PSYCOPG_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without driver
    psycopg = None  # type: ignore
    psycopg_pool = None  # type: ignore
    _PSYCOPG_AVAILABLE = False

# ``set_json_loaders`` is psycopg 3's hook to register custom (dumps,
# loads) callables for json / jsonb values. It lives in
# ``psycopg.types.json`` from psycopg 3.1+. We import it lazily so the
# module still loads on older psycopg 3 builds; the jsonb adapter
# is a best-effort upgrade and not required for the basic Protocol
# contract.
try:
    from psycopg.types.json import set_json_loaders  # type: ignore

    _JSON_LOADERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on old psycopg
    set_json_loaders = None  # type: ignore
    _JSON_LOADERS_AVAILABLE = False

# pgvector is an optional companion package for psycopg 3. When it is
# importable, the ``configure`` callback calls ``register_vector`` so
# ``List[float]`` round-trips natively. When it is not (the common dev
# env case), the patterns helpers fall back to encoding vectors as
# strings the ``::vector`` cast can parse.
try:
    import pgvector.psycopg as _pgvector_psycopg  # type: ignore

    _HAS_PGVECTOR = True
except ImportError:  # pragma: no cover - exercised only without pgvector
    _pgvector_psycopg = None  # type: ignore
    _HAS_PGVECTOR = False

if TYPE_CHECKING:
    # Type-only imports for annotations. These are erased at runtime
    # so they don't need the try/except above. Used by the per-connection
    # ``configure`` callback's ``conn: psycopg.AsyncConnection`` hint.
    import psycopg as _psycopg_typing  # noqa: F401

from common.db.backends.base import BackendConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL constants for the patterns helpers.
#
# Kept as module-level strings so the manager and any other caller can
# see exactly which queries the backend owns. The differences from the
# asyncpg versions are the placeholder style (``%s`` everywhere, in
# positional order) and the ``::vector`` cast, which both backends
# accept but only asyncpg is happy to take a Python list for. Here we
# always pass a string for the vector parameter; see ``_encode_vector``.
# ---------------------------------------------------------------------------

_INSERT_PATTERN_SQL = """
    INSERT INTO patterns (repo_id, name, description, context, embedding)
    VALUES (%s, %s, %s, %s, %s::vector)
    ON CONFLICT (repo_id, name)
    DO UPDATE SET
        description = EXCLUDED.description,
        context = EXCLUDED.context,
        embedding = EXCLUDED.embedding
    RETURNING id
"""

_FIND_SIMILAR_PATTERNS_SQL = """
    SELECT
        p.id,
        p.name,
        p.description,
        p.context,
        p.repo_id,
        r.name as repo_name,
        1 - (p.embedding <=> %s::vector) as similarity
    FROM patterns p
    JOIN repositories r ON p.repo_id = r.id
    WHERE p.embedding IS NOT NULL
        AND 1 - (p.embedding <=> %s::vector) >= %s
        AND (%s::integer IS NULL OR p.repo_id != %s)
    ORDER BY p.embedding <=> %s::vector
    LIMIT %s
"""

_UPDATE_PATTERN_EMBEDDING_SQL = (
    "UPDATE patterns SET embedding = %s::vector WHERE id = %s"
)

_GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL = """
    SELECT
        p.id,
        p.name,
        p.description,
        p.context,
        p.repo_id,
        r.name as repo_name
    FROM patterns p
    JOIN repositories r ON p.repo_id = r.id
    WHERE p.embedding IS NULL
    LIMIT %s
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_vector(vec: Optional[List[float]]) -> Optional[str]:
    """Encode a Python list of floats as a Postgres ``vector`` literal.

    Returns ``None`` for ``None`` input; the SQL ``::vector`` cast in
    the patterns methods then produces a SQL NULL, which matches the
    "no embedding yet" case the patterns table allows.

    The returned string is a Postgres vector literal (square-bracket
    list of floats). The ``::vector`` cast in SQL parses this format
    without any driver-side help, so the fallback works whether or not
    the ``pgvector`` Python package is installed.

    Note: we coerce each element to ``float`` so callers that pass
    ``List[int]`` (common for short embeddings) still produce a valid
    vector literal — Postgres' vector parser is strict about the
    ``1.0`` form and will reject ``1``.
    """
    if vec is None:
        return None
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _build_pool_kwargs(config: BackendConfig) -> dict:
    """Translate a :class:`BackendConfig` into ``psycopg_pool`` kwargs.

    The mapping is documented in the issue #28 plan. Key differences
    from the asyncpg backend:

    * ``database`` → ``dbname`` (Postgres / libpq naming).
    * ``min_size`` / ``max_size`` keep their names.
    * No pool-level ``command_timeout``; we set the per-statement
      timeout server-side via ``options='-c statement_timeout=NNNms'``.
    * ``max_idle=300`` replaces asyncpg's
      ``max_inactive_connection_lifetime=300``.
    * SSL is negotiated through ``sslmode`` / ``sslrootcert`` in the
      connection parameters, not through Python's ``ssl`` module.
    * ``search_path`` and ``application_name`` are per-connection
      options; ``search_path`` is applied in the ``configure``
      callback because there is no per-pool setting for it.

    The returned dict does NOT include ``configure``; the caller
    (the backend's :meth:`connect`) attaches the configure callback
    separately so the closure can capture the per-instance state.
    """
    if not _PSYCOPG_AVAILABLE:
        raise ImportError(
            "psycopg 3 is not installed; the 'psycopg3' backend requires "
            "the 'psycopg[binary,pool]' (or equivalent) package. Install "
            "it via the py-daroja-libs[psycopg3] extra (Phase 3) or "
            "pin psycopg>=3.1 and psycopg_pool>=3.2 directly."
        )

    # Build a libpq conninfo string from connection parameters.
    # psycopg_pool.AsyncConnectionPool accepts connection params via the
    # ``conninfo`` positional arg (a libpq connection string) or the
    # ``kwargs`` dict, NOT as direct pool constructor kwargs.
    conninfo_parts = [f"host={config.host}"]
    if config.port:
        conninfo_parts.append(f"port={config.port}")
    conninfo_parts.append(f"dbname={config.database}")
    if config.user:
        conninfo_parts.append(f"user={config.user}")
    if config.password:
        conninfo_parts.append(f"password={config.password}")

    # SSL translation. ``disable`` / ``false`` / ``0`` → ``sslmode=disable``.
    # ``require`` / ``true`` / ``1`` → ``sslmode=require``; the
    # ``sslrootcert=''`` trick disables cert verification while still
    # encrypting the wire (the libpq-documented way to say
    # "encrypt-don't-verify" without writing a Python ssl context).
    ssl_mode = (config.ssl_mode or "").lower()
    if ssl_mode in ("disable", "false", "0"):
        conninfo_parts.append("sslmode=disable")
    elif ssl_mode in ("require", "true", "1"):
        conninfo_parts.append("sslmode=require")
        if config.ssl_no_verify:
            conninfo_parts.append("sslrootcert=")

    conninfo = " ".join(conninfo_parts)

    # Pool-level kwargs (not connection params).
    pool_kwargs: dict = {
        # Pool sizing. ``min_size=0`` is allowed (matches asyncpg) and
        # is the "lazy connect" opt-out for tests / single-shot CLIs.
        "min_size": config.min_size,
        "max_size": config.max_size,
        # Idle lifetime. asyncpg's ``max_inactive_connection_lifetime"
        # is called ``max_idle`` here. 5 minutes matches the asyncpg
        # default the rest of the codebase assumes.
        "max_idle": 300,
        # Don't open the pool until we explicitly call ``open()``; the
        # caller controls the connect-vs-open sequence so we can apply
        # retries and emit a clear log line.
        "open": False,
    }

    # Per-connection options passed via the pool's ``kwargs`` param.
    # These are forwarded to psycopg.connect() on every new connection.
    conn_kwargs: dict = {}

    # Application name is a per-connection option, not a pool option.
    # psycopg forwards it to libpq which sets ``application_name`` in
    # the startup packet; the value shows up in ``pg_stat_activity``.
    if config.application_name:
        conn_kwargs["application_name"] = config.application_name

    # Server-side statement timeout. We pass it via libpq's ``options``
    # connection parameter, which the driver forwards verbatim. The
    # ``ms`` suffix means milliseconds, which matches the
    # ``BackendConfig.statement_timeout_ms`` field. The Postgres GUC
    # accepts ms / s / min; the unit is just for readability.
    if config.statement_timeout_ms and config.statement_timeout_ms > 0:
        conn_kwargs["options"] = f"-c statement_timeout={config.statement_timeout_ms}ms"

    # search_path is set in the ``configure`` callback (no pool-level
    # setting), but we also pass it as a conn_kwarg so the ``configure``
    # callback can read it from the pool's conn_kwargs if needed.
    if config.search_path:
        conn_kwargs["options"] = (
            conn_kwargs.get("options", "") + f" -c search_path={config.search_path}"
        )
        conn_kwargs["options"] = conn_kwargs["options"].strip()

    if conn_kwargs:
        pool_kwargs["kwargs"] = conn_kwargs

    return {"conninfo": conninfo, **pool_kwargs}


# ---------------------------------------------------------------------------
# Row adapter
# ---------------------------------------------------------------------------


class _RowAdapter:
    """Adapter to make psycopg 3's ``Row`` behave like asyncpg's ``Record``.

    asyncpg's ``Record`` supports ``row[0]`` (positional), ``row["col"]``
    (column-name), ``row.keys()``, ``row.values()``, ``len(row)``, and
    membership tests. psycopg 3's ``Row`` is dict-like — ``row["col"]``
    works in 3.1+ — but its support for ``row[0]``, ``.keys()``, and
    numeric indexing has been uneven across versions. The adapter
    normalizes both so the same call sites work against either backend.

    Kept on ``__slots__`` to avoid a per-row dict allocation in
    ``fetch()`` paths that return large result sets.
    """

    __slots__ = ("_row",)

    def __init__(self, row: Any) -> None:
        self._row = row

    def __getitem__(self, key: Any) -> Any:
        # psycopg 3.1+ supports both str and int on Row; we delegate
        # unconditionally and let psycopg raise if the key is wrong.
        return self._row[key]

    def keys(self) -> list:
        # psycopg 3.1+ exposes ``Row.keys()`` returning the column
        # names in declaration order. ``_index`` is the private dict
        # psycopg uses internally; falling back to it keeps the adapter
        # working on older 3.0 builds that don't expose ``keys()``.
        if hasattr(self._row, "keys"):
            return list(self._row.keys())
        return list(self._row._index)

    def values(self) -> list:
        return [self._row[k] for k in self.keys()]

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, k: Any) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Psycopg3Backend:
    """psycopg 3-backed implementation of the ``DatabaseBackend`` Protocol.

    Lifecycle, contract, and method semantics mirror
    :class:`AsyncpgBackend`. The differences are confined to pool
    construction (``_build_pool_kwargs``), per-connection setup
    (the ``configure`` callback), and the row adapter.

    The instance is a regular class, not a dataclass: it holds mutable
    driver state (``self.pool``) that changes over the lifetime of
    the connection, and every Protocol method besides ``name`` and
    ``acquire`` is a coroutine.

    Note: ``self.pool`` is the public reference the ``DatabaseManager``
    syncs from (``self.pool = self._backend.pool`` in
    ``DatabaseManager.connect``). It was previously named ``self._pool``;
    the rename aligns with :class:`AsyncpgBackend` so both backends
    expose the same attribute name and the manager's connect path works
    uniformly. ``self._pool`` and ``self._backend._pool`` are no longer
    valid; the attribute is now ``self.pool`` on every backend.
    """

    def __init__(self) -> None:
        # Set by connect(); None until the pool is open.
        self.pool: Optional[Any] = None
        # Stored so health_check() can report host/database; not used
        # for pool config after connect() returns.
        self._config: Optional[BackendConfig] = None
        # NOTE: psycopg 3 does NOT expose ``cur.timeout`` as a cursor
        # attribute and ``cur.execute(query, args, timeout=...)`` is
        # not a valid kwarg (that is asyncpg's API). The server-side
        # ``statement_timeout`` GUC set via libpq ``options=`` in
        # ``_build_pool_kwargs`` is the only correct per-statement
        # timeout. An earlier client-side backstop here was removed
        # because the two wrong-API attempts (8ade1a2, 5b8da61, beb969f)
        # each broke psycopg 3 in a different way. PR #73.

    # ------------------------------------------------------------------
    # Protocol: identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "psycopg3"

    # ------------------------------------------------------------------
    # Protocol: lifecycle
    # ------------------------------------------------------------------

    async def connect(self, config: BackendConfig) -> None:
        """Open the psycopg 3 connection pool.

        Builds the pool kwargs from ``config`` (see :func:`_build_pool_kwargs`),
        constructs the pool with ``open=False`` so the caller's
        connect-vs-retry sequence stays explicit, then ``await pool.open(wait=True)``
        to wait for the initial connections to be established.

        After the pool is open we run a best-effort ``SELECT extversion
        FROM pg_extension WHERE extname = 'vector'`` probe. The probe
        logs the version when pgvector is installed and a warning when
        it isn't (the patterns helpers degrade gracefully either way).
        """
        self._config = config
        pool_kwargs = _build_pool_kwargs(config)
        pool_kwargs["configure"] = self._build_configure(config)

        if psycopg_pool is None:  # pragma: no cover - guard above covers this
            raise ImportError("psycopg_pool is not installed")

        # psycopg_pool.AsyncConnectionPool.open's default timeout is 30s.
        # That default is too short for cold-start scenarios (CI
        # containers, serverless cold starts, fresh DB clusters where
        # initdb + first connection + TLS handshake have to fit in
        # the budget). 30s was confirmed too short on the
        # py-daroja-libs-stress.yml job (PR #73) where the psycopg 3
        # path timed out with "couldn't get a connection after 30.00
        # sec" while the asyncpg path on the same container succeeded.
        # 60s gives the cold-start room without making genuinely-broken
        # connections hang the caller indefinitely.
        self.pool = psycopg_pool.AsyncConnectionPool(**pool_kwargs)
        await self.pool.open(wait=True, timeout=60.0)

        # Best-effort pgvector probe. The probe runs against a
        # connection from the pool so the same per-connection settings
        # (search_path, jsonb loader) are in effect; if the probe
        # itself fails (network blip, missing privileges) we just log
        # a warning rather than failing the whole connect().
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    row = await cur.fetchone()
            if row is not None:
                extversion = (
                    row[0] if not isinstance(row, dict) else row.get("extversion")
                )
                logger.info(f"pgvector extension detected: v{extversion}")
            else:
                logger.warning("pgvector extension not found")
        except Exception as e:  # pragma: no cover - exercised in real PG only
            logger.warning(f"pgvector probe failed (non-fatal): {e}")

        logger.info("Database connection pool established (psycopg3)")

    def _build_configure(self, config: BackendConfig):
        """Build the per-connection ``configure`` callback.

        The callback is invoked by psycopg_pool on every new connection
        created in the pool. We use it for three things:

        1. ``SET search_path TO ...`` — there is no per-pool setting
           for ``search_path`` in psycopg_pool, so it has to go here.
        2. Register the jsonb loader so ``json/jsonb`` columns
           round-trip as Python objects (asyncpg-parity). Falls back
           to a no-op if the ``set_json_loaders`` API isn't available
           in this psycopg build.
        3. If the ``pgvector`` package is importable, register its
           adapter so ``List[float]`` round-trips natively. The
           patterns helpers additionally always pass a string-encoded
           vector (see :func:`_encode_vector`) so the patterns code
           works whether or not pgvector is installed.
        """
        search_path = config.search_path
        has_json = _JSON_LOADERS_AVAILABLE
        has_pgvector = _HAS_PGVECTOR
        pgvector_mod = _pgvector_psycopg

        async def _configure(conn: Any) -> None:
            if search_path:
                # The schema name is operator-controlled (it comes
                # from the ``search_path`` config field, not from a
                # SQL string the caller provided), so interpolation
                # is safe. Avoids the f-string-around-quotes trap
                # where ``SET search_path TO 'public'`` becomes
                # syntactically invalid because of the surrounding
                # quotes.
                #
                # CRITICAL: psycopg_pool's configure callback must not
                # leave the connection in an open transaction. Without
                # autocommit, ``SET search_path`` runs inside an
                # implicit transaction; when the pool tries to reuse
                # the connection, it sees INTRANS state and discards
                # the connection with "connection left in status
                # INTRANS by configure function". The correct psycopg 3
                # API is ``conn.set_autocommit(True)`` -- ``conn.execute``
                # does NOT accept an ``autocommit`` kwarg (the prior
                # attempt did and was rejected with "got an unexpected
                # keyword argument 'autocommit'" on PR #73).
                # We restore autocommit=False afterwards so the
                # connection's normal transactional mode is preserved
                # for the callers that pull it from the pool.
                prev_autocommit = conn.autocommit
                try:
                    await conn.set_autocommit(True)
                    await conn.execute(f"SET search_path TO {search_path}")
                finally:
                    await conn.set_autocommit(prev_autocommit)
            if has_json and set_json_loaders is not None:
                # json.dumps / json.loads gives the same behavior as
                # asyncpg's auto-decode of jsonb to a Python object.
                # The loader set is per-connection so a connection
                # pulled from the pool will have it active.
                set_json_loaders(json.dumps, json.loads, conn)
            if has_pgvector and pgvector_mod is not None:
                pgvector_mod.register_vector(conn)

        return _configure

    async def disconnect(self) -> None:
        """Close the connection pool. Idempotent."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed (psycopg3)")

    # ------------------------------------------------------------------
    # Protocol: health, acquire
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Return ``{status, version, pgvector_version, host, database}``.

        Mirrors the asyncpg backend's report shape so the manager can
        layer its pool-stats metrics on top uniformly. The ``status``
        field is one of ``"healthy"``, ``"unhealthy"``, or
        ``"disconnected"``.
        """
        if self.pool is None:
            return {"status": "disconnected", "message": "No active connection pool"}

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT version()")
                    version_row = await cur.fetchone()
                    version = version_row[0] if version_row is not None else None

                    await cur.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    pgvector_row = await cur.fetchone()

            host = self._config.host if self._config else None
            database = self._config.database if self._config else None
            pgvector_version: Optional[str] = None
            if pgvector_row is not None:
                # Cursor ``fetchone`` returns a Row; ``[0]`` is the
                # first column, which is what we want.
                pgvector_version = (
                    pgvector_row[0]
                    if not isinstance(pgvector_row, dict)
                    else pgvector_row.get("extversion")
                )
            return {
                "status": "healthy",
                "version": version.split(",")[0] if version else None,
                "pgvector_version": pgvector_version,
                "host": host,
                "database": database,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def acquire(self):
        """Return an async context manager yielding a psycopg connection.

        Unlike the asyncpg backend, we do NOT wrap with
        ``@asynccontextmanager``: psycopg_pool's ``connection()`` is
        already an async context manager, so we just return it. The
        caller does ``async with backend.acquire() as conn: ...``.

        Raises ``RuntimeError`` if the pool is not connected. The
        manager's :meth:`DatabaseManager.acquire` does not yet dispatch
        to this method; psycopg 3 callers should go through the
        backend's own ``acquire()`` (or through the four ``_*_impl``
        helpers which already dispatch).
        """
        if self.pool is None:
            raise RuntimeError("Database not connected")
        return self.pool.connection()

    # ------------------------------------------------------------------
    # Protocol: basic query helpers
    # ------------------------------------------------------------------

    async def execute(self, query: str, *args: Any) -> str:
        """Run a single execute against a freshly-acquired connection.

        The query is expected to already be in psycopg 3 style
        (``%s`` placeholders). The manager's translator at the
        boundary rewrites ``$1, $2`` → ``%s, %s`` before invoking
        the backend, so the backend never sees ``$N`` placeholders.

        Returns the command's status string (``"INSERT 0 1"`` etc.),
        matching the asyncpg backend's return contract.
        """
        assert self.pool is not None, "Database not connected"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                return cur.statusmessage or ""

    async def fetch(self, query: str, *args: Any) -> List[_RowAdapter]:
        """Fetch multiple rows. Each row is wrapped in :class:`_RowAdapter`."""
        assert self.pool is not None, "Database not connected"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                rows = await cur.fetchall()
                return [_RowAdapter(row) for row in rows]

    async def fetchrow(self, query: str, *args: Any) -> Optional[_RowAdapter]:
        """Fetch a single row, or ``None`` if no rows match."""
        assert self.pool is not None, "Database not connected"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                row = await cur.fetchone()
                return _RowAdapter(row) if row is not None else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single scalar value, or ``None`` if no rows match."""
        assert self.pool is not None, "Database not connected"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                row = await cur.fetchone()
                if row is None:
                    return None
                # ``row[0]`` works for both Row and tuple results.
                return row[0] if not isinstance(row, dict) else next(iter(row.values()))

    async def executemany(self, query: str, args: List[Any]) -> None:
        """Execute ``query`` once per row in ``args`` (psycopg-style batch).

        psycopg 3's ``cur.executemany`` is the same shape as
        asyncpg's. The manager is expected to pass a list of
        positional-arg tuples; psycopg will adapt it to a single
        multi-row INSERT / UPDATE / DELETE under the hood.
        """
        assert self.pool is not None, "Database not connected"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, args)

    # ------------------------------------------------------------------
    # Protocol: patterns helpers (rag_research_tool)
    # ------------------------------------------------------------------

    async def insert_pattern_with_embedding(
        self,
        repo_id: int,
        name: str,
        description: str,
        context: str,
        embedding: Optional[List[float]] = None,
    ) -> Any:
        """Insert a pattern with optional embedding. Return the new id.

        The vector is string-encoded via :func:`_encode_vector` so the
        call works whether or not the ``pgvector`` Python package is
        installed: the SQL ``::vector`` cast parses the
        ``'[1.0,2.0,3.0]'`` literal directly. With pgvector
        installed, the same string is also accepted (the adapter
        tolerates text-encoded vectors).
        """
        await self.fetchval(
            _INSERT_PATTERN_SQL,
            repo_id,
            name,
            description,
            context,
            _encode_vector(embedding),
        )

    async def find_similar_patterns(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        exclude_repo_id: Optional[int] = None,
    ) -> List[dict]:
        """Find patterns whose embedding is similar to ``embedding``.

        Returns a list of dicts with the same shape the asyncpg backend
        returns, so the manager can dispatch uniformly. ``similarity``
        is cast to ``float`` to keep the contract identical (asyncpg
        returns a ``Decimal`` for the cosine-distance expression;
        psycopg 3 returns a ``float`` already, but the cast is cheap
        and keeps the contract obvious).
        """
        rows = await self.fetch(
            _FIND_SIMILAR_PATTERNS_SQL,
            _encode_vector(embedding),
            _encode_vector(embedding),
            threshold,
            exclude_repo_id,
            exclude_repo_id,
            _encode_vector(embedding),
            limit,
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "context": row["context"],
                "repo_id": row["repo_id"],
                "repo_name": row["repo_name"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    async def update_pattern_embedding(
        self, pattern_id: int, embedding: List[float]
    ) -> None:
        """Backfill the embedding for an existing pattern."""
        await self.execute(
            _UPDATE_PATTERN_EMBEDDING_SQL,
            _encode_vector(embedding),
            pattern_id,
        )

    async def get_patterns_without_embeddings(self, limit: int = 100) -> List[dict]:
        """Return patterns whose embedding column is NULL."""
        rows = await self.fetch(_GET_PATTERNS_WITHOUT_EMBEDDINGS_SQL, limit)
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "context": row["context"],
                "repo_id": row["repo_id"],
                "repo_name": row["repo_name"],
            }
            for row in rows
        ]
