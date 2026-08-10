"""asyncpg implementation of the ``DatabaseBackend`` Protocol.

This module is a pure extraction of the asyncpg-specific code that used
to live inline in ``common.db.postgres``. The lifecycle (connect /
disconnect / SSL retry / pgvector probe), the basic query methods
(execute / fetch / fetchrow / fetchval), and the patterns helpers
(insert_pattern_with_embedding, find_similar_patterns,
update_pattern_embedding, get_patterns_without_embeddings) all moved
here verbatim. The ``DatabaseManager`` (common.db.manager) holds an
instance of this class and delegates to it.

The shape of this file is intentionally close to the old ``postgres.py``
so a reviewer can diff method-by-method. Comments that explained
*why* a particular pattern was used (SSL retry, pgvector probe, etc.)
were preserved.

Why the SQL strings live here, not on the manager:
  - The patterns helpers use asyncpg-specific syntax: ``$1::vector``
    casts, ``ON CONFLICT (repo_id, name) DO UPDATE``, the
    ``1 - (embedding <=> $1::vector)`` cosine-distance idiom. Moving
    them behind the Protocol means a future psycopg 3 backend only has
    to reimplement the same four methods — the manager never sees the
    raw SQL.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import asyncpg

from common.db.backends.base import BackendConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL constants for the patterns helpers.
#
# Kept as module-level strings so the manager and any other caller can
# see exactly which queries the backend owns. The ``$1::vector`` cast is
# asyncpg-specific — psycopg 3 will need its own copy of these strings
# (or a translator at the boundary, which is the point of issue #27).
# ---------------------------------------------------------------------------

_INSERT_PATTERN_SQL = """
    INSERT INTO patterns (repo_id, name, description, context, embedding)
    VALUES ($1, $2, $3, $4, $5::vector)
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
        1 - (p.embedding <=> $1::vector) as similarity
    FROM patterns p
    JOIN repositories r ON p.repo_id = r.id
    WHERE p.embedding IS NOT NULL
        AND 1 - (p.embedding <=> $1::vector) >= $2
        AND ($3::integer IS NULL OR p.repo_id != $3)
    ORDER BY p.embedding <=> $1::vector
    LIMIT $4
"""

_UPDATE_PATTERN_EMBEDDING_SQL = (
    "UPDATE patterns SET embedding = $1::vector WHERE id = $2"
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
    LIMIT $1
"""


def _build_ssl_arg(ssl_mode: str, ssl_no_verify: bool) -> Any:
    """Translate the ``POSTGRES_SSLMODE`` value into an asyncpg ``ssl`` kwarg.

    ``disable`` / ``false`` / ``0`` -> ``False`` (no TLS).
    ``require`` / ``true`` / ``1``   -> ``ssl.create_default_context()``,
                                       optionally with cert verification
                                       disabled when ``ssl_no_verify`` is set.
    anything else                    -> ``None`` (let asyncpg negotiate).

    Kept as a module-level helper so it's testable in isolation and so
    the retry path in ``connect()`` can reuse the same logic.
    """
    if ssl_mode in ("disable", "false", "0"):
        return False
    if ssl_mode in ("require", "true", "1"):
        if ssl_no_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return ssl.create_default_context()
    return None


class AsyncpgBackend:
    """asyncpg-backed implementation of the ``DatabaseBackend`` Protocol.

    The manager constructs one of these in its ``__init__`` and calls
    ``connect(config)`` from its own ``connect()`` (which then syncs
    ``self.pool`` to the backend's pool so the manager's own helpers
    — ``acquire()``, ``execute()``, ``fetch()``, etc. — keep working
    with the same pool reference the tests set directly).

    The instance is a regular class, not a dataclass: it holds mutable
    driver state (``self.pool``) that changes over the lifetime of
    the connection, and the Protocol methods are all coroutines.
    """

    def __init__(self) -> None:
        # Set by connect(); None until the pool is open.
        self.pool: Optional[asyncpg.Pool] = None
        # Stored so the manager can read SSL settings if it ever needs
        # to (it currently doesn't — the backend owns all of this).
        self._config: Optional[BackendConfig] = None

    def _build_setup(self, config: BackendConfig):
        """Return an asyncpg ``setup`` coroutine that applies the
        per-connection settings that ``server_settings`` can't express
        correctly (currently just ``search_path``).

        asyncpg's ``server_settings`` sends each value as a single
        quoted string in the startup packet, which Postgres treats as
        a single value. For GUCs that accept a comma-separated
        identifier list (``search_path``, ``session_preload_libraries``,
        ``shared_preload_libraries``, etc.) this turns
        ``"a, b"`` into a single bogus schema named ``"a, b"`` rather
        than the list ``[a, b]`` the caller wanted.

        The fix is to run ``SET search_path TO a, b`` as raw SQL on
        each new connection via the ``setup`` callback — Postgres'
        SQL parser correctly treats the unquoted, comma-separated
        identifiers as a list.

        The value of ``config.search_path`` is operator-controlled
        (it comes from the ``POSTGRES_SEARCH_PATH`` env var, not from
        a SQL string the caller provided), so direct f-string
        interpolation is safe. We also defensively reject anything
        that contains characters which would be unsafe to splice
        into a SET statement (quotes, semicolons, newlines).
        """
        import re

        search_path = (config.search_path or "").strip()
        if not search_path:
            return None
        if not re.fullmatch(r"[A-Za-z_][\w]*(?:[ ,]+[A-Za-z_][\w]*)*", search_path):
            raise ValueError(
                "POSTGRES_SEARCH_PATH must be a comma-separated list of "
                "valid SQL identifiers; got "
                f"{search_path!r}"
            )

        async def _setup(conn: asyncpg.Connection) -> None:
            # The schema list is operator-controlled, not user-supplied
            # SQL, so f-string interpolation is safe (and required —
            # asyncpg's parameterized queries would quote the whole
            # thing and reproduce the server_settings bug).
            await conn.execute(f"SET search_path TO {search_path}")

        return _setup

    # ------------------------------------------------------------------
    # Protocol: identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "asyncpg"

    # ------------------------------------------------------------------
    # Protocol: lifecycle
    # ------------------------------------------------------------------

    async def connect(self, config: BackendConfig) -> None:
        """Open the asyncpg pool with SSL retry and pgvector probe.

        Mirrors the original DatabaseManager.connect() flow:
          1. Up to 6 attempts with exponential backoff (1s, 2s, 4s, ...)
             capped at 30s.
          2. On the first SSL/cert/TLS error, retry once with
             ``ssl=False``. This handles the dev/prod skew where the
             local Postgres accepts plaintext but a misconfigured
             cloud instance demands TLS.
          3. After the pool is open, run a ``SELECT extversion FROM
             pg_extension WHERE extname = 'vector'`` probe so we log
             clearly when pgvector is missing (it's required for the
             patterns helpers).
        """
        self._config = config
        max_attempts = 6
        delay = 1

        ssl_arg = _build_ssl_arg(config.ssl_mode, config.ssl_no_verify)

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    f"Connecting to PostgreSQL at {config.host}:{config.port}/"
                    f"{config.database} (attempt {attempt}/{max_attempts})"
                )
                logger.info(f"asyncpg ssl argument: {ssl_arg!r}")

                try:
                    self.pool = await asyncio.wait_for(
                        asyncpg.create_pool(
                            host=config.host,
                            port=config.port,
                            database=config.database,
                            user=config.user,
                            password=config.password,
                            ssl=ssl_arg,
                            min_size=config.min_size,
                            max_size=config.max_size,
                            command_timeout=config.statement_timeout_ms / 1000.0,
                            max_inactive_connection_lifetime=300,
                            max_queries=1000,
                            # NOTE: search_path is set via the `setup`
                            # callback below, NOT via server_settings.
                            # asyncpg's server_settings sends each value
                            # as a single quoted string in the startup
                            # packet (e.g. `SET search_path 'a, b'`),
                            # which Postgres stores as a single schema
                            # name containing a comma — not a list.
                            # The setup callback runs raw SQL
                            # (`SET search_path TO a, b`) which Postgres
                            # correctly parses as an identifier list.
                            # See issue: dev-nexus PR #1077 / dev-nexus
                            # `relation "repositories" does not exist`
                            # when POSTGRES_SEARCH_PATH contained a
                            # comma. application_name and statement_timeout
                            # are still safe as server_settings (no
                            # comma / no list parsing).
                            setup=self._build_setup(config),
                            server_settings={
                                "application_name": config.application_name,
                                "statement_timeout": str(config.statement_timeout_ms),
                            },
                        ),
                        timeout=10,
                    )
                except Exception as e:
                    err_text = str(e).lower()
                    logger.error(f"asyncpg.create_pool failed (ssl={ssl_arg!r}): {e}")
                    if ssl_arg not in (False,) and (
                        "ssl" in err_text
                        or "certificate" in err_text
                        or "tls" in err_text
                    ):
                        logger.warning(
                            "Detected SSL-related failure, retrying once with ssl=False"
                        )
                        try:
                            self.pool = await asyncio.wait_for(
                                asyncpg.create_pool(
                                    host=config.host,
                                    port=config.port,
                                    database=config.database,
                                    user=config.user,
                                    password=config.password,
                                    ssl=False,
                                    min_size=config.min_size,
                                    max_size=config.max_size,
                                    command_timeout=(
                                        config.statement_timeout_ms / 1000.0
                                    ),
                                    max_inactive_connection_lifetime=300,
                                    max_queries=1000,
                                    setup=self._build_setup(config),
                                    server_settings={
                                        "application_name": config.application_name,
                                        "statement_timeout": str(
                                            config.statement_timeout_ms
                                        ),
                                    },
                                ),
                                timeout=10,
                            )
                        except Exception as e2:
                            logger.error(f"Fallback (ssl=False) also failed: {e2}")
                            raise
                    else:
                        raise

                # Verify pgvector extension is available
                async with self.pool.acquire() as conn:
                    result = await conn.fetchrow(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    if result:
                        logger.info(
                            f"pgvector extension detected: v{result['extversion']}"
                        )
                    else:
                        logger.warning("pgvector extension not found")

                logger.info("Database connection pool established")
                return

            except Exception as e:
                logger.error(
                    f"Failed to connect to PostgreSQL (attempt {attempt}): {e}"
                )
                try:
                    if self.pool is not None:
                        await self.pool.close()
                        self.pool = None
                except Exception:
                    pass

                if attempt == max_attempts:
                    logger.error(
                        "Exceeded max connection attempts to PostgreSQL — giving up"
                    )
                    raise

                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def disconnect(self) -> None:
        """Close the connection pool. Idempotent."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    # ------------------------------------------------------------------
    # Protocol: health, acquire
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Return ``{status, version, pgvector_version, host, database}``.

        The manager layers on its pool-stats saturation metrics; this
        method only reports the driver-visible state.
        """
        if self.pool is None:
            return {"status": "disconnected", "message": "No active connection pool"}

        try:
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                pgvector = await conn.fetchrow(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
            host = self._config.host if self._config else None
            database = self._config.database if self._config else None
            return {
                "status": "healthy",
                "version": version.split(",")[0],
                "pgvector_version": pgvector["extversion"] if pgvector else None,
                "host": host,
                "database": database,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    @asynccontextmanager
    async def acquire(self):
        """Return an async context manager yielding an asyncpg connection.

        The manager wraps this with pool-stats timing; this method just
        delegates to the underlying pool.
        """
        if self.pool is None:
            raise RuntimeError("Database not connected")
        async with self.pool.acquire() as conn:
            yield conn

    # ------------------------------------------------------------------
    # Protocol: basic query helpers
    # ------------------------------------------------------------------

    async def execute(self, query: str, *args: Any) -> str:
        """Run a single execute against a freshly-acquired connection."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def executemany(self, query: str, args: List[Any]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args)

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
        await self.fetchval(
            _INSERT_PATTERN_SQL, repo_id, name, description, context, embedding
        )

    async def find_similar_patterns(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        exclude_repo_id: Optional[int] = None,
    ) -> List[dict]:
        rows = await self.fetch(
            _FIND_SIMILAR_PATTERNS_SQL,
            embedding,
            threshold,
            exclude_repo_id,
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
        await self.execute(_UPDATE_PATTERN_EMBEDDING_SQL, embedding, pattern_id)

    async def get_patterns_without_embeddings(self, limit: int = 100) -> List[dict]:
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
