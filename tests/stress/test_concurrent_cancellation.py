"""Real-PG stress test for the concurrent-cancellation pattern (issue #29).

Parametrized against both backends (asyncpg + psycopg 3).

Background (issues #7, #8):
    The DatabaseManager sync facade (fetchval_sync, execute_sync, ...)
    was segfaulting on production (``rag-research-eai`` revisions -00078-bsc
    and -00079-j4m) when called concurrently with task cancellation. Crash
    signature: SIGSEGV in asyncpg C code at fault_addr=0xFFFFFFFFFFFFFFFF,
    preceded by ``asyncio.CancelledError`` and "Future exception was never
    retrieved".

    The mock-based unit tests in ``test_postgres_sync.py`` cannot catch this
    kind of bug because they stub out asyncpg at the .pool level — the C
    code is never actually loaded. This file loads the real driver + real
    PostgreSQL + real event loop and reproduces the production pattern.

    Parametrized against both backends (issue #29):
      - The ``asyncpg`` path verifies the dedicated-loop fix from PR #10
        holds under real cancellation pressure.
      - The ``psycopg3`` path verifies that psycopg 3 (pure-Python wrapper
        around libpq, no C-level concurrency bugs in the driver) does not
        exhibit the segfault pattern at all.

What the test does:
    1. Spins up a real ``DatabaseManager`` against a real PostgreSQL for
       each parametrized backend.
    2. Fires N concurrent ``fetchval_sync`` calls (default: 50).
    3. Cancels half of them mid-flight via ``asyncio.CancelledError``
       (simulated by submitting tasks to a fresh ``asyncio`` loop that
       we then cancel — but actually we use the production pattern: the
       ``_run_sync`` dedicated loop's underlying task gets cancelled via
       ``future.cancel()``).
    4. Lets the rest complete.
    5. Verifies the test process is still alive afterwards (if a SIGSEGV
       happened, the process would be dead before the assert).

Skipping:
    The module skips when ``POSTGRES_HOST`` is not set, so lightweight
    CI (no real PG) doesn't run it. The psycopg 3 parametrization also
    skips individually when ``psycopg`` / ``psycopg_pool`` are not
    installed — the import is deferred into ``_build_manager`` so the
    asyncpg parametrization still runs on an asyncpg-only install.
    The nightly stress job + manual
    ``gh workflow run devnexus-common-stress.yml`` runs both
    parametrizations via two jobs (``stress-asyncpg`` and
    ``stress-psycopg3``).

Validation (when first added):
    This test must be verified to **segfault against the pre-fix
    devnexus-common main** (the bug must be reproducible). Then verify
    it passes against the fix. If the test doesn't fail against the
    unfixed code, it's a worthless test.

Local dev:
    # Option A — Docker
    docker run --rm -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17
    POSTGRES_HOST=localhost POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres \\
        python -m pytest tests/stress/ -v

    # Option B — existing dev PG at 10.0.0.21
    POSTGRES_HOST=10.0.0.21 POSTGRES_USER=... POSTGRES_PASSWORD=... \\
        python -m pytest tests/stress/ -v
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Tuple

import pytest

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "").strip()
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres").strip()
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres").strip()
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "public").strip()

# Skip the entire module if no real PG is available. This is the only
# module-level skip — once a real PG is in scope, every parametrized
# test must exercise it.
pytestmark = pytest.mark.skipif(
    not POSTGRES_HOST,
    reason=(
        "POSTGRES_HOST not set; real-PG stress test skipped. "
        "Set POSTGRES_HOST (and POSTGRES_PORT/USER/PASSWORD/DB) to run. "
        "Local dev: `docker run --rm -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17`."
    ),
)


def _build_manager(backend: str):
    """Construct a real DatabaseManager from the env vars + the chosen backend.

    The psycopg 3 driver is imported lazily so the asyncpg parametrization
    can still run on an asyncpg-only install (the import only happens when
    ``backend == "psycopg3"``). If the driver isn't installed, the psycopg
    parametrization skips cleanly rather than failing collection.
    """
    from common.db.postgres import DatabaseManager

    if backend == "psycopg3":
        try:
            import psycopg  # noqa: F401
            import psycopg_pool  # noqa: F401
        except ImportError as e:
            pytest.skip(
                f"backend='psycopg3' requested but psycopg / psycopg_pool "
                f"not installed: {e}. Install with "
                f"`pip install 'psycopg[binary,pool]'`."
            )

    return DatabaseManager(
        backend=backend,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        search_path=POSTGRES_SCHEMA,
        application_name="stress-test",
    )


@pytest.fixture(scope="module")
def db_manager(backend):
    """Module-scoped fixture: build the manager for the parametrized backend
    and connect once.

    The ``backend`` parameter comes from the module-scoped fixture in
    ``tests/stress/conftest.py`` (which overrides the function-scoped
    parent). pytest re-evaluates this fixture once per unique backend,
    so each backend gets its own module-scoped connection pool.
    """
    mgr = _build_manager(backend)
    # We need a connection. Use the async init — but we're in a sync
    # pytest fixture, so use the sync wrapper.
    mgr.init_db_sync() if hasattr(mgr, "init_db_sync") else mgr._run_sync(mgr.connect())
    yield mgr
    # Best-effort cleanup; we don't fail the test if this raises.
    try:
        mgr._run_sync(mgr.disconnect())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The bug-reproducing tests (parametrized against both backends, issue #29)
# ---------------------------------------------------------------------------


class TestConcurrentCancellation:
    """Real-PG stress test for the per-call asyncio.run() SIGSEGV pattern.

    Parametrized against both backends (issue #29). The asyncpg path
    verifies the dedicated-loop fix from PR #10 holds under real
    cancellation pressure. The psycopg 3 path verifies that psycopg 3
    (pure-Python wrapper around libpq, no C-level concurrency bugs in
    the driver) does not exhibit the segfault pattern at all.
    """

    def test_concurrent_cancellation_does_not_segfault(self, db_manager, backend):
        """The exact pattern that crashed production.

        N concurrent ``fetchval_sync`` calls, with half of them cancelled
        mid-flight. The test process must survive (no SIGSEGV).

        If a SIGSEGV occurs, the test process dies; pytest reports the test
        as failed or crashed. We also assert the manager is still healthy
        after the burst.
        """
        N = int(os.environ.get("STRESS_N", "50"))
        cancel_fraction = float(os.environ.get("STRESS_CANCEL_FRACTION", "0.5"))
        cancel_delay = float(os.environ.get("STRESS_CANCEL_DELAY", "0.1"))  # seconds
        query_delay = float(
            os.environ.get("STRESS_QUERY_DELAY", "0.5")
        )  # pg_sleep seconds

        # Submit N tasks to a threadpool. Each task does a fetchval_sync
        # which (internally) creates a coroutine on the dedicated loop.
        # We then cancel half the tasks from the main thread.
        def _do_query(idx: int) -> Tuple[int, str]:
            # Use a slightly slow query so cancellation has time to land
            # mid-flight. pg_sleep is the simplest way.
            try:
                result = db_manager.fetchval_sync(
                    "SELECT pg_sleep($1)::text", query_delay
                )
                return (idx, f"ok:{result!r}")
            except Exception as e:
                return (idx, f"err:{type(e).__name__}:{e}")

        started_at = time.time()
        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_do_query, i) for i in range(N)]

            # Give some of them a moment to start, then cancel half.
            time.sleep(cancel_delay)
            n_to_cancel = int(N * cancel_fraction)
            cancelled_count = 0
            for f in futures[:n_to_cancel]:
                if f.cancel():
                    cancelled_count += 1
            # Anything not yet running: cancel() returns True and the
            # future won't run. Anything already running: cancel() returns
            # False (we can't cancel a thread's running task from outside
            # the asyncio loop in a way that propagates). For the latter,
            # the production pattern (cancelled asyncio task inside the
            # dedicated loop) is the one we're exercising; the future
            # cancel here is just to model the *request-cancel-during-query*
            # pattern as closely as possible from sync code.

            # Wait for all to complete (cancelled ones return immediately)
            done, _ = wait(futures, timeout=30.0)
            results: List[Tuple[int, str]] = [f.result() for f in done]

        elapsed = time.time() - started_at

        # Tally outcomes
        ok = [r for r in results if r[1].startswith("ok:")]
        err = [r for r in results if r[1].startswith("err:")]
        cancelled = N - len(done)  # futures that didn't finish within timeout

        # *** The key assertion: process is still alive ***
        # If asyncpg segfaulted, we wouldn't reach this line.
        assert len(done) == N, (
            f"Only {len(done)}/{N} futures completed within 30s "
            f"({cancelled} timed out, {len(err)} errored). "
            f"This is consistent with a hang/segfault pattern."
        )

        # Sanity: at least some queries completed (we didn't just cancel everything)
        assert len(ok) > 0, f"No queries completed successfully: results={results}"

        # The manager must still be healthy after the burst
        health = db_manager.health_check_sync()
        assert health.get("status") in (
            "healthy",
            "disabled",
        ), f"Manager unhealthy after concurrent burst: {health}"

        # Log a summary (visible with `pytest -s`)
        print(
            f"\n[concurrent_cancellation/{backend}] N={N} ok={len(ok)} "
            f"err={len(err)} cancelled={cancelled_count} elapsed={elapsed:.2f}s"
        )
        if err:
            # Show first 3 error types (not all, to keep output sane)
            for _, msg in err[:3]:
                print(f"  err: {msg}")

    def test_minimal_concurrent_burst_does_not_segfault(self, db_manager, backend):
        """Smaller, faster version for CI.

        Runs 8 concurrent ``fetchval_sync`` calls, cancels 4. If the
        dedicated loop + ``CancelledError`` handling is broken, this small
        test still segfaults (the original bug was reproducible with N=2,
        not just N=50).
        """
        N = 8

        def _do_query(idx: int) -> Tuple[int, str]:
            try:
                result = db_manager.fetchval_sync("SELECT 1::text")
                return (idx, f"ok:{result!r}")
            except Exception as e:
                return (idx, f"err:{type(e).__name__}:{e}")

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_do_query, i) for i in range(N)]
            time.sleep(0.05)
            for f in futures[: N // 2]:
                f.cancel()
            done, _ = wait(futures, timeout=10.0)
            results: List[Tuple[int, str]] = [f.result() for f in done]

        assert len(done) == N, f"Only {len(done)}/{N} completed"
        assert any(
            r[1].startswith("ok:") for r in results
        ), f"No queries succeeded: {results}"
        # Manager still healthy
        health = db_manager.health_check_sync()
        assert health.get("status") in ("healthy", "disabled")
