"""Regression test for DatabaseManager._run_sync() called from a
running event loop.

Bug history:
  - 2026-06-14 audit of rag_research_tool/eai found that
    FastAPI routers (async def endpoints) calling sync DB
    methods (db.fetch_sync, db.execute_sync, ...) hit a
    RuntimeError because the calling thread has a running
    event loop.
  - The error was caught by the routers' broad `except Exception`
    handlers, which fell back to a disk cache. This made the
    pipeline appear successful (HTTP 200 + completion message)
    while *nothing* was actually persisted to Postgres.
  - Root cause: the original _run_sync had a branch that
    rejected calls from a running event loop, with a docstring
    suggesting callers switch to the async methods instead.
    The routers hadn't been switched, so they hit the dead end
    every request.

The fix:
  - _run_sync now schedules on the *dedicated* DB loop
    (a separate event loop running in a daemon thread) and
    blocks the caller via future.result(). The caller's
    event loop is technically blocked during the DB call,
    but the round-trip is bounded (a few ms for typical PG)
    and the behavior is correct (the operation actually runs,
    the result is returned).

This test asserts:
  1. _run_sync works from a thread with no event loop
     (the original happy path).
  2. _run_sync works from a thread with a running event loop
     (the new path; previously raised RuntimeError).
  3. _run_sync works from inside an `async def` running in
     the main event loop (the FastAPI router pattern).
  4. The scheduled coroutine actually executes (not just
     silently returns None).
  5. Concurrent calls from a running event loop don't deadlock
     (the dedicated loop handles them serially, not in parallel
     within the same loop, but multiple calls don't block each
     other on shared state).
"""

from __future__ import annotations

import asyncio
import threading
import time


from common.db.postgres import DatabaseManager


# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------


class _FakePool:
    """Stand-in for asyncpg.Pool that records what was awaited on it
    and returns a canned row. Used so the test doesn't need a real PG."""

    def __init__(self) -> None:
        self.awaits: list[tuple[str, tuple]] = []
        self._counter = 0

    def acquire(self):
        """Return an async context manager that yields a fake conn."""
        return _FakeConnection(self)

    async def fetch(self, query: str, *args):
        self.awaits.append((query, args))
        self._counter += 1
        return [{"id": f"row-{self._counter}"}]

    async def fetchrow(self, query: str, *args):
        self.waits_count = getattr(self, "waits_count", 0) + 1
        return {"id": "single-row"}

    async def execute(self, query: str, *args):
        self.awaits.append((query, args))
        return "INSERT 0 1"


class _FakeConnection:
    """Stand-in for an asyncpg.Connection acquired from a pool.
    Passes the fetch/fetchrow/execute calls through to the parent pool
    so the test assertions still see the queries recorded."""

    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def fetch(self, query: str, *args):
        return await self.pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        return await self.pool.fetchrow(query, *args)

    async def execute(self, query: str, *args):
        return await self.pool.execute(query, *args)


def _make_manager_with_fake_pool() -> tuple[DatabaseManager, _FakePool]:
    """Construct a DatabaseManager and inject a fake pool. We don't
    call connect() — we patch self.pool and skip the lazy-init
    machinery for this unit test."""
    fake = _FakePool()
    mgr = DatabaseManager(
        host="localhost.example.invalid",
        port=5432,
        database="x",
        user="u",
        password="p",
    )
    mgr.pool = fake
    # _ensure_loop will start a real daemon thread for the loop,
    # but our fake pool's async methods will be executed on it.
    return mgr, fake


# ---------------------------------------------------------------------------
# Test 1: _run_sync from a thread with no event loop
# ---------------------------------------------------------------------------


def test_run_sync_from_thread_without_event_loop():
    """The original happy path. Worker thread, no loop, scheduled
    onto the dedicated DB loop, future.result() blocks until done."""
    mgr, fake = _make_manager_with_fake_pool()

    def worker():
        # We're in a thread with no event loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = mgr.fetch_sync("SELECT 1")
            assert result == [{"id": "row-1"}]
        finally:
            loop.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "worker thread deadlocked"
    assert len(fake.awaits) == 1, f"expected 1 query, got {len(fake.awaits)}"


# ---------------------------------------------------------------------------
# Test 2: _run_sync from a thread with a running event loop (the bug)
# ---------------------------------------------------------------------------


def test_run_sync_from_thread_with_running_event_loop():
    """The new path. Caller has a running event loop (e.g. a
    FastAPI request handler). Previously this raised RuntimeError
    and silently fell back to the disk cache. Now it should work."""
    mgr, fake = _make_manager_with_fake_pool()
    started = threading.Event()
    proceed = threading.Event()
    errors: list[BaseException] = []

    def worker():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run_in_loop(mgr, fake, started, proceed))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    async def _run_in_loop(mgr, fake, started, proceed):
        started.set()
        # We're inside a running event loop now. _run_sync used to
        # raise RuntimeError here; now it should work.
        try:
            result = mgr.fetch_sync("SELECT 1")
            assert result == [{"id": "row-1"}]
            return result
        finally:
            proceed.set()

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(timeout=5), "worker thread did not start"
    assert proceed.wait(timeout=5), "worker thread deadlocked in fetch_sync"
    t.join(timeout=5)
    assert not t.is_alive(), "worker thread is hung"
    assert not errors, f"worker raised: {errors!r}"
    assert len(fake.awaits) == 1, f"expected 1 query, got {len(fake.awaits)}"


# ---------------------------------------------------------------------------
# Test 3: _run_sync from inside an async def (the FastAPI router pattern)
# ---------------------------------------------------------------------------


def test_run_sync_from_async_def():
    """The exact pattern in rag_research_tool's routers. An
    async def (the FastAPI endpoint) calls mgr.fetch_sync() directly
    (sync) — not awaited. Previously this raised RuntimeError."""
    mgr, fake = _make_manager_with_fake_pool()
    errors: list[BaseException] = []

    async def handler():
        # mgr.fetch_sync is sync; called from inside a running loop.
        result = mgr.fetch_sync("SELECT 1")
        return result

    try:
        result = asyncio.run(handler())
        assert result == [{"id": "row-1"}]
    except BaseException as e:  # noqa: BLE001
        errors.append(e)

    assert not errors, f"handler raised: {errors!r}"
    assert len(fake.awaits) == 1, f"expected 1 query, got {len(fake.awaits)}"


# ---------------------------------------------------------------------------
# Test 4: scheduled coroutine actually executes (no silent skip)
# ---------------------------------------------------------------------------


def test_run_sync_actually_runs_coroutine():
    """Make sure _run_sync doesn't silently return None or skip
    the coroutine. The previous bug (the audit root cause) was
    that the runtime check rejected the call; we want to
    confirm the fix actually runs the work."""
    mgr, fake = _make_manager_with_fake_pool()
    result = mgr.fetch_sync("SELECT * FROM source_documents")
    assert result is not None
    assert result == [{"id": "row-1"}]
    assert len(fake.awaits) == 1
    assert "source_documents" in fake.awaits[0][0]


# ---------------------------------------------------------------------------
# Test 5: concurrent calls from a running loop don't deadlock
# ---------------------------------------------------------------------------


def test_concurrent_run_sync_from_running_loop():
    """Multiple sync calls from a running event loop, in quick
    succession. They serialize on the dedicated DB loop (single
    thread), but each completes within a bounded time."""
    mgr, fake = _make_manager_with_fake_pool()

    async def handler():
        results = []
        for _ in range(5):
            r = mgr.fetch_sync("SELECT 1")
            results.append(r)
        return results

    start = time.time()
    results = asyncio.run(handler())
    elapsed = time.time() - start
    assert len(results) == 5
    assert all(r == [{"id": f"row-{i + 1}"}] for i, r in enumerate(results))
    assert len(fake.awaits) == 5
    # Each call should be fast (a few ms each, not seconds).
    # Allow generous slack for CI overhead.
    assert elapsed < 5.0, f"5 calls took {elapsed:.2f}s, suspected deadlock"
