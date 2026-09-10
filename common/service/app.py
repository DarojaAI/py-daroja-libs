"""Service app factory (RFC `.github#52`, 0004).

``create_service_app`` wires the org wire-contract standard:

  * ``GET /healthz``       — liveness
  * ``GET /healthz/ready`` — readiness (delegates to a caller probe)
  * ``/<domain>/...``      — the domain prefix routers should mount under
  * OpenAPI ``title``/``version`` pinning
  * CORS defaults

Interiors stay per-service: callers mount their own routers under
``f"/{domain}"`` and register their own auth.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

# A readiness probe returns a JSON-ready dict on success, or raises to
# signal 503. Callers pass a zero-arg async callable.
ReadinessProbe = Callable[[], Awaitable[dict[str, Any]]]

DEFAULT_CORS_ORIGINS = ["*"]


def create_service_app(
    title: str,
    version: str,
    domain: str,
    *,
    ready_probe: ReadinessProbe | None = None,
    cors_origins: list[str] | None = None,
    description: str | None = None,
) -> FastAPI:
    """Build a FastAPI app carrying the org's standard surface.

    Parameters
    ----------
    title, version:
        Pinned into the OpenAPI doc (stable for generated clients).
    domain:
        Route prefix services mount routers under (e.g. ``"tenancy"``
        -> ``/tenancy/...``). Exposed on ``app.state.domain``.
    ready_probe:
        Optional async callable; returns a dict on readiness success,
        raises to signal not-ready (mapped to 503).
    cors_origins:
        CORS allow-list; defaults to ``["*"]`` for private/operator
        surfaces. Tighten per service.
    description:
        OpenAPI description.
    """
    domain = domain.strip().strip("/").lower()
    if not domain:
        raise ValueError("create_service_app: domain must be non-empty")

    app = FastAPI(
        title=title,
        version=version,
        description=description,
        openapi_tags=[{"name": domain}],
    )
    app.state.domain = domain
    app.state.service_title = title
    app.state.service_version = version

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins if cors_origins is not None else DEFAULT_CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        """Liveness — 200 whenever the process is up."""
        return {"status": "ok"}

    @app.get("/healthz/ready", tags=["health"])
    async def healthz_ready(request: Request) -> dict[str, Any]:
        """Readiness — 200 when deps reachable, 503 otherwise."""
        probe: ReadinessProbe | None = request.app.state.ready_probe
        if probe is None:
            # No probe wired: process-up is the readiness signal.
            return {"status": "ready"}
        try:
            detail = await probe()
        except Exception as exc:  # noqa: BLE001 - surface any probe failure as 503
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc) or "not ready",
            ) from exc
        return {"status": "ready", **detail}

    app.state.ready_probe = ready_probe
    return app
