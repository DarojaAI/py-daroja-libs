"""API surface standard — service bootstrap (RFC `.github#52`, 0004).

Provisioning vehicle for the org's wire-contract standard. Import
``create_service_app`` and a new FastAPI service gets, for free:

  * ``GET /healthz``        — liveness (always 200 when the process is up)
  * ``GET /healthz/ready``  — readiness (503 until the service's own probe passes)
  * ``/<domain>/...``       — stable per-domain route prefix
  * envelope provenance     — every JSON response wrapped as
                              ``{"data": ..., "provenance": {source, captured_at, signature}}``
                              matching ``ApiEnvelope<T>`` / ``isLyingEnvelope``
                              in ``daroja-frontend-starter``.
  * CORS defaults + OpenAPI ``title``/``version`` pinning

The standard is deliberately **wire-contract only** — service interiors
(routers, models, auth) stay per-service. See RFC `.github#52` for the
explicit non-goals.

FastAPI is an optional dependency:

    pip install "py-daroja-libs[service]"
"""

from common.service.app import create_service_app
from common.service.envelope import ApiEnvelope, build_provenance, envelop

__all__ = [
    "ApiEnvelope",
    "build_provenance",
    "create_service_app",
    "envelop",
]
