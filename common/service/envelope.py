"""Response envelope with provenance (RFC `.github#52`, 0004).

Wire shape — matches ``ApiEnvelope<T>`` in ``daroja-frontend-starter``:

    {
      "data": <payload>,
      "provenance": {
        "source": "<service-name>",
        "captured_at": "<ISO-8601 UTC>",
        "signature": "<opaque; optional integrity tag>"
      }
    }

The console's ``isLyingEnvelope`` guard treats a payload with no
``provenance`` (or missing any provenance field) as untrusted. This
module makes services emit trustworthy envelopes by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Provenance:
    """Where a response body came from, and when."""

    source: str
    captured_at: str = field(default_factory=_utc_now_iso)
    signature: str | None = None

    def as_dict(self) -> dict[str, str]:
        out = {"source": self.source, "captured_at": self.captured_at}
        if self.signature is not None:
            out["signature"] = self.signature
        return out


@dataclass(frozen=True)
class ApiEnvelope(Generic[T]):
    """Envelope wrapping a payload with its provenance."""

    data: T
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        return {"data": self.data, "provenance": self.provenance.as_dict()}


def build_provenance(source: str, signature: str | None = None) -> Provenance:
    """Construct provenance for ``source``, stamped now."""
    return Provenance(source=source, signature=signature)


def envelop(
    data: T,
    source: str,
    signature: str | None = None,
) -> dict[str, Any]:
    """Wrap ``data`` in the standard envelope. Returns a JSON-ready dict."""
    return ApiEnvelope(data=data, provenance=build_provenance(source, signature)).as_dict()
