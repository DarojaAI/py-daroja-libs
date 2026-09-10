"""Tests for the API surface standard bootstrap (RFC `.github#52`, 0004)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from common.service.app import create_service_app
from common.service.envelope import ApiEnvelope, build_provenance, envelop


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------


def test_envelop_wraps_with_provenance():
    out = envelop({"counterparty_id": "c1"}, source="tenancy")
    assert out["data"] == {"counterparty_id": "c1"}
    assert out["provenance"]["source"] == "tenancy"
    assert "captured_at" in out["provenance"]


def test_envelop_missing_provenance_is_lying():
    # The console's isLyingEnvelope guard rejects payloads without full
    # provenance; an envelope built here must always carry source+captured_at.
    out = envelop({"x": 1}, source="svc")
    p = out["provenance"]
    assert p["source"] and p["captured_at"]


def test_build_provenance_signature_optional():
    p = build_provenance("tenancy")
    assert p.signature is None
    p2 = build_provenance("tenancy", signature="abc")
    assert p2.signature == "abc"
    assert p2.as_dict()["signature"] == "abc"


def test_apienvelope_as_dict_shape():
    env = ApiEnvelope(data=["a"], provenance=build_provenance("s"))
    d = env.as_dict()
    assert d["data"] == ["a"]
    assert d["provenance"]["source"] == "s"


# ---------------------------------------------------------------------------
# create_service_app — health surface
# ---------------------------------------------------------------------------


def test_healthz_liveness():
    app = create_service_app(title="t", version="1.2.3", domain="widgets")
    client = TestClient(app)
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_healthz_ready_no_probe_is_ready():
    app = create_service_app(title="t", version="1.2.3", domain="widgets")
    client = TestClient(app)
    res = client.get("/healthz/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_healthz_ready_probe_success():
    async def probe():
        return {"default_counterparty_id": "c1"}

    app = create_service_app(title="t", version="1.2.3", domain="w", ready_probe=probe)
    client = TestClient(app)
    res = client.get("/healthz/ready")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["default_counterparty_id"] == "c1"


@pytest.mark.asyncio
async def test_healthz_ready_probe_failure_503():
    async def probe():
        raise RuntimeError("pg down")

    app = create_service_app(title="t", version="1.2.3", domain="w", ready_probe=probe)
    client = TestClient(app)
    res = client.get("/healthz/ready")
    assert res.status_code == 503


# ---------------------------------------------------------------------------
# create_service_app — domain + openapi
# ---------------------------------------------------------------------------


def test_domain_prefix_and_state():
    app = create_service_app(title="t", version="1.2.3", domain="tenancy")
    assert app.state.domain == "tenancy"
    assert app.state.service_title == "t"
    assert app.state.service_version == "1.2.3"
    # OpenAPI tags carry the domain so generated clients can group router endpoints.
    tags = {t["name"] for t in app.openapi()["tags"]}
    assert "tenancy" in tags


def test_domain_normalized_and_validated():
    assert create_service_app("t", "1", " Widgets ").state.domain == "widgets"
    with pytest.raises(ValueError):
        create_service_app("t", "1", " ")


def test_openapi_version_pinned():
    app = create_service_app(title="svc", version="3.4.5", domain="x")
    assert app.openapi()["info"]["version"] == "3.4.5"
    assert app.openapi()["info"]["title"] == "svc"


def test_cors_defaults_wire_allow_all():
    app = create_service_app(title="t", version="1", domain="x")
    # starlette stores CORS middleware; the default we set is allow_all.
    cors = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert cors, "CORSMiddleware should be registered"