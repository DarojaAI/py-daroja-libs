"""TENANCY MODULE — PLANNED, NOT YET IMPLEMENTED.

This module is reserved for tenant/context primitives used across the
content/data tier of the DarojaAI org. Adding the module file as a
doc-only marker so the planned home is visible to maintainers and
so the API surface can be designed (and imports stubbed) before the
``daroja-tenancy`` service emits its first stable JWKS.

Planned public API (subject to revision after the read-pass converges):

  * ``TenantContext`` — dataclass carrying
    ``(user_id, counterparty_id, client_id, project_id, roles)``
    extracted from a verified JWT. The unit that flows through every
    request after auth.

  * ``scope_query(table, ctx)`` — helper that appends the
    ``(counterparty_id, client_id, project_id)`` WHERE filter to a
    raw query, for code paths that run against databases without
    Postgres RLS. Where RLS *is* enforced, prefer the DB policy via
    ``DatabaseManager.search_path`` and skip this helper.

  * ``require_role(role)`` — FastAPI dependency; 401s unless the
    active ``TenantContext`` holds ``role`` at the active project
    or at client level.

  * JWT verification helper that pulls the JWKS from the
    ``daroja-tenancy`` service and validates the bearer token per
    request.

Architecture (per ``darojaai_architect`` OPEN_QUESTIONS.md Q20, 2026-08):

    * ``daroja-tenancy`` (new shared service) owns authoritative
      tenant state — counterparties, clients, projects, users,
      memberships, roles, magic-link auth, audit. It issues JWTs with
      RS256 signatures and exposes a JWKS endpoint.

    * rag_research_tool / document-pipeline / research-orchestrator /
      intelligent-feed all verify the JWT, populate a ``TenantContext``,
      and thread ``(counterparty, client, project)`` through downstream
      hops. ``rag_research_tool`` enforces isolation at the data layer
      via Postgres RLS keyed on the three columns.

    * The activist factory in ``intel.activation.factory`` keeps its
      product-slug keying. Tenancy context arrives at activators as a
      separate parameter; we do NOT replace the slug.

DO NOT add the ``daroja-tenancy`` Python client here — keep this module
pure plumbing. The service owns state, this module owns verification +
plumbing helpers only.

WHAT TO DO NEXT (in order, after ``daroja-tenancy`` is provisioned):

  1. Wait for a stable ``daroja-tenancy`` JWKS URL.
  2. Implement ``TenantContext`` dataclass + the JWT verifier.
  3. Add ``scope_query`` against the Postgres backend in
     ``common.db.manager``; piggy-back on the existing ``search_path``
     parameter for schema-level switching.
  4. Add ``require_role`` as a FastAPI dependency.
"""
