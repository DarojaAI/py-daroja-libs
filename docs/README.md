# py-daroja-libs documentation

This directory hosts cross-cutting documentation for the
DarojaAI/py-daroja-libs repository. The Python utilities themselves
live in `common/` (and are installable via `pip install -e .`) — see
the top-level [README.md](../README.md) for the module catalog.

## Contents

| File | Purpose |
|---|---|
| [STANDARDS_COMPLIANCE_FRAMEWORK.md](STANDARDS_COMPLIANCE_FRAMEWORK.md) | Which standards apply to backend, frontend, and shared repositories, with compliance requirements and verification procedures. |

## Why these docs live here

`STANDARDS_COMPLIANCE_FRAMEWORK.md` describes compliance patterns that
**every DarojaAI project should follow** — it is the meta-document that
specifies which standards (API documentation, breaking-changes tracking,
type-generation, A2A manifest endpoints, etc.) apply to which repo
type. Its natural home is alongside the cross-repo shared modules
where those standards are enforced and consumed.

If you're consuming these standards from another DarojaAI repo, see
the top-level [README.md](../README.md) for the module catalog.
