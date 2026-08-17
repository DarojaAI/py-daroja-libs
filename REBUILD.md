# VPC Runner Base — Operator Runbook (Q5 #1)

> **Audience:** the operator. This file is the artifact's single source of
> truth for "what _you_ have to do, after the staging PR lands, to actually
> publish `us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base`."
>
> It is not a `dev-nexus`-side runbook. It is `py-daroja-libs`-side runbook.
> The consumer-side `Dockerfile.vpc-runner` (dev-nexus + rag_research_tool
> wrapper) already points at the new AR path — see `OPEN_QUESTIONS.md` Q5 #1
> history.

---

## TL;DR

The Dockerfile, `.dockerignore`, README, publish workflow, and GCP setup
guide are **already committed** to `py-daroja-libs:5f832c0`. The publish
workflow has never fired since the rename. To close Q5 #1:

```bash
# ─── 1. Create the AR repo (one-time, idempotent) ───────────────────────────
gcloud artifacts repositories create py-daroja-libs \
  --repository-format=docker \
  --location=us-central1 \
  --project=globalbiting-dev \
  --description="Shared base images for DarojaAI projects (vpc-runner, etc.)"

# ─── 2. Grant writer to the existing org-deploy SA ──────────────────────────
gcloud artifacts repositories add-iam-policy-binding py-daroja-libs \
  --location=us-central1 \
  --project=globalbiting-dev \
  --member="serviceAccount:github-actions-deploy@globalbiting-dev.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
# If the repo already exists: this is a no-op idempotent update.

# ─── 3. Set the 4 workflow variables on this repo ───────────────────────────
gh variable set GCP_PROJECT_ID   --repo DarojaAI/py-daroja-libs --body "globalbiting-dev"
gh variable set GCP_REGION       --repo DarojaAI/py-daroja-libs --body "us-central1"
gh variable set GCP_WIF_PROVIDER --repo DarojaAI/py-daroja-libs \
  --body "projects/665374072631/locations/global/workloadIdentityPools/github-pool/providers/github-provider-daroja"
gh variable set GCP_PUBLISH_SA   --repo DarojaAI/py-daroja-libs \
  --body "github-actions-deploy@globalbiting-dev.iam.gserviceaccount.com"

# ─── 4. Run the workflow once on a throwaway tag (workflow_dispatch) ─────────
# Actions tab → "Publish VPC Runner Base Image" → Run workflow → tag=dev
# (the workflow reads inputs.tag when fired via dispatch; defaults to "dev")

# ─── 5. Smoke-test the published image is actually live at the path ─────────
gcloud artifacts docker tags list \
  us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base \
  --format="value(tag,update_time)"
# Expected output:
#   dev           <this run's timestamp>
#   latest        <this run's timestamp>

# ─── 6. Roll-forward to the first real semver tag ───────────────────────────
# Once the throwaway tag round-trips clean, re-run on the actual current
# py-daroja-libs release tag (e.g. `v1.18.0`). Trigger options:
#   a) Push the tag locally and let the workflow fire on tag-push: not
#      recommended — semantic-release cuts tags, you don't cut them in
#      git directly.
#   b) Fire workflow_dispatch with `tag=v1.18.0` (the workflow accepts any
#      matching tag regex from the manual input).
# The published image will live at:
#   us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base:v1.18.0
#   us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base:latest

# ─── 7. Bump dev-nexus + rag_research_tool wrapper builds ───────────────────
# Once the real tag exists, follow-up PRs in dev-nexus and rag_research_tool
# change `ARG VPC_RUNNER_BASE=…:v0.0.1-baseline` → `…:v1.18.0` (or whatever
# tag you published in step 6). Those follow-up PRs are NOT in this repo.
```

Steps 1–4 are required for first publish. Steps 5–6 close the Q5 #1
definition. Step 7 is a follow-up that lives outside this repo.

---

## What this repo already has (and didn't need re-staging)

The files behind this runbook already exist at `py-daroja-libs@5f832c0`:

| File | Why it exists |
|---|---|
| `docker/vpc-runner/Dockerfile` (6302 B) | The single-source-of-truth base image. Multi-stage, alpine runtime, pinned gcloud CLI (`572.0.0-alpine`), pinned Terraform `1.15.1`, pinned Atlas `v1.2.0`, three-branch ENTRYPOINT matching the contract locked by `rag_research_tool/tests/deploy/test_terraform_vpc_runner_entrypoint.py::TestEntrypointMirrorsDevnexusCommon::EXPECTED_SHAPE`. |
| `docker/vpc-runner/.dockerignore` | Canonical exclusions. The big win is `**/.terraform/` (provider cache ~170 MB; re-downloaded from lock file). |
| `docker/vpc-runner/README.md` | Three consumption patterns + "Pattern 1 is the right long-term answer." Pattern 1 is what this runbook enables. |
| `.github/workflows/publish-vpc-runner-base.yml` | WIF-authenticated build-and-push. Triggers on `v*` tag push; manual dispatch accepts custom tag (e.g. `dev`). Serialise concurrency to avoid tag-push races. |
| `.github/workflows/README-publish-vpc-runner-base.md` | The full one-time setup guide — GCP project, AR repo, WIF pool attribute condition, repo variables grid. Has the existing-`globalbiting-dev` short path (skip WIF + SA setup entirely). |
| `.github/workflows/devnexus-common-stress.yml` | **Renamed to `py-daroja-libs-stress.yml`** in the staging PR (Q5 #1 cleanup-only). Body already had `name: py-daroja-libs stress tests`; only the filename was stale post-rename. |

Q5 #1's "draft and stage Dockerfile" reading was inverted: the artifact
was drafted and staged before this session. The remaining work is the
five operator commands above plus the follow-up `v0.0.1-baseline` →
`vX.Y.Z` consumer rollout.

---

## Why this PR is a draft, not a real PR

Two reasons:

1. **The `v0.0.1-baseline` ARG defaults in `dev-nexus/Dockerfile.vpc-runner`
   and `rag_research_tool/config/Dockerfile.terraform-vpc` are fictional.**
   They were written under the assumption that semantic-release's
   pattern meant `vX.Y.Z-baseline` would be auto-cut (which it isn't —
   real semantic-release only cuts plain `vX.Y.Z` for Python packages).
   Bumping those defaults to a real, published tag happens in step 7 —
   a cross-repo follow-up, gated on a successful first publish.
2. **Consumer `TestEntrypointMirrorsDevnexusCommon::EXPECTED_SHAPE` is
   hardcoded against the upstream string shape**, but the test currently
   has no test for the cross-repo runtime invariant (does dev-nexus's
   `FROM` actually land on the published AR path?). After step 6, a
   follow-up issue should add that assertion.

Promote this PR out of draft once steps 1–4 complete and the first
`workflow_dispatch` lands a clean `:dev` tag.

---

## Failure-mode map

The staging PR **does not** fix the rank-order runtime failure. The
current runtime state is:

- **dev-nexus** `terraform-apply-v2.yml` builds `dev-nexus/vpc-runner:TAG`
  with `VPC_RUNNER_BASE=…/py-daroja-libs/vpc-runner-base:v0.0.1-baseline`.
  `docker pull` of that base **fails** because nothing has been pushed
  at that path under that tag. Builds that hit this branch surface in
  the `auth.docker.config` step with "unauthorized" or "manifest unknown".
- **rag_research_tool** `terraform-apply.yml` does the same with
  `…/devnexus-common/vpc-runner-base:v0.0.1-baseline` — the OLD repo,
  OLD path. Per PR #1309 the AR path was deliberately preserved
  ("AR paths are independent of GH renames"), so the historical
  `…/devnexus-common/…` image (if it ever existed) would still be
  load-bearing until rag_research_tool's follow-up PR flips it.

Until steps 6–7 complete, **avoid** running terraform-apply-v2 in
either repo. If a deploy is urgent, pin `VPC_RUNNER_BASE` to a known
existing image (`…/devnexus-common/vpc-runner-base:v0.0.1-baseline` may
or may not exist — verify with `gcloud artifacts docker images list` —
it predates the rename).

---

## What the Q5 #1 staging PR **does not** change

- The Dockerfile content (still needs the toolchain versions you bump
  intentionally; semantic-release does not auto-bump them).
- The publish workflow (still triggers on `v*` tag push OR
  `workflow_dispatch` with custom tag).
- The 4 repo variables (still need to be set; PR's REBUILD.md is the
  reminder). The `Validate required configuration` step in the workflow
  blocks first-run until they're set.
- dev-nexus / rag_research_tool wrappers (cross-repo roll-forward, step 7).
- The `README-publish-vpc-runner-base.md` 1-time setup guide (already
  covers the `globalbiting-dev` short path).

---

## Related

- `OPEN_QUESTIONS.md` Q5 #1 — the open question this closes.
- `OPEN_QUESTIONS.md` Q5 #2 — `intelligent-feed` PR #14 CHANGELOG rewrite.
- `OPEN_QUESTIONS.md` Q5 #3 — `rag_research_tool/docs/dependencies/devnexus-common.md`
  filename carve-out (independent of this file).
- `docker/vpc-runner/README.md` — the container-facing README.
- `.github/workflows/README-publish-vpc-runner-base.md` — the GCP-side
  1-time setup guide.
- `dev-nexus/Dockerfile.vpc-runner` — Pattern 1 consumer (the wrapper
  the publish workflow is meant to back).
- `rag_research_tool/config/Dockerfile.terraform-vpc` — second consumer
  (still pointing at the OLD `devnexus-common` AR path; needs a
  follow-up PR after first publish succeeds).
