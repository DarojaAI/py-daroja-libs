# Publish VPC Runner Base Image — setup guide

This document explains the one-time GCP + repo configuration needed before
`.github/workflows/publish-vpc-runner-base.yml` will run successfully.

The workflow is triggered on every `v*` tag push. After a `feat:` commit
merges to `main`, semantic-release cuts a tag (e.g. `v1.9.0`) and this
workflow publishes a corresponding base image to Artifact Registry:

```
${REGION}-docker.pkg.dev/${PROJECT_ID}/py-daroja-libs/vpc-runner-base:v1.9.0
${REGION}-docker.pkg.dev/${PROJECT_ID}/py-daroja-libs/vpc-runner-base:latest
```

Consumer repos (dev-nexus, rag-research-tool) then `FROM` the version-pinned
URL instead of rebuilding the toolchain on every CI run.

## TL;DR for the existing `globalbiting-dev` setup

This org already has a WIF provider that accepts any repo under
`DarojaAI/*`, and the `github-actions-deploy` SA has a WIF binding for
that pool. So you can skip WIF + SA setup entirely and only do:

```bash
# 1. Create the AR repo (one-time)
gcloud artifacts repositories create py-daroja-libs \
  --repository-format=docker \
  --location=us-central1 \
  --project=globalbiting-dev \
  --description="Shared base images for DarojaAI projects (vpc-runner, etc.)"

# 2. Grant writer to the existing github-actions-deploy SA (AR-repo-scoped, not project-wide)
gcloud artifacts repositories add-iam-policy-binding py-daroja-libs \
  --repository=py-daroja-libs \
  --location=us-central1 \
  --project=globalbiting-dev \
  --member="serviceAccount:github-actions-deploy@globalbiting-dev.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# 3. Set the 4 vars on this repo
gh variable set GCP_PROJECT_ID   --repo DarojaAI/py-daroja-libs --body "globalbiting-dev"
gh variable set GCP_REGION       --repo DarojaAI/py-daroja-libs --body "us-central1"
gh variable set GCP_WIF_PROVIDER --repo DarojaAI/py-daroja-libs --body "projects/665374072631/locations/global/workloadIdentityPools/github-pool/providers/github-provider-daroja"
gh variable set GCP_PUBLISH_SA   --repo DarojaAI/py-daroja-libs --body "github-actions-deploy@globalbiting-dev.iam.gserviceaccount.com"

# 4. Run the workflow once via the Actions tab (workflow_dispatch with tag=dev)
#    — this publishes the first image to py-daroja-libs/vpc-runner-base:dev.
```

If you're setting this up on a new GCP project or new GitHub org, follow
the longer path below.

## One-time setup (full path, for new projects)

### 1. Artifact Registry: create the repo

```bash
gcloud artifacts repositories create py-daroja-libs \
  --repository-format=docker \
  --location=us-central1 \
  --project="$PROJECT_ID" \
  --description="Shared base images for DarojaAI projects (vpc-runner, etc.)"
```

The `--location` value must match the `GCP_REGION` variable you set below.

### 2. Service account: a SA that can write to the AR repo

You have two options.

**(a) Reuse an existing SA** if it already has a WIF binding that
covers this repo. In `globalbiting-dev`, `github-actions-deploy@…` is
already bound to the org-scoped WIF pool (`attribute.repository_owner=DarojaAI`),
which accepts tokens from `DarojaAI/py-daroja-libs`. So we just grant it
`roles/artifactregistry.writer` on the new AR repo:

```bash
gcloud artifacts repositories add-iam-policy-binding py-daroja-libs \
  --location=us-central1 \
  --project="$PROJECT_ID" \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

**(b) Create a dedicated publish SA** if you want the tightest blast
radius. Use this when your WIF provider is repo-scoped (e.g.
`attribute.repository=DarojaAI/dev-nexus` only) and you don't want the
publish workflow inheriting the deploy SA's broader permissions:

```bash
SA_NAME="github-publish-py-daroja-libs"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --project="$PROJECT_ID" \
  --display-name="Publish py-daroja-libs base images from CI"

# Grant write access — AR-repo-scoped, NOT project-wide
gcloud artifacts repositories add-iam-policy-binding py-daroja-libs \
  --location=us-central1 \
  --project="$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"
```

Then bind the SA to the WIF pool (see step 3).

Either way, the SA has no project-level admin roles, no service account
key material, no compute/storage permissions. The WIF binding (or the
publish-SAO assumption) constrains *when* the SA can be used.

### 3. Workload Identity Federation: bind the SA so this repo can assume it

Use Terraform, `gcloud iam`, or the GitHub OIDC provider of your choice.
The pattern in this org is the `DarojaAI/infra-actions/gcp/auth` action,
which expects:

- A WIF provider resource name (`projects/.../locations/.../workloadIdentityPools/.../providers/...`)
- A pool that trusts `token.actions.githubusercontent.com`
- An attribute condition that limits which repo + branch/ref can mint tokens
- An IAM binding granting `roles/iam.workloadIdentityUser` on the SA to the
  pool's principalSet

The condition should look something like:

```
attribute.repository == "DarojaAI/py-daroja-libs"
```

If you also want `workflow_dispatch` runs to be able to publish (e.g. for a
manual `dev` tag), the condition can be relaxed to allow any ref; or you
can add explicit `main` and `v*` allowances. Keep it tight for prod.

**For `globalbiting-dev`, this step is already done** — the existing WIF
provider's attribute condition (`attribute.repository_owner=DarojaAI`)
covers this repo, and `github-actions-deploy` already has the WIF binding.
Skip unless you're setting up a new project.

### 4. Repo variables: set on DarojaAI/py-daroja-libs

**These are VARIABLES, not secrets.** Settings → Secrets and variables
→ Actions → **Variables** tab → New repository variable. Use `gh variable set`,
not `gh secret set`. Putting them under Secrets won't work — the workflow
reads them as `${{ vars.GCP_* }}` and ignores `${{ secrets.GCP_* }}`. WIF
handles auth, so there's no credential to protect here — these are
identifiers (project ID, region, WIF provider name, SA email).

| Name | Required | Example | Notes |
|---|---|---|---|
| `GCP_PROJECT_ID` | yes | `globalbiting-dev` | Project that hosts the AR repo. |
| `GCP_REGION` | no | `us-central1` | Defaults to `us-central1` if unset. |
| `GCP_WIF_PROVIDER` | yes | `projects/665374072631/locations/global/workloadIdentityPools/github-pool/providers/github-provider-daroja` | WIF provider resource name. |
| `GCP_PUBLISH_SA` | yes | `github-actions-deploy@globalbiting-dev.iam.gserviceaccount.com` | SA email that has writer on the AR repo. |

Verify with `gh variable list --repo DarojaAI/py-daroja-libs` — you should
see the 4 GCP_* entries. If `gh secret list --repo DarojaAI/py-daroja-libs`
shows them instead, they were set in the wrong tab (see troubleshooting).

### 5. Verify

The cleanest verification is to dispatch the workflow manually with a
throwaway tag, confirm the image lands, then clean up:

```bash
# On the Actions tab: workflow_dispatch → tag=dev (or any non-prod value)
# After it succeeds:
gcloud artifacts docker tags list \
  us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base \
  --format="value(tag,version)"
# You should see at least :dev and :latest.

# To clean up a throwaway image entirely (e.g. :dev-test-1):
gcloud artifacts docker tags delete \
  us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base:dev \
  --quiet
```

The actual `v1.9.0` publish will happen automatically when a release
PR merges and semantic-release cuts the tag.

## Consumer side (downstream repos)

Once the image is published, each consumer switches from vendoring the
Dockerfile to `FROM`-ing the published image. The Dockerfile becomes:

```dockerfile
# Set at build time. Pin to a specific tag, not :latest, so builds
# are reproducible.
ARG VPC_RUNNER_BASE=us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base:v1.9.0
FROM ${VPC_RUNNER_BASE}

WORKDIR /workspace

COPY atlas/      ./atlas/
COPY terraform/ ./terraform/

ENTRYPOINT ["/bin/sh", "-c", "if [ \"$#\" -eq 1 ]; then eval \"$1\"; else exec \"$@\"; fi", "--"]
```

The consumer's CI workflow must configure Docker to authenticate against
the AR region before pulling:

```yaml
- name: Auth to AR (for pulling the base image)
  run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

The consumer's WIF SA (typically the same `github-actions-deploy` as
above) needs `roles/artifactregistry.reader` on the `py-daroja-libs` AR
repo so it can pull the base. The `DarojaAI/infra-actions/docs/github-actions-wif-setup.md`
script grants this when run with `CONSUMER_AR_REPO_NAME=py-daroja-libs`.

## Why WIF, not a service account key

WIF (Workload Identity Federation) lets the workflow assume the SA via
short-lived OIDC tokens minted by GitHub Actions. There is no
long-lived service account key to leak. This is the modern pattern and
matches what the rest of the DarojaAI org already does (see
`DarojaAI/dev-nexus/.github/workflows/terraform-apply-v2.yml` for an example).

## Defending the published image

- **Consumers must pin to a specific semver tag**, not `:latest`. This
  makes the supply chain auditable — every CI run pulls a known image.
- **Branch protection on `main` + required reviews on `py-daroja-libs`**
  gates who can change the Dockerfile that backs the image.
- **Image signing (cosign)** is a future hardening — see roadmap.

## Troubleshooting

- **`Missing required repo variables`** — the `Validate required
  configuration` step prints the missing variable names. First check
  `gh variable list --repo DarojaAI/py-daroja-libs` — if the values
  aren't there, check `gh secret list` instead. Most common cause:
  setting them as secrets via the GitHub UI Secrets tab. Move them to
  the Variables tab. (Workflow reads `${{ vars.X }}`, never `${{ secrets.X }}`.)
- **`Permission denied` on `docker push`** — the SA doesn't have
  `roles/artifactregistry.writer` on the AR repo. Re-check step 2.
- **`Permission denied` on `docker pull` (consumer side)** — consumer's
  WIF SA lacks `roles/artifactregistry.reader` on `py-daroja-libs`. Run
  `infra-actions/docs/github-actions-wif-setup.md` again with the right
  `CONSUMER_AR_REPO_*` env vars.
- **`Invalid authentication credentials`** — the WIF pool doesn't trust
  this repo, or the attribute condition is too tight. Re-check step 3.
- **Tag exists in AR but the workflow says `image not found`** — old
  cached `gcloud` config. Add a `gcloud auth login --update-adc` step.

## Why we don't auto-rotate `latest`

The workflow tags `:latest` on every push as a convenience for local
exploration. **Production consumers should never pin `:latest`** — every
build would silently pick up the next published image, defeating
reproducibility. The pinning is by-tag at the Dockerfile level; the
`:latest` tag is for `docker run …-dev` style interactive use only.

## Related

- `docker/vpc-runner/README.md` — the three consumption patterns (1: prebuilt
  base, 2: copy at build time, 3: vendor). This workflow enables Pattern 1.
- `DarojaAI/infra-actions/docs/github-actions-wif-setup.md` — the WIF
  bootstrap script that grants the cross-AR consumer reader grant.
