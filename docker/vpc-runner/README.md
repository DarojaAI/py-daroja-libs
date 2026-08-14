# VPC Runner — canonical image

This is the canonical source of truth for the `vpc-runner` / `terraform-vpc`
Cloud Run Job image used across DarojaAI projects. It exists here so that:

- All consumers (dev-nexus, rag-research-tool, future projects) build the
  same toolchain versions.
- Terraform and Atlas version bumps happen in one place and propagate
  predictably.
- The image is small (~400 MB compressed vs the previous ~1.4 GB) because
  the base is a pinned slim variant and the build is multi-stage.

## What the image provides

- `gcloud` CLI (pinned to a specific version, not `:latest`)
- `terraform` (default `1.15.1`, override via `--build-arg`)
- `atlas` (default `v1.2.0`, override via `--build-arg`) with the
  community-edition driver set so `atlas migrate apply` works against
  the project DB
- `psql`, `jq`, `git`, `bash`, `ca-certificates`

It does **not** copy project files. Consumers must extend the image.

## Consumption patterns

Pick one based on your build-action's capabilities.

### Pattern 1 — `FROM` a prebuilt base image (best)

If your project publishes a prebuilt vpc-runner base image to Artifact
Registry (e.g. `${REGION}-docker.pkg.dev/${PROJECT}/py-daroja-libs/vpc-runner-base:v1.9.0`),
the consumer's `Dockerfile.vpc-runner` is just:

```dockerfile
ARG VPC_RUNNER_BASE=us-central1-docker.pkg.dev/globalbiting-dev/py-daroja-libs/vpc-runner-base:v1.9.0
FROM ${VPC_RUNNER_BASE}

WORKDIR /workspace

# Copy only the runtime-needed files. .terraform/ is excluded via the
# canonical .dockerignore shipped in this directory.
COPY atlas/    ./atlas/
COPY terraform/ ./terraform/

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["echo 'VPC Runner ready.'"]
```

Pros: fastest build, no per-build network calls, layer caching is
optimal. Cons: requires a separate pipeline to publish the base image on
release of `py-daroja-libs`.

### Pattern 2 — `COPY` the canonical Dockerfile into the build context (pragmatic)

In your build workflow, before `docker build`:

```bash
curl -fsSL \
  "https://raw.githubusercontent.com/DarojaAI/py-daroja-libs/v1.9.0/docker/vpc-runner/Dockerfile" \
  -o Dockerfile.vpc-runner.canonical
cp Dockerfile.vpc-runner.canonical Dockerfile.vpc-runner.build
# Append consumer-specific COPYs:
cat >> Dockerfile.vpc-runner.build <<'EOF'

WORKDIR /workspace
COPY atlas/      ./atlas/
COPY terraform/ ./terraform/
EOF
docker build -f Dockerfile.vpc-runner.build ...
```

Pros: no prebuilt base needed. Cons: pinning a tag URL is fragile
(raw.githubusercontent.com doesn't 404 cleanly on missing tags), and
the consumer ends up with a generated Dockerfile in their CI flow.

### Pattern 3 — Vendor the Dockerfile (simplest, lowest ceremony)

Copy this entire `docker/vpc-runner/` directory into your repo at a stable
path (e.g. `docker/vpc-runner/Dockerfile`). Pin the upstream copy via a
script in CI that opens a PR when the canonical version drifts.

Pros: zero network calls in CI, full control. Cons: drift. You have to
remember to re-vendor.

## Recommendation

Pattern 1 is the right long-term answer. To set it up:

1. Add a release workflow in `py-daroja-libs` that, on a tagged release,
   builds this Dockerfile and pushes it to
   `${REGION}-docker.pkg.dev/${PROJECT}/py-daroja-libs/vpc-runner-base:${TAG}`.
2. Update each consumer's build workflow to `FROM` the published image.
3. Update each consumer's `Dockerfile.vpc-runner` to be a thin wrapper
   that adds the project-specific COPYs.

Patterns 2 and 3 are interim solutions until the base-image pipeline
exists.

## Versioning policy

- **Base image** (`gcr.io/google.com/cloudsdktool/google-cloud-cli`): pinned
  to a specific version tag inside the Dockerfile. Bump in this file; the
  bump propagates on the next release of `py-daroja-libs`.
- **Terraform / Atlas versions**: ARG defaults in the Dockerfile. Override
  at build time if you need to. When you bump a major version, coordinate
  with other consumers that share Terraform state with you.

## File layout

```
docker/vpc-runner/
├── Dockerfile          # the canonical multi-stage build
├── .dockerignore       # canonical exclusions (especially **/.terraform/)
└── README.md           # this file
```

## Related

- `DarojaAI/dev-nexus#1015` — parent tracking issue in the dev-nexus repo
- `DarojaAI/rag_research_tool#733` — sibling tracking issue (was
  separate; now consolidated here)
- `docs/standards/VPC_RUNNER_PATTERN.md` in dev-nexus — the runtime
  pattern this image backs
