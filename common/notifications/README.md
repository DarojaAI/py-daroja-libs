# GCP → Discord Notification Forwarder

Generic Cloud Run service that forwards GCP Pub/Sub events
(Cloud Build, etc.) to Discord via rich embeds.

Replaces the project-specific `elastica-notifier` with a shared,
reusable module that any GCP project can use. One project deploys
the notifier service; all others create their own Pub/Sub topic
and push-subscribe to the shared Cloud Run URL.

## Install

```bash
pip install "py-daroja-libs[notifications]"
```

Or, to ship this as a container:

```bash
FROM python:3.11-slim
RUN pip install "py-daroja-libs[notifications]"
CMD ["uvicorn", "common.notifications:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Local Development

```bash
pip install -e ".[dev]"
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
uvicorn common.notifications:app --host 0.0.0.0 --port 8080
```

Test the endpoints:

```bash
# Health
curl http://localhost:8080/health

# Push (simulating Pub/Sub)
curl -X POST http://localhost:8080/push \
  -H 'Content-Type: application/json' \
  -d '{"message": {"data": "'$(echo -n '{"build":{"status":"SUCCESS"}}' | base64)'"}}'
```

## Deployment

### Step 1 — Deploy the shared notifier (one project only)

```hcl
# shared-notifier-project/main.tf
module "notifier" {
  source              = "github.com/DarojaAI/py-daroja-libs//common/notifications/terraform"
  project_id          = "shared-notifier-project"
  app_name            = "gcp-discord-notifier"
  discord_webhook_url = var.discord_webhook_url
  deploy_notifier     = true
}

output "notifier_url" {
  value = module.notifier.notifier_service_url
}
```

### Step 2 — Consumer projects subscribe to the shared notifier

```hcl
# my-app-project/main.tf
module "build_notifications" {
  source                = "github.com/DarojaAI/py-daroja-libs//common/notifications/terraform"
  project_id            = "my-app-project"
  app_name              = "my-app"
  discord_webhook_url   = var.discord_webhook_url
  existing_notifier_url = "https://gcp-discord-notifier-xxx.a.run.app"
  notification_trigger_enabled = true
  github_owner          = "DarojaAI"
  github_repo           = "my-app"
}
```

Each consumer project gets:

- Its own Pub/Sub topic (`my-app-build-events`)
- Its own push-subscription pointing at the shared notifier's `/push` endpoint
- Optionally, a Cloud Build GitHub trigger that publishes build events to its topic

No per-project Cloud Run deployment. Shared notifier scales to zero when idle.

## Supported Event Types

- **Cloud Build** — SUCCESS, FAILURE, TIMEOUT, CANCELLED, INTERNAL_ERROR, QUEUED, WORKING

To add a new event type, edit `common/notifications/notifier.py` and add a
handler function. The routing is in `handle_pubsub_push`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_WEBHOOK_URL` | Yes | Target Discord webhook URL |
| `SERVICE_NAME` | No | Service name shown in `/health` response (default: `gcp-discord-notifier`) |

## Architecture

```
GitHub push to main
    │
    ▼
Cloud Build Trigger (per-project)
    │
    ├─► Build & Deploy app
    │
    └─► Publishes build event → Pub/Sub topic (per-project)
              │
              ▼
        Push Subscription (per-project)
              │
              ▼
        Shared Cloud Run: gcp-discord-notifier (/push)
              │
              ▼
        Discord Webhook → target channel
```

## Migration from `dev-nexus/notifications/`

If you were using the old `elastica-notifier`:

1. Deploy the new shared notifier in your central project.
2. Update consumer terraform to use `existing_notifier_url`.
3. Remove old `elastica-notifier-*` resources: `terraform destroy` in the
   project where they were originally deployed.
4. Remove `elastica-notifier` from `requirements.txt` / imports.

## Refs

- Issue: dev-nexus #958
- Previous: `dev-nexus/notifications/` (project-specific, will be removed in dev-nexus #951)
