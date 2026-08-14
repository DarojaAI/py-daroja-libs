# ====================================
# GCP → Discord Notification Forwarder (Generic)
#
# Generic Terraform module for deploying a shared Cloud Run notifier
# that receives Pub/Sub events (Cloud Build, etc.) and forwards them
# to Discord. The same notifier service can be reused by multiple
# GCP projects — each project creates its own Pub/Sub topic and
# push-subscription pointing at the shared Cloud Run URL.
#
# Usage (consumer project — uses shared notifier):
#   module "notifier" {
#     source              = "github.com/DarojaAI/py-daroja-libs//common/notifications/terraform"
#     project_id          = "my-gcp-project"
#     app_name            = "my-app"  # becomes topic and subscription prefix
#     discord_webhook_url = var.discord_webhook_url
#     # existing_notifier_url = "https://gcp-discord-notifier-xxx.a.run.app"
#   }
#
# Usage (initial project — deploys the shared notifier):
#   module "notifier" {
#     source              = "github.com/DarojaAI/py-daroja-libs//common/notifications/terraform"
#     project_id          = "shared-notifier-project"
#     app_name            = "gcp-discord-notifier"
#     discord_webhook_url = var.discord_webhook_url
#     deploy_notifier     = true  # create the Cloud Run service
#   }
# ====================================

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ====================================
# Enable APIs
# ====================================

resource "google_project_service" "pubsub" {
  service            = "pubsub.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild_api" {
  count              = var.deploy_notifier ? 1 : 0
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry_api" {
  count              = var.deploy_notifier ? 1 : 0
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# ====================================
# Pub/Sub Topic
# Cloud Build publishes build events here (or another source publishes)
# ====================================

resource "google_pubsub_topic" "build_events" {
  name = "${var.app_name}-build-events"
  labels = {
    app  = var.app_name
    role = "notification"
  }

  depends_on = [google_project_service.pubsub]
}

# ====================================
# Cloud Run Notifier (only when deploy_notifier = true)
# Receives Pub/Sub push messages and forwards to Discord
# ====================================

resource "google_artifact_registry_repository" "notifier" {
  count         = var.deploy_notifier ? 1 : 0
  location      = var.region
  repository_id = "${var.app_name}-notifier"
  description   = "Container registry for ${var.app_name} notifier"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry_api]
}

resource "google_service_account" "notifier" {
  count        = var.deploy_notifier ? 1 : 0
  account_id   = "${var.app_name}-notifier-sa"
  display_name = "${var.app_name} Notification Relay"
  description  = "Cloud Run service account for ${var.app_name} Discord notification relay"
}

resource "google_project_iam_member" "notifier_sa_pubsub" {
  count   = var.deploy_notifier ? 1 : 0
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.notifier[0].email}"
}

resource "google_cloud_run_service" "notifier" {
  count    = var.deploy_notifier ? 1 : 0
  name     = var.app_name
  location = var.region
  project  = var.project_id

  template {
    spec {
      service_account_name = google_service_account.notifier[0].email
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}-notifier/${var.app_name}:latest"
        ports {
          container_port = 8080
        }
        env {
          name  = "DISCORD_WEBHOOK_URL"
          value = var.discord_webhook_url
        }
        env {
          name  = "SERVICE_NAME"
          value = var.app_name
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "256Mi"
          }
        }
      }
    }

    metadata {
      labels = {
        app = var.app_name
      }
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.pubsub,
    google_project_service.cloudbuild_api,
  ]
}

resource "google_cloud_run_service_iam_member" "notifier_public" {
  count    = var.deploy_notifier && var.allow_unauthenticated ? 1 : 0
  service  = google_cloud_run_service.notifier[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ====================================
# Pub/Sub Push Subscription
# Delivers messages from build-events topic → Cloud Run /push endpoint
# Uses existing_notifier_url when provided, otherwise the notifier we just deployed
# ====================================

locals {
  notifier_url = var.existing_notifier_url != "" ? var.existing_notifier_url : (
    var.deploy_notifier ? google_cloud_run_service.notifier[0].status[0].url : ""
  )
}

resource "google_pubsub_subscription" "notifier_push" {
  name  = "${var.app_name}-notifier-push"
  topic = google_pubsub_topic.build_events.name

  ack_deadline_seconds = 600

  push_config {
    push_endpoint = "${local.notifier_url}/push"
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  retain_acked_messages = false

  labels = {
    app  = var.app_name
    role = "notification"
  }

  depends_on = [google_cloud_run_service.notifier]
}

# Grant Pub/Sub subscriber SA permission to invoke Cloud Run (only when we deploy the notifier)
data "google_project" "current" {}

resource "google_cloud_run_service_iam_member" "notifier_pubsub_invoker" {
  count    = var.deploy_notifier ? 1 : 0
  service  = google_cloud_run_service.notifier[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# ====================================
# Cloud Build Trigger (optional)
# When enabled, creates a GitHub push trigger that builds the consumer app
# and publishes build events to the topic.
# ====================================

resource "google_cloudbuild_trigger" "on_push_to_main" {
  count       = var.notification_trigger_enabled ? 1 : 0
  name        = "${var.app_name}-notify-on-main"
  description = "Build ${var.app_name} on push to main + publish events to Discord"
  disabled    = false

  github {
    owner = var.github_owner

    push {
      branch = "^${var.github_branch}$"
    }
  }

  pubsub_config {
    topic = google_pubsub_topic.build_events.id
  }

  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-t", "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}/${var.app_name}:latest",
        ".",
      ]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "push",
        "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}/${var.app_name}:latest",
      ]
    }
    step {
      name       = "gcr.io/google.com/cloudsdktool/cloud-sdk"
      entrypoint = "gcloud"
      args = [
        "run", "deploy", var.app_name,
        "--image", "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}/${var.app_name}:latest",
        "--region", var.region,
        "--port", "8080",
      ]
    }

    images = ["${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}/${var.app_name}:latest"]
  }

  depends_on = [
    google_pubsub_topic.build_events,
    google_pubsub_subscription.notifier_push,
    google_project_service.cloudbuild_api,
  ]
}

# ====================================
# Outputs
# ====================================

output "notifier_service_url" {
  description = "Cloud Run notifier service URL (empty if using existing)"
  value       = var.deploy_notifier ? google_cloud_run_service.notifier[0].status[0].url : local.notifier_url
}

output "pubsub_topic" {
  description = "Pub/Sub topic for build events"
  value       = google_pubsub_topic.build_events.name
}

output "pubsub_subscription" {
  description = "Pub/Sub push subscription"
  value       = google_pubsub_subscription.notifier_push.name
}

output "notifier_sa_email" {
  description = "Notifier service account email (only when deploy_notifier = true)"
  value       = var.deploy_notifier ? google_service_account.notifier[0].email : ""
}
