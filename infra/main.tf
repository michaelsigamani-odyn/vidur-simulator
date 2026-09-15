provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "results" {
  name                        = var.results_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_service_account" "cloud_run" {
  account_id   = "${var.service_name}-sa"
  display_name = "${var.service_name} runtime"
}

resource "google_storage_bucket_iam_member" "runtime_writer" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_cloud_run_v2_service" "dashboard" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run.email
    timeout         = "3600s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      env {
        name  = "ODYN_OUTPUT_ROOT"
        value = "gs://${google_storage_bucket.results.name}/${var.results_prefix}"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }
    }
  }

  depends_on = [google_storage_bucket_iam_member.runtime_writer]
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  location = google_cloud_run_v2_service.dashboard.location
  service  = google_cloud_run_v2_service.dashboard.name
  role     = "roles/run.invoker"
  member   = var.allowed_invoker
}
