output "service_url" {
  value = google_cloud_run_v2_service.dashboard.uri
}

output "results_bucket" {
  value = google_storage_bucket.results.name
}
