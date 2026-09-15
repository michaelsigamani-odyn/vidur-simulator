#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="${ODYN_SERVICE_NAME:-odyn-simulator}"
AR_REPO="${ODYN_ARTIFACT_REPO:-odyn-simulator}"
RESULTS_BUCKET="${ODYN_RESULTS_BUCKET:?Set ODYN_RESULTS_BUCKET}"
ALLOWED_INVOKER="${ODYN_ALLOWED_INVOKER:?Set ODYN_ALLOWED_INVOKER, example group:engineering@odyn.ai}"

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:${IMAGE_TAG}"

gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}" --project="${PROJECT_ID}"

gcloud builds submit --project="${PROJECT_ID}" --tag "${IMAGE_URI}" .

terraform -chdir=infra init
terraform -chdir=infra apply -auto-approve \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -var="service_name=${SERVICE_NAME}" \
  -var="image=${IMAGE_URI}" \
  -var="results_bucket_name=${RESULTS_BUCKET}" \
  -var="allowed_invoker=${ALLOWED_INVOKER}"

terraform -chdir=infra output service_url
