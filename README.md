# Odyn Simulator Dashboard

Odyn Simulator Dashboard is a Streamlit interface for running PD-disaggregation simulations with `python -m vidur.main` and visualizing request-level latency metrics.

## What changed

- Rebranded dashboard title and defaults for Odyn.
- Fixed startup failure caused by incorrect imports of `Request` from `vidur.entities.batch`.
- Added full stderr and stdout rendering in the UI to keep complete tracebacks visible.
- Added optional GCS-backed output storage for Cloud Run using `gs://bucket/prefix` in **Output root**.
- Added Terraform infrastructure under `infra/` and automated deployment script `deploy.sh`.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_simulator_dashboard.py --server.port=8502
```

Open `http://localhost:8502`.

### CLI simulation with dashboard defaults

```bash
python -m vidur.main \
  --request_generator_config_type synthetic \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 100.0 \
  --synthetic_request_generator_config_num_requests 6 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 100 \
  --fixed_request_length_generator_config_decode_tokens 8 \
  --trace_request_length_generator_config_trace_file data/processed_traces/splitwise_conv.csv \
  --cluster_config_num_replicas 8 \
  --replica_config_pd_node_ratio 0.25 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name qwen3-next-80B \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype fp8 \
  --metrics_config_output_dir odyn_simulator_output \
  --replica_config_num_prefill_replicas 2 \
  --replica_config_prefill_tensor_parallel_size 1 \
  --replica_config_decode_tensor_parallel_size 1
```

## Docker run

```bash
docker build -t odyn-simulator:local .
docker run --rm -p 8080:8080 \
  -e ODYN_OUTPUT_ROOT=odyn_simulator_output \
  odyn-simulator:local
```

Open `http://localhost:8080`.

To persist in GCS, set `ODYN_OUTPUT_ROOT=gs://YOUR_BUCKET/simulations` and mount Google credentials locally.

## One-command GCP deploy (Cloud Run)

### Prerequisites

- `gcloud` authenticated to target project.
- `terraform` >= 1.5.
- Enabled APIs: Cloud Run, Cloud Build, Artifact Registry, IAM, Storage.

### Deploy

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="europe-west1"
export ODYN_RESULTS_BUCKET="odyn-simulator-results-your-project-id"
export ODYN_ALLOWED_INVOKER="group:engineering@odyn.ai"
./deploy.sh
```

`deploy.sh` performs:

1. Create/check Artifact Registry repository.
2. Build and push container with Cloud Build.
3. Apply Terraform in `infra/` to deploy Cloud Run and storage.

### Access control

The service is not configured for unauthenticated access. Invocation is restricted with Cloud Run IAM via `allowed_invoker` (`group:...`, `user:...`, or `serviceAccount:...`).

### Long simulations

The deployment uses Cloud Run service mode with:

- request timeout set to `3600s` (maximum for Cloud Run services), and
- `min_instance_count >= 1` to reduce cold starts.

This avoids request timeout failures for longer simulation runs while keeping the dashboard interactive.

## Infrastructure variables

See `infra/terraform.tfvars.example`:

- `project_id`
- `region`
- `service_name`
- `image`
- `results_bucket_name`
- `results_prefix`
- `allowed_invoker`
- `min_instances`
- `max_instances`

No credentials are stored in this repository.

## Test

```bash
python -m pytest -q
```

Includes `tests/test_smoke_simulation.py` for a short end-to-end simulation run.
