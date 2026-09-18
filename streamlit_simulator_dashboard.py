from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from vidur.odyn_registry import (
    ModelEntry,
    PresetEntry,
    SKUEntry,
    load_model_registry,
    load_presets,
    load_sku_registry,
    validate_presets,
    valid_pp_values,
    valid_tp_values,
)

try:
    import boto3
except ImportError:
    boto3 = None

try:
    from google.cloud import storage
except ImportError:
    storage = None


@dataclass
class RunForm:
    model_id: str
    sku_id: str
    model_name: str
    profile_name: str
    backend: str
    scheduler: str
    replica_scheduler: str
    replicas: int
    pd_ratio: float
    tp: int
    pp: int
    prefill_replicas: int
    prefill_tp: int
    decode_tp: int
    qps: float
    requests: int
    prefill_tokens: int
    decode_tokens: int
    trace_file: str
    length_type: str
    interval_type: str
    pd_bandwidth: int
    nvlink_bandwidth: int
    rdma_bandwidth: int
    pd_dtype: str
    output_root: str
    extra_args: str
    device: str
    network_device: str
    hourly_cost_usd: float
    node_size: int


@dataclass
class RunResult:
    success: bool
    return_code: int
    output_text: str
    stdout_text: str
    stderr_text: str
    run_dir: Optional[Path]
    command: List[str]


@dataclass
class OutputTarget:
    is_gcs: bool
    local_root: Path
    bucket: Optional[str]
    prefix: str


@dataclass
class RunSummary:
    run_name: str
    run_dir: Path
    model: str
    trace: str
    sku: str
    scheduler: str
    tp: int
    pp: int
    batch_size: int
    qps: float
    observed_qps: float
    ttft_p90: float
    tbt_p99_ms: float
    qps_per_dollar: float
    backend: str


SKU_REGISTRY: Dict[str, SKUEntry] = load_sku_registry(Path(__file__).resolve().parent)
MODEL_REGISTRY: Dict[str, ModelEntry] = load_model_registry(Path(__file__).resolve().parent)
PRESET_REGISTRY: Dict[str, PresetEntry] = load_presets(Path(__file__).resolve().parent)


def validate_registry() -> None:
    errors = validate_presets(PRESET_REGISTRY, MODEL_REGISTRY, SKU_REGISTRY)
    if errors:
        raise RuntimeError("\n".join(errors))


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_output_root() -> str:
    return os.environ.get("ODYN_OUTPUT_ROOT", "/tmp/odyn-dashboard/odyn-beta-v1/simulations")


def default_trace_path() -> str:
    return "data/processed_traces/splitwise_conv.csv"


def to_run_form(preset_id: str, preset: PresetEntry) -> RunForm:
    model = MODEL_REGISTRY[preset.model_id]
    sku = SKU_REGISTRY[preset.sku_id]
    return RunForm(
        preset.model_id,
        preset.sku_id,
        model.model_name,
        model.profile_name,
        preset.backend,
        preset.scheduler,
        preset.replica_scheduler,
        preset.replicas,
        preset.pd_ratio,
        preset.tp,
        preset.pp,
        preset.prefill_replicas,
        preset.prefill_tp,
        preset.decode_tp,
        preset.qps,
        preset.requests,
        preset.prefill_tokens,
        preset.decode_tokens,
        default_trace_path(),
        preset.length_type,
        preset.interval_type,
        800,
        1600,
        800,
        "fp8",
        default_output_root(),
        "",
        sku.device,
        sku.network_device,
        sku.hourly_cost_usd,
        sku.node_sizes[0],
    )


def scenario_defaults() -> Dict[str, RunForm]:
    return {preset.label: to_run_form(preset_id, preset) for preset_id, preset in PRESET_REGISTRY.items()}


def split_args(arg_string: str) -> List[str]:
    return [part for part in arg_string.split() if part.strip()]


def add_pd_args(args: List[str], form: RunForm) -> None:
    if form.prefill_replicas > 0:
        args.extend(["--replica_config_num_prefill_replicas", str(form.prefill_replicas)])
    if form.pd_ratio < 1.0:
        args.extend(["--replica_config_prefill_tensor_parallel_size", str(form.prefill_tp)])
        args.extend(["--replica_config_decode_tensor_parallel_size", str(form.decode_tp)])


def base_cli_args(form: RunForm) -> List[str]:
    args = [
        "python",
        "-m",
        "vidur.main",
        "--request_generator_config_type",
        "synthetic" if form.length_type == "fixed" else "trace_replay",
        "--interval_generator_config_type",
        form.interval_type,
        "--poisson_request_interval_generator_config_qps",
        str(form.qps),
        "--synthetic_request_generator_config_num_requests",
        str(form.requests),
        "--trace_request_generator_config_trace_file",
        form.trace_file,
        "--length_generator_config_type",
        form.length_type,
        "--fixed_request_length_generator_config_prefill_tokens",
        str(form.prefill_tokens),
        "--fixed_request_length_generator_config_decode_tokens",
        str(form.decode_tokens),
        "--trace_request_length_generator_config_trace_file",
        form.trace_file,
        "--cluster_config_num_replicas",
        str(form.replicas),
        "--replica_config_pd_node_ratio",
        str(form.pd_ratio),
        "--global_scheduler_config_type",
        form.scheduler,
        "--replica_scheduler_config_type",
        form.replica_scheduler,
        "--replica_config_model_name",
        form.model_name,
        "--replica_config_device",
        form.device,
        "--replica_config_network_device",
        form.network_device,
        "--replica_config_tensor_parallel_size",
        str(form.tp),
        "--replica_config_num_pipeline_stages",
        str(form.pp),
        "--random_forrest_execution_time_predictor_config_backend",
        form.backend,
        "--replica_config_pd_p2p_comm_bandwidth",
        str(form.pd_bandwidth),
        "--replica_config_nvlink_bandwidth",
        str(form.nvlink_bandwidth),
        "--replica_config_rdma_bandwidth",
        str(form.rdma_bandwidth),
        "--replica_config_pd_p2p_comm_dtype",
        form.pd_dtype,
        "--metrics_config_output_dir",
        form.output_root,
    ]
    add_pd_args(args, form)
    args.extend(split_args(form.extra_args))
    return args


def parse_output_target(raw_output_root: str) -> OutputTarget:
    normalized = raw_output_root.strip()
    if normalized.startswith("gs://"):
        _, remainder = normalized.split("gs://", 1)
        bucket, _, prefix = remainder.partition("/")
        cache_root = Path("/tmp") / "odyn-dashboard" / bucket / (prefix or "results")
        cache_root.mkdir(parents=True, exist_ok=True)
        return OutputTarget(True, cache_root, bucket, prefix.strip("/"))
    local_path = Path(normalized)
    local_root = local_path if local_path.is_absolute() else repo_root() / normalized
    local_root.mkdir(parents=True, exist_ok=True)
    return OutputTarget(False, local_root, None, "")


def expected_profile_paths(form: RunForm) -> List[Path]:
    root = repo_root() / "data" / "profiling"
    return [
        root / "compute" / form.device / form.profile_name / "mlp.csv",
        root / "compute" / form.device / form.profile_name / "attention.csv",
        root / "network" / form.network_device / "all_reduce.csv",
        root / "network" / form.network_device / "send_recv.csv",
    ]


def missing_profile_paths(form: RunForm) -> List[Path]:
    return [path for path in expected_profile_paths(form) if not path.exists()]


def get_storage_client() -> Optional["storage.Client"]:
    if storage is None:
        return None
    return storage.Client()


def list_remote_runs(target: OutputTarget) -> List[str]:
    client = get_storage_client()
    if client is None or not target.bucket:
        return []
    bucket = client.bucket(target.bucket)
    prefix = f"{target.prefix}/" if target.prefix else ""
    names = set()
    for blob in client.list_blobs(bucket, prefix=prefix):
        suffix = blob.name[len(prefix):]
        if not suffix.endswith("request_metrics.csv"):
            continue
        run_name = Path(suffix).parent.as_posix()
        if run_name and run_name != ".":
            names.add(run_name)
    return sorted(names, reverse=True)


def upload_run_dir_to_gcs(run_dir: Path, target: OutputTarget) -> None:
    client = get_storage_client()
    if client is None or not target.bucket:
        return
    bucket = client.bucket(target.bucket)
    for file_path in run_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(run_dir).as_posix()
        blob_name = "/".join(part for part in [target.prefix, run_dir.name, rel_path] if part)
        bucket.blob(blob_name).upload_from_filename(str(file_path))


def download_run_dir_from_gcs(run_name: str, target: OutputTarget) -> Optional[Path]:
    client = get_storage_client()
    if client is None or not target.bucket:
        return None
    local_dir = target.local_root / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(target.bucket)
    prefix = "/".join(part for part in [target.prefix, run_name] if part) + "/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        return None
    for blob in blobs:
        rel_path = blob.name[len(prefix):]
        if not rel_path:
            continue
        destination = local_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
    return local_dir


def ensure_remote_runs_cached(target: OutputTarget) -> None:
    for run_name in list_remote_runs(target):
        local_run_dir = target.local_root / run_name
        if (local_run_dir / "request_metrics.csv").exists():
            continue
        download_run_dir_from_gcs(run_name, target)


def resolve_s3_trace(trace_uri: str) -> str:
    if not trace_uri.startswith("s3://"):
        return trace_uri
    if boto3 is None:
        raise RuntimeError("boto3 is required for s3:// traces")
    _, remainder = trace_uri.split("s3://", 1)
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise RuntimeError(f"Invalid S3 URI: {trace_uri}")
    trace_dir = Path(tempfile.gettempdir()) / "odyn-dashboard" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    target = trace_dir / f"{bucket}__{key.replace('/', '__')}"
    boto3.client("s3").download_file(bucket, key, str(target))
    return str(target)


def is_valid_run_dir(run_dir: Path) -> bool:
    return (run_dir / "request_metrics.csv").exists()


def list_valid_run_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    dirs = {metrics_path.parent for metrics_path in output_root.rglob("request_metrics.csv")}
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def get_new_run_dir(before: Sequence[Path], output_root: Path) -> Optional[Path]:
    old_set = {path.resolve() for path in before}
    after = list_valid_run_dirs(output_root)
    new_runs = [path for path in after if path.resolve() not in old_set]
    return new_runs[0] if new_runs else (after[0] if after else None)


def make_failed_run_dir(output_root: Path, command: List[str], process: subprocess.CompletedProcess[str]) -> Path:
    run_dir = output_root / f"failed-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")
    (run_dir / "stderr.log").write_text(process.stderr or "", encoding="utf-8")
    (run_dir / "stdout.log").write_text(process.stdout or "", encoding="utf-8")
    return run_dir


def run_simulation(form: RunForm) -> RunResult:
    cwd = repo_root()
    target = parse_output_target(form.output_root)
    local_trace = resolve_s3_trace(form.trace_file)
    adapted = RunForm(**{**form.__dict__, "output_root": str(target.local_root), "trace_file": local_trace})
    command = base_cli_args(adapted)
    known_runs = list_valid_run_dirs(target.local_root)
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    run_dir = get_new_run_dir(known_runs, target.local_root)
    if process.returncode != 0 and run_dir is None:
        run_dir = make_failed_run_dir(target.local_root, command, process)
    if run_dir is not None:
        (run_dir / "stderr.log").write_text(process.stderr or "", encoding="utf-8")
        (run_dir / "stdout.log").write_text(process.stdout or "", encoding="utf-8")
        (run_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")
    if run_dir and target.is_gcs and is_valid_run_dir(run_dir):
        upload_run_dir_to_gcs(run_dir, target)
    combined = f"{process.stdout}\n{process.stderr}".strip()
    return RunResult(process.returncode == 0, process.returncode, combined, process.stdout, process.stderr, run_dir, command)


def read_request_metrics(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "request_metrics.csv")
    if "request_num_decode_tokens" in frame and "decode_time" in frame:
        den = frame["request_num_decode_tokens"].replace(0, pd.NA)
        frame["tbt"] = frame["decode_time"] / den
    return frame


def percentile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    return float(clean.quantile(q)) if not clean.empty else float("nan")


def safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def read_run_config(run_dir: Path) -> Dict:
    config_json = run_dir / "config.json"
    config_yml = run_dir / "config.yml"
    if config_json.exists():
        return json.loads(config_json.read_text(encoding="utf-8"))
    if config_yml.exists():
        return yaml.safe_load(config_yml.read_text(encoding="utf-8"))
    return {}


def extract_trace_name(config: Dict) -> str:
    trace = config.get("request_generator_config", {}).get("trace_file")
    if trace:
        return str(trace)
    trace = config.get("trace_request_generator_config_trace_file")
    if trace:
        return str(trace)
    return "unknown-trace"


def hourly_cost_for_device(device: str) -> float:
    for sku in SKU_REGISTRY.values():
        if sku.device == device:
            return sku.hourly_cost_usd
    return 1.0


def summarize_run(run_dir: Path) -> Optional[RunSummary]:
    metrics_path = run_dir / "request_metrics.csv"
    if not metrics_path.exists():
        return None
    frame = read_request_metrics(run_dir)
    config = read_run_config(run_dir)
    if frame.empty:
        return None
    replica_cfg = config.get("cluster_config", {}).get("replica_config", {})
    req_cfg = config.get("request_generator_config", {})
    scheduler_cfg = config.get("cluster_config", {}).get("replica_scheduler_config", {})
    backend = str(config.get("execution_time_predictor_config", {}).get("backend", "unknown"))
    device = str(replica_cfg.get("device", "a100"))
    network_device = str(replica_cfg.get("network_device", ""))
    tp = int(replica_cfg.get("tensor_parallel_size", 1))
    pp = int(replica_cfg.get("num_pipeline_stages", 1))
    replicas = int(config.get("cluster_config", {}).get("num_replicas", 1))
    model = str(replica_cfg.get("model_name", "unknown-model"))
    scheduler = str(scheduler_cfg.get("name", replica_cfg.get("scheduler", "unknown")))
    batch_size = int(scheduler_cfg.get("batch_size_cap", 0))
    qps = float(req_cfg.get("interval_generator_config", {}).get("qps", 0.0))
    runtime = float(frame["completed_at"].max() - frame["arrived_at"].min()) if {"completed_at", "arrived_at"}.issubset(frame.columns) else 0.0
    observed_qps = float(len(frame) / runtime) if runtime > 0 else qps
    ttft_p90 = percentile(frame["prefill_e2e_time"], 0.9)
    tbt_col = frame["tbt"] if "tbt" in frame else pd.Series(dtype=float)
    tbt_p99_ms = percentile(tbt_col, 0.99) * 1000.0 if not tbt_col.empty else float("nan")
    gpu_count = max(tp * pp * replicas, 1)
    qps_per_dollar = observed_qps / (hourly_cost_for_device(device) * gpu_count)
    sku = next((s.label for s in SKU_REGISTRY.values() if s.device == device and s.network_device == network_device), device)
    return RunSummary(run_dir.name, run_dir, model, extract_trace_name(config), sku, scheduler, tp, pp, batch_size, qps, observed_qps, ttft_p90, tbt_p99_ms, qps_per_dollar, backend)


def render_kpis(frame: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", f"{len(frame)}")
    c2.metric("P50 E2E (s)", f"{percentile(frame['request_e2e_time'], 0.5):.4f}")
    c3.metric("P95 E2E (s)", f"{percentile(frame['request_e2e_time'], 0.95):.4f}")
    c4.metric("Mean TTFT (s)", f"{safe_mean(frame['prefill_e2e_time']):.4f}")


def render_latency_plots(frame: pd.DataFrame) -> None:
    st.plotly_chart(px.histogram(frame, x="request_e2e_time", nbins=30, title="E2E Latency"), use_container_width=True)
    st.plotly_chart(px.scatter(frame, x="request_num_prefill_tokens", y="request_e2e_time", title="Prefill Tokens vs E2E"), use_container_width=True)
    st.plotly_chart(px.scatter(frame, x="request_num_decode_tokens", y="decode_time", title="Decode Tokens vs Decode Time"), use_container_width=True)


def render_timeline(frame: pd.DataFrame) -> None:
    columns = ["arrived_at", "scheduled_at", "prefill_completed_at", "decode_arrived_at", "completed_at"]
    present = [name for name in columns if name in frame.columns]
    melted = frame[["Request Id", *present]].melt(id_vars=["Request Id"], var_name="phase", value_name="time")
    chart = px.line(melted, x="phase", y="time", color="Request Id", title="Per-request Timing Phases", markers=True)
    st.plotly_chart(chart, use_container_width=True)


def render_pd_chart(frame: pd.DataFrame) -> None:
    if "pd_p2p_comm_time" not in frame.columns:
        return
    chart = px.histogram(frame, x="pd_p2p_comm_time", nbins=30, title="PD P2P Communication Time")
    st.plotly_chart(chart, use_container_width=True)


def render_run_metadata(run_dir: Path) -> None:
    st.json(read_run_config(run_dir), expanded=False)


def run_controls(defaults: Dict[str, RunForm]) -> RunForm:
    preset_labels = list(defaults)
    default_index = next((idx for idx, label in enumerate(preset_labels) if not missing_profile_paths(defaults[label]) and defaults[label].backend != "aicb"), 0)
    selected = st.sidebar.selectbox("Preset", preset_labels, index=default_index)
    if default_index > 0:
        st.sidebar.caption("Default preset moved to first runnable option for this deployment.")
    seed = defaults[selected]
    model_items = list(MODEL_REGISTRY.items())
    model_labels = [entry.label for _, entry in model_items]
    model_to_id = {entry.label: model_id for model_id, entry in model_items}
    seed_model_label = MODEL_REGISTRY[seed.model_id].label if seed.model_id in MODEL_REGISTRY else model_labels[0]
    selected_model_label = st.sidebar.selectbox("Model", model_labels, index=model_labels.index(seed_model_label))
    selected_model_id = model_to_id[selected_model_label]
    selected_model = MODEL_REGISTRY[selected_model_id]
    sku_items = list(SKU_REGISTRY.items())
    sku_labels = [entry.label for _, entry in sku_items]
    sku_to_id = {entry.label: sku_id for sku_id, entry in sku_items}
    seed_sku_label = SKU_REGISTRY[seed.sku_id].label if seed.sku_id in SKU_REGISTRY else sku_labels[0]
    selected_sku_label = st.sidebar.selectbox("SKU / Node", sku_labels, index=sku_labels.index(seed_sku_label))
    selected_sku_id = sku_to_id[selected_sku_label]
    selected_sku = SKU_REGISTRY[selected_sku_id]
    node_size = selected_sku.node_sizes[0]
    tp_choices = valid_tp_values(node_size)
    tp_default = seed.tp if seed.tp in tp_choices else tp_choices[0]
    tp = st.sidebar.selectbox("Tensor parallel", tp_choices, index=tp_choices.index(tp_default))
    pp_choices = valid_pp_values(node_size, tp)
    pp_default = seed.pp if seed.pp in pp_choices else pp_choices[0]
    pp = st.sidebar.selectbox("Pipeline stages", pp_choices, index=pp_choices.index(pp_default))
    st.sidebar.caption(f"Node size {node_size} GPUs constrains TP x PP <= {node_size}.")
    backend_choices = ["vidur", "simai_simulation", "simai_analytical", "aicb"]
    backend = st.sidebar.selectbox("Backend", backend_choices, index=backend_choices.index(seed.backend if seed.backend in backend_choices else "vidur"))
    scheduler = st.sidebar.selectbox("Global scheduler", ["split_wise", "lor", "round_robin"], index=["split_wise", "lor", "round_robin"].index(seed.scheduler))
    replica_scheduler = st.sidebar.selectbox("Replica scheduler", ["split_wise", "sarathi", "vllm"], index=["split_wise", "sarathi", "vllm"].index(seed.replica_scheduler))
    replicas = st.sidebar.number_input("Replicas", 1, 512, seed.replicas)
    pd_ratio = st.sidebar.slider("PD ratio", 0.01, 1.0, float(seed.pd_ratio), 0.01)
    prefill_replicas = st.sidebar.number_input("Prefill replicas (0=auto)", 0, 512, seed.prefill_replicas)
    prefill_tp = st.sidebar.number_input("Prefill TP", 1, 512, seed.prefill_tp)
    decode_tp = st.sidebar.number_input("Decode TP", 1, 512, seed.decode_tp)
    qps = st.sidebar.number_input("QPS", 0.01, 100000.0, float(seed.qps), 0.1)
    requests = st.sidebar.number_input("Requests", 1, 500000, seed.requests)
    prefill_tokens = st.sidebar.number_input("Prefill tokens", 1, 32768, seed.prefill_tokens)
    decode_tokens = st.sidebar.number_input("Decode tokens", 0, 32768, seed.decode_tokens)
    trace_file = st.sidebar.text_input("Trace file (local path or s3://bucket/key.csv)", seed.trace_file)
    length_type = st.sidebar.selectbox("Length type", ["fixed", "trace"], index=["fixed", "trace"].index(seed.length_type))
    interval_type = st.sidebar.selectbox("Interval type", ["poisson", "gamma", "trace"], index=["poisson", "gamma", "trace"].index(seed.interval_type))
    pd_bandwidth = st.sidebar.number_input("PD bandwidth (Gbps)", 1, 100000, seed.pd_bandwidth)
    nvlink_bandwidth = st.sidebar.number_input("NVLink bandwidth (Gbps)", 1, 100000, seed.nvlink_bandwidth)
    rdma_bandwidth = st.sidebar.number_input("RDMA bandwidth (Gbps)", 1, 100000, seed.rdma_bandwidth)
    pd_dtype = st.sidebar.selectbox("PD dtype", ["fp8", "float16", "float32"], index=["fp8", "float16", "float32"].index(seed.pd_dtype))
    output_root = st.sidebar.text_input("Output root", seed.output_root)
    extra_args = st.sidebar.text_input("Extra CLI args", seed.extra_args)
    return RunForm(selected_model_id, selected_sku_id, selected_model.model_name, selected_model.profile_name, backend, scheduler, replica_scheduler, int(replicas), float(pd_ratio), int(tp), int(pp), int(prefill_replicas), int(prefill_tp), int(decode_tp), float(qps), int(requests), int(prefill_tokens), int(decode_tokens), trace_file, length_type, interval_type, int(pd_bandwidth), int(nvlink_bandwidth), int(rdma_bandwidth), pd_dtype, output_root, extra_args, selected_sku.device, selected_sku.network_device, float(selected_sku.hourly_cost_usd), int(node_size))


def run_and_store(form: RunForm) -> None:
    missing_paths = missing_profile_paths(form)
    if missing_paths:
        details = "\n".join(str(path.relative_to(repo_root())) for path in missing_paths)
        message = "Missing profiling data; run blocked.\n" + details
        st.session_state["last_result"] = RunResult(False, 2, message, "", message, None, [])
        st.session_state["selected_run"] = ""
        return
    if form.backend == "aicb":
        message = "Backend aicb is blocked for capacity claims because this fork uses synthetic timing formulas instead of empirical profiling."
        st.session_state["last_result"] = RunResult(False, 2, message, "", message, None, [])
        st.session_state["selected_run"] = ""
        return
    with st.spinner("Running simulation..."):
        result = run_simulation(form)
    st.session_state["last_result"] = result
    st.session_state["selected_run"] = str(result.run_dir) if result.run_dir else ""


def render_run_selector(output_root: str) -> Optional[Path]:
    target = parse_output_target(output_root)
    if target.is_gcs:
        labels = list_remote_runs(target)
        if not labels:
            return None
        selected = st.selectbox("Existing runs", labels, index=0)
        chosen = download_run_dir_from_gcs(selected, target)
        if chosen is None:
            st.warning(f"Could not load run from {output_root}: {selected}")
            return None
        st.session_state["selected_run"] = str(chosen)
        return chosen
    run_dirs = list_valid_run_dirs(target.local_root)
    if not run_dirs:
        return None
    labels = [str(path.relative_to(target.local_root)) for path in run_dirs]
    selected = st.selectbox("Existing runs", labels, index=0)
    chosen = target.local_root / selected
    st.session_state["selected_run"] = str(chosen)
    return chosen


def render_execution_log() -> None:
    result = st.session_state.get("last_result")
    if not isinstance(result, RunResult):
        return
    status = "Success" if result.success else f"Failed ({result.return_code})"
    st.subheader(f"Last run: {status}")
    st.code(result.output_text or "(no stdout/stderr)", language="text")
    with st.expander("Full stderr", expanded=not result.success):
        st.code(result.stderr_text or "(empty stderr)", language="text")
    with st.expander("Full stdout", expanded=False):
        st.code(result.stdout_text or "(empty stdout)", language="text")


def render_dashboard(run_dir: Path) -> None:
    metrics_path = run_dir / "request_metrics.csv"
    if not metrics_path.exists():
        stderr_path = run_dir / "stderr.log"
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        st.error(f"Simulation did not produce request metrics in {run_dir}")
        st.code(stderr_text or "(no stderr captured)", language="text")
        return
    frame = read_request_metrics(run_dir)
    render_kpis(frame)
    render_latency_plots(frame)
    render_timeline(frame)
    render_pd_chart(frame)
    with st.expander("Raw request metrics", expanded=False):
        st.dataframe(frame, use_container_width=True)
    with st.expander("Run config", expanded=False):
        render_run_metadata(run_dir)


def build_capacity_df(output_root: str) -> pd.DataFrame:
    target = parse_output_target(output_root)
    if target.is_gcs:
        ensure_remote_runs_cached(target)
    summaries = [summarize_run(path) for path in list_valid_run_dirs(target.local_root)]
    records = [summary.__dict__ for summary in summaries if summary is not None]
    return pd.DataFrame(records)


def annotate_best(fig: go.Figure, df: pd.DataFrame, x: str, y: str) -> go.Figure:
    if df.empty:
        return fig
    best = df.sort_values("qps_per_dollar", ascending=False).iloc[0]
    fig.add_trace(go.Scatter(x=[best[x]], y=[best[y]], mode="markers", marker_symbol="star", marker_size=14, marker_color="black", name="Best QPS/$"))
    return fig


def render_capacity_views(output_root: str) -> None:
    st.subheader("Capacity Planning")
    df = build_capacity_df(output_root)
    if df.empty:
        st.info("No valid runs found for capacity planning.")
        return
    empirical = df[df["backend"] == "vidur"].copy()
    if empirical.empty:
        st.info("No empirical runs found. Capacity views require backend=vidur.")
        return
    excluded_count = len(df) - len(empirical)
    if excluded_count > 0:
        st.warning(f"Excluded {excluded_count} non-empirical run(s) from capacity charts.")
    df = empirical
    st.dataframe(df[["run_name", "model", "sku", "trace", "backend", "scheduler", "tp", "pp", "batch_size", "observed_qps", "ttft_p90", "tbt_p99_ms", "qps_per_dollar"]], use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        model_filter = st.multiselect("Models", sorted(df["model"].unique()), default=sorted(df["model"].unique()))
    with c2:
        sku_filter = st.multiselect("SKUs", sorted(df["sku"].unique()), default=sorted(df["sku"].unique()))
    filtered = df[df["model"].isin(model_filter) & df["sku"].isin(sku_filter)]
    fig_a = px.scatter(filtered, x="ttft_p90", y="qps_per_dollar", color="sku", hover_data=["run_name", "trace", "scheduler", "tp", "pp", "batch_size"], title="QPS/$ vs TTFT-P90")
    st.plotly_chart(annotate_best(fig_a, filtered, "ttft_p90", "qps_per_dollar"), use_container_width=True)
    fig_b = px.scatter(filtered, x="tbt_p99_ms", y="qps_per_dollar", color="sku", hover_data=["run_name", "trace", "scheduler", "tp", "pp", "batch_size"], title="QPS/$ vs TBT-P99")
    st.plotly_chart(annotate_best(fig_b, filtered, "tbt_p99_ms", "qps_per_dollar"), use_container_width=True)
    fig_c = px.scatter(filtered, x="ttft_p90", y="tbt_p99_ms", color="qps_per_dollar", symbol="sku", hover_data=["run_name", "trace", "scheduler", "tp", "pp", "batch_size"], title="TBT-P99 vs TTFT-P90 (colored by QPS/$)")
    st.plotly_chart(annotate_best(fig_c, filtered, "ttft_p90", "tbt_p99_ms"), use_container_width=True)
    scheduler_codes = {name: idx for idx, name in enumerate(sorted(filtered["scheduler"].unique()))}
    sku_codes = {name: idx for idx, name in enumerate(sorted(filtered["sku"].unique()))}
    trace_codes = {name: idx for idx, name in enumerate(sorted(filtered["trace"].unique()))}
    parallel_df = filtered.copy()
    parallel_df["scheduler_code"] = parallel_df["scheduler"].map(scheduler_codes)
    parallel_df["sku_code"] = parallel_df["sku"].map(sku_codes)
    parallel_df["trace_code"] = parallel_df["trace"].map(trace_codes)
    parcoords = go.Figure(data=go.Parcoords(
        line=dict(color=parallel_df["qps_per_dollar"], colorscale="Viridis", showscale=True, cmin=float(parallel_df["qps_per_dollar"].min()), cmax=float(parallel_df["qps_per_dollar"].max()), colorbar=dict(title="QPS/$")),
        dimensions=[
            dict(label="PP", values=parallel_df["pp"]),
            dict(label="TP", values=parallel_df["tp"]),
            dict(label="Scheduler", values=parallel_df["scheduler_code"], tickvals=list(scheduler_codes.values()), ticktext=list(scheduler_codes.keys())),
            dict(label="Batch", values=parallel_df["batch_size"]),
            dict(label="SKU", values=parallel_df["sku_code"], tickvals=list(sku_codes.values()), ticktext=list(sku_codes.keys())),
            dict(label="Trace", values=parallel_df["trace_code"], tickvals=list(trace_codes.values()), ticktext=list(trace_codes.keys())),
        ],
    ))
    parcoords.update_layout(title="Optimal Configuration Parallel Coordinates")
    st.plotly_chart(parcoords, use_container_width=True)
    penalty_df = filtered.copy()
    penalty_df["config"] = penalty_df.apply(lambda row: f"pp={row['pp']} tp={row['tp']} sch={row['scheduler']} bs={row['batch_size']} sku={row['sku']}", axis=1)
    best_by_trace = penalty_df.groupby("trace")["qps_per_dollar"].transform("max")
    penalty_df["misconfiguration_cost_pct"] = ((best_by_trace - penalty_df["qps_per_dollar"]) / best_by_trace.replace(0, pd.NA) * 100.0).fillna(0.0)
    heatmap = penalty_df.pivot_table(index="trace", columns="config", values="misconfiguration_cost_pct", aggfunc="min").fillna(0.0)
    fig_heatmap = px.imshow(heatmap, aspect="auto", color_continuous_scale="Reds", labels={"color": "Cost of misconfiguration (%)"}, title="Cost of Misconfiguration Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)


def main() -> None:
    validate_registry()
    st.set_page_config(page_title="Odyn Simulator Dashboard", layout="wide")
    st.title("Odyn Interactive Simulator Dashboard")
    st.caption("Runs simulator with Odyn/Vidur profiles and renders request diagnostics plus capacity-planning views.")
    defaults = scenario_defaults()
    form = run_controls(defaults)
    st.sidebar.code(" ".join(base_cli_args(form)), language="bash")
    if st.sidebar.button("Run simulation", use_container_width=True):
        run_and_store(form)
    render_execution_log()
    selected = render_run_selector(form.output_root)
    if selected:
        st.subheader(f"Run: {selected}")
        render_dashboard(selected)
    render_capacity_views(form.output_root)


if __name__ == "__main__":
    main()
