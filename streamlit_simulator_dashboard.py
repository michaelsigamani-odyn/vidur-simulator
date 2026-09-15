from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from google.cloud import storage
except ImportError:
    storage = None


@dataclass
class RunForm:
    model_name: str
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


@dataclass
class RunResult:
    success: bool
    return_code: int
    output_text: str
    stdout_text: str
    stderr_text: str
    run_dir: Optional[Path]


@dataclass
class OutputTarget:
    is_gcs: bool
    local_root: Path
    bucket: Optional[str]
    prefix: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_trace_path() -> str:
    return "data/processed_traces/splitwise_conv.csv"


def scenario_defaults() -> Dict[str, RunForm]:
    configured_output_root = os.environ.get("ODYN_OUTPUT_ROOT", "odyn_simulator_output")
    common = dict(
        backend="aicb",
        interval_type="poisson",
        qps=100.0,
        requests=6,
        length_type="fixed",
        prefill_tokens=100,
        decode_tokens=8,
        trace_file=default_trace_path(),
        pd_bandwidth=800,
        nvlink_bandwidth=1600,
        rdma_bandwidth=800,
        pd_dtype="fp8",
        output_root=configured_output_root,
        extra_args="",
    )
    return {
        "Scenario 1: Qwen3-Next-80B No-PD": RunForm(
            model_name="qwen3-next-80B",
            scheduler="lor",
            replica_scheduler="sarathi",
            replicas=32,
            pd_ratio=1.0,
            tp=1,
            pp=1,
            prefill_replicas=0,
            prefill_tp=1,
            decode_tp=1,
            **common,
        ),
        "Scenario 2: Qwen3-Next-80B PD": RunForm(
            model_name="qwen3-next-80B",
            scheduler="split_wise",
            replica_scheduler="split_wise",
            replicas=8,
            pd_ratio=0.25,
            tp=1,
            pp=1,
            prefill_replicas=2,
            prefill_tp=1,
            decode_tp=1,
            **common,
        ),
        "Scenario 3: DeepSeek-671B PD": RunForm(
            model_name="deepseek-671B",
            scheduler="split_wise",
            replica_scheduler="split_wise",
            replicas=8,
            pd_ratio=0.25,
            tp=8,
            pp=1,
            prefill_replicas=0,
            prefill_tp=8,
            decode_tp=8,
            **common,
        ),
        "Scenario 4: Qwen3-MoE-235B PD": RunForm(
            model_name="qwen3-moe-235B",
            scheduler="split_wise",
            replica_scheduler="split_wise",
            replicas=8,
            pd_ratio=0.25,
            tp=4,
            pp=1,
            prefill_replicas=0,
            prefill_tp=4,
            decode_tp=4,
            **common,
        ),
    }


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
        "synthetic",
        "--interval_generator_config_type",
        form.interval_type,
        "--poisson_request_interval_generator_config_qps",
        str(form.qps),
        "--synthetic_request_generator_config_num_requests",
        str(form.requests),
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


def list_run_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


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
        run_name = suffix.split("/", 1)[0]
        if run_name:
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


def get_new_run_dir(before: Sequence[Path], output_root: Path) -> Optional[Path]:
    old_set = {path.resolve() for path in before}
    after = list_run_dirs(output_root)
    new_runs = [path for path in after if path.resolve() not in old_set]
    return new_runs[0] if new_runs else (after[0] if after else None)


def run_simulation(form: RunForm) -> RunResult:
    cwd = repo_root()
    target = parse_output_target(form.output_root)
    adapted = RunForm(**{**form.__dict__, "output_root": str(target.local_root)})
    known_runs = list_run_dirs(target.local_root)
    process = subprocess.run(base_cli_args(adapted), cwd=cwd, capture_output=True, text=True)
    run_dir = get_new_run_dir(known_runs, target.local_root)
    if run_dir and target.is_gcs:
        upload_run_dir_to_gcs(run_dir, target)
    combined = f"{process.stdout}\n{process.stderr}".strip()
    return RunResult(process.returncode == 0, process.returncode, combined, process.stdout, process.stderr, run_dir)


def read_request_metrics(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "request_metrics.csv")
    if "request_num_decode_tokens" in frame and "decode_time" in frame:
        den = frame["request_num_decode_tokens"].replace(0, pd.NA)
        frame["tbt" ] = frame["decode_time"] / den
    return frame


def percentile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    return float(clean.quantile(q)) if not clean.empty else float("nan")


def safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


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
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as handle:
        st.json(json.load(handle), expanded=False)


def run_controls(defaults: Dict[str, RunForm]) -> RunForm:
    selected = st.sidebar.selectbox("Preset", list(defaults))
    seed = defaults[selected]
    model_name = st.sidebar.text_input("Model", seed.model_name)
    backend = st.sidebar.selectbox("Backend", ["aicb", "vidur", "simai_simulation", "simai_analytical"], index=["aicb", "vidur", "simai_simulation", "simai_analytical"].index(seed.backend))
    scheduler = st.sidebar.selectbox("Global scheduler", ["split_wise", "lor", "round_robin"], index=["split_wise", "lor", "round_robin"].index(seed.scheduler))
    replica_scheduler = st.sidebar.selectbox("Replica scheduler", ["split_wise", "sarathi", "vllm"], index=["split_wise", "sarathi", "vllm"].index(seed.replica_scheduler))
    replicas = st.sidebar.number_input("Replicas", 1, 512, seed.replicas)
    pd_ratio = st.sidebar.slider("PD ratio", 0.01, 1.0, float(seed.pd_ratio), 0.01)
    tp = st.sidebar.number_input("Tensor parallel", 1, 512, seed.tp)
    pp = st.sidebar.number_input("Pipeline stages", 1, 128, seed.pp)
    prefill_replicas = st.sidebar.number_input("Prefill replicas (0=auto)", 0, 512, seed.prefill_replicas)
    prefill_tp = st.sidebar.number_input("Prefill TP", 1, 512, seed.prefill_tp)
    decode_tp = st.sidebar.number_input("Decode TP", 1, 512, seed.decode_tp)
    qps = st.sidebar.number_input("QPS", 0.01, 100000.0, float(seed.qps), 0.1)
    requests = st.sidebar.number_input("Requests", 1, 500000, seed.requests)
    prefill_tokens = st.sidebar.number_input("Prefill tokens", 1, 32768, seed.prefill_tokens)
    decode_tokens = st.sidebar.number_input("Decode tokens", 0, 32768, seed.decode_tokens)
    trace_file = st.sidebar.text_input("Trace file", seed.trace_file)
    length_type = st.sidebar.selectbox("Length type", ["fixed", "trace"], index=["fixed", "trace"].index(seed.length_type))
    interval_type = st.sidebar.selectbox("Interval type", ["poisson", "gamma", "trace"], index=["poisson", "gamma", "trace"].index(seed.interval_type))
    pd_bandwidth = st.sidebar.number_input("PD bandwidth (Gbps)", 1, 100000, seed.pd_bandwidth)
    nvlink_bandwidth = st.sidebar.number_input("NVLink bandwidth (Gbps)", 1, 100000, seed.nvlink_bandwidth)
    rdma_bandwidth = st.sidebar.number_input("RDMA bandwidth (Gbps)", 1, 100000, seed.rdma_bandwidth)
    pd_dtype = st.sidebar.selectbox("PD dtype", ["fp8", "float16", "float32"], index=["fp8", "float16", "float32"].index(seed.pd_dtype))
    output_root = st.sidebar.text_input("Output root", seed.output_root)
    extra_args = st.sidebar.text_input("Extra CLI args", seed.extra_args)
    return RunForm(model_name, backend, scheduler, replica_scheduler, int(replicas), float(pd_ratio), int(tp), int(pp), int(prefill_replicas), int(prefill_tp), int(decode_tp), float(qps), int(requests), int(prefill_tokens), int(decode_tokens), trace_file, length_type, interval_type, int(pd_bandwidth), int(nvlink_bandwidth), int(rdma_bandwidth), pd_dtype, output_root, extra_args)


def run_and_store(form: RunForm) -> None:
    with st.spinner("Running simulation..."):
        result = run_simulation(form)
    st.session_state["last_result"] = result
    st.session_state["selected_run"] = str(result.run_dir) if result.run_dir else ""


def render_run_selector(output_root: str) -> Optional[Path]:
    target = parse_output_target(output_root)
    labels = list_remote_runs(target) if target.is_gcs else [run.name for run in list_run_dirs(target.local_root)]
    if not labels:
        return None
    selected = st.selectbox("Existing runs", labels, index=0)
    chosen = download_run_dir_from_gcs(selected, target) if target.is_gcs else target.local_root / selected
    if chosen is None:
        st.warning(f"Could not load run from {output_root}: {selected}")
        return None
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
        st.warning(f"No request metrics found in {run_dir}")
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


def main() -> None:
    st.set_page_config(page_title="Odyn Simulator Dashboard", layout="wide")
    st.title("Odyn Interactive Simulator Dashboard")
    st.caption("Run new simulations and analyze request-level latency and PD communication metrics.")
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


if __name__ == "__main__":
    main()
