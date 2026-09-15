from pathlib import Path
import subprocess
import sys


def _build_command(output_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vidur.main",
        "--request_generator_config_type",
        "synthetic",
        "--interval_generator_config_type",
        "poisson",
        "--poisson_request_interval_generator_config_qps",
        "20",
        "--synthetic_request_generator_config_num_requests",
        "1",
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        "32",
        "--fixed_request_length_generator_config_decode_tokens",
        "4",
        "--trace_request_length_generator_config_trace_file",
        "data/processed_traces/splitwise_conv.csv",
        "--cluster_config_num_replicas",
        "8",
        "--replica_config_pd_node_ratio",
        "0.25",
        "--global_scheduler_config_type",
        "split_wise",
        "--replica_scheduler_config_type",
        "split_wise",
        "--replica_config_model_name",
        "qwen3-next-80B",
        "--replica_config_tensor_parallel_size",
        "1",
        "--replica_config_num_pipeline_stages",
        "1",
        "--random_forrest_execution_time_predictor_config_backend",
        "aicb",
        "--replica_config_pd_p2p_comm_bandwidth",
        "800",
        "--replica_config_nvlink_bandwidth",
        "1600",
        "--replica_config_rdma_bandwidth",
        "800",
        "--replica_config_pd_p2p_comm_dtype",
        "fp8",
        "--replica_config_num_prefill_replicas",
        "2",
        "--replica_config_prefill_tensor_parallel_size",
        "1",
        "--replica_config_decode_tensor_parallel_size",
        "1",
        "--metrics_config_output_dir",
        str(output_root),
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_plots",
    ]


def test_simulation_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke-output"
    command = _build_command(output_root)
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert run_dirs, "No run directory was generated"
    assert (run_dirs[0] / "request_metrics.csv").exists()
