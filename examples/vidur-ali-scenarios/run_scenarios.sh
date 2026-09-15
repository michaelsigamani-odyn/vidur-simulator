#!/usr/bin/env bash
# =============================================================================
# run_scenarios.sh — SimAI / AICB Vidur 
#
#  examples/vidur-ali-scenarios/ :
#   examples/vidur-ali-scenarios/
#   ├── run_scenarios.sh               ← 
#   ├── logs/                          ← tee 
#   │   └── scenario_<N>_<TIMESTAMP>.log
#   └── simulator_output/              ← vidur  ( --output_dir )
#       └── <YYYY-MM-DD_HH-MM-SS>/
#
# :
#   bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario <1|2|3|4>
#   bash examples/vidur-ali-scenarios/run_scenarios.sh --all
#   bash examples/vidur-ali-scenarios/run_scenarios.sh -h | --help
#
# :
#   1  Qwen3-Next-80B  PD  ws=32 (dp=32, tp=1, pp=1, ep=32)     : lor
#   2  Qwen3-Next-80B  PD    ws=8  (P=2, D=6, tp=1, pp=1)         : split_wise
#   3  DeepSeek-671B   PD    ws=8  (P=2, D=6, tp=8, pp=1, EP auto)  : split_wise
#   4  Qwen3-MoE-235B  PD    ws=8  (P=2, D=6, tp=4, pp=1, EP auto)  : split_wise
#
# :
#   conda activate vidur
#   conda : /root/miniconda3/envs/vidur
#   python:     /root/miniconda3/envs/vidur/bin/python
# =============================================================================

set -euo pipefail

# =====================  =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDUR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/simulator_output"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# =====================  =====================

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "[WARN] Script exited abnormally (), exit_code=$exit_code"
        echo "       Log dir (): $LOG_DIR"
        echo "       Output dir (): $OUTPUT_DIR"
    fi
}
trap 'cleanup' EXIT INT TERM

validate_environment() {
    local conda_env="${CONDA_DEFAULT_ENV:-}"
    if [[ "$conda_env" != "vidur" ]]; then
        echo "[ERROR] vidur conda env not detected ( vidur conda )"
        echo "        Current env (): ${conda_env:-N/A}"
        echo "        Please run (): conda activate vidur"
        echo "        conda path (): /root/miniconda3/envs/vidur"
        exit 1
    fi
    local python_bin
    python_bin="$(which python 2>/dev/null || true)"
    if [[ "$python_bin" != */miniconda3/envs/vidur/* ]]; then
        echo "[ERROR] python not in vidur env (python  vidur )"
        echo "        Current python ( python): ${python_bin:-not found}"
        echo "        Expected path (): /root/miniconda3/envs/vidur/bin/python"
        exit 1
    fi
    echo "[INFO] Env check passed (): conda=$conda_env, python=$python_bin"
}

check_disk_space() {
    local required_gb=10
    local available
    available=$(df "$SCRIPT_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ "$available" -lt "$required_gb" ]]; then
        echo "[ERROR] Insufficient disk space (): need ${required_gb}GB, available ${available}GB"
        exit 1
    fi
    echo "[INFO] Disk check passed (): available ${available}GB, need ${required_gb}GB"
}

progress_bar() {
    local current=$1 total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local bar
    bar=$(printf "%${filled}s" | tr ' ' '=')
    local empty=$((20 - filled))
    local space
    space=$(printf "%${empty}s")
    printf "\n[%-20s] %d%% (%d/%d)\n" "${bar}${space}" "$percent" "$current" "$total"
}

validate_scenario_output() {
    local scenario_num=$1
    local output_dir=$2
    # Find the latest timestamped output directory
    # Use || true to prevent SIGPIPE when ls outputs multiple entries under set -eo pipefail
    local latest_dir
    latest_dir=$(ls -td "$output_dir"/*/ 2>/dev/null | head -1) || true
    if [[ -z "$latest_dir" ]]; then
        echo "[WARN] Scenario $scenario_num: no output directory found in $output_dir"
        return 1
    fi
    if [[ -f "$latest_dir/chrome_trace.json" ]]; then
        echo "[INFO] Scenario $scenario_num: validated (chrome_trace.json found)"
    else
        echo "[WARN] Scenario $scenario_num: chrome_trace.json NOT found in $latest_dir"
        return 1
    fi
}

# =====================  =====================
# //
COMMON_ARGS=(
    # 
    --replica_config_pd_p2p_comm_bandwidth  800
    --replica_config_nvlink_bandwidth       1600
    --replica_config_rdma_bandwidth         800
    --replica_config_pd_p2p_comm_dtype      fp8
    --replica_config_network_device         h20_dgx
    --replica_config_device                 h20
    # : Poisson QPS=100,  prefill=100 / decode=8
    --request_generator_config_type         synthetic
    --interval_generator_config_type        poisson
    --poisson_request_interval_generator_config_qps 100
    --synthetic_request_generator_config_num_requests 4
    --length_generator_config_type          fixed
    --fixed_request_length_generator_config_prefill_tokens 100
    --fixed_request_length_generator_config_decode_tokens  8
    --trace_request_length_generator_config_trace_file \
        ./data/processed_traces/splitwise_conv.csv
    # 
    --random_forrest_execution_time_predictor_config_backend aicb
    #  → examples/vidur-ali-scenarios/simulator_output/
    --metrics_config_output_dir "$OUTPUT_DIR"
)

# =====================  =====================

# -----------------------------------------------------------------------
#  1: Qwen3-Next-80B PD
#   cluster_config_num_replicas = 32 ( dp=32)
#   ws = tp(1) × pp(1) × dp(32) = 32，ep = ws = 32（）
#   : global=lor, replica=sarathi
# -----------------------------------------------------------------------
run_scenario_1() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_1_${ts}.log"
    echo "[INFO] === Scenario 1: Qwen3-Next-80B, no PD, ws=32, lor (1: PD, ws=32, lor) ==="
    echo "[INFO] Log (): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas         32 \
        --replica_config_pd_node_ratio        1 \
        --global_scheduler_config_type        lor \
        --replica_scheduler_config_type       sarathi \
        --replica_config_model_name           qwen3-next-80B \
        --replica_config_tensor_parallel_size 1 \
        --replica_config_num_pipeline_stages  1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 1 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 1 "$OUTPUT_DIR"
    echo "[INFO] Scenario 1 done (1 )"
}

# -----------------------------------------------------------------------
#  2: Qwen3-Next-80B PD
#    replica=8; num_prefill_replicas=2 → prefill dp=2, decode dp=6
#   prefill: ws = tp(1) × pp(1) × dp(2) = 2，ep = 2
#   decode:  ws = tp(1) × pp(1) × dp(6) = 6，ep = 6
#   : global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_2() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_2_${ts}.log"
    echo "[INFO] === Scenario 2: Qwen3-Next-80B, PD, P=2 D=6, split_wise (2: PD, P=2 D=6) ==="
    echo "[INFO] Log (): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --replica_config_num_prefill_replicas           2 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    qwen3-next-80B \
        --replica_config_tensor_parallel_size          1 \
        --replica_config_num_pipeline_stages           1 \
        --replica_config_prefill_tensor_parallel_size  1 \
        --replica_config_prefill_num_pipeline_stages   1 \
        --replica_config_decode_tensor_parallel_size   1 \
        --replica_config_decode_num_pipeline_stages    1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 2 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 2 "$OUTPUT_DIR"
    echo "[INFO] Scenario 2 done (2 )"
}

# -----------------------------------------------------------------------
#  3: DeepSeek-671B PD
#    replica=8; pd_node_ratio=0.25 → prefill dp=2, decode dp=6
#   ws = tp(8) × pp(1) × dp = 16(P)/48(D)，EP auto-set to world_size
#   : global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_3() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_3_${ts}.log"
    echo "[INFO] === Scenario 3: DeepSeek-671B, PD, tp=8, EP=auto, split_wise (3: PD, tp=8, EP=auto) ==="
    echo "[INFO] Log (): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    deepseek-671B \
        --replica_config_tensor_parallel_size          8 \
        --replica_config_num_pipeline_stages           1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 3 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 3 "$OUTPUT_DIR"
    echo "[INFO] Scenario 3 done (3 )"
}

# -----------------------------------------------------------------------
#  4: Qwen3-MoE-235B PD
#    replica=8; pd_node_ratio=0.25 → prefill dp=2, decode dp=6
#   ws = tp(4) × pp(1) × dp = 8(P)/24(D)，EP auto-set to world_size
#   : global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_4() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_4_${ts}.log"
    echo "[INFO] === Scenario 4: Qwen3-MoE-235B, PD, tp=4, EP=auto, split_wise (4: PD, tp=4, EP=auto) ==="
    echo "[INFO] Log (): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    qwen3-moe-235B \
        --replica_config_tensor_parallel_size          4 \
        --replica_config_num_pipeline_stages           1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 4 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 4 "$OUTPUT_DIR"
    echo "[INFO] Scenario 4 done (4 )"
}

# =====================  =====================

print_help() {
    cat <<'EOF'
Usage ():
  bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario <N>   Run single scenario (, N=1~4)
  bash examples/vidur-ali-scenarios/run_scenarios.sh --all            Run all 4 scenarios ()
  bash examples/vidur-ali-scenarios/run_scenarios.sh -h | --help      Print help ()

Scenarios ():
  1  Qwen3-Next-80B  no PD (PD)  ws=32             scheduler: lor
  2  Qwen3-Next-80B  PD (PD)      ws=8 (P=2,D=6)    scheduler: split_wise
  3  DeepSeek-671B   PD (PD)      tp=8, EP=auto     scheduler: split_wise
  4  Qwen3-MoE-235B  PD (PD)      tp=4, EP=auto     scheduler: split_wise

Output dir (): examples/vidur-ali-scenarios/simulator_output/<TIMESTAMP>/
Log dir (): examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log
EOF
}

# =====================  =====================

main() {
    # --help / -h ，
    case "${1:-}" in
        -h|--help|"") print_help; exit 0 ;;
    esac

    echo "============================================================"
    echo " SimAI / AICB Vidur 4-Scenario Runner ()"
    echo " Root dir (): $SCRIPT_DIR"
    echo "============================================================"

    validate_environment
    check_disk_space

    case "${1:-}" in
        --scenario)
            case "${2:-}" in
                1) run_scenario_1 ;;
                2) run_scenario_2 ;;
                3) run_scenario_3 ;;
                4) run_scenario_4 ;;
                *) echo "[ERROR] Invalid scenario (): ${2:-}, use 1~4"; exit 1 ;;
            esac
            ;;
        --all)
            local total=4
            run_scenario_1
            progress_bar 1 $total

            run_scenario_2
            progress_bar 2 $total

            run_scenario_3
            progress_bar 3 $total

            run_scenario_4
            progress_bar 4 $total

            echo ""
            echo "[INFO] All 4 scenarios completed ( 4 )!"
            echo "       Logs (): $LOG_DIR/"
            echo "       Output (): $OUTPUT_DIR/"
            ;;
        *)
            echo "[ERROR] Unknown argument (): $1"
            print_help
            exit 1
            ;;
    esac
}

main "$@"
