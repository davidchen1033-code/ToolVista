#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
source "${TOOLVISTA_ROOT}/scripts/autodl/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: benchmark.sh --stage c3|c4 --variant VARIANT [--algorithm ALGORITHM] [--scope SCOPE] [mode]

C4 requires --algorithm grpo|bnpo|dapo|drgrpo and accepts --scope light|full.
C3 ignores algorithms and supports light only.
Set ADAPTER explicitly for checkpoints without a completed formal default.

Modes:
  --tmux         Launch merge, services, and evaluation in tmux (default).
  --foreground   Run orchestration in the current shell.
  --attach       Enter the live percentage monitor after launch (default).
  --detach       Launch tmux and return to the shell.
EOF
}

pipeline_stage="${PIPELINE_STAGE:-}"
variant_input="${EXPERIMENT_VARIANT:-}"
algorithm_input="${ALGORITHM:-}"
mode="${MODE:-tmux}"
attach="${ATTACH:-true}"
scope="${SCOPE:-light}"

while (( $# )); do
  case "$1" in
    --stage) pipeline_stage="${2:?missing value for --stage}"; shift 2 ;;
    --variant) variant_input="${2:?missing value for --variant}"; shift 2 ;;
    --algorithm) algorithm_input="${2:?missing value for --algorithm}"; shift 2 ;;
    --scope) scope="${2:?missing value for --scope}"; shift 2 ;;
    --tmux) mode=tmux; shift ;;
    --foreground|--orchestrate) mode=foreground; shift ;;
    --attach) attach=true; shift ;;
    --detach) attach=false; shift ;;
    --help|-h) usage; exit 0 ;;
    *) autodl_die "unknown argument: $1" ;;
  esac
done

case "${pipeline_stage}" in c3|c4) ;; *) autodl_die "--stage must be c3 or c4" ;; esac
case "${scope}" in light|full) ;; *) autodl_die "--scope must be light or full" ;; esac
if [[ "${pipeline_stage}" == c3 && "${scope}" != light ]]; then
  autodl_die "C3 benchmark supports --scope light only"
fi
[[ -n "${variant_input}" ]] || autodl_die "--variant is required"
variant="$(autodl_normalize_variant "${variant_input}")"
if [[ "${pipeline_stage}" == c4 ]]; then
  [[ -n "${algorithm_input}" ]] || autodl_die "C4 benchmark requires --algorithm"
  algorithm="$(autodl_normalize_algorithm "${algorithm_input}")"
else
  algorithm=router
fi
autodl_require_bool ATTACH "${attach}"
case "${mode}" in tmux|foreground) ;; *) autodl_die "MODE must be tmux or foreground: ${mode}" ;; esac

seed="${SEED:-42}"
timestamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
default_adapter=""
default_merged=""
default_baseline=""

case "${pipeline_stage}:${variant}" in
  c3:raw)
    expected_steps=3414
    suite=c3_light
    default_adapter="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_replay_sft/v0-20260719-151204/checkpoint-3414"
    history_default="${TOOLVISTA_ROOT}/runs/c2_toolvqa_light_20260719T064824Z_seed42/summary.json"
    ;;
  c3:datafixed)
    expected_steps=1063
    suite=c3_datafixed_light
    default_adapter="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed_sft/v1-20260721-203434/checkpoint-1063"
    default_merged="${TOOLVISTA_ROOT}/outputs/final/c3-mllm-router-datafixed-seed42-step1063-merged"
    history_default="${TOOLVISTA_ROOT}/runs/c2_datafixed_light_seed42/summary.json"
    ;;
  c3:datafixed-withnegative)
    expected_steps=1215
    suite=c3_datafixed_agentretention_light
    default_adapter="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention_sft/20260723T052946Z/v0-20260723-133020/checkpoint-1215"
    default_merged="${TOOLVISTA_ROOT}/outputs/final/c3-mllm-router-datafixed-agentretention-seed42-step1215-merged"
    history_default="${TOOLVISTA_ROOT}/runs/c2_datafixed_light_seed42/summary.json"
    ;;
  c3:datafixed-withnegative-terminal)
    expected_steps=1700
    suite=c3_datafixed_agentretention_light
    default_adapter="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention-terminalfix_sft/terminalfix20260723T100256Z/v0-20260723-180402/checkpoint-1700"
    default_merged="${TOOLVISTA_ROOT}/outputs/final/c3-mllm-router-datafixed-agentretention-terminalfix-seed42-step1700-merged"
    history_default="${TOOLVISTA_ROOT}/runs/c2_datafixed_light_seed42/summary.json"
    ;;
  c4:raw)
    expected_steps=6000
    suite=c4_light
    history_default="${TOOLVISTA_ROOT}/runs/c3_router_light_20260719T135247Z_seed42/summary.json"
    ;;
  c4:datafixed)
    expected_steps=6000
    suite=c4_datafixed_light
    history_default="${TOOLVISTA_ROOT}/runs/c3_datafixed_light_seed42_step1063/summary.json"
    if [[ "${algorithm}" == dapo ]]; then
      default_adapter="${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed_seed42_run20260722T060254Z/v0-20260722-141110/checkpoint-6000"
      default_merged="${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-dapo-datafixed-seed42-run20260722T060254Z-step6000-merged"
    fi
    ;;
  c4:datafixed-withnegative)
    expected_steps=6000
    suite=c4_datafixed_agentretention_light
    history_default="${TOOLVISTA_ROOT}/runs/c3_datafixed-agentretention_light_seed42_step1215/summary.json"
    ;;
  c4:datafixed-withnegative-terminal)
    expected_steps=6000
    suite=c4_datafixed_agentretention_light
    history_default="${TOOLVISTA_ROOT}/runs/c3_datafixed-agentretention-terminalfix_light_seed42_step1700/summary.json"
    case "${algorithm}" in
      dapo)
        default_adapter="${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed-withnegative_seed42_runterminalfix20260723T142400Z/v0-20260723-222537/checkpoint-6000"
        default_merged="${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-dapo-datafixed-withnegative-terminalfix-seed42-runterminalfix20260723T142400Z-step6000-merged"
        default_baseline="${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative_light_seed42_step6000/summary.json"
        ;;
      grpo)
        default_adapter="${TOOLVISTA_ROOT}/outputs/rl/grpo_datafixed-withnegative_seed42_runterminalfix-grpo20260724T055108Z/v0-20260724-135218/checkpoint-6000"
        default_merged="${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-grpo-datafixed-withnegative-terminalfix-seed42-runterminalfix-grpo20260724T055108Z-step6000-merged"
        default_baseline="${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative-terminalfix_light_seed42_step6000/summary.json"
        ;;
      bnpo)
        default_adapter="${TOOLVISTA_ROOT}/outputs/rl/bnpo_datafixed-withnegative_seed42_runterminalfix-bnpo20260724T114133Z/v0-20260724-194247/checkpoint-6000"
        default_merged="${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-bnpo-datafixed-withnegative-terminalfix-seed42-runterminalfix-bnpo20260724T114133Z-step6000-merged"
        default_baseline="${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative-terminalfix_light_seed42_step6000/summary.json"
        ;;
      drgrpo)
        default_baseline="${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative-terminalfix_light_seed42_step6000/summary.json"
        ;;
    esac
    ;;
esac

if [[ "${pipeline_stage}" == c4 && "${scope}" == full ]]; then
  case "${variant}" in
    raw) suite=c4_full ;;
    datafixed) suite=c4_datafixed_full ;;
    datafixed-withnegative|datafixed-withnegative-terminal)
      suite=c4_datafixed_agentretention_full
      ;;
  esac
  # Existing defaults point to light runs and are not valid full baselines.
  default_baseline=""
fi

expected_steps="${EXPECTED_STEPS:-${expected_steps}}"
adapter="${ADAPTER:-${default_adapter}}"
[[ -n "${adapter}" ]] || autodl_die "ADAPTER is required; no completed default exists for ${pipeline_stage}/${variant}/${algorithm}"
merged="${MERGED:-${default_merged:-${TOOLVISTA_ROOT}/outputs/final/${pipeline_stage}-${algorithm}-${variant}-seed${seed}-step${expected_steps}-merged}}"
history_summary="${HISTORY_SUMMARY:-${history_default}}"
baseline_summary="${BASELINE_SUMMARY:-${default_baseline}}"
session="${SESSION:-${pipeline_stage}-${variant}-${algorithm}-benchmark}"
run_id="${RUN_ID:-${pipeline_stage}_${algorithm}_${variant}_${scope}_${timestamp}_seed${seed}_step${expected_steps}}"
api_model="${API_MODEL:-${pipeline_stage}-${algorithm}-${variant}-step${expected_steps}}"
api_base="${API_BASE:-http://127.0.0.1:8010/v1}"
judge_api_base="${JUDGE_API_BASE:-http://127.0.0.1:8011/v1}"
judge_model="${JUDGE_MODEL:-Qwen3-VL-4B-Instruct}"
stage_name="${RUN_STAGE:-${pipeline_stage}-${algorithm}-${variant}}"
wandb_group="${WANDB_GROUP:-${pipeline_stage}-${algorithm}-${variant}}"
log_dir="${LOG_DIR:-${TOOLVISTA_ROOT}/logs/${pipeline_stage}}"
orchestrator_log="${ORCH_LOG:-${log_dir}/${run_id}.orchestrator.log}"
candidate_log="${CANDIDATE_LOG:-${log_dir}/${run_id}.candidate-8010.log}"
judge_log="${JUDGE_LOG:-${log_dir}/${run_id}.judge-8011.log}"
benchmark_log="${BENCHMARK_LOG:-${log_dir}/${run_id}.benchmark.log}"
if [[ "${variant}" == raw ]]; then
  benchmark_config="${TOOLVISTA_ROOT}/configs/evaluation/benchmarks.yaml"
  suite_config="${TOOLVISTA_ROOT}/configs/evaluation/suites.yaml"
else
  benchmark_config="${TOOLVISTA_ROOT}/configs/evaluation/benchmarks_datafixed.yaml"
  suite_config="${TOOLVISTA_ROOT}/configs/evaluation/suites_datafixed.yaml"
fi

verify_merged() {
  [[ -d "$1" && -s "$1/config.json" ]] || return 1
  find "$1" -maxdepth 1 -type f -name '*.safetensors' -size +0c | grep -q .
}

wait_for_api_model() {
  local label="$1" base="$2" model="$3" pid="$4" attempt
  for attempt in $(seq 1 180); do
    kill -0 "${pid}" 2>/dev/null || { echo "${label} exited before becoming ready" >&2; return 1; }
    if curl -fsS "${base}/models" 2>/dev/null | "${PYTHON_BIN}" -c 'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if expected in {item.get("id") for item in data.get("data", [])} else 1)' "${model}" >/dev/null 2>&1; then
      echo "${label} ready: ${model}"
      return 0
    fi
    sleep 5
  done
  echo "timed out waiting for ${label}: ${model}" >&2
  return 1
}

run_evaluator() {
  common_env=(
    env TOOLVISTA_ROOT="${TOOLVISTA_ROOT}" PYTHON_BIN="${PYTHON_BIN}" SWIFT_BIN="${SWIFT_BIN}"
    RUN_ID="${run_id}" API_BASE="${api_base}" API_MODEL="${api_model}" EVAL_URL="${api_base}"
    JUDGE_API_BASE="${judge_api_base}" JUDGE_MODEL="${judge_model}" HISTORY_SUMMARY="${history_summary}"
    SEED="${seed}" WANDB_PROJECT=toolvista-qwen3vl-eval WANDB_GROUP="${wandb_group}"
  )
  [[ -z "${baseline_summary}" ]] || common_env+=(BASELINE_SUMMARY="${baseline_summary}")

  case "${pipeline_stage}:${variant}" in
    c3:raw)
      "${common_env[@]}" CHECKPOINT="${adapter}" STAGE="${stage_name}" SUITE="${suite}" C3_ADAPTER_CHECKPOINT="${adapter}" C3_RETENTION_GATE="${adapter}/toolvista_c3_retention_gate.json" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_raw.sh"
      ;;
    c3:datafixed)
      "${common_env[@]}" CHECKPOINT="${adapter}" C3_DATAFIXED_ADAPTER_CHECKPOINT="${adapter}" STAGE="${stage_name}" EVAL_SUITE="${suite}" RETENTION_VARIANT=datafixed ROUTER_BENCHMARK_ID=mllm_router_datafixed WANDB_GROUP="${wandb_group}" C3_DATAFIXED_RETENTION_GATE="${adapter}/toolvista_c3_datafixed_retention_gate.json" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_c3_datafixed.sh"
      ;;
    c3:datafixed-withnegative)
      "${common_env[@]}" CHECKPOINT="${adapter}" C3_DATAFIXED_ADAPTER_CHECKPOINT="${adapter}" STAGE="${stage_name}" EVAL_SUITE="${suite}" RETENTION_VARIANT=datafixed-agentretention ROUTER_BENCHMARK_ID=mllm_router_datafixed_agentretention TOOLVQA_ANSWER_RETENTION=0.80 BFCL_IRRELEVANCE_MINIMUM=0.20 ROUTER_VIDEO_MINIMUM=0.25 ROUTER_INVALID_MAXIMUM=0.05 C3_DATAFIXED_RETENTION_GATE="${adapter}/toolvista_c3_datafixed_agentretention_retention_gate.json" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_c3_datafixed.sh"
      ;;
    c3:datafixed-withnegative-terminal)
      "${common_env[@]}" CHECKPOINT="${adapter}" RUN_STAGE="${stage_name}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_c3_datafixed_withnegative_terminal.sh"
      ;;
    c4:raw)
      eval_history="${history_summary}"
      [[ "${scope}" != full ]] || eval_history=""
      "${common_env[@]}" HISTORY_SUMMARY="${eval_history}" CHECKPOINT="${merged}" STAGE="${stage_name}" SUITE="${suite}" ALGORITHM="${algorithm}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_raw.sh"
      ;;
    c4:datafixed|c4:datafixed-withnegative|c4:datafixed-withnegative-terminal)
      if [[ "${variant}" == datafixed ]]; then
        eval_variant=datafixed
        history_router=mllm_router_datafixed
      else
        eval_variant=datafixed-withnegative
        history_router=mllm_router_datafixed_agentretention
      fi
      "${common_env[@]}" EVALUATION_SCOPE="${scope}" CHECKPOINT="${merged}" SUITE="${suite}" ALGORITHM="${algorithm}" EXPERIMENT_VARIANT="${eval_variant}" HISTORY_ROUTER_BENCHMARK_ID="${history_router}" STAGE="${stage_name}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_c4_datafixed.sh"
      ;;
  esac
}

run_orchestrator() {
  local candidate_pid="" judge_pid="" staging="${merged}.partial.$$"
  cleanup() {
    local pid
    for pid in "${candidate_pid}" "${judge_pid}"; do
      [[ -n "${pid}" ]] || continue
      kill -TERM -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    done
    [[ ! -e "${staging}" ]] || rm -rf "${staging}"
  }
  trap cleanup EXIT INT TERM
  mkdir -p "${log_dir}" "${TOOLVISTA_ROOT}/outputs/final"
  exec > >(tee -a "${orchestrator_log}") 2>&1

  "${PYTHON_BIN}" - "${adapter}" "${expected_steps}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve(strict=True)
expected = int(sys.argv[2])
state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
if state.get("global_step") != expected or state.get("max_steps") != expected:
    raise SystemExit(f"checkpoint is incomplete: {state.get('global_step')}/{state.get('max_steps')} != {expected}/{expected}")
if not (checkpoint / "adapter_model.safetensors").is_file():
    raise SystemExit("checkpoint is missing adapter_model.safetensors")
print(f"Verified checkpoint: {expected}/{expected}")
PY

  if verify_merged "${merged}"; then
    echo "Merged model already exists and passed verification: ${merged}"
  else
    [[ ! -e "${merged}" ]] || autodl_die "refusing invalid pre-existing merged path: ${merged}"
    env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" export --adapters "${adapter}" --merge_lora true --torch_dtype bfloat16 --safe_serialization true --max_shard_size 5GB --output_dir "${staging}"
    verify_merged "${staging}" || autodl_die "merged model verification failed: ${staging}"
    mv "${staging}" "${merged}"
  fi

  export PATH="$(dirname "${SWIFT_BIN}"):${PATH}"
  export PYTHONPATH="/root/autodl-tmp/ms-swift:${TOOLVISTA_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"

  setsid env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" deploy --model "${merged}" --served_model_name "${api_model}" --infer_backend vllm --vllm_max_model_len 32768 --vllm_max_num_seqs 16 --vllm_gpu_memory_utilization 0.55 --host 127.0.0.1 --port 8010 > "${candidate_log}" 2>&1 &
  candidate_pid="$!"
  wait_for_api_model candidate "${api_base}" "${api_model}" "${candidate_pid}" || { tail -n 160 "${candidate_log}" || true; return 4; }

  setsid env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" deploy --model "${BASE_MODEL}" --served_model_name "${judge_model}" --infer_backend vllm --vllm_max_model_len 8192 --vllm_max_num_seqs 4 --vllm_gpu_memory_utilization 0.35 --host 127.0.0.1 --port 8011 > "${judge_log}" 2>&1 &
  judge_pid="$!"
  wait_for_api_model judge "${judge_api_base}" "${judge_model}" "${judge_pid}" || { tail -n 160 "${judge_log}" || true; return 4; }

  set +e
  run_evaluator 2>&1 | tee "${benchmark_log}"
  benchmark_rc="${PIPESTATUS[0]}"
  set -e
  echo "BENCHMARK_EXIT=${benchmark_rc}"
  echo "Summary: ${TOOLVISTA_ROOT}/runs/${run_id}/summary.json"
  cleanup
  trap - EXIT INT TERM
  return "${benchmark_rc}"
}

autodl_require_dir TOOLVISTA_ROOT "${TOOLVISTA_ROOT}"
autodl_require_dir BASE_MODEL "${BASE_MODEL}"
autodl_require_dir ADAPTER "${adapter}"
autodl_require_file ADAPTER_MODEL "${adapter}/adapter_model.safetensors"
autodl_require_file TRAINER_STATE "${adapter}/trainer_state.json"
autodl_require_file HISTORY_SUMMARY "${history_summary}"
[[ -z "${baseline_summary}" ]] || autodl_require_file BASELINE_SUMMARY "${baseline_summary}"
[[ -x "${PYTHON_BIN}" ]] || autodl_die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -x "${SWIFT_BIN}" ]] || autodl_die "SWIFT_BIN is not executable: ${SWIFT_BIN}"

if [[ "${mode}" == foreground ]]; then
  run_orchestrator
  exit "$?"
fi

case "${DRY_RUN:-0}" in
  1|true)
    echo "DRY_RUN=OK"
    echo "STAGE=${pipeline_stage}"
    echo "VARIANT=${variant}"
    echo "SCOPE=${scope}"
    echo "ALGORITHM=${algorithm}"
    echo "ADAPTER=${adapter}"
    echo "MERGED=${merged}"
    echo "SUITE=${suite}"
    echo "RUN_ID=${run_id}"
    echo "SESSION=${session}"
    exit 0
    ;;
  0|false) ;;
  *) autodl_die "DRY_RUN must be 0, 1, false, or true" ;;
esac

command -v tmux >/dev/null || autodl_die "tmux is not installed"
command -v nvidia-smi >/dev/null || autodl_die "nvidia-smi is not installed"
tmux has-session -t "${session}" 2>/dev/null && autodl_die "tmux session already exists: ${session}"
[[ ! -e "${TOOLVISTA_ROOT}/runs/${run_id}" ]] || autodl_die "RUN_ID already exists: ${run_id}"
if pgrep -af '[s]wift deploy|[d]eploy.py|VLLM::EngineCore|[s]cripts/evaluate.py' >/dev/null; then
  autodl_die "an inference or evaluation process is already running"
fi
gpu_pids="$(autodl_gpu_pids)"
[[ -z "${gpu_pids}" ]] || autodl_die "GPU has active compute process IDs: ${gpu_pids}"

mkdir -p "${log_dir}"
printf -v run_command \
  'cd %q; env TOOLVISTA_ROOT=%q PYTHON_BIN=%q SWIFT_BIN=%q BASE_MODEL=%q PIPELINE_STAGE=%q EXPERIMENT_VARIANT=%q ALGORITHM=%q SCOPE=%q ADAPTER=%q MERGED=%q EXPECTED_STEPS=%q HISTORY_SUMMARY=%q BASELINE_SUMMARY=%q SESSION=%q RUN_ID=%q API_MODEL=%q API_BASE=%q JUDGE_API_BASE=%q JUDGE_MODEL=%q RUN_STAGE=%q WANDB_GROUP=%q LOG_DIR=%q ORCH_LOG=%q CANDIDATE_LOG=%q JUDGE_LOG=%q BENCHMARK_LOG=%q bash %q --stage %q --variant %q --scope %q %s --foreground; status=$?; echo "ORCHESTRATOR_EXIT=${status}"; exec bash' \
  "${TOOLVISTA_ROOT}" "${TOOLVISTA_ROOT}" "${PYTHON_BIN}" "${SWIFT_BIN}" "${BASE_MODEL}" "${pipeline_stage}" "${variant}" "${algorithm}" \
  "${scope}" "${adapter}" "${merged}" "${expected_steps}" "${history_summary}" "${baseline_summary}" "${session}" "${run_id}" "${api_model}" "${api_base}" "${judge_api_base}" "${judge_model}" \
  "${stage_name}" "${wandb_group}" "${log_dir}" "${orchestrator_log}" "${candidate_log}" "${judge_log}" "${benchmark_log}" "${BASH_SOURCE[0]}" "${pipeline_stage}" "${variant}" \
  "${scope}" "$([[ "${pipeline_stage}" == c4 ]] && printf '%q %q' --algorithm "${algorithm}")"
printf -v monitor_command 'while true; do clear; %q %q/scripts/monitor_eval.py --run-id %q --suite %q --api-base %q --benchmark-config %q --suite-config %q --candidate-log %q 2>/dev/null | awk '\''/^ToolVista eval snapshot/{show=1} show{print} /^\[6\/6\]/{exit}'\''; sleep 5; done' "${PYTHON_BIN}" "${TOOLVISTA_ROOT}" "${run_id}" "${suite}" "${api_base}" "${benchmark_config}" "${suite_config}" "${candidate_log}"
printf -v logs_command 'tail --retry -n 80 -F %q %q %q %q; exec bash' "${orchestrator_log}" "${candidate_log}" "${judge_log}" "${benchmark_log}"
printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

tmux new-session -d -s "${session}" -n run "${run_command}"
tmux new-window -d -t "${session}" -n monitor "${monitor_command}"
tmux new-window -d -t "${session}" -n logs "${logs_command}"
tmux new-window -d -t "${session}" -n gpu "${gpu_command}"
tmux select-window -t "${session}:monitor"

echo "SESSION=${session}"
echo "RUN_ID=${run_id}"
echo "SUMMARY=${TOOLVISTA_ROOT}/runs/${run_id}/summary.json"
echo "Attach: tmux attach-session -t ${session}"

if [[ "${attach}" == true ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "${session}"
  fi
  exec tmux attach-session -t "${session}"
fi
