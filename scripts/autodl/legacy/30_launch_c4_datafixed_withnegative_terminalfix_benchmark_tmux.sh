#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed-withnegative_seed42_runterminalfix20260723T142400Z/v0-20260723-222537/checkpoint-6000}"
MERGED="${MERGED:-${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-dapo-datafixed-withnegative-terminalfix-seed42-runterminalfix20260723T142400Z-step6000-merged}"
HISTORY_SUMMARY="${HISTORY_SUMMARY:-${TOOLVISTA_ROOT}/runs/c3_datafixed-agentretention-terminalfix_light_seed42_step1700/summary.json}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative_light_seed42_step6000/summary.json}"
SESSION="${SESSION:-c4-datafixed-withnegative-terminalfix-benchmark}"
RUN_ID="${RUN_ID:-c4_visualrft_coco_dapo_datafixed-withnegative-terminalfix_light_seed42_step6000}"
API_MODEL="${API_MODEL:-c4-dapo-datafixed-withnegative-terminalfix-step6000}"
API_BASE="${API_BASE:-http://127.0.0.1:8010/v1}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:8011/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-VL-4B-Instruct}"
SUITE="${SUITE:-c4_datafixed_agentretention_light}"
ALGORITHM="${ALGORITHM:-dapo}"
STAGE="${STAGE:-c4-visualrft-datafixed-withnegative-terminalfix}"
WANDB_GROUP="${WANDB_GROUP:-c4-visualrft-datafixed-withnegative-terminalfix}"
LOG_DIR="${LOG_DIR:-${TOOLVISTA_ROOT}/logs/c4}"
ORCH_LOG="${ORCH_LOG:-${LOG_DIR}/${RUN_ID}.orchestrator.log}"
CANDIDATE_LOG="${CANDIDATE_LOG:-${LOG_DIR}/${RUN_ID}.candidate-8010.log}"
JUDGE_LOG="${JUDGE_LOG:-${LOG_DIR}/${RUN_ID}.judge-8011.log}"
BENCHMARK_LOG="${BENCHMARK_LOG:-${LOG_DIR}/${RUN_ID}.benchmark.log}"
SCRIPT_PATH="$(realpath "$0")"

require_file() {
  [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

verify_merged() {
  local path="$1"
  [[ -d "${path}" ]] || return 1
  [[ -s "${path}/config.json" ]] || return 1
  find "${path}" -maxdepth 1 -type f -name '*.safetensors' -size +0c | grep -q .
}

wait_for_api_model() {
  local label="$1"
  local base="$2"
  local model="$3"
  local pid="$4"
  local attempt
  for attempt in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${label} exited before becoming ready." >&2
      return 1
    fi
    if curl -fsS "${base}/models" 2>/dev/null |
      "${PYTHON_BIN}" -c 'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if expected in {item.get("id") for item in data.get("data", [])} else 1)' "${model}" >/dev/null 2>&1; then
      echo "${label} ready: ${model}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${label}: ${model}" >&2
  return 1
}

run_orchestrator() {
  local candidate_pid=""
  local judge_pid=""
  local staging="${MERGED}.partial.$$"

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

  mkdir -p "${LOG_DIR}" "${TOOLVISTA_ROOT}/outputs/final"
  exec > >(tee -a "${ORCH_LOG}") 2>&1
  echo "Benchmark orchestration started: $(date --iso-8601=seconds)"
  echo "Adapter: ${ADAPTER}"
  echo "Merged: ${MERGED}"
  echo "Run ID: ${RUN_ID}"

  "${PYTHON_BIN}" - "${ADAPTER}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve(strict=True)
state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
if state.get("global_step") != 6000 or state.get("max_steps") != 6000:
    raise SystemExit(f"checkpoint is incomplete: {state.get('global_step')}/{state.get('max_steps')}")
if not (checkpoint / "adapter_model.safetensors").is_file():
    raise SystemExit("checkpoint is missing adapter_model.safetensors")
print("Verified C4 terminal-fix RL checkpoint: 6000/6000")
PY

  if verify_merged "${MERGED}"; then
    echo "Merged model already exists and passed verification."
  else
    [[ ! -e "${MERGED}" ]] || {
      echo "Refusing invalid pre-existing merged path: ${MERGED}" >&2
      exit 3
    }
    echo "Merging C4 terminal-fix ${ALGORITHM^^} adapter."
    env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" export --adapters "${ADAPTER}" --merge_lora true --torch_dtype bfloat16 --safe_serialization true --max_shard_size 5GB --output_dir "${staging}"
    verify_merged "${staging}" || {
      echo "Merged model verification failed: ${staging}" >&2
      exit 3
    }
    mv "${staging}" "${MERGED}"
    echo "Merged model created: ${MERGED}"
  fi

  export PATH="$(dirname "${SWIFT_BIN}"):${PATH}"
  export PYTHONPATH="/root/autodl-tmp/ms-swift:${TOOLVISTA_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"

  echo "Starting C4 terminal-fix candidate on 8010."
  setsid env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" deploy --model "${MERGED}" --served_model_name "${API_MODEL}" --infer_backend vllm --vllm_max_model_len 32768 --vllm_max_num_seqs 16 --vllm_gpu_memory_utilization 0.55 --host 127.0.0.1 --port 8010 > "${CANDIDATE_LOG}" 2>&1 &
  candidate_pid="$!"
  wait_for_api_model candidate "${API_BASE}" "${API_MODEL}" "${candidate_pid}" || {
    tail -n 160 "${CANDIDATE_LOG}" || true
    exit 4
  }

  echo "Starting fixed base judge on 8011."
  setsid env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" deploy --model "${BASE_MODEL}" --served_model_name "${JUDGE_MODEL}" --infer_backend vllm --vllm_max_model_len 8192 --vllm_max_num_seqs 4 --vllm_gpu_memory_utilization 0.35 --host 127.0.0.1 --port 8011 > "${JUDGE_LOG}" 2>&1 &
  judge_pid="$!"
  wait_for_api_model judge "${JUDGE_API_BASE}" "${JUDGE_MODEL}" "${judge_pid}" || {
    tail -n 160 "${JUDGE_LOG}" || true
    exit 4
  }

  echo "Starting C4 terminal-fix datafixed-withnegative ${ALGORITHM} light benchmark."
  set +e
  env TOOLVISTA_ROOT="${TOOLVISTA_ROOT}" CHECKPOINT="${MERGED}" RUN_ID="${RUN_ID}" API_BASE="${API_BASE}" API_MODEL="${API_MODEL}" EVAL_URL="${API_BASE}" JUDGE_API_BASE="${JUDGE_API_BASE}" JUDGE_MODEL="${JUDGE_MODEL}" HISTORY_SUMMARY="${HISTORY_SUMMARY}" BASELINE_SUMMARY="${BASELINE_SUMMARY}" SUITE="${SUITE}" HISTORY_ROUTER_BENCHMARK_ID=mllm_router_datafixed_agentretention EXPERIMENT_VARIANT=datafixed-withnegative STAGE="${STAGE}" WANDB_GROUP="${WANDB_GROUP}" ALGORITHM="${ALGORITHM}" bash "${TOOLVISTA_ROOT}/scripts/autodl/17_run_c4_datafixed_light_eval.sh" 2>&1 | tee "${BENCHMARK_LOG}"
  local benchmark_rc="${PIPESTATUS[0]}"
  set -e
  echo "BENCHMARK_EXIT=${benchmark_rc}"
  echo "Summary: ${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json"
  cleanup
  trap - EXIT INT TERM
  return "${benchmark_rc}"
}

if [[ "${ORCHESTRATOR:-0}" == "1" ]]; then
  run_orchestrator
  exit "$?"
fi

command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
test -d "${TOOLVISTA_ROOT}"
test -d "${ADAPTER}"
test -d "${BASE_MODEL}"
test -x "${PYTHON_BIN}"
test -x "${SWIFT_BIN}"
require_file "${ADAPTER}/adapter_model.safetensors"
require_file "${ADAPTER}/trainer_state.json"
require_file "${HISTORY_SUMMARY}"
require_file "${BASELINE_SUMMARY}"
require_file "${TOOLVISTA_ROOT}/configs/evaluation/benchmarks_datafixed.yaml"
require_file "${TOOLVISTA_ROOT}/configs/evaluation/suites_datafixed.yaml"
require_file "${TOOLVISTA_ROOT}/scripts/autodl/17_run_c4_datafixed_light_eval.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${TOOLVISTA_ROOT}/runs/${RUN_ID}" ]]; then
  echo "Run ID already exists: ${RUN_ID}" >&2
  exit 2
fi
if pgrep -af '[s]wift deploy|[d]eploy.py|VLLM::EngineCore|[s]cripts/evaluate.py' >/dev/null; then
  echo "An inference or evaluation process is already running." >&2
  pgrep -af '[s]wift deploy|[d]eploy.py|VLLM::EngineCore|[s]cripts/evaluate.py' || true
  exit 2
fi
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ -n "${gpu_pids}" ]]; then
  echo "GPU is still occupied by process IDs: ${gpu_pids}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
printf -v run_command 'cd %q; env ORCHESTRATOR=1 TOOLVISTA_ROOT=%q PYTHON_BIN=%q SWIFT_BIN=%q BASE_MODEL=%q ADAPTER=%q MERGED=%q HISTORY_SUMMARY=%q BASELINE_SUMMARY=%q SESSION=%q RUN_ID=%q API_MODEL=%q API_BASE=%q JUDGE_API_BASE=%q JUDGE_MODEL=%q SUITE=%q ALGORITHM=%q STAGE=%q WANDB_GROUP=%q LOG_DIR=%q ORCH_LOG=%q CANDIDATE_LOG=%q JUDGE_LOG=%q BENCHMARK_LOG=%q bash %q; status=$?; echo "ORCHESTRATOR_EXIT=${status}"; exec bash' "${TOOLVISTA_ROOT}" "${TOOLVISTA_ROOT}" "${PYTHON_BIN}" "${SWIFT_BIN}" "${BASE_MODEL}" "${ADAPTER}" "${MERGED}" "${HISTORY_SUMMARY}" "${BASELINE_SUMMARY}" "${SESSION}" "${RUN_ID}" "${API_MODEL}" "${API_BASE}" "${JUDGE_API_BASE}" "${JUDGE_MODEL}" "${SUITE}" "${ALGORITHM}" "${STAGE}" "${WANDB_GROUP}" "${LOG_DIR}" "${ORCH_LOG}" "${CANDIDATE_LOG}" "${JUDGE_LOG}" "${BENCHMARK_LOG}" "${SCRIPT_PATH}"
printf -v monitor_command 'while true; do clear; %q %q/scripts/monitor_eval.py --run-id %q --suite %q --api-base %q --benchmark-config %q/configs/evaluation/benchmarks_datafixed.yaml --suite-config %q/configs/evaluation/suites_datafixed.yaml --candidate-log %q 2>/dev/null | awk '\''/^ToolVista eval snapshot/{show=1} show{print} /^\[6\/6\]/{exit}'\''; sleep 5; done' "${PYTHON_BIN}" "${TOOLVISTA_ROOT}" "${RUN_ID}" "${SUITE}" "${API_BASE}" "${TOOLVISTA_ROOT}" "${TOOLVISTA_ROOT}" "${CANDIDATE_LOG}"
printf -v logs_command 'tail --retry -n 80 -F %q %q %q %q; exec bash' "${ORCH_LOG}" "${CANDIDATE_LOG}" "${JUDGE_LOG}" "${BENCHMARK_LOG}"
printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

tmux new-session -d -s "${SESSION}" -n run "${run_command}"
tmux new-window -d -t "${SESSION}" -n monitor "${monitor_command}"
tmux new-window -d -t "${SESSION}" -n logs "${logs_command}"
tmux new-window -d -t "${SESSION}" -n gpu "${gpu_command}"
tmux select-window -t "${SESSION}:monitor"

echo "Benchmark tmux started: ${SESSION}"
echo "Attach: tmux attach-session -t ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Summary: ${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json"
