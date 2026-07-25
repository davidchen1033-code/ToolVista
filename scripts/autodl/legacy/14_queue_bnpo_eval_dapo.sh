#!/usr/bin/env bash
set -Eeuo pipefail

# Legacy queue for the already-running router-v1 RL A/B sequence. New C4 runs
# must use the router-v2 C3 checkpoint and its exact-checkpoint retention gate
# through 08-11_train_visual_rft_*.sh; do not reuse this queue for v2.

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
ACTION="${1:-run}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
C3_CHECKPOINT="${C3_CHECKPOINT:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_replay_sft/c3-final-seed42}"
DATASET="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl}"
C0_BASELINE="${C0_BASELINE:-${TOOLVISTA_ROOT}/runs/c0_base_light_fixed_20260717T120256Z_seed42/summary.json}"
C3_HISTORY="${C3_HISTORY:-${TOOLVISTA_ROOT}/runs/c3_router_light_20260719T135247Z_seed42/summary.json}"
SEED="${SEED:-42}"
CANDIDATE_PORT="${CANDIDATE_PORT:-8010}"
JUDGE_PORT="${JUDGE_PORT:-8011}"
CHECK_ONLY="${CHECK_ONLY:-false}"
RECOVER_RUN_ID="${RECOVER_RUN_ID:-}"

LOG_DIR="${TOOLVISTA_ROOT}/logs/c4"
QUEUE_STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE_LOG="${LOG_DIR}/c4-bnpo-eval-dapo-queue-${QUEUE_STARTED_AT}-seed${SEED}.log"
LOCK_FILE="${LOG_DIR}/c4-bnpo-eval-dapo-queue.lock"
QUEUE_PID_FILE="${LOG_DIR}/latest_bnpo_eval_dapo_queue.pid"
QUEUE_STATE_FILE="${LOG_DIR}/c4-bnpo-eval-dapo-queue-${QUEUE_STARTED_AT}-seed${SEED}.state"
CANDIDATE_PID=""
JUDGE_PID=""
MERGE_PID=""
BENCHMARK_PID=""
DAPO_PID=""
CANCEL_REQUESTED=false
CANCEL_SIGNAL=""

if [[ "${ACTION}" == "--cancel" ]]; then
  if [[ ! -s "${QUEUE_PID_FILE}" ]]; then
    echo "No active queue PID file was found: ${QUEUE_PID_FILE}" >&2
    exit 1
  fi
  queue_pid="$(<"${QUEUE_PID_FILE}")"
  if [[ ! "${queue_pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${queue_pid}" 2>/dev/null; then
    echo "The queue PID file is stale: ${queue_pid}" >&2
    exit 1
  fi
  queue_identity_ok=false
  queue_cmdline=""
  while IFS= read -r -d '' argument; do
    queue_cmdline+="${argument} "
    if [[ "${argument}" == "${TOOLVISTA_ROOT}/scripts/autodl/14_queue_bnpo_eval_dapo.sh" ||
          "${argument}" == "scripts/autodl/14_queue_bnpo_eval_dapo.sh" ]]; then
      queue_identity_ok=true
    fi
  done < "/proc/${queue_pid}/cmdline"
  if [[ "${queue_identity_ok}" != "true" ]]; then
    echo "Refusing to signal PID ${queue_pid}; no exact ToolVista queue script argument was found: ${queue_cmdline}" >&2
    exit 2
  fi
  echo "Requesting graceful cancellation of queue PID ${queue_pid}."
  kill -TERM "${queue_pid}"
  for _ in $(seq 1 360); do
    current_pid=""
    if [[ -s "${QUEUE_PID_FILE}" ]]; then
      current_pid="$(<"${QUEUE_PID_FILE}")"
    fi
    if [[ "${current_pid}" != "${queue_pid}" ]] || \
       ! kill -0 "${queue_pid}" 2>/dev/null; then
      echo "Queue PID ${queue_pid} exited."
      exit 0
    fi
    sleep 1
  done
  echo "Queue PID ${queue_pid} did not exit within 360 seconds; inspect the queue log before taking further action." >&2
  exit 3
elif [[ "${ACTION}" != "run" ]]; then
  echo "Usage: $0 [--cancel]" >&2
  exit 2
fi

BNPO_OUTPUT_ROOT="${BNPO_OUTPUT_ROOT:-$(<"${TOOLVISTA_ROOT}/logs/c4/latest_bnpo_output_dir.txt")}"
BNPO_TRAIN_LOG="${BNPO_TRAIN_LOG:-$(<"${TOOLVISTA_ROOT}/logs/c4/latest_bnpo_train_log.txt")}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another BNPO -> C4 light -> DAPO queue already holds ${LOCK_FILE}." >&2
  exit 2
fi

printf '%s\n' "${QUEUE_LOG}" > "${LOG_DIR}/latest_bnpo_eval_dapo_queue_log.txt"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

set_state() {
  local state="$1"
  printf '%s\n' "${state}" > "${QUEUE_STATE_FILE}"
  log "Queue state: ${state}."
}

process_group_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 1
  ps -eo pgid=,stat= \
    | awk -v pgid="${pid}" '$1 == pgid && $2 !~ /^Z/ { alive = 1 } END { exit(alive ? 0 : 1) }'
}

descendant_process_groups() {
  local root_pid="$1"
  { printf '%s\n' "${root_pid}"; ps -eo pid=,ppid=,pgid=,stat= \
    | awk -v root="${root_pid}" '
        {
          pid[NR] = $1
          ppid[NR] = $2
          pgid[NR] = $3
          stat[NR] = $4
        }
        END {
          included[root] = 1
          changed = 1
          while (changed) {
            changed = 0
            for (i = 1; i <= NR; i++) {
              if (included[ppid[i]] && !included[pid[i]]) {
                included[pid[i]] = 1
                changed = 1
              }
            }
          }
          for (i = 1; i <= NR; i++) {
            if (included[pid[i]] && stat[i] !~ /^Z/) {
              print pgid[i]
            }
          }
        }
      ' \
  ; } | sort -nu
}

process_groups_alive() {
  local pgid
  for pgid in "$@"; do
    if process_group_alive "${pgid}"; then
      return 0
    fi
  done
  return 1
}

stop_process_group() {
  local name="$1"
  local pid="$2"
  local attempt
  local pgid
  local -a pgids=()

  if [[ -z "${pid}" ]]; then
    return 0
  fi

  mapfile -t pgids < <(descendant_process_groups "${pid}")
  if [[ "${#pgids[@]}" -eq 0 ]]; then
    return 0
  fi

  log "Stopping ${name} process tree rooted at ${pid}; process groups: ${pgids[*]}."
  for pgid in "${pgids[@]}"; do
    kill -INT -- "-${pgid}" 2>/dev/null || true
  done
  for attempt in $(seq 1 60); do
    if ! process_groups_alive "${pgids[@]}"; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done

  for pgid in "${pgids[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
  for attempt in $(seq 1 30); do
    if ! process_groups_alive "${pgids[@]}"; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done

  log "WARNING: ${name} process tree did not exit after SIGINT/SIGTERM; sending SIGKILL."
  for pgid in "${pgids[@]}"; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done
  for attempt in $(seq 1 10); do
    if ! process_groups_alive "${pgids[@]}"; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done

  log "ERROR: ${name} process tree survived SIGKILL."
  return 1
}

cleanup_owned_processes() {
  local rc=0
  stop_process_group "DAPO training" "${DAPO_PID}" || rc=1
  stop_process_group "BNPO benchmark" "${BENCHMARK_PID}" || rc=1
  stop_process_group "BNPO merge" "${MERGE_PID}" || rc=1
  stop_process_group "fixed judge" "${JUDGE_PID}" || rc=1
  stop_process_group "BNPO candidate" "${CANDIDATE_PID}" || rc=1
  DAPO_PID=""
  BENCHMARK_PID=""
  MERGE_PID=""
  JUDGE_PID=""
  CANDIDATE_PID=""
  return "${rc}"
}

on_exit() {
  local rc=$?
  local cleanup_rc=0
  trap - EXIT HUP INT TERM
  cleanup_owned_processes || cleanup_rc=$?
  if [[ "${CANCEL_REQUESTED}" == "true" && "${cleanup_rc}" -eq 0 ]]; then
    set_state cancelled
    log "Queue cancelled by ${CANCEL_SIGNAL}; the independently started BNPO run was not stopped."
  elif [[ "${CANCEL_REQUESTED}" == "true" ]]; then
    set_state cancel-cleanup-failed
    log "ERROR: cancellation was requested by ${CANCEL_SIGNAL}, but one or more queue-owned process groups survived cleanup."
  elif (( rc == 0 && cleanup_rc == 0 )); then
    set_state completed
    log "Queue finished successfully."
  else
    set_state failed
    log "Queue stopped with exit code ${rc}; DAPO was not advanced past a failed prerequisite."
  fi
  if (( rc == 0 && cleanup_rc != 0 )); then
    rc=1
  fi
  if [[ -f "${QUEUE_PID_FILE}" ]] && [[ "$(<"${QUEUE_PID_FILE}")" == "$$" ]]; then
    rm -f "${QUEUE_PID_FILE}"
  fi
  exit "${rc}"
}

on_signal() {
  local signal="$1"
  local exit_code="$2"
  CANCEL_REQUESTED=true
  CANCEL_SIGNAL="${signal}"
  log "Cancellation requested by ${signal}."
  exit "${exit_code}"
}

trap on_exit EXIT
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || {
    log "ERROR: required non-empty file is missing: ${path}"
    exit 2
  }
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || {
    log "ERROR: required directory is missing: ${path}"
    exit 2
  }
}

bnpo_processes() {
  pgrep -af '[s]wift rlhf|[r]lhf.py' \
    | grep -F -- '--loss_type bnpo' \
    | grep -F -- "--output_dir ${BNPO_OUTPUT_ROOT}" \
    || true
}

latest_progress() {
  tr '\r' '\n' < "${BNPO_TRAIN_LOG}" 2>/dev/null \
    | grep -oE 'global_step/max_steps[^0-9]*[0-9]+/[0-9]+' \
    | tail -1 \
    | grep -oE '[0-9]+/[0-9]+' \
    || true
}

wait_for_no_gpu_processes() {
  local timeout_seconds="$1"
  local elapsed=0
  local processes

  while (( elapsed < timeout_seconds )); do
    processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "${processes//[[:space:]]/}" ]]; then
      return 0
    fi
    if (( elapsed % 30 == 0 )); then
      log "Waiting for GPU processes to exit: ${processes//$'\n'/; }."
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  log "ERROR: GPU is still occupied after ${timeout_seconds}s."
  nvidia-smi || true
  return 1
}

wait_for_api_model() {
  local name="$1"
  local base_url="$2"
  local model_id="$3"
  local pid="$4"
  local timeout_seconds="$5"
  local elapsed=0

  while (( elapsed < timeout_seconds )); do
    if ! process_group_alive "${pid}"; then
      log "ERROR: ${name} exited before becoming ready."
      return 1
    fi
    if curl -sf --max-time 5 "${base_url}/models" 2>/dev/null | grep -Fq "${model_id}"; then
      log "${name} is ready at ${base_url}; model=${model_id}."
      return 0
    fi
    if (( elapsed % 30 == 0 )); then
      log "Waiting for ${name} API (${elapsed}/${timeout_seconds}s)."
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  log "ERROR: timed out waiting for ${name} at ${base_url}."
  return 1
}

verify_summary() {
  local summary_path="$1"
  "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)

expected = {
    "xlam_function_calling",
    "toolvqa_step_light",
    "bfcl_non_live_light",
    "mllm_router",
    "qwen3vl_general_light",
    "visual_rft_coco_light",
}
benchmarks = summary.get("benchmarks")
if summary.get("status") != "passed" or not isinstance(benchmarks, dict):
    raise SystemExit(f"C4 light summary did not pass: status={summary.get('status')!r}")
if set(benchmarks) != expected:
    raise SystemExit(f"C4 light benchmark set mismatch: {sorted(benchmarks)}")
failed = {name: result.get("status") for name, result in benchmarks.items()
          if result.get("status") != "passed"}
if failed:
    raise SystemExit(f"C4 light contains failed benchmarks: {failed}")
print(f"Verified C4 light 6/6 passed: {path}")
PY
}

verify_merged_model() {
  local model_dir="$1"
  "${PYTHON_BIN}" - "${model_dir}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
config_path = root / "config.json"
if not index_path.is_file() or not config_path.is_file():
    raise SystemExit(f"Merged model is incomplete: {root}")
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in shards if not (root / name).is_file()]
if not shards or missing:
    raise SystemExit(f"Merged model shards are incomplete: shards={shards}, missing={missing}")
print(f"Verified merged model with {len(shards)} shard(s): {root}")
PY
}

require_dir "${TOOLVISTA_ROOT}"
require_dir "${MS_SWIFT_ROOT}"
require_dir "${BASE_MODEL}"
require_dir "${C3_CHECKPOINT}"
require_file "${C3_CHECKPOINT}/adapter_model.safetensors"
require_file "${DATASET}"
require_file "${C0_BASELINE}"
require_file "${C3_HISTORY}"
require_file "${BNPO_TRAIN_LOG}"
[[ -x "${PYTHON_BIN}" ]] || { log "ERROR: Python is not executable: ${PYTHON_BIN}"; exit 2; }
[[ -x "${SWIFT_BIN}" ]] || { log "ERROR: swift is not executable: ${SWIFT_BIN}"; exit 2; }
[[ "$(wc -l < "${DATASET}")" -eq 6000 ]] || { log "ERROR: expected 6000 RL rows."; exit 2; }
[[ "${BNPO_OUTPUT_ROOT}" == "${TOOLVISTA_ROOT}"/outputs/rl/bnpo_seed${SEED}_run* ]] || {
  log "ERROR: unexpected BNPO output root: ${BNPO_OUTPUT_ROOT}"
  exit 2
}

BNPO_RUN_TAG="${BNPO_OUTPUT_ROOT##*bnpo_seed${SEED}_run}"
[[ "${BNPO_RUN_TAG}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  log "ERROR: could not derive a stable BNPO run tag from ${BNPO_OUTPUT_ROOT}."
  exit 2
}

FINAL_ADAPTER="${TOOLVISTA_ROOT}/outputs/final/c4-c3-bnpo-visualrft-coco-base65-seed${SEED}-run${BNPO_RUN_TAG}-step6000-adapter"
FINAL_MERGED="${TOOLVISTA_ROOT}/outputs/final/c4-c3-bnpo-visualrft-coco-base65-seed${SEED}-run${BNPO_RUN_TAG}-step6000-merged"
CANDIDATE_MODEL_ID="c4-c3-bnpo-visualrft-coco-seed${SEED}"

if [[ -n "${RECOVER_RUN_ID}" ]]; then
  RUN_ID="${RECOVER_RUN_ID}"
  [[ "${RUN_ID}" == c4_visualrft_coco_bnpo_light_*_seed"${SEED}" ]] || {
    log "ERROR: unexpected BNPO recovery run ID: ${RUN_ID}"
    exit 4
  }
  RECOVERY_SUMMARY="${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json"
  require_file "${RECOVERY_SUMMARY}"
  "${PYTHON_BIN}" - "${RECOVERY_SUMMARY}" "${FINAL_MERGED}" <<'PY'
import json
import os
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected_checkpoint = os.path.realpath(sys.argv[2])
if summary.get("status") != "failed":
    raise SystemExit(f"Recovery requires a failed summary, got {summary.get('status')!r}")
if os.path.realpath(summary.get("checkpoint") or "") != expected_checkpoint:
    raise SystemExit(
        f"Recovery checkpoint mismatch: {summary.get('checkpoint')!r} != {expected_checkpoint!r}"
    )
benchmarks = summary.get("benchmarks") or {}
failed = {name for name, result in benchmarks.items() if result.get("status") != "passed"}
if failed != {"toolvqa_step_light"}:
    raise SystemExit(f"Expected only toolvqa_step_light to need recovery, got {sorted(failed)}")
print("Verified recovery target: five passed benchmarks plus failed toolvqa_step_light")
PY
  EVAL_RESUME=true
else
  EVAL_RESUME=false
fi

printf '%s\n' "$$" > "${QUEUE_PID_FILE}"
printf '%s\n' "${QUEUE_STATE_FILE}" > "${LOG_DIR}/latest_bnpo_eval_dapo_queue_state_file.txt"
log "Queue initialized."
log "BNPO output root: ${BNPO_OUTPUT_ROOT}"
log "BNPO train log: ${BNPO_TRAIN_LOG}"
log "BNPO final adapter: ${FINAL_ADAPTER}"
log "BNPO merged model: ${FINAL_MERGED}"

if [[ "${CHECK_ONLY}" == "true" ]]; then
  set_state check-only-passed
  log "CHECK_ONLY=true: preflight passed; no pipeline stage was started."
  exit 0
elif [[ "${CHECK_ONLY}" != "false" ]]; then
  log "ERROR: CHECK_ONLY must be true or false."
  exit 2
fi

if [[ -n "$(pgrep -af '[s]wift rlhf|[r]lhf.py' | grep -vF -- '--loss_type bnpo' || true)" ]]; then
  log "ERROR: an RL process other than the expected BNPO run is active."
  pgrep -af '[s]wift rlhf|[r]lhf.py' || true
  exit 2
fi

log "Waiting for the active BNPO run to finish."
set_state waiting-for-bnpo
bnpo_wait_ticks=0
while [[ -n "$(bnpo_processes)" ]]; do
  if (( bnpo_wait_ticks % 12 == 0 )); then
    log "BNPO progress: $(latest_progress | sed 's/^$/initializing/')."
  fi
  sleep 5
  bnpo_wait_ticks=$((bnpo_wait_ticks + 1))
done

set_state validating-bnpo
log "BNPO process exited; validating its final checkpoint."
mapfile -t bnpo_checkpoints < <(find "${BNPO_OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type d -name checkpoint-6000 -print)
if [[ "${#bnpo_checkpoints[@]}" -ne 1 ]]; then
  log "ERROR: expected exactly one checkpoint-6000 under ${BNPO_OUTPUT_ROOT}, found ${#bnpo_checkpoints[@]}."
  exit 3
fi
BNPO_CHECKPOINT="${bnpo_checkpoints[0]}"
require_file "${BNPO_CHECKPOINT}/adapter_model.safetensors"
require_file "${BNPO_CHECKPOINT}/adapter_config.json"
require_file "${BNPO_CHECKPOINT}/trainer_state.json"
require_file "${BNPO_CHECKPOINT}/args.json"
"${PYTHON_BIN}" - \
  "${BNPO_CHECKPOINT}/trainer_state.json" \
  "${BNPO_CHECKPOINT}/args.json" \
  "${C3_CHECKPOINT}" \
  "${BASE_MODEL}" <<'PY'
import json
import os
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
if state.get("global_step") != 6000 or state.get("max_steps") != 6000:
    raise SystemExit(f"Unexpected BNPO trainer state: global_step={state.get('global_step')}, max_steps={state.get('max_steps')}")

args = json.load(open(sys.argv[2], encoding="utf-8"))
c3_checkpoint = os.path.realpath(sys.argv[3])
base_model = os.path.realpath(sys.argv[4])
adapters = {os.path.realpath(path) for path in args.get("adapters") or []}
ref_adapters = {os.path.realpath(path) for path in args.get("ref_adapters") or []}
if args.get("loss_type") != "bnpo":
    raise SystemExit(f"Expected loss_type=bnpo, got {args.get('loss_type')!r}")
if os.path.realpath(args.get("model") or "") != base_model:
    raise SystemExit(f"BNPO base model mismatch: {args.get('model')!r}")
if adapters != {c3_checkpoint} or ref_adapters != {c3_checkpoint}:
    raise SystemExit(
        "BNPO lineage mismatch: expected both adapters and ref_adapters "
        f"to be the fixed C3 checkpoint {c3_checkpoint}; "
        f"got adapters={sorted(adapters)}, ref_adapters={sorted(ref_adapters)}"
    )
print("Verified BNPO trainer_state: 6000/6000")
print(f"Verified model lineage: Qwen3-VL base -> C3 adapter -> BNPO continuation ({c3_checkpoint})")
PY

if [[ -e "${FINAL_ADAPTER}" ]]; then
  require_file "${FINAL_ADAPTER}/adapter_model.safetensors"
  cmp -s "${BNPO_CHECKPOINT}/adapter_model.safetensors" "${FINAL_ADAPTER}/adapter_model.safetensors" || {
    log "ERROR: existing BNPO final adapter differs from checkpoint-6000: ${FINAL_ADAPTER}"
    exit 3
  }
  log "BNPO final adapter already exists and matches checkpoint-6000."
else
  adapter_staging="${FINAL_ADAPTER}.partial.$$"
  cp -a "${BNPO_CHECKPOINT}" "${adapter_staging}"
  sha256sum "${adapter_staging}/adapter_model.safetensors" > "${adapter_staging}/adapter_model.safetensors.sha256"
  mv "${adapter_staging}" "${FINAL_ADAPTER}"
  log "Archived BNPO adapter: ${FINAL_ADAPTER}"
fi
printf '%s\n' "${FINAL_ADAPTER}" > "${LOG_DIR}/latest_bnpo_final_adapter.txt"

wait_for_no_gpu_processes 300

if [[ -e "${FINAL_MERGED}" ]]; then
  verify_merged_model "${FINAL_MERGED}"
  log "BNPO merged model already exists and passed verification."
else
  merged_staging="${FINAL_MERGED}.partial.$$"
  log "Merging the complete C3-initialized, BNPO-updated adapter into the original Qwen3-VL base model."
  set_state merging-bnpo
  setsid env CUDA_VISIBLE_DEVICES=0 "${SWIFT_BIN}" export \
    --adapters "${FINAL_ADAPTER}" \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --safe_serialization true \
    --max_shard_size 5GB \
    --output_dir "${merged_staging}" &
  MERGE_PID=$!
  set +e
  wait "${MERGE_PID}"
  merge_rc=$?
  set -e
  MERGE_PID=""
  (( merge_rc == 0 )) || { log "ERROR: BNPO merge exited with code ${merge_rc}."; exit "${merge_rc}"; }
  verify_merged_model "${merged_staging}"
  mv "${merged_staging}" "${FINAL_MERGED}"
  log "Created BNPO merged model: ${FINAL_MERGED}"
fi
printf '%s\n' "${FINAL_MERGED}" > "${LOG_DIR}/latest_bnpo_final_merged.txt"

if pgrep -af '[s]wift deploy|[d]eploy.py|VLLM::EngineCore' >/dev/null; then
  log "ERROR: an inference service was already running before the BNPO benchmark."
  pgrep -af '[s]wift deploy|[d]eploy.py|VLLM::EngineCore' || true
  exit 4
fi
wait_for_no_gpu_processes 300

export PATH="$(dirname "${SWIFT_BIN}"):${PATH}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${TOOLVISTA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export CUDA_VISIBLE_DEVICES=0

CANDIDATE_LOG="${LOG_DIR}/c4-bnpo-candidate-${QUEUE_STARTED_AT}-seed${SEED}.log"
JUDGE_LOG="${LOG_DIR}/c4-bnpo-judge-${QUEUE_STARTED_AT}-seed${SEED}.log"

log "Starting merged BNPO candidate on port ${CANDIDATE_PORT}."
setsid "${SWIFT_BIN}" deploy \
  --model "${FINAL_MERGED}" \
  --served_model_name "${CANDIDATE_MODEL_ID}" \
  --infer_backend vllm \
  --vllm_max_model_len 32768 \
  --vllm_max_num_seqs 16 \
  --vllm_gpu_memory_utilization 0.55 \
  --host 127.0.0.1 \
  --port "${CANDIDATE_PORT}" \
  > "${CANDIDATE_LOG}" 2>&1 &
CANDIDATE_PID=$!
wait_for_api_model "BNPO candidate" "http://127.0.0.1:${CANDIDATE_PORT}/v1" "${CANDIDATE_MODEL_ID}" "${CANDIDATE_PID}" 600 || {
  tail -n 160 "${CANDIDATE_LOG}" || true
  exit 4
}

sleep 5
log "Starting the fixed base judge on port ${JUDGE_PORT}."
setsid "${SWIFT_BIN}" deploy \
  --model "${BASE_MODEL}" \
  --served_model_name Qwen3-VL-4B-Instruct \
  --infer_backend vllm \
  --vllm_max_model_len 8192 \
  --vllm_max_num_seqs 4 \
  --vllm_gpu_memory_utilization 0.35 \
  --host 127.0.0.1 \
  --port "${JUDGE_PORT}" \
  > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
wait_for_api_model "fixed base judge" "http://127.0.0.1:${JUDGE_PORT}/v1" "Qwen3-VL-4B-Instruct" "${JUDGE_PID}" 600 || {
  tail -n 160 "${JUDGE_LOG}" || true
  exit 4
}

if [[ -n "${RECOVER_RUN_ID}" ]]; then
  BENCHMARK_LOG="${LOG_DIR}/${RUN_ID}.resume-${QUEUE_STARTED_AT}.benchmark.log"
  log "Recovering BNPO C4 light run ${RUN_ID}; only failed results will rerun."
else
  RUN_ID="c4_visualrft_coco_bnpo_light_$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}"
  BENCHMARK_LOG="${LOG_DIR}/${RUN_ID}.benchmark.log"
fi
printf '%s\n' "${RUN_ID}" > "${LOG_DIR}/latest_c4_bnpo_benchmark_run_id.txt"
printf '%s\n' "${BENCHMARK_LOG}" > "${LOG_DIR}/latest_c4_bnpo_benchmark_log.txt"

log "Starting BNPO C4 light benchmark: ${RUN_ID}."
set_state benchmarking-bnpo
CHECKPOINT="${FINAL_MERGED}" \
RUN_ID="${RUN_ID}" \
STAGE=c4-visualrft-coco-bnpo \
ALGORITHM=bnpo \
SUITE=c4_light \
API_BASE="http://127.0.0.1:${CANDIDATE_PORT}/v1" \
API_MODEL="${CANDIDATE_MODEL_ID}" \
EVAL_URL="http://127.0.0.1:${CANDIDATE_PORT}/v1" \
JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1" \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
BASELINE_SUMMARY="${C0_BASELINE}" \
HISTORY_SUMMARY="${C3_HISTORY}" \
WANDB_PROJECT=toolvista-qwen3vl-eval \
WANDB_GROUP=c4-visualrft-rl-ab \
RESUME="${EVAL_RESUME}" \
TOOLVISTA_ROOT="${TOOLVISTA_ROOT}" \
BENCHMARK_LOG="${BENCHMARK_LOG}" \
setsid bash -c 'bash "${TOOLVISTA_ROOT}/scripts/autodl/12_run_light_eval.sh" 2>&1 | tee "${BENCHMARK_LOG}"; exit "${PIPESTATUS[0]}"' &
BENCHMARK_PID=$!
set +e
wait "${BENCHMARK_PID}"
benchmark_rc=$?
set -e
BENCHMARK_PID=""
if (( benchmark_rc != 0 )); then
  log "ERROR: BNPO C4 light exited with code ${benchmark_rc}."
  exit "${benchmark_rc}"
fi

BNPO_SUMMARY="${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json"
require_file "${BNPO_SUMMARY}"
verify_summary "${BNPO_SUMMARY}"
printf '%s\n' "${BNPO_SUMMARY}" > "${LOG_DIR}/latest_c4_bnpo_summary.txt"

log "BNPO benchmark passed; stopping candidate and judge before DAPO."
cleanup_owned_processes
wait_for_no_gpu_processes 300

if pgrep -af '[s]wift rlhf|[r]lhf.py|[s]wift deploy|[d]eploy.py|VLLM::EngineCore|[s]cripts/evaluate.py' >/dev/null; then
  log "ERROR: a GPU/evaluation process remains before DAPO."
  pgrep -af '[s]wift rlhf|[r]lhf.py|[s]wift deploy|[d]eploy.py|VLLM::EngineCore|[s]cripts/evaluate.py' || true
  exit 5
fi

DAPO_STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
DAPO_OUTPUT_DIR="${TOOLVISTA_ROOT}/outputs/rl/dapo_seed${SEED}_run${DAPO_STARTED_AT}"
DAPO_LOG="${LOG_DIR}/c4-dapo-${DAPO_STARTED_AT}-seed${SEED}.train.log"
DAPO_WANDB_NAME="c4-visualrft-coco-dapo-${DAPO_STARTED_AT}-seed${SEED}"
[[ ! -e "${DAPO_OUTPUT_DIR}" ]] || { log "ERROR: DAPO output already exists: ${DAPO_OUTPUT_DIR}"; exit 5; }

printf '%s\n' "${DAPO_OUTPUT_DIR}" > "${LOG_DIR}/latest_dapo_output_dir.txt"
printf '%s\n' "${DAPO_LOG}" > "${LOG_DIR}/latest_dapo_train_log.txt"
printf '%s\n' "${DAPO_WANDB_NAME}" > "${LOG_DIR}/latest_dapo_wandb_name.txt"

log "Starting DAPO from the shared C3 checkpoint."
set_state training-dapo
TOOLVISTA_ROOT="${TOOLVISTA_ROOT}" \
MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
C3_CHECKPOINT="${C3_CHECKPOINT}" \
DATASET="${DATASET}" \
PYTHON_BIN="${PYTHON_BIN}" \
SWIFT_BIN="${SWIFT_BIN}" \
SEED="${SEED}" \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
EPOCHS=1 \
MAX_STEPS=-1 \
LEARNING_RATE=1e-6 \
GRAD_ACC=4 \
STEPS_PER_GENERATION=4 \
NUM_GENERATIONS=4 \
MAX_LENGTH=4096 \
VLLM_MAX_MODEL_LEN=4096 \
MAX_COMPLETION_LENGTH=512 \
MAX_PIXELS=1003520 \
TEMPERATURE=1.0 \
TOP_P=0.9 \
BETA=0.04 \
VLLM_GPU_MEMORY_UTILIZATION=0.4 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
SLEEP_LEVEL=0 \
OFFLOAD_MODEL=true \
OFFLOAD_OPTIMIZER=true \
LORA_RANK=8 \
LORA_ALPHA=32 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=1 \
LOG_COMPLETIONS=false \
DATALOADER_NUM_WORKERS=2 \
VALIDATE_ROWS=100 \
REWARD_VALIDATE_ROWS=0 \
REPORT_TO=wandb \
WANDB_PROJECT=toolvista-qwen3vl \
WANDB_GROUP=rl-ab \
WANDB_NAME="${DAPO_WANDB_NAME}" \
STAGE=rl-dapo \
OUTPUT_DIR="${DAPO_OUTPUT_DIR}" \
DAPO_LOG="${DAPO_LOG}" \
setsid bash -c 'bash "${TOOLVISTA_ROOT}/scripts/autodl/10_train_visual_rft_dapo.sh" 2>&1 | tee "${DAPO_LOG}"; exit "${PIPESTATUS[0]}"' &
DAPO_PID=$!
set +e
wait "${DAPO_PID}"
dapo_rc=$?
set -e
DAPO_PID=""

log "C4 DAPO formal training exit code: ${dapo_rc}."
exit "${dapo_rc}"
