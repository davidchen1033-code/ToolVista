#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C3_DATAFIXED_CHECKPOINT="${C3_DATAFIXED_CHECKPOINT:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed_sft/v1-20260721-203434/checkpoint-1063}"
C3_DATAFIXED_RETENTION_GATE="${C3_DATAFIXED_RETENTION_GATE:-${C3_DATAFIXED_CHECKPOINT}/toolvista_c3_datafixed_retention_gate.json}"
EXPECTED_C3_VARIANT="${EXPECTED_C3_VARIANT:-datafixed}"

SESSION="${SESSION:-c4fixed-withnegative-dapo}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_STEPS="${MAX_STEPS:-6000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.04}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-13}"
REPORT_TO="${REPORT_TO:-wandb}"
ATTACH="${ATTACH:-true}"

OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed-withnegative_seed${SEED}_run${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c4/c4-dapo-datafixed-withnegative-${RUN_TAG}-seed${SEED}.train.log}"
PROVENANCE_MANIFEST="${PROVENANCE_MANIFEST:-${TRAIN_LOG%.train.log}.manifest.json}"
TRAIN_LAUNCHER="${TOOLVISTA_ROOT}/scripts/autodl/16_train_visual_rft_dapo_datafixed_withnegative.sh"

command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
test -x "${PYTHON_BIN}"
test -d "${C3_DATAFIXED_CHECKPOINT}"
test -f "${TRAIN_LAUNCHER}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach-session -t ${SESSION}" >&2
  exit 2
fi

for path in "${OUTPUT_DIR}" "${TRAIN_LOG}" "${PROVENANCE_MANIFEST}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    exit 2
  fi
done

gpu_pids="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]'
)"
if [[ -n "${gpu_pids}" && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "GPU has active compute process IDs: ${gpu_pids}" >&2
  echo "Set ALLOW_BUSY_GPU=1 only after confirming resource sharing is safe." >&2
  exit 2
fi

printf -v train_command \
  'cd %q; env TOOLVISTA_ROOT=%q MS_SWIFT_ROOT=%q BASE_MODEL=%q PYTHON_BIN=%q C3_DATAFIXED_CHECKPOINT=%q C3_DATAFIXED_RETENTION_GATE=%q EXPECTED_C3_VARIANT=%q SEED=%q RUN_TAG=%q MAX_STEPS=%q LEARNING_RATE=%q BETA=%q SAVE_STEPS=%q SAVE_TOTAL_LIMIT=%q REPORT_TO=%q OUTPUT_DIR=%q TRAIN_LOG=%q PROVENANCE_MANIFEST=%q bash %q; status=$?; echo "TRAIN_EXIT=${status}"; exec bash' \
  "${TOOLVISTA_ROOT}" \
  "${TOOLVISTA_ROOT}" \
  "${MS_SWIFT_ROOT}" \
  "${BASE_MODEL}" \
  "${PYTHON_BIN}" \
  "${C3_DATAFIXED_CHECKPOINT}" \
  "${C3_DATAFIXED_RETENTION_GATE}" \
  "${EXPECTED_C3_VARIANT}" \
  "${SEED}" \
  "${RUN_TAG}" \
  "${MAX_STEPS}" \
  "${LEARNING_RATE}" \
  "${BETA}" \
  "${SAVE_STEPS}" \
  "${SAVE_TOTAL_LIMIT}" \
  "${REPORT_TO}" \
  "${OUTPUT_DIR}" \
  "${TRAIN_LOG}" \
  "${PROVENANCE_MANIFEST}" \
  "${TRAIN_LAUNCHER}"

printf -v log_command \
  'tail --retry -n 100 -F %q; exec bash' \
  "${TRAIN_LOG}"
printf -v gpu_command \
  'watch -n 5 nvidia-smi; exec bash'

echo "Session: ${SESSION}"
echo "Run tag: ${RUN_TAG}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${TRAIN_LOG}"
echo "Provenance: ${PROVENANCE_MANIFEST}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=OK"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n train "${train_command}"
tmux new-window -d -t "${SESSION}" -n log "${log_command}"
tmux new-window -d -t "${SESSION}" -n gpu "${gpu_command}"
tmux select-window -t "${SESSION}:train"

case "${ATTACH}" in
  true) exec tmux attach-session -t "${SESSION}" ;;
  false) echo "Attach with: tmux attach-session -t ${SESSION}" ;;
  *) echo "ATTACH must be true or false" >&2; exit 2 ;;
esac
