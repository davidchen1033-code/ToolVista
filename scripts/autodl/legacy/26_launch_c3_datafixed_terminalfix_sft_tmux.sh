#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
C2_CHECKPOINT="${C2_CHECKPOINT:-${TOOLVISTA_ROOT}/outputs/c2_toolvqa_replay_sft/v2-20260719-113121/checkpoint-1616}"
TRAIN_LAUNCHER="${TOOLVISTA_ROOT}/scripts/autodl/25_train_mllm_router_datafixed_terminalfix_sft.sh"
SESSION="${SESSION:-c3-datafixed-terminalfix-sft}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_STEPS="${MAX_STEPS:-1700}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention-terminalfix_sft/${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c3/c3-datafixed-agentretention-terminalfix-${RUN_TAG}-seed${SEED}.train.log}"
WANDB_NAME="${WANDB_NAME:-c3-datafixed-agentretention-terminalfix-${RUN_TAG}-seed${SEED}}"
ATTACH="${ATTACH:-true}"

command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
test -d "${TOOLVISTA_ROOT}"
test -d "${C2_CHECKPOINT}"
test -x "${TRAIN_LAUNCHER}"
test -s "${TOOLVISTA_ROOT}/data/reports/c3/router_datafixed-agentretention-terminalfix.report.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach-session -t ${SESSION}" >&2
  exit 2
fi
for path in "${OUTPUT_DIR}" "${TRAIN_LOG}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    exit 2
  fi
done

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ -n "${gpu_pids}" && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "GPU has active compute process IDs: ${gpu_pids}" >&2
  exit 2
fi

mkdir -p "$(dirname "${TRAIN_LOG}")"
train_env=(
  env
  "TOOLVISTA_ROOT=${TOOLVISTA_ROOT}"
  "C2_CHECKPOINT=${C2_CHECKPOINT}"
  "SEED=${SEED}"
  "MAX_STEPS=${MAX_STEPS}"
  "LEARNING_RATE=${LEARNING_RATE}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "WANDB_NAME=${WANDB_NAME}"
  "REPORT_TO=${REPORT_TO:-wandb}"
)
printf -v quoted_train '%q ' "${train_env[@]}" bash "${TRAIN_LAUNCHER}"
printf -v train_command 'cd %q; set -o pipefail; %s 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo "TRAIN_EXIT=${status}"; exec bash' "${TOOLVISTA_ROOT}" "${quoted_train}" "${TRAIN_LOG}"
printf -v log_command 'tail --retry -n 100 -F %q; exec bash' "${TRAIN_LOG}"
printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

echo "Session: ${SESSION}"
echo "C2 base: ${C2_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${TRAIN_LOG}"
echo "Steps: ${MAX_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"

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
