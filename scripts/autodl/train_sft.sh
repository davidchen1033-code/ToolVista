#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
source "${TOOLVISTA_ROOT}/scripts/autodl/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: train_sft.sh --stage toolvqa|c1|c2|c3 --variant VARIANT [mode]

Modes:
  --tmux        Launch in tmux (default).
  --foreground  Run in the current shell.
  --attach      Enter the tmux session after launch (default).
  --detach      Launch tmux and return to the shell.

Variants:
  toolvqa/c1/c2: raw
  c3: raw | datafixed | datafixed-withnegative | datafixed-withnegative-terminal
EOF
}

pipeline_stage="${PIPELINE_STAGE:-}"
variant_input="${EXPERIMENT_VARIANT:-}"
mode="${MODE:-tmux}"
attach="${ATTACH:-true}"

while (( $# )); do
  case "$1" in
    --stage) pipeline_stage="${2:?missing value for --stage}"; shift 2 ;;
    --variant) variant_input="${2:?missing value for --variant}"; shift 2 ;;
    --tmux) mode=tmux; shift ;;
    --foreground|--worker) mode=foreground; shift ;;
    --attach) attach=true; shift ;;
    --detach) attach=false; shift ;;
    --help|-h) usage; exit 0 ;;
    *) autodl_die "unknown argument: $1" ;;
  esac
done

[[ -n "${pipeline_stage}" ]] || autodl_die "--stage is required"
[[ -n "${variant_input}" ]] || autodl_die "--variant is required"
variant="$(autodl_normalize_variant "${variant_input}")"
autodl_require_bool ATTACH "${attach}"
case "${mode}" in tmux|foreground) ;; *) autodl_die "MODE must be tmux or foreground: ${mode}" ;; esac

seed="${SEED:-42}"
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
case "${pipeline_stage}:${variant}" in
  toolvqa:raw)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_toolvqa_raw.sh"
    default_steps=-1
    default_lr=1e-4
    ;;
  c1:raw)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c1_raw.sh"
    default_steps=-1
    default_lr=1e-4
    ;;
  c2:raw)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c2_raw.sh"
    default_steps=-1
    default_lr=1e-4
    : "${C1_CHECKPOINT:=/root/autodl-tmp/ToolVista/outputs/c1_xlam_sft/v0-20260718-112823/checkpoint-11429}"
    export C1_CHECKPOINT
    ;;
  c3:raw)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c3_raw.sh"
    default_steps=3414
    default_lr=2e-5
    ;;
  c3:datafixed)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c3_datafixed.sh"
    default_steps=1063
    default_lr=2e-5
    ;;
  c3:datafixed-withnegative)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c3_datafixed_withnegative.sh"
    default_steps=1215
    default_lr=2e-5
    ;;
  c3:datafixed-withnegative-terminal)
    worker="${TOOLVISTA_ROOT}/scripts/autodl/lib/train_c3_datafixed_withnegative_terminal.sh"
    default_steps=1700
    default_lr=3e-5
    ;;
  *) autodl_die "unsupported SFT combination: stage=${pipeline_stage}, variant=${variant}" ;;
esac

if [[ "${pipeline_stage}" == c3 ]]; then
  : "${C2_CHECKPOINT:=/root/autodl-tmp/ToolVista/outputs/c2_toolvqa_replay_sft/v2-20260719-113121/checkpoint-1616}"
  export C2_CHECKPOINT
fi

max_steps="${MAX_STEPS:-${default_steps}}"
learning_rate="${LEARNING_RATE:-${default_lr}}"
stage_name="${RUN_STAGE:-${pipeline_stage}-${variant}-sft}"
session="${SESSION:-${pipeline_stage}-${variant}-sft}"
output_dir="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/${pipeline_stage}_${variant}_sft/${run_tag}}"
train_log="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/${pipeline_stage}/${pipeline_stage}-${variant}-${run_tag}-seed${seed}.train.log}"
wandb_group="${WANDB_GROUP:-${pipeline_stage}-${variant}-sft}"
wandb_name="${WANDB_NAME:-${pipeline_stage}-${variant}-${run_tag}-seed${seed}}"

autodl_require_dir TOOLVISTA_ROOT "${TOOLVISTA_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || autodl_die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
autodl_require_file SFT_WORKER "${worker}"

export TOOLVISTA_ROOT PYTHON_BIN
export PIPELINE_STAGE="${pipeline_stage}" EXPERIMENT_VARIANT="${variant}"
export SEED="${seed}" MAX_STEPS="${max_steps}" LEARNING_RATE="${learning_rate}"
export STAGE="${stage_name}" OUTPUT_DIR="${output_dir}"
export WANDB_GROUP="${wandb_group}" WANDB_NAME="${wandb_name}"

if [[ "${mode}" == foreground ]]; then
  exec bash "${worker}"
fi

case "${DRY_RUN:-0}" in
  1|true)
    echo "DRY_RUN=OK"
    echo "STAGE=${pipeline_stage}"
    echo "VARIANT=${variant}"
    echo "WORKER=${worker}"
    echo "SESSION=${session}"
    echo "OUTPUT_DIR=${output_dir}"
    echo "MAX_STEPS=${max_steps}"
    exit 0
    ;;
  0|false) ;;
  *) autodl_die "DRY_RUN must be 0, 1, false, or true" ;;
esac

command -v tmux >/dev/null || autodl_die "tmux is not installed"
command -v nvidia-smi >/dev/null || autodl_die "nvidia-smi is not installed"
tmux has-session -t "${session}" 2>/dev/null && autodl_die "tmux session already exists: ${session}"
[[ ! -e "${output_dir}" ]] || autodl_die "refusing to overwrite OUTPUT_DIR: ${output_dir}"
[[ ! -e "${train_log}" ]] || autodl_die "refusing to overwrite TRAIN_LOG: ${train_log}"

gpu_pids="$(autodl_gpu_pids)"
if [[ -n "${gpu_pids}" && "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
  autodl_die "GPU has active compute process IDs: ${gpu_pids}"
fi

mkdir -p "$(dirname "${output_dir}")" "$(dirname "${train_log}")"
printf -v worker_command \
  'cd %q; env TOOLVISTA_ROOT=%q PYTHON_BIN=%q PIPELINE_STAGE=%q EXPERIMENT_VARIANT=%q C1_CHECKPOINT=%q C2_CHECKPOINT=%q SEED=%q MAX_STEPS=%q LEARNING_RATE=%q STAGE=%q OUTPUT_DIR=%q WANDB_GROUP=%q WANDB_NAME=%q REPORT_TO=%q bash %q --stage %q --variant %q --foreground 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo "TRAIN_EXIT=${status}"; exec bash' \
  "${TOOLVISTA_ROOT}" "${TOOLVISTA_ROOT}" "${PYTHON_BIN}" "${pipeline_stage}" "${variant}" \
  "${C1_CHECKPOINT:-}" "${C2_CHECKPOINT:-}" "${seed}" "${max_steps}" "${learning_rate}" \
  "${stage_name}" "${output_dir}" "${wandb_group}" "${wandb_name}" "${REPORT_TO:-wandb}" \
  "${BASH_SOURCE[0]}" "${pipeline_stage}" "${variant}" "${train_log}"
printf -v log_command 'tail --retry -n 100 -F %q; exec bash' "${train_log}"
printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

tmux new-session -d -s "${session}" -n train "${worker_command}"
tmux new-window -d -t "${session}" -n log "${log_command}"
tmux new-window -d -t "${session}" -n gpu "${gpu_command}"
tmux select-window -t "${session}:train"

echo "SESSION=${session}"
echo "VARIANT=${variant}"
echo "OUTPUT_DIR=${output_dir}"
echo "TRAIN_LOG=${train_log}"
echo "Attach: tmux attach-session -t ${session}"

if [[ "${attach}" == true ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "${session}"
  fi
  exec tmux attach-session -t "${session}"
fi
