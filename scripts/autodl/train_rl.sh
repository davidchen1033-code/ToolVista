#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
source "${TOOLVISTA_ROOT}/scripts/autodl/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: train_rl.sh --variant VARIANT --algorithm ALGORITHM [mode]

Variants:
  raw | datafixed | datafixed-withnegative | datafixed-withnegative-terminal

Algorithms:
  grpo | bnpo | dapo | drgrpo

Modes:
  --tmux        Launch in tmux (default).
  --foreground  Run in the current shell.
  --attach      Enter the tmux session after launch (default).
  --detach      Launch tmux and return to the shell.
EOF
}

variant_input="${EXPERIMENT_VARIANT:-}"
algorithm_input="${ALGORITHM:-}"
mode="${MODE:-tmux}"
attach="${ATTACH:-true}"

while (( $# )); do
  case "$1" in
    --variant) variant_input="${2:?missing value for --variant}"; shift 2 ;;
    --algorithm) algorithm_input="${2:?missing value for --algorithm}"; shift 2 ;;
    --tmux) mode=tmux; shift ;;
    --foreground|--worker) mode=foreground; shift ;;
    --attach) attach=true; shift ;;
    --detach) attach=false; shift ;;
    --help|-h) usage; exit 0 ;;
    *) autodl_die "unknown argument: $1" ;;
  esac
done

[[ -n "${variant_input}" ]] || autodl_die "--variant is required"
[[ -n "${algorithm_input}" ]] || autodl_die "--algorithm is required"
variant="$(autodl_normalize_variant "${variant_input}")"
algorithm="$(autodl_normalize_algorithm "${algorithm_input}")"
internal_c3_variant="$(autodl_internal_c3_variant "${variant}")"
autodl_require_bool ATTACH "${attach}"
case "${mode}" in tmux|foreground) ;; *) autodl_die "MODE must be tmux or foreground: ${mode}" ;; esac

seed="${SEED:-42}"
run_tag="${RUN_TAG:-${variant}-${algorithm}-$(date -u +%Y%m%dT%H%M%SZ)}"
max_steps="${MAX_STEPS:-6000}"
save_steps="${SAVE_STEPS:-500}"
save_total_limit="${SAVE_TOTAL_LIMIT:-13}"

case "${algorithm}" in
  grpo) loss_type=grpo ;;
  bnpo) loss_type=bnpo ;;
  dapo) loss_type=dapo ;;
  drgrpo) loss_type=dr_grpo ;;
esac

case "${variant}" in
  raw)
    default_checkpoint="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_replay_sft/v0-20260719-151204/checkpoint-3414"
    default_gate=""
    expected_gate_variant=""
    require_gate=false
    default_dataset="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl"
    default_report=""
    default_media=""
    default_reward_plugin="${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft.py"
    reward_funcs="visual_rft format"
    ;;
  datafixed)
    default_checkpoint="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed_sft/v1-20260721-203434/checkpoint-1063"
    default_gate="${default_checkpoint}/toolvista_c3_datafixed_retention_gate.json"
    expected_gate_variant=datafixed
    require_gate=true
    default_dataset="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl"
    default_report=""
    default_media=""
    default_reward_plugin="${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft.py"
    reward_funcs="visual_rft format"
    ;;
  datafixed-withnegative)
    default_checkpoint="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention_sft/20260723T052946Z/v0-20260723-133020/checkpoint-1215"
    default_gate="${default_checkpoint}/toolvista_c3_datafixed_agentretention_retention_gate.json"
    expected_gate_variant=datafixed-agentretention
    require_gate=true
    default_dataset="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl"
    default_report="${TOOLVISTA_ROOT}/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json"
    default_media="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_media"
    default_reward_plugin="${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft_withnegative.py"
    reward_funcs="visual_rft_withnegative visual_rft_withnegative_format"
    ;;
  datafixed-withnegative-terminal)
    default_checkpoint="${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention-terminalfix_sft/terminalfix20260723T100256Z/v0-20260723-180402/checkpoint-1700"
    default_gate="${default_checkpoint}/toolvista_c3_datafixed_agentretention_terminalfix_retention_gate.json"
    expected_gate_variant=datafixed-agentretention-terminalfix
    require_gate=true
    default_dataset="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl"
    default_report="${TOOLVISTA_ROOT}/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json"
    default_media="${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_media"
    default_reward_plugin="${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft_withnegative.py"
    reward_funcs="visual_rft_withnegative visual_rft_withnegative_format"
    ;;
esac

checkpoint="${C3_CHECKPOINT:-${C3_DATAFIXED_CHECKPOINT:-${default_checkpoint}}}"
gate="${C3_RETENTION_GATE:-${C3_DATAFIXED_RETENTION_GATE:-${default_gate}}}"
dataset="${DATASET:-${default_dataset}}"
dataset_report="${DATASET_REPORT:-${default_report}}"
media_dir="${MEDIA_DIR:-${default_media}}"
reward_plugin="${REWARD_PLUGIN:-${default_reward_plugin}}"
session="${SESSION:-c4-${variant}-${algorithm}}"
stage_name="${RUN_STAGE:-rl-${algorithm}-${variant}}"
wandb_group="${WANDB_GROUP:-rl-${algorithm}-${variant}}"
wandb_name="${WANDB_NAME:-c4-visualrft-${algorithm}-${variant}-${run_tag}-seed${seed}}"
output_dir="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/${algorithm}_${variant}_seed${seed}_run${run_tag}}"
train_log="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c4/c4-${algorithm}-${variant}-${run_tag}-seed${seed}.train.log}"
provenance_manifest="${PROVENANCE_MANIFEST:-${train_log%.train.log}.manifest.json}"

verify_contract() {
  [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || autodl_die "MAX_STEPS must be a positive integer: ${max_steps}"
  [[ "${save_steps}" =~ ^[1-9][0-9]*$ ]] || autodl_die "SAVE_STEPS must be a positive integer: ${save_steps}"
  [[ "${save_total_limit}" =~ ^[1-9][0-9]*$ ]] || autodl_die "SAVE_TOTAL_LIMIT must be a positive integer: ${save_total_limit}"
  if [[ "${variant}" == datafixed-withnegative-terminal && "${max_steps}" != 6000 ]]; then
    autodl_die "the strict terminal comparison requires MAX_STEPS=6000"
  fi
  [[ "${output_dir}" == *"${variant}"* ]] || autodl_die "OUTPUT_DIR must contain the canonical variant '${variant}': ${output_dir}"
  [[ "${train_log}" == *"${variant}"* ]] || autodl_die "TRAIN_LOG must contain the canonical variant '${variant}': ${train_log}"
  autodl_require_dir TOOLVISTA_ROOT "${TOOLVISTA_ROOT}"
  autodl_require_dir MS_SWIFT_ROOT "${MS_SWIFT_ROOT}"
  autodl_require_dir BASE_MODEL "${BASE_MODEL}"
  autodl_require_dir C3_CHECKPOINT "${checkpoint}"
  autodl_require_file C3_ADAPTER "${checkpoint}/adapter_model.safetensors"
  autodl_require_file DATASET "${dataset}"
  autodl_require_file REWARD_PLUGIN "${reward_plugin}"
  if [[ "${require_gate}" == true ]]; then
    autodl_require_file C3_RETENTION_GATE "${gate}"
  fi
  if [[ "${variant}" == datafixed-withnegative || "${variant}" == datafixed-withnegative-terminal ]]; then
    autodl_require_file DATASET_REPORT "${dataset_report}"
    autodl_require_dir MEDIA_DIR "${media_dir}"
    autodl_require_file MEDIA_MANIFEST "${media_dir}/manifest.jsonl"
  fi
}

write_manifest() {
  "${PYTHON_BIN}" - "${provenance_manifest}" "${variant}" "${internal_c3_variant}" "${algorithm}" "${loss_type}" "${checkpoint}" "${gate}" "${dataset}" "${dataset_report}" "${media_dir}" "${reward_plugin}" "${output_dir}" "${train_log}" "${run_tag}" "${seed}" "${max_steps}" <<'PY'
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

(
    destination, variant, c3_variant, algorithm, loss_type, checkpoint, gate,
    dataset, report, media_dir, reward_plugin, output_dir, train_log, run_tag,
    seed, max_steps,
) = sys.argv[1:]


def file_info(value: str) -> dict[str, object] | None:
    if not value:
        return None
    path = Path(value).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


files = {
    "dataset": file_info(dataset),
    "dataset_report": file_info(report),
    "reward_plugin": file_info(reward_plugin),
    "retention_gate": file_info(gate),
}
if media_dir:
    files["canonical_media_manifest"] = file_info(str(Path(media_dir) / "manifest.jsonl"))
payload = {
    "schema": "toolvista.rl.provenance.v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "variant": variant,
    "c3_variant": c3_variant,
    "algorithm": algorithm,
    "loss_type": loss_type,
    "run_tag": run_tag,
    "seed": int(seed),
    "max_steps": int(max_steps),
    "checkpoint": str(Path(checkpoint).resolve(strict=True)),
    "output_dir": str(Path(output_dir).resolve(strict=False)),
    "train_log": str(Path(train_log).resolve(strict=False)),
    "files": {name: value for name, value in files.items() if value is not None},
}
path = Path(destination)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"Provenance manifest: {path}")
PY
}

run_foreground() {
  verify_contract
  [[ ! -e "${output_dir}" ]] || autodl_die "refusing to overwrite OUTPUT_DIR: ${output_dir}"
  [[ ! -e "${provenance_manifest}" ]] || autodl_die "refusing to overwrite provenance: ${provenance_manifest}"

  if [[ "${require_gate}" == true ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; actual=json.load(open(sys.argv[1], encoding="utf-8")).get("variant"); expected=sys.argv[2]; actual == expected or (_ for _ in ()).throw(SystemExit(f"C3 gate variant mismatch: {actual!r} != {expected!r}"))' "${gate}" "${expected_gate_variant}"
  fi
  if [[ "${variant}" == datafixed-withnegative || "${variant}" == datafixed-withnegative-terminal ]]; then
    "${PYTHON_BIN}" "${TOOLVISTA_ROOT}/scripts/prepare_visual_rft_datafixed_withnegative.py" --output "${dataset}" --report "${dataset_report}" --media-dir "${media_dir}" --verify-only
  fi

  mkdir -p "$(dirname "${output_dir}")" "$(dirname "${train_log}")"
  write_manifest

  export TOOLVISTA_ROOT MS_SWIFT_ROOT BASE_MODEL PYTHON_BIN
  export C3_CHECKPOINT="${checkpoint}" C3_RETENTION_GATE="${gate}" REQUIRE_C3_RETENTION_GATE="${require_gate}"
  export DATASET="${dataset}" REWARD_PLUGIN="${reward_plugin}" REWARD_FUNCS="${reward_funcs}"
  export SEED="${seed}" MAX_STEPS="${max_steps}" SAVE_STEPS="${save_steps}" SAVE_TOTAL_LIMIT="${save_total_limit}"
  export OUTPUT_DIR="${output_dir}" STAGE="${stage_name}" WANDB_GROUP="${wandb_group}" WANDB_NAME="${wandb_name}"
  export LOSS_TYPE="${loss_type}"

  case "${algorithm}" in
    grpo|bnpo) ALGORITHM_ARGS=() ;;
    dapo) ALGORITHM_ARGS=(--epsilon_high 0.28 --dynamic_sample true --max_resample_times 3 --overlong_filter true) ;;
    drgrpo) ALGORITHM_ARGS=(--scale_rewards none) ;;
  esac
  source "${TOOLVISTA_ROOT}/scripts/autodl/lib/rl_common.sh"
}

if [[ "${mode}" == foreground ]]; then
  run_foreground
  exit "$?"
fi

verify_contract
case "${DRY_RUN:-0}" in
  1|true)
    echo "DRY_RUN=OK"
    echo "VARIANT=${variant}"
    echo "C3_VARIANT=${internal_c3_variant}"
    echo "ALGORITHM=${algorithm}"
    echo "LOSS_TYPE=${loss_type}"
    echo "C3_CHECKPOINT=${checkpoint}"
    echo "DATASET=${dataset}"
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
gpu_pids="$(autodl_gpu_pids)"
[[ -z "${gpu_pids}" ]] || autodl_die "GPU has active compute process IDs: ${gpu_pids}"
for path in "${output_dir}" "${train_log}" "${provenance_manifest}"; do
  [[ ! -e "${path}" ]] || autodl_die "refusing to overwrite existing path: ${path}"
done
mkdir -p "$(dirname "${output_dir}")" "$(dirname "${train_log}")"

printf -v worker_command \
  'cd %q; env TOOLVISTA_ROOT=%q MS_SWIFT_ROOT=%q BASE_MODEL=%q PYTHON_BIN=%q EXPERIMENT_VARIANT=%q ALGORITHM=%q RUN_TAG=%q SEED=%q MAX_STEPS=%q SAVE_STEPS=%q SAVE_TOTAL_LIMIT=%q C3_CHECKPOINT=%q C3_RETENTION_GATE=%q DATASET=%q DATASET_REPORT=%q MEDIA_DIR=%q REWARD_PLUGIN=%q OUTPUT_DIR=%q TRAIN_LOG=%q PROVENANCE_MANIFEST=%q RUN_STAGE=%q WANDB_GROUP=%q WANDB_NAME=%q bash %q --variant %q --algorithm %q --foreground 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo "TRAIN_EXIT=${status}"; exec bash' \
  "${TOOLVISTA_ROOT}" "${TOOLVISTA_ROOT}" "${MS_SWIFT_ROOT}" "${BASE_MODEL}" "${PYTHON_BIN}" \
  "${variant}" "${algorithm}" "${run_tag}" "${seed}" "${max_steps}" "${save_steps}" "${save_total_limit}" \
  "${checkpoint}" "${gate}" "${dataset}" "${dataset_report}" "${media_dir}" "${reward_plugin}" \
  "${output_dir}" "${train_log}" "${provenance_manifest}" "${stage_name}" "${wandb_group}" "${wandb_name}" \
  "${BASH_SOURCE[0]}" "${variant}" "${algorithm}" "${train_log}"
printf -v log_command 'tail --retry -n 100 -F %q; exec bash' "${train_log}"
printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

tmux new-session -d -s "${session}" -n train "${worker_command}"
tmux new-window -d -t "${session}" -n log "${log_command}"
tmux new-window -d -t "${session}" -n gpu "${gpu_command}"
tmux select-window -t "${session}:train"

echo "SESSION=${session}"
echo "VARIANT=${variant}"
echo "ALGORITHM=${algorithm}"
echo "OUTPUT_DIR=${output_dir}"
echo "TRAIN_LOG=${train_log}"
echo "Attach: tmux attach-session -t ${session}"

if [[ "${attach}" == true ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "${session}"
  fi
  exec tmux attach-session -t "${session}"
fi
