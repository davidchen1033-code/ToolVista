#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
CHECKPOINT="${C3_DATAFIXED_CHECKPOINT:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention-terminalfix_sft/terminalfix20260723T100256Z/v0-20260723-180402/checkpoint-1700}"
GATE="${C3_DATAFIXED_RETENTION_GATE:-${CHECKPOINT}/toolvista_c3_datafixed_agentretention_terminalfix_retention_gate.json}"
ALGORITHM="${ALGORITHM:-grpo}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
STAGE="${STAGE:-rl-${LOSS_TYPE}-datafixed-withnegative-terminalfix}"
WANDB_GROUP="${WANDB_GROUP:-rl-${LOSS_TYPE}-datafixed-withnegative-terminalfix}"

SESSION="${SESSION:-c4-datafixed-withnegative-terminalfix-grpo}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-terminalfix-grpo$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_STEPS="${MAX_STEPS:-6000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-13}"

DATASET="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl}"
DATASET_REPORT="${DATASET_REPORT:-${TOOLVISTA_ROOT}/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json}"
MEDIA_DIR="${MEDIA_DIR:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_media}"
DATASET_BUILDER="${DATASET_BUILDER:-${TOOLVISTA_ROOT}/scripts/prepare_visual_rft_datafixed_withnegative.py}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft_withnegative.py}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${TOOLVISTA_ROOT}/scripts/autodl/08_train_visual_rft_grpo.sh}"

OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/grpo_datafixed-withnegative_seed${SEED}_run${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c4/c4-grpo-datafixed-withnegative-${RUN_TAG}-seed${SEED}.train.log}"
PROVENANCE_MANIFEST="${PROVENANCE_MANIFEST:-${TRAIN_LOG%.train.log}.manifest.json}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_file() {
  [[ -f "$2" ]] || die "$1 does not exist: $2"
}

verify_contract() {
  [[ "${MAX_STEPS}" == "6000" ]] || die "strict ${ALGORITHM^^} comparison requires MAX_STEPS=6000"
  [[ "${ALGORITHM}" == "grpo" && "${LOSS_TYPE}" == "grpo" || \
     "${ALGORITHM}" == "bnpo" && "${LOSS_TYPE}" == "bnpo" || \
     "${ALGORITHM}" == "drgrpo" && "${LOSS_TYPE}" == "dr_grpo" ]] || \
    die "ALGORITHM/LOSS_TYPE must be grpo/grpo, bnpo/bnpo, or drgrpo/dr_grpo"
  [[ "${OUTPUT_DIR}" == *datafixed-withnegative* ]] || die "OUTPUT_DIR must stay in the datafixed-withnegative track"
  [[ "${TRAIN_LOG}" == *datafixed-withnegative* ]] || die "TRAIN_LOG must stay in the datafixed-withnegative track"
  [[ "${DATASET}" == *datafixed-withnegative_canonical* ]] || die "DATASET must be the canonical withnegative dataset"
  [[ "${CHECKPOINT}" == *datafixed-agentretention-terminalfix* ]] || die "CHECKPOINT must be the terminal-fix C3 adapter"
  [[ -d "${TOOLVISTA_ROOT}" ]] || die "TOOLVISTA_ROOT does not exist: ${TOOLVISTA_ROOT}"
  [[ -d "${MS_SWIFT_ROOT}" ]] || die "MS_SWIFT_ROOT does not exist: ${MS_SWIFT_ROOT}"
  [[ -d "${BASE_MODEL}" ]] || die "BASE_MODEL does not exist: ${BASE_MODEL}"
  [[ -d "${CHECKPOINT}" ]] || die "CHECKPOINT does not exist: ${CHECKPOINT}"
  [[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
  require_file GATE "${GATE}"
  require_file DATASET_BUILDER "${DATASET_BUILDER}"
  require_file REWARD_PLUGIN "${REWARD_PLUGIN}"
  require_file TRAIN_LAUNCHER "${TRAIN_LAUNCHER}"
}

write_manifest() {
  "${PYTHON_BIN}" - "${PROVENANCE_MANIFEST}" "${CHECKPOINT}" "${GATE}" "${DATASET}" "${DATASET_REPORT}" "${MEDIA_DIR}" "${REWARD_PLUGIN}" "${TRAIN_LAUNCHER}" "${OUTPUT_DIR}" "${TRAIN_LOG}" "${RUN_TAG}" "${SEED}" "${MAX_STEPS}" "${ALGORITHM}" "${LOSS_TYPE}" <<'PY'
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

(
    destination,
    checkpoint,
    gate,
    dataset,
    dataset_report,
    media_dir,
    reward_plugin,
    launcher,
    output_dir,
    train_log,
    run_tag,
    seed,
    max_steps,
    algorithm,
    loss_type,
) = sys.argv[1:]


def file_info(value: str) -> dict[str, object]:
    path = Path(value).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


payload = {
    "schema": "toolvista.rl.provenance.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "algorithm": algorithm,
    "loss_type": loss_type,
    "variant": "datafixed-withnegative",
    "c3_variant": "datafixed-agentretention-terminalfix",
    "run_tag": run_tag,
    "seed": int(seed),
    "max_steps": int(max_steps),
    "checkpoint": str(Path(checkpoint).resolve(strict=True)),
    "output_dir": str(Path(output_dir).resolve(strict=False)),
    "train_log": str(Path(train_log).resolve(strict=False)),
    "media_dir": str(Path(media_dir).resolve(strict=True)),
    "reward_funcs": ["visual_rft_withnegative", "visual_rft_withnegative_format"],
    "files": {
        "retention_gate": file_info(gate),
        "dataset": file_info(dataset),
        "dataset_report": file_info(dataset_report),
        "reward_plugin": file_info(reward_plugin),
        "rl_launcher": file_info(launcher),
    },
}

path = Path(destination)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"Provenance manifest: {path}")
PY
}

run_worker() {
  verify_contract
  [[ ! -e "${OUTPUT_DIR}" ]] || die "refusing to overwrite OUTPUT_DIR: ${OUTPUT_DIR}"
  [[ ! -e "${TRAIN_LOG}" ]] || die "refusing to overwrite TRAIN_LOG: ${TRAIN_LOG}"
  [[ ! -e "${PROVENANCE_MANIFEST}" ]] || die "refusing to overwrite provenance: ${PROVENANCE_MANIFEST}"

  "${PYTHON_BIN}" -c 'import json,sys; gate=json.load(open(sys.argv[1], encoding="utf-8")); expected="datafixed-agentretention-terminalfix"; actual=gate.get("variant"); actual == expected or (_ for _ in ()).throw(SystemExit(f"C3 retention gate variant mismatch: {actual!r} != {expected!r}"))' "${GATE}"

  cd "${TOOLVISTA_ROOT}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${DATASET}" --report "${DATASET_REPORT}" --media-dir "${MEDIA_DIR}" --verify-only
  require_file DATASET "${DATASET}"
  require_file DATASET_REPORT "${DATASET_REPORT}"
  require_file MEDIA_MANIFEST "${MEDIA_DIR}/manifest.jsonl"

  mkdir -p "$(dirname "${TRAIN_LOG}")" "$(dirname "${OUTPUT_DIR}")"
  write_manifest

  export TOOLVISTA_ROOT MS_SWIFT_ROOT BASE_MODEL PYTHON_BIN
  export C3_CHECKPOINT="${CHECKPOINT}"
  export C3_RETENTION_GATE="${GATE}"
  export DATASET REWARD_PLUGIN SEED MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT OUTPUT_DIR
  export REWARD_FUNCS="visual_rft_withnegative visual_rft_withnegative_format"
  export STAGE WANDB_GROUP
  export WANDB_NAME="c4-visualrft-coco-${ALGORITHM}-datafixed-withnegative-${RUN_TAG}-seed${SEED}"

  echo "ALGORITHM=${ALGORITHM}"
  echo "LOSS_TYPE=${LOSS_TYPE}"
  echo "C3_CHECKPOINT=${C3_CHECKPOINT}"
  echo "DATASET=${DATASET}"
  echo "REWARD_FUNCS=${REWARD_FUNCS}"
  echo "MAX_STEPS=${MAX_STEPS}"
  set -o pipefail
  bash "${TRAIN_LAUNCHER}" 2>&1 | tee "${TRAIN_LOG}"
  exit "${PIPESTATUS[0]}"
}

launch_tmux() {
  verify_contract
  command -v tmux >/dev/null || die "tmux is not installed"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is not installed"
  tmux has-session -t "${SESSION}" 2>/dev/null && die "tmux session already exists: ${SESSION}"

  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -z "${gpu_pids}" ]] || die "GPU has active compute process IDs: ${gpu_pids}"

  for path in "${OUTPUT_DIR}" "${TRAIN_LOG}" "${PROVENANCE_MANIFEST}"; do
    [[ ! -e "${path}" ]] || die "refusing to overwrite existing path: ${path}"
  done
  mkdir -p "$(dirname "${TRAIN_LOG}")" "$(dirname "${OUTPUT_DIR}")"

  printf -v worker_command 'cd %q; env ALGORITHM=%q LOSS_TYPE=%q STAGE=%q WANDB_GROUP=%q TRAIN_LAUNCHER=%q RUN_TAG=%q OUTPUT_DIR=%q TRAIN_LOG=%q PROVENANCE_MANIFEST=%q bash %q --worker; status=$?; echo "TRAIN_EXIT=${status}"; exec bash' "${TOOLVISTA_ROOT}" "${ALGORITHM}" "${LOSS_TYPE}" "${STAGE}" "${WANDB_GROUP}" "${TRAIN_LAUNCHER}" "${RUN_TAG}" "${OUTPUT_DIR}" "${TRAIN_LOG}" "${PROVENANCE_MANIFEST}" "${BASH_SOURCE[0]}"
  printf -v log_command 'tail --retry -n 100 -F %q; exec bash' "${TRAIN_LOG}"
  printf -v gpu_command 'watch -n 5 nvidia-smi; exec bash'

  tmux new-session -d -s "${SESSION}" -n train "${worker_command}"
  tmux new-window -d -t "${SESSION}" -n log "${log_command}"
  tmux new-window -d -t "${SESSION}" -n gpu "${gpu_command}"
  tmux select-window -t "${SESSION}:train"

  echo "SESSION=${SESSION}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "TRAIN_LOG=${TRAIN_LOG}"
  echo "Attach with: tmux attach-session -t ${SESSION}"
}

case "${1:-}" in
  --worker) run_worker ;;
  "") launch_tmux ;;
  *) die "unknown argument: $1" ;;
esac
