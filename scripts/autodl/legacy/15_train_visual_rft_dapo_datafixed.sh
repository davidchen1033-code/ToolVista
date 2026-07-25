#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C3_DATAFIXED_CHECKPOINT="${C3_DATAFIXED_CHECKPOINT:?Set C3_DATAFIXED_CHECKPOINT to the passed C3 datafixed checkpoint-* directory}"
C3_DATAFIXED_RETENTION_GATE="${C3_DATAFIXED_RETENTION_GATE:-${C3_DATAFIXED_CHECKPOINT}/toolvista_c3_datafixed_retention_gate.json}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
}

checkpoint_resolved="$(realpath -- "${C3_DATAFIXED_CHECKPOINT}")"
gate_resolved="$(realpath -- "${C3_DATAFIXED_RETENTION_GATE}")"
has_datafixed_marker "${C3_DATAFIXED_CHECKPOINT}" && has_datafixed_marker "${checkpoint_resolved}" || {
  echo "C3_DATAFIXED_CHECKPOINT must stay in the datafixed experiment track: ${C3_DATAFIXED_CHECKPOINT}" >&2
  exit 2
}
[[ "$(dirname "${gate_resolved}")" == "${checkpoint_resolved}" ]] || {
  echo "C3_DATAFIXED_RETENTION_GATE must be inside C3_DATAFIXED_CHECKPOINT" >&2
  exit 2
}

"${PYTHON_BIN}" - "${gate_resolved}" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("variant") != "datafixed":
    raise SystemExit("C3 retention gate variant is not datafixed")
PY

export C3_CHECKPOINT="${checkpoint_resolved}"
export C3_RETENTION_GATE="${gate_resolved}"
export SEED
export STAGE="${STAGE:-rl-dapo-datafixed}"
export WANDB_GROUP="${WANDB_GROUP:-rl-datafixed}"
export WANDB_NAME="${WANDB_NAME:-c4-visualrft-coco-dapo-datafixed-${RUN_TAG}-seed${SEED}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed_seed${SEED}_run${RUN_TAG}}"

has_datafixed_marker "${OUTPUT_DIR}" || {
  echo "OUTPUT_DIR must stay in the datafixed experiment track: ${OUTPUT_DIR}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite an existing RL output: ${OUTPUT_DIR}" >&2
  exit 2
}

if [[ -n "${TRAIN_LOG:-}" ]]; then
  mkdir -p "$(dirname "${TRAIN_LOG}")"
  set -o pipefail
  bash "${TOOLVISTA_ROOT}/scripts/autodl/10_train_visual_rft_dapo.sh" \
    2>&1 | tee "${TRAIN_LOG}"
  exit "${PIPESTATUS[0]}"
fi

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/10_train_visual_rft_dapo.sh"
