#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C2_CHECKPOINT="${C2_CHECKPOINT:?Set C2_CHECKPOINT to the completed C2 checkpoint directory}"
DEFAULT_AGENT_PREP_REPORT="${TOOLVISTA_ROOT}/data/reports/c3/router_datafixed-agentretention.report.json"
AGENT_PREP_REPORT="${AGENT_PREP_REPORT:-${DEFAULT_AGENT_PREP_REPORT}}"
ROUTER_DATASET="${ROUTER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_datafixed-agentretention_train.jsonl}"
DIRECT_ANSWER_DATASET="${DIRECT_ANSWER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl}"
TOOLVQA_COMPLETE_DATASET="${TOOLVQA_COMPLETE_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_complete_datafixed-agentretention_train.jsonl}"
TOOLVQA_TERMINAL_DATASET="${TOOLVQA_TERMINAL_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_terminal_answer_datafixed-agentretention_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
XLAM_ABSTENTION_DATASET="${XLAM_ABSTENTION_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_abstention_datafixed-agentretention_train.jsonl}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_router_datafixed-agentretention_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention_sft}"
STAGE="${STAGE:-c3-router-datafixed-agentretention}"
WANDB_GROUP="${WANDB_GROUP:-c3-router-datafixed-agentretention}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1215}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

has_variant_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])(datafixed-agentretention|datafixed-withnegative)([/_.-]|$) ]]
}

require_variant_path() {
  local label="$1"
  local value="$2"
  if ! has_variant_marker "${value}" || ! has_variant_marker "$(realpath -m -- "${value}")"; then
    echo "${label} must remain in the datafixed-agentretention namespace: ${value}" >&2
    exit 2
  fi
}

if [[ ! "${MAX_STEPS}" =~ ^[0-9]+$ ]] || (( 10#${MAX_STEPS} < 1 || 10#${MAX_STEPS} > 1215 )); then
  echo "MAX_STEPS must be an integer between 1 and 1215 for datafixed-agentretention" >&2
  exit 2
fi
MAX_STEPS="$((10#${MAX_STEPS}))"

test -d "${TOOLVISTA_ROOT}"
test -d "${C2_CHECKPOINT}"
test -x "${PYTHON_BIN}"
test -s "${AGENT_PREP_REPORT}"
if [[ "$(realpath "${AGENT_PREP_REPORT}")" != "$(realpath "${DEFAULT_AGENT_PREP_REPORT}")" ]]; then
  echo "AGENT_PREP_REPORT must be the formal datafixed-agentretention report" >&2
  exit 2
fi
for path in \
  "${ROUTER_DATASET}" \
  "${DIRECT_ANSWER_DATASET}" \
  "${TOOLVQA_COMPLETE_DATASET}" \
  "${TOOLVQA_TERMINAL_DATASET}" \
  "${XLAM_DATASET}" \
  "${XLAM_ABSTENTION_DATASET}"; do
  test -s "${path}"
done
require_variant_path AGENT_PREP_REPORT "${AGENT_PREP_REPORT}"
require_variant_path ROUTER_DATASET "${ROUTER_DATASET}"
require_variant_path TOOLVQA_COMPLETE_DATASET "${TOOLVQA_COMPLETE_DATASET}"
require_variant_path TOOLVQA_TERMINAL_DATASET "${TOOLVQA_TERMINAL_DATASET}"
require_variant_path XLAM_ABSTENTION_DATASET "${XLAM_ABSTENTION_DATASET}"
require_variant_path REPLAY_DATASET "${REPLAY_DATASET}"
require_variant_path REPLAY_REPORT "${REPLAY_REPORT}"
require_variant_path OUTPUT_DIR "${OUTPUT_DIR}"
case "${STAGE}" in c3-*) ;; *) echo "STAGE must identify c3" >&2; exit 2;; esac
require_variant_path STAGE "${STAGE}"
require_variant_path WANDB_GROUP "${WANDB_GROUP}"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/prepare_c3_datafixed_agentretention.py --verify-only
"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
actual = {
    name: Path(value).resolve(strict=True)
    for name, value in zip(
        ("router", "direct", "complete", "terminal", "xlam", "abstention"),
        sys.argv[2:],
        strict=True,
    )
}
bindings = (
    ("router", "artifacts", "router_train"),
    ("direct", "inputs", "direct_answer"),
    ("complete", "artifacts", "toolvqa_complete"),
    ("terminal", "artifacts", "toolvqa_terminal"),
    ("xlam", "inputs", "xlam"),
    ("abstention", "artifacts", "xlam_abstention"),
)
for name, section, key in bindings:
    value = report[section][key]
    if section == "inputs":
        value = value["path"]
    expected = Path(value).resolve(strict=True)
    if actual[name] != expected:
        raise SystemExit(
            f"{name} dataset is not authorized by {section}.{key}: "
            f"{actual[name]} != {expected}"
        )
' "${AGENT_PREP_REPORT}" "${ROUTER_DATASET}" "${DIRECT_ANSWER_DATASET}" "${TOOLVQA_COMPLETE_DATASET}" "${TOOLVQA_TERMINAL_DATASET}" "${XLAM_DATASET}" "${XLAM_ABSTENTION_DATASET}"

case "${DRY_RUN:-0}" in
  1)
    echo "DRY_RUN=OK"
    echo "C2_CHECKPOINT=${C2_CHECKPOINT}"
    echo "REPLAY_DATASET=${REPLAY_DATASET}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "MAX_STEPS=${MAX_STEPS}"
    exit 0
    ;;
  0) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "router_agentretention=${ROUTER_DATASET}" \
  --source "toolvqa_terminal_agentretention=${TOOLVQA_TERMINAL_DATASET}" \
  --source "toolvqa_complete_agentretention=${TOOLVQA_COMPLETE_DATASET}" \
  --source "direct_answer_datafixed=${DIRECT_ANSWER_DATASET}" \
  --source "xlam_general=${XLAM_DATASET}" \
  --source "xlam_abstention_agentretention=${XLAM_ABSTENTION_DATASET}" \
  --weight "router_agentretention=0.35" \
  --weight "toolvqa_terminal_agentretention=0.265" \
  --weight "toolvqa_complete_agentretention=0.15" \
  --weight "direct_answer_datafixed=0.10" \
  --weight "xlam_general=0.10" \
  --weight "xlam_abstention_agentretention=0.035" \
  --anchor router_agentretention \
  --seed "${SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE DATASET OUTPUT_DIR WANDB_GROUP PREVIOUS_CHECKPOINT
export LEARNING_RATE MAX_STEPS SEED
export PREVIOUS_CHECKPOINT="${C2_CHECKPOINT}"
export DATASET="${REPLAY_DATASET}"
export OUTPUT_DIR
export WANDB_GROUP
export LEARNING_RATE MAX_STEPS
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

source "${TOOLVISTA_ROOT}/scripts/autodl/lib/sft_common.sh"
