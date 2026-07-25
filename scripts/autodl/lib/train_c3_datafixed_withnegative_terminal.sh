#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C2_CHECKPOINT="${C2_CHECKPOINT:?Set C2_CHECKPOINT to the completed C2 checkpoint directory}"
VARIANT="datafixed-agentretention-terminalfix"
PREP_REPORT="${PREP_REPORT:-${TOOLVISTA_ROOT}/data/reports/c3/router_${VARIANT}.report.json}"
ROUTER_DATASET="${ROUTER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_${VARIANT}_train.jsonl}"
DIRECT_ANSWER_DATASET="${DIRECT_ANSWER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_direct_answer_${VARIANT}_train.jsonl}"
TOOLVQA_COMPLETE_DATASET="${TOOLVQA_COMPLETE_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_complete_${VARIANT}_train.jsonl}"
TOOLVQA_TERMINAL_DATASET="${TOOLVQA_TERMINAL_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_terminal_answer_${VARIANT}_train.jsonl}"
TOOLVQA_REACT_ACTION_DATASET="${TOOLVQA_REACT_ACTION_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_react_action_${VARIANT}_train.jsonl}"
TOOLVQA_REACT_TERMINAL_DATASET="${TOOLVQA_REACT_TERMINAL_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_react_terminal_${VARIANT}_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
XLAM_ABSTENTION_DATASET="${XLAM_ABSTENTION_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_abstention_${VARIANT}_train.jsonl}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_router_${VARIANT}_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_${VARIANT}_sft}"
STAGE="${STAGE:-c3-router-${VARIANT}}"
WANDB_GROUP="${WANDB_GROUP:-c3-router-${VARIANT}}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1700}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"

has_variant_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])(datafixed-agentretention-terminalfix|datafixed-withnegative-terminal)([/_.-]|$) ]]
}

require_variant_path() {
  local label="$1"
  local value="$2"
  if ! has_variant_marker "${value}" || ! has_variant_marker "$(realpath -m -- "${value}")"; then
    echo "${label} must remain in the ${VARIANT} namespace: ${value}" >&2
    exit 2
  fi
}

if [[ ! "${MAX_STEPS}" =~ ^[0-9]+$ ]] || (( 10#${MAX_STEPS} < 1 || 10#${MAX_STEPS} > 1700 )); then
  echo "MAX_STEPS must be an integer between 1 and 1700 for ${VARIANT}" >&2
  exit 2
fi
MAX_STEPS="$((10#${MAX_STEPS}))"

test -d "${TOOLVISTA_ROOT}"
test -d "${C2_CHECKPOINT}"
test -x "${PYTHON_BIN}"
for path in \
  "${PREP_REPORT}" \
  "${ROUTER_DATASET}" \
  "${DIRECT_ANSWER_DATASET}" \
  "${TOOLVQA_COMPLETE_DATASET}" \
  "${TOOLVQA_TERMINAL_DATASET}" \
  "${TOOLVQA_REACT_ACTION_DATASET}" \
  "${TOOLVQA_REACT_TERMINAL_DATASET}" \
  "${XLAM_DATASET}" \
  "${XLAM_ABSTENTION_DATASET}"; do
  test -s "${path}"
done
for path in \
  "${PREP_REPORT}" \
  "${ROUTER_DATASET}" \
  "${DIRECT_ANSWER_DATASET}" \
  "${TOOLVQA_COMPLETE_DATASET}" \
  "${TOOLVQA_TERMINAL_DATASET}" \
  "${TOOLVQA_REACT_ACTION_DATASET}" \
  "${TOOLVQA_REACT_TERMINAL_DATASET}" \
  "${XLAM_ABSTENTION_DATASET}" \
  "${REPLAY_DATASET}" \
  "${REPLAY_REPORT}" \
  "${OUTPUT_DIR}" \
  "${STAGE}" \
  "${WANDB_GROUP}"; do
  require_variant_path terminalfix "${path}"
done

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/prepare_c3_datafixed_terminalfix.py --verify-only
"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("variant") != "datafixed-agentretention-terminalfix":
    raise SystemExit("terminal-fix preparation report variant mismatch")
if report.get("official_test_exclusion", {}).get("postbuild_overlap_rows") != 0:
    raise SystemExit("terminal-fix preparation report does not prove zero test overlap")
bindings = {
    "router_train": sys.argv[2],
    "direct_answer": sys.argv[3],
    "toolvqa_complete": sys.argv[4],
    "toolvqa_terminal": sys.argv[5],
    "toolvqa_react_action": sys.argv[6],
    "toolvqa_react_terminal": sys.argv[7],
    "xlam_abstention": sys.argv[8],
}
for key, actual in bindings.items():
    expected = Path(report["artifacts"][key]).resolve(strict=True)
    if Path(actual).resolve(strict=True) != expected:
        raise SystemExit(f"{key} is not authorized by the terminal-fix report")
' "${PREP_REPORT}" "${ROUTER_DATASET}" "${DIRECT_ANSWER_DATASET}" "${TOOLVQA_COMPLETE_DATASET}" "${TOOLVQA_TERMINAL_DATASET}" "${TOOLVQA_REACT_ACTION_DATASET}" "${TOOLVQA_REACT_TERMINAL_DATASET}" "${XLAM_ABSTENTION_DATASET}"

case "${DRY_RUN:-0}" in
  1)
    echo "DRY_RUN=OK"
    echo "C2_CHECKPOINT=${C2_CHECKPOINT}"
    echo "REPLAY_DATASET=${REPLAY_DATASET}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    exit 0
    ;;
  0) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "router_terminalfix=${ROUTER_DATASET}" \
  --source "toolvqa_react_terminal=${TOOLVQA_REACT_TERMINAL_DATASET}" \
  --source "toolvqa_react_action=${TOOLVQA_REACT_ACTION_DATASET}" \
  --source "toolvqa_complete=${TOOLVQA_COMPLETE_DATASET}" \
  --source "toolvqa_terminal=${TOOLVQA_TERMINAL_DATASET}" \
  --source "direct_answer=${DIRECT_ANSWER_DATASET}" \
  --source "xlam_general=${XLAM_DATASET}" \
  --source "xlam_abstention=${XLAM_ABSTENTION_DATASET}" \
  --weight "router_terminalfix=0.25" \
  --weight "toolvqa_react_terminal=0.25" \
  --weight "toolvqa_react_action=0.15" \
  --weight "toolvqa_complete=0.12" \
  --weight "toolvqa_terminal=0.08" \
  --weight "direct_answer=0.05" \
  --weight "xlam_general=0.075" \
  --weight "xlam_abstention=0.025" \
  --anchor router_terminalfix \
  --seed "${SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE DATASET OUTPUT_DIR WANDB_GROUP PREVIOUS_CHECKPOINT
export LEARNING_RATE MAX_STEPS SEED
export PREVIOUS_CHECKPOINT="${C2_CHECKPOINT}"
export DATASET="${REPLAY_DATASET}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-18}"

source "${TOOLVISTA_ROOT}/scripts/autodl/lib/sft_common.sh"
