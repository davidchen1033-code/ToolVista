#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C2_CHECKPOINT="${C2_CHECKPOINT:?Set C2_CHECKPOINT to the completed C2 checkpoint-* directory}"
ROUTER_DATASET="${ROUTER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl}"
DIRECT_ANSWER_DATASET="${DIRECT_ANSWER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl}"
TOOLVQA_DATASET="${TOOLVQA_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
DATAFIXED_PREP_REPORT="${DATAFIXED_PREP_REPORT:-${TOOLVISTA_ROOT}/data/reports/c3/router_datafixed.report.json}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_router_datafixed_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
REPLAY_SEED="${REPLAY_SEED:-${SEED:-42}}"
ROUTER_WEIGHT="${ROUTER_WEIGHT:-0.4}"
DIRECT_ANSWER_WEIGHT="${DIRECT_ANSWER_WEIGHT:-0.3}"
TOOLVQA_WEIGHT="${TOOLVQA_WEIGHT:-0.15}"
XLAM_WEIGHT="${XLAM_WEIGHT:-0.15}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_STEPS="${MAX_STEPS:-1063}"
STAGE="${STAGE:-c3-router-datafixed-retention}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed_sft}"
WANDB_GROUP="${WANDB_GROUP:-c3-router-datafixed-retention}"

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
}

require_datafixed_name() {
  local label="$1"
  local value="$2"
  if ! has_datafixed_marker "${value}"; then
    echo "${label} must contain a safe 'datafixed' marker: ${value}" >&2
    exit 2
  fi
}

require_datafixed_path() {
  local label="$1"
  local value="$2"
  local resolved
  if ! has_datafixed_marker "${value}"; then
    echo "${label} lexical path must contain a safe 'datafixed' marker: ${value}" >&2
    exit 2
  fi
  resolved="$(realpath -m -- "${value}")"
  if ! has_datafixed_marker "${resolved}"; then
    echo "${label} resolves outside the datafixed namespace: ${value} -> ${resolved}" >&2
    exit 2
  fi
}

if [[ ! "${MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_STEPS must be an integer between 1 and 1063 for the datafixed track" >&2
  exit 2
fi
max_steps_decimal="${MAX_STEPS#"${MAX_STEPS%%[!0]*}"}"
max_steps_decimal="${max_steps_decimal:-0}"
if (( ${#max_steps_decimal} > 4 )) || (( 10#${max_steps_decimal} < 1 || 10#${max_steps_decimal} > 1063 )); then
  echo "MAX_STEPS must be an integer between 1 and 1063 for the datafixed track" >&2
  exit 2
fi
MAX_STEPS="$((10#${max_steps_decimal}))"

test -d "${TOOLVISTA_ROOT}"
cd "${TOOLVISTA_ROOT}"
test -d "${C2_CHECKPOINT}"
test -s "${ROUTER_DATASET}"
test -s "${DIRECT_ANSWER_DATASET}"
test -s "${TOOLVQA_DATASET}"
test -s "${XLAM_DATASET}"
test -s "${DATAFIXED_PREP_REPORT}"
require_datafixed_path ROUTER_DATASET "${ROUTER_DATASET}"
require_datafixed_path DIRECT_ANSWER_DATASET "${DIRECT_ANSWER_DATASET}"
require_datafixed_path DATAFIXED_PREP_REPORT "${DATAFIXED_PREP_REPORT}"
require_datafixed_path REPLAY_DATASET "${REPLAY_DATASET}"
require_datafixed_path REPLAY_REPORT "${REPLAY_REPORT}"
require_datafixed_path OUTPUT_DIR "${OUTPUT_DIR}"
require_datafixed_name STAGE "${STAGE}"
require_datafixed_name WANDB_GROUP "${WANDB_GROUP}"

"${PYTHON_BIN}" -c '
import hashlib
import json
import sys
from pathlib import Path


def require_mapping(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def resolve_artifact(value, root, label):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    report_path = Path(sys.argv[1]).expanduser().resolve(strict=True)
    root = Path(sys.argv[2]).expanduser().resolve(strict=False)
    actual = {
        "router_train": Path(sys.argv[3]).expanduser().resolve(strict=True),
        "direct_answer_train": Path(sys.argv[4]).expanduser().resolve(strict=True),
    }
    with report_path.open("r", encoding="utf-8") as handle:
        report = require_mapping(json.load(handle), "preparation report")
    if report.get("variant") != "datafixed":
        raise ValueError("preparation report variant must be datafixed")

    artifacts = require_mapping(report.get("artifacts"), "preparation report artifacts")
    audits = require_mapping(report.get("audit"), "preparation report audit")
    bindings = (
        ("router_train", "router_train"),
        ("direct_answer_train", "direct_answer"),
    )
    for artifact_key, audit_key in bindings:
        reported_path = resolve_artifact(
            artifacts.get(artifact_key), root, f"artifacts.{artifact_key}"
        )
        if reported_path != actual[artifact_key]:
            raise ValueError(
                f"artifacts.{artifact_key} does not authorize the selected input: "
                f"reported={reported_path}, selected={actual[artifact_key]}"
            )
        audit = require_mapping(audits.get(audit_key), f"audit.{audit_key}")
        reported_sha = audit.get("sha256")
        actual_sha = sha256(actual[artifact_key])
        if reported_sha != actual_sha:
            raise ValueError(
                f"audit.{audit_key}.sha256 does not match {artifact_key}: "
                f"reported={reported_sha!r}, actual={actual_sha}"
            )
except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
    raise SystemExit(f"datafixed preparation report validation failed: {exc}") from exc
' "${DATAFIXED_PREP_REPORT}" "${TOOLVISTA_ROOT}" "${ROUTER_DATASET}" "${DIRECT_ANSWER_DATASET}"

"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "router_datafixed=${ROUTER_DATASET}" \
  --source "direct_answer_datafixed=${DIRECT_ANSWER_DATASET}" \
  --source "toolvqa_datafixed_replay=${TOOLVQA_DATASET}" \
  --source "xlam_datafixed_replay=${XLAM_DATASET}" \
  --weight "router_datafixed=${ROUTER_WEIGHT}" \
  --weight "direct_answer_datafixed=${DIRECT_ANSWER_WEIGHT}" \
  --weight "toolvqa_datafixed_replay=${TOOLVQA_WEIGHT}" \
  --weight "xlam_datafixed_replay=${XLAM_WEIGHT}" \
  --anchor router_datafixed \
  --seed "${REPLAY_SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE
export DATASET="${REPLAY_DATASET}"
export OUTPUT_DIR
export WANDB_GROUP
export PREVIOUS_CHECKPOINT="${C2_CHECKPOINT}"
export LEARNING_RATE
export MAX_STEPS
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

source "${TOOLVISTA_ROOT}/scripts/autodl/lib/sft_common.sh"
