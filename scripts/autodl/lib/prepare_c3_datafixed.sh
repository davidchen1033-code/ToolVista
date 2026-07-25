#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
CONFIG="${CONFIG:-${TOOLVISTA_ROOT}/configs/autodl.yaml}"
SEED="${SEED:-42}"
MLLM_EVAL_FRACTION="${MLLM_EVAL_FRACTION:-0.20}"
LIGHT_LIMIT="${LIGHT_LIMIT:-200}"
ROUTER_CATALOG_SIZE="${ROUTER_CATALOG_SIZE:-32}"
DIRECT_MAX_ROWS="${DIRECT_MAX_ROWS:-12000}"
OVERWRITE="${OVERWRITE:-false}"
SKIP_MEDIA_VALIDATION="${SKIP_MEDIA_VALIDATION:-false}"

ROUTER_TRAIN="${ROUTER_TRAIN:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl}"
ROUTER_FULL="${ROUTER_FULL:-${TOOLVISTA_ROOT}/data/eval/mllm_router_datafixed/full.jsonl}"
ROUTER_LIGHT="${ROUTER_LIGHT:-${TOOLVISTA_ROOT}/data/eval/mllm_router_datafixed/light.jsonl}"
ROUTER_SPLIT="${ROUTER_SPLIT:-${TOOLVISTA_ROOT}/data/splits/mllm_router_datafixed/seed${SEED}.json}"
DIRECT_SOURCE="${DIRECT_SOURCE:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
DIRECT_OUTPUT="${DIRECT_OUTPUT:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl}"
DIRECT_REPORT="${DIRECT_REPORT:-${DIRECT_OUTPUT%.jsonl}.report.json}"
DATAFIXED_REPORT="${DATAFIXED_REPORT:-${TOOLVISTA_ROOT}/data/reports/c3/router_datafixed.report.json}"

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
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

test -d "${TOOLVISTA_ROOT}"
cd "${TOOLVISTA_ROOT}"
test -x "${PYTHON_BIN}"
test -s "${CONFIG}"
test -s "${DIRECT_SOURCE}"
require_datafixed_path ROUTER_TRAIN "${ROUTER_TRAIN}"
require_datafixed_path ROUTER_FULL "${ROUTER_FULL}"
require_datafixed_path ROUTER_LIGHT "${ROUTER_LIGHT}"
require_datafixed_path ROUTER_SPLIT "${ROUTER_SPLIT}"
require_datafixed_path DIRECT_OUTPUT "${DIRECT_OUTPUT}"
require_datafixed_path DIRECT_REPORT "${DIRECT_REPORT}"
require_datafixed_path DATAFIXED_REPORT "${DATAFIXED_REPORT}"

overwrite_args=()
case "${OVERWRITE}" in
  true) overwrite_args+=(--overwrite) ;;
  false) ;;
  *) echo "OVERWRITE must be true or false" >&2; exit 2 ;;
esac

media_args=()
case "${SKIP_MEDIA_VALIDATION}" in
  true) media_args+=(--skip-media-validation) ;;
  false) ;;
  *) echo "SKIP_MEDIA_VALIDATION must be true or false" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" scripts/prepare_c3_router_datafixed.py \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --eval-fraction "${MLLM_EVAL_FRACTION}" \
  --light-limit "${LIGHT_LIMIT}" \
  --catalog-size "${ROUTER_CATALOG_SIZE}" \
  --file-keys text image video \
  --router-train "${ROUTER_TRAIN}" \
  --router-full "${ROUTER_FULL}" \
  --router-light "${ROUTER_LIGHT}" \
  --router-split "${ROUTER_SPLIT}" \
  --direct-source "${DIRECT_SOURCE}" \
  --direct-output "${DIRECT_OUTPUT}" \
  --direct-report "${DIRECT_REPORT}" \
  --direct-max-rows "${DIRECT_MAX_ROWS}" \
  --report "${DATAFIXED_REPORT}" \
  "${media_args[@]}" \
  "${overwrite_args[@]}"

for artifact in \
  "${ROUTER_TRAIN}" \
  "${ROUTER_FULL}" \
  "${ROUTER_LIGHT}" \
  "${ROUTER_SPLIT}" \
  "${DIRECT_OUTPUT}" \
  "${DIRECT_REPORT}" \
  "${DATAFIXED_REPORT}"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "Missing datafixed artifact: ${artifact}" >&2
    exit 1
  fi
done

echo "C3 datafixed preparation completed: ${DATAFIXED_REPORT}"

