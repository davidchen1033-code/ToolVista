#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
source "${TOOLVISTA_ROOT}/scripts/autodl/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: prepare_data.sh --stage STAGE --variant VARIANT [options] [-- EXTRA_ARGS]

Stages:
  c0       Prepare shared raw training/evaluation assets.
  toolvqa  Export the raw ToolVQA SFT dataset.
  c3       Prepare the selected router/replay data lineage.
  c4       Prepare or verify the selected Visual-RFT dataset.

Variants:
  raw | datafixed | datafixed-withnegative | datafixed-withnegative-terminal

Options:
  --overwrite    Permit deterministic builders to replace their own artifacts.
  --verify-only  Verify an existing C4 with-negative export without rebuilding it.
EOF
}

pipeline_stage="${PIPELINE_STAGE:-}"
variant_input="${EXPERIMENT_VARIANT:-}"
overwrite="${OVERWRITE:-false}"
verify_only="${VERIFY_ONLY:-false}"
extra_args=()

while (( $# )); do
  case "$1" in
    --stage) pipeline_stage="${2:?missing value for --stage}"; shift 2 ;;
    --variant) variant_input="${2:?missing value for --variant}"; shift 2 ;;
    --overwrite) overwrite=true; shift ;;
    --verify-only) verify_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) autodl_die "unknown argument: $1" ;;
  esac
done

[[ -n "${pipeline_stage}" ]] || autodl_die "--stage is required"
[[ -n "${variant_input}" ]] || autodl_die "--variant is required"
variant="$(autodl_normalize_variant "${variant_input}")"
autodl_require_bool OVERWRITE "${overwrite}"
autodl_require_bool VERIFY_ONLY "${verify_only}"
autodl_require_dir TOOLVISTA_ROOT "${TOOLVISTA_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || autodl_die "PYTHON_BIN is not executable: ${PYTHON_BIN}"

case "${pipeline_stage}:${variant}" in
  c0:raw)
    (( ${#extra_args[@]} == 0 )) || autodl_die "c0 raw preparation does not accept extra arguments"
    exec bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/prepare_c0_raw.sh"
    ;;
  toolvqa:raw)
    (( ${#extra_args[@]} == 0 )) || autodl_die "ToolVQA raw export does not accept extra arguments"
    exec bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/prepare_toolvqa_raw.sh"
    ;;
  c3:raw)
    autodl_die "raw C3 inputs are produced by: prepare_data.sh --stage c0 --variant raw"
    ;;
  c3:datafixed)
    exec env OVERWRITE="${overwrite}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/prepare_c3_datafixed.sh" "${extra_args[@]}"
    ;;
  c3:datafixed-withnegative)
    exec env OVERWRITE="${overwrite}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/prepare_c3_datafixed_withnegative.sh" "${extra_args[@]}"
    ;;
  c3:datafixed-withnegative-terminal)
    exec env OVERWRITE="${overwrite}" bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/prepare_c3_datafixed_withnegative_terminal.sh" "${extra_args[@]}"
    ;;
  c4:raw|c4:datafixed)
    dataset="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl}"
    autodl_require_file DATASET "${dataset}"
    exec "${PYTHON_BIN}" "${TOOLVISTA_ROOT}/scripts/validate_visual_rft_rl_dataset.py" "${dataset}" --max-rows "${VALIDATE_ROWS:-0}"
    ;;
  c4:datafixed-withnegative|c4:datafixed-withnegative-terminal)
    dataset="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl}"
    report="${DATASET_REPORT:-${TOOLVISTA_ROOT}/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json}"
    media_dir="${MEDIA_DIR:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_media}"
    builder_args=(--output "${dataset}" --report "${report}" --media-dir "${media_dir}")
    [[ "${overwrite}" == false ]] || builder_args+=(--overwrite)
    [[ "${verify_only}" == false ]] || builder_args+=(--verify-only)
    exec "${PYTHON_BIN}" "${TOOLVISTA_ROOT}/scripts/prepare_visual_rft_datafixed_withnegative.py" "${builder_args[@]}" "${extra_args[@]}"
    ;;
  *) autodl_die "unsupported preparation combination: stage=${pipeline_stage}, variant=${variant}" ;;
esac
