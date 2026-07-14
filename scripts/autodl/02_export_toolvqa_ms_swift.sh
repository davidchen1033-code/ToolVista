#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
CONFIG="${CONFIG:-${TOOLVISTA_ROOT}/configs/autodl.yaml}"
MAX_ITEMS="${MAX_ITEMS:-50}"
OUTPUT="${OUTPUT:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"

cd "${TOOLVISTA_ROOT}"

python scripts/export_toolvqa.py \
  --config "${CONFIG}" \
  --split train \
  --max-items "${MAX_ITEMS}" \
  --output "${OUTPUT}"

echo "Exported dataset: ${OUTPUT}"
echo "Summary report: ${OUTPUT%.jsonl}.report.json"
echo "Per-record issues: ${OUTPUT%.jsonl}.errors.jsonl"
wc -l "${OUTPUT}"
