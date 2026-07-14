#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ToolVista/data/raw/ToolVQA}"
REPO_ID="${REPO_ID:-DietCoke4671/ToolVQA}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${DATA_ROOT}"

hf download "${REPO_ID}" \
  --repo-type dataset \
  --include "train.jsonl" "test.jsonl" "images/**" \
  --local-dir "${DATA_ROOT}"

test -s "${DATA_ROOT}/train.jsonl"
test -d "${DATA_ROOT}/images"

echo "ToolVQA downloaded to ${DATA_ROOT}"
du -sh "${DATA_ROOT}"
find "${DATA_ROOT}/images" -type f | wc -l
