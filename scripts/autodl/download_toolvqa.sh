#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ToolVista/data/raw/ToolVQA}"
REPO_ID="${REPO_ID:-DietCoke4671/ToolVQA}"
HF_BIN="${HF_BIN:-/root/autodl-tmp/venvs/toolvista/bin/hf}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

test -x "${HF_BIN}"
mkdir -p "${DATA_ROOT}"

"${HF_BIN}" download "${REPO_ID}" \
  --repo-type dataset \
  --include "train.jsonl" "test.jsonl" "images/**" \
  --local-dir "${DATA_ROOT}"

test -s "${DATA_ROOT}/train.jsonl"
test -d "${DATA_ROOT}/images"

echo "ToolVQA downloaded to ${DATA_ROOT}"
du -sh "${DATA_ROOT}"
find "${DATA_ROOT}/images" -type f | wc -l
