#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
HF_BIN="${HF_BIN:-/root/autodl-tmp/venvs/toolvista/bin/hf}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

test -x "${HF_BIN}"
mkdir -p "${MODEL_DIR}"

"${HF_BIN}" download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}"

test -s "${MODEL_DIR}/config.json"

echo "Model downloaded to ${MODEL_DIR}"
du -sh "${MODEL_DIR}"
