#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${MODEL_DIR}"

hf download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}"

test -s "${MODEL_DIR}/config.json"

echo "Model downloaded to ${MODEL_DIR}"
du -sh "${MODEL_DIR}"
