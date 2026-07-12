#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -U huggingface_hub pyyaml pillow tensorboard

if [[ ! -d "${MS_SWIFT_ROOT}" ]]; then
  git clone https://github.com/modelscope/ms-swift.git "${MS_SWIFT_ROOT}"
fi

"${PYTHON_BIN}" -m pip install -e "${MS_SWIFT_ROOT}"
"${PYTHON_BIN}" -m pip install -U "transformers>=4.57" "qwen_vl_utils>=0.0.14" decord

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" -m py_compile \
  toolvista/adapters/toolvqa.py \
  toolvista/exporters/ms_swift.py \
  scripts/export_toolvqa.py

echo "Environment ready."
echo "ToolVista: ${TOOLVISTA_ROOT}"
echo "ms-swift: ${MS_SWIFT_ROOT}"
