#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_VLLM="${INSTALL_VLLM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -U huggingface_hub pyyaml pillow tensorboard wandb

if [[ ! -d "${MS_SWIFT_ROOT}" ]]; then
  git clone https://github.com/modelscope/ms-swift.git "${MS_SWIFT_ROOT}"
fi

"${PYTHON_BIN}" -m pip install -e "${MS_SWIFT_ROOT}"
"${PYTHON_BIN}" -m pip install -U "transformers>=4.57" "qwen_vl_utils>=0.0.14" decord

if [[ "${INSTALL_VLLM}" == "true" ]]; then
  "${PYTHON_BIN}" -m pip install -U "vllm>=0.11.0"
elif ! "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed. SFT is ready, but RL and vLLM evaluation require it."
  echo "Re-run with INSTALL_VLLM=true after checking the CUDA/PyTorch compatibility of the AutoDL image."
fi

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" -m py_compile \
  scripts/build_sft_replay.py \
  scripts/validate_visual_rft_rl_dataset.py \
  toolvista/adapters/toolvqa.py \
  toolvista/adapters/visual_rft.py \
  toolvista/exporters/ms_swift.py \
  toolvista/exporters/ms_swift_grpo.py \
  toolvista/rewards/visual_rft.py \
  scripts/export_toolvqa.py \
  scripts/export_visual_rft.py

swift --help >/dev/null

echo "Environment ready."
echo "ToolVista: ${TOOLVISTA_ROOT}"
echo "ms-swift: ${MS_SWIFT_ROOT}"
