#!/usr/bin/env bash
set -euo pipefail

export MAX_ITEMS="${MAX_ITEMS:-50}"
export EPOCHS="${EPOCHS:-1}"
export GRAD_ACC="${GRAD_ACC:-8}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-512}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/ToolVista/output/toolvqa_smoke}"

bash "${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}/scripts/autodl/02_export_toolvqa_ms_swift.sh"
bash "${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}/scripts/autodl/03_train_toolvqa_sft.sh"
