#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
TOOLVQA_ROOT="${TOOLVQA_ROOT:-/root/autodl-tmp/ToolVQA-release}"
VISUAL_RFT_ROOT="${VISUAL_RFT_ROOT:-/root/autodl-tmp/Visual-RFT}"
COCO_ROOT="${COCO_ROOT:-/root/autodl-tmp/datasets/COCO2017}"
LMUData="${LMUData:-/root/autodl-tmp/cache/vlmevalkit}"
BFCL_PYTHON="${BFCL_PYTHON:-/root/autodl-tmp/venvs/bfcl/bin/python}"
SEED="${SEED:-42}"
XLAM_EVAL_FRACTION="${XLAM_EVAL_FRACTION:-0.05}"
MLLM_EVAL_FRACTION="${MLLM_EVAL_FRACTION:-0.20}"
LIGHT_LIMIT="${LIGHT_LIMIT:-200}"

test -d "${TOOLVISTA_ROOT}"
cd "${TOOLVISTA_ROOT}"

"${PYTHON_BIN}" scripts/prepare_training_and_eval_data.py \
  --config configs/autodl.yaml \
  --seed "${SEED}" \
  --xlam-eval-fraction "${XLAM_EVAL_FRACTION}" \
  --mllm-eval-fraction "${MLLM_EVAL_FRACTION}" \
  --light-limit "${LIGHT_LIMIT}" \
  --mllm-file-keys text image video \
  --overwrite

"${PYTHON_BIN}" scripts/prepare_c3_direct_answer_retention.py \
  --source data/exports/msswift_sft/toolvqa_train.jsonl \
  --output data/exports/msswift_sft/toolvqa_direct_answer_retention_train.jsonl \
  --max-rows 12000 \
  --seed "${SEED}" \
  --overwrite

"$PYTHON_BIN" scripts/prepare_visual_rft_coco_eval.py \
  --annotations "$COCO_ROOT/annotations/instances_val2017.json" \
  --images-dir "$COCO_ROOT/val2017" \
  --candidate-map "$VISUAL_RFT_ROOT/coco_evaluation/exist_map_coco_Qwen2_vl_2B_baseline.json" \
  --seed "$SEED" \
  --light-query-limit "$LIGHT_LIMIT" \
  --overwrite

"${BFCL_PYTHON}" scripts/prepare_bfcl_eval.py \
  --seed "${SEED}" \
  --light-limit "${LIGHT_LIMIT}" \
  --overwrite

"${PYTHON_BIN}" scripts/prefetch_vlmevalkit_data.py \
  --cache-dir "${LMUData}"

"${PYTHON_BIN}" scripts/check_c0_readiness.py \
  --root "${TOOLVISTA_ROOT}" \
  --model "${MODEL}" \
  --toolvqa-root "${TOOLVQA_ROOT}" \
  --visual-rft-root "${VISUAL_RFT_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --lmu-data "${LMUData}" \
  --bfcl-python "${BFCL_PYTHON}"
