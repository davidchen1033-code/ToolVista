#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
MODEL="${MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/output/qwen3_vl_4b_toolvqa_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
GRAD_ACC="${GRAD_ACC:-16}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

test -d "${MS_SWIFT_ROOT}"
test -e "${MODEL}"
test -s "${DATASET}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export IMAGE_MAX_TOKEN_NUM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"

cd "${TOOLVISTA_ROOT}"
python scripts/validate_ms_swift_dataset.py "${DATASET}" --max-rows 100

cd "${MS_SWIFT_ROOT}"

swift sft \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --agent_template hermes \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --max_length "${MAX_LENGTH}" \
  --split_dataset_ratio 0.02 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
