#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
DATASET="${DATASET:?DATASET must point to a non-empty ms-swift SFT JSONL file}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set by the stage launcher}"
STAGE="${STAGE:?STAGE must be set by the stage launcher}"
PREVIOUS_CHECKPOINT="${PREVIOUS_CHECKPOINT:-}"

SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
GRAD_ACC="${GRAD_ACC:-16}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.02}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
VALIDATE_ROWS="${VALIDATE_ROWS:-100}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl}"
WANDB_GROUP="${WANDB_GROUP:-${STAGE}}"
WANDB_NAME="${WANDB_NAME:-${STAGE}-seed${SEED}}"

test -d "${TOOLVISTA_ROOT}"
test -d "${MS_SWIFT_ROOT}"
test -e "${BASE_MODEL}"
test -s "${DATASET}"
if [[ -n "${PREVIOUS_CHECKPOINT}" ]]; then
  test -d "${PREVIOUS_CHECKPOINT}"
fi

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export IMAGE_MAX_TOKEN_NUM
export WANDB_PROJECT
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${TOOLVISTA_ROOT}:${PYTHONPATH:-}"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/validate_ms_swift_dataset.py "${DATASET}" --max-rows "${VALIDATE_ROWS}"

resume_args=()
if [[ -n "${PREVIOUS_CHECKPOINT}" ]]; then
  resume_args=(
    --resume_from_checkpoint "${PREVIOUS_CHECKPOINT}"
    --resume_only_model true
  )
fi

cd "${MS_SWIFT_ROOT}"
"${SWIFT_BIN}" sft \
  --model "${BASE_MODEL}" \
  --dataset "${DATASET}" \
  --load_args false \
  "${resume_args[@]}" \
  --agent_template hermes \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --max_length "${MAX_LENGTH}" \
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to "${REPORT_TO}"
