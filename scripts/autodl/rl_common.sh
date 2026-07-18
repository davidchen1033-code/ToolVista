#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
C3_CHECKPOINT="${C3_CHECKPOINT:?Set C3_CHECKPOINT to the completed C3 SFT checkpoint-* directory}"
DATASET="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft.py}"
LOSS_TYPE="${LOSS_TYPE:?LOSS_TYPE must be set by the RL branch launcher}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set by the RL branch launcher}"
STAGE="${STAGE:?STAGE must be set by the RL branch launcher}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}"

SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
GRAD_ACC="${GRAD_ACC:-4}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-${GRAD_ACC}}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
BETA="${BETA:-0.04}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_LENGTH}}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
VALIDATE_ROWS="${VALIDATE_ROWS:-100}"
REWARD_VALIDATE_ROWS="${REWARD_VALIDATE_ROWS:-0}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-false}"
REPORT_TO="${REPORT_TO:-wandb}"
REWARD_FUNCS="${REWARD_FUNCS:-visual_rft format}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl}"
WANDB_GROUP="${WANDB_GROUP:-rl-ab}"
WANDB_NAME="${WANDB_NAME:-${STAGE}-seed${SEED}}"

test -d "${TOOLVISTA_ROOT}"
test -d "${MS_SWIFT_ROOT}"
test -e "${BASE_MODEL}"
test -d "${C3_CHECKPOINT}"
test -s "${DATASET}"
test -f "${REWARD_PLUGIN}"
if ! "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is required for RL rollout. Install a version compatible with this AutoDL CUDA/PyTorch image." >&2
  echo "The local ms-swift documentation recommends vllm>=0.11.0 for Qwen3-VL." >&2
  exit 2
fi

if ! declare -p ALGORITHM_ARGS >/dev/null 2>&1; then
  ALGORITHM_ARGS=()
fi
read -r -a reward_func_args <<< "${REWARD_FUNCS}"
if [[ "${#reward_func_args[@]}" -eq 0 ]]; then
  echo "REWARD_FUNCS must contain at least one ms-swift reward name." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export MAX_PIXELS
export WANDB_PROJECT
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${TOOLVISTA_ROOT}:${PYTHONPATH:-}"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/validate_visual_rft_rl_dataset.py "${DATASET}" --max-rows "${REWARD_VALIDATE_ROWS}"
"${PYTHON_BIN}" scripts/validate_ms_swift_dataset.py "${DATASET}" --max-rows "${VALIDATE_ROWS}"

cd "${MS_SWIFT_ROOT}"
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${BASE_MODEL}" \
  --adapters "${C3_CHECKPOINT}" \
  --ref_adapters "${C3_CHECKPOINT}" \
  --load_args false \
  --dataset "${DATASET}" \
  --external_plugins "${REWARD_PLUGIN}" \
  --reward_funcs "${reward_func_args[@]}" \
  --loss_type "${LOSS_TYPE}" \
  "${ALGORITHM_ARGS[@]}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "${TORCH_DTYPE}" \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --sleep_level 1 \
  --offload_model "${OFFLOAD_MODEL}" \
  --offload_optimizer "${OFFLOAD_OPTIMIZER}" \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --steps_per_generation "${STEPS_PER_GENERATION}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --beta "${BETA}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --log_completions "${LOG_COMPLETIONS}" \
  --log_entropy true \
  --log_rollout_offpolicy_metrics true \
  --output_dir "${OUTPUT_DIR}" \
  --report_to "${REPORT_TO}"
