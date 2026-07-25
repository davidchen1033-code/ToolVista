#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"

export ADAPTER="${ADAPTER:-${TOOLVISTA_ROOT}/outputs/rl/bnpo_datafixed-withnegative_seed42_runterminalfix-bnpo20260724T114133Z/v0-20260724-194247/checkpoint-6000}"
export MERGED="${MERGED:-${TOOLVISTA_ROOT}/outputs/final/c4-visualrft-coco-bnpo-datafixed-withnegative-terminalfix-seed42-runterminalfix-bnpo20260724T114133Z-step6000-merged}"
export HISTORY_SUMMARY="${HISTORY_SUMMARY:-${TOOLVISTA_ROOT}/runs/c3_datafixed-agentretention-terminalfix_light_seed42_step1700/summary.json}"
export BASELINE_SUMMARY="${BASELINE_SUMMARY:-${TOOLVISTA_ROOT}/runs/c4_visualrft_coco_dapo_datafixed-withnegative-terminalfix_light_seed42_step6000/summary.json}"
export SESSION="${SESSION:-c4-datafixed-withnegative-terminalfix-bnpo-benchmark}"
export RUN_ID="${RUN_ID:-c4_visualrft_coco_bnpo_datafixed-withnegative-terminalfix_light_seed42_step6000}"
export API_MODEL="${API_MODEL:-c4-bnpo-datafixed-withnegative-terminalfix-step6000}"
export ALGORITHM=bnpo
export STAGE=c4-visualrft-bnpo-datafixed-withnegative-terminalfix
export WANDB_GROUP=c4-visualrft-bnpo-datafixed-withnegative-terminalfix

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/30_launch_c4_datafixed_withnegative_terminalfix_benchmark_tmux.sh"
