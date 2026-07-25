#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the evaluated terminal-fix checkpoint}"
RUN_ID="${RUN_ID:?Set a unique terminal-fix C3 RUN_ID}"
HISTORY_SUMMARY="${HISTORY_SUMMARY:-${TOOLVISTA_ROOT}/runs/c2_datafixed_light_seed42/summary.json}"
CORE_GATE="${CORE_GATE:-${CHECKPOINT}/toolvista_c3_datafixed_agentretention_terminalfix_core_gate.json}"
FINAL_GATE="${FINAL_GATE:-${CHECKPOINT}/toolvista_c3_datafixed_agentretention_terminalfix_retention_gate.json}"

export TOOLVISTA_ROOT PYTHON_BIN CHECKPOINT RUN_ID HISTORY_SUMMARY
export C3_DATAFIXED_ADAPTER_CHECKPOINT="${CHECKPOINT}"
export STAGE="${RUN_STAGE:-c3-router-datafixed-withnegative-terminal}"
export EVAL_SUITE="c3_datafixed_agentretention_light"
export RETENTION_VARIANT="datafixed-agentretention-terminalfix"
export ROUTER_BENCHMARK_ID="mllm_router_datafixed_agentretention"
export TOOLVQA_ANSWER_MINIMUM="${TOOLVQA_ANSWER_MINIMUM:-60.0}"
export BFCL_IRRELEVANCE_MINIMUM="${BFCL_IRRELEVANCE_MINIMUM:-0.20}"
export ROUTER_VIDEO_MINIMUM="${ROUTER_VIDEO_MINIMUM:-0.25}"
export ROUTER_INVALID_MAXIMUM="${ROUTER_INVALID_MAXIMUM:-0.05}"
export WANDB_GROUP="${WANDB_GROUP:-c3-router-datafixed-agentretention-terminalfix}"
export C3_DATAFIXED_RETENTION_GATE="${CORE_GATE}"

bash "${TOOLVISTA_ROOT}/scripts/autodl/lib/evaluate_c3_datafixed.sh" "$@"

"${PYTHON_BIN}" "${TOOLVISTA_ROOT}/scripts/check_c3_terminalfix_retention.py" \
  --candidate-summary "${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json" \
  --baseline-summary "${HISTORY_SUMMARY}" \
  --authorized-checkpoint "${CHECKPOINT}" \
  --output "${FINAL_GATE}" \
  --toolvqa-answer-minimum "${TOOLVQA_ANSWER_MINIMUM}" \
  --bfcl-irrelevance-minimum "${BFCL_IRRELEVANCE_MINIMUM}" \
  --router-video-minimum "${ROUTER_VIDEO_MINIMUM}" \
  --router-invalid-maximum "${ROUTER_INVALID_MAXIMUM}" \
  --terminal-emission-minimum "${TERMINAL_EMISSION_MINIMUM:-80.0}" \
  --terminal-format-error-maximum "${TERMINAL_FORMAT_ERROR_MAXIMUM:-20.0}" \
  --conditional-answer-minimum "${CONDITIONAL_ANSWER_MINIMUM:-75.0}"

echo "Terminal-fix gate: ${FINAL_GATE}"
