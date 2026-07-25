#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the evaluated C3 datafixed-agentretention checkpoint}"
export TOOLVISTA_ROOT CHECKPOINT
export C3_DATAFIXED_ADAPTER_CHECKPOINT="${C3_DATAFIXED_ADAPTER_CHECKPOINT:-${CHECKPOINT}}"
export RUN_ID="${RUN_ID:?Set a unique c3_datafixed_agentretention RUN_ID}"
export STAGE="${STAGE:-c3-router-datafixed-agentretention}"
export EVAL_SUITE="c3_datafixed_agentretention_light"
export RETENTION_VARIANT="datafixed-agentretention"
export ROUTER_BENCHMARK_ID="mllm_router_datafixed_agentretention"
export TOOLVQA_ANSWER_MINIMUM="${TOOLVQA_ANSWER_MINIMUM:-}"
if [[ -n "${TOOLVQA_ANSWER_MINIMUM}" ]]; then
  export TOOLVQA_ANSWER_RETENTION="${TOOLVQA_ANSWER_RETENTION:-}"
else
  export TOOLVQA_ANSWER_RETENTION="${TOOLVQA_ANSWER_RETENTION:-0.80}"
fi
export BFCL_IRRELEVANCE_MINIMUM="${BFCL_IRRELEVANCE_MINIMUM:-0.20}"
export ROUTER_VIDEO_MINIMUM="${ROUTER_VIDEO_MINIMUM:-0.25}"
export ROUTER_INVALID_MAXIMUM="${ROUTER_INVALID_MAXIMUM:-0.05}"
export WANDB_GROUP="${WANDB_GROUP:-c3-router-datafixed-agentretention}"
export C3_DATAFIXED_RETENTION_GATE="${C3_DATAFIXED_RETENTION_GATE:-${C3_DATAFIXED_ADAPTER_CHECKPOINT}/toolvista_c3_datafixed_agentretention_retention_gate.json}"

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/12_run_datafixed_light_eval.sh" "$@"
