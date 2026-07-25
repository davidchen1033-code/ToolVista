#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
CHECKPOINT="${C3_DATAFIXED_CHECKPOINT:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_datafixed-agentretention-terminalfix_sft/terminalfix20260723T100256Z/v0-20260723-180402/checkpoint-1700}"
GATE="${C3_DATAFIXED_RETENTION_GATE:-${CHECKPOINT}/toolvista_c3_datafixed_agentretention_terminalfix_retention_gate.json}"
SESSION="${SESSION:-c4-datafixed-withnegative-terminalfix-dapo}"
RUN_TAG="${RUN_TAG:-terminalfix$(date -u +%Y%m%dT%H%M%SZ)}"

export C3_DATAFIXED_CHECKPOINT="${CHECKPOINT}"
export C3_DATAFIXED_RETENTION_GATE="${GATE}"
export EXPECTED_C3_VARIANT="datafixed-agentretention-terminalfix"
export SESSION RUN_TAG
export ATTACH="${ATTACH:-false}"

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/18_launch_visual_rft_dapo_datafixed_withnegative_tmux.sh"
