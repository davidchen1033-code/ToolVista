#!/usr/bin/env bash
set -Eeuo pipefail
TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
# Compatibility entry: use --checkpoint/--data-dir/--output/--mode, as train_agent_rl.sh.
# GiGPO now shares pinned upstream verl with GRPO; no verl-agent fork is imported.
exec bash "${TOOLVISTA_ROOT}/scripts/autodl/train_agent_rl.sh" "$@" --algorithm "${RUN_MODE:-grpo}"
