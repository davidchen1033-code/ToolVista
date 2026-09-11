#!/usr/bin/env bash
set -Eeuo pipefail
TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
exec bash "${TOOLVISTA_ROOT}/scripts/autodl/train_agent_rl.sh" "$@" --algorithm gigpo
