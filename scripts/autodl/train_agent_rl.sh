#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
AGENTRL_PYTHON="${AGENTRL_PYTHON:-/root/autodl-tmp/venvs/toolvista-agentrl/bin/python}"
export PYTHONPATH="${TOOLVISTA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${TOOLVISTA_ROOT}"
exec "${AGENTRL_PYTHON}" -m toolvista.agentrl.launch "$@"
