#!/usr/bin/env bash
set -Eeuo pipefail
TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
AGENTRL_VENV="${AGENTRL_VENV:-/root/autodl-tmp/venvs/toolvista-agentrl}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if [[ ! -x "${AGENTRL_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${AGENTRL_VENV}"
fi
"${AGENTRL_VENV}/bin/python" -m pip install --upgrade pip
"${AGENTRL_VENV}/bin/python" -m pip install -r "${TOOLVISTA_ROOT}/configs/verl/requirements-agentrl.txt"
"${AGENTRL_VENV}/bin/python" -m pip check
"${AGENTRL_VENV}/bin/python" -m playwright install --with-deps chromium
export PYTHONPATH="${TOOLVISTA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${AGENTRL_VENV}/bin/python" -c "from toolvista.agentrl.dependencies import check_versions, load_verl_api; check_versions(); load_verl_api()"
