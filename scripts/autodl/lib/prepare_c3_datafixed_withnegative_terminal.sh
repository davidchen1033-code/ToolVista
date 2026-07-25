#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
OVERWRITE="${OVERWRITE:-false}"

test -d "${TOOLVISTA_ROOT}"
test -x "${PYTHON_BIN}"
cd "${TOOLVISTA_ROOT}"

overwrite_args=()
case "${OVERWRITE}" in
  true) overwrite_args+=(--overwrite) ;;
  false) ;;
  *) echo "OVERWRITE must be true or false" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" scripts/prepare_c3_datafixed_terminalfix.py "${overwrite_args[@]}" "$@"
"${PYTHON_BIN}" scripts/prepare_c3_datafixed_terminalfix.py --verify-only "$@"

echo "C3 datafixed-agentretention-terminalfix preparation and verification passed."
