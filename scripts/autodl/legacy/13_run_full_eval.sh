#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
SUITE="${SUITE:-c0_full}"
export SUITE

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/12_run_light_eval.sh"
