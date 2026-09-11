#!/usr/bin/env bash
set -Eeuo pipefail
# Legacy filename: GRPO/GiGPO share upstream verl; only MiniWoB assets are needed.
MINIWOB_ROOT="${MINIWOB_ROOT:-/root/autodl-tmp/miniwob-plusplus}"
MINIWOB_REF="${MINIWOB_REF:-33c3b4ddef8c6eb67c57a29663d844b1eda7e614}"
if [[ "${PRINT_ONLY:-true}" == true ]]; then
  echo "No download performed (PRINT_ONLY=true)."
  echo "miniwob=${MINIWOB_ROOT}@${MINIWOB_REF}"
  exit 0
fi
if [[ ! -e "${MINIWOB_ROOT}" ]]; then
  git clone --no-checkout --filter=blob:none https://github.com/Farama-Foundation/miniwob-plusplus.git "${MINIWOB_ROOT}"
else
  test "$(git -C "${MINIWOB_ROOT}" rev-parse HEAD)" = "${MINIWOB_REF}"
  test -z "$(git -C "${MINIWOB_ROOT}" status --porcelain)"
fi
git -C "${MINIWOB_ROOT}" fetch --depth 1 origin "${MINIWOB_REF}"
git -C "${MINIWOB_ROOT}" checkout --detach "${MINIWOB_REF}"
test -f "${MINIWOB_ROOT}/miniwob/html/miniwob/click-test.html"
