#!/usr/bin/env bash
set -euo pipefail

MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_ONLY="${CHECK_ONLY:-false}"
TARGET="${MS_SWIFT_ROOT}/swift/pipelines/eval/eval.py"
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
MATHV_TARGET="${SITE_PACKAGES}/vlmeval/dataset/utils/mathv.py"

contains_fixed() {
  if command -v rg >/dev/null 2>&1; then
    rg -qF -- "$1" "$2"
  else
    grep -qF -- "$1" "$2"
  fi
}

count_fixed() {
  if command -v rg >/dev/null 2>&1; then
    rg -cF -- "$1" "$2" || true
  else
    grep -cF -- "$1" "$2" || true
  fi
}

test -f "${TARGET}"
test -f "${MATHV_TARGET}"
patched="'type': args.served_model_name or args.model_suffix"
original="'type': args.model_suffix"

if contains_fixed "${patched}" "${TARGET}"; then
  echo "ms-swift VLMEval candidate route: compatible"
elif [[ "$(count_fixed "${original}" "${TARGET}")" != "1" ]]; then
  echo "Unsupported ms-swift VLMEval implementation: ${TARGET}" >&2
  exit 1
elif [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "ms-swift VLMEval candidate route patch is required: ${TARGET}" >&2
  exit 1
else
  sed -i "s/'type': args\.model_suffix/'type': args.served_model_name or args.model_suffix/" "${TARGET}"
  echo "ms-swift VLMEval candidate route: patched"
fi

mathv_patched="from latex2sympy2_extended import latex2sympy"
mathv_original="    from latex2sympy2 import latex2sympy"

if contains_fixed "${mathv_patched}" "${MATHV_TARGET}"; then
  echo "VLMEvalKit MathVision latex parser: compatible"
elif [[ "$(count_fixed "${mathv_original}" "${MATHV_TARGET}")" != "1" ]]; then
  echo "Unsupported VLMEvalKit MathVision implementation: ${MATHV_TARGET}" >&2
  exit 1
elif [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "VLMEvalKit MathVision latex fallback patch is required: ${MATHV_TARGET}" >&2
  exit 1
else
  sed -i '/^    from latex2sympy2 import latex2sympy$/a\except ImportError:\n    from latex2sympy2_extended import latex2sympy' "${MATHV_TARGET}"
  echo "VLMEvalKit MathVision latex parser: patched"
fi

"${PYTHON_BIN}" -m py_compile "${TARGET}" "${MATHV_TARGET}"
