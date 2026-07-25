#!/usr/bin/env bash
set -euo pipefail

CALLER_ROOT="$(pwd -P)"

lexical_path() {
  local value="$1"
  if [[ "${value}" != /* ]]; then
    value="${CALLER_ROOT}/${value}"
  fi
  realpath -m -s -- "${value}"
}

resolved_path() {
  realpath -m -- "$(lexical_path "$1")"
}

TOOLVISTA_ROOT="$(resolved_path "${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}")"
PYTHON_BIN="$(lexical_path "${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}")"
SWIFT_BIN="$(lexical_path "${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}")"
CHECKPOINT_LEXICAL="$(lexical_path "${CHECKPOINT:?Set CHECKPOINT to the served C3 datafixed adapter checkpoint}")"
CHECKPOINT="$(resolved_path "${CHECKPOINT_LEXICAL}")"
C3_DATAFIXED_ADAPTER_CHECKPOINT_LEXICAL="$(lexical_path "${C3_DATAFIXED_ADAPTER_CHECKPOINT:?Set C3_DATAFIXED_ADAPTER_CHECKPOINT to the evaluated C3 datafixed adapter}")"
C3_DATAFIXED_ADAPTER_CHECKPOINT="$(resolved_path "${C3_DATAFIXED_ADAPTER_CHECKPOINT_LEXICAL}")"
RUN_ID="${RUN_ID:?Set a unique datafixed RUN_ID}"
STAGE="${STAGE:-c3-router-datafixed-retention}"
EVAL_SUITE="${EVAL_SUITE:-c3_datafixed_light}"
RETENTION_VARIANT="${RETENTION_VARIANT:-datafixed}"
ROUTER_BENCHMARK_ID="${ROUTER_BENCHMARK_ID:-mllm_router_datafixed}"
TOOLVQA_ANSWER_MINIMUM="${TOOLVQA_ANSWER_MINIMUM:-}"
TOOLVQA_ANSWER_RETENTION="${TOOLVQA_ANSWER_RETENTION:-}"
BFCL_IRRELEVANCE_MINIMUM="${BFCL_IRRELEVANCE_MINIMUM:-}"
ROUTER_VIDEO_MINIMUM="${ROUTER_VIDEO_MINIMUM:-}"
ROUTER_INVALID_MAXIMUM="${ROUTER_INVALID_MAXIMUM:-}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
API_MODEL="${API_MODEL:?Set API_MODEL to the datafixed adapter alias exposed by the API}"
SEED="${SEED:-42}"
RESUME="${RESUME:-false}"
ALGORITHM="${ALGORITHM:-}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl-eval}"
WANDB_GROUP="${WANDB_GROUP:-c3-router-datafixed-retention}"
BASELINE_SUMMARY_RAW="${BASELINE_SUMMARY:-}"
HISTORY_SUMMARY_LEXICAL="$(lexical_path "${HISTORY_SUMMARY:?Set HISTORY_SUMMARY to the C2 datafixed-suite summary}")"
HISTORY_SUMMARY="$(resolved_path "${HISTORY_SUMMARY_LEXICAL}")"
HISTORY_MANIFEST_LEXICAL="$(dirname "${HISTORY_SUMMARY_LEXICAL}")/manifest.json"
HISTORY_MANIFEST="$(resolved_path "${HISTORY_MANIFEST_LEXICAL}")"
EVAL_URL="${EVAL_URL:-${API_BASE}}"
JUDGE_API_BASE="${JUDGE_API_BASE:?Set JUDGE_API_BASE to the fixed C0-C4 judge endpoint}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to the fixed C0-C4 judge model ID}"
LMUData="$(resolved_path "${LMUData:-/root/autodl-tmp/cache/vlmevalkit}")"
HF_HOME="$(resolved_path "${HF_HOME:-/root/autodl-tmp/cache/huggingface}")"
TOOLVQA_PYTHON="$(lexical_path "${TOOLVQA_PYTHON:-/root/autodl-tmp/venvs/toolvqa/bin/python}")"
BFCL_PYTHON="$(lexical_path "${BFCL_PYTHON:-/root/autodl-tmp/venvs/bfcl/bin/python}")"
MS_SWIFT_ROOT="$(resolved_path "${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}")"
DATAFIXED_BENCHMARK_CONFIG_LEXICAL="$(lexical_path "${DATAFIXED_BENCHMARK_CONFIG:-${TOOLVISTA_ROOT}/configs/evaluation/benchmarks_datafixed.yaml}")"
DATAFIXED_BENCHMARK_CONFIG="$(resolved_path "${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}")"
DATAFIXED_SUITE_CONFIG_LEXICAL="$(lexical_path "${DATAFIXED_SUITE_CONFIG:-${TOOLVISTA_ROOT}/configs/evaluation/suites_datafixed.yaml}")"
DATAFIXED_SUITE_CONFIG="$(resolved_path "${DATAFIXED_SUITE_CONFIG_LEXICAL}")"
C3_DATAFIXED_RETENTION_GATE_LEXICAL="$(lexical_path "${C3_DATAFIXED_RETENTION_GATE:-${C3_DATAFIXED_ADAPTER_CHECKPOINT}/toolvista_c3_datafixed_retention_gate.json}")"
C3_DATAFIXED_RETENTION_GATE="$(resolved_path "${C3_DATAFIXED_RETENTION_GATE_LEXICAL}")"

BASELINE_SUMMARY=""
if [[ -n "${BASELINE_SUMMARY_RAW}" ]]; then
  BASELINE_SUMMARY="$(resolved_path "${BASELINE_SUMMARY_RAW}")"
fi

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
}

require_datafixed_name() {
  local label="$1"
  local value="$2"
  if ! has_datafixed_marker "${value}"; then
    echo "${label} must contain a safe 'datafixed' marker: ${value}" >&2
    exit 2
  fi
}

require_datafixed_path() {
  local label="$1"
  local value="$2"
  local resolved
  if ! has_datafixed_marker "${value}"; then
    echo "${label} lexical path must contain a safe 'datafixed' marker: ${value}" >&2
    exit 2
  fi
  resolved="$(realpath -m -- "${value}")"
  if ! has_datafixed_marker "${resolved}"; then
    echo "${label} resolves outside the datafixed namespace: ${value} -> ${resolved}" >&2
    exit 2
  fi
}

require_datafixed_run_id() {
  local value="$1"
  if [[ "${value}" == */* || "${value}" == "." || "${value}" == ".." ]]; then
    echo "RUN_ID must be one datafixed path component: ${value}" >&2
    exit 2
  fi
  require_datafixed_name RUN_ID "${value}"
}

test -d "${TOOLVISTA_ROOT}"
test -d "${C3_DATAFIXED_ADAPTER_CHECKPOINT}"
test -e "${CHECKPOINT}"
test -s "${HISTORY_SUMMARY}"
test -s "${HISTORY_MANIFEST}"
test -s "${DATAFIXED_BENCHMARK_CONFIG}"
test -s "${DATAFIXED_SUITE_CONFIG}"
if [[ "$(realpath "${CHECKPOINT}")" != "$(realpath "${C3_DATAFIXED_ADAPTER_CHECKPOINT}")" ]]; then
  echo "CHECKPOINT must equal C3_DATAFIXED_ADAPTER_CHECKPOINT" >&2
  exit 2
fi
require_datafixed_path CHECKPOINT "${CHECKPOINT_LEXICAL}"
require_datafixed_path C3_DATAFIXED_ADAPTER_CHECKPOINT "${C3_DATAFIXED_ADAPTER_CHECKPOINT_LEXICAL}"
require_datafixed_path HISTORY_SUMMARY "${HISTORY_SUMMARY_LEXICAL}"
require_datafixed_path HISTORY_MANIFEST "${HISTORY_MANIFEST_LEXICAL}"
require_datafixed_path DATAFIXED_BENCHMARK_CONFIG "${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}"
require_datafixed_path DATAFIXED_SUITE_CONFIG "${DATAFIXED_SUITE_CONFIG_LEXICAL}"
require_datafixed_path C3_DATAFIXED_RETENTION_GATE "${C3_DATAFIXED_RETENTION_GATE_LEXICAL}"
require_datafixed_run_id "${RUN_ID}"
stage_lower="${STAGE,,}"
if [[ "${stage_lower}" != c3* ]]; then
  echo "STAGE must start with 'c3': ${STAGE}" >&2
  exit 2
fi
require_datafixed_name STAGE "${STAGE}"
require_datafixed_name EVAL_SUITE "${EVAL_SUITE}"
require_datafixed_name RETENTION_VARIANT "${RETENTION_VARIANT}"
require_datafixed_name ROUTER_BENCHMARK_ID "${ROUTER_BENCHMARK_ID}"
require_datafixed_name API_MODEL "${API_MODEL}"
require_datafixed_name WANDB_GROUP "${WANDB_GROUP}"
run_root_resolved="$(realpath -m -- "${TOOLVISTA_ROOT}/runs")"
run_dir_resolved="$(realpath -m -- "${TOOLVISTA_ROOT}/runs/${RUN_ID}")"
if [[ "$(dirname "${run_dir_resolved}")" != "${run_root_resolved}" ]]; then
  echo "RUN_ID must resolve directly under ${run_root_resolved}: ${RUN_ID}" >&2
  exit 2
fi
gate_resolved="$(realpath -m -- "${C3_DATAFIXED_RETENTION_GATE}")"
if [[ "$(dirname "${gate_resolved}")" != "$(realpath "${C3_DATAFIXED_ADAPTER_CHECKPOINT}")" ]]; then
  echo "C3_DATAFIXED_RETENTION_GATE must be inside the authorized datafixed checkpoint" >&2
  exit 2
fi
if [[ "$(dirname "${HISTORY_SUMMARY}")" != "$(dirname "${HISTORY_MANIFEST}")" ]]; then
  echo "HISTORY_MANIFEST must resolve next to HISTORY_SUMMARY" >&2
  exit 2
fi

# Freeze the candidate judge protocol to the baseline manifest before spending
# resources on the benchmark. Equivalent credential/query/trailing-slash forms
# are accepted, then the baseline's sanitized spelling is reused verbatim.
JUDGE_API_BASE="$(
  "${PYTHON_BIN}" -c '
import json
import sys
import urllib.parse


def safe_endpoint(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None


manifest_path, requested_endpoint, requested_model = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
evaluator = manifest.get("evaluator")
if not isinstance(evaluator, dict):
    raise SystemExit("HISTORY_MANIFEST is missing evaluator protocol")
if evaluator.get("policy") != "fixed_across_c0_c4":
    raise SystemExit("HISTORY_MANIFEST evaluator policy must be fixed_across_c0_c4")
baseline_endpoint = safe_endpoint(evaluator.get("judge_api_base"))
candidate_endpoint = safe_endpoint(requested_endpoint)
if baseline_endpoint is None or candidate_endpoint is None:
    raise SystemExit("fixed judge endpoint must be a valid absolute URL")
if baseline_endpoint.rstrip("/") != candidate_endpoint.rstrip("/"):
    raise SystemExit(
        "JUDGE_API_BASE does not match the HISTORY_MANIFEST evaluator endpoint"
    )
baseline_model = evaluator.get("judge_model")
if not isinstance(baseline_model, str) or not baseline_model.strip():
    raise SystemExit("HISTORY_MANIFEST evaluator judge_model is missing")
if baseline_model != requested_model:
    raise SystemExit("JUDGE_MODEL does not match HISTORY_MANIFEST evaluator judge_model")
print(baseline_endpoint)
' "${HISTORY_MANIFEST}" "${JUDGE_API_BASE}" "${JUDGE_MODEL}"
)"


[[ -x "${SWIFT_BIN}" ]] || {
  echo "SWIFT_BIN is not executable: ${SWIFT_BIN}" >&2
  exit 2
}
export PATH="$(dirname "${SWIFT_BIN}"):${PATH}"

test -x "${TOOLVQA_PYTHON}"
"${TOOLVQA_PYTHON}" -c "from lagent.agents import ReAct; from opencompass.datasets.gta_bench import GTABenchEvaluator; from opencompass.models.lagent import LagentAgentVisual"
test -x "${BFCL_PYTHON}"
"${BFCL_PYTHON}" -c "import bfcl_eval"
mkdir -p "${LMUData}"
export LMUData HF_HOME TOOLVQA_PYTHON BFCL_PYTHON

resume_args=()
case "${RESUME}" in
  true) resume_args+=(--resume) ;;
  false) ;;
  *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

algorithm_args=()
if [[ -n "${ALGORITHM}" ]]; then
  algorithm_args+=(--algorithm "${ALGORITHM}")
fi

strict_gate_args=()
if [[ -n "${TOOLVQA_ANSWER_RETENTION}" ]]; then
  strict_gate_args+=(--toolvqa-answer-retention "${TOOLVQA_ANSWER_RETENTION}")
fi
if [[ -n "${TOOLVQA_ANSWER_MINIMUM}" ]]; then
  strict_gate_args+=(--toolvqa-answer-minimum "${TOOLVQA_ANSWER_MINIMUM}")
fi
if [[ -n "${BFCL_IRRELEVANCE_MINIMUM}" ]]; then
  strict_gate_args+=(--bfcl-irrelevance-minimum "${BFCL_IRRELEVANCE_MINIMUM}")
fi
if [[ -n "${ROUTER_VIDEO_MINIMUM}" ]]; then
  strict_gate_args+=(--router-video-minimum "${ROUTER_VIDEO_MINIMUM}")
fi
if [[ -n "${ROUTER_INVALID_MAXIMUM}" ]]; then
  strict_gate_args+=(--router-invalid-maximum "${ROUTER_INVALID_MAXIMUM}")
fi

comparison_args=(--history-summary "${HISTORY_SUMMARY}")
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  test -s "${BASELINE_SUMMARY}"
  comparison_args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi

CHECK_ONLY=true \
MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${TOOLVISTA_ROOT}/scripts/autodl/patch_msswift_vlmeval.sh"

rm -f "${C3_DATAFIXED_RETENTION_GATE}"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/evaluate.py \
  --config "${DATAFIXED_BENCHMARK_CONFIG}" \
  --suite-config "${DATAFIXED_SUITE_CONFIG}" \
  --suite "${EVAL_SUITE}" \
  --checkpoint "${CHECKPOINT}" \
  --run-id "${RUN_ID}" \
  --stage "${STAGE}" \
  --api-base "${API_BASE}" \
  --api-model "${API_MODEL}" \
  --eval-url "${EVAL_URL}" \
  --judge-api-base "${JUDGE_API_BASE}" \
  --judge-model "${JUDGE_MODEL}" \
  --swift-executable "${SWIFT_BIN}" \
  --seed "${SEED}" \
  "${resume_args[@]}" \
  "${algorithm_args[@]}" \
  "${comparison_args[@]}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}"

"${PYTHON_BIN}" scripts/check_c3_datafixed_retention.py \
  --candidate-summary "${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json" \
  --baseline-summary "${HISTORY_SUMMARY}" \
  --authorized-checkpoint "${C3_DATAFIXED_ADAPTER_CHECKPOINT}" \
  --output "${C3_DATAFIXED_RETENTION_GATE}" \
  --variant "${RETENTION_VARIANT}" \
  --router-benchmark-id "${ROUTER_BENCHMARK_ID}" \
  "${strict_gate_args[@]}"

