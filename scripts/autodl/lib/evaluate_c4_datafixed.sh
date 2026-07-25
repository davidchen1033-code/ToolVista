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

fail() {
  echo "$*" >&2
  exit 2
}

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
}

has_withnegative_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed-withnegative([/_.-]|$) ]]
}

require_datafixed_path() {
  local label="$1"
  local lexical="$2"
  local resolved="$3"
  has_datafixed_marker "${lexical}" || fail "${label} lexical path must contain a safe 'datafixed' marker: ${lexical}"
  has_datafixed_marker "${resolved}" || fail "${label} resolves outside the datafixed namespace: ${lexical} -> ${resolved}"
}

require_variant_value() {
  local label="$1"
  local value="$2"
  case "${EXPERIMENT_VARIANT}" in
    datafixed-withnegative)
      has_withnegative_marker "${value}" || fail "${label} must contain the exact 'datafixed-withnegative' marker: ${value}"
      ;;
    datafixed)
      has_datafixed_marker "${value}" || fail "${label} must contain a safe 'datafixed' marker: ${value}"
      if has_withnegative_marker "${value}"; then
        fail "${label} belongs to datafixed-withnegative, not the datafixed control: ${value}"
      fi
      ;;
  esac
}

TOOLVISTA_ROOT="$(resolved_path "${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}")"
PYTHON_BIN="$(lexical_path "${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}")"
SWIFT_BIN="$(lexical_path "${SWIFT_BIN:-/root/autodl-tmp/venvs/toolvista/bin/swift}")"
CHECKPOINT_LEXICAL="$(lexical_path "${CHECKPOINT:?Set CHECKPOINT to the served C4 datafixed RL checkpoint or merged model}")"
CHECKPOINT="$(resolved_path "${CHECKPOINT_LEXICAL}")"
RUN_ID="${RUN_ID:?Set a unique C4 datafixed RUN_ID}"
API_BASE="${API_BASE:-http://127.0.0.1:8010/v1}"
API_MODEL="${API_MODEL:?Set API_MODEL to the C4 datafixed alias exposed by the API}"
EVAL_URL="${EVAL_URL:-${API_BASE}}"
JUDGE_API_BASE="${JUDGE_API_BASE:?Set JUDGE_API_BASE to the fixed C0-C4 judge endpoint}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to the fixed C0-C4 judge model ID}"
SEED="${SEED:-42}"
RESUME="${RESUME:-false}"
ALGORITHM="${ALGORITHM:-}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TIMEOUT="${TIMEOUT:-120}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl-eval}"

HISTORY_SUMMARY_LEXICAL="$(lexical_path "${HISTORY_SUMMARY:?Set HISTORY_SUMMARY to the passed C3 datafixed light summary}")"
HISTORY_SUMMARY="$(resolved_path "${HISTORY_SUMMARY_LEXICAL}")"
HISTORY_MANIFEST_LEXICAL="$(dirname "${HISTORY_SUMMARY_LEXICAL}")/manifest.json"
HISTORY_MANIFEST="$(resolved_path "${HISTORY_MANIFEST_LEXICAL}")"

BASELINE_SUMMARY_RAW="${BASELINE_SUMMARY:-}"
BASELINE_SUMMARY=""
if [[ -n "${BASELINE_SUMMARY_RAW}" ]]; then
  BASELINE_SUMMARY="$(resolved_path "${BASELINE_SUMMARY_RAW}")"
fi

LMUData="$(resolved_path "${LMUData:-/root/autodl-tmp/cache/vlmevalkit}")"
HF_HOME="$(resolved_path "${HF_HOME:-/root/autodl-tmp/cache/huggingface}")"
TOOLVQA_PYTHON="$(lexical_path "${TOOLVQA_PYTHON:-/root/autodl-tmp/venvs/toolvqa/bin/python}")"
BFCL_PYTHON="$(lexical_path "${BFCL_PYTHON:-/root/autodl-tmp/venvs/bfcl/bin/python}")"
MS_SWIFT_ROOT="$(resolved_path "${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}")"
DATAFIXED_BENCHMARK_CONFIG_LEXICAL="$(lexical_path "${DATAFIXED_BENCHMARK_CONFIG:-${TOOLVISTA_ROOT}/configs/evaluation/benchmarks_datafixed.yaml}")"
DATAFIXED_BENCHMARK_CONFIG="$(resolved_path "${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}")"
DATAFIXED_SUITE_CONFIG_LEXICAL="$(lexical_path "${DATAFIXED_SUITE_CONFIG:-${TOOLVISTA_ROOT}/configs/evaluation/suites_datafixed.yaml}")"
DATAFIXED_SUITE_CONFIG="$(resolved_path "${DATAFIXED_SUITE_CONFIG_LEXICAL}")"

SUITE="${SUITE:-c4_datafixed_light}"
case "${SUITE}" in
  c4_datafixed_light)
    expected_history_router="mllm_router_datafixed"
    suite_scope=light
    ;;
  c4_datafixed_full)
    expected_history_router="mllm_router_datafixed"
    suite_scope=full
    ;;
  c4_datafixed_agentretention_light)
    expected_history_router="mllm_router_datafixed_agentretention"
    suite_scope=light
    ;;
  c4_datafixed_agentretention_full)
    expected_history_router="mllm_router_datafixed_agentretention"
    suite_scope=full
    ;;
  *)
    fail "Unsupported C4 datafixed suite: ${SUITE}"
    ;;
esac
EVALUATION_SCOPE="${EVALUATION_SCOPE:-${suite_scope}}"
[[ "${EVALUATION_SCOPE}" == "${suite_scope}" ]] || fail "EVALUATION_SCOPE does not match SUITE: ${EVALUATION_SCOPE} != ${suite_scope}"
HISTORY_ROUTER_BENCHMARK_ID="${HISTORY_ROUTER_BENCHMARK_ID:-${expected_history_router}}"
[[ "${HISTORY_ROUTER_BENCHMARK_ID}" == "${expected_history_router}" ]] || fail "HISTORY_ROUTER_BENCHMARK_ID does not match SUITE: ${HISTORY_ROUTER_BENCHMARK_ID}"

requested_variant="${EXPERIMENT_VARIANT:-auto}"
case "${requested_variant}" in
  auto)
    if has_withnegative_marker "${CHECKPOINT_LEXICAL}" && has_withnegative_marker "${CHECKPOINT}"; then
      EXPERIMENT_VARIANT="datafixed-withnegative"
    else
      EXPERIMENT_VARIANT="datafixed"
    fi
    ;;
  datafixed|datafixed-withnegative)
    EXPERIMENT_VARIANT="${requested_variant}"
    ;;
  *) fail "EXPERIMENT_VARIANT must be auto, datafixed or datafixed-withnegative: ${requested_variant}" ;;
esac

STAGE="${STAGE:-c4-visualrft-${EXPERIMENT_VARIANT}}"
WANDB_GROUP="${WANDB_GROUP:-c4-visualrft-${EXPERIMENT_VARIANT}}"

[[ -d "${TOOLVISTA_ROOT}" ]] || fail "TOOLVISTA_ROOT is not a directory: ${TOOLVISTA_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -x "${SWIFT_BIN}" ]] || fail "SWIFT_BIN is not executable: ${SWIFT_BIN}"
[[ -e "${CHECKPOINT}" ]] || fail "CHECKPOINT does not exist: ${CHECKPOINT}"
[[ -s "${HISTORY_SUMMARY}" ]] || fail "HISTORY_SUMMARY is missing or empty: ${HISTORY_SUMMARY}"
[[ -s "${HISTORY_MANIFEST}" ]] || fail "C3 history manifest is missing or empty next to HISTORY_SUMMARY: ${HISTORY_MANIFEST}"
[[ -s "${DATAFIXED_BENCHMARK_CONFIG}" ]] || fail "Datafixed benchmark config is missing or empty: ${DATAFIXED_BENCHMARK_CONFIG}"
[[ -s "${DATAFIXED_SUITE_CONFIG}" ]] || fail "Datafixed suite config is missing or empty: ${DATAFIXED_SUITE_CONFIG}"
[[ -d "${MS_SWIFT_ROOT}" ]] || fail "MS_SWIFT_ROOT is not a directory: ${MS_SWIFT_ROOT}"

require_datafixed_path CHECKPOINT "${CHECKPOINT_LEXICAL}" "${CHECKPOINT}"
require_datafixed_path HISTORY_SUMMARY "${HISTORY_SUMMARY_LEXICAL}" "${HISTORY_SUMMARY}"
require_datafixed_path HISTORY_MANIFEST "${HISTORY_MANIFEST_LEXICAL}" "${HISTORY_MANIFEST}"
require_datafixed_path DATAFIXED_BENCHMARK_CONFIG "${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}" "${DATAFIXED_BENCHMARK_CONFIG}"
require_datafixed_path DATAFIXED_SUITE_CONFIG "${DATAFIXED_SUITE_CONFIG_LEXICAL}" "${DATAFIXED_SUITE_CONFIG}"

[[ "$(basename "${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}")" == "benchmarks_datafixed.yaml" ]] || \
  fail "DATAFIXED_BENCHMARK_CONFIG must name benchmarks_datafixed.yaml: ${DATAFIXED_BENCHMARK_CONFIG_LEXICAL}"
[[ "$(basename "${DATAFIXED_SUITE_CONFIG_LEXICAL}")" == "suites_datafixed.yaml" ]] || \
  fail "DATAFIXED_SUITE_CONFIG must name suites_datafixed.yaml: ${DATAFIXED_SUITE_CONFIG_LEXICAL}"

if [[ "${RUN_ID}" == */* || "${RUN_ID}" == "." || "${RUN_ID}" == ".." || ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail "RUN_ID must be one path-safe datafixed component: ${RUN_ID}"
fi
if [[ "${STAGE,,}" != c4* ]]; then
  fail "STAGE must start with 'c4': ${STAGE}"
fi
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer: ${SEED}"

require_variant_value CHECKPOINT "${CHECKPOINT_LEXICAL}"
require_variant_value CHECKPOINT "${CHECKPOINT}"
require_variant_value RUN_ID "${RUN_ID}"
require_variant_value STAGE "${STAGE}"
require_variant_value API_MODEL "${API_MODEL}"
require_variant_value WANDB_GROUP "${WANDB_GROUP}"

run_root="$(realpath -m -- "${TOOLVISTA_ROOT}/runs")"
run_dir="$(realpath -m -- "${TOOLVISTA_ROOT}/runs/${RUN_ID}")"
[[ "$(dirname "${run_dir}")" == "${run_root}" ]] || fail "RUN_ID must resolve directly under ${run_root}: ${RUN_ID}"

resume_args=()
case "${RESUME}" in
  true) resume_args+=(--resume) ;;
  false)
    [[ ! -e "${run_dir}" ]] || fail "RUN_ID already exists; set RESUME=true to reuse it: ${run_dir}"
    ;;
  *) fail "RESUME must be true or false: ${RESUME}" ;;
esac

case "${WANDB_ENABLED}" in
  true) wandb_args=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}") ;;
  false) wandb_args=() ;;
  *) fail "WANDB_ENABLED must be true or false: ${WANDB_ENABLED}" ;;
esac

algorithm_args=()
if [[ -n "${ALGORITHM}" ]]; then
  algorithm_args+=(--algorithm "${ALGORITHM}")
fi

comparison_args=()
if [[ "${suite_scope}" == light ]]; then
  comparison_args+=(--history-summary "${HISTORY_SUMMARY}")
fi
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  [[ -s "${BASELINE_SUMMARY}" ]] || fail "BASELINE_SUMMARY is missing or empty: ${BASELINE_SUMMARY}"
  comparison_args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi

# The C3 history manifest is the source of truth for the fixed judge protocol.
# This preflight is read-only and deliberately has no retention-gate output.
JUDGE_API_BASE="$(
  "${PYTHON_BIN}" - "${HISTORY_SUMMARY}" "${HISTORY_MANIFEST}" "${JUDGE_API_BASE}" "${JUDGE_MODEL}" "${SEED}" "${HISTORY_ROUTER_BENCHMARK_ID}" <<'PY'
import json
import re
import sys
import urllib.parse


def load_mapping(path, label):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return value


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


def has_datafixed_marker(value):
    return isinstance(value, str) and re.search(
        r"(^|[/_.-])datafixed([/_.-]|$)", value, flags=re.IGNORECASE
    ) is not None


(
    summary_path,
    manifest_path,
    requested_endpoint,
    requested_model,
    seed_text,
    router_benchmark_id,
) = sys.argv[1:]
EXPECTED_BENCHMARKS = [
    "xlam_function_calling",
    "toolvqa_step_light",
    "bfcl_non_live_light",
    router_benchmark_id,
    "qwen3vl_general_light",
    "visual_rft_coco_light",
]
summary = load_mapping(summary_path, "HISTORY_SUMMARY")
manifest = load_mapping(manifest_path, "C3 history manifest")

if summary.get("status") != "passed":
    raise SystemExit("HISTORY_SUMMARY status must be passed")
if summary.get("run_id") != manifest.get("run_id"):
    raise SystemExit("C3 history summary/manifest run_id mismatch")
if summary.get("checkpoint") != manifest.get("checkpoint"):
    raise SystemExit("C3 history summary/manifest checkpoint mismatch")
if not has_datafixed_marker(manifest.get("run_id")):
    raise SystemExit("C3 history run_id must belong to the datafixed track")
if not has_datafixed_marker(manifest.get("checkpoint")):
    raise SystemExit("C3 history checkpoint must belong to the datafixed track")
stage = manifest.get("stage")
if not isinstance(stage, str) or not stage.lower().startswith("c3") or not has_datafixed_marker(stage):
    raise SystemExit("C3 history manifest stage must start with c3 and contain datafixed")
if manifest.get("seed") != int(seed_text):
    raise SystemExit("C3 history seed does not match the requested C4 evaluation seed")
if manifest.get("benchmarks") != EXPECTED_BENCHMARKS:
    raise SystemExit("C3 history manifest is not the datafixed light benchmark suite")
summary_benchmarks = summary.get("benchmarks")
if not isinstance(summary_benchmarks, dict) or list(summary_benchmarks) != EXPECTED_BENCHMARKS:
    raise SystemExit("C3 history summary is not the complete datafixed light benchmark suite")

evaluator = manifest.get("evaluator")
if not isinstance(evaluator, dict):
    raise SystemExit("C3 history manifest is missing evaluator protocol")
if evaluator.get("policy") != "fixed_across_c0_c4":
    raise SystemExit("C3 history evaluator policy must be fixed_across_c0_c4")
baseline_endpoint = safe_endpoint(evaluator.get("judge_api_base"))
candidate_endpoint = safe_endpoint(requested_endpoint)
if baseline_endpoint is None or candidate_endpoint is None:
    raise SystemExit("fixed judge endpoint must be a valid absolute URL")
if baseline_endpoint.rstrip("/") != candidate_endpoint.rstrip("/"):
    raise SystemExit("JUDGE_API_BASE does not match the C3 history evaluator endpoint")
baseline_model = evaluator.get("judge_model")
if not isinstance(baseline_model, str) or not baseline_model.strip():
    raise SystemExit("C3 history evaluator judge_model is missing")
if baseline_model != requested_model:
    raise SystemExit("JUDGE_MODEL does not match the C3 history evaluator judge_model")
print(baseline_endpoint)
PY
)"

[[ -x "${TOOLVQA_PYTHON}" ]] || fail "TOOLVQA_PYTHON is not executable: ${TOOLVQA_PYTHON}"
"${TOOLVQA_PYTHON}" -c "from lagent.agents import ReAct; from opencompass.datasets.gta_bench import GTABenchEvaluator; from opencompass.models.lagent import LagentAgentVisual"
[[ -x "${BFCL_PYTHON}" ]] || fail "BFCL_PYTHON is not executable: ${BFCL_PYTHON}"
"${BFCL_PYTHON}" -c "import bfcl_eval"
mkdir -p "${LMUData}" "${HF_HOME}"
export LMUData HF_HOME TOOLVQA_PYTHON BFCL_PYTHON
export PATH="$(dirname "${SWIFT_BIN}"):${PATH}"

CHECK_ONLY=true \
MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${TOOLVISTA_ROOT}/scripts/autodl/patch_msswift_vlmeval.sh"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/evaluate.py \
  --config "${DATAFIXED_BENCHMARK_CONFIG}" \
  --suite-config "${DATAFIXED_SUITE_CONFIG}" \
  --suite "${SUITE}" \
  --checkpoint "${CHECKPOINT}" \
  --run-id "${RUN_ID}" \
  --stage "${STAGE}" \
  --api-base "${API_BASE}" \
  --api-model "${API_MODEL}" \
  --infer-backend "${INFER_BACKEND}" \
  --eval-url "${EVAL_URL}" \
  --judge-api-base "${JUDGE_API_BASE}" \
  --judge-model "${JUDGE_MODEL}" \
  --swift-executable "${SWIFT_BIN}" \
  --timeout "${TIMEOUT}" \
  --seed "${SEED}" \
  "${resume_args[@]}" \
  "${algorithm_args[@]}" \
  "${comparison_args[@]}" \
  "${wandb_args[@]}"
