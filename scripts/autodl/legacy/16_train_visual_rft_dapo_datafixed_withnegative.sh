#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C3_DATAFIXED_CHECKPOINT="${C3_DATAFIXED_CHECKPOINT:?Set C3_DATAFIXED_CHECKPOINT to the passed C3 datafixed checkpoint-* directory}"
C3_DATAFIXED_RETENTION_GATE="${C3_DATAFIXED_RETENTION_GATE:-${C3_DATAFIXED_CHECKPOINT}/toolvista_c3_datafixed_retention_gate.json}"
EXPECTED_C3_VARIANT="${EXPECTED_C3_VARIANT:-datafixed}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

DATASET="${DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl}"
DATASET_REPORT="${DATASET_REPORT:-${TOOLVISTA_ROOT}/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json}"
MEDIA_DIR="${MEDIA_DIR:-${TOOLVISTA_ROOT}/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_canonical_media}"
DATASET_BUILDER="${DATASET_BUILDER:-${TOOLVISTA_ROOT}/scripts/prepare_visual_rft_datafixed_withnegative.py}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${TOOLVISTA_ROOT}/toolvista/rewards/visual_rft_withnegative.py}"

MAX_STEPS="${MAX_STEPS:-6000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.04}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-13}"

STAGE="${STAGE:-rl-dapo-datafixed-withnegative}"
WANDB_GROUP="${WANDB_GROUP:-rl-datafixed-withnegative}"
WANDB_NAME="${WANDB_NAME:-c4-visualrft-coco-dapo-datafixed-withnegative-${RUN_TAG}-seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/dapo_datafixed-withnegative_seed${SEED}_run${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c4/c4-dapo-datafixed-withnegative-${RUN_TAG}-seed${SEED}.train.log}"
PROVENANCE_MANIFEST="${PROVENANCE_MANIFEST:-${TRAIN_LOG%.train.log}.manifest.json}"

has_datafixed_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed([/_.-]|$) ]]
}

has_withnegative_marker() {
  local value="${1,,}"
  [[ "${value}" =~ (^|[/_.-])datafixed-withnegative([/_.-]|$) ]]
}

require_withnegative_value() {
  local label="$1"
  local value="$2"
  if ! has_withnegative_marker "${value}"; then
    echo "${label} must contain the exact 'datafixed-withnegative' experiment marker: ${value}" >&2
    exit 2
  fi
}

require_withnegative_path() {
  local label="$1"
  local value="$2"
  local resolved
  require_withnegative_value "${label} lexical path" "${value}"
  resolved="$(realpath -m -- "${value}")"
  require_withnegative_value "${label} resolved path" "${resolved}"
}

require_canonical_path() {
  local label="$1"
  local value="$2"
  local resolved
  require_withnegative_path "${label}" "${value}"
  resolved="$(realpath -m -- "${value}")"
  if [[ "${value,,}" != *canonical* || "${resolved,,}" != *canonical* ]]; then
    echo "${label} must stay in the canonical datafixed-withnegative track: ${value}" >&2
    exit 2
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${label} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

test -d "${TOOLVISTA_ROOT}"
test -d "${MS_SWIFT_ROOT}"
test -d "${BASE_MODEL}"
test -x "${PYTHON_BIN}"
test -f "${DATASET_BUILDER}"
test -f "${REWARD_PLUGIN}"

toolvista_resolved="$(realpath -- "${TOOLVISTA_ROOT}")"
ms_swift_resolved="$(realpath -- "${MS_SWIFT_ROOT}")"
base_model_resolved="$(realpath -- "${BASE_MODEL}")"
checkpoint_resolved="$(realpath -- "${C3_DATAFIXED_CHECKPOINT}")"
gate_resolved="$(realpath -- "${C3_DATAFIXED_RETENTION_GATE}")"
reward_resolved="$(realpath -- "${REWARD_PLUGIN}")"
builder_resolved="$(realpath -- "${DATASET_BUILDER}")"
launcher_resolved="$(realpath -- "${BASH_SOURCE[0]}")"
dapo_launcher_resolved="$(realpath -- "${TOOLVISTA_ROOT}/scripts/autodl/10_train_visual_rft_dapo.sh")"
rl_common_resolved="$(realpath -- "${TOOLVISTA_ROOT}/scripts/autodl/rl_common.sh")"
adapter_resolved="$(realpath -- "${checkpoint_resolved}/adapter_model.safetensors")"
base_config_resolved="$(realpath -- "${base_model_resolved}/config.json")"
has_datafixed_marker "${C3_DATAFIXED_CHECKPOINT}" && has_datafixed_marker "${checkpoint_resolved}" || {
  echo "C3_DATAFIXED_CHECKPOINT must stay in the datafixed experiment track: ${C3_DATAFIXED_CHECKPOINT}" >&2
  exit 2
}
[[ "$(dirname "${gate_resolved}")" == "${checkpoint_resolved}" ]] || {
  echo "C3_DATAFIXED_RETENTION_GATE must be inside C3_DATAFIXED_CHECKPOINT" >&2
  exit 2
}

"${PYTHON_BIN}" - "${gate_resolved}" "${EXPECTED_C3_VARIANT}" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2]
if expected not in {
    "datafixed",
    "datafixed-agentretention",
    "datafixed-agentretention-terminalfix",
}:
    raise SystemExit(f"unsupported expected C3 variant: {expected}")
if gate.get("variant") != expected:
    raise SystemExit(
        f"C3 retention gate variant mismatch: {gate.get('variant')!r} != {expected!r}"
    )
PY

require_canonical_path DATASET "${DATASET}"
require_canonical_path DATASET_REPORT "${DATASET_REPORT}"
require_canonical_path MEDIA_DIR "${MEDIA_DIR}"
require_withnegative_path OUTPUT_DIR "${OUTPUT_DIR}"
require_withnegative_path TRAIN_LOG "${TRAIN_LOG}"
require_withnegative_path PROVENANCE_MANIFEST "${PROVENANCE_MANIFEST}"
require_withnegative_value STAGE "${STAGE}"
require_withnegative_value WANDB_GROUP "${WANDB_GROUP}"
require_withnegative_value WANDB_NAME "${WANDB_NAME}"
[[ "${REWARD_PLUGIN,,}" == *withnegative* && "${reward_resolved,,}" == *withnegative* ]] || {
  echo "REWARD_PLUGIN must stay in the withnegative experiment track: ${REWARD_PLUGIN}" >&2
  exit 2
}

DATASET="$(realpath -m -- "${DATASET}")"
DATASET_REPORT="$(realpath -m -- "${DATASET_REPORT}")"
MEDIA_DIR="$(realpath -m -- "${MEDIA_DIR}")"
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
TRAIN_LOG="$(realpath -m -- "${TRAIN_LOG}")"
PROVENANCE_MANIFEST="$(realpath -m -- "${PROVENANCE_MANIFEST}")"

require_positive_integer MAX_STEPS "${MAX_STEPS}"
require_positive_integer SAVE_STEPS "${SAVE_STEPS}"
require_positive_integer SAVE_TOTAL_LIMIT "${SAVE_TOTAL_LIMIT}"
if [[ "${MAX_STEPS}" != "6000" ]]; then
  echo "WARNING: MAX_STEPS=${MAX_STEPS}; the strict datafixed versus datafixed-withnegative comparison uses 6000 updates." >&2
fi

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite an existing datafixed-withnegative RL output: ${OUTPUT_DIR}" >&2
  exit 2
}
[[ ! -e "${TRAIN_LOG}" ]] || {
  echo "Refusing to overwrite an existing datafixed-withnegative training log: ${TRAIN_LOG}" >&2
  exit 2
}
[[ ! -e "${PROVENANCE_MANIFEST}" ]] || {
  echo "Refusing to overwrite an existing datafixed-withnegative provenance manifest: ${PROVENANCE_MANIFEST}" >&2
  exit 2
}

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" "${builder_resolved}" \
  --output "${DATASET}" \
  --report "${DATASET_REPORT}" \
  --media-dir "${MEDIA_DIR}" \
  --verify-only
test -s "${DATASET}"
test -s "${DATASET_REPORT}"
test -d "${MEDIA_DIR}"
test -s "${MEDIA_DIR}/manifest.jsonl"

dataset_resolved="$(realpath -- "${DATASET}")"
report_resolved="$(realpath -- "${DATASET_REPORT}")"
media_dir_resolved="$(realpath -- "${MEDIA_DIR}")"
media_manifest_resolved="$(realpath -- "${MEDIA_DIR}/manifest.jsonl")"
output_resolved="$(realpath -m -- "${OUTPUT_DIR}")"
train_log_resolved="$(realpath -m -- "${TRAIN_LOG}")"
manifest_resolved="$(realpath -m -- "${PROVENANCE_MANIFEST}")"
REWARD_FUNCS="visual_rft_withnegative visual_rft_withnegative_format"

export C3_CHECKPOINT="${checkpoint_resolved}"
export C3_RETENTION_GATE="${gate_resolved}"
export TOOLVISTA_ROOT="${toolvista_resolved}"
export MS_SWIFT_ROOT="${ms_swift_resolved}"
export BASE_MODEL="${base_model_resolved}"
export PYTHON_BIN
export DATASET="${dataset_resolved}"
export MEDIA_DIR="${media_dir_resolved}"
export REWARD_PLUGIN="${reward_resolved}"
export REWARD_FUNCS
export SEED
export MAX_STEPS
export LEARNING_RATE
export BETA
export SAVE_STEPS
export SAVE_TOTAL_LIMIT
export STAGE
export WANDB_GROUP
export WANDB_NAME
export OUTPUT_DIR="${output_resolved}"
export RUN_TAG
export TRAIN_LOG="${train_log_resolved}"
export PROVENANCE_MANIFEST="${manifest_resolved}"

mkdir -p "$(dirname "${TRAIN_LOG}")"
mkdir -p "$(dirname "${manifest_resolved}")"
"${PYTHON_BIN}" - \
  "${manifest_resolved}" \
  "${dataset_resolved}" \
  "${report_resolved}" \
  "${media_dir_resolved}" \
  "${media_manifest_resolved}" \
  "${builder_resolved}" \
  "${reward_resolved}" \
  "${launcher_resolved}" \
  "${dapo_launcher_resolved}" \
  "${rl_common_resolved}" \
  "${gate_resolved}" \
  "${adapter_resolved}" \
  "${base_config_resolved}" \
  "${toolvista_resolved}" \
  "${ms_swift_resolved}" \
  "${base_model_resolved}" \
  "${output_resolved}" \
  "${train_log_resolved}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

(
    destination_arg,
    dataset_arg,
    report_arg,
    media_dir_arg,
    media_manifest_arg,
    builder_arg,
    reward_plugin_arg,
    launcher_arg,
    dapo_launcher_arg,
    rl_common_arg,
    gate_arg,
    adapter_arg,
    base_config_arg,
    toolvista_root_arg,
    ms_swift_root_arg,
    base_model_arg,
    output_arg,
    train_log_arg,
) = sys.argv[1:]


def file_provenance(value: str, *, embed_utf8: bool = False) -> dict[str, object]:
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"provenance input is not a regular file: {path}")
    digest = hashlib.sha256()
    payload = None
    if embed_utf8:
        payload = path.read_bytes()
        digest.update(payload)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    result = {
        "resolved_path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    if payload is not None:
        result["utf8_content"] = payload.decode("utf-8")
    return result


def canonical_media_provenance(
    report_value: str,
    media_dir_value: str,
    manifest_value: str,
    manifest_file: dict[str, object],
) -> dict[str, object]:
    report = json.loads(Path(report_value).read_text(encoding="utf-8"))
    media = report.get("canonical_media")
    if not isinstance(media, dict):
        raise RuntimeError("dataset report has no canonical_media mapping")
    report_media_path = Path(str(media.get("path") or "")).resolve(strict=True)
    media_dir = Path(media_dir_value).resolve(strict=True)
    if report_media_path != media_dir:
        raise RuntimeError(
            f"canonical media path mismatch: report={report_media_path}, launcher={media_dir}"
        )

    report_manifest = media.get("manifest")
    if not isinstance(report_manifest, dict):
        raise RuntimeError("dataset report has no canonical_media.manifest mapping")
    report_manifest_path = Path(
        str(report_manifest.get("path") or "")
    ).resolve(strict=True)
    manifest_path = Path(manifest_value).resolve(strict=True)
    if report_manifest_path != manifest_path:
        raise RuntimeError(
            "canonical media manifest path mismatch: "
            f"report={report_manifest_path}, launcher={manifest_path}"
        )
    if report_manifest.get("sha256") != manifest_file["sha256"]:
        raise RuntimeError("canonical media manifest SHA-256 does not match dataset report")
    if int(report_manifest.get("bytes", -1)) != manifest_file["size_bytes"]:
        raise RuntimeError("canonical media manifest size does not match dataset report")

    with manifest_path.open("r", encoding="utf-8") as handle:
        entries = sum(1 for line in handle if line.strip())
    if int(report_manifest.get("entries", -1)) != entries:
        raise RuntimeError("canonical media manifest entry count does not match dataset report")
    if int(media.get("file_count", -1)) != entries:
        raise RuntimeError("canonical media file count does not match its manifest")
    return {
        "resolved_path": str(media_dir),
        "file_count": int(media["file_count"]),
        "bytes": int(media["bytes"]),
        "directory_bytes": int(media["directory_bytes"]),
        "manifest_entries": entries,
        "manifest_sha256": manifest_file["sha256"],
    }


def git_provenance(value: str) -> dict[str, object]:
    path = Path(value).resolve(strict=True)
    common = {
        "resolved_path": str(path),
        "git_commit": None,
        "git_dirty": None,
        "git_available": False,
    }
    try:
        probe = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return common
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return common
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "resolved_path": str(path),
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_available": True,
    }


def swift_runtime_provenance(value: str) -> dict[str, object]:
    root = Path(value).resolve(strict=True)
    sys.path.insert(0, str(root))
    import swift

    if not getattr(swift, "__file__", None):
        raise RuntimeError("imported swift package has no __file__")
    module_file = Path(swift.__file__).resolve(strict=True)
    try:
        module_file.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"imported swift package is outside MS_SWIFT_ROOT: {module_file}"
        ) from exc
    return {
        "swift_version": str(getattr(swift, "__version__", "unknown")),
        "swift_file": file_provenance(str(module_file)),
    }


def setting(name: str, default: str) -> str:
    return os.environ.get(name) or default


grad_acc = setting("GRAD_ACC", "4")
ms_swift_provenance = git_provenance(ms_swift_root_arg)
ms_swift_provenance.update(swift_runtime_provenance(ms_swift_root_arg))
media_manifest_provenance = file_provenance(media_manifest_arg)
canonical_media = canonical_media_provenance(
    report_arg, media_dir_arg, media_manifest_arg, media_manifest_provenance
)
manifest = {
    "schema": "toolvista.rl.provenance.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "variant": "datafixed-withnegative",
    "run_tag": os.environ["RUN_TAG"],
    "seed": os.environ["SEED"],
    "stage": os.environ["STAGE"],
    "output": {"resolved_path": str(Path(output_arg).resolve(strict=False))},
    "log": {"resolved_path": str(Path(train_log_arg).resolve(strict=False))},
    "manifest": {"resolved_path": str(Path(destination_arg).resolve(strict=False))},
    "base_model": {"resolved_path": str(Path(base_model_arg).resolve(strict=True))},
    "canonical_media": canonical_media,
    "files": {
        "dataset": file_provenance(dataset_arg),
        "dataset_report": file_provenance(report_arg),
        "canonical_media_manifest": media_manifest_provenance,
        "dataset_builder": file_provenance(builder_arg, embed_utf8=True),
        "reward_plugin": file_provenance(reward_plugin_arg, embed_utf8=True),
        "launcher_datafixed_withnegative": file_provenance(launcher_arg, embed_utf8=True),
        "launcher_dapo": file_provenance(dapo_launcher_arg, embed_utf8=True),
        "rl_common": file_provenance(rl_common_arg, embed_utf8=True),
        "c3_retention_gate": file_provenance(gate_arg),
        "c3_adapter_model": file_provenance(adapter_arg),
        "base_model_config": file_provenance(base_config_arg),
    },
    "reward_funcs": {
        "raw": os.environ["REWARD_FUNCS"],
        "names": os.environ["REWARD_FUNCS"].split(),
    },
    "training_parameters": {
        "rlhf_type": "grpo",
        "loss_type": "dapo",
        "max_steps": os.environ["MAX_STEPS"],
        "learning_rate": os.environ["LEARNING_RATE"],
        "beta": os.environ["BETA"],
        "save_steps": os.environ["SAVE_STEPS"],
        "save_total_limit": os.environ["SAVE_TOTAL_LIMIT"],
        "num_train_epochs": setting("EPOCHS", "1"),
        "per_device_train_batch_size": "1",
        "gradient_accumulation_steps": grad_acc,
        "steps_per_generation": setting("STEPS_PER_GENERATION", grad_acc),
        "num_generations": setting("NUM_GENERATIONS", "4"),
        "max_length": setting("MAX_LENGTH", "4096"),
        "max_completion_length": setting("MAX_COMPLETION_LENGTH", "512"),
        "temperature": setting("TEMPERATURE", "1.0"),
        "top_p": setting("TOP_P", "0.9"),
        "lora_rank": setting("LORA_RANK", "8"),
        "lora_alpha": setting("LORA_ALPHA", "32"),
        "torch_dtype": setting("TORCH_DTYPE", "bfloat16"),
        "max_pixels": setting("MAX_PIXELS", "1003520"),
        "epsilon_high": "0.28",
        "dynamic_sample": "true",
        "max_resample_times": "3",
        "overlong_filter": "true",
    },
    "repositories": {
        "toolvista": git_provenance(toolvista_root_arg),
        "ms_swift": ms_swift_provenance,
    },
}

destination = Path(destination_arg).resolve(strict=False)
if destination.exists():
    raise SystemExit(f"Refusing to overwrite an existing provenance manifest: {destination}")

fd, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise SystemExit(
            f"Refusing to overwrite an existing provenance manifest: {destination}"
        ) from exc
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)

print(f"Provenance manifest: {destination}")
PY

set -o pipefail
bash "${TOOLVISTA_ROOT}/scripts/autodl/10_train_visual_rft_dapo.sh" \
  2>&1 | tee "${TRAIN_LOG}"
exit "${PIPESTATUS[0]}"
