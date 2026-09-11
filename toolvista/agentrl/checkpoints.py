"""Export LoRA weights and verify the immutable starting policy for AgentRL."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

MANIFEST = "toolvista_verl_checkpoint.json"
SCHEMA = "toolvista.verl.checkpoint.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_info(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def model_files(path: Path) -> list[Path]:
    config = read_json(path / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"expected a Qwen3-VL model: {path}")
    if (path / "adapter_config.json").exists():
        raise ValueError(f"expected merged full model, found an adapter: {path}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = set(read_json(index).get("weight_map", {}).values())
        if not names or any(not isinstance(n, str) or Path(n).name != n for n in names):
            raise ValueError("invalid safetensors shard index")
        weights = [path / name for name in sorted(names)]
    else:
        weights = [path / "model.safetensors"]
    for required in [path / "tokenizer_config.json", path / "preprocessor_config.json", *weights]:
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"incomplete model: {required}")
    if not (path / "tokenizer.json").is_file():
        raise ValueError(f"missing tokenizer.json: {path}")
    return sorted({*weights, *(p for p in path.iterdir() if p.suffix in (".json", ".jinja", ".txt") and p.name != MANIFEST)})


def validate_sources(base: Path, adapter: Path, *, retention_gate: Path | None = None,
                     provenance: Path | None = None) -> dict:
    base, adapter = base.resolve(strict=True), adapter.resolve(strict=True)
    base_files = model_files(base)
    adapter_config = read_json(adapter / "adapter_config.json")
    if adapter_config.get("peft_type", "").upper() != "LORA":
        raise ValueError("adapter_config.json must describe LoRA")
    identity = adapter_config.get("base_model_name_or_path")
    if not isinstance(identity, str) or not identity:
        raise ValueError("adapter has no base_model_name_or_path")
    if Path(identity).is_absolute():
        matches = Path(identity).resolve() == base
    else:
        matches = identity.rstrip("/").split("/")[-1] == base.name
    if not matches:
        raise ValueError(f"adapter base identity {identity!r} does not match {base}")
    adapter_files = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    evidence = {}
    state_path = adapter / "trainer_state.json"
    if state_path.exists():
        state = read_json(state_path)
        current, maximum = state.get("global_step"), state.get("max_steps")
        if type(current) is not int or type(maximum) is not int or maximum <= 0 or current < maximum:
            raise ValueError("trainer_state.json does not describe completed training (global_step >= max_steps)")
        evidence["trainer_state"] = file_info(state_path)
    if retention_gate is not None:
        gate = read_json(retention_gate)
        if gate.get("passed") is not True or Path(gate.get("authorized_checkpoint", "")).resolve() != adapter:
            raise ValueError("retention gate did not pass for this exact adapter")
        evidence["retention_gate"] = file_info(retention_gate)
    if provenance is not None:
        source = read_json(provenance)
        # Source training manifests may name the run output directory rather than a leaf checkpoint.
        authorized = source.get("authorized_checkpoint") or source.get("output_dir")
        if not isinstance(authorized, str) or not adapter.is_relative_to(Path(authorized).resolve()):
            raise ValueError("provenance must bind authorized_checkpoint/output_dir to the adapter or its training run")
        evidence["provenance"] = file_info(provenance)
    return {"base_model": {"path": str(base), "identity": base.name, "files": [file_info(p) for p in base_files]},
            "adapter": {"path": str(adapter), "base_identity": identity, "files": [file_info(p) for p in adapter_files]},
            "evidence": evidence}


def export_checkpoint(base: Path, adapter: Path, output: Path, *, swift_bin: str = "swift",
                      retention_gate: Path | None = None, provenance: Path | None = None,
                      overwrite: bool = False, dry_run: bool = False, runner=subprocess.run) -> dict:
    base, adapter, output = base.resolve(), adapter.resolve(), output.resolve()
    for source in (base, adapter):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must not overlap the base model or adapter")
    for evidence in (retention_gate, provenance):
        if evidence is not None and evidence.resolve().is_relative_to(output):
            raise ValueError("output must not contain source evidence")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; use --overwrite explicitly")
    source_info = validate_sources(base, adapter, retention_gate=retention_gate, provenance=provenance)
    staging = output.with_name(f".{output.name}.merge-{uuid4().hex}")
    command = [swift_bin, "export", "--model", str(base), "--adapters", str(adapter),
               "--load_args", "false", "--merge_lora", "true", "--safe_serialization", "true",
               "--output_dir", str(staging)]
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(output), **source_info, "merge_command": command}
    if dry_run:
        return {**manifest, "dry_run": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    try:
        runner(command, check=True)
        files = model_files(staging)
        # Refuse an export taken while the source files were being updated.
        if source_info != validate_sources(base, adapter, retention_gate=retention_gate, provenance=provenance):
            raise ValueError("checkpoint sources changed during export")
        manifest["merged_files"] = [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in files]
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def verify_checkpoint(output: Path) -> dict:
    output = output.resolve(strict=True)
    manifest = read_json(output / MANIFEST)
    if manifest.get("schema") != SCHEMA or manifest.get("output_dir") != str(output):
        raise ValueError("unverified or relocated merged checkpoint; re-export at the final training path")
    expected = manifest.get("merged_files", [])
    actual = model_files(output)
    if {p.name for p in actual} != {item["name"] for item in expected}:
        raise ValueError("merged checkpoint file inventory changed")
    for entry in expected:
        path = output / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"merged checkpoint hash mismatch: {path}")
    for group in (manifest["base_model"]["files"], manifest["adapter"]["files"], manifest["evidence"].values()):
        for entry in group:
            if file_info(Path(entry["path"])) != entry:
                raise ValueError(f"source/evidence hash mismatch: {entry['path']}")
    manifest["manifest_sha256"] = sha256(output / MANIFEST)
    return manifest
