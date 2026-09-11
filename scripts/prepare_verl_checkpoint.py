"""Merge a completed ms-swift LoRA checkpoint into a verified verl starting policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolvista.agentrl.checkpoints import export_checkpoint, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--retention-gate", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_checkpoint(args.output)
        else:
            if args.base_model is None or args.adapter is None:
                parser.error("--base-model and --adapter are required for export")
            result = export_checkpoint(args.base_model, args.adapter, args.output,
                                       swift_bin=args.swift_bin, retention_gate=args.retention_gate,
                                       provenance=args.provenance, overwrite=args.overwrite, dry_run=args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Checkpoint preflight failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
