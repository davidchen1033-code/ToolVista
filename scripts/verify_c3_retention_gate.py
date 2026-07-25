"""Verify that a passed C3 retention gate authorizes one exact checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        gate = json.loads(args.gate.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"C3 retention gate is unavailable: {exc}") from exc
    if gate.get("passed") is not True:
        raise SystemExit("C3 retention gate did not pass.")
    authorized = gate.get("authorized_checkpoint")
    if not isinstance(authorized, str) or not authorized.strip():
        raise SystemExit("C3 retention gate has no authorized_checkpoint.")
    if os.path.realpath(authorized) != os.path.realpath(args.checkpoint):
        raise SystemExit(
            "C3 retention gate checkpoint mismatch: "
            f"{authorized!r} != {str(args.checkpoint)!r}"
        )
    print(f"C3 retention gate passed: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
