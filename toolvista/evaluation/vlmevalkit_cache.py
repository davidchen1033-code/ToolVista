"""Verified local cache support for the frozen VLMEvalKit benchmark set."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


VLMEVALKIT_DATASETS: dict[str, dict[str, Any]] = {
    "RealWorldQA": {
        "filename": "RealWorldQA.tsv",
        "size": 183_675_999,
        "md5": "4de008f55dc4fd008ca9e15321dc44b7",
    },
    "MathVision_MINI": {
        "filename": "MathVision_MINI.tsv",
        "size": 12_764_058,
        "md5": "060fe4fa5d868987ce179307bd5f8a33",
    },
    "MMMU_DEV_VAL": {
        "filename": "MMMU_DEV_VAL.tsv",
        "size": 80_999_563,
        "md5": "585e8ad75e73f75dcad265dfd0417d64",
    },
    "OCRBench": {
        "filename": "OCRBench.tsv",
        "size": 113_986_299,
        "md5": "e953d98a987cc6e26ef717b61260b778",
    },
}


def audit_vlmevalkit_cache(cache_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for dataset, metadata in VLMEVALKIT_DATASETS.items():
        path = cache_dir / str(metadata["filename"])
        actual_size = path.stat().st_size if path.is_file() else None
        expected_size = int(metadata["size"])
        actual_md5 = _md5(path) if actual_size == expected_size else None
        expected_md5 = str(metadata["md5"])
        files[dataset] = {
            "path": path.as_posix(),
            "expected_size": expected_size,
            "actual_size": actual_size,
            "expected_md5": expected_md5,
            "actual_md5": actual_md5,
            "ready": actual_size == expected_size and actual_md5 == expected_md5,
        }
    missing = [dataset for dataset, detail in files.items() if detail["actual_size"] is None]
    invalid = [
        dataset
        for dataset, detail in files.items()
        if detail["actual_size"] is not None and not detail["ready"]
    ]
    return {
        "ready": not missing and not invalid,
        "files": files,
        "missing": missing,
        "invalid": invalid,
        "first_evaluation_may_download": bool(missing or invalid),
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_and_verify(
    url: str,
    output: Path,
    *,
    expected_size: int,
    expected_md5: str,
    timeout: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Downloaded size mismatch for {output.name}: "
                f"expected={expected_size}, actual={actual_size}"
            )
        actual_md5 = _md5(temporary)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Downloaded MD5 mismatch for {output.name}: "
                f"expected={expected_md5}, actual={actual_md5}"
            )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prefetch_vlmevalkit_cache(
    cache_dir: Path,
    *,
    base_url: str,
    datasets: Sequence[str] | None = None,
    retries: int = 3,
    timeout: float = 120.0,
) -> Mapping[str, str]:
    selected = list(datasets or VLMEVALKIT_DATASETS)
    unknown = sorted(set(selected) - set(VLMEVALKIT_DATASETS))
    if unknown:
        raise ValueError(f"Unknown VLMEvalKit datasets: {unknown}")
    if retries <= 0:
        raise ValueError("retries must be positive")

    results: dict[str, str] = {}
    for dataset in selected:
        metadata = VLMEVALKIT_DATASETS[dataset]
        output = cache_dir / str(metadata["filename"])
        if output.is_file() and output.stat().st_size == int(metadata["size"]):
            if _md5(output) == metadata["md5"]:
                results[dataset] = "cached"
                continue
        url = base_url.rstrip("/") + "/" + str(metadata["filename"])
        failure: Exception | None = None
        for attempt in range(retries):
            try:
                _download_and_verify(
                    url,
                    output,
                    expected_size=int(metadata["size"]),
                    expected_md5=str(metadata["md5"]),
                    timeout=timeout,
                )
                results[dataset] = "downloaded"
                failure = None
                break
            except Exception as exc:
                failure = exc
                if attempt + 1 < retries:
                    time.sleep(min(2**attempt, 5))
        if failure is not None:
            raise RuntimeError(f"Unable to prefetch {dataset} from {url}: {failure}") from failure
    return results
