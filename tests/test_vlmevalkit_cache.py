from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from pathlib import Path

from toolvista.evaluation.vlmevalkit_cache import (
    _download_and_verify,
    audit_vlmevalkit_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class VLMEvalKitCacheTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-vlmeval-cache-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_download_is_verified_and_promoted_atomically(self) -> None:
        root = self.make_temp_dir()
        source = root / "source.tsv"
        source.write_bytes(b"dataset")
        output = root / "cache" / "test.tsv"

        _download_and_verify(
            source.as_uri(),
            output,
            expected_size=7,
            expected_md5=hashlib.md5(b"dataset").hexdigest(),
            timeout=5,
        )

        self.assertEqual(output.read_bytes(), b"dataset")
        self.assertFalse((output.parent / ".test.tsv.part").exists())

    def test_failed_verification_does_not_replace_existing_cache(self) -> None:
        root = self.make_temp_dir()
        source = root / "source.tsv"
        source.write_bytes(b"invalid")
        output = root / "test.tsv"
        output.write_bytes(b"existing")

        with self.assertRaisesRegex(ValueError, "size mismatch"):
            _download_and_verify(
                source.as_uri(),
                output,
                expected_size=99,
                expected_md5="0" * 32,
                timeout=5,
            )

        self.assertEqual(output.read_bytes(), b"existing")
        self.assertFalse((root / ".test.tsv.part").exists())

    def test_audit_reports_every_missing_dataset(self) -> None:
        detail = audit_vlmevalkit_cache(self.make_temp_dir())
        self.assertFalse(detail["ready"])
        self.assertEqual(len(detail["missing"]), 4)


if __name__ == "__main__":
    unittest.main()
