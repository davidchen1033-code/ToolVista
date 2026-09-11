from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from toolvista.agentrl.checkpoints import MANIFEST, export_checkpoint, verify_checkpoint


def write_model(path):
    path.mkdir(parents=True, exist_ok=True)
    for name, content in {"config.json": '{"model_type":"qwen3_vl"}', "tokenizer_config.json": '{}',
                          "preprocessor_config.json": '{}', "tokenizer.json": '{}', "model.safetensors": 'weights'}.items():
        (path / name).write_text(content)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.adapter, self.output = [self.root / name for name in ("Qwen3-VL-4B-Instruct", "adapter", "merged")]
        write_model(self.base)
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(self.base)}))
        (self.adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    def fake_export(self, command, **kwargs):
        self.command = command
        write_model(Path(command[command.index("--output_dir") + 1]))

    def export(self, **kwargs):
        return export_checkpoint(self.base, self.adapter, self.output, runner=self.fake_export, **kwargs)

    def test_export_verification_and_tamper_detection(self):
        self.export()
        self.assertTrue(verify_checkpoint(self.output)["manifest_sha256"])
        self.assertEqual(self.command[:2], ["swift", "export"])
        self.assertIn("--safe_serialization", self.command)
        (self.output / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_checkpoint(self.output)

    def test_output_refusal_and_explicit_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        self.export(overwrite=True)
        verify_checkpoint(self.output)

    def test_overlap_is_rejected_even_with_overwrite(self):
        for output in (self.base, self.adapter / "merged", self.root):
            with self.assertRaisesRegex(ValueError, "overlap"):
                export_checkpoint(self.base, self.adapter, output, overwrite=True)

    def test_failed_export_preserves_existing_model(self):
        self.export()
        previous = (self.output / MANIFEST).read_bytes()
        def fail(command, **kwargs):
            raise RuntimeError("export failed")
        with self.assertRaises(RuntimeError):
            export_checkpoint(self.base, self.adapter, self.output, runner=fail, overwrite=True)
        self.assertEqual((self.output / MANIFEST).read_bytes(), previous)

    def test_optional_evidence_must_match_and_training_must_be_complete(self):
        state = self.adapter / "trainer_state.json"
        state.write_text('{"global_step":9,"max_steps":10}')
        with self.assertRaisesRegex(ValueError, "completed"):
            self.export()
        state.write_text('{"global_step":10,"max_steps":10}')
        gate = self.root / "gate.json"
        gate.write_text(json.dumps({"passed": True, "authorized_checkpoint": str(self.base)}))
        with self.assertRaisesRegex(ValueError, "exact adapter"):
            self.export(retention_gate=gate)
        gate.write_text(json.dumps({"passed": True, "authorized_checkpoint": str(self.adapter)}))
        self.export(retention_gate=gate)
        state.write_text('{"global_step":11,"max_steps":10}')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_checkpoint(self.output)

    def test_invalid_model_and_adapter_and_dry_run(self):
        plan = self.export(dry_run=True)
        self.assertTrue(plan["dry_run"])
        self.assertFalse(self.output.exists())
        (self.base / "config.json").write_text('{"model_type":"qwen2_vl"}')
        with self.assertRaisesRegex(ValueError, "Qwen3-VL"):
            self.export()
        (self.base / "config.json").write_text('{"model_type":"qwen3_vl"}')
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaises(OSError):
            self.export()


if __name__ == "__main__":
    unittest.main()
