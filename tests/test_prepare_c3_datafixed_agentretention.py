from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_c3_datafixed_agentretention as builder


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.router_train_source = root / "base/router_train.jsonl"
        self.router_full_source = root / "base/router_full.jsonl"
        self.direct_source = root / "base/direct.jsonl"
        self.toolvqa_source = root / "base/toolvqa.jsonl"
        self.xlam_source = root / "base/xlam.jsonl"
        self.base_report = root / "base/router_datafixed.report.json"

        write_jsonl(
            self.router_train_source,
            [{"messages": [{"role": "assistant", "content": f"tool-{i}"}]} for i in range(3)],
        )
        full_rows = [
            {
                "sample_id": f"video-{i}",
                "input_modality": "video",
                "messages": [{"role": "user", "content": "<video>route"}],
            }
            for i in range(2)
        ]
        full_rows.extend(
            {
                "sample_id": f"image-{i}",
                "input_modality": "image",
                "messages": [{"role": "user", "content": "<image>route"}],
            }
            for i in range(2)
        )
        full_rows.extend(
            {
                "sample_id": f"text-{i}",
                "input_modality": "text",
                "messages": [{"role": "user", "content": "route"}],
            }
            for i in range(3)
        )
        write_jsonl(self.router_full_source, full_rows)
        write_jsonl(
            self.direct_source,
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>question"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ],
        )

        tool_schema = json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ]
        )
        write_jsonl(
            self.toolvqa_source,
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>question"},
                        {"role": "tool_call", "content": '{"name":"lookup","arguments":{}}'},
                        {"role": "tool_response", "content": "evidence"},
                        {"role": "assistant", "content": "final answer"},
                    ],
                    "images": ["image.jpg"],
                    "tools": tool_schema,
                },
                {
                    "messages": [
                        {"role": "user", "content": "<image>unfinished"},
                        {"role": "tool_call", "content": '{"name":"lookup","arguments":{}}'},
                    ],
                    "images": ["image.jpg"],
                    "tools": tool_schema,
                },
            ],
        )
        write_jsonl(
            self.xlam_source,
            [
                {
                    "messages": [
                        {"role": "system", "content": "Use relevant tools only."},
                        {"role": "user", "content": "Generate an image."},
                        {
                            "role": "assistant",
                            "content": (
                                "The provided functions do not support image generation, "
                                "so I cannot complete this request using these tools."
                            ),
                        },
                    ],
                    "tools": tool_schema,
                },
                {
                    "messages": [
                        {"role": "system", "content": "Use tools."},
                        {"role": "user", "content": "Look it up."},
                        {"role": "tool_call", "content": '{"name":"lookup","arguments":{}}'},
                    ],
                    "tools": tool_schema,
                },
            ],
        )

        self.base_report.parent.mkdir(parents=True, exist_ok=True)
        self.base_report.write_text(
            json.dumps(
                {
                    "variant": "datafixed",
                    "artifacts": {
                        "router_train": str(self.router_train_source.resolve()),
                        "router_full": str(self.router_full_source.resolve()),
                        "direct_answer_train": str(self.direct_source.resolve()),
                    },
                    "audit": {
                        "router_train": {"sha256": sha256(self.router_train_source)},
                        "router_full": {"sha256": sha256(self.router_full_source)},
                        "direct_answer": {"sha256": sha256(self.direct_source)},
                    },
                }
            ),
            encoding="utf-8",
        )

        prefix = root / "out/datafixed-agentretention"
        self.router_train = prefix / "router_train.jsonl"
        self.router_full = prefix / "router_full.jsonl"
        self.router_light = prefix / "router_light.jsonl"
        self.complete = prefix / "toolvqa_complete.jsonl"
        self.terminal = prefix / "toolvqa_terminal.jsonl"
        self.abstention = prefix / "xlam_abstention.jsonl"
        self.report = prefix / "report.json"

    def argv(self) -> list[str]:
        return [
            "--seed",
            "42",
            "--light-limit",
            "4",
            "--video-light-min",
            "2",
            "--terminal-max-rows",
            "2",
            "--abstention-max-rows",
            "2",
            "--base-report",
            str(self.base_report),
            "--router-train-source",
            str(self.router_train_source),
            "--router-full-source",
            str(self.router_full_source),
            "--direct-answer-source",
            str(self.direct_source),
            "--toolvqa-source",
            str(self.toolvqa_source),
            "--xlam-source",
            str(self.xlam_source),
            "--router-train",
            str(self.router_train),
            "--router-full",
            str(self.router_full),
            "--router-light",
            str(self.router_light),
            "--toolvqa-complete",
            str(self.complete),
            "--toolvqa-terminal",
            str(self.terminal),
            "--xlam-abstention",
            str(self.abstention),
            "--report",
            str(self.report),
        ]


class C3DatafixedAgentRetentionTest(unittest.TestCase):
    def test_builds_terminal_abstention_and_balanced_router_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c3-agentretention-") as value:
            fixture = Fixture(Path(value))
            builder.main(fixture.argv())

            report = json.loads(fixture.report.read_text(encoding="utf-8"))
            self.assertEqual(report["variant"], builder.VARIANT)
            self.assertEqual(
                report["router"]["light_by_modality"],
                {"image": 1, "text": 1, "video": 2},
            )

            complete = [json.loads(line) for line in fixture.complete.read_text().splitlines()]
            terminal = [json.loads(line) for line in fixture.terminal.read_text().splitlines()]
            abstention = [json.loads(line) for line in fixture.abstention.read_text().splitlines()]
            self.assertEqual(len(complete), 1)
            self.assertEqual(len(terminal), 1)
            self.assertEqual(len(abstention), 1)
            self.assertNotIn("loss_scale", complete[0]["messages"][1])
            self.assertEqual(terminal[0]["messages"][1]["loss_scale"], 0.0)
            self.assertEqual(terminal[0]["messages"][-1]["loss_scale"], 1.0)
            self.assertEqual(abstention[0]["messages"][-1]["loss_scale"], 1.0)
            self.assertIn("tools", abstention[0])

    def test_verify_only_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c3-agentretention-") as value:
            fixture = Fixture(Path(value))
            builder.main(fixture.argv())
            builder.main([*fixture.argv(), "--verify-only"])

            fixture.terminal.write_bytes(fixture.terminal.read_bytes() + b"\n")
            with self.assertRaisesRegex(SystemExit, "deterministic verification"):
                builder.main([*fixture.argv(), "--verify-only"])

    def test_refuses_non_isolated_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c3-agentretention-") as value:
            fixture = Fixture(Path(value))
            argv = fixture.argv()
            index = argv.index("--router-light") + 1
            argv[index] = str(fixture.root / "control/router_light.jsonl")
            with self.assertRaisesRegex(SystemExit, "output must remain"):
                builder.main(argv)


if __name__ == "__main__":
    unittest.main()
