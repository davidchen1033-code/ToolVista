from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_c3_datafixed_terminalfix as builder


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolvqa_row(question: str, answer: str = "final answer") -> dict[str, object]:
    tools = [
        {
            "name": "lookup",
            "description": "Look up evidence.",
            "parameters": [
                {"name": "query", "type": "STRING", "required": True}
            ],
        }
    ]
    return {
        "messages": [
            {"role": "user", "content": f"<image>{question}"},
            {
                "role": "tool_call",
                "content": '{"name":"lookup","arguments":{"query":"x"}}',
            },
            {"role": "tool_response", "content": "evidence"},
            {"role": "assistant", "content": answer},
        ],
        "images": ["image.jpg"],
        "tools": json.dumps(tools),
    }


class C3TerminalFixBuilderTest(unittest.TestCase):
    def test_react_transition_matches_benchmark_markers_and_masks_history(self) -> None:
        row = toolvqa_row("kept question")

        transition, converted = builder._react_prefix(
            row, target_index=3, source_row=7
        )

        self.assertEqual(transition, "terminal")
        self.assertIn("Final Answer:final answer", converted["messages"][-1]["content"])
        self.assertEqual(converted["messages"][-1]["loss_scale"], 1.0)
        history_actions = [
            message
            for message in converted["messages"]
            if message["role"] == "assistant" and "Action:" in message["content"]
        ]
        self.assertEqual(len(history_actions), 1)
        self.assertEqual(history_actions[0]["loss_scale"], 0.0)
        self.assertIn("Final Answer:", converted["messages"][0]["content"])
        self.assertTrue(converted["messages"][-2]["content"].startswith("<image>"))
        self.assertNotIn("tools", converted)

    def test_build_excludes_official_test_questions_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c3-terminalfix-") as value:
            root = Path(value)
            base_root = root / "base"
            router_train = base_root / "router.jsonl"
            router_full = base_root / "router_full.jsonl"
            direct = base_root / "direct.jsonl"
            toolvqa = base_root / "toolvqa.jsonl"
            xlam = base_root / "xlam.jsonl"
            raw_test = base_root / "test.jsonl"
            canonical_eval = base_root / "dataset.json"
            base_report = base_root / "report.json"

            write_jsonl(
                router_train,
                [
                    {
                        "messages": [
                            {"role": "user", "content": f"route-{index}"},
                            {"role": "assistant", "content": "tool"},
                        ]
                    }
                    for index in range(3)
                ],
            )
            write_jsonl(
                router_full,
                [
                    {"sample_id": "video", "input_modality": "video"},
                    {"sample_id": "image", "input_modality": "image"},
                    {"sample_id": "text", "input_modality": "text"},
                ],
            )
            write_jsonl(
                direct,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "excluded question"},
                            {"role": "assistant", "content": "leaked"},
                        ]
                    },
                    {
                        "messages": [
                            {"role": "user", "content": "kept question"},
                            {"role": "assistant", "content": "answer"},
                        ]
                    },
                ],
            )
            write_jsonl(
                toolvqa,
                [toolvqa_row("excluded question"), toolvqa_row("kept question")],
            )
            tools = json.loads(toolvqa_row("x")["tools"])
            write_jsonl(
                xlam,
                [
                    {
                        "messages": [
                            {"role": "system", "content": "Use relevant tools only."},
                            {"role": "user", "content": "Generate an image."},
                            {
                                "role": "assistant",
                                "content": (
                                    "The provided tools do not support image generation, "
                                    "so I cannot complete this request using these tools."
                                ),
                            },
                        ],
                        "tools": json.dumps(tools),
                    }
                ],
            )
            write_jsonl(
                raw_test,
                [
                    {
                        "question": "rewritten excluded question",
                        "ori_question": "excluded question",
                    }
                ],
            )
            canonical_eval.write_text(
                json.dumps(
                    {
                        "0": {
                            "dialogs": [
                                {"role": "user", "content": "excluded question"}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            base_report.write_text(
                json.dumps(
                    {
                        "variant": "datafixed",
                        "artifacts": {
                            "router_train": str(router_train.resolve()),
                            "router_full": str(router_full.resolve()),
                            "direct_answer_train": str(direct.resolve()),
                        },
                        "audit": {
                            "router_train": {"sha256": sha256(router_train)},
                            "router_full": {"sha256": sha256(router_full)},
                            "direct_answer": {"sha256": sha256(direct)},
                        },
                    }
                ),
                encoding="utf-8",
            )

            output = root / builder.VARIANT
            argv = [
                "--light-limit",
                "3",
                "--video-light-min",
                "1",
                "--toolvqa-max-rows",
                "2",
                "--abstention-max-rows",
                "1",
                "--base-report",
                str(base_report),
                "--router-train-source",
                str(router_train),
                "--router-full-source",
                str(router_full),
                "--direct-answer-source",
                str(direct),
                "--toolvqa-source",
                str(toolvqa),
                "--toolvqa-test-source",
                str(raw_test),
                "--toolvqa-eval-source",
                str(canonical_eval),
                "--xlam-source",
                str(xlam),
                "--router-train",
                str(output / "router.jsonl"),
                "--router-full",
                str(output / "router_full.jsonl"),
                "--router-light",
                str(output / "router_light.jsonl"),
                "--direct-answer",
                str(output / "direct.jsonl"),
                "--toolvqa-complete",
                str(output / "complete.jsonl"),
                "--toolvqa-terminal",
                str(output / "terminal.jsonl"),
                "--toolvqa-react-action",
                str(output / "react_action.jsonl"),
                "--toolvqa-react-terminal",
                str(output / "react_terminal.jsonl"),
                "--xlam-abstention",
                str(output / "abstention.jsonl"),
                "--report",
                str(output / "report.json"),
            ]

            builder.main(argv)
            builder.main([*argv, "--verify-only"])

            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["variant"], builder.VARIANT)
            self.assertEqual(report["official_test_exclusion"]["postbuild_overlap_rows"], 0)
            self.assertEqual(report["official_test_exclusion"]["direct_answer_excluded_rows"], 1)
            self.assertEqual(report["toolvqa"]["official_test_excluded_rows"], 1)
            direct_rows = [json.loads(line) for line in (output / "direct.jsonl").read_text().splitlines()]
            self.assertEqual(len(direct_rows), 1)
            self.assertEqual(direct_rows[0]["messages"][0]["content"], "kept question")

    def test_rejects_output_outside_terminalfix_namespace(self) -> None:
        args = builder.parse_args(["--report", "/tmp/control/report.json"])
        with self.assertRaisesRegex(ValueError, "output must remain"):
            builder.build_outputs(args)


if __name__ == "__main__":
    unittest.main()
