from __future__ import annotations

import unittest

from scripts.check_c3_terminalfix_retention import terminal_metrics


class C3TerminalFixRetentionTest(unittest.TestCase):
    def test_terminal_metrics_separate_emission_and_correctness(self) -> None:
        predictions = {
            "0": {"prediction": [{"role": "assistant", "content": "correct"}]},
            "1": {"prediction": [{"role": "assistant", "content": "wrong"}]},
            "2": {
                "prediction": [
                    {
                        "role": "assistant",
                        "error": {"type": "API_ERROR"},
                        "tool_calls": [],
                    }
                ]
            },
            "3": {
                "prediction": [
                    {"role": "assistant", "tool_calls": [{"type": "function"}]}
                ]
            },
        }

        metrics = terminal_metrics(predictions, answer_accuracy=25.0)

        self.assertEqual(
            metrics["toolvqa_sbs/terminal_answer_emission_rate"], 50.0
        )
        self.assertEqual(metrics["toolvqa_sbs/terminal_format_error_rate"], 25.0)
        self.assertEqual(
            metrics["toolvqa_sbs/conditional_answer_accuracy"], 50.0
        )

    def test_terminal_metrics_reject_empty_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            terminal_metrics({}, answer_accuracy=0.0)


if __name__ == "__main__":
    unittest.main()
