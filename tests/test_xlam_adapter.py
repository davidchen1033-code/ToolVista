from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.export_xlam import write_export
from toolvista.adapters.xlam import XLAMAdapter
from toolvista.exporters.ms_swift import canonical_to_ms_swift_row
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "examples" / "data" / "xlam"


def make_paths() -> ToolVistaPaths:
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "xlam": {
                    "root": str(SAMPLE_ROOT),
                    "unified_data_dir": ".",
                    "default_subset": "actionstudio-98k",
                    "default_split": "train",
                }
            }
        },
    )


class XLAMAdapterTest(unittest.TestCase):
    def test_iter_raw_streams_json_array_and_honors_limit(self) -> None:
        adapter = XLAMAdapter(make_paths(), max_trajectories=2)

        items = list(adapter.iter_raw())

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["_toolvista_raw_index"], 0)
        self.assertTrue(items[0]["_toolvista_file_path"].endswith("actionstudio-98k/train/train.json"))

    def test_each_assistant_turn_becomes_one_target_sample(self) -> None:
        adapter = XLAMAdapter(make_paths(), max_trajectories=1)

        results = list(adapter.iter_results())

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.ok for result in results), [result.to_report_dict() for result in results])
        first = results[0].sample
        second = results[1].sample
        assert first is not None and second is not None
        self.assertEqual(first.sample_id, "actionstudio-sample---train---1---2")
        self.assertEqual(second.sample_id, "actionstudio-sample---train---1---4")
        self.assertEqual([message.role for message in first.messages], ["system", "user", "tool_call"])
        self.assertNotIn("I will retrieve", " ".join(message.content for message in first.messages))
        self.assertIsNone(first.messages[-1].loss)

        self.assertEqual(
            [message.role for message in second.messages],
            ["system", "user", "assistant", "tool_call", "tool_response", "tool_call"],
        )
        self.assertFalse(second.messages[2].loss)
        self.assertFalse(second.messages[3].loss)
        self.assertEqual(second.messages[4].content, '{"name":"John Doe","age":35}')
        self.assertEqual(json.loads(second.messages[-1].content)["name"], "finish")
        self.assertEqual(second.labels["target_tool_call_ids"], ["call-2"])

    def test_parallel_tool_calls_remain_consecutive_for_ms_swift(self) -> None:
        adapter = XLAMAdapter(make_paths(), max_trajectories=2)
        results = list(adapter.iter_results())
        parallel = results[2].sample
        assert parallel is not None

        self.assertEqual([message.role for message in parallel.messages], ["system", "user", "tool_call", "tool_call"])
        calls = [json.loads(message.content) for message in parallel.messages[-2:]]
        self.assertEqual([call["arguments"]["id"] for call in calls], [13579, 24680])
        self.assertTrue(parallel.metadata["parallel_tool_calls"])

        row = canonical_to_ms_swift_row(parallel)
        self.assertEqual([message["role"] for message in row["messages"][-2:]], ["tool_call", "tool_call"])
        self.assertEqual(len(json.loads(row["tools"])), 1)

    def test_non_string_tool_output_is_json_and_history_has_no_loss(self) -> None:
        adapter = XLAMAdapter(make_paths(), max_trajectories=2)
        results = list(adapter.iter_results())
        final_text = results[3].sample
        assert final_text is not None

        self.assertEqual(final_text.messages[-1].role, "assistant")
        self.assertEqual(final_text.messages[-1].content, "I found First Game and Second Game.")
        history_calls = [message for message in final_text.messages if message.role == "tool_call"]
        self.assertEqual(len(history_calls), 2)
        self.assertTrue(all(message.loss is False for message in history_calls))
        response = next(message for message in final_text.messages if message.role == "tool_response")
        self.assertEqual(json.loads(response.content)[0]["title"], "First Game")

    def test_empty_tool_calls_with_text_is_a_text_target(self) -> None:
        adapter = XLAMAdapter(make_paths())
        results = list(adapter.iter_results())
        text_sample = results[-1].sample
        assert text_sample is not None

        self.assertEqual(text_sample.metadata["target_kind"], "assistant_text")
        self.assertEqual(text_sample.messages[-1].role, "assistant")
        self.assertNotEqual(text_sample.messages[-1].content, "[]")
        self.assertEqual(text_sample.labels["final_answer"], text_sample.messages[-1].content)

    def test_invalid_tool_call_arguments_produce_an_error_result(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["conversation"] = [
            raw["conversation"][0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_guest", "arguments": "not-an-object"},
                    }
                ],
            },
        ]

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertIn("arguments must be a mapping", results[0].errors[0])

    def test_python_style_tool_schema_types_are_normalized_recursively(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_points",
                    "description": "Analyze named point groups.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "str, optional"},
                            "count": {"type": "int", "default": "2"},
                            "groups": {"type": "Dict[str, List[Tuple[float, float]]]"},
                        },
                        "required": ["label", "count", "groups"],
                    },
                },
            }
        ]

        result = list(adapter.adapt(raw))[0]

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        parameters = result.sample.tools[0].parameters
        self.assertEqual(parameters["properties"]["label"]["type"], "string")
        self.assertNotIn("label", parameters["required"])
        self.assertEqual(parameters["properties"]["count"]["type"], "integer")
        self.assertEqual(parameters["properties"]["count"]["default"], 2)
        groups = parameters["properties"]["groups"]
        self.assertEqual(groups["type"], "object")
        points = groups["additionalProperties"]["items"]
        self.assertEqual(points["type"], "array")
        self.assertEqual(points["items"], {"type": "number"})
        self.assertEqual(points["minItems"], 2)
        self.assertEqual(points["maxItems"], 2)

    def test_ambiguous_python_tool_schema_type_rejects_trajectory(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["tools"][0]["function"]["parameters"]["properties"]["callback"] = {
            "type": "Callable[[float], float]"
        }

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertIn("unsupported or ambiguous Python type", results[0].errors[0])
        self.assertIn("Callable[[float], float]", results[0].errors[0])

    def test_identical_duplicate_tools_are_deduplicated(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        original_tool_count = len(raw["tools"])
        raw["tools"].append(json.loads(json.dumps(raw["tools"][0])))

        result = list(adapter.adapt(raw))[0]

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(len(result.sample.tools), original_tool_count)

    def test_conflicting_duplicate_tool_names_reject_trajectory(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        conflicting = json.loads(json.dumps(raw["tools"][0]))
        conflicting["function"]["description"] = "A conflicting definition."
        raw["tools"].append(conflicting)

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertIn("conflicting definitions", results[0].errors[0])

    def test_consecutive_assistant_target_is_rejected_before_ms_swift_can_merge_loss(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["conversation"] = [
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "First action"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_guest", "arguments": {"guest_id": 7}},
                    }
                ],
            },
        ]

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].ok, results[0].to_report_dict())
        self.assertFalse(results[1].ok)
        self.assertIn("consecutive assistant messages", results[1].errors[0])

    def test_history_conversion_is_incremental(self) -> None:
        class CountingXLAMAdapter(XLAMAdapter):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.converted_history_source_messages = 0

            def _convert_history(self, messages, *args, **kwargs):
                self.converted_history_source_messages += len(messages)
                return super()._convert_history(messages, *args, **kwargs)

        adapter = CountingXLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["conversation"] = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
        ]

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.ok for result in results), [result.to_report_dict() for result in results])
        self.assertLessEqual(adapter.converted_history_source_messages, len(raw["conversation"]))

    def test_empty_assistant_target_does_not_poison_later_history(self) -> None:
        adapter = XLAMAdapter(make_paths())
        raw = next(adapter.iter_raw())
        raw["conversation"] = [
            {"role": "user", "content": "First request"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Second request"},
            {"role": "assistant", "content": "Second response"},
        ]

        results = list(adapter.adapt(raw))

        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok, results[1].to_report_dict())
        self.assertEqual(results[1].warnings, ["Dropped empty assistant history message 1."])
        assert results[1].sample is not None
        self.assertEqual(
            [message.role for message in results[1].sample.messages],
            ["system", "user", "user", "assistant"],
        )

    def test_streaming_export_writes_all_assistant_targets(self) -> None:
        adapter = XLAMAdapter(make_paths())
        output = REPO_ROOT / "examples" / "ms_swift" / ".xlam-test.jsonl"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".xlam-test.errors.jsonl"
        try:
            stats = write_export(adapter.iter_results(), output, errors, include_metadata=True)

            self.assertEqual(stats["converted"], 5)
            self.assertEqual(stats["skipped"], 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[2]["messages"][-2]["role"], "tool_call")
            self.assertEqual(rows[2]["messages"][-1]["role"], "tool_call")
            self.assertEqual(rows[-1]["messages"][-1]["role"], "assistant")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")
        finally:
            output.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
