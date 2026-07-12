"""Adapter for ToolVQA multimodal multi-step tool-use trajectories."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence

from toolvista.adapters.base import BaseDatasetAdapter
from toolvista.schema.canonical import (
    AdaptResult,
    CanonicalMessage,
    CanonicalSample,
    ContentPart,
    MediaBundle,
    ToolSpec,
)


def _object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: Sequence[str] = (),
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {name: dict(value) for name, value in properties.items()},
    }
    if required:
        schema["required"] = list(required)
    return schema


IMAGE_ARGUMENT = {
    "type": "string",
    "description": "Reference to the input image, for example images[0].",
}


class ToolVQAAdapter(BaseDatasetAdapter):
    """Convert ToolVQA JSON/JSONL records into canonical agent trajectories."""

    source_name: ClassVar[str] = "toolvqa"
    image_output_tools: ClassVar[frozenset[str]] = frozenset(
        {"TextToImage", "ImageStylization", "DrawBox", "AddText", "Plot"}
    )
    output_image_suffixes: ClassVar[Sequence[str]] = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    image_output_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"(?i)(?:[a-z]:)?(?:[^\s\"'<>()[\],;]+[\\/])*[^\s\"'<>()[\],;]+\.(?:jpe?g|png|webp|gif)"
    )
    tool_catalog: ClassVar[Mapping[str, Mapping[str, Any]]] = {
        "ImageDescription": {
            "description": "A useful tool that returns a brief description of the input image.",
            "parameters": _object_schema({"image": IMAGE_ARGUMENT}, required=("image",)),
        },
        "OCR": {
            "description": "This tool can recognize all text on the input image.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "lang": {
                        "type": "string",
                        "description": "Language code to recognize. Defaults to en.",
                    },
                },
                required=("image",),
            ),
        },
        "CountGivenObject": {
            "description": "The tool can count the number of a certain object in the image.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "text": {
                        "type": "string",
                        "description": "The object description in English.",
                    },
                    "bbox": {
                        "type": "string",
                        "description": "Optional region in (x1, y1, x2, y2) format.",
                    },
                },
                required=("image", "text"),
            ),
        },
        "RegionAttributeDescription": {
            "description": "Describe an attribute of a region of the input image.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "bbox": {
                        "type": "string",
                        "description": "Region in (x1, y1, x2, y2) format.",
                    },
                    "attribute": {
                        "type": "string",
                        "description": "The attribute to describe.",
                    },
                },
                required=("image", "bbox", "attribute"),
            ),
        },
        "TextToBbox": {
            "description": "The tool can detect an object location according to its description.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "text": {
                        "type": "string",
                        "description": "The object description in English.",
                    },
                    "top1": {
                        "type": "boolean",
                        "description": "Return only the highest-scoring object when true. Defaults to true.",
                    },
                },
                required=("image", "text"),
            ),
        },
        "GoogleSearch": {
            "description": "The tool can search a text query on Google and return related results.",
            "parameters": _object_schema(
                {
                    "query": {"type": "string", "description": "The search query."},
                    "k": {
                        "type": "integer",
                        "description": "Number of leading results to return. Defaults to 1.",
                    },
                },
                required=("query",),
            ),
        },
        "Calculator": {
            "description": (
                "A calculator tool. The input must be one Python expression without package imports; "
                "functions from the math package are available."
            ),
            "parameters": _object_schema(
                {"expression": {"type": "string", "description": "The Python expression to evaluate."}},
                required=("expression",),
            ),
        },
        "Plot": {
            "description": (
                "Execute Python code that defines solution() and returns a matplotlib figure."
            ),
            "parameters": _object_schema(
                {"command": {"type": "string", "description": "Markdown-formatted Python code."}},
                required=("command",),
            ),
        },
        "MathOCR": {
            "description": "Recognize mathematical expressions in an image and return LaTeX-style text.",
            "parameters": _object_schema({"image": IMAGE_ARGUMENT}, required=("image",)),
        },
        "Solver": {
            "description": "Execute Python code with sympy to solve mathematical equations.",
            "parameters": _object_schema(
                {"command": {"type": "string", "description": "Markdown-formatted Python code."}},
                required=("command",),
            ),
        },
        "TextToImage": {
            "description": "Generate an image according to a series of textual keywords.",
            "parameters": _object_schema(
                {
                    "keywords": {
                        "type": "string",
                        "description": "A series of image-generation keywords separated by commas.",
                    }
                },
                required=("keywords",),
            ),
        },
        "ImageStylization": {
            "description": "Modify an input image according to a textual instruction.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "instruction": {
                        "type": "string",
                        "description": "The image editing or stylization instruction.",
                    },
                },
                required=("image", "instruction"),
            ),
        },
        "DrawBox": {
            "description": "Draw a box on a specified region of the input image.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "bbox": {
                        "type": "string",
                        "description": "Region in (x1, y1, x2, y2) format.",
                    },
                    "annotation": {
                        "type": "string",
                        "description": "Optional annotation text for the box.",
                    },
                },
                required=("image", "bbox"),
            ),
        },
        "AddText": {
            "description": "Add text at a specified position on the input image.",
            "parameters": _object_schema(
                {
                    "image": IMAGE_ARGUMENT,
                    "text": {"type": "string", "description": "Text to add to the image."},
                    "position": {
                        "type": "string",
                        "description": "Text position as coordinates or a placement code such as mt.",
                    },
                    "color": {
                        "type": "string",
                        "description": "Text color. Defaults to red.",
                    },
                },
                required=("image", "text", "position"),
            ),
        },
    }

    def __init__(
        self,
        *args: Any,
        split: str = "train",
        annotation_file: Optional[str] = None,
        max_items: Optional[int] = None,
        validate_media_files: bool = True,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.split = split
        self.annotation_file = annotation_file
        self.max_items = max_items
        self.validate_media_files = validate_media_files
        self.include_reasoning = include_reasoning

    def iter_raw(self) -> Iterable[Dict[str, Any]]:
        annotation_path = self._annotation_path()
        if annotation_path.suffix.lower() == ".jsonl":
            yield from self._iter_jsonl(annotation_path)
        else:
            yield from self._iter_json(annotation_path)

    def adapt(self, raw_item: Any) -> AdaptResult:
        raw_ref = self._raw_ref(raw_item)
        if not isinstance(raw_item, Mapping):
            return AdaptResult(errors=["Raw item is not a mapping."], raw_ref=raw_ref)

        question = self._non_empty_string(raw_item.get("question"))
        answer = self._non_empty_string(raw_item.get("answer"))
        image_path = self._non_empty_string(raw_item.get("image_path"))
        context = raw_item.get("context")

        errors: List[str] = []
        if not question:
            errors.append("Missing or empty question.")
        if not answer:
            errors.append("Missing or empty answer.")
        if not image_path:
            errors.append("Missing or empty image_path.")
        if not isinstance(context, list) or not context:
            errors.append("Missing or empty tool context.")
        if errors:
            return AdaptResult(errors=errors, raw_ref=raw_ref)
        resolved_image_path = self._resolve_image_path(image_path)
        if self.validate_media_files and not Path(resolved_image_path).is_file():
            return AdaptResult(errors=[f"Image file does not exist: {resolved_image_path}"], raw_ref=raw_ref)

        messages = [
            CanonicalMessage(
                role="user",
                content=question,
                content_parts=[ContentPart(type="image", media_ref="images[0]"), ContentPart(type="text", text=question)],
            )
        ]
        tool_trace: List[Dict[str, Any]] = []
        reasoning_trace: List[Dict[str, Any]] = []
        tool_parameters: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        generation_step_index: Optional[int] = None
        generation_tool_name: Optional[str] = None
        generated_media_placeholder: Optional[str] = None

        for step_index, step in enumerate(context):
            if not isinstance(step, Mapping):
                return AdaptResult(
                    errors=[f"Tool context step {step_index} is not a mapping."],
                    warnings=warnings,
                    raw_ref=raw_ref,
                )

            tool_name = self._non_empty_string(step.get("name"))
            tool_output = self._value_to_string(step.get("output"))
            if not tool_name:
                return AdaptResult(
                    errors=[f"Tool context step {step_index} has no tool name."],
                    warnings=warnings,
                    raw_ref=raw_ref,
                )
            if not tool_output:
                return AdaptResult(
                    errors=[f"Tool context step {step_index} has no tool output."],
                    warnings=warnings,
                    raw_ref=raw_ref,
                )

            arguments, argument_warning = self._parse_arguments(step.get("input"))
            if argument_warning:
                warnings.append(f"Tool context step {step_index}: {argument_warning}")
            normalized_arguments, unsupported_media_paths = self._normalize_media_arguments(arguments, image_path)
            if unsupported_media_paths:
                return AdaptResult(
                    errors=[
                        f"Tool context step {step_index} references image media other than the original input, "
                        "which the current canonical schema cannot represent safely: "
                        + ", ".join(unsupported_media_paths)
                    ],
                    warnings=warnings,
                    raw_ref=raw_ref,
                )
            if tool_name not in tool_parameters:
                tool_parameters[tool_name] = self._parameters_for_tool(tool_name, normalized_arguments)
            elif tool_name not in self.tool_catalog:
                tool_parameters[tool_name] = self._merge_parameter_schemas(
                    tool_parameters[tool_name], self._infer_parameters(normalized_arguments)
                )

            call = {"name": tool_name, "arguments": normalized_arguments}
            response = {"name": tool_name, "content": tool_output}
            reasoning = self._compact_dict(
                {
                    "tool_name": tool_name,
                    "thought": step.get("thought"),
                    "thought_choose": step.get("thought_choose"),
                    "thought_query": step.get("thought_query"),
                }
            )
            reasoning_trace.append(reasoning)
            if self.include_reasoning:
                reasoning_text = self._non_empty_string(step.get("thought"))
                if reasoning_text:
                    messages.append(CanonicalMessage(role="assistant", content=reasoning_text, loss=False))
            messages.append(
                CanonicalMessage(role="tool_call", content=json.dumps(call, ensure_ascii=False, separators=(",", ":")))
            )
            image_output_path = self._image_output_path(tool_name, tool_output)
            if image_output_path is not None:
                generation_step_index = step_index
                generation_tool_name = tool_name
                generated_media_placeholder = image_output_path
                tool_trace.append({"tool_call": call})
                break

            messages.append(CanonicalMessage(role="tool_response", content=tool_output))
            tool_trace.append({"tool_call": call, "tool_response": response})

        source_answer_contains_media = self._answer_references_generated_media(
            answer,
            generated_media_placeholder,
        )
        if source_answer_contains_media and generation_step_index is None:
            return AdaptResult(
                errors=[
                    "ToolVQA final answer references image media but the trajectory has no matching "
                    f"image-producing tool call: {answer}"
                ],
                warnings=warnings,
                raw_ref=raw_ref,
            )

        routing_only = generation_step_index is not None
        if not routing_only:
            messages.append(CanonicalMessage(role="assistant", content=answer))
        tools = [
            ToolSpec(
                name=name,
                description=str(self.tool_catalog.get(name, {}).get("description", f"ToolVQA tool: {name}")),
                parameters=parameters,
            )
            for name, parameters in tool_parameters.items()
        ]
        raw_index = raw_item.get("_toolvista_raw_index", "unknown")
        sample = CanonicalSample(
            sample_id=f"toolvqa-{self.split}-{raw_index}",
            source=self.source_name,
            task_type="agent_trajectory",
            messages=messages,
            media=MediaBundle(images=[resolved_image_path]),
            tools=tools,
            labels=self._compact_dict(
                {
                    "final_answer": None if routing_only else answer,
                    "target_tool_calls": [item["tool_call"] for item in tool_trace],
                    "tool_trace": tool_trace,
                }
            ),
            split=self.split,
            provenance=self._compact_dict(
                {
                    "source_dataset": "ToolVQA",
                    "source_repository": "DietCoke4671/ToolVQA",
                    "source_file": raw_item.get("_toolvista_file_path"),
                    "source_url": raw_item.get("_toolvista_source_url"),
                    "raw_index": raw_item.get("_toolvista_raw_index"),
                    "adapter": self.__class__.__name__,
                }
            ),
            metadata=self._compact_dict(
                {
                    "dataset_family": "ToolVQA",
                    "sample_type": raw_item.get("type"),
                    "original_question": raw_item.get("ori_question"),
                    "tool_count": len(context),
                    "retained_tool_count": len(tool_trace),
                    "reasoning_trace": reasoning_trace,
                    "conversion_mode": "tool_routing_only" if routing_only else "full_trajectory",
                    "final_output_is_media": source_answer_contains_media,
                    "source_final_answer": answer if routing_only else None,
                    "generated_media_available": False if routing_only else None,
                    "generated_media_placeholder": generated_media_placeholder,
                    "trajectory_truncated": routing_only,
                    "truncated_after_tool": generation_tool_name,
                    "discarded_tool_steps": (
                        len(context) - generation_step_index - 1 if generation_step_index is not None else None
                    ),
                }
            ),
        )
        return AdaptResult(sample=sample, warnings=warnings, raw_ref=raw_ref)

    def _annotation_path(self) -> Path:
        if self.annotation_file:
            return self.dataset_config.path(self.annotation_file)
        files = self.dataset_config.setting("files", {})
        if isinstance(files, Mapping):
            filename = files.get(self.split)
            if isinstance(filename, str) and filename:
                return self.dataset_config.path(filename)
        annotation_dir = self.dataset_config.setting("annotation_dir", "")
        return self.dataset_config.path(annotation_dir, f"{self.split}.jsonl")

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        emitted = 0
        with path.open("r", encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                if self.max_items is not None and emitted >= self.max_items:
                    return
                item = json.loads(line)
                yield self._with_source_fields(item, path, index)
                emitted += 1

    def _iter_json(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"ToolVQA annotation must be a JSON list or JSONL file: {path}")
        for index, item in enumerate(data):
            if self.max_items is not None and index >= self.max_items:
                return
            yield self._with_source_fields(item, path, index)

    def _with_source_fields(self, item: Any, path: Path, index: int) -> Dict[str, Any]:
        raw_item = dict(item) if isinstance(item, Mapping) else {"_toolvista_invalid_item": item}
        raw_item.setdefault("_toolvista_raw_index", index)
        raw_item["_toolvista_file_path"] = path.as_posix()
        return raw_item

    def _resolve_image_path(self, image_path: str) -> str:
        normalized = image_path.replace("\\", "/")
        if normalized.startswith("/") or Path(normalized).is_absolute():
            return normalized
        normalized = normalized.lstrip("/")
        google_drive_prefix = "datasets/toolvqa/"
        if normalized.lower().startswith(google_drive_prefix):
            normalized = normalized[len(google_drive_prefix):]
        media_dir = self.dataset_config.setting("media_dir", "")
        if media_dir and normalized.startswith(f"{str(media_dir).strip('/')}/"):
            return self.dataset_config.posix_path(normalized)
        if normalized.startswith("images/"):
            return self.dataset_config.posix_path(normalized)
        return self.dataset_config.posix_path(media_dir, normalized)

    @staticmethod
    def _parse_arguments(value: Any) -> tuple[Dict[str, Any], Optional[str]]:
        if isinstance(value, Mapping):
            return dict(value), None
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"raw_input": value}, "tool input is not valid JSON; preserved as raw_input."
            if isinstance(parsed, Mapping):
                return dict(parsed), None
            return {"value": parsed}, "tool input JSON is not an object; preserved under value."
        return {}, "tool input is empty; using empty arguments."

    @staticmethod
    def _normalize_media_arguments(
        arguments: Mapping[str, Any],
        original_image_path: str,
    ) -> tuple[Dict[str, Any], List[str]]:
        normalized = dict(arguments)
        unsupported_media_paths: List[str] = []
        original_image_name = Path(original_image_path).name
        for key, value in list(normalized.items()):
            if key.lower() in {"image", "image_path", "img"} and isinstance(value, str):
                if Path(value).name == original_image_name:
                    normalized[key] = "images[0]"
                else:
                    unsupported_media_paths.append(value)
        return normalized, unsupported_media_paths

    @staticmethod
    def _infer_parameters(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for key, value in arguments.items():
            json_type = "string"
            if isinstance(value, bool):
                json_type = "boolean"
            elif isinstance(value, int):
                json_type = "integer"
            elif isinstance(value, float):
                json_type = "number"
            elif isinstance(value, list):
                json_type = "array"
            elif isinstance(value, Mapping):
                json_type = "object"
            properties[str(key)] = {"type": json_type}
            required.append(str(key))
        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _parameters_for_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        catalog_entry = self.tool_catalog.get(tool_name)
        if catalog_entry is not None:
            return deepcopy(catalog_entry["parameters"])
        return self._infer_parameters(arguments)

    @staticmethod
    def _merge_parameter_schemas(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
        properties = dict(left.get("properties", {}))
        properties.update(right.get("properties", {}))
        right_required = set(right.get("required", []))
        required = [name for name in left.get("required", []) if name in right_required]
        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def _image_output_path(cls, tool_name: str, value: str) -> Optional[str]:
        if tool_name not in cls.image_output_tools:
            return None
        return cls._first_local_image_path(value)

    @classmethod
    def _first_local_image_path(cls, value: str) -> Optional[str]:
        normalized = value.replace("\\", "/")
        for match in cls.image_output_pattern.finditer(normalized):
            candidate = match.group(0)
            if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", candidate):
                continue
            return candidate
        return None

    @classmethod
    def _answer_references_generated_media(
        cls,
        answer: str,
        generated_media_placeholder: Optional[str],
    ) -> bool:
        normalized_answer = answer.replace("\\", "/")
        if generated_media_placeholder is not None:
            return generated_media_placeholder.replace("\\", "/") in normalized_answer

        stripped_answer = normalized_answer.strip().strip("\"'")
        match = cls.image_output_pattern.fullmatch(stripped_answer)
        if match is None:
            return False
        return re.match(r"(?i)^[a-z][a-z0-9+.-]*://", match.group(0)) is None

    def _raw_ref(self, raw_item: Any) -> Dict[str, Any]:
        if not isinstance(raw_item, Mapping):
            return {"source": self.source_name}
        return self._compact_dict(
            {
                "source": self.source_name,
                "source_file": raw_item.get("_toolvista_file_path"),
                "raw_index": raw_item.get("_toolvista_raw_index"),
                "image_path": raw_item.get("image_path"),
            }
        )

    @staticmethod
    def _value_to_string(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _non_empty_string(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _compact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}
