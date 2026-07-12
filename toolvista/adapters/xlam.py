"""Adapter for xLAM ActionStudio unified tool-use trajectories."""

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import ijson
except ImportError:  # pragma: no cover - exercised only in incomplete environments.
    ijson = None

from toolvista.adapters.base import BaseDatasetAdapter
from toolvista.schema.canonical import AdaptResult, CanonicalMessage, CanonicalSample, ToolSpec


JSON_SCHEMA_TYPES = {"null", "boolean", "object", "array", "number", "string", "integer"}
TYPE_ALIASES = {
    **{schema_type: schema_type for schema_type in JSON_SCHEMA_TYPES},
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "none": "null",
    "nonetype": "null",
}
OPTIONAL_TYPE_SUFFIX = re.compile(r"\s*,\s*optional\s*$", re.IGNORECASE)


class XLAMAdapter(BaseDatasetAdapter):
    """Split each xLAM trajectory into one canonical sample per assistant turn."""

    source_name = "xlam"

    def __init__(
        self,
        *args: Any,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        annotation_file: Optional[str] = None,
        max_trajectories: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.subset = subset or str(self.dataset_config.setting("default_subset", "actionstudio-98k"))
        self.split = split or str(self.dataset_config.setting("default_split", "train"))
        self.annotation_file = annotation_file
        self.max_trajectories = max_trajectories

    def iter_raw(self) -> Iterable[Dict[str, Any]]:
        if ijson is None:
            raise RuntimeError("xLAM streaming conversion requires ijson. Install it with: pip install ijson")

        annotation_path = self._annotation_path()
        with annotation_path.open("rb") as handle:
            for index, item in enumerate(ijson.items(handle, "item", use_float=True)):
                if self.max_trajectories is not None and index >= self.max_trajectories:
                    return
                raw_item = dict(item) if isinstance(item, Mapping) else {"_xlam_invalid_item": item}
                raw_item.setdefault("_toolvista_raw_index", index)
                raw_item["_toolvista_file_path"] = annotation_path.as_posix()
                yield raw_item

    def adapt(self, raw_item: Any) -> Iterable[AdaptResult]:
        raw_ref = self._raw_ref(raw_item)
        if not isinstance(raw_item, Mapping):
            return [AdaptResult(errors=["Raw item is not a mapping."], raw_ref=raw_ref)]

        trajectory_id = self._non_empty_string(raw_item.get("unique_trajectory_id"))
        task_instruction = self._non_empty_string(raw_item.get("task_instruction"))
        conversation = raw_item.get("conversation")
        if not trajectory_id:
            return [AdaptResult(errors=["Missing or empty unique_trajectory_id."], raw_ref=raw_ref)]
        if not task_instruction:
            return [AdaptResult(errors=["Missing or empty task_instruction."], raw_ref=raw_ref)]
        if not isinstance(conversation, list) or not conversation:
            return [AdaptResult(errors=["Missing or empty conversation."], raw_ref=raw_ref)]

        tools, tool_errors = self._parse_tools(raw_item.get("tools"))
        if tool_errors:
            return [AdaptResult(errors=tool_errors, raw_ref=raw_ref)]

        results: List[AdaptResult] = []
        history_messages: List[CanonicalMessage] = []
        history_warnings: List[str] = []
        history_errors: List[str] = []
        for target_index, target_message in enumerate(conversation):
            if isinstance(target_message, Mapping) and target_message.get("role") == "assistant":
                results.append(
                    self._adapt_target(
                        raw_item=raw_item,
                        message=target_message,
                        message_index=target_index,
                        previous_message=conversation[target_index - 1] if target_index else None,
                        trajectory_id=trajectory_id,
                        task_instruction=task_instruction,
                        tools=tools,
                        history_messages=history_messages,
                        history_warnings=history_warnings,
                        history_errors=history_errors,
                    )
                )

            converted, warnings, errors = self._convert_history([target_message], start_index=target_index)
            history_messages.extend(converted)
            history_warnings.extend(warnings)
            history_errors.extend(errors)

        if not results:
            return [AdaptResult(errors=["Conversation contains no assistant target messages."], raw_ref=raw_ref)]
        return results

    def _adapt_target(
        self,
        *,
        raw_item: Mapping[str, Any],
        message: Mapping[str, Any],
        message_index: int,
        previous_message: Any,
        trajectory_id: str,
        task_instruction: str,
        tools: Sequence[ToolSpec],
        history_messages: Sequence[CanonicalMessage],
        history_warnings: Sequence[str],
        history_errors: Sequence[str],
    ) -> AdaptResult:
        raw_ref = self._raw_ref(raw_item, assistant_index=message_index)
        warnings = list(history_warnings)
        errors = list(history_errors)
        if isinstance(previous_message, Mapping) and previous_message.get("role") == "assistant":
            errors.append(
                f"Assistant target {message_index} follows another assistant message; "
                "consecutive assistant messages would be merged by ms-swift and can corrupt the target loss."
            )
            return AdaptResult(errors=errors, warnings=warnings, raw_ref=raw_ref)

        target_messages, details, target_errors = self._convert_assistant(
            message,
            message_index,
            history=False,
        )
        errors.extend(target_errors)
        if errors:
            return AdaptResult(errors=errors, warnings=warnings, raw_ref=raw_ref)

        target_calls = details.get("target_tool_calls", [])
        target_position = message_index + 1
        sample = CanonicalSample(
            sample_id=f"{trajectory_id}---{target_position}",
            source=self.source_name,
            task_type="agent_trajectory",
            messages=[CanonicalMessage(role="system", content=task_instruction)]
            + list(history_messages)
            + target_messages,
            tools=deepcopy(list(tools)),
            labels=self._compact_dict(
                {
                    "final_answer": details.get("final_answer"),
                    "target_kind": details["target_kind"],
                    "target_tool_calls": target_calls or None,
                    "target_tool_call_ids": details.get("target_tool_call_ids"),
                }
            ),
            split=self.split,
            provenance=self._compact_dict(
                {
                    "source_dataset": "xLAM ActionStudio",
                    "source_subset": self.subset,
                    "source_file": raw_item.get("_toolvista_file_path"),
                    "raw_index": raw_item.get("_toolvista_raw_index"),
                    "source_assistant_index": message_index,
                    "adapter": self.__class__.__name__,
                }
            ),
            metadata=self._compact_dict(
                {
                    "dataset_family": "xLAM",
                    "trajectory_id": trajectory_id,
                    "source_message_index": target_position,
                    "source_assistant_content": details.get("source_assistant_content"),
                    "source_had_empty_tool_calls": details.get("source_had_empty_tool_calls"),
                    "target_kind": details["target_kind"],
                    "target_call_count": len(target_calls),
                    "parallel_tool_calls": len(target_calls) > 1,
                    "history_source_message_count": message_index,
                    "tool_count": len(tools),
                    "few_shot_example_count": self._sequence_length(raw_item.get("few_shot_examples")),
                    "source_format_instruction": raw_item.get("format_instruction"),
                }
            ),
        )
        return AdaptResult(sample=sample, warnings=warnings, raw_ref=raw_ref)

    def _annotation_path(self) -> Path:
        if self.annotation_file:
            return self.dataset_config.path(self.annotation_file)
        unified_data_dir = str(
            self.dataset_config.setting("unified_data_dir", "actionstudio/datasets/unified_data")
        )
        return self.dataset_config.path(unified_data_dir, self.subset, self.split, f"{self.split}.json")

    def _parse_tools(self, value: Any) -> Tuple[List[ToolSpec], List[str]]:
        if not isinstance(value, list):
            return [], ["Tools must be a list."]

        tools: List[ToolSpec] = []
        tools_by_name: Dict[str, ToolSpec] = {}
        errors: List[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                errors.append(f"Tool {index} is not a mapping.")
                continue
            if item.get("type", "function") != "function":
                errors.append(f"Tool {index} has unsupported type: {item.get('type')!r}.")
                continue
            function = item.get("function")
            if not isinstance(function, Mapping):
                errors.append(f"Tool {index} has no function mapping.")
                continue
            name = self._non_empty_string(function.get("name"))
            if not name:
                errors.append(f"Tool {index} has no function name.")
                continue
            description_value = function.get("description", "")
            description = description_value if isinstance(description_value, str) else str(description_value)
            parameters = function.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, Mapping):
                errors.append(f"Tool {index} parameters must be a mapping.")
                continue
            try:
                normalized_parameters, _ = self._normalize_schema(
                    parameters,
                    path=f"tools[{index}].function.parameters",
                )
            except ValueError as exc:
                errors.append(f"Tool {index} ({name!r}) has invalid parameters: {exc}")
                continue

            tool = ToolSpec(name=name, description=description, parameters=normalized_parameters)
            previous = tools_by_name.get(name)
            if previous is None:
                tools_by_name[name] = tool
                tools.append(tool)
            elif previous != tool:
                errors.append(f"Tool name {name!r} has conflicting definitions at tool index {index}.")
        return tools, errors

    def _normalize_schema(self, value: Mapping[str, Any], *, path: str) -> Tuple[Dict[str, Any], bool]:
        schema = deepcopy(dict(value))
        normalized: Dict[str, Any] = {}
        optional = False

        if "type" in schema:
            type_fragment, optional = self._normalize_type(schema.pop("type"), path=f"{path}.type")
            self._merge_schema_fragments(normalized, type_fragment, path=path)

        properties = schema.pop("properties", None)
        required = schema.pop("required", None)
        optional_properties: set[str] = set()
        if properties is not None:
            if not isinstance(properties, Mapping):
                raise ValueError(f"{path}.properties must be a mapping.")
            normalized_properties: Dict[str, Any] = {}
            for property_name, property_schema in properties.items():
                if not isinstance(property_name, str) or not isinstance(property_schema, Mapping):
                    raise ValueError(f"{path}.properties entries must map string names to schemas.")
                normalized_property, property_optional = self._normalize_schema(
                    property_schema,
                    path=f"{path}.properties.{property_name}",
                )
                normalized_properties[property_name] = normalized_property
                if property_optional:
                    optional_properties.add(property_name)
            normalized["properties"] = normalized_properties

        if required is not None:
            if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
                raise ValueError(f"{path}.required must be a list of property names.")
            normalized["required"] = [name for name in required if name not in optional_properties]

        for keyword in ("items", "additionalProperties"):
            nested = schema.pop(keyword, None)
            if nested is None:
                continue
            if isinstance(nested, bool) and keyword == "additionalProperties":
                normalized_nested: Any = nested
            elif isinstance(nested, Mapping):
                normalized_nested, _ = self._normalize_schema(nested, path=f"{path}.{keyword}")
            else:
                raise ValueError(f"{path}.{keyword} must be a schema mapping.")
            self._merge_schema_fragments(normalized, {keyword: normalized_nested}, path=path)

        self._merge_schema_fragments(normalized, schema, path=path)
        self._coerce_exact_default(normalized)
        return normalized, optional

    def _normalize_type(self, value: Any, *, path: str) -> Tuple[Dict[str, Any], bool]:
        if isinstance(value, list):
            normalized_types: List[str] = []
            optional = False
            for index, item in enumerate(value):
                fragment, item_optional = self._normalize_type(item, path=f"{path}[{index}]")
                if set(fragment) != {"type"} or not isinstance(fragment["type"], str):
                    raise ValueError(f"{path} contains a complex type that cannot be represented in a JSON Schema type array.")
                normalized_types.append(fragment["type"])
                optional = optional or item_optional
            return {"type": list(dict.fromkeys(normalized_types))}, optional

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a non-empty string or list of strings.")

        type_text = value.strip()
        suffix_optional = OPTIONAL_TYPE_SUFFIX.search(type_text) is not None
        if suffix_optional:
            type_text = OPTIONAL_TYPE_SUFFIX.sub("", type_text).strip()
        try:
            expression = ast.parse(type_text, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"{path} has unsupported or ambiguous Python type {value!r}.") from exc

        fragment, expression_optional = self._python_type_to_schema(expression, source=value, path=path)
        return fragment, suffix_optional or expression_optional

    def _python_type_to_schema(
        self,
        expression: ast.expr,
        *,
        source: str,
        path: str,
    ) -> Tuple[Dict[str, Any], bool]:
        name = self._type_expression_name(expression)
        if name is not None:
            lowered = name.lower()
            if lowered in TYPE_ALIASES:
                return {"type": TYPE_ALIASES[lowered]}, False
            if lowered == "dict":
                return {"type": "object"}, False
            if lowered == "list":
                return {"type": "array"}, False
            if lowered == "set":
                return {"type": "array", "uniqueItems": True}, False
            if lowered == "tuple":
                return {"type": "array"}, False
            raise ValueError(f"{path} has unsupported or ambiguous Python type {source!r}.")

        if not isinstance(expression, ast.Subscript):
            raise ValueError(f"{path} has unsupported or ambiguous Python type {source!r}.")

        container_name = self._type_expression_name(expression.value)
        if container_name is None:
            raise ValueError(f"{path} has unsupported or ambiguous Python type {source!r}.")
        container = container_name.lower()
        arguments = list(expression.slice.elts) if isinstance(expression.slice, ast.Tuple) else [expression.slice]

        if container in {"list", "set"} and len(arguments) == 1:
            item_schema, item_optional = self._python_type_to_schema(arguments[0], source=source, path=path)
            if item_optional:
                item_schema = self._with_null_type(item_schema)
            result: Dict[str, Any] = {"type": "array", "items": item_schema}
            if container == "set":
                result["uniqueItems"] = True
            return result, False

        if container == "dict" and len(arguments) == 2:
            key_schema, key_optional = self._python_type_to_schema(arguments[0], source=source, path=path)
            if key_optional or key_schema != {"type": "string"}:
                raise ValueError(f"{path} has a mapping key type that cannot be represented by JSON object keys: {source!r}.")
            value_schema, value_optional = self._python_type_to_schema(arguments[1], source=source, path=path)
            if value_optional:
                value_schema = self._with_null_type(value_schema)
            return {"type": "object", "additionalProperties": value_schema}, False

        if container == "tuple" and arguments:
            item_schemas = []
            for argument in arguments:
                item_schema, item_optional = self._python_type_to_schema(argument, source=source, path=path)
                if item_optional:
                    item_schema = self._with_null_type(item_schema)
                item_schemas.append(item_schema)
            if any(item != item_schemas[0] for item in item_schemas[1:]):
                raise ValueError(f"{path} has a heterogeneous tuple type that is not safely representable: {source!r}.")
            return {
                "type": "array",
                "items": item_schemas[0],
                "minItems": len(item_schemas),
                "maxItems": len(item_schemas),
            }, False

        if container == "optional" and len(arguments) == 1:
            fragment, _ = self._python_type_to_schema(arguments[0], source=source, path=path)
            return fragment, True

        if container == "union" and arguments:
            fragments: List[Dict[str, Any]] = []
            optional = False
            for argument in arguments:
                fragment, argument_optional = self._python_type_to_schema(argument, source=source, path=path)
                if fragment == {"type": "null"}:
                    optional = True
                    continue
                optional = optional or argument_optional
                if fragment not in fragments:
                    fragments.append(fragment)
            if not fragments:
                return {"type": "null"}, optional
            if len(fragments) == 1:
                return fragments[0], optional
            return {"anyOf": fragments}, optional

        raise ValueError(f"{path} has unsupported or ambiguous Python type {source!r}.")

    @staticmethod
    def _type_expression_name(expression: ast.expr) -> Optional[str]:
        if isinstance(expression, ast.Name):
            return expression.id
        if isinstance(expression, ast.Constant) and expression.value is None:
            return "None"
        return None

    @staticmethod
    def _with_null_type(schema: Dict[str, Any]) -> Dict[str, Any]:
        if not schema:
            return schema
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            return {**schema, "type": [schema_type, "null"]}
        return {"anyOf": [schema, {"type": "null"}]}

    @staticmethod
    def _merge_schema_fragments(base: Dict[str, Any], incoming: Mapping[str, Any], *, path: str) -> None:
        for key, value in incoming.items():
            if key in base and base[key] != value:
                raise ValueError(f"{path} contains conflicting schema definitions for {key!r}.")
            base[key] = value

    @staticmethod
    def _coerce_exact_default(schema: Dict[str, Any]) -> None:
        value = schema.get("default")
        schema_type = schema.get("type")
        if not isinstance(value, str) or not isinstance(schema_type, str):
            return
        stripped = value.strip()
        if schema_type == "integer" and re.fullmatch(r"[+-]?\d+", stripped):
            schema["default"] = int(stripped)
        elif schema_type == "number":
            try:
                schema["default"] = float(stripped)
            except ValueError:
                pass
        elif schema_type == "boolean" and stripped.lower() in {"true", "false"}:
            schema["default"] = stripped.lower() == "true"

    def _convert_history(
        self,
        messages: Sequence[Any],
        *,
        start_index: int = 0,
    ) -> Tuple[List[CanonicalMessage], List[str], List[str]]:
        converted: List[CanonicalMessage] = []
        warnings: List[str] = []
        errors: List[str] = []
        for index, message in enumerate(messages, start=start_index):
            if not isinstance(message, Mapping):
                errors.append(f"Conversation message {index} is not a mapping.")
                continue
            role = message.get("role")
            if role in {"system", "user"}:
                content = self._non_empty_string(message.get("content"))
                if not content:
                    errors.append(f"Conversation {role} message {index} has empty content.")
                    continue
                converted.append(CanonicalMessage(role=role, content=content))
            elif role == "assistant":
                content = self._non_empty_string(message.get("content"))
                no_calls = "tool_calls" not in message or message.get("tool_calls") == []
                if not content and no_calls:
                    warnings.append(f"Dropped empty assistant history message {index}.")
                    continue
                assistant_messages, _, assistant_errors = self._convert_assistant(
                    message,
                    index,
                    history=True,
                )
                converted.extend(assistant_messages)
                errors.extend(assistant_errors)
            elif role in {"tool", "tool_response"}:
                converted.append(CanonicalMessage(role="tool_response", content=self._value_to_string(message.get("content"))))
            else:
                errors.append(f"Conversation message {index} has unsupported role: {role!r}.")
        return converted, warnings, errors

    def _convert_assistant(
        self,
        message: Mapping[str, Any],
        message_index: int,
        *,
        history: bool,
    ) -> Tuple[List[CanonicalMessage], Dict[str, Any], List[str]]:
        content = self._non_empty_string(message.get("content"))
        tool_calls_present = "tool_calls" in message
        tool_calls_value = message.get("tool_calls")
        if tool_calls_present and not isinstance(tool_calls_value, list):
            return [], {}, [f"Assistant message {message_index} tool_calls must be a list."]

        tool_calls = tool_calls_value if isinstance(tool_calls_value, list) else []
        if tool_calls:
            call_messages, call_details, call_errors = self._convert_tool_calls(
                tool_calls,
                message_index,
                loss=False if history else None,
            )
            if call_errors:
                return [], {}, call_errors
            messages: List[CanonicalMessage] = []
            if history and content:
                messages.append(CanonicalMessage(role="assistant", content=content, loss=False))
            messages.extend(call_messages)
            return (
                messages,
                {
                    "target_kind": "tool_calls",
                    "target_tool_calls": call_details["calls"],
                    "target_tool_call_ids": call_details["ids"],
                    "source_assistant_content": content or None,
                },
                [],
            )

        if content:
            return (
                [CanonicalMessage(role="assistant", content=content, loss=False if history else None)],
                {
                    "target_kind": "assistant_text",
                    "final_answer": content,
                    "source_had_empty_tool_calls": tool_calls_present,
                },
                [],
            )

        return [], {}, [f"Assistant message {message_index} has neither tool calls nor non-empty content."]

    def _convert_tool_calls(
        self,
        tool_calls: Sequence[Any],
        message_index: int,
        *,
        loss: Optional[bool],
    ) -> Tuple[List[CanonicalMessage], Dict[str, Any], List[str]]:
        messages: List[CanonicalMessage] = []
        calls: List[Dict[str, Any]] = []
        call_ids: List[Any] = []
        errors: List[str] = []
        for call_index, call in enumerate(tool_calls):
            prefix = f"Assistant message {message_index} tool call {call_index}"
            if not isinstance(call, Mapping):
                errors.append(f"{prefix} is not a mapping.")
                continue
            if call.get("type", "function") != "function":
                errors.append(f"{prefix} has unsupported type: {call.get('type')!r}.")
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                errors.append(f"{prefix} has no function mapping.")
                continue
            name = self._non_empty_string(function.get("name"))
            if not name:
                errors.append(f"{prefix} has no function name.")
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, Mapping):
                errors.append(f"{prefix} arguments must be a mapping.")
                continue
            normalized_call = {"name": name, "arguments": dict(arguments)}
            calls.append(normalized_call)
            call_ids.append(call.get("id"))
            messages.append(
                CanonicalMessage(
                    role="tool_call",
                    content=json.dumps(normalized_call, ensure_ascii=False, separators=(",", ":")),
                    loss=loss,
                )
            )
        return messages, {"calls": calls, "ids": call_ids}, errors

    def _raw_ref(self, raw_item: Any, *, assistant_index: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(raw_item, Mapping):
            return {"source": self.source_name}
        return self._compact_dict(
            {
                "source": self.source_name,
                "source_file": raw_item.get("_toolvista_file_path"),
                "raw_index": raw_item.get("_toolvista_raw_index"),
                "trajectory_id": raw_item.get("unique_trajectory_id"),
                "assistant_index": assistant_index,
            }
        )

    @staticmethod
    def _value_to_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _non_empty_string(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _sequence_length(value: Any) -> Optional[int]:
        return len(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else None

    @staticmethod
    def _compact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}
