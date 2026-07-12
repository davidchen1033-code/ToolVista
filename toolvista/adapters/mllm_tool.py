"""Adapter for MLLM-Tool T+X-T instruction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence

from toolvista.adapters.base import BaseDatasetAdapter
from toolvista.schema.canonical import AdaptResult, CanonicalMessage, CanonicalSample, ContentPart, MediaBundle


class MLLMToolAdapter(BaseDatasetAdapter):
    """Convert MLLM-Tool JSON annotations into ToolVista canonical samples."""

    source_name: ClassVar[str] = "mllm_tool"
    default_file_keys: ClassVar[Sequence[str]] = ("image", "video", "audio", "text")

    role_map: ClassVar[Dict[str, str]] = {
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "tool_call": "tool_call",
        "tool_response": "tool_response",
    }
    media_field_by_modality: ClassVar[Dict[str, str]] = {
        "image": "image_path",
        "video": "video_path",
        "audio": "audio_path",
    }

    def __init__(
        self,
        *args: Any,
        file_keys: Optional[Sequence[str] | str] = None,
        max_items: Optional[int] = None,
        validate_media_files: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(file_keys, str):
            self.file_keys = (file_keys,)
        else:
            self.file_keys = tuple(file_keys or self.default_file_keys)
        self.max_items = max_items
        self.validate_media_files = validate_media_files

    def iter_raw(self) -> Iterable[Dict[str, Any]]:
        emitted = 0
        for file_key in self.file_keys:
            data_path = self._annotation_path(file_key)
            with data_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"MLLM-Tool annotation must be a list: {data_path}")

            for raw_index, item in enumerate(data):
                if self.max_items is not None and emitted >= self.max_items:
                    return
                emitted += 1

                if isinstance(item, Mapping):
                    raw_item = dict(item)
                else:
                    raw_item = {"_toolvista_invalid_item": item}
                raw_item["_toolvista_file_key"] = file_key
                raw_item["_toolvista_file_path"] = data_path.as_posix()
                raw_item["_toolvista_raw_index"] = raw_index
                yield raw_item

    def adapt(self, raw_item: Any) -> AdaptResult:
        raw_ref = self._raw_ref(raw_item)
        if not isinstance(raw_item, Mapping):
            return AdaptResult(errors=["Raw item is not a mapping."], raw_ref=raw_ref)

        conversations = raw_item.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return AdaptResult(errors=["Missing or empty conversations."], raw_ref=raw_ref)

        warnings: List[str] = []
        messages, message_source_indices = self._convert_messages(conversations, warnings)
        if not messages:
            return AdaptResult(errors=["No valid conversation messages."], warnings=warnings, raw_ref=raw_ref)

        input_modality = (self._string_or_empty(raw_item.get("input_modality")) or "text").lower()
        supported_modalities = {"text", *self.media_field_by_modality}
        if input_modality not in supported_modalities:
            return AdaptResult(
                errors=[f"Unsupported input modality: {input_modality}"],
                warnings=warnings,
                raw_ref=raw_ref,
            )

        media, media_warnings, media_occurrences = self._build_media(raw_item, input_modality, conversations)
        warnings.extend(media_warnings)

        errors = self._validate_media(input_modality, media_occurrences)
        errors.extend(self._attach_media_parts(conversations, messages, message_source_indices, media_occurrences))

        target_value = self._last_assistant_value(conversations)
        target_tools = self._normalize_target_tools(target_value)
        if not target_tools:
            errors.append("Could not infer a non-empty target tool from an assistant message.")
        if errors:
            return AdaptResult(errors=errors, warnings=warnings, raw_ref=raw_ref)

        target_tool_calls = self._build_target_tool_calls(target_tools, input_modality, media_occurrences)

        file_key = self._string_or_empty(raw_item.get("_toolvista_file_key")) or "unknown"
        raw_id = raw_item.get("id", raw_item.get("_toolvista_raw_index", "unknown"))
        raw_index = raw_item.get("_toolvista_raw_index", "unknown")
        sample = CanonicalSample(
            sample_id=f"mllm_tool-{file_key}-{raw_index}-{raw_id}",
            source=self.source_name,
            task_type="tool_selection",
            messages=messages,
            media=media,
            labels=self._compact_dict(
                {
                    "target_tools": target_tools,
                    "target_tool_calls": target_tool_calls,
                    "input_modality": input_modality,
                    "output_modality": raw_item.get("output_modality"),
                    "ambiguity": self._top_level_or_last_assistant_field(raw_item, conversations, "ambiguity"),
                    "ifambiguity": raw_item.get("ifambiguity"),
                    "onetomulti": raw_item.get("onetomulti"),
                }
            ),
            provenance=self._compact_dict(
                {
                    "source_dataset": "MLLM-Tool",
                    "source_file_key": file_key,
                    "source_file": raw_item.get("_toolvista_file_path"),
                    "raw_index": raw_item.get("_toolvista_raw_index"),
                    "raw_id": raw_item.get("id"),
                    "adapter": self.__class__.__name__,
                }
            ),
            metadata=self._compact_dict(
                {
                    "dataset_family": "MLLM-Tool",
                    "input_modality": input_modality,
                    "output_modality": raw_item.get("output_modality"),
                    "conversation_count": len(conversations),
                }
            ),
        )
        return AdaptResult(sample=sample, warnings=warnings, raw_ref=raw_ref)

    def _annotation_path(self, file_key: str) -> Path:
        files = self.dataset_config.setting("files", {})
        if not isinstance(files, Mapping):
            raise ValueError("MLLM-Tool dataset config field 'files' must be a mapping.")
        filename = files.get(file_key, file_key)
        if not isinstance(filename, str) or not filename:
            raise KeyError(f"Unknown MLLM-Tool file key: {file_key}")
        return self.dataset_config.path(self.dataset_config.required_setting("instruction_dir"), filename)

    def _build_media(
        self,
        raw_item: Mapping[str, Any],
        input_modality: str,
        conversations: Sequence[Any],
    ) -> tuple[MediaBundle, List[str], List[Dict[str, Any]]]:
        warnings: List[str] = []
        media = MediaBundle()
        occurrences: List[Dict[str, Any]] = []

        for source_message_index, message in enumerate(conversations):
            if not isinstance(message, Mapping):
                continue
            for modality in self.media_field_by_modality:
                for relative_media_path in self._extract_media_paths(message, modality):
                    media_ref, media_path = self._append_media(media, modality, relative_media_path)
                    occurrences.append(
                        {
                            "media_ref": media_ref,
                            "path": media_path,
                            "input_modality": modality,
                            "source": "conversation",
                            "source_message_index": source_message_index,
                        }
                    )

        if occurrences:
            return media, warnings, occurrences

        field_name = self.media_field_by_modality.get(input_modality)
        if field_name is None:
            if input_modality != "text":
                warnings.append(f"Unsupported input modality: {input_modality}")
            return media, warnings, occurrences

        relative_media_paths = self._extract_media_paths(raw_item, input_modality)
        if not relative_media_paths:
            return media, warnings, occurrences

        for relative_media_path in relative_media_paths:
            media_ref, media_path = self._append_media(media, input_modality, relative_media_path)
            occurrences.append(
                {
                    "media_ref": media_ref,
                    "path": media_path,
                    "input_modality": input_modality,
                    "source": "sample",
                    "source_message_index": None,
                }
            )
        return media, warnings, occurrences

    def _validate_media(
        self,
        input_modality: str,
        media_occurrences: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        errors: List[str] = []
        if input_modality != "text" and not any(
            occurrence.get("input_modality") == input_modality for occurrence in media_occurrences
        ):
            field_name = self.media_field_by_modality[input_modality]
            errors.append(f"Missing required media path for modality {input_modality}: {field_name}")

        if self.validate_media_files:
            for occurrence in media_occurrences:
                media_path = occurrence.get("path")
                if isinstance(media_path, str) and not Path(media_path).is_file():
                    errors.append(f"Media file does not exist: {media_path}")
        return errors

    def _media_path(self, input_modality: str, relative_media_path: str) -> str:
        media_dir = self.dataset_config.required_setting("media_dir")
        if input_modality == "image":
            return self.dataset_config.posix_path(media_dir, "image", relative_media_path)
        if input_modality == "audio":
            return self.dataset_config.posix_path(media_dir, "audio", relative_media_path)
        return self.dataset_config.posix_path(media_dir, relative_media_path)

    def _append_media(self, media: MediaBundle, input_modality: str, relative_media_path: str) -> tuple[str, str]:
        media_path = self._media_path(input_modality, relative_media_path)
        if input_modality == "image":
            media.images.append(media_path)
            return f"images[{len(media.images) - 1}]", media_path
        if input_modality == "video":
            media.videos.append(media_path)
            return f"videos[{len(media.videos) - 1}]", media_path
        if input_modality == "audio":
            media.audios.append(media_path)
            return f"audios[{len(media.audios) - 1}]", media_path
        raise ValueError(f"Unsupported media modality: {input_modality}")

    def _extract_media_paths(self, item: Mapping[str, Any], input_modality: str) -> List[str]:
        field_name = self.media_field_by_modality.get(input_modality)
        if field_name is None:
            return []
        aliases = (
            field_name,
            field_name.replace("_path", "_paths"),
            field_name.replace("_path", ""),
            f"{input_modality}s",
        )
        paths: List[str] = []
        for alias in aliases:
            value = item.get(alias)
            if isinstance(value, str) and value:
                paths.append(value)
            elif isinstance(value, list):
                paths.extend(path for path in value if isinstance(path, str) and path)
        return paths

    def _convert_messages(
        self,
        conversations: Sequence[Any],
        warnings: List[str],
    ) -> tuple[List[CanonicalMessage], List[int]]:
        messages: List[CanonicalMessage] = []
        source_indices: List[int] = []
        for index, message in enumerate(conversations):
            if not isinstance(message, Mapping):
                warnings.append(f"Skip non-mapping conversation message at index {index}.")
                continue

            raw_role = self._string_or_empty(message.get("from")).lower()
            role = self.role_map.get(raw_role)
            if role is None:
                warnings.append(f"Skip unsupported conversation role at index {index}: {raw_role!r}")
                continue

            content = self._value_to_content(message.get("value"))
            has_media = any(self._extract_media_paths(message, modality) for modality in self.media_field_by_modality)
            if not content and not (role == "user" and has_media):
                warnings.append(f"Skip empty conversation content at index {index}.")
                continue
            content_parts = [ContentPart(type="text", text=content)] if content else []
            messages.append(CanonicalMessage(role=role, content=content, content_parts=content_parts))
            source_indices.append(index)
        return messages, source_indices

    def _attach_media_parts(
        self,
        conversations: Sequence[Any],
        messages: List[CanonicalMessage],
        message_source_indices: Sequence[int],
        media_occurrences: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        if not media_occurrences:
            return []

        user_targets = self._user_message_targets(conversations, messages, message_source_indices)
        if not user_targets:
            return ["Media is present but no valid user message is available to reference it."]

        parts_by_message_index: Dict[int, List[ContentPart]] = {}
        attached_media_refs: set[str] = set()
        for occurrence in media_occurrences:
            target = self._target_for_media_occurrence(occurrence, user_targets)
            if target is None:
                continue
            message_index = int(target["message_index"])
            modality = occurrence.get("input_modality")
            media_ref = occurrence.get("media_ref")
            if modality not in self.media_field_by_modality or not isinstance(media_ref, str):
                continue
            parts_by_message_index.setdefault(message_index, []).append(ContentPart(type=modality, media_ref=media_ref))
            attached_media_refs.add(media_ref)

        for message_index, media_parts in parts_by_message_index.items():
            message = messages[message_index]
            text_parts = [part for part in message.content_parts if part.type == "text"]
            other_parts = [part for part in message.content_parts if part.type != "text"]
            message.content_parts = media_parts + other_parts + text_parts

        expected_media_refs = {
            str(occurrence["media_ref"])
            for occurrence in media_occurrences
            if isinstance(occurrence.get("media_ref"), str)
        }
        missing_media_refs = sorted(expected_media_refs - attached_media_refs)
        if missing_media_refs:
            return [f"Could not attach media references to a user message: {', '.join(missing_media_refs)}"]
        return []

    def _user_message_targets(
        self,
        conversations: Sequence[Any],
        messages: Sequence[CanonicalMessage],
        message_source_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        user_turn_index = 0
        targets: List[Dict[str, Any]] = []
        for message_index, message in enumerate(messages):
            if message.role != "user":
                continue
            source_message_index = message_source_indices[message_index]
            source_message = conversations[source_message_index]
            source_modality = ""
            if isinstance(source_message, Mapping):
                source_modality = self._string_or_empty(source_message.get("input_modality"))
            targets.append(
                {
                    "message_index": message_index,
                    "source_message_index": source_message_index,
                    "turn_index": user_turn_index,
                    "role": message.role,
                    "input_modality": source_modality,
                }
            )
            user_turn_index += 1
        return targets

    def _target_for_media_occurrence(
        self,
        occurrence: Mapping[str, Any],
        user_targets: Sequence[Mapping[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        source_message_index = occurrence.get("source_message_index")
        if source_message_index is not None:
            for target in user_targets:
                if target["source_message_index"] == source_message_index:
                    return target

        occurrence_modality = occurrence.get("input_modality")
        for target in user_targets:
            if target.get("input_modality") == occurrence_modality:
                return target

        return user_targets[0] if user_targets else None

    def _build_target_tool_calls(
        self,
        target_tools: Sequence[str],
        input_modality: str,
        media_occurrences: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        media_refs = [
            str(occurrence["media_ref"])
            for occurrence in media_occurrences
            if isinstance(occurrence.get("media_ref"), str)
        ]

        calls: List[Dict[str, Any]] = []
        for tool_id in target_tools:
            arguments: Dict[str, Any] = {"input_modality": input_modality}
            if media_refs:
                arguments["media_refs"] = list(media_refs)
            calls.append(
                {
                    "action": "select_tool",
                    "tool_id": tool_id,
                    "arguments": arguments,
                }
            )
        return calls

    def _last_assistant_value(self, conversations: Sequence[Any]) -> Any:
        return self._last_assistant_field(conversations, "value")

    def _last_assistant_field(self, conversations: Sequence[Any], field_name: str) -> Any:
        for message in reversed(conversations):
            if not isinstance(message, Mapping):
                continue
            role = self.role_map.get(self._string_or_empty(message.get("from")).lower())
            if role == "assistant":
                return message.get(field_name)
        return None

    def _top_level_or_last_assistant_field(
        self,
        raw_item: Mapping[str, Any],
        conversations: Sequence[Any],
        field_name: str,
    ) -> Any:
        top_level_value = raw_item.get(field_name)
        if top_level_value is not None:
            return top_level_value
        return self._last_assistant_field(conversations, field_name)

    def _normalize_target_tools(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [self._value_to_content(item) for item in value if self._value_to_content(item)]
        content = self._value_to_content(value)
        return [content] if content else []

    def _raw_ref(self, raw_item: Any) -> Dict[str, Any]:
        if not isinstance(raw_item, Mapping):
            return {"source": self.source_name}
        return self._compact_dict(
            {
                "source": self.source_name,
                "file_key": raw_item.get("_toolvista_file_key"),
                "source_file": raw_item.get("_toolvista_file_path"),
                "raw_index": raw_item.get("_toolvista_raw_index"),
                "raw_id": raw_item.get("id"),
            }
        )

    @staticmethod
    def _value_to_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""

    @staticmethod
    def _compact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}
