"""Adapter for Visual-RFT visual reasoning and grounding datasets."""

from __future__ import annotations

import io
import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional

from PIL import Image

from toolvista.adapters.base import BaseDatasetAdapter
from toolvista.schema.canonical import AdaptResult, CanonicalMessage, CanonicalSample, ContentPart, MediaBundle


DEFAULT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The assistant first reasons inside <think> </think> tags, "
    "then provides the final answer inside <answer> </answer> tags."
)


class VisualRFTAdapter(BaseDatasetAdapter):
    """Convert Visual-RFT prompt/image/solution rows into canonical RL prompts."""

    source_name: ClassVar[str] = "visual_rft"
    grounding_tasks: ClassVar[set[str]] = {"grounding", "detection", "object_detection", "gui_grounding"}

    def __init__(
        self,
        *args: Any,
        data_path: Optional[str] = None,
        split: Optional[str] = None,
        task: Optional[str] = None,
        max_items: Optional[int] = None,
        validate_media_files: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_path = data_path or str(self.dataset_config.setting("data_path", "train"))
        self.split = split or str(self.dataset_config.setting("default_split", "train"))
        configured_task = task if task is not None else self.dataset_config.setting("default_task")
        self.default_task = self._non_empty_string(configured_task)
        self.max_items = max_items
        self.validate_media_files = validate_media_files
        self.system_prompt = system_prompt or str(
            self.dataset_config.setting("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        media_dir = str(self.dataset_config.setting("materialized_media_dir", "materialized_images"))
        self.materialized_media_dir = self.dataset_config.path(media_dir, self.split)

    def iter_raw(self) -> Iterable[Dict[str, Any]]:
        source_path = self.dataset_config.path(self.data_path)
        for raw_index, item in enumerate(self._iter_source(source_path)):
            if self.max_items is not None and raw_index >= self.max_items:
                return
            raw_item = dict(item) if isinstance(item, Mapping) else {"_toolvista_invalid_item": item}
            raw_item["_toolvista_raw_index"] = raw_index
            raw_item["_toolvista_source_path"] = source_path.as_posix()
            raw_item.setdefault("_toolvista_split", self.split)
            yield raw_item

    def adapt(self, raw_item: Any) -> AdaptResult:
        raw_ref = self._raw_ref(raw_item)
        if not isinstance(raw_item, Mapping):
            return AdaptResult(errors=["Raw item is not a mapping."], raw_ref=raw_ref)
        read_error = self._non_empty_string(raw_item.get("_toolvista_read_error"))
        if read_error:
            return AdaptResult(errors=[read_error], raw_ref=raw_ref)

        problem = self._non_empty_string(raw_item.get("problem"))
        if not problem:
            problem = self._non_empty_string(raw_item.get("question"))
        if not problem:
            return AdaptResult(errors=["Missing or empty problem."], raw_ref=raw_ref)

        solution = self._non_empty_string(raw_item.get("solution"))
        if not solution:
            return AdaptResult(errors=["Missing or empty solution."], raw_ref=raw_ref)

        images, image_errors = self._resolve_images(raw_item.get("image", raw_item.get("images")), raw_item)
        if image_errors:
            return AdaptResult(errors=image_errors, raw_ref=raw_ref)
        if not images:
            return AdaptResult(errors=["Missing image."], raw_ref=raw_ref)

        task = self._normalize_task(raw_item.get("task"))
        if not task:
            return AdaptResult(
                errors=["Missing task; set a source task or pass an explicit default task."],
                raw_ref=raw_ref,
            )
        sample_id = self._sample_id(raw_item)
        content_parts = [ContentPart(type="image", media_ref=f"images[{index}]") for index in range(len(images))]
        content_parts.append(ContentPart(type="text", text=problem))

        original_answer = self._non_empty_string(raw_item.get("original_answer"))
        ground_truth = raw_item.get("ground_truth")
        if ground_truth is None:
            ground_truth = original_answer or solution

        labels = self._compact_dict(
            {
                "solution": solution,
                "ground_truth": ground_truth,
                "task": task,
                "original_answer": original_answer or None,
                "class_label": raw_item.get("class_label", raw_item.get("label")),
                "bbox": raw_item.get("bbox"),
                "objects": raw_item.get("objects"),
                "image_width": raw_item.get("image_width"),
                "image_height": raw_item.get("image_height"),
            }
        )
        sample = CanonicalSample(
            sample_id=sample_id,
            source=self.source_name,
            task_type="grounding" if task in self.grounding_tasks else "vqa",
            messages=[
                CanonicalMessage(role="system", content=self.system_prompt),
                CanonicalMessage(role="user", content=problem, content_parts=content_parts),
            ],
            media=MediaBundle(images=images),
            labels=labels,
            split=self._non_empty_string(raw_item.get("_toolvista_split")) or self.split,
            provenance=self._compact_dict(
                {
                    "source_dataset": "Visual-RFT",
                    "source_file": raw_item.get("_toolvista_source_path"),
                    "raw_index": raw_item.get("_toolvista_raw_index"),
                    "raw_id": raw_item.get("id"),
                    "adapter": self.__class__.__name__,
                }
            ),
            metadata=self._compact_dict(
                {
                    "dataset_family": "Visual-RFT",
                    "task": task,
                    "original_question": raw_item.get("original_question"),
                    "image_count": len(images),
                }
            ),
        )
        return AdaptResult(sample=sample, raw_ref=raw_ref)

    def _iter_source(self, path: Path) -> Iterable[Any]:
        if path.is_dir():
            try:
                from datasets import DatasetDict, load_from_disk
            except ImportError as exc:
                raise RuntimeError(
                    "Reading a Visual-RFT save_to_disk directory requires the 'datasets' package."
                ) from exc
            dataset = load_from_disk(str(path))
            if isinstance(dataset, DatasetDict):
                if self.split not in dataset:
                    raise KeyError(f"Visual-RFT dataset has no split {self.split!r}: {path}")
                dataset = dataset[self.split]
            yield from dataset
            return

        if not path.is_file():
            raise FileNotFoundError(f"Visual-RFT data path does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {
                            "_toolvista_read_error": f"Invalid JSONL at {path}:{line_number}: {exc}",
                            "_toolvista_source_line": line_number,
                        }
            return
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, Mapping) and self.split in data:
                data = data[self.split]
            if not isinstance(data, list):
                raise ValueError(f"Visual-RFT JSON must contain a list or split mapping: {path}")
            yield from data
            return
        raise ValueError(f"Unsupported Visual-RFT data source: {path}")

    def _resolve_images(self, value: Any, raw_item: Mapping[str, Any]) -> tuple[List[str], List[str]]:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        images: List[str] = []
        errors: List[str] = []
        for image_index, image_value in enumerate(values):
            if image_value is None:
                continue
            try:
                image_path = self._resolve_image(image_value, raw_item, image_index)
                if self.validate_media_files and not image_path.is_file():
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                images.append(image_path.resolve().as_posix())
            except (OSError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        return images, errors

    def _resolve_image(self, value: Any, raw_item: Mapping[str, Any], image_index: int) -> Path:
        if isinstance(value, (str, Path)):
            path = Path(value)
            return path if path.is_absolute() else self.dataset_config.path(path)
        if isinstance(value, Mapping):
            path_value = value.get("path")
            if isinstance(path_value, str) and path_value:
                path = Path(path_value)
                candidate = path if path.is_absolute() else self.dataset_config.path(path)
                if candidate.is_file() or value.get("bytes") is None:
                    return candidate
            byte_value = value.get("bytes")
            if isinstance(byte_value, (bytes, bytearray)):
                return self._materialize_image(Image.open(io.BytesIO(byte_value)), raw_item, image_index)
            raise ValueError("Image mapping must contain a path or bytes.")
        if isinstance(value, Image.Image):
            return self._materialize_image(value, raw_item, image_index)
        raise TypeError(f"Unsupported image value: {type(value)!r}")

    def _materialize_image(self, image: Image.Image, raw_item: Mapping[str, Any], image_index: int) -> Path:
        self.materialized_media_dir.mkdir(parents=True, exist_ok=True)
        token = self._safe_token(raw_item.get("id", raw_item.get("_toolvista_raw_index", "unknown")))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        output_path = self.materialized_media_dir / f"{token}-{digest}-{image_index}.png"
        if not output_path.is_file():
            temp_path = output_path.with_suffix(".tmp")
            temp_path.write_bytes(payload)
            temp_path.replace(output_path)
        return output_path

    def _sample_id(self, raw_item: Mapping[str, Any]) -> str:
        token = self._safe_token(raw_item.get("id", raw_item.get("_toolvista_raw_index", "unknown")))
        split = self._safe_token(raw_item.get("_toolvista_split", self.split))
        return f"visual-rft-{split}-{token}"

    def _normalize_task(self, value: Any) -> str:
        task = self._non_empty_string(value).lower().replace("-", "_").replace(" ", "_")
        return task or self.default_task.lower().replace("-", "_").replace(" ", "_")

    def _raw_ref(self, raw_item: Any) -> Dict[str, Any]:
        if not isinstance(raw_item, Mapping):
            return {"source": self.source_name}
        return self._compact_dict(
            {
                "source": self.source_name,
                "source_file": raw_item.get("_toolvista_source_path"),
                "raw_index": raw_item.get("_toolvista_raw_index"),
                "source_line": raw_item.get("_toolvista_source_line"),
                "raw_id": raw_item.get("id"),
            }
        )

    @staticmethod
    def _safe_token(value: Any) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
        return token or "unknown"

    @staticmethod
    def _non_empty_string(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _compact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}
