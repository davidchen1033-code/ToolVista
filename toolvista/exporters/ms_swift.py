"""Export ToolVista canonical samples to ms-swift JSONL rows."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from toolvista.schema.canonical import (
    CanonicalMessage,
    CanonicalSample,
    ContentPart,
    MediaBundle,
    PreferencePair,
)

# 负责把 ToolVista 内部的媒体类型转换成 ms-swift 消息文本中的媒体占位符
MEDIA_TAGS = {
    "image": "<image>",
    "video": "<video>",
    "audio": "<audio>",
}


class MsSwiftExportError(ValueError):
    """
    定义一个新的异常类 MsSwiftExportError，并让它继承 ValueError，当一个规范化样本无法被转换成 ms-swift 格式时抛出
    Raised when a canonical sample cannot be lowered to ms-swift format.
    """


def canonical_to_ms_swift_row(sample: CanonicalSample, *, include_metadata: bool = False) -> Dict[str, Any]:
    """
    单个样本转换函数
    Lower one canonical sample to one ms-swift-compatible dataset row.
    """

    media_state = _MediaExportState(sample.media)
    # 每个 CanonicalMessage 会被 _message_to_ms_swift() 转换成普通字典
    row: Dict[str, Any] = {
        "messages": [_message_to_ms_swift(message, media_state) for message in sample.messages],
    }

    # 如果没有图片，images 键会被直接省略，而不是输出
    if media_state.images:
        row["images"] = media_state.images
    if media_state.videos:
        row["videos"] = media_state.videos
    if media_state.audios:
        row["audios"] = media_state.audios
    # 把每个 ToolSpec 转成 OpenAI 风格工具定义，然后json.dumps把工具列表序列化为 JSON 字符串，所以row["tools"] 不是 Python 列表，而是一个字符串
    if sample.tools:
        row["tools"] = json.dumps([tool.to_openai_tool() for tool in sample.tools], ensure_ascii=False)
    if sample.objects is not None:
        row["objects"] = sample.objects
    # 如果样本包含 PreferencePair，说明这不是普通 SFT 样本，而是带有 chosen/rejected 的偏好训练数据，例如 DPO 数据
    if sample.preference is not None:
        _assert_chosen_matches_messages(sample)
        row.update(_preference_to_ms_swift(sample.preference, sample.messages))
    if include_metadata:
        row["metadata"] = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "labels": sample.labels,
            "provenance": sample.provenance,
            "metadata": sample.metadata,
        }

    # 检查 sample.media 中的每一个媒体项是否至少被消息引用过一次
    media_state.assert_all_referenced()
    return row


def export_jsonl(
    samples: Iterable[CanonicalSample], # 输入不是限定为 List，而是更宽泛的 Iterable
    output_path: str | Path,
    *,
    include_metadata: bool = False,
) -> int:
    """
    批量导出 JSONL
    Write canonical samples as ms-swift JSONL and return the row count.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True) # parents=True 表示缺少多层父目录时递归创建；exist_ok=True 表示目录已经存在也不报错
    temp_path: Optional[Path] = None # temp_path 用来记录临时文件路径
    count = 0
    try:
        with tempfile.NamedTemporaryFile( # 创建一个有名字的临时文件，并用上下文管理器确保文件最后会被关闭
            "w",
            encoding="utf-8",
            newline="\n", # 强制使用 \n 作为换行符
            dir=path.parent, # 把临时文件创建在正式输出文件的同一个目录中
            prefix=f".{path.name}.", # 临时文件名前缀由目标文件名生成
            suffix=".tmp", # 临时文件后缀为 .tmp
            delete=False, # 离开 with 后不让 Python自动删除临时文件
        ) as handle:
            temp_path = Path(handle.name) # 保存临时文件路径，handle.name 是临时文件的实际路径字符串
            for row_index, sample in enumerate(samples):
                try:
                    # 调用前面分析的单样本转换函数
                    row = canonical_to_ms_swift_row(sample, include_metadata=include_metadata)
                except (MsSwiftExportError, TypeError, ValueError) as exc:
                    raise MsSwiftExportError(
                        f"Failed to export row {row_index}, sample_id={sample.sample_id!r}: {exc}"
                    ) from exc # raise ... from exc 建立异常链
                handle.write(json.dumps(row, ensure_ascii=False) + "\n") # 序列化并写入一行
                count += 1
        temp_path.replace(path) # 会用临时文件替换目标文件
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True) # 失败时清理临时文件，missing_ok=True 表示如果临时文件已经不存在，也不要再抛出 FileNotFoundError
        raise
    return count # 返回行数

# 这个辅助函数把一个 CanonicalMessage 转换为 ms-swift 消息字典
def _message_to_ms_swift(message: CanonicalMessage, media_state: "_MediaExportState") -> Dict[str, Any]:
    row = {
        "role": message.role,
        "content": _render_message_content(message, media_state),
    }
    if message.loss is not None:
        row["loss"] = message.loss
    if message.loss_scale is not None:
        row["loss_scale"] = message.loss_scale
    return row


def _render_message_content(message: CanonicalMessage, media_state: "_MediaExportState") -> str:
    if not message.content_parts: # 如果 content_parts 是空列表，说明这是一条普通文本消息，直接返回 message.content
        return message.content 

    text_content = "".join(part.text or "" for part in message.content_parts if part.type == "text") # 遍历所有 content_parts，只取 type == "text" 的部分，再拼接成一个字符串
    if text_content != message.content:
        raise MsSwiftExportError(
            "Message content does not match the text in content_parts: "
            f"content={message.content!r}, content_parts_text={text_content!r}."
        )

    rendered: List[str] = []
    for part in message.content_parts:
        if part.type == "text":
            rendered.append(part.text if part.text is not None else "")
        else:
            media_tag = MEDIA_TAGS.get(part.type) # 从媒体标签字典中查询
            if media_tag is None:
                raise MsSwiftExportError(f"Unsupported content part type: {part.type!r}.")
            rendered.append(media_tag) # 把媒体占位符加入消息字符串，注意这里只加入 <image>，不会加入图片路径。路径由 media_state 单独收集
            media_state.add_ref(part)
    return "".join(rendered)

# 这个函数只进行检查，不返回数据，所以返回类型是 None
def _assert_chosen_matches_messages(sample: CanonicalSample) -> None:
    if not sample.messages or sample.messages[-1].role != "assistant": # 偏好样本的 messages 必须以 chosen assistant 回答结束，否则无法确定主对话中的 chosen 是哪一条消息
        raise MsSwiftExportError("Preference sample messages must end with an assistant chosen response.")
    chosen_message = sample.messages[-1]
    chosen_text = "".join(
        part.text or "" for part in chosen_message.content_parts if part.type == "text" # 如果存在结构化部分，只提取文本部分；媒体标签不会成为 chosen 文本的一部分
    ) if chosen_message.content_parts else chosen_message.content
    if chosen_text != sample.preference.chosen:
        raise MsSwiftExportError(
            "Preference chosen response does not match the final assistant message: "
            f"chosen={sample.preference.chosen!r}, assistant={chosen_text!r}."
        )

# 这个函数把偏好信息转换成 ms-swift 字段
def _preference_to_ms_swift(
    preference: PreferencePair,
    chosen_messages: Sequence[CanonicalMessage],
) -> Dict[str, Any]:
    if preference.rejected_messages is None:
        row: Dict[str, Any] = {"rejected_response": preference.rejected}
        if preference.rejected_media is not None: # 如果 rejected 分支还携带媒体资源，就需要处理其媒体引用
            rejected_state = _MediaExportState(preference.rejected_media) # 为 rejected 分支单独创建媒体状态，避免与 chosen 分支的 images/videos/audios 混在一起
            for message in chosen_messages:
                _message_to_ms_swift(message, rejected_state) # 根据 chosen 对话中的媒体占位位置，收集 rejected_media 中对应的媒体路径
            _merge_media_state(row, rejected_state, prefix="rejected_")
            rejected_state.assert_all_referenced()
        return row

    if not preference.rejected_messages: # 这个判断专门捕获空列表
        raise MsSwiftExportError("Preference rejected_messages must not be empty.")

    rejected_state = _MediaExportState(preference.rejected_media or MediaBundle())
    row: Dict[str, Any] = {
        "rejected_messages": [
            _message_to_ms_swift(message, rejected_state) for message in preference.rejected_messages
        ]
    }
    _merge_media_state(row, rejected_state, prefix="rejected_")
    rejected_state.assert_all_referenced() # 检查 rejected 媒体全部被使用
    return row

# 把媒体状态中的非空列表合并进 row，比如"rejected_images"
def _merge_media_state(row: Dict[str, Any], media_state: "_MediaExportState", *, prefix: str = "") -> None:
    for key in ("images", "videos", "audios"):
        value = getattr(media_state, key)
        if value:
            row[f"{prefix}{key}"] = list(value)


class _MediaExportState:
    def __init__(self, media: MediaBundle) -> None:
        self.media = media
        self.images: List[str] = []
        self.videos: List[str] = []
        self.audios: List[str] = []
        self._seen_refs: set[str] = set()

    def add_ref(self, part: ContentPart) -> None:
        if part.media_ref is None: # 媒体部分必须明确指向某个媒体
            raise MsSwiftExportError(f"Missing media_ref for {part.type} content part.")

        kind, index = _parse_media_ref(part.media_ref)
        expected_kind = f"{part.type}s"
        if kind != expected_kind: # 检查内容类型和引用类型是否匹配
            raise MsSwiftExportError(
                f"Media ref kind mismatch: part type {part.type!r} points to {part.media_ref!r}."
            )

        values = getattr(self.media, kind)
        if index >= len(values): # 检查索引上界，例如列表长度为 2，只允许索引 0 和 1；images[2] 会越界
            raise MsSwiftExportError(f"Media ref out of range: {part.media_ref!r}.")

        getattr(self, kind).append(values[index])
        self._seen_refs.add(part.media_ref)

    # 检查原始媒体集合中的每项资源是否都被引用
    def assert_all_referenced(self) -> None:
        for kind in ("images", "videos", "audios"):
            values: Sequence[str] = getattr(self.media, kind)
            for index in range(len(values)):
                ref = f"{kind}[{index}]"
                if ref not in self._seen_refs:
                    raise MsSwiftExportError(f"Unreferenced media item: {ref}.")

# 把字符串引用解析成 (媒体类别, 索引)，如_parse_media_ref("videos[3]") 返回：("videos", 3)
def _parse_media_ref(media_ref: str) -> tuple[str, int]:
    if not media_ref.endswith("]") or "[" not in media_ref:
        raise MsSwiftExportError(f"Invalid media_ref: {media_ref!r}.")
    kind, index_text = media_ref[:-1].split("[", 1)
    if kind not in {"images", "videos", "audios"}:
        raise MsSwiftExportError(f"Unsupported media_ref kind: {media_ref!r}.")
    try:
        index = int(index_text) # 尝试把索引文本转成整数
    except ValueError as exc:
        raise MsSwiftExportError(f"Invalid media_ref index: {media_ref!r}.") from exc
    if index < 0:
        raise MsSwiftExportError(f"Negative media_ref index: {media_ref!r}.")
    return kind, index
