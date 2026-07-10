"""Canonical data structures used between adapters and exporters.

Adapters convert raw datasets into these structures. Exporters then convert
them into Qwen official sanity data, ms-swift SFT/DPO JSONL, or verl-agent
rollout inputs.
"""
"""
定义了 ToolVista 的内部标准样本结构，包括：
CanonicalMessage
MediaBundle
ToolSpec
PreferencePair
CanonicalSample
AdaptResult
normalize_path()

这个 schema 的定位是：
raw dataset
  -> adapter
  -> CanonicalSample
  -> exporter
  -> ms-swift / Qwen official sanity / verl-agent
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


MessageRole = Literal["system", "user", "assistant", "tool_call", "tool_response", "tool"]
Modality = Literal["text", "image", "video", "audio"]
TaskType = Literal[
    "sft",
    "vqa",
    "tool_selection",
    "tool_augmented_vqa",
    "agent_trajectory",
    "preference",
    "grounding",
]

# dataclass 会自动帮你生成 __init__、__repr__、__eq__ 等方法。也就是说你不用手写构造函数。
@dataclass
class CanonicalMessage:
    """
    表示一条标准化后的聊天/Agent 消息
    A normalized chat or agent message.
    不要把太多额外字段塞进单条 message 里，如果某些信息只和原始数据来源有关，
    比如数据集名称、原始 ID、标注来源、转换记录等，不应该放在 CanonicalMessage 里，
    而应该放到更外层的 CanonicalSample.metadata 或 CanonicalSample.provenance 里
    Keep this close to ms-swift's allowed message shape. ms-swift keeps only
    role/content/loss/loss_scale inside messages, so source-specific metadata
    belongs on CanonicalSample.metadata or CanonicalSample.provenance.
    """

    role: MessageRole # 上面定义的MessageRole角色，包括："system","user","assistant","tool_call","tool_response","tool"
    content: str
    loss: Optional[bool] = None # loss 通常用来控制这一条消息是否参与 loss 计算
    loss_scale: Optional[float] = None # loss 的权重缩放，比如 0.5 表示这条消息的 loss 权重可能乘以 0.5

    # 作用是把当前 CanonicalMessage 对象转换成字典。
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class MediaBundle:
    """
    表示消息里引用的图片、视频、音频路径
    Media paths referenced by standard placeholders in messages.
    """
    # default_factory=list 会确保每次创建对象时都生成一个新的空列表
    # field是 dataclasses 模块提供的一个函数，主要用于给 @dataclass 里的字段提供更精细的配置，给 dataclass 字段设置默认值、默认工厂、显示规则、比较规则等配置。
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)

    # 根据当前对象里有没有图片、视频、音频，推断这个样本包含哪些模态
    def modalities(self) -> List[Modality]:
        result: List[Modality] = []
        if self.images:
            result.append("image")
        if self.videos:
            result.append("video")
        if self.audios:
            result.append("audio")
        if not result:
            result.append("text")
        return result

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "images": list(self.images),
            "videos": list(self.videos),
            "audios": list(self.audios),
        }


@dataclass
class ToolSpec:
    """
    表示 OpenAI function calling 风格的工具 schema
    OpenAI-style function tool schema.
    """

    name: str
    description: str = ""
    # 这里使用 lambda 是因为默认值是一个字典，字典也是可变对象。和列表一样，不能直接写成共享默认值
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    # Literal 来自 typing 模块：在类型标注里限制某个变量只能取几个固定的字面量值。
    type: Literal["function"] = "function" # 它只能是字符串 "function"

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class PreferencePair:
    """
    表示偏好数据，比如 DPO 里的 chosen/rejected
    Chosen/rejected response data for DPO-style exporters.
    """

    chosen: str
    rejected: str
    rejected_messages: Optional[List[CanonicalMessage]] = None # 表示 rejected 一侧如果不是单纯字符串，而是完整消息列表，可以放在这里
    rejected_media: Optional[MediaBundle] = None # 用于保存 rejected 回答关联的媒体文件。例如 rejected 回答里引用了错误图片、视频或音频，就可以把媒体路径放在这里。

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"chosen": self.chosen, "rejected": self.rejected}
        if self.rejected_messages is not None:
            data["rejected_messages"] = [message.to_dict() for message in self.rejected_messages]
        if self.rejected_media is not None:
            data["rejected_media"] = self.rejected_media.to_dict()
        return data


@dataclass
class CanonicalSample:
    """
    这是 ToolVista 的内部样本表示
    ToolVista's internal sample representation.
    """

    sample_id: str
    source: str # 它表示这条数据原本来自哪个数据集或哪个转换器
    task_type: TaskType # task_type 只能是前面定义过的这些任务类型之一
    messages: List[CanonicalMessage]
    media: MediaBundle = field(default_factory=MediaBundle)
    tools: List[ToolSpec] = field(default_factory=list)
    labels: Dict[str, Any] = field(default_factory=dict)
    preference: Optional[PreferencePair] = None
    objects: Optional[Dict[str, Any]] = None # 可以是字典，也可以是 None。这个字段常用于视觉 grounding、检测、场景图等任务
    split: Optional[str] = None # 数据集划分
    provenance: Dict[str, Any] = field(default_factory=dict) # 通常保存这条样本是怎么来的，比如原始文件、原始行号、转换器版本等
    metadata: Dict[str, Any] = field(default_factory=dict) # 数据集特有字段、统计信息、语言、难度、类别等

    # @property 的作用是：把一个方法变成像字段一样访问
    @property
    def modalities(self) -> List[Modality]:
        return self.media.modalities()

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "source": self.source,
            "task_type": self.task_type,
            "modalities": self.modalities,
            "messages": [message.to_dict() for message in self.messages],
            "media": self.media.to_dict(),
            "tools": [tool.to_openai_tool() for tool in self.tools],
            "labels": self.labels,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }
        if self.preference is not None:
            data["preference"] = self.preference.to_dict()
        if self.objects is not None:
            data["objects"] = self.objects
        if self.split is not None:
            data["split"] = self.split
        return data


@dataclass
class AdaptResult:
    """
    适配器输出，里面包含可恢复的 warnings 和 errors
    Adapter output with recoverable warnings and errors.
    """

    sample: Optional[CanonicalSample] = None # 转换出来的标准样本，如果转换成功，这里是一个 CanonicalSample，如果转换失败，这里可能是 None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_ref: Dict[str, Any] = field(default_factory=dict) # 原始样本引用信息，它通常不是完整原始数据，而是能帮助定位原始数据的引用
    
    # 定义一个属性 ok，表示这次转换是否成功
    @property
    def ok(self) -> bool:
        return self.sample is not None and not self.errors

    # 把适配结果转换成报告用的字典，导出转换报告，比如统计成功失败、定位错误
    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "sample_id": self.sample.sample_id if self.sample is not None else None,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "raw_ref": self.raw_ref,
        }


def normalize_path(path: str | Path) -> str:
    """
    文档字符串：返回一种稳定的 POSIX 风格路径，用于 JSONL 导出
    Return a stable POSIX-style path for JSONL exports.
    POSIX-style path 指使用 / 作为路径分隔符
    """

    # Path(path) 把传入的字符串或 Path 对象转换成 pathlib.Path 对象
    # .as_posix() 把路径转换成 POSIX 风格字符串。
    return Path(path).as_posix()
