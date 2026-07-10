"""Path configuration helpers for local and AutoDL runs.

Adapters should receive paths from this module instead of hardcoding Windows
or AutoDL absolute paths. The default config target is `configs/autodl.yaml`.
"""
# 读取 configs/autodl.yaml
# 提供 ToolVistaPaths 统一访问 project/model/data/dataset/export 路径
# 提供 DatasetPathConfig 给 adapter 获取某个数据源的 root 和子路径
# 提供 as_posix_path() / join_posix()，保证导出给 AutoDL/ms-swift 的 JSONL 路径是 /root/autodl-tmp/... 风格
# 避免 adapter 里写死 Windows 路径或 AutoDL 路径
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional

import yaml


ConfigDict = Dict[str, Any]


def load_yaml_config(config_path: str | Path) -> ConfigDict:
    """
    文档字符串：把 YAML 配置文件加载成字典
    Load a YAML config file into a dictionary.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {} # 用 PyYAML 的 safe_load 解析 YAML 文件
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data

# 函数名前面的 _ 表示：这是模块内部使用的函数，不建议外部直接调用
# Mapping是一个比dict更宽泛的类型，只要支持key获取value都算Mapping
def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping): # 这个 key 对应的值必须也是一个 mapping
        raise KeyError(f"Missing or invalid mapping: {key}")
    return value

# 要求某个 key 必须存在，并且对应值必须是非空字符串
def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(f"Missing or invalid string: {key}")
    return value


def as_posix_path(path: str | Path) -> str:
    """
    文档字符串：返回适合导出到 Linux/AutoDL JSONL 的路径字符串
    Return a path string suitable for JSONL exported on Linux/AutoDL.
    """

    return Path(path).as_posix()


def join_posix(*parts: str | Path) -> str: # *parts 表示可以传任意多个路径片段
    """
    Join path fragments and return a POSIX-style string.
    说明这个函数适合导出数据集路径，给 AutoDL 这类 Linux 环境使用
    This is useful for exported dataset paths that will be consumed on AutoDL.
    """

    cleaned = [str(part).replace("\\", "/").strip("/") for part in parts if str(part)] # 如果 part 转成字符串后不是空字符串，才保留
    if not cleaned:
        return ""
    first = str(parts[0]).replace("\\", "/")
    prefix = "/" if first.startswith("/") else "" # 如果第一个路径片段以 / 开头，就说明用户想要绝对路径，于是保留根路径前缀 /
    return prefix + str(PurePosixPath(*cleaned)) # PurePosixPath 是 pathlib 里的纯 POSIX 路径类，它不依赖当前操作系统


@dataclass(frozen=True)# frozen=True 表示这个 dataclass 是“冻结的”，创建后字段不能再被修改
class DatasetPathConfig:
    """Resolved configuration for one raw dataset source."""

    name: str
    root: Path
    settings: Mapping[str, Any]

    def path(self, *parts: str | Path) -> Path:
        return self.root.joinpath(*map(Path, parts)) # 接收任意多个路径片段，返回一个 Path 对象

    @property
    def root_posix(self) -> str: 
        return as_posix_path(self.root) # 返回数据集 root 的 POSIX 风格路径字符串

    def posix_path(self, *parts: str | Path) -> str:
        return as_posix_path(self.path(*parts)) # 拼接 root 和 parts，然后返回 POSIX 风格字符串

    def setting(self, key: str, default: Optional[Any] = None) -> Any:
        return self.settings.get(key, default) # 从 settings 里读取某个配置项

    def required_setting(self, key: str) -> str:
        return _require_str(self.settings, key) # 读取一个必填字符串配置项，如果读取不到就抛异常

# 把 YAML 配置文件里的各种路径封装成属性和方法，让其他代码不用反复手动从字典里取字段
@dataclass(frozen=True)
class ToolVistaPaths:
    """
    这是项目路径、模型路径、数据路径、数据集路径的便捷访问器
    Convenience accessor for project, model, data, and dataset paths.
    """

    config_path: Path
    raw: ConfigDict

    # @classmethod 的意思是：这个方法属于类本身，不是某个具体对象
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ToolVistaPaths":
        path = Path(config_path)
        return cls(config_path=path, raw=load_yaml_config(path))

    # 定义属性 project_root，返回项目根目录
    @property
    def project_root(self) -> Path:
        # _require_mapping(self.raw, "project")要求 raw 配置里必须有 "project"，并且它必须是一个字典
        # 要求 project 这个字典里必须有 "base_path"，并且它必须是非空字符串
        return Path(_require_str(_require_mapping(self.raw, "project"), "base_path"))

    # 返回项目根目录的 POSIX 风格字符串
    @property
    def project_root_posix(self) -> str:
        return as_posix_path(self.project_root)

    # 返回模型根目录
    @property
    def model_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "model"), "base_path"))

    # 返回模型根目录的 POSIX 风格路径字符串
    @property
    def model_root_posix(self) -> str:
        return as_posix_path(self.model_root)

    # 返回 Qwen3-VL 模型路径
    @property
    def qwen3_vl_model(self) -> Path:
        # 它直接从配置里读完整路径，并没有自动拼接 model.base_path 和 qwen3_vl；也就是说，如果配置里写的是相对路径，返回的就是相对路径对应的 Path。
        return Path(_require_str(_require_mapping(self.raw, "model"), "qwen3_vl"))

    # 返回 Qwen3-VL 模型路径的 POSIX 字符串
    @property
    def qwen3_vl_model_posix(self) -> str:
        return as_posix_path(self.qwen3_vl_model)

    # 返回数据根目录
    @property
    def data_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "data"), "base_path"))

    # 返回数据根目录的 POSIX 字符串
    @property
    def data_root_posix(self) -> str:
        return as_posix_path(self.data_root)

    # 返回原始数据目录
    @property
    def raw_data_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "data"), "raw_path"))

    # 返回原始数据目录的 POSIX 字符串
    @property
    def raw_data_root_posix(self) -> str:
        return as_posix_path(self.raw_data_root)

    # 返回标准化数据目录
    # 所谓 canonical data，就是不同来源数据经过 adapter 转换后，统一成 CanonicalSample 格式并保存的位置
    @property
    def canonical_data_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "data"), "canonical_path"))

    # 返回标准化数据目录的 POSIX 字符串
    @property
    def canonical_data_root_posix(self) -> str:
        return as_posix_path(self.canonical_data_root)

    # 返回导出目录
    @property
    def export_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "data"), "export_path"))

    # 返回导出目录的 POSIX 字符串
    @property
    def export_root_posix(self) -> str:
        return as_posix_path(self.export_root)

    # 返回报告目录
    # 例如 adapter 转换报告、错误报告、统计报告可能放这里
    @property
    def report_root(self) -> Path:
        return Path(_require_str(_require_mapping(self.raw, "data"), "report_path"))

    # 返回报告目录的 POSIX 字符
    @property
    def report_root_posix(self) -> str:
        return as_posix_path(self.report_root)

    # 根据数据集名称，返回一个 DatasetPathConfig 对象
    def dataset(self, name: str) -> DatasetPathConfig:
        datasets = _require_mapping(self.raw, "datasets") # 要求配置里有 "datasets"，并且它必须是字典
        settings = datasets.get(name)
        if not isinstance(settings, Mapping):
            raise KeyError(f"Unknown dataset config: {name}")
        return DatasetPathConfig(name=name, root=Path(_require_str(settings, "root")), settings=settings)

    # 根据导出名称，返回某个导出文件/目录路径
    def export_path(self, name: str) -> Path:
        exports = _require_mapping(self.raw, "exports")
        return Path(_require_str(exports, name))

    # 传入任意路径，返回 POSIX 风格字符串
    def posix(self, path: str | Path) -> str:
        return as_posix_path(path)


def load_path_config(config_path: str | Path = "configs/autodl.yaml") -> ToolVistaPaths:
    """Load ToolVista path settings from a YAML config file."""
    # 加载 ToolVista 的路径配置
    return ToolVistaPaths.from_yaml(config_path)
