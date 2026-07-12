"""Base adapter interfaces for converting raw datasets to canonical samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional

from toolvista.schema.canonical import AdaptResult, CanonicalSample
from toolvista.utils.paths import DatasetPathConfig, ToolVistaPaths


@dataclass
class AdapterStats:
    """
    统计 total_raw / converted / skipped / warnings / errors / error_counts
    Conversion counters collected during one adapter run.
    """

    source: str # 数据来源名称，表示这次统计对应哪个数据源/adapter
    total_raw: int = 0 # 原始数据总数，默认是 0
    converted: int = 0 # 成功转换的数量，默认是 0
    skipped: int = 0 # 如果某条样本转换失败、缺字段、格式不合法，或者没有生成有效 CanonicalSample，就会被统计为 skipped
    warnings: int = 0 # 注意它统计的是 warning 条数，不是“有 warning 的样本数”
    errors: int = 0 # 它统计的是 error 条数，不是失败样本数
    error_counts: Counter[str] = field(default_factory=Counter) # Counter 来自：from collections import Counter 是一个专门用来计数的字典

    # 作用是把一条样本的转换结果 AdaptResult 记录到统计里
    def record_result(self, result: AdaptResult) -> None:
        """Update counters from one normalized adapter result."""

        self.warnings += len(result.warnings)
        self.errors += len(result.errors)
        for error in result.errors:
            self.error_counts[error] += 1

        if result.ok:
            self.converted += 1 # 注意：有 warning 但没有 error，仍然算成功
        else:
            self.skipped += 1

    # 把统计结果转换成普通字典
    def to_dict(self) -> Dict[str, Any]:
        """
        返回一个 JSON 可序列化的统计字典
        Return a JSON-serializable stats dictionary.
        """

        return {
            "source": self.source,
            "total_raw": self.total_raw,
            "converted": self.converted,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "errors": self.errors,
            "error_counts": dict(self.error_counts),
        }


@dataclass
class AdapterRunResult:
    """
    保存一次 adapter 运行后的 samples / reports / stats
    In-memory result for a complete adapter run.
    """
    # 前面 AdapterStats 只负责统计数量；而这里的 AdapterRunResult 会把一次运行产生的三类东西放在一起
    source: str # 表示这次 adapter 运行的数据来源
    samples: List[CanonicalSample] = field(default_factory=list) # 保存成功转换出来的标准样本列表
    reports: List[Dict[str, Any]] = field(default_factory=list) # 保存每条样本的转换报告
    stats: AdapterStats = field(default_factory=lambda: AdapterStats(source="unknown")) # 保存整体统计信息，也就是前面定义的统计类
    # 这里用 lambda 是因为 AdapterStats 创建时需要传参数。如果只是无参构造，可以写成：field(default_factory=AdapterStats)

    def __post_init__(self) -> None:
        if self.stats.source == "unknown":
            self.stats.source = self.source

    def to_report_dict(self) -> Dict[str, Any]:
        """
        返回一个 JSON 可序列化的转换报告
        Return a JSON-serializable conversion report.
        """

        return {
            "source": self.source,
            "stats": self.stats.to_dict(),
            "items": list(self.reports),
        }

# ABC 是 Abstract Base Class，抽象基类，意思是：这个类本身不是拿来直接实例化使用的，而是给子类继承的
class BaseDatasetAdapter(ABC):
    """
    这个基类定义了从原始数据项转换成 CanonicalSample 的通用生命周期
    Common lifecycle for converting raw dataset items into CanonicalSample.
    """
    # ClassVar[str] 表示这是一个类级别变量，不是每个实例单独的数据字段
    source_name: ClassVar[str] = "base"# ClassVar 的意义主要是给类型检查器看的：告诉它这个字段属于类，不属于 dataclass/实例字段，子类通常会覆盖它

    def __init__(self, paths: ToolVistaPaths, dataset_name: Optional[str] = None) -> None:
        self.paths = paths # 把传进来的 ToolVistaPaths 保存到实例上
        self.dataset_name = dataset_name or self.source_name # 设置当前 adapter 对应的数据集名称

    @property
    def dataset_config(self) -> DatasetPathConfig:
        """
        返回当前 adapter 对应的数据集路径配置
        Resolved path config for this adapter's dataset source.
        """

        return self.paths.dataset(self.dataset_name)

    # @abstractmethod 表示：子类必须实现这个方法，否则子类也不能被实例化
    @abstractmethod
    def iter_raw(self) -> Iterable[Any]: # 意思是它可以返回任何可迭代对象，里面的原始 item 类型不限
        """
        逐条产生原始数据记录
        Yield raw dataset records from files or another local source.
        """
        # yield 不是方法，它是 Python 的一个关键字，它的作用是：让一个函数变成生成器函数，可以一边运行、一边产出结果，而不是一次性返回所有结果

    @abstractmethod
    def adapt(self, raw_item: Any) -> AdaptResult | Iterable[AdaptResult]: # 允许返回多个 AdaptResult: 因为有些原始数据一条记录可能拆成多个训练样本
        """
        把一条原始数据转换成标准结果
        Convert one raw record into one or more AdaptResult objects.
        """

    def iter_results(self) -> Iterator[AdaptResult]:
        """
        逐条产出标准化后的 AdaptResult，产出规范化的 AdaptResult，同时保留 adapter 失败信息
        Yield normalized AdaptResult objects while preserving adapter failures.
        """

        for index, raw_item in enumerate(self.iter_raw()): # 这里的 index 后面用于记录出错的是第几条原始数据
            try:
                adapted = self.adapt(raw_item) # 尝试转换这一条原始数据。
                yield from self._ensure_result_list(adapted)
            except Exception as exc:  # pragma: no cover - exercised by adapter subclasses.
                yield self._error_result(exc, raw_index=index)

    def iter_samples(self) -> Iterator[CanonicalSample]: # 它是一个迭代器，会不断 yield 结果
        """
        只产出成功转换的 CanonicalSample
        Yield successfully converted canonical samples only.
        """

        for result in self.iter_results():
            if result.ok and result.sample is not None:
                yield result.sample

    def run(self, *, collect_samples: bool = True) -> AdapterRunResult:
        """
        执行整个 adapter，并收集运行结果
        Execute the adapter and collect samples, per-item reports, and stats.
        """

        stats = AdapterStats(source=self.source_name)
        samples: List[CanonicalSample] = [] # 用来保存成功转换的样本
        reports: List[Dict[str, Any]] = [] # 用来保存每条结果的报告字典

        for raw_item in self.iter_raw():
            stats.total_raw += 1 # 每读一条原始数据，原始总数加一
            try:
                adapted = self.adapt(raw_item) # 尝试转换当前原始 item
                results = self._ensure_result_list(adapted)
            except Exception as exc:  # pragma: no cover - exercised by adapter subclasses.
                results = [self._error_result(exc, raw_index=stats.total_raw - 1)] # 如果转换过程出错，就生成一个失败的 AdaptResult

            for result in results:
                stats.record_result(result)
                reports.append(result.to_report_dict())
                if collect_samples and result.ok and result.sample is not None:
                    samples.append(result.sample)

        return AdapterRunResult(
            source=self.source_name,
            samples=samples,
            reports=reports,
            stats=stats,
        )

    # 把 adapt() 的返回值统一成 List[AdaptResult]
    def _ensure_result_list(self, result: AdaptResult | Iterable[AdaptResult]) -> List[AdaptResult]:
        if isinstance(result, AdaptResult): # 如果它本来就是单个 AdaptResult，就包成列表
            return [result]

        results = list(result)
        for item in results:
            if not isinstance(item, AdaptResult): # 检查列表里每一项必须都是 AdaptResult
                raise TypeError(f"Adapter returned non-AdaptResult item: {type(item)!r}")
        return results

    # 把异常转换成一个失败的 AdaptResult
    def _error_result(self, exc: Exception, *, raw_index: int) -> AdaptResult: # 这里的 * 表示 raw_index 必须用关键字传
        return AdaptResult(
            errors=[f"{type(exc).__name__}: {exc}"],
            raw_ref={
                "source": self.source_name, # adapter 来源名，比如 "llava"
                "dataset": self.dataset_name, # YAML 里的数据集配置名，比如 "llava"
                "raw_index": raw_index, # 第几条原始数据出错，比如 123
            },
        )
