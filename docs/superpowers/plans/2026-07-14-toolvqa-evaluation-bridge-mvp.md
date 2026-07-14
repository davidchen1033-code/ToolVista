# ToolVQA Evaluation Bridge MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated ToolVQA bridge that generates an upstream OpenCompass config, runs the official evaluator, and converts its official JSON metrics into one compact local result.

**Architecture:** Keep ToolVista's code at the process boundary. One focused module prepares media paths, renders an upstream config, invokes `ToolVQA-release/opencompass/run.py`, and parses `gta_bench.json`; one thin script exposes those functions. The ToolVQA loader, agent, multimodal prompt construction, tools, and scoring remain entirely upstream.

**Tech Stack:** Python standard library, existing ToolVista JSON helpers, ToolVQA's modified OpenCompass/Lagent/AgentLego stack, `unittest`.

---

## File Map

| File | Responsibility |
|---|---|
| `toolvista/evaluation/toolvqa.py` | Data preparation, media remapping, runtime config rendering, upstream command execution, official result discovery, and metric normalization |
| `scripts/evaluate_toolvqa.py` | Minimal CLI validation and call into `run_toolvqa` |
| `examples/data/toolvqa/dataset.json` | One static upstream-format ToolVQA training fixture that references the existing sample image |
| `tests/test_toolvqa_evaluation.py` | Tests only ToolVista-owned boundaries with a fake subprocess and temporary files |
| `docs/phase2_toolvqa_evaluation_usage.md` | Chinese handoff: what is tested, upstream source, commands, metrics, and artifact paths |

Existing `BenchmarkSpec`, `toolvista/evaluation/runner.py`, `scripts/evaluate.py`, benchmark registry, suites, and W&B code are out of scope.

### Task 1: Static Fixture, Media Preparation, and Runtime Config

**Files:**
- Create: `examples/data/toolvqa/dataset.json`
- Create: `tests/test_toolvqa_evaluation.py`
- Create: `toolvista/evaluation/toolvqa.py`

- [ ] **Step 1: Add the one-record upstream-format smoke fixture**

Create `examples/data/toolvqa/dataset.json` with one object keyed by `"0"`. Use this exact shape and populate the three tool responses with the text already present in `train_sample.jsonl`:

```json
{
  "0": {
    "_toolvista_fixture": true,
    "tools": [
      {"name": "ImageDescription"},
      {"name": "OCR"},
      {"name": "GoogleSearch"}
    ],
    "files": [
      {
        "type": "image",
        "path": "images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg",
        "url": null
      }
    ],
    "dialogs": [
      {
        "role": "user",
        "content": "Are these leggings likely part of a men's or women's collection?"
      },
      {
        "role": "assistant",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "ImageDescription",
            "arguments": {"image": "images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg"}
          }
        }]
      },
      {
        "role": "tool",
        "name": "ImageDescription",
        "content": {
          "type": "text",
          "content": "The image shows three shirtless male models wearing differently colored thermal leggings: gray, black, and white."
        }
      },
      {
        "role": "assistant",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "OCR",
            "arguments": {"image": "images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg"}
          }
        }]
      },
      {
        "role": "tool",
        "name": "OCR",
        "content": {"type": "text", "content": "alvin Kle n; 29; 365"}
      },
      {
        "role": "assistant",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "GoogleSearch",
            "arguments": {"query": "Calvin Klein 365 thermal leggings collection"}
          }
        }]
      },
      {
        "role": "tool",
        "name": "GoogleSearch",
        "content": {
          "type": "text",
          "content": "Calvin Klein Womens Performance Logo-Tape Thermal High-Waisted Leggings Black"
        }
      },
      {"role": "assistant", "content": "Men's collection"}
    ],
    "gt_answer": {
      "whitelist": [["Men's collection", "men's collection", "men"]],
      "blacklist": null
    }
  }
}
```

- [ ] **Step 2: Write failing tests for API normalization, fixture preparation, media errors, and both config modes**

Start `tests/test_toolvqa_evaluation.py` with these imports and helpers:

```python
from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from toolvista.evaluation.toolvqa import (
    normalize_api_base,
    prepare_data,
    render_runtime_config,
    resolve_toolmeta,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "data" / "toolvqa"


class ToolVQAEvaluationTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-toolvqa-eval-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path
```

Add these tests:

```python
    def test_normalize_api_base_keeps_exactly_one_v1(self) -> None:
        self.assertEqual(normalize_api_base("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(normalize_api_base("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")

    def test_prepare_data_rewrites_fixture_image_to_absolute_path(self) -> None:
        output = self.make_temp_dir() / "prepared"
        prepared, is_fixture, sample_count = prepare_data(FIXTURE_ROOT, output)
        row = json.loads(prepared.read_text(encoding="utf-8"))["0"]
        image = Path(row["files"][0]["path"])
        self.assertTrue(image.is_absolute())
        self.assertTrue(image.is_file())
        self.assertEqual(row["dialogs"][1]["tool_calls"][0]["function"]["arguments"]["image"], str(image))
        self.assertTrue(is_fixture)
        self.assertEqual(sample_count, 1)

    def test_prepare_data_rejects_missing_and_ambiguous_media(self) -> None:
        root = self.make_temp_dir()
        source = root / "source"
        source.mkdir()
        manifest = {
            "0": {
                "tools": [{"name": "OCR"}],
                "files": [{"type": "image", "path": "/old/path/a.jpg"}],
                "dialogs": [],
                "gt_answer": []
            }
        }
        (source / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        media = root / "media"
        media.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "a.jpg"):
            prepare_data(source, root / "missing", media_root=media)
        (media / "one").mkdir()
        (media / "two").mkdir()
        (media / "one" / "a.jpg").write_bytes(b"one")
        (media / "two" / "a.jpg").write_bytes(b"two")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            prepare_data(source, root / "ambiguous", media_root=media)

    def test_render_runtime_config_separates_step_and_e2e(self) -> None:
        root = self.make_temp_dir()
        common = dict(
            data_root=root / "data",
            toolmeta=root / "toolmeta.json",
            api_base="http://127.0.0.1:8000/v1",
            api_model="qwen3-vl",
            limit=1,
        )
        step = render_runtime_config(mode="step", tool_url=None, **common)
        self.assertIn("every_with_gt", step)
        self.assertIn("tool_meta=", step)
        self.assertNotIn("tool_server=", step)
        self.assertIn("test_range='[0:1]'", step)
        e2e = render_runtime_config(mode="e2e", tool_url="http://127.0.0.1:16181", **common)
        self.assertIn("mode='every'", e2e)
        self.assertIn("tool_server=", e2e)
        self.assertNotIn("tool_meta=", e2e)

    def test_resolve_toolmeta_falls_back_to_upstream_data(self) -> None:
        root = self.make_temp_dir()
        upstream = root / "ToolVQA-release" / "opencompass" / "data" / "ToolVQA"
        upstream.mkdir(parents=True)
        expected = upstream / "toolmeta.json"
        expected.write_text("{}", encoding="utf-8")
        data_root = root / "fixture"
        data_root.mkdir()
        self.assertEqual(resolve_toolmeta(data_root, root / "ToolVQA-release"), expected.resolve())
```

- [ ] **Step 3: Run the focused test and verify it fails before implementation**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'toolvista.evaluation.toolvqa'`.

- [ ] **Step 4: Implement the pure data and config functions**

Create `toolvista/evaluation/toolvqa.py` with these public signatures:

```python
from __future__ import annotations

import copy
import json
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from toolvista.evaluation.io import load_json, write_json

ToolVQAMode = Literal["step", "e2e"]
MODEL_ABBR = "toolvista_qwen3vl"


def normalize_api_base(value: str) -> str:
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid OpenAI-compatible API base: {value!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        normalized_path = path
    elif not path:
        normalized_path = "/v1"
    else:
        normalized_path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def resolve_toolmeta(data_root: Path, toolvqa_root: Path) -> Path:
    candidates = (
        data_root / "toolmeta.json",
        toolvqa_root / "opencompass" / "data" / "ToolVQA" / "toolmeta.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ToolVQA toolmeta.json was not found in the data or upstream repository.")
```

Implement media resolution without a generic asset index:

```python
def _resolve_media(source_root: Path, raw_path: str, media_root: Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    relative = source_root / candidate
    if relative.is_file():
        return relative.resolve()
    if media_root is None:
        raise FileNotFoundError(f"Missing ToolVQA media: {raw_path}")
    matches = [path.resolve() for path in media_root.rglob(candidate.name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Missing ToolVQA media: {raw_path}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous ToolVQA media {candidate.name!r}: {len(matches)} matches")
    return matches[0]


def _rewrite_matching_arguments(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_matching_arguments(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_matching_arguments(item, old, new) for item in value]
    return new if value == old else value


def prepare_data(
    data_root: Path,
    output_root: Path,
    *,
    media_root: Path | None = None,
) -> tuple[Path, bool, int]:
    source = data_root / "dataset.json"
    dataset = load_json(source)
    prepared = copy.deepcopy(dataset)
    is_fixture = bool(prepared) and all(
        isinstance(row, Mapping) and row.get("_toolvista_fixture") is True
        for row in prepared.values()
    )
    for sample_id, row in prepared.items():
        if not isinstance(row, dict):
            raise ValueError(f"ToolVQA sample {sample_id!r} must be an object.")
        for file_item in row.get("files", []):
            old = file_item.get("path")
            if not isinstance(old, str) or not old:
                raise ValueError(f"ToolVQA sample {sample_id!r} has an invalid media path.")
            resolved = str(_resolve_media(data_root, old, media_root))
            file_item["path"] = resolved
            row["dialogs"] = _rewrite_matching_arguments(row.get("dialogs", []), old, resolved)
    output = output_root / "dataset.json"
    write_json(output, prepared)
    return output, is_fixture, len(prepared)
```

Implement `render_runtime_config` as this deterministic renderer. It validates `mode`, requires `tool_url` for `e2e`, adds `reader_cfg.test_range` only when `limit` is present, and uses `repr` for injected strings:

```python
def render_runtime_config(
    *,
    mode: ToolVQAMode,
    data_root: Path,
    toolmeta: Path,
    api_base: str,
    api_model: str,
    tool_url: str | None,
    limit: int | None,
) -> str:
    if mode not in {"step", "e2e"}:
        raise ValueError(f"Unsupported ToolVQA mode: {mode!r}")
    if mode == "e2e" and not tool_url:
        raise ValueError("ToolVQA e2e mode requires an AgentLego tool URL.")
    if limit is not None and limit <= 0:
        raise ValueError("ToolVQA limit must be positive.")
    if not api_model.strip():
        raise ValueError("ToolVQA API model must be non-empty.")

    infer_mode = "every_with_gt" if mode == "step" else "every"
    max_turn = 6 if mode == "step" else 8
    range_line = "" if limit is None else f"    test_range={f'[0:{limit}]'!r},\n"
    tool_line = (
        f"        tool_meta={str(toolmeta.resolve())!r},\n"
        if mode == "step"
        else f"        tool_server={tool_url!r},\n"
    )
    normalized_base = normalize_api_base(api_base)

    return f'''from lagent.agents import ReAct
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
from opencompass.models import OpenAIVisual
from opencompass.models.lagent import LagentAgentVisual
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

gta_bench_reader_cfg = dict(
    input_columns=["dialogs", "resources"],
    output_column="gt_answer",
    train_split="test",
    test_split="test",
{range_line})

gta_bench_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{{questions}}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode={infer_mode!r}),
)

gta_bench_eval_cfg = dict(
    evaluator=dict(type=GTABenchEvaluator, mode={infer_mode!r}),
)

datasets = [dict(
    abbr="gta_bench",
    type=GTABenchDataset,
    path={str(data_root.resolve())!r},
    reader_cfg=gta_bench_reader_cfg,
    infer_cfg=gta_bench_infer_cfg,
    eval_cfg=gta_bench_eval_cfg,
)]

models = {{
    "{MODEL_ABBR}": dict(
        abbr="{MODEL_ABBR}",
        type=LagentAgentVisual,
        agent_type=ReAct,
        max_turn={max_turn},
        llm=dict(
            type=OpenAIVisual,
            path={api_model.strip()!r},
            key="ENV",
            openai_api_base={normalized_base!r},
            query_per_second=1,
            max_seq_len=4096,
            retry=3,
            temperature=0.0,
        ),
{tool_line}        batch_size=1,
    ),
}}

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=4000, gen_task_coef=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)
'''
```

- [ ] **Step 5: Run the focused tests and commit Task 1**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation -v
```

Expected: 5 tests pass.

Commit only Task 1 files:

```bash
git add toolvista/evaluation/toolvqa.py tests/test_toolvqa_evaluation.py examples/data/toolvqa/dataset.json
git commit -m "feat: prepare ToolVQA upstream evaluation config"
```

### Task 2: Official Result Parser and Upstream Process Runner

**Files:**
- Modify: `tests/test_toolvqa_evaluation.py`
- Modify: `toolvista/evaluation/toolvqa.py`

- [ ] **Step 1: Add failing tests for official metrics and fake upstream execution**

Add imports:

```python
import subprocess

from toolvista.evaluation.toolvqa import map_metrics, run_toolvqa
```

Add metric tests:

```python
    def test_map_metrics_uses_only_official_mode_fields(self) -> None:
        step = map_metrics(
            {"inst_align": 80.0, "tool_acc": 70.0, "arg_acc": 60.0, "answer_acc": 50.0},
            "step",
        )
        self.assertEqual(step["toolvqa_sbs/tool_accuracy"], 70.0)
        e2e = map_metrics(
            {"answer_acc": 40.0, "p_f1": 30.0, "o_f1": 20.0, "l_f1": 10.0, "c_f1": 5.0},
            "e2e",
        )
        self.assertEqual(e2e["toolvqa_e2e/creativity_f1"], 5.0)
        with self.assertRaisesRegex(ValueError, "tool_acc"):
            map_metrics({"inst_align": 1.0}, "step")
```

Add a fake runner that creates the same stable result layout as OpenCompass:

```python
    def test_run_toolvqa_preserves_logs_and_parses_official_json(self) -> None:
        root = self.make_temp_dir()
        toolvqa_root = root / "ToolVQA-release"
        opencompass = toolvqa_root / "opencompass"
        opencompass.mkdir(parents=True)
        (opencompass / "run.py").write_text("# fake", encoding="utf-8")
        upstream_meta = opencompass / "data" / "ToolVQA"
        upstream_meta.mkdir(parents=True)
        (upstream_meta / "toolmeta.json").write_text("{}", encoding="utf-8")
        output = root / "run"
        captured: dict[str, object] = {}

        def fake_runner(command, **kwargs):
            captured["command"] = command
            captured["cwd"] = kwargs["cwd"]
            result_path = output / "upstream" / "20260714_120000" / "results" / "toolvista_qwen3vl" / "gta_bench.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(json.dumps({
                "inst_align": 80.0,
                "tool_acc": 70.0,
                "arg_acc": 60.0,
                "answer_acc": 50.0,
                "tool_call": 3,
                "tool_call_error": 0
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="official stdout", stderr="")

        result = run_toolvqa(
            mode="step",
            toolvqa_root=toolvqa_root,
            data_root=FIXTURE_ROOT,
            output_dir=output,
            api_base="http://127.0.0.1:8000",
            api_model="qwen3-vl",
            python_executable="python",
            workers=1,
            limit=1,
            command_runner=fake_runner,
        )
        self.assertEqual(captured["cwd"], str(opencompass.resolve()))
        self.assertIn("--eval_mode", captured["command"])
        self.assertEqual(result["metrics"]["toolvqa_sbs/tool_accuracy"], 70.0)
        self.assertFalse(result["comparable"])
        self.assertEqual((output / "stdout.log").read_text(encoding="utf-8"), "official stdout")
        self.assertTrue((output / "runtime_config.py").is_file())
        self.assertTrue((output / "result.json").is_file())
```

- [ ] **Step 2: Run the two new tests and verify they fail**

Run:

```powershell
python -m unittest \
  tests.test_toolvqa_evaluation.ToolVQAEvaluationTest.test_map_metrics_uses_only_official_mode_fields \
  tests.test_toolvqa_evaluation.ToolVQAEvaluationTest.test_run_toolvqa_preserves_logs_and_parses_official_json -v
```

Expected: import errors for `map_metrics` and `run_toolvqa`.

- [ ] **Step 3: Implement official result mapping and discovery**

Add constants and functions to `toolvista/evaluation/toolvqa.py`:

```python
STEP_METRICS = {
    "inst_align": "toolvqa_sbs/inst_align",
    "tool_acc": "toolvqa_sbs/tool_accuracy",
    "arg_acc": "toolvqa_sbs/argument_accuracy",
    "answer_acc": "toolvqa_sbs/answer_accuracy",
}
E2E_METRICS = {
    "answer_acc": "toolvqa_e2e/answer_accuracy",
    "p_f1": "toolvqa_e2e/perception_f1",
    "o_f1": "toolvqa_e2e/operation_f1",
    "l_f1": "toolvqa_e2e/logic_f1",
    "c_f1": "toolvqa_e2e/creativity_f1",
}


def map_metrics(upstream: Mapping[str, Any], mode: ToolVQAMode) -> dict[str, float]:
    mapping = STEP_METRICS if mode == "step" else E2E_METRICS
    metrics: dict[str, float] = {}
    for source, target in mapping.items():
        value = upstream.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError(f"Official ToolVQA result requires finite metric {source!r}.")
        metrics[target] = float(value)
    return metrics


def _latest_official_result(upstream_root: Path) -> Path:
    matches = [
        path for path in upstream_root.rglob("gta_bench.json")
        if path.parent.name == MODEL_ABBR and path.parent.parent.name == "results"
    ]
    if not matches:
        raise FileNotFoundError(f"Official ToolVQA gta_bench.json not found under {upstream_root}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)
```

- [ ] **Step 4: Implement the minimal process runner**

Add imports `os`, `shlex`, `subprocess`, `sys`, and `Callable`. Implement this exact public boundary and body:

```python
def run_toolvqa(
    *,
    mode: ToolVQAMode,
    toolvqa_root: Path,
    data_root: Path,
    output_dir: Path,
    api_base: str,
    api_model: str,
    tool_url: str | None = None,
    media_root: Path | None = None,
    python_executable: str = sys.executable,
    workers: int = 1,
    limit: int | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if mode not in {"step", "e2e"}:
        raise ValueError(f"Unsupported ToolVQA mode: {mode!r}")
    if workers <= 0:
        raise ValueError("ToolVQA workers must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("ToolVQA limit must be positive.")

    toolvqa_root = toolvqa_root.resolve()
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    opencompass_root = toolvqa_root / "opencompass"
    run_py = opencompass_root / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"ToolVQA OpenCompass entry point does not exist: {run_py}")
    if not (data_root / "dataset.json").is_file():
        raise FileNotFoundError(f"ToolVQA dataset.json does not exist: {data_root / 'dataset.json'}")

    toolmeta = resolve_toolmeta(data_root, toolvqa_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dataset, is_fixture, sample_count = prepare_data(
        data_root,
        output_dir / "data",
        media_root=media_root.resolve() if media_root is not None else None,
    )
    runtime_config = output_dir / "runtime_config.py"
    runtime_config.write_text(
        render_runtime_config(
            mode=mode,
            data_root=prepared_dataset.parent,
            toolmeta=toolmeta,
            api_base=api_base,
            api_model=api_model,
            tool_url=tool_url,
            limit=limit,
        ),
        encoding="utf-8",
        newline="\n",
    )

    upstream_root = output_dir / "upstream"
    command = [
        python_executable,
        str(run_py),
        str(runtime_config),
        "--work-dir",
        str(upstream_root),
        "--eval_mode",
        MODEL_ABBR,
        "--max-num-workers",
        str(workers),
        "--debug",
    ]
    try:
        completed = command_runner(
            command,
            cwd=str(opencompass_root),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        (output_dir / "stdout.log").write_text("", encoding="utf-8")
        (output_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise RuntimeError(f"Unable to start ToolVQA OpenCompass: {exc}") from exc

    (output_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ToolVQA OpenCompass failed with exit code {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )

    official_result = _latest_official_result(upstream_root)
    upstream_metrics = load_json(official_result)
    metrics = map_metrics(upstream_metrics, mode)
    result = {
        "mode": mode,
        "dataset_kind": "train_fixture" if is_fixture else "official_test",
        "comparable": not is_fixture,
        "warning": "Not an official ToolVQA test score." if is_fixture else None,
        "sample_count": min(sample_count, limit) if limit is not None else sample_count,
        "upstream_result": official_result.as_posix(),
        "upstream_metrics": upstream_metrics,
        "metrics": metrics,
        "command": shlex.join(command),
        "exit_code": completed.returncode,
    }
    write_json(output_dir / "result.json", result)
    return result
```

No API key is added to the command or config. The inherited environment supplies `OPENAI_API_KEY` to upstream `key='ENV'`.

- [ ] **Step 5: Run all focused tests and commit Task 2**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation -v
```

Expected: 7 tests pass.

Commit:

```bash
git add toolvista/evaluation/toolvqa.py tests/test_toolvqa_evaluation.py
git commit -m "feat: run official ToolVQA evaluator"
```

### Task 3: Dedicated CLI and Chinese Usage Handoff

**Files:**
- Create: `scripts/evaluate_toolvqa.py`
- Modify: `tests/test_toolvqa_evaluation.py`
- Create: `docs/phase2_toolvqa_evaluation_usage.md`

- [ ] **Step 1: Add failing CLI parsing tests**

Add:

```python
from scripts.evaluate_toolvqa import parse_args
```

Add tests:

```python
    def test_cli_requires_tool_url_only_for_e2e(self) -> None:
        common = [
            "--toolvqa-root", "F:/code/ToolVQA-release",
            "--data-root", "F:/data/ToolVQA",
            "--api-base", "http://127.0.0.1:8000/v1",
            "--api-model", "qwen3-vl",
            "--output-dir", "runs/toolvqa",
        ]
        step = parse_args(["--mode", "step", *common, "--limit", "10"])
        self.assertEqual(step.mode, "step")
        self.assertEqual(step.limit, 10)
        with self.assertRaises(SystemExit):
            parse_args(["--mode", "e2e", *common])
        e2e = parse_args(["--mode", "e2e", *common, "--tool-url", "http://127.0.0.1:16181"])
        self.assertEqual(e2e.tool_url, "http://127.0.0.1:16181")
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation.ToolVQAEvaluationTest.test_cli_requires_tool_url_only_for_e2e -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.evaluate_toolvqa'`.

- [ ] **Step 3: Implement the thin CLI**

Create `scripts/evaluate_toolvqa.py`:

```python
"""Run ToolVQA's official OpenCompass evaluator through a generated config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.toolvqa import run_toolvqa


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("step", "e2e"), required=True)
    parser.add_argument("--toolvqa-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--tool-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--limit", type=_positive_int)
    args = parser.parse_args(argv)
    if args.mode == "e2e" and not args.tool_url:
        parser.error("--tool-url is required when --mode e2e")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_toolvqa(
        mode=args.mode,
        toolvqa_root=args.toolvqa_root,
        data_root=args.data_root,
        media_root=args.media_root,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_model=args.api_model,
        tool_url=args.tool_url,
        python_executable=args.python,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the Chinese usage handoff**

Create `docs/phase2_toolvqa_evaluation_usage.md` with these sections:

1. **测试来源：** ToolVQA official repository, `GTABenchDataset`, `AgentInferencer`, and `GTABenchEvaluator`.
2. **两种模式：** step tests annotated-history next-step policy; e2e tests the real AgentLego loop.
3. **Smoke data:** explicitly name `examples/data/toolvqa/dataset.json`, the existing JPG, and `comparable=false`.
4. **Prerequisites:** the ToolVQA OpenCompass environment, a running Qwen3-VL OpenAI-compatible API, `OPENAI_API_KEY=EMPTY` for local servers, and AgentLego only for e2e.
5. **Windows CMD commands:** use one physical line and no PowerShell backticks.
6. **PowerShell commands:** use backticks only in the PowerShell-labeled block.
7. **AutoDL/Linux commands:** use backslash continuation.
8. **Metrics:** list the four step and five e2e normalized keys and state that values come unchanged from official `gta_bench.json`.
9. **Artifacts:** `runtime_config.py`, `data/dataset.json`, `upstream/`, `stdout.log`, `stderr.log`, and `result.json`.

The Windows CMD smoke command must be:

```cmd
set OPENAI_API_KEY=EMPTY
python scripts\evaluate_toolvqa.py --mode step --toolvqa-root F:\code\ToolVQA-release --data-root F:\code\ToolVista\examples\data\toolvqa --api-base http://127.0.0.1:8000/v1 --api-model qwen3-vl --output-dir runs\toolvqa-smoke --limit 1
```

State clearly that running this command requires the model API to be started first and that the resulting one-record training-fixture score is not an official benchmark score.

- [ ] **Step 5: Run focused tests and CLI help, then commit Task 3**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation -v
python scripts\evaluate_toolvqa.py --help
```

Expected: 8 tests pass; help lists `step`, `e2e`, `--media-root`, `--tool-url`, and `--limit`.

Commit:

```bash
git add scripts/evaluate_toolvqa.py tests/test_toolvqa_evaluation.py docs/phase2_toolvqa_evaluation_usage.md
git commit -m "docs: add ToolVQA evaluation command"
```

### Task 4: Verification and Upstream-Reuse Audit

**Files:**
- Modify only if verification exposes a ToolVQA MVP defect in the five planned files.

- [ ] **Step 1: Run syntax compilation**

Run:

```powershell
python -m compileall toolvista scripts tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Run focused ToolVQA tests**

Run:

```powershell
python -m unittest tests.test_toolvqa_evaluation -v
```

Expected: 8 tests pass.

- [ ] **Step 3: Run the complete repository regression suite**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all existing tests plus the new ToolVQA tests pass. Record the exact count in the final handoff.

- [ ] **Step 4: Verify generated config without calling a live model**

Run a short read-only Python command that calls `prepare_data` and `render_runtime_config` for the local fixture into a temporary directory, then assert:

```text
prepared dataset exists
prepared image path exists
runtime config imports GTABenchEvaluator
runtime config contains every_with_gt
runtime config contains tool_meta
runtime config does not contain tool_server
runtime config does not contain an actual OPENAI_API_KEY value
```

Expected: all assertions pass. Remove only the temporary directory created by this verification.

- [ ] **Step 5: Audit repository boundaries**

Run:

```powershell
git diff --check
git status --short
git -C F:\code\ToolVQA-release status --short
```

Expected:

- `git diff --check` produces no warnings for MVP files.
- ToolVista still shows the user's unrelated `.gitignore` and `data/` changes untouched.
- `ToolVQA-release` shows no changes caused by this work.

- [ ] **Step 6: Attempt the real smoke only when its external prerequisites exist**

Check whether a Qwen3-VL API is already running and whether the selected ToolVQA Python can import its OpenCompass/Lagent dependencies. If both are available, run the documented one-record `step` command and verify the official `gta_bench.json` plus compact `result.json` exist.

If either external prerequisite is absent, do not install, deploy, or fake it in this MVP. Report that automated boundary verification passed and list the exact missing prerequisite preventing a live smoke.

- [ ] **Step 7: Commit verification fixes only when needed**

If verification required code or documentation fixes, commit only those planned files:

```bash
git add toolvista/evaluation/toolvqa.py scripts/evaluate_toolvqa.py examples/data/toolvqa/dataset.json tests/test_toolvqa_evaluation.py docs/phase2_toolvqa_evaluation_usage.md
git commit -m "fix: harden ToolVQA evaluation bridge"
```

If no fixes were required, do not create an empty commit.
