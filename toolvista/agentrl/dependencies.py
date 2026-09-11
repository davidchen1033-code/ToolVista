"""Optional dependency checks, kept outside the base ToolVista import path."""
from __future__ import annotations

from importlib import import_module, metadata
import inspect
from pathlib import Path

MINIWOB_REVISION = "33c3b4ddef8c6eb67c57a29663d844b1eda7e614"

PINS = {"verl": "0.7.0", "browsergym-core": "0.13.3", "browsergym-miniwob": "0.13.3",
        "gymnasium": "1.0.0", "playwright": "1.51.0", "transformers": "4.57.3",
        "vllm": "0.11.2", "qwen-vl-utils": "0.0.14", "torch": "2.9.0",
        "peft": "0.17.1", "tensordict": "0.10.0", "ray": "2.49.2",
        "numpy": "1.26.4", "opencv-python-headless": "4.11.0.86",
        "hydra-core": "1.3.2", "pyarrow": "19.0.1"}


def check_versions(names=None) -> dict:
    versions = {}
    for name in (PINS if names is None else names):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing {name}; install configs/verl/requirements-agentrl.txt in the AgentRL venv") from exc
        if versions[name].split("+", 1)[0] != PINS[name]:
            raise RuntimeError(f"Incompatible {name} {versions[name]}; expected {PINS[name]}")
    return versions


def check_chromium():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            raise RuntimeError("Chromium missing; run python -m playwright install --with-deps chromium")


def load_verl_api():
    check_versions(["verl"])
    try:
        api = import_module("verl.experimental.agent_loop.agent_loop")
        parameters = inspect.signature(api.AgentLoopBase.__init__).parameters
        if not {"trainer_config", "server_manager", "tokenizer", "processor", "dataset_cls", "dataset_config"} <= set(parameters):
            raise TypeError("AgentLoopBase constructor changed")
        if not {"prompt_ids", "response_ids", "response_mask", "reward_score", "multi_modal_data"} <= set(api.AgentLoopOutput.model_fields):
            raise TypeError("AgentLoopOutput contract changed")
        return api
    except (ImportError, AttributeError, TypeError) as exc:
        raise RuntimeError(f"Incompatible verl AgentLoop API; expected verl==0.7.0: {exc}") from exc
