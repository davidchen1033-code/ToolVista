"""BrowserGym adapter and a process boundary for its synchronous Playwright API."""
from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
from urllib.parse import urlparse

from .actions import browsergym_action
from .dependencies import check_versions
from .gigpo import BROWSER_STATE_JS, state_anchor


def local_miniwob_url(value: str | None = None) -> str:
    value = value or os.environ.get("MINIWOB_URL", "")
    parsed = urlparse(value)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        pass
    elif parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        pass
    else:
        raise ValueError("MINIWOB_URL must point to local MiniWoB assets (file:///.../miniwob/html/miniwob/ or http://127.0.0.1:PORT/)")
    if parsed.query or parsed.fragment:
        raise ValueError("MINIWOB_URL must be a directory URL")
    return value.rstrip("/") + "/"


class MiniWoBEnv:
    def __init__(self, artifact_dir: str | Path, *, base_url: str | None = None, max_steps: int = 15):
        self.artifact_dir = Path(artifact_dir)
        self.base_url = local_miniwob_url(base_url)
        self.max_steps = max_steps
        self.env = None
        self.index = 0
        self.last = None
        self.versions = {}

    def reset(self, *, task_id: str, seed: int) -> dict:
        if not re.fullmatch(r"browsergym/miniwob\.[a-z][a-z0-9-]*", task_id):
            raise ValueError(f"invalid MiniWoB task id: {task_id}")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.versions = check_versions(["browsergym-core", "browsergym-miniwob", "gymnasium", "playwright"])
        import browsergym.miniwob  # noqa: F401 - registers Gymnasium environments
        from browsergym.core.action.highlevel import HighLevelActionSet
        import gymnasium

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        action_set = HighLevelActionSet(subsets=["coord"], multiaction=False, strict=True)
        self.env = gymnasium.make(task_id, headless=True, disable_env_checker=True,
                                  action_mapping=action_set.to_python_code,
                                  task_kwargs={"base_url": self.base_url, "episode_max_time": 1000000},
                                  max_episode_steps=self.max_steps)
        observation, info = self.env.reset(seed=seed)
        self.index = 0
        self.last = self._observation(observation, info)
        return self.last

    def _observation(self, observation: dict, info: dict) -> dict:
        from PIL import Image

        image = Image.fromarray(observation["screenshot"]).convert("RGB")
        destination = (self.artifact_dir / f"{self.index:03d}.png").resolve()
        image.save(destination)
        # DOM and accessibility trees are diagnostics only; never include them in the actor observation.
        diagnostics = {"info": info, "dom_object": observation.get("dom_object"),
                       "axtree_object": observation.get("axtree_object")}
        destination.with_suffix(".diagnostics.json").write_text(
            json.dumps(diagnostics, default=str, ensure_ascii=False), encoding="utf-8")
        result = {"image_path": str(destination), "image_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "width": image.width, "height": image.height, "goal": str(observation["goal"]),
                "step": self.index, "feedback": "action_error" if observation.get("last_action_error") else "ready",
                "environment": {**self.versions, "base_url": self.base_url,
                                "asset_revision": os.environ.get("MINIWOB_ASSET_REVISION", "unrecorded")}}
        result["state_anchor"] = state_anchor(result, self.env.unwrapped.page.evaluate(BROWSER_STATE_JS))
        return result

    def step(self, action: dict) -> dict:
        if self.env is None or self.last is None:
            raise RuntimeError("reset must be called before step")
        command = browsergym_action(action, self.last["width"], self.last["height"])
        observation, reward, terminated, truncated, info = self.env.step(command)
        self.index += 1
        self.last = self._observation(observation, info)
        return {"observation": self.last, "environment_reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "success": bool(terminated and reward > 0),
                "invalid_action": bool(observation.get("last_action_error")),
                "feedback": self.last["feedback"]}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def _environment_worker(connection, options):
    env = None
    try:
        if os.name == "posix":
            os.setsid()  # A timeout can reap Playwright/Chromium descendants as well.
        env = MiniWoBEnv(**options)
        while True:
            method, kwargs = connection.recv()
            try:
                result = getattr(env, method)(**kwargs)
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
            if method == "close":
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class AsyncMiniWoBEnv:
    """One spawned process per episode, with deadline-bounded IPC and cleanup."""
    def __init__(self, artifact_dir: str | Path, *, max_steps: int = 15, timeout: float = 60,
                 base_url: str | None = None):
        self.options = {"artifact_dir": str(artifact_dir), "max_steps": max_steps, "base_url": base_url}
        self.timeout = timeout
        self.process = None
        self.connection = None

    def _rpc(self, method: str, **kwargs):
        self.connection.send((method, kwargs))
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"BrowserGym {method} exceeded {self.timeout}s")
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(f"BrowserGym {method}: {result}")
        return result

    async def reset(self, **kwargs):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_environment_worker, args=(child, self.options), daemon=True)
        self.process.start()
        child.close()
        return await asyncio.to_thread(self._rpc, "reset", **kwargs)

    async def step(self, action):
        return await asyncio.to_thread(self._rpc, "step", action=action)

    async def close(self):
        if self.process is None:
            return
        try:
            if self.process.is_alive():
                self.timeout = min(self.timeout, 5)
                try:
                    await asyncio.to_thread(self._rpc, "close")
                except (RuntimeError, TimeoutError, EOFError, BrokenPipeError, OSError):
                    pass
                await asyncio.to_thread(self.process.join, 2)
                if self.process.is_alive():
                    if os.name == "posix" and os.getpgid(self.process.pid) == self.process.pid:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    else:
                        self.process.terminate()
                    await asyncio.to_thread(self.process.join, 2)
                if self.process.is_alive():
                    self.process.kill()
                    await asyncio.to_thread(self.process.join, 2)
        finally:
            self.connection.close()
            self.process.close()
            self.process = None
