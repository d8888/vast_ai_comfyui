"""In-process fake of the ComfyUI HTTP API for remote smoke-runner tests.

``vc_smoke`` talks to ComfyUI exclusively through ``vc_comfy.http_json`` and
``vc_comfy.gpu_check``; tests monkeypatch those two with this fake and never
open a socket.

The fake:

* answers ``/system_stats``, ``/queue``, ``/object_info``, ``/prompt`` and
  ``/history/{id}``,
* writes a real PNG into ``<comfy_dir>/output/`` for every accepted prompt so
  the runner's PIL validation has something to check,
* lets the test choose the image builder (good noise / flat single color /
  wrong size / undecodable / stale mtime) and which workflow prefixes fail.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from PIL import Image

ImageBuilder = Callable[[Path], None]


def noise_png(path: Path, size: tuple[int, int] = (1024, 1024)) -> None:
    """Deterministic, multi-colour PNG (many distinct colors)."""
    rng = random.Random(20261015)
    small = Image.frombytes("L", (256, 256), rng.randbytes(256 * 256))
    img = small.convert("RGB").resize(size, Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def flat_png(path: Path, size: tuple[int, int] = (1024, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (250, 248, 246)).save(path, format="PNG")


def wrong_size_png(path: Path) -> None:
    noise_png(path, size=(512, 512))


def undecodable_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not a PNG file\n")


def old_png(path: Path) -> None:
    noise_png(path)
    stale = time.time() - 3600.0
    os.utime(path, (stale, stale))


class FakeComfyUI:
    def __init__(
        self,
        comfy_dir: Path,
        *,
        object_info: dict[str, Any] | None = None,
        busy: bool = False,
        image_builder: ImageBuilder = noise_png,
        fail_prefixes: tuple[str, ...] = (),
        system_stats: dict[str, Any] | None = None,
    ) -> None:
        self.comfy_dir = Path(comfy_dir)
        self.object_info = object_info if object_info is not None else {}
        self.busy = busy
        self.image_builder = image_builder
        self.fail_prefixes = set(fail_prefixes)
        self.system_stats = system_stats or {
            "system": {"comfyui_version": "0.3.60", "python_version": "3.12", "os": "posix"},
            "devices": [
                {
                    "name": "cuda:0 NVIDIA GeForce RTX 4060 Ti",
                    "type": "cuda",
                    "vram_total": 17163091968,
                    "vram_free": 16000000000,
                    "torch_version": "2.7.0",
                }
            ],
        }
        self.prompts: list[dict[str, Any]] = []
        self.history_calls: list[str] = []
        self._prefix_by_prompt: dict[str, str] = {}
        self._counter = 0

    # -- vc_comfy.http_json replacement ------------------------------------

    def http_json(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        method: str = "GET",
        body: dict | None = None,
    ) -> Any:
        path = urlsplit(url).path
        if path.endswith("/system_stats"):
            return self.system_stats
        if path.endswith("/queue"):
            return {
                "queue_running": [["1", "prompt"]] if self.busy else [],
                "queue_pending": [],
            }
        if path.endswith("/object_info"):
            return dict(self.object_info)
        if path.endswith("/prompt") and method == "POST":
            assert body is not None
            prompt = body["prompt"]
            self.prompts.append(prompt)
            prompt_id = f"prompt-{len(self.prompts)}"
            self._prefix_by_prompt[prompt_id] = self._filename_prefix(prompt)
            return {"prompt_id": prompt_id, "number": len(self.prompts)}
        if "/history/" in path:
            prompt_id = path.rsplit("/", 1)[1]
            self.history_calls.append(prompt_id)
            prefix = self._prefix_by_prompt.get(prompt_id, "unknown")
            if prefix in self.fail_prefixes:
                return {
                    prompt_id: {
                        "status": {
                            "completed": True,
                            "status_str": "error",
                            "messages": [["execution_error", {"error": "simulated failure"}]],
                        },
                        "outputs": {},
                    }
                }
            filename = f"{prefix}_{self._counter:04d}.png"
            self._counter += 1
            self.image_builder(self.comfy_dir / "output" / filename)
            return {
                prompt_id: {
                    "status": {"completed": True, "status_str": "success", "messages": []},
                    "outputs": {
                        "9": {
                            "images": [
                                {"type": "output", "subfolder": "", "filename": filename}
                            ]
                        }
                    },
                }
            }
        raise AssertionError(f"FakeComfyUI got an unexpected URL: {url}")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _filename_prefix(prompt: dict[str, Any]) -> str:
        for node in prompt.values():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                return str(node.get("inputs", {}).get("filename_prefix", "output"))
        return "output"


def gpu_check_stub() -> dict[str, Any]:
    return {"gpus": ["NVIDIA GeForce RTX 4060 Ti, 16376 MiB, 550.54"]}
