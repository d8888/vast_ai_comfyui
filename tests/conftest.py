"""Shared pytest fixtures for the vast-comfy test suite.

Nothing here touches the network, real SSH or the real Vast.ai API: every
integration point is replaced by a local mock (see ``tests/mocks/``).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Must run before any test module is imported: several modules build the deploy
# package at import time, and that package is profile-dependent. Set here rather
# than only in a fixture so import-time evaluation sees the same profile.
os.environ.setdefault("VAST_COMFY_PROFILE", "anima")

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_AGENT_DIR = REPO_ROOT / "src" / "vast_comfy" / "remote_agent"

_REMOTE_MODULES = ("vc_common", "vc_comfy", "vc_smoke")


def ensure_remote_agent_on_path() -> None:
    """Put the deployed-on-the-instance scripts on sys.path (idempotent)."""
    if str(REMOTE_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_AGENT_DIR))


@pytest.fixture(autouse=True)
def _default_test_profile(monkeypatch):
    """Pin the suite-wide default to ``anima``.

    Production defaults to ``noobai``, but the bulk of the existing tests assert
    the Anima suite (LLLite cases, Qwen encoders, the 19-file package). Tests
    that exercise the NoobAI profile request ``noobai_profile`` explicitly,
    which runs after this fixture and therefore wins.
    """
    monkeypatch.setenv("VAST_COMFY_PROFILE", "anima")
    return "anima"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def anima_profile(monkeypatch):
    """Pin the legacy Anima expectations to the ``anima`` profile.

    Profile selection lives in config/env, so tests that assert the Anima suite
    (LLLite cases, Qwen encoders) must say which profile they mean instead of
    relying on whatever the default happens to be.
    """
    monkeypatch.setenv("VAST_COMFY_PROFILE", "anima")
    return "anima"


@pytest.fixture()
def noobai_profile(monkeypatch):
    monkeypatch.setenv("VAST_COMFY_PROFILE", "noobai")
    return "noobai"


@pytest.fixture()
def cfg():
    """Built-in default configuration (no YAML, no CLI overrides)."""
    from vast_comfy.config import load_config

    return load_config()


@pytest.fixture()
def mapping():
    """The committed GPU alias mapping (config/gpu_aliases.json)."""
    from vast_comfy.gpu import load_mapping

    return load_mapping()


@pytest.fixture()
def remote_agent(tmp_path, monkeypatch):
    """Import the remote-agent scripts with tmp runtime/ComfyUI directories.

    The scripts read ``VAST_COMFY_RUNTIME_DIR`` / ``VAST_COMFY_COMFY_DIR`` at
    import time, so the modules are reloaded with the environment pointed at
    per-test tmp dirs. They keep the ``import vc_common as C`` style.
    """
    runtime_dir = tmp_path / "runtime"
    comfy_dir = tmp_path / "ComfyUI"
    monkeypatch.setenv("VAST_COMFY_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("VAST_COMFY_COMFY_DIR", str(comfy_dir))
    ensure_remote_agent_on_path()
    importlib.invalidate_caches()

    modules = {}
    for name in _REMOTE_MODULES:
        module = importlib.import_module(name)
        modules[name] = importlib.reload(module)

    return SimpleNamespace(
        C=modules["vc_common"],
        CM=modules["vc_comfy"],
        SMOKE=modules["vc_smoke"],
        runtime_dir=runtime_dir,
        comfy_dir=comfy_dir,
    )
