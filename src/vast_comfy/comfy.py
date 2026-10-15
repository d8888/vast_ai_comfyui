"""ComfyUI service operations from the control side (thin wrappers)."""

from __future__ import annotations

from .config import Config
from .remote import invoke
from .ssh import SSHRunner


def _comfy_url(cfg: Config) -> str:
    return f"http://localhost:{int(cfg.require('image.internal_port'))}"


def status(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        ["status", "--session-id", session_id, "--comfy-url", _comfy_url(cfg)],
        timeout_seconds=120,
    )


def health(runner: SSHRunner, cfg: Config, session_id: str, *, gpu: bool = False) -> dict:
    argv = ["health", "--session-id", session_id, "--comfy-url", _comfy_url(cfg)]
    if gpu:
        argv.append("--gpu")
    return invoke(runner, cfg, "vc_comfy.py", argv, timeout_seconds=180)


def restart(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        [
            "restart",
            "--session-id",
            session_id,
            "--comfy-url",
            _comfy_url(cfg),
            "--wait-seconds",
            str(cfg.get("comfy.restart_wait_seconds", 240)),
        ],
        timeout_seconds=int(cfg.get("comfy.restart_wait_seconds", 240)) + 120,
    )


def verify_nodes(runner: SSHRunner, cfg: Config, session_id: str, nodes: list[str]) -> dict:
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        [
            "verify-nodes",
            "--session-id",
            session_id,
            "--nodes",
            ",".join(nodes),
            "--comfy-url",
            _comfy_url(cfg),
        ],
        timeout_seconds=180,
    )


def gpu_check(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    return invoke(
        runner, cfg, "vc_comfy.py", ["gpu-check", "--session-id", session_id], timeout_seconds=120
    )


def queue(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        ["queue", "--session-id", session_id, "--comfy-url", _comfy_url(cfg)],
        timeout_seconds=120,
    )


def logs(runner: SSHRunner, cfg: Config, session_id: str, *, tail: int = 200) -> dict:
    """ComfyUI service logs (management-job logs use ``jobs logs`` instead)."""
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        ["logs", "--session-id", session_id, "--tail", str(int(tail))],
        timeout_seconds=120,
    )
