"""ComfyUI service operations from the control side (thin wrappers)."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .errors import UsageError
from .remote import invoke
from .ssh import SSHRunner

# A single reference image, not a model: anything beyond this is a mistake.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _comfy_url(cfg: Config) -> str:
    return f"http://localhost:{int(cfg.require('image.internal_port'))}"


def status(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        ["status", "--session-id", session_id, "--comfy-url", _comfy_url(cfg),
         "--service", str(cfg.require("image.supervisor_service"))],
        timeout_seconds=120,
    )


def health(runner: SSHRunner, cfg: Config, session_id: str, *, gpu: bool = False) -> dict:
    argv = ["health", "--session-id", session_id, "--comfy-url", _comfy_url(cfg)]
    if gpu:
        argv.append("--gpu")
    return invoke(
        runner,
        cfg,
        "vc_comfy.py",
        argv,
        timeout_seconds=int(cfg.require("comfy.health_timeout_seconds")),
    )


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
            "--service", str(cfg.require("image.supervisor_service")),
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
        ["logs", "--session-id", session_id, "--tail", str(int(tail)),
         "--service", str(cfg.require("image.supervisor_service"))],
        timeout_seconds=120,
    )


def upload_image(
    runner: SSHRunner,
    cfg: Config,
    session_id: str,
    *,
    local_path: Path,
    name: str,
) -> dict:
    """Copy a PNG from the control machine into ComfyUI's ``input`` directory.

    The file is scp'd to a runtime staging path first; the remote side verifies
    the checksum and only then publishes it, so a truncated transfer is never
    visible to a workflow and an existing different file is never overwritten.
    """
    from . import remote as remote_mod

    local = Path(local_path)
    if not local.is_file():
        raise UsageError(f"image not found: {local}")
    if local.suffix.lower() != ".png":
        raise UsageError(f"only PNG images can be uploaded; got {local.name!r}")
    if local.stat().st_size > MAX_UPLOAD_BYTES:
        raise UsageError(
            f"image is {local.stat().st_size} bytes; the upload limit is {MAX_UPLOAD_BYTES}"
        )
    digest = remote_mod._sha256(local)
    staging_dir = f"{remote_mod.runtime_root(cfg)}/uploads/{session_id}"
    staged = f"{staging_dir}/{digest}.png"
    runner.run_remote(["mkdir", "-p", staging_dir], check=True)
    runner.upload(local, staged)
    result = invoke(
        runner,
        cfg,
        "vc_comfy.py",
        [
            "upload-image",
            "--session-id", session_id,
            "--name", name,
            "--staged-path", staged,
            "--sha256", digest,
        ],
        timeout_seconds=120,
    )
    result["local_path"] = str(local)
    return result
