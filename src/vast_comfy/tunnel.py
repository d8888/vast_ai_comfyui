"""SSH tunnel helpers (SPEC.md 7.2): local-only port forwarding to ComfyUI.

The tunnel binds to 127.0.0.1 on the control machine only, and OpenSSH options
must precede the destination host so that ``-N``/``-L`` really create a forward
instead of being treated as a remote command. The official Vast.ai Portal
(with its authentication) remains the other access path.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from .config import Config
from .errors import E_SSH_COMMAND, UsageError, VastComfyError
from .ssh import SSHRunner


def tunnel_plan(cfg: Config, runner: SSHRunner, *, local_port: int = 8188) -> dict:
    local_port = int(local_port)
    if not 1 <= local_port <= 65535:
        raise UsageError(f"--local-port must be between 1 and 65535 (got {local_port})")
    remote_port = int(cfg.require("image.internal_port"))
    base = runner.tunnel_argv(local_port, remote_port)
    argv = [base[0], "-o", "ExitOnForwardFailure=yes", *base[1:]]
    return {
        "ok": True,
        "local_port": local_port,
        "remote_port": remote_port,
        "argv": argv,
        "command": shlex.join(argv),
        "url": f"http://127.0.0.1:{local_port}/",
        "bind": "127.0.0.1 (control machine only)",
        "note": (
            "The tunnel reaches ComfyUI's internal backend port directly. "
            "The official Portal (Open button) remains the authenticated alternative."
        ),
    }


def start_tunnel(
    cfg: Config,
    runner: SSHRunner,
    *,
    instance_id: int | str,
    state,
    local_port: int = 8188,
) -> dict:
    plan = tunnel_plan(cfg, runner, local_port=local_port)
    log_dir = cfg.resolved_paths()["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tunnel-{instance_id}-{local_port}.log"
    try:
        with open(log_path, "ab") as log_fh:
            kwargs: dict = {"stdout": log_fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
            if os.name == "nt":
                kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(plan["argv"], **kwargs)
    except FileNotFoundError as exc:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message="The tunnel process could not be started",
            stage="tunnel",
            details={"log": str(log_path)},
        ) from exc
    time.sleep(1.0)
    if proc.poll() is not None:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message="The tunnel process exited immediately; check the tunnel log",
            stage="tunnel",
            details={"log": str(log_path)},
        )
    state.put(
        "tunnels",
        str(instance_id),
        {
            "instance_id": str(instance_id),
            "pid": proc.pid,
            "local_port": int(local_port),
            "url": plan["url"],
            "log": str(log_path),
        },
    )
    plan.update({"started": True, "pid": proc.pid, "log": str(log_path)})
    return plan


def stop_tunnel(instance_id: int | str, state) -> dict:
    record = state.get("tunnels", str(instance_id))
    if not record:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message=f"No tunnel is recorded for instance {instance_id}",
            stage="tunnel",
        )
    pid = int(record.get("pid", 0))
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    state.remove("tunnels", str(instance_id))
    return {"ok": True, "stopped": True, "pid": pid, "note": "The tunnel was local only; the instance is unaffected."}
