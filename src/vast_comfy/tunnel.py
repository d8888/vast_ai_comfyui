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
    forward = f"127.0.0.1:{local_port}:localhost:{remote_port}"
    endpoint = f"{runner.ctx.user}@{runner.ctx.host}"
    return {
        "ok": True,
        "local_port": local_port,
        "remote_port": remote_port,
        "forward": forward,
        "endpoint": endpoint,
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
            "remote_port": plan["remote_port"],
            # Identity fields: stop_tunnel verifies the live process against
            # these before ever sending a signal (PID reuse protection).
            "forward": plan["forward"],
            "endpoint": plan["endpoint"],
            "url": plan["url"],
            "log": str(log_path),
        },
    )
    plan.update({"started": True, "pid": proc.pid, "log": str(log_path)})
    return plan


_RECORD_HINT = (
    "Nothing was signalled. Verify the process yourself (ps / tasklist) and, if it is not a "
    "vast-comfy SSH tunnel, remove the stale record from the local state file instead of "
    "stopping it."
)


def _process_command_text(pid: int) -> tuple[str | None, bool]:
    """Current command line of a local PID, for identity verification only.

    Returns ``(text, exists)``: ``exists=False`` means the process is gone;
    ``text=None`` with ``exists=True`` means it exists but its command line
    cannot be read on this platform (verification impossible). Read-only.
    """
    if os.name == "posix":
        if not Path("/proc").is_dir():
            return None, True  # e.g. macOS: cannot verify the identity
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError:
            return None, False
        except OSError:
            return None, True
        text = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        return (text, True) if text else (None, False)  # empty = zombie/kernel thread
    if os.name == "nt":
        command = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, True
        if result.returncode != 0:
            return None, True
        text = (result.stdout or "").strip()
        return (text, True) if text else (None, False)
    return None, True


def stop_tunnel(instance_id: int | str, state) -> dict:
    record = state.get("tunnels", str(instance_id))
    if not record:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message=f"No tunnel is recorded for instance {instance_id}",
            stage="tunnel",
        )
    pid = record.get("pid")
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message="The stored tunnel record does not contain a safe PID; nothing was signalled.",
            stage="tunnel",
            details={"pid": pid, "instance_id": str(instance_id)},
            hints=[_RECORD_HINT],
        )
    forward = record.get("forward")
    endpoint = record.get("endpoint")
    if not (isinstance(forward, str) and forward and isinstance(endpoint, str) and endpoint):
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message=(
                "The stored tunnel record has no identity fields to verify against "
                "(from an older version); nothing was signalled."
            ),
            stage="tunnel",
            details={"pid": pid, "instance_id": str(instance_id)},
            hints=[_RECORD_HINT],
        )
    text, exists = _process_command_text(pid)
    if not exists:
        state.remove("tunnels", str(instance_id))
        return {
            "ok": True,
            "stopped": False,
            "already_exited": True,
            "pid": pid,
            "note": "The recorded tunnel process is already gone; nothing was signalled.",
        }
    if text is None:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message=(
                "The command line of the recorded PID cannot be read, so its identity cannot "
                "be confirmed; nothing was signalled."
            ),
            stage="tunnel",
            details={"pid": pid, "instance_id": str(instance_id)},
            hints=[_RECORD_HINT],
        )
    normalized = text.replace('"', "")
    if "ssh" not in normalized or forward not in normalized or endpoint not in normalized:
        raise VastComfyError(
            code=E_SSH_COMMAND,
            message=(
                "The recorded PID no longer runs this tunnel command (the PID may have been "
                "reused); nothing was signalled."
            ),
            stage="tunnel",
            details={"pid": pid, "instance_id": str(instance_id),
                     "expected_forward": forward, "expected_endpoint": endpoint},
            hints=[_RECORD_HINT],
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    state.remove("tunnels", str(instance_id))
    return {
        "ok": True,
        "stopped": True,
        "pid": pid,
        "note": "The tunnel was local only; the instance is unaffected.",
    }
