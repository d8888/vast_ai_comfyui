"""Remote background jobs from the control side (SPEC.md section 10)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import Config
from .errors import E_JOB_FAILED, E_SMOKE_FAILED, JobTimeoutError, VastComfyError
from .remote import current_link, invoke
from .ssh import SSHRunner

_TERMINAL_STATES = {"succeeded", "failed", "timeout", "crashed"}


def start(
    runner: SSHRunner,
    cfg: Config,
    session_id: str,
    name: str,
    command: list[str],
    *,
    timeout_seconds: float,
    env: list[str] | None = None,
    start_timeout_seconds: int = 60,
) -> dict:
    argv = [
        "start",
        "--session-id",
        session_id,
        "--name",
        name,
        "--timeout-seconds",
        str(float(timeout_seconds)),
    ]
    for item in env or []:
        argv += ["--env", item]
    argv += ["--", *command]
    return invoke(runner, cfg, "vc_jobs.py", argv, timeout_seconds=start_timeout_seconds)


def status(runner: SSHRunner, cfg: Config, job_id: str) -> dict:
    return invoke(runner, cfg, "vc_jobs.py", ["status", "--job-id", job_id], timeout_seconds=60)


def logs(runner: SSHRunner, cfg: Config, job_id: str, *, tail: int | None = None) -> dict:
    tail_lines = int(tail if tail is not None else cfg.require("jobs.log_tail_lines"))
    return invoke(
        runner, cfg, "vc_jobs.py", ["logs", "--job-id", job_id, "--tail", str(tail_lines)], timeout_seconds=60
    )


def list_jobs(runner: SSHRunner, cfg: Config, *, session_id: str | None = None) -> dict:
    argv = ["list"]
    if session_id:
        argv += ["--session-id", session_id]
    return invoke(runner, cfg, "vc_jobs.py", argv, timeout_seconds=60)


def wait(
    runner: SSHRunner,
    cfg: Config,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float | None = None,
    on_poll: Callable[[dict], None] | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Bounded wait. On timeout the job is NOT stopped; the error says so."""
    poll = float(poll_seconds if poll_seconds is not None else cfg.require("jobs.wait_poll_seconds"))
    deadline = clock() + timeout_seconds
    last: dict[str, Any] = {}
    while True:
        last = status(runner, cfg, job_id)
        if on_poll:
            on_poll(last)
        state = str(last.get("state"))
        if state in _TERMINAL_STATES:
            return last
        if state == "overdue":
            raise JobTimeoutError(job_id, timeout_seconds, details={"last_status": last})
        if clock() >= deadline:
            raise JobTimeoutError(job_id, timeout_seconds, details={"last_status": last})
        sleep(poll)


def wait_or_fail(runner: SSHRunner, cfg: Config, job_id: str, *, timeout_seconds: float, **kw: Any) -> dict:
    final = wait(runner, cfg, job_id, timeout_seconds=timeout_seconds, **kw)
    state = str(final.get("state"))
    if state != "succeeded":
        tail = ""
        try:
            tail = logs(runner, cfg, job_id, tail=60).get("log", "")
        except VastComfyError:
            tail = "<log unavailable>"
        raise VastComfyError(
            code=E_JOB_FAILED,
            message=f"Remote job {job_id} finished in state {state!r} (exit_code={final.get('exit_code')})",
            stage="jobs",
            details={"job_id": job_id, "state": state, "log_tail": tail[-2000:]},
        )
    return final


def prepare_command(cfg: Config, tool: str, argv: list[str]) -> list[str]:
    venv_python = str(cfg.require("image.venv_python"))
    return [venv_python, f"{current_link(cfg)}/remote/{tool}", *argv]


def final_json_from_log(log_text: str) -> dict[str, Any] | None:
    """Extract the last ``{"ok": ...}`` JSON object printed by a remote tool."""
    for line in reversed(log_text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return None
