"""Exclusive session lifecycle on the control side (SPEC.md section 8)."""

from __future__ import annotations

import contextlib
import getpass
import json
import socket
import sys
import time
import uuid

from .config import Config
from .errors import E_LOCK_NOT_HELD, VastComfyError
from .remote import ensure_bootstrap, invoke
from .ssh import SSHRunner


def new_session_id() -> str:
    return "vc-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def controller_info() -> dict:
    return {"hostname": socket.gethostname(), "user": getpass.getuser(), "platform": sys.platform}


def acquire(
    runner: SSHRunner,
    cfg: Config,
    instance_id: int | str,
    *,
    session_id: str | None = None,
    accept_stale: bool = False,
    timeout_seconds: int = 60,
) -> dict:
    """Acquire (or recover) the exclusive session lock.

    A conflict raises ``E_LOCK_CONFLICT`` with the owner's information; a stale
    owner record raises ``E_LOCK_STALE`` unless ``accept_stale`` is explicitly set
    by the operator (user-driven recovery).
    """
    sid = session_id or new_session_id()
    ensure_bootstrap(runner, cfg)
    argv = [
        "acquire",
        "--session-id",
        sid,
        "--instance-id",
        str(instance_id),
        "--controller",
        json.dumps(controller_info()),
        "--heartbeat-seconds",
        str(cfg.require("lock.heartbeat_seconds")),
        "--ttl-seconds",
        str(cfg.require("lock.heartbeat_ttl_seconds")),
        "--buffer-seconds",
        str(cfg.require("lock.heartbeat_buffer_seconds")),
    ]
    if accept_stale:
        argv.append("--accept-stale")
    payload = invoke(runner, cfg, "vc_session.py", argv, timeout_seconds=timeout_seconds)
    payload["session_id"] = sid
    return payload


def status(runner: SSHRunner, cfg: Config, *, session_id: str | None = None) -> dict:
    ensure_bootstrap(runner, cfg)
    return invoke(
        runner,
        cfg,
        "vc_session.py",
        ["status", "--session-id", session_id or ""],
        timeout_seconds=60,
    )


def release(runner: SSHRunner, cfg: Config, session_id: str, *, timeout_seconds: int = 90) -> dict:
    ensure_bootstrap(runner, cfg)
    return invoke(
        runner,
        cfg,
        "vc_session.py",
        ["release", "--session-id", session_id],
        timeout_seconds=timeout_seconds,
    )


def recover(
    runner: SSHRunner, cfg: Config, instance_id: int | str, *, session_id: str | None = None
) -> dict:
    """Explicit, user-driven recovery after confirming the previous owner is dead.

    Still refuses when the stale session's jobs are alive or when the flock is
    actually held; the runtime decides, this call only expresses user intent.
    """
    return acquire(runner, cfg, instance_id, session_id=session_id, accept_stale=True)


def assert_owns(runner: SSHRunner, cfg: Config, session_id: str) -> dict:
    """Verify (live) that this session owns the remote lock; raise otherwise.

    Used as a controller-side guard for read/query operations that SPEC.md 8.1
    subjects to the session rules.
    """
    payload = status(runner, cfg, session_id=session_id)
    if not payload.get("lock_held") or not payload.get("matches_me"):
        raise VastComfyError(
            code=E_LOCK_NOT_HELD,
            message="This session does not own the live remote lock; the operation is refused.",
            stage="session",
            details={
                "session_id": session_id,
                "lock_held": payload.get("lock_held"),
                "owner": payload.get("owner"),
            },
            hints=["Acquire the session first (vast-comfy session acquire)."],
        )
    return payload


class OperationHandle:
    def __init__(self, token):
        self.token, self.retained = token, False

    def retain(self):
        """Keep exclusion when stop/destroy is accepted and SSH will disappear."""
        self.retained = True


@contextlib.contextmanager
def operation(runner: SSHRunner, cfg: Config, session_id: str, *, label: str = "operation"):
    """Hold remote exclusion across multiple SSH/SCP or platform API calls."""
    existing = getattr(runner, "_vast_operation_token", None)
    if isinstance(existing, str):
        assert_owns(runner, cfg, session_id)
        yield OperationHandle(existing)
        return
    ensure_bootstrap(runner, cfg)
    token = uuid.uuid4().hex
    invoke(runner, cfg, "vc_session.py", ["begin-operation", "--session-id", session_id,
           "--operation-token", token, "--label", label], timeout_seconds=60)
    runner._vast_operation_token = token
    handle = OperationHandle(token)
    error = None
    try:
        yield handle
    except BaseException as exc:
        error = exc
        raise
    finally:
        runner._vast_operation_token = None
        if not handle.retained:
            try:
                invoke(runner, cfg, "vc_session.py", ["end-operation", "--session-id", session_id,
                       "--operation-token", token], timeout_seconds=60)
            except VastComfyError as exc:
                if error is None:
                    raise
                if isinstance(error, VastComfyError):
                    error.details["operation_release_error"] = exc.to_dict()
