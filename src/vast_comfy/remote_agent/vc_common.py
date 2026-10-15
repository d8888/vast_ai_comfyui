#!/usr/bin/env python3
"""vast-comfy remote runtime: shared helpers (stdlib only).

This file is part of the whitelisted management package that vast-comfy
auto-uploads to managed instances. It only ever runs on the Linux instance,
using the ComfyUI image's venv python (``/venv/main/bin/python``) when PIL is
needed. Nothing here reads or writes outside:
  * <runtime_dir>            default /workspace/.vast_comfy
  * the ComfyUI directory    default /workspace/ComfyUI
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

RUNTIME_DIR = Path(os.environ.get("VAST_COMFY_RUNTIME_DIR", "/workspace/.vast_comfy"))
COMFY_DIR = Path(os.environ.get("VAST_COMFY_COMFY_DIR", "/workspace/ComfyUI"))
PKG_DIR = RUNTIME_DIR / "pkg"
CURRENT_PKG = RUNTIME_DIR / "current"
LOCK_DIR = RUNTIME_DIR / "lock"
LOCK_FILE = LOCK_DIR / "owner.lock"
OWNER_FILE = LOCK_DIR / "owner.json"
OWNER_HISTORY = LOCK_DIR / "owner_history.jsonl"
RELEASE_REQUEST = LOCK_DIR / "release.request"
OPERATION_FILE = LOCK_DIR / "operation.lock"
OPERATION_OWNER = LOCK_DIR / "operation.json"
JOBS_DIR = RUNTIME_DIR / "jobs"
RUNS_DIR = RUNTIME_DIR / "runs"
DOWNLOADS_DIR = RUNTIME_DIR / "downloads"
ARTIFACTS_DIR = RUNTIME_DIR / "artifacts"

ALLOWED_MODEL_DIRS = {
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "vae",
    "controlnet",
    "loras",
    "embeddings",
    "clip_vision",
    "upscale_models",
    "model_patches",
    "ipadapter",
    "style_models",
    "audio_encoders",
}


def now_epoch() -> float:
    return time.time()


def iso_utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else time.time()))


def ensure_dirs() -> None:
    for d in (RUNTIME_DIR, PKG_DIR, LOCK_DIR, JOBS_DIR, RUNS_DIR, DOWNLOADS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(LOCK_DIR, 0o700)
    except OSError:
        pass


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload, sort_keys=True))
    sys.stdout.flush()
    sys.exit(exit_code)


class RemoteFail(Exception):
    """Structured remote failure; caught by the module dispatcher."""

    def __init__(self, code: str, message: str, *, exit_code: int = 1, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}


def fail(code: str, message: str, *, exit_code: int = 1, **details: object) -> None:
    raise RemoteFail(code, message, exit_code=exit_code, details=dict(details))


def redact(text: str) -> str:
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1<redacted>", str(text))
    for name, value in os.environ.items():
        if len(value) >= 8 and any(word in name.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")):
            text = text.replace(value, "<redacted>")
    return text


def run_cli(func, args, *, guard: bool = False) -> None:
    """Dispatch a subcommand, converting RemoteFail into structured JSON."""
    try:
        if guard:
            with operation(args.session_id, label=func.__name__):
                result = func(args)
        else:
            result = func(args)
    except RemoteFail as exc:
        payload = {"ok": False, "code": exc.code, "message": exc.message}
        payload.update(exc.details)
        emit(payload, exit_code=exc.exit_code)
    except Exception as exc:
        emit({"ok": False, "code": "E_INTERNAL", "message": redact(str(exc)),
              "stage": func.__name__, "takeover_required": True}, exit_code=1)
    if isinstance(result, dict):
        emit(result)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "linux":
        try:
            if (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_model_relpath(rel: str) -> str:
    """Validate a model destination relative to ComfyUI/models/.

    Only a single allowed top-level directory plus a safe relative path is
    accepted; absolute paths and '..' escapes are rejected.
    """
    if not rel or rel.startswith("/") or "\\" in rel:
        raise ValueError(f"unsafe model path: {rel!r}")
    parts = [p for p in rel.split("/") if p]
    if any(p in ("..", ".") or p.startswith("~") for p in parts) or len(parts) < 2:
        raise ValueError(f"unsafe model path: {rel!r}")
    if parts[0] not in ALLOWED_MODEL_DIRS:
        raise ValueError(f"model path must start with an allowed directory, got {parts[0]!r}")
    return "/".join(parts)


def validate_plugin_relpath(rel: str) -> str:
    if not rel.startswith("custom_nodes/") or ".." in rel or rel.startswith("/"):
        raise ValueError(f"unsafe plugin path: {rel!r}")
    name = rel.split("/", 1)[1]
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"unsafe plugin path: {rel!r}")
    return rel


# ---------------------------------------------------------------------------
# Lock ownership check used by every modifying tool
# ---------------------------------------------------------------------------


def read_owner():
    return read_json(OWNER_FILE, default=None)


def probe_lock_held() -> bool:
    """True when some process currently holds the flock (non-destructive probe)."""
    import fcntl  # Linux-only, import here so the module stays importable for tests

    ensure_dirs()
    fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def ensure_owns(session_id: str) -> dict:
    """Verify this session owns the live remote lock; return the owner record."""
    if not probe_lock_held():
        fail(
            "E_LOCK_NOT_HELD",
            "The remote exclusive lock is not held by any session; no modifying action was performed.",
            exit_code=4,
            session_id=session_id,
            hint="Acquire the lock first (controller: `vast-comfy session acquire`).",
        )
    owner = read_owner() or {}
    if owner.get("state") != "active" or owner.get("session_id") != session_id:
        fail(
            "E_LOCK_NOT_HELD",
            "This session does not own the remote exclusive lock; no modifying action was performed.",
            exit_code=4,
            session_id=session_id,
            owner=owner,
        )
    return owner


def operation_held() -> bool:
    """Probe the separate per-operation kernel lock, without changing its inode."""
    import fcntl
    ensure_dirs()
    fd = os.open(OPERATION_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def child_process_options() -> dict:
    """Carry the actual operation lock through our managed child processes."""
    raw = os.environ.get("VAST_COMFY_OPERATION_FD")
    if not raw:
        return {}
    fd = int(raw)
    os.fstat(fd)  # Fail closed if the inherited descriptor was lost.
    return {"pass_fds": (fd,)}


class OperationLease:
    def __init__(self, fd, token):
        self.fd, self.token, self.handed_off = fd, token, False

    def handoff(self):
        self.handed_off = True


def finish_operation(token: str) -> None:
    owner = read_json(OPERATION_OWNER, {}) or {}
    if owner.get("token") == token:
        owner.update(state="released", ended_at=now_epoch())
        atomic_write_json(OPERATION_OWNER, owner)


def check_deadline() -> None:
    deadline = os.environ.get("VAST_COMFY_JOB_DEADLINE")
    if deadline and now_epoch() >= float(deadline):
        fail("E_JOB_TIMEOUT", "The job deadline was exceeded; no next stage was started.",
             job_may_still_be_running=False, takeover_required=True)


@contextlib.contextmanager
def operation(session_id: str, *, label: str = "operation", token: str | None = None):
    """Nonblocking operation lock; delegated work must name the live lease.

    The session daemon and this lock are separate: a second operation in the
    same session fails too. Jobs inherit the descriptor so controller disconnect
    or session-daemon death cannot remove their actual exclusion.
    """
    import fcntl
    ensure_dirs()
    delegated = os.environ.get("VAST_COMFY_OPERATION_TOKEN")
    if delegated:
        record = read_json(OPERATION_OWNER, {}) or {}
        if (record.get("token") != delegated or record.get("session_id") != session_id
                or record.get("state") != "active" or not operation_held()):
            fail("E_LOCK_NOT_HELD", "The delegated operation lease is no longer valid.", exit_code=4)
        ensure_owns(session_id)
        raw_fd = os.environ.get("VAST_COMFY_OPERATION_FD")
        if raw_fd:
            try:
                actual, expected = os.fstat(int(raw_fd)), os.stat(OPERATION_FILE)
            except OSError:
                fail("E_LOCK_NOT_HELD", "The inherited operation descriptor is gone.", exit_code=4)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                fail("E_LOCK_NOT_HELD", "Inherited operation descriptor does not match the lock.", exit_code=4)
        yield OperationLease(None, delegated)
        return
    fd = os.open(OPERATION_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    lease = OperationLease(fd, token or uuid.uuid4().hex)
    acquired = False
    previous = {k: os.environ.get(k) for k in
                ("VAST_COMFY_OPERATION_TOKEN", "VAST_COMFY_OPERATION_FD", "VAST_COMFY_SESSION_ID")}
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            fail("E_LOCK_CONFLICT", "Another operation is running, including in this session.",
                 exit_code=4, operation=read_json(OPERATION_OWNER, {}))
        ensure_owns(session_id)
        alive = session_jobs(session_id, alive_only=True)
        if alive:
            fail("E_LOCK_JOBS_RUNNING", "Existing remote work is still alive.", exit_code=4, jobs=alive)
        atomic_write_json(OPERATION_OWNER, {"state": "active", "token": lease.token,
                          "session_id": session_id, "label": label,
                          "pid": os.getpid(), "started_at": now_epoch()})
        os.environ.update(VAST_COMFY_OPERATION_TOKEN=lease.token,
                          VAST_COMFY_OPERATION_FD=str(fd), VAST_COMFY_SESSION_ID=session_id)
        yield lease
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if acquired and not lease.handed_off:
            finish_operation(lease.token)
        os.close(fd)  # Do not LOCK_UN: a handed-off descriptor belongs to the job.


def session_jobs(session_id: str, *, alive_only: bool = False) -> list:
    """Job records belonging to a session; optionally only those still alive."""
    records = []
    if not JOBS_DIR.is_dir():
        return records
    for job_dir in sorted(JOBS_DIR.iterdir()):
        meta = read_json(job_dir / "meta.json", default=None)
        if not isinstance(meta, dict) or meta.get("session_id") != session_id:
            continue
        status = read_json(job_dir / "status.json", default=None) or {}
        state = status.get("state", "running")
        pid = status.get("pid") or meta.get("pid")
        record = {
            "job_id": meta.get("job_id"),
            "name": meta.get("name"),
            "state": state,
            "pid": pid,
            "pid_alive": pid_alive(pid) if pid else False,
        }
        record["runner_alive"] = pid_alive(meta.get("pid"))
        if alive_only and not (record["pid_alive"] or record["runner_alive"]):
            continue
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Subcommands shared by every remote module
# ---------------------------------------------------------------------------


def cmd_verify_package(args) -> None:
    """Verify the deployed package against its sha256 manifest."""
    directory = Path(args.directory)
    problems: list[str] = []
    checked = 0
    manifest = Path(args.manifest)
    if not manifest.is_file():
        fail("E_DEPLOY", f"manifest not found: {manifest}", exit_code=1)
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sha_expected, _, rel = line.partition("  ")
        if not sha_expected or not rel:
            problems.append(f"malformed manifest line: {line!r}")
            continue
        target = directory / rel
        if not target.is_file():
            problems.append(f"missing file: {rel}")
            continue
        actual = sha256_file(target)
        if actual != sha_expected:
            problems.append(f"sha256 mismatch: {rel}")
        checked += 1
    emit({"ok": not problems, "checked": checked, "problems": problems})


def cmd_instance_time(args) -> None:
    emit({"ok": True, "epoch": now_epoch(), "iso_utc": iso_utc()})


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="vast-comfy remote common helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-package")
    p.add_argument("--directory", required=True)
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_verify_package)

    p = sub.add_parser("instance-time")
    p.set_defaults(func=cmd_instance_time)

    args = parser.parse_args(argv)
    run_cli(args.func, args)


if __name__ == "__main__":
    main()
