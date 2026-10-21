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
from urllib.parse import unquote, urlsplit

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
INFERENCE_DIR = LOCK_DIR / "inference"
JOBS_DIR = RUNTIME_DIR / "jobs"
RUNS_DIR = RUNTIME_DIR / "runs"
DOWNLOADS_DIR = RUNTIME_DIR / "downloads"
ARTIFACTS_DIR = RUNTIME_DIR / "artifacts"

ALLOWED_MODEL_DIRS = {
    "checkpoints",
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
    text = re.sub(r"(?i)(https?://)[^\s/?#]*(?:@|%(?:25){0,2}40)", r"\1<redacted>@", text)

    def query_secret(match):
        key = match.group(2)
        for _ in range(3):
            key = unquote(key)
        key = key.lower().replace("-", "_")
        if any(word in key for word in ("token", "api_key", "apikey", "password", "secret")):
            return match.group(1) + match.group(2) + match.group(3) + "<redacted>"
        return match.group(0)

    text = re.sub(r"(?i)([?&]|%(?:25){0,2}3f|%(?:25){0,2}26)([^=&#\s]+?)(=|%(?:25){0,2}3d)"
                  r"((?:(?!%(?:25){0,2}(?:26|23))[^&#\s])*)",
                  query_secret, text)
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
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):  # includes JSONDecodeError and UnicodeDecodeError
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


def validate_identifier(value, label: str = "identifier") -> str:
    """Portable, single-component identifiers; never normalize an unsafe value."""
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            or value.endswith(".") or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])",
                                                   value.split(".", 1)[0])):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def confined_path(root: Path, rel: str) -> Path:
    if (not isinstance(rel, str) or not rel or any(c in rel for c in ("\\", ":", "\0"))
            or any(part in ("", ".", "..") for part in rel.split("/"))
            or any(ord(c) < 32 or ord(c) == 127 for c in rel)):
        raise ValueError(f"unsafe relative path: {rel!r}")
    root = Path(root)
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes its directory: {rel!r}")
    return path


def validate_public_url(url) -> str:
    if (not isinstance(url, str) or not url or any(ord(c) <= 32 or ord(c) >= 127 for c in url)
            or any(c in url for c in ("%", "\\", "?", "#"))):
        raise ValueError("URL must be credential-free, unencoded HTTPS without query or fragment")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or "@" in parsed.netloc or parsed.port not in (None, 443)
            or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", parsed.hostname)
            or any(not label or label.startswith("-") or label.endswith("-")
                   for label in parsed.hostname.split("."))):
        raise ValueError("URL must be credential-free HTTPS on the default port")
    if parsed.path not in ("", "/") and any(
        part in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.~-]+", part)
        for part in parsed.path.removeprefix("/").split("/")
    ):
        raise ValueError("URL path must contain safe ASCII components")
    return url


def reserve_run_directory(run_id: str, *, stage: str, from_prepare: bool = False) -> Path:
    """Claim a run once, including failures; never overwrite evidence from a retry."""
    try:
        validate_identifier(run_id, "run id")
        path = confined_path(RUNS_DIR, run_id)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        if from_prepare:
            claim = read_json(confined_path(path, ".prepare.claim"), {})
            if not isinstance(claim, dict) or claim.get("token") != os.environ.get("VAST_COMFY_OPERATION_TOKEN"):
                raise ValueError("run was not reserved by this prepare operation")
        else:
            path.mkdir(exist_ok=False)
        marker = confined_path(path, f".{stage}.claim")
        with marker.open("x", encoding="utf-8") as fh:
            json.dump({"token": os.environ.get("VAST_COMFY_OPERATION_TOKEN"), "created_at": now_epoch()}, fh)
    except (ValueError, FileExistsError) as exc:
        fail("E_USAGE", f"Invalid or already used run id: {exc}")
    return path


def inference_resolved(record) -> bool:
    """True only when the record carries positive completion evidence.

    A terminal/rejected state without evidence is UNKNOWN and keeps the fence;
    only the watcher or the operator's explicit resolution may end it.
    """
    if not isinstance(record, dict) or record.get("state") not in ("terminal", "rejected"):
        return False
    if record.get("state") == "rejected" and record.get("rejection_confirmed") is True:
        return True
    status = record.get("terminal_status")
    if not isinstance(status, dict) or not isinstance(record.get("prompt_id"), str):
        return False
    messages = status.get("messages")
    execution_error = isinstance(messages, list) and any(
        isinstance(item, (list, tuple)) and len(item) == 2
        and item[0] in ("execution_error", "execution_interrupted") for item in messages
    )
    return (status.get("status_str") in ("error", "failed", "interrupted")
            or (status.get("completed") is True and status.get("status_str") == "success")
            or execution_error)


def pending_inferences(*, session_id=None, operation_token=None, job_id=None) -> list[dict]:
    """Persistent safety fence, independent of kernel locks and watcher PIDs.

    Unreadable/malformed records are UNKNOWN and block every session. Only the
    watcher may record terminal/rejected after obtaining positive evidence.
    """
    if not INFERENCE_DIR.exists():
        return []
    records = []
    for path in sorted(INFERENCE_DIR.glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("state"), str):
            records.append({"intent_id": path.stem, "state": "unknown", "record_unreadable": True})
            continue
        if record["state"] in ("terminal", "rejected"):
            if inference_resolved(record):
                continue
            records.append({"intent_id": path.stem, "state": "unknown", "record_unreadable": True})
            continue
        if any(value is not None and record.get(key) != value for key, value in (
            ("session_id", session_id), ("operation_token", operation_token), ("job_id", job_id)
        )):
            continue
        records.append(record)
    return records


def assert_no_pending_inferences() -> None:
    pending = pending_inferences()
    if pending:
        fail("E_LOCK_JOBS_RUNNING", "Unresolved HTTP inference remains; no unlock or takeover is safe.",
             exit_code=4, pending_inferences=pending, persistent_hold=True, takeover_required=True)


def write_inference(record: dict) -> None:
    intent_id = validate_identifier(record["intent_id"], "inference intent id")
    atomic_write_json(confined_path(INFERENCE_DIR, intent_id + ".json"), record)


def write_job_progress(payload: dict) -> None:
    job_id = os.environ.get("VAST_COMFY_JOB_ID")
    if job_id:
        directory = confined_path(JOBS_DIR, validate_identifier(job_id, "job id"))
        atomic_write_json(confined_path(directory, "progress.json"), {**payload, "updated_at": now_epoch()})


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
        if self.fd is None:
            fail("E_LOCK_NOT_HELD", "A token-only lease cannot hand off a kernel lock to a job.", exit_code=4)
        self.handed_off = True


def finish_operation(token: str) -> bool:
    owner = read_json(OPERATION_OWNER, {}) or {}
    if owner.get("token") == token:
        if pending_inferences():
            owner.update(persistent_hold=True)
            atomic_write_json(OPERATION_OWNER, owner)
            return False
        owner.update(state="released", ended_at=now_epoch())
        owner["persistent_hold"] = False
        atomic_write_json(OPERATION_OWNER, owner)
    return True


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
    ensure_dirs()
    assert_no_pending_inferences()
    import fcntl
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
        yield OperationLease(int(raw_fd) if raw_fd else None, delegated)
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
        assert_no_pending_inferences()
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
        try:
            if acquired and not lease.handed_off:
                finish_operation(lease.token)
        finally:
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
        record["persistent_hold"] = bool(pending_inferences(job_id=meta.get("job_id")))
        # A trusted terminal state wins over stored PIDs: those numbers may have
        # been reused by unrelated processes after the job ended.
        terminal = state in ("succeeded", "failed", "timeout") and not record["persistent_hold"]
        if alive_only and (terminal or not (record["pid_alive"]
                                            or record["runner_alive"] or record["persistent_hold"])):
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
    entries = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()]
    entries = [line for line in entries if line]
    if not entries:
        problems.append("manifest contains no entries")
    seen: set[str] = set()
    for line in entries:
        sha_expected, _, rel = line.partition("  ")
        if not sha_expected or not rel:
            problems.append(f"malformed manifest line: {line!r}")
            continue
        if rel in seen:
            problems.append(f"duplicate manifest entry: {rel}")
            continue
        seen.add(rel)
        target = directory / rel
        if not target.is_file():
            problems.append(f"missing file: {rel}")
            continue
        actual = sha256_file(target)
        if actual != sha_expected:
            problems.append(f"sha256 mismatch: {rel}")
        checked += 1
    emit({"ok": not problems, "checked": checked, "problems": problems,
          "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
         exit_code=0 if not problems else 1)


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
