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

import hashlib
import json
import os
import sys
import time
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


def run_cli(func, args) -> None:
    """Dispatch a subcommand, converting RemoteFail into structured JSON."""
    try:
        result = func(args)
    except RemoteFail as exc:
        payload = {"ok": False, "code": exc.code, "message": exc.message}
        payload.update(exc.details)
        emit(payload, exit_code=exc.exit_code)
    if isinstance(result, dict):
        emit(result)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
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
        if alive_only and not (state == "running" and record["pid_alive"]):
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
    C.run_cli(args.func, args)


if __name__ == "__main__":
    main()
