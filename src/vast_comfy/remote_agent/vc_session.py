#!/usr/bin/env python3
"""vast-comfy remote session lock: acquire / status / release / takeover.

The mutual exclusion is guaranteed by a real ``flock`` held for the entire
lifetime of a small daemon process. The owner record (JSON) carries the session
id, timestamps taken from the *instance* clock, heartbeat settings and the
daemon pid; it is used for diagnostics and stale-detection, never as the
locking primitive itself.

Rules enforced here (SPEC.md section 8):

* Acquire is atomic (the daemon's flock acquisition is the arbiter).
* Conflict => immediate failure with owner info; no queueing, no actions taken.
* Stale owner records are reported; taking over requires the explicit
  ``--accept-stale`` flag, and is refused while the stale session still has
  alive jobs. The lock file is never deleted or recreated.
* Release refuses while this session still has alive jobs.
* Heartbeats update automatically while the daemon lives.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402


def _probe():
    return C.probe_lock_held()


def _owner():
    return C.read_owner()


def _stale_report(owner) -> dict:
    if not owner:
        return {
            "owner": None,
            "heartbeat_age_seconds": None,
            "expired": None,
            "jobs": [],
        }
    hb = owner.get("heartbeat_at")
    age = (C.now_epoch() - float(hb)) if isinstance(hb, (int, float)) else None
    ttl = float(owner.get("ttl_seconds", 300))
    buffer = float(owner.get("buffer_seconds", 60))
    return {
        "owner": owner,
        "heartbeat_age_seconds": round(age, 1) if age is not None else None,
        "expired": (age is not None and age > ttl + buffer),
        "jobs": C.session_jobs(str(owner.get("session_id", ""))),
    }


def _archive_owner(owner) -> None:
    if owner:
        C.append_jsonl(C.OWNER_HISTORY, {"archived_at": C.now_epoch(), "owner": owner})


def cmd_acquire(args) -> None:
    C.ensure_dirs()
    session_id = args.session_id
    held = _probe()
    owner = _owner()

    if held:
        if owner and owner.get("state") == "active" and owner.get("session_id") == session_id:
            C.emit({"ok": True, "status": "held_by_me", "owner": owner})
        C.fail(
            "E_LOCK_CONFLICT",
            "Another session holds the remote exclusive lock; nothing was done.",
            exit_code=4,
            status="conflict",
            **_stale_report(owner),
        )

    # Lock is free: is there a stale foreign owner record?
    stale_foreign = bool(owner and owner.get("state") == "active" and owner.get("session_id") != session_id)
    if stale_foreign:
        report = _stale_report(owner)
        alive_jobs = [j for j in report["jobs"] if j.get("state") == "running" and j.get("pid_alive")]
        if alive_jobs:
            C.fail(
                "E_LOCK_JOBS_RUNNING",
                "The previous owner's daemon is gone but its jobs are still alive; refusing takeover.",
                exit_code=4,
                status="stale_jobs_running",
                jobs=alive_jobs,
                hint="Stop the old remote work first (user decision). The lock is never taken over automatically.",
            )
        if not args.accept_stale:
            C.fail(
                "E_LOCK_STALE",
                "A stale owner record from another session exists (its daemon is gone). "
                "Nothing was done; takeover requires an explicit user decision.",
                exit_code=4,
                status="stale",
                **report,
            )
        _archive_owner(owner)

    # Spawn the daemon; it performs the atomic flock acquisition.
    daemon_log = C.LOCK_DIR / "daemon.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "daemon",
        "--session-id",
        session_id,
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--ttl-seconds",
        str(args.ttl_seconds),
        "--buffer-seconds",
        str(args.buffer_seconds),
        "--instance-id",
        str(args.instance_id),
        "--controller",
        args.controller,
    ]
    if args.accept_stale:
        cmd.append("--takeover")
    with open(daemon_log, "ab") as log_fh:
        log_fh.write(f"\n--- {C.iso_utc()} acquire -> daemon ---\n".encode())
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(C.LOCK_DIR),
        )

    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        time.sleep(0.25)
        if proc.poll() is not None and proc.returncode != 0:
            break
        current = _owner()
        if (
            _probe()
            and current
            and current.get("state") == "active"
            and current.get("session_id") == session_id
            and C.pid_alive(current.get("daemon_pid"))
        ):
            C.emit(
                {
                    "ok": True,
                    "status": "acquired",
                    "owner": current,
                    "takeover": bool(args.accept_stale),
                }
            )

    # Daemon failed or lost the race.
    current = _owner()
    if _probe() and current and current.get("session_id") != session_id:
        C.fail(
            "E_LOCK_CONFLICT",
            "Another session acquired the lock first; nothing was done.",
            exit_code=4,
            status="conflict",
            **_stale_report(current),
        )
    C.fail(
        "E_INTERNAL",
        "The session daemon did not start correctly. Read lock/daemon.log on the instance.",
        exit_code=1,
        status="daemon_failed",
        daemon_log=str(daemon_log),
    )


def cmd_daemon(args) -> None:
    import fcntl

    C.ensure_dirs()
    fd = os.open(C.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        sys.exit(3)

    owner = _owner()
    if (
        owner
        and owner.get("state") == "active"
        and owner.get("session_id") != args.session_id
        and not args.takeover
    ):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        sys.exit(4)

    record = {
        "state": "active",
        "session_id": args.session_id,
        "instance_id": args.instance_id,
        "controller": json.loads(args.controller) if args.controller else {},
        "acquired_at": C.now_epoch(),
        "heartbeat_at": C.now_epoch(),
        "heartbeat_seconds": args.heartbeat_seconds,
        "ttl_seconds": args.ttl_seconds,
        "buffer_seconds": args.buffer_seconds,
        "daemon_pid": os.getpid(),
        "takeover": bool(args.takeover),
    }
    C.atomic_write_json(C.OWNER_FILE, record)

    last_heartbeat = time.time()
    try:
        while True:
            time.sleep(1)
            if C.RELEASE_REQUEST.exists():
                request = C.read_json(C.RELEASE_REQUEST, default={}) or {}
                if request.get("session_id") == args.session_id:
                    # Consume the request exactly once: a refused release must be
                    # retried explicitly by the controller, never acted on later.
                    C.RELEASE_REQUEST.unlink(missing_ok=True)
                    blocked_file = C.LOCK_DIR / "release.blocked"
                    blocked_file.unlink(missing_ok=True)
                    alive = [
                        j
                        for j in C.session_jobs(args.session_id)
                        if j.get("state") == "running" and j.get("pid_alive")
                    ]
                    if alive:
                        # Refuse to release: work is still running and must not be
                        # left behind unsynchronized. Report and keep holding.
                        C.atomic_write_json(
                            blocked_file,
                            {"at": C.now_epoch(), "jobs": alive},
                        )
                        continue
                    record = _owner() or record
                    record.update(
                        {
                            "state": "released",
                            "released_at": C.now_epoch(),
                            "released_reason": request.get("reason", "explicit release"),
                        }
                    )
                    _archive_owner(record)
                    C.atomic_write_json(C.OWNER_FILE, record)
                    break
            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                record = _owner() or record
                record["heartbeat_at"] = C.now_epoch()
                C.atomic_write_json(C.OWNER_FILE, record)
                last_heartbeat = time.time()
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def cmd_status(args) -> None:
    C.ensure_dirs()
    owner = _owner()
    report = _stale_report(owner)
    daemon_alive = bool(owner and C.pid_alive(owner.get("daemon_pid")))
    held = _probe()
    payload = {
        "ok": True,
        "lock_held": held,
        "daemon_alive": daemon_alive,
        "matches_me": bool(
            args.session_id
            and held
            and owner
            and owner.get("session_id") == args.session_id
            and owner.get("state") == "active"
        ),
        **report,
    }
    if report["expired"]:
        payload["note"] = (
            "Heartbeat appears expired. This is REPORTED ONLY: no takeover, no unlock, "
            "no work is stopped. A user decision is required."
        )
    C.emit(payload)


def cmd_release(args) -> None:
    C.ensure_dirs()
    owner = _owner()
    if not owner or owner.get("session_id") != args.session_id or owner.get("state") != "active":
        C.fail(
            "E_LOCK_NOT_HELD",
            "This session does not own the remote lock; nothing was released.",
            exit_code=4,
            owner=owner,
        )
    alive = [
        j for j in C.session_jobs(args.session_id) if j.get("state") == "running" and j.get("pid_alive")
    ]
    if alive:
        C.fail(
            "E_LOCK_JOBS_RUNNING",
            "Refusing to release while this session still has running jobs.",
            exit_code=4,
            jobs=alive,
            hint="Wait for the jobs to finish (or stop them yourself); release never abandons running work.",
        )
    # Clear any stale "blocked" marker from a previous attempt before sending the request.
    (C.LOCK_DIR / "release.blocked").unlink(missing_ok=True)
    C.atomic_write_json(
        C.RELEASE_REQUEST, {"session_id": args.session_id, "at": C.now_epoch(), "reason": args.reason}
    )
    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        time.sleep(0.25)
        current = _owner() or {}
        blocked = C.read_json(C.LOCK_DIR / "release.blocked", default=None)
        if blocked:
            alive2 = blocked.get("jobs", [])
            C.fail(
                "E_LOCK_JOBS_RUNNING",
                "Release was blocked because jobs are still alive.",
                exit_code=4,
                jobs=alive2,
            )
        if current.get("state") == "released" and current.get("session_id") == args.session_id:
            C.emit(
                {
                    "ok": True,
                    "status": "released",
                    "released_at": current.get("released_at"),
                    "note": "Releasing the session does not stop the running ComfyUI service.",
                }
            )
    C.fail(
        "E_INTERNAL",
        "The daemon did not release within the wait window; nothing else was done.",
        exit_code=1,
        hint="Inspect lock/daemon.log on the instance; do not delete lock files.",
    )


def cmd_list_jobs(args) -> None:
    C.emit({"ok": True, "jobs": C.session_jobs(args.session_id) if args.session_id else []})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote session lock")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("acquire")
    p.add_argument("--session-id", required=True)
    p.add_argument("--instance-id", default="")
    p.add_argument("--controller", default="")
    p.add_argument("--heartbeat-seconds", type=float, default=30.0)
    p.add_argument("--ttl-seconds", type=float, default=300.0)
    p.add_argument("--buffer-seconds", type=float, default=60.0)
    p.add_argument("--accept-stale", action="store_true")
    p.add_argument("--wait-seconds", type=float, default=10.0)
    p.set_defaults(func=cmd_acquire)

    p = sub.add_parser("daemon")
    p.add_argument("--session-id", required=True)
    p.add_argument("--instance-id", default="")
    p.add_argument("--controller", default="")
    p.add_argument("--heartbeat-seconds", type=float, default=30.0)
    p.add_argument("--ttl-seconds", type=float, default=300.0)
    p.add_argument("--buffer-seconds", type=float, default=60.0)
    p.add_argument("--takeover", action="store_true")
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("status")
    p.add_argument("--session-id", default="")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("release")
    p.add_argument("--session-id", required=True)
    p.add_argument("--reason", default="explicit release")
    p.add_argument("--wait-seconds", type=float, default=15.0)
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("list-jobs")
    p.add_argument("--session-id", default="")
    p.set_defaults(func=cmd_list_jobs)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    C.run_cli(args.func, args)


if __name__ == "__main__":
    main()
