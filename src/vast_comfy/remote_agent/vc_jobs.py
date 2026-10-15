#!/usr/bin/env python3
"""vast-comfy remote background jobs: start / status / logs / list.

A job is an independent process (`start_new_session=True`) so it survives the
controller disconnecting. Jobs remember their owning session; ownership is
re-verified at start time against the live flock, so a session that lost the
lock cannot spawn work. Jobs are never cancelled implicitly; release refuses
while a session's jobs are alive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402


def _new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + os.urandom(4).hex()


def _job_dir(job_id: str) -> Path:
    if "/" in job_id or ".." in job_id:
        C.fail("E_USAGE", f"invalid job id: {job_id!r}")
    return C.JOBS_DIR / job_id


def cmd_start(args) -> None:
    with C.operation(args.session_id, label=args.name) as lease:
        _start_job(args, lease)


def _start_job(args, lease) -> None:
    C.ensure_dirs()
    owner = C.ensure_owns(args.session_id)
    if not args.argv:
        C.fail("E_USAGE", "job start requires the command after '--'")
    job_id = _new_job_id()
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True)
    log_path = job_dir / "log.txt"
    meta = {
        "job_id": job_id,
        "name": args.name,
        "session_id": args.session_id,
        "instance_id": owner.get("instance_id"),
        "argv": list(args.argv),
        "created_at": C.now_epoch(),
        "timeout_seconds": args.timeout_seconds,
        "pid": None,
    }
    C.atomic_write_json(job_dir / "meta.json", meta)
    C.atomic_write_json(
        job_dir / "status.json",
        {"state": "running", "started_at": C.now_epoch(), "pid": None, "exit_code": None},
    )

    env = dict(os.environ)
    env["VAST_COMFY_JOB_DEADLINE"] = str(C.now_epoch() + args.timeout_seconds)
    for item in args.env or []:
        key, _, value = item.partition("=")
        if not key:
            C.fail("E_USAGE", f"invalid --env value: {item!r}")
        env[key] = value

    runner = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run",
        "--job-dir",
        str(job_dir),
        "--timeout",
        str(args.timeout_seconds),
        "--",
        *list(args.argv),
    ]
    with open(log_path, "ab") as log_fh:
        log_fh.write(
            f"=== job {job_id} name={args.name} session={args.session_id} at {C.iso_utc()} ===\n".encode()
        )
        log_fh.write(("$ " + C.redact(" ".join(list(args.argv))) + "\n").encode())
        log_fh.flush()
        proc = subprocess.Popen(
            runner,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=str(job_dir),
            **C.child_process_options(),
        )
    lease.handoff()
    meta["pid"] = proc.pid
    C.atomic_write_json(job_dir / "meta.json", meta)
    status = C.read_json(job_dir / "status.json", default={}) or {}
    status["pid"] = proc.pid
    C.atomic_write_json(job_dir / "status.json", status)
    C.emit(
        {
            "ok": True,
            "job_id": job_id,
            "pid": proc.pid,
            "name": args.name,
            "timeout_seconds": args.timeout_seconds,
        }
    )


def cmd_run(args) -> None:  # internal runner (_run)
    job_dir = Path(args.job_dir)
    status = {
        "state": "running",
        "started_at": C.now_epoch(),
        "pid": None,
        "exit_code": None,
        "timeout_seconds": args.timeout,
    }
    C.atomic_write_json(job_dir / "status.json", status)
    log_path = job_dir / "log.txt"
    state = "failed"
    rc: int | None = None
    try:
        with open(log_path, "ab") as log_fh:
            try:
                proc = subprocess.Popen(
                    list(args.argv),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    **C.child_process_options(),
                )
            except FileNotFoundError as exc:
                log_fh.write(f"\n[vast-comfy] command not found: {exc}\n".encode())
                status.update({"state": "failed", "exit_code": 127, "ended_at": C.now_epoch()})
                C.atomic_write_json(job_dir / "status.json", status)
                sys.exit(1)
            status["pid"] = proc.pid
            C.atomic_write_json(job_dir / "status.json", status)
            try:
                rc = proc.wait(timeout=args.timeout)
                state = "succeeded" if rc == 0 else "failed"
            except subprocess.TimeoutExpired:
                log_fh.write(f"\n[vast-comfy] deadline exceeded after {args.timeout}s; work is NOT cancelled\n".encode())
                log_fh.flush()
                status.update(state="overdue", job_may_still_be_running=True)
                C.atomic_write_json(job_dir / "status.json", status)
                rc = proc.wait()  # Retain the actual lock until work really ends.
                state = "timeout"
            log_fh.write(f"\n[vast-comfy] job finished: state={state} exit_code={rc}\n".encode())
    finally:
        status.update({"state": state, "exit_code": rc, "ended_at": C.now_epoch()})
        C.atomic_write_json(job_dir / "status.json", status)
        token = os.environ.get("VAST_COMFY_OPERATION_TOKEN")
        if token:
            C.finish_operation(token)
    sys.exit(0 if state == "succeeded" else 1)


def _load_job(job_id: str) -> tuple[Path, dict, dict]:
    job_dir = _job_dir(job_id)
    meta = C.read_json(job_dir / "meta.json", default=None)
    if not isinstance(meta, dict):
        C.fail("E_JOB_NOT_FOUND", f"No such job: {job_id}", exit_code=1, job_id=job_id)
    status = C.read_json(job_dir / "status.json", default={}) or {}
    return job_dir, meta, status


def _effective_state(status: dict) -> str:
    state = str(status.get("state", "running"))
    if state in ("running", "overdue"):
        pid = status.get("pid")
        if pid and not C.pid_alive(pid):
            return "crashed"
    return state


def cmd_status(args) -> None:
    job_dir, meta, status = _load_job(args.job_id)
    state = _effective_state(status)
    C.emit(
        {
            "ok": True,
            "job_id": args.job_id,
            "name": meta.get("name"),
            "session_id": meta.get("session_id"),
            "state": state,
            "exit_code": status.get("exit_code"),
            "pid": status.get("pid"),
            "pid_alive": C.pid_alive(status.get("pid")),
            "started_at": status.get("started_at"),
            "ended_at": status.get("ended_at"),
            "timeout_seconds": status.get("timeout_seconds"),
            "job_dir": str(job_dir),
        }
    )


def cmd_logs(args) -> None:
    job_dir, meta, status = _load_job(args.job_id)
    log_path = job_dir / "log.txt"
    text = ""
    if log_path.is_file():
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        text = "".join(lines[-args.tail :])
    C.emit(
        {
            "ok": True,
            "job_id": args.job_id,
            "state": _effective_state(status),
            "tail_lines": args.tail,
            "log": text,
        }
    )


def cmd_list(args) -> None:
    C.ensure_dirs()
    items = []
    for job_dir in sorted(C.JOBS_DIR.iterdir(), reverse=True):
        meta = C.read_json(job_dir / "meta.json", default=None)
        if not isinstance(meta, dict):
            continue
        if args.session_id and meta.get("session_id") != args.session_id:
            continue
        status = C.read_json(job_dir / "status.json", default={}) or {}
        items.append(
            {
                "job_id": meta.get("job_id"),
                "name": meta.get("name"),
                "session_id": meta.get("session_id"),
                "state": _effective_state(status),
                "created_at": meta.get("created_at"),
                "ended_at": status.get("ended_at"),
            }
        )
        if len(items) >= args.limit:
            break
    C.emit({"ok": True, "jobs": items})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote jobs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start")
    p.add_argument("--session-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--timeout-seconds", type=float, required=True)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("_run")
    p.add_argument("--job-dir", required=True)
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status")
    p.add_argument("--job-id", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs")
    p.add_argument("--job-id", required=True)
    p.add_argument("--tail", type=int, default=200)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("list")
    p.add_argument("--session-id", default="")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_list)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "start" and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]
    if args.command == "_run" and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]
    C.run_cli(args.func, args)


if __name__ == "__main__":
    main()
