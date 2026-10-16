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
import math
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
    try:
        return C.confined_path(C.JOBS_DIR, C.validate_identifier(job_id, "job id"))
    except ValueError as exc:
        C.fail("E_USAGE", str(exc))


def cmd_start(args) -> None:
    # An SSH-delegated token does not carry the operation daemon's FD. Even
    # inherited nested starts have ambiguous finalizer ownership: refuse both.
    if os.environ.get("VAST_COMFY_OPERATION_TOKEN"):
        C.fail("E_LOCK_CONFLICT", "Background jobs must acquire their own lease; nested job start is refused.",
               exit_code=4)
    with C.operation(args.session_id, label=args.name) as lease:
        _start_job(args, lease)


def _start_job(args, lease) -> None:
    C.ensure_dirs()
    owner = C.ensure_owns(args.session_id)
    if not args.argv:
        C.fail("E_USAGE", "job start requires the command after '--'")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        C.fail("E_USAGE", "job timeout must be finite and positive")
    if lease.fd is None:
        C.fail("E_LOCK_NOT_HELD", "Job start requires an actual operation descriptor.", exit_code=4)
    env = dict(os.environ)
    for item in args.env or []:
        key, separator, value = item.partition("=")
        if not separator or not key or key.startswith("VAST_COMFY_"):
            C.fail("E_USAGE", "Invalid or reserved job environment variable")
        env[key] = value
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

    env["VAST_COMFY_JOB_DEADLINE"] = str(C.now_epoch() + args.timeout_seconds)
    env["VAST_COMFY_JOB_ID"] = job_id

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
    # The runner is the sole status writer after spawn. A parent read/replace
    # here could overwrite a child's already-committed terminal status.
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
    job_dir = _job_dir(Path(args.job_dir).name)
    if job_dir.resolve() != Path(args.job_dir).resolve():
        C.fail("E_USAGE", "job directory escapes the managed jobs directory")
    os.environ["VAST_COMFY_JOB_ID"] = job_dir.name
    status = {
        "state": "running",
        "started_at": C.now_epoch(),
        "pid": None,
        "exit_code": None,
        "timeout_seconds": args.timeout,
        "runner_pid": os.getpid(),
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
                rc = 127
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
            # A crashed smoke watcher is not proof that its HTTP prompt ended.
            # Keep the inherited FD as well as the durable fence; never retry
            # submission, cancel inference, or turn this into terminal success.
            while C.pending_inferences(job_id=job_dir.name):
                state = "timeout"
                status.update(state="overdue", child_exit_code=rc, job_may_still_be_running=True)
                C.atomic_write_json(job_dir / "status.json", status)
                time.sleep(1.0)
            log_fh.write(f"\n[vast-comfy] job finished: state={state} exit_code={rc}\n".encode())
    finally:
        pending = C.pending_inferences(job_id=job_dir.name)
        status.update({"state": "overdue" if pending else state, "exit_code": rc,
                       "job_may_still_be_running": bool(pending)})
        if not pending:
            status["ended_at"] = C.now_epoch()
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


def _effective_state(status: dict, *, meta=None, progress=None, pending=None) -> str:
    state = str(status.get("state", "running"))
    runner_pid = status.get("runner_pid") or (meta or {}).get("pid")
    if pending and (state not in ("running", "overdue") or not C.pid_alive(runner_pid)):
        return "overdue"
    if state in ("running", "overdue") and (progress or {}).get("state") == "overdue":
        return "overdue"
    if state in ("running", "overdue"):
        pid = runner_pid or status.get("pid")
        if pid and not C.pid_alive(pid):
            return "crashed"
    return state


def _job_state(job_dir, meta, status):
    pending = C.pending_inferences(job_id=job_dir.name)
    progress = C.read_json(job_dir / "progress.json", {}) or {}
    if (pending and progress.get("state") != "overdue"
            and isinstance(progress.get("deadline_at"), (int, float))
            and C.now_epoch() >= progress["deadline_at"]):
        # Report overdue even while the watcher is blocked in an HTTP request.
        progress = {**progress, "state": "overdue", "reason": "SMOKE_TIMEOUT"}
    state = _effective_state(status, meta=meta, progress=progress, pending=pending)
    return state, pending, progress


def cmd_status(args) -> None:
    job_dir, meta, status = _load_job(args.job_id)
    state, pending, progress = _job_state(job_dir, meta, status)
    kernel_held = C.operation_held()
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
            "runner_pid": status.get("runner_pid") or meta.get("pid"),
            "runner_alive": C.pid_alive(status.get("runner_pid") or meta.get("pid")),
            "operation_kernel_held": kernel_held,
            "persistent_hold": bool(pending),
            "pending_inferences": pending,
            "progress": progress,
            "job_may_still_be_running": bool(pending) or state in ("running", "overdue"),
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
            "state": _job_state(job_dir, meta, status)[0],
            "tail_lines": args.tail,
            "log": C.redact(text),
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
                "state": _job_state(job_dir, meta, status)[0],
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
