"""Actual flock/FD inference lifetime tests. Synthetic HTTP only; no sockets/GPU."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from test_linux_session import SessionEnv
from test_audit_linux_operations import jobs, wait_terminal

pytestmark = [pytest.mark.linux_only,
              pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux flock")]
AGENT = Path(__file__).resolve().parents[1] / "src/vast_comfy/remote_agent"


@pytest.fixture
def inference_env(tmp_path):
    value = SessionEnv(tmp_path)
    yield value
    value.teardown()


def start_watcher(env, tmp_path):
    terminal = tmp_path / "prompt-terminal"
    crash = tmp_path / "watcher-crash"
    submitted = tmp_path / "submitted.json"
    # This child is the smoke watcher under the real job runner. HTTP is a
    # deterministic in-process fake, including a simulated abrupt watcher crash.
    script = f'''
import sys, time, os, json
from pathlib import Path
sys.path.insert(0, {str(AGENT)!r})
import vc_common as C
import vc_smoke as S
real_sleep = time.sleep
S.time.sleep = lambda seconds: real_sleep(min(seconds, .03))
def http(url, **kwargs):
    if kwargs.get("method") == "POST":
        assert C.pending_inferences()[0]["state"] == "submitting"
        Path({str(submitted)!r}).write_text(json.dumps({{"pid": os.getpid()}}))
        return {{"prompt_id": "synthetic-prompt"}}
    if Path({str(crash)!r}).exists():
        os._exit(99)
    assert "/history/" in url
    if Path({str(terminal)!r}).exists():
        return {{"synthetic-prompt": {{"status": {{"completed": True, "status_str": "success"}}}}}}
    return {{}}
S.CM.http_json = http
with C.operation("s1", label="synthetic-smoke"):
    S.submit_and_watch("http://not-used.invalid", {{}}, .15,
                       session_id="s1", run_id="synthetic", case_id="base_t2i")
'''
    rc, payload = jobs(env, "start", "--session-id", "s1", "--name", "synthetic-inference",
                       "--timeout-seconds", "60", "--", sys.executable, "-B", "-c", script)
    assert rc == 0, payload
    env.spawned_pids.append(payload["pid"])
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        _, status = jobs(env, "status", "--job-id", payload["job_id"])
        if status["state"] == "overdue":
            return payload, status, terminal, crash, submitted
        time.sleep(.03)
    pytest.fail("case-level overdue was not exposed promptly")


def test_case_deadline_retains_real_fd_until_http_terminal_and_keeps_session(inference_env, tmp_path):
    env = inference_env
    assert env.acquire("s1")[0] == 0
    first, status, terminal, _, submitted = start_watcher(env, tmp_path)
    assert status["operation_kernel_held"] and status["persistent_hold"]
    assert status["runner_alive"] and status["pid_alive"]
    watcher_pid = json.loads(submitted.read_text())["pid"]
    env.spawned_pids.append(watcher_pid)
    for command in (("release", "--session-id", "s1"),
                    ("begin-operation", "--session-id", "s1", "--operation-token", "a" * 32)):
        assert env.run_json(*command)[0] == 4
    terminal.touch()  # Positive completion evidence, not a cancellation.
    final = wait_terminal(env, first["job_id"])
    assert final["state"] == "failed"  # late success cannot erase the timeout
    _, session, _ = env.run_json("status", "--session-id", "s1")
    assert session["lock_held"] and session["matches_me"]
    assert not session["persistent_hold"]
    # Runner may be completing its last few instructions after its status write.
    deadline = time.monotonic() + 3
    while session["operation_held"] and time.monotonic() < deadline:
        time.sleep(.03)
        _, session, _ = env.run_json("status", "--session-id", "s1")
    assert not session["operation_held"]
    assert env.run_json("release", "--session-id", "s1")[0] == 0


def test_crashed_watcher_and_runner_leave_persistent_fence_without_fake_kernel_claim(inference_env, tmp_path):
    env = inference_env
    assert env.acquire("s1")[0] == 0
    first, _, _, crash, submitted = start_watcher(env, tmp_path)
    env.spawned_pids.append(json.loads(submitted.read_text())["pid"])
    crash.touch()  # Crash only the synthetic mock watcher, never real work.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, status = jobs(env, "status", "--job-id", first["job_id"])
        if not status["pid_alive"]:
            break
        time.sleep(.03)
    assert not status["pid_alive"] and status["runner_alive"]
    assert status["state"] == "overdue" and status["operation_kernel_held"]
    # Emulate loss of all management processes. HTTP work has no such lifetime
    # relationship, so a dead PID must not permit release or stale recovery.
    os.kill(first["pid"], signal.SIGKILL)
    os.kill(env.owner()["daemon_pid"], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, status, _ = env.run_json("status", "--session-id", "s1")
        if not status["operation_kernel_held"] and not status["lock_held"]:
            break
        time.sleep(.03)
    assert not status["operation_kernel_held"] and not status["lock_held"]
    assert status["persistent_hold"] and status["pending_inferences"]
    assert env.acquire("s2", "--accept-stale")[0] == 4
    assert env.run_json("release", "--session-id", "s1")[0] == 4
    rc, blocked = jobs(env, "start", "--session-id", "s1", "--name", "blocked",
                       "--timeout-seconds", "1", "--", sys.executable, "-c", "pass")
    assert rc == 4 and blocked["persistent_hold"]


def test_token_only_delegation_cannot_start_a_background_job(inference_env):
    env = inference_env
    assert env.acquire("s1")[0] == 0
    token = "b" * 32
    assert env.run_json("begin-operation", "--session-id", "s1", "--operation-token", token)[0] == 0
    env.env["VAST_COMFY_OPERATION_TOKEN"] = token
    try:
        rc, blocked = jobs(env, "start", "--session-id", "s1", "--name", "nested",
                           "--timeout-seconds", "1", "--", sys.executable, "-c", "pass")
        assert rc == 4 and blocked["code"] == "E_LOCK_CONFLICT"
        assert not list((env.runtime / "jobs").iterdir())
    finally:
        env.env.pop("VAST_COMFY_OPERATION_TOKEN")
        assert env.run_json("end-operation", "--session-id", "s1", "--operation-token", token)[0] == 0
    assert env.run_json("release", "--session-id", "s1")[0] == 0
