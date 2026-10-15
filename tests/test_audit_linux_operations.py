"""Real kernel-lock / inherited-descriptor tests; never substitute mocks for flock."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from test_linux_session import SessionEnv, last_json

pytestmark = [pytest.mark.linux_only,
              pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux flock")]
JOBS = Path(__file__).resolve().parents[1] / "src/vast_comfy/remote_agent/vc_jobs.py"


@pytest.fixture
def env(tmp_path):
    value = SessionEnv(tmp_path)
    yield value
    value.teardown()


def jobs(env, *args):
    result = subprocess.run([sys.executable, str(JOBS), *args], env=env.env,
                            capture_output=True, text=True, timeout=15)
    return result.returncode, last_json(result.stdout)


def start(env, *, timeout=15, duration=3):
    rc, payload = jobs(env, "start", "--session-id", "s1", "--name", "audit",
                       "--timeout-seconds", str(timeout), "--", sys.executable, "-c",
                       f"import time; time.sleep({duration})")
    if rc == 0:
        env.spawned_pids.append(payload["pid"])
    return rc, payload


def wait_terminal(env, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        _, status = jobs(env, "status", "--job-id", job_id)
        if status["state"] not in ("running", "overdue"):
            return status
        time.sleep(.1)
    pytest.fail("job did not complete")


def test_same_session_operations_and_release_cannot_overlap(env):
    assert env.acquire("s1")[0] == 0
    token = uuid.uuid4().hex
    rc, _, _ = env.run_json("begin-operation", "--session-id", "s1", "--operation-token", token)
    assert rc == 0
    rc, blocked, _ = env.run_json("begin-operation", "--session-id", "s1", "--operation-token", uuid.uuid4().hex)
    assert rc == 4 and blocked["code"] == "E_LOCK_CONFLICT"
    rc, blocked, _ = env.run_json("release", "--session-id", "s1")
    assert rc == 4
    assert env.run_json("end-operation", "--session-id", "s1", "--operation-token", token)[0] == 0
    assert env.run_json("release", "--session-id", "s1")[0] == 0


def test_job_inherits_lock_and_survives_session_daemon_death(env):
    assert env.acquire("s1")[0] == 0
    rc, first = start(env)
    assert rc == 0, first
    rc, second = start(env)
    assert rc == 4 and second["code"] == "E_LOCK_CONFLICT"
    os.kill(env.owner()["daemon_pid"], signal.SIGTERM)
    time.sleep(.2)
    rc, blocked = env.acquire("s2", "--accept-stale")
    assert rc == 4 and blocked["code"] == "E_LOCK_JOBS_RUNNING"
    assert wait_terminal(env, first["job_id"])["state"] == "succeeded"
    assert env.acquire("s2", "--accept-stale")[0] == 0
    assert env.run_json("release", "--session-id", "s2")[0] == 0


def test_deadline_reports_live_work_without_cancelling_it(env):
    assert env.acquire("s1")[0] == 0
    rc, first = start(env, timeout=.2, duration=2)
    assert rc == 0
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        _, status = jobs(env, "status", "--job-id", first["job_id"])
        if status["state"] == "overdue":
            break
        time.sleep(.05)
    assert status["state"] == "overdue" and status["pid_alive"]
    assert start(env)[0] == 4
    assert wait_terminal(env, first["job_id"])["state"] == "timeout"
    assert env.run_json("release", "--session-id", "s1")[0] == 0
