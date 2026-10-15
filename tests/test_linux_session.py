"""SPEC 17 items 9-11, 15 and 18: the real remote ``flock`` session daemon.

These tests drive the real ``src/vast_comfy/remote_agent/vc_session.py`` script
through subprocesses with ``VAST_COMFY_RUNTIME_DIR`` / ``VAST_COMFY_COMFY_DIR``
pointed at pytest tmp dirs. They require Linux (``fcntl``/``flock``) and are
skipped on Windows.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.linux_only,
    pytest.mark.skipif(sys.platform != "linux", reason="requires Linux fcntl/flock"),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
VC_SESSION = REPO_ROOT / "src" / "vast_comfy" / "remote_agent" / "vc_session.py"
VC_COMMON = REPO_ROOT / "src" / "vast_comfy" / "remote_agent" / "vc_common.py"

ACQUIRE_TIMEOUT = 30
RELEASE_TIMEOUT = 30


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise AssertionError(f"no JSON payload in output: {text!r}")


class SessionEnv:
    """Runs vc_session.py against an isolated tmp runtime directory."""

    def __init__(self, tmp_path: Path):
        self.runtime = tmp_path / "runtime"
        self.comfy = tmp_path / "ComfyUI"
        self.env = dict(os.environ)
        self.env["VAST_COMFY_RUNTIME_DIR"] = str(self.runtime)
        self.env["VAST_COMFY_COMFY_DIR"] = str(self.comfy)
        self.spawned_pids: list[int] = []

    # -- process helpers ---------------------------------------------------

    def run(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(VC_SESSION), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_json(self, *args: str, timeout: float = 60.0) -> tuple[int, dict, subprocess.CompletedProcess]:
        proc = self.run(*args, timeout=timeout)
        return proc.returncode, last_json(proc.stdout), proc

    def acquire(self, session_id: str, *extra: str, instance_id: str = "42") -> tuple[int, dict]:
        proc = self.run(
            "acquire",
            "--session-id",
            session_id,
            "--instance-id",
            instance_id,
            "--controller",
            json.dumps({"hostname": "ctrl", "user": "tester", "platform": "linux"}),
            "--wait-seconds",
            "10",
            *extra,
            timeout=ACQUIRE_TIMEOUT,
        )
        return proc.returncode, last_json(proc.stdout)

    def spawn_sleep(self, seconds: int = 60) -> int:
        proc = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
        self.spawned_pids.append(proc.pid)
        return proc.pid

    # -- state inspection --------------------------------------------------

    @property
    def lock_dir(self) -> Path:
        return self.runtime / "lock"

    def owner(self) -> dict | None:
        path = self.lock_dir / "owner.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def lock_file(self) -> Path:
        return self.lock_dir / "owner.lock"

    def write_owner(self, owner: dict) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        (self.lock_dir / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
        self.lock_file().touch()

    def register_job(self, job_id: str, session_id: str, pid: int, state: str = "running") -> None:
        job_dir = self.runtime / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "meta.json").write_text(
            json.dumps({"job_id": job_id, "name": "fake", "session_id": session_id, "pid": pid}),
            encoding="utf-8",
        )
        (job_dir / "status.json").write_text(
            json.dumps({"state": state, "pid": pid, "started_at": time.time()}),
            encoding="utf-8",
        )

    def teardown(self) -> None:
        for pid in self.spawned_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        owner = self.owner() or {}
        daemon_pid = owner.get("daemon_pid")
        if daemon_pid:
            try:
                os.kill(int(daemon_pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture()
def session_env(tmp_path):
    env = SessionEnv(tmp_path)
    yield env
    env.teardown()


# ---------------------------------------------------------------------------
# acquire / status / conflict / release
# ---------------------------------------------------------------------------


def test_acquire_conflict_status_and_release(session_env):
    rc, payload = session_env.acquire("s1")
    assert rc == 0, payload
    assert payload["ok"] is True
    assert payload["status"] == "acquired"
    owner = payload["owner"]
    assert owner["session_id"] == "s1"
    assert owner["state"] == "active"
    assert owner["instance_id"] == "42"
    assert owner["heartbeat_seconds"] == 30.0
    assert owner["ttl_seconds"] == 300.0
    assert owner["buffer_seconds"] == 60.0
    # Timestamps come from the instance clock.
    assert abs(owner["heartbeat_at"] - time.time()) < 30
    # The lock file exists and is never deleted by the flow.
    assert session_env.lock_file().is_file()

    # A second acquire for another session fails immediately while the daemon
    # still holds the flock (exit 4, E_LOCK_CONFLICT) — no queueing.
    rc2, conflict = session_env.acquire("s2")
    assert rc2 == 4, conflict
    assert conflict["code"] == "E_LOCK_CONFLICT"
    assert conflict["owner"]["session_id"] == "s1"

    # Status: held by me, daemon alive; also reports the (young) heartbeat age.
    rc3, status, _ = session_env.run_json("status", "--session-id", "s1")
    assert rc3 == 0
    assert status["lock_held"] is True
    assert status["daemon_alive"] is True
    assert status["matches_me"] is True
    assert status["expired"] is False
    assert status["owner"]["session_id"] == "s1"

    # A foreign session sees lock_held but matches_me False.
    rc4, foreign, _ = session_env.run_json("status", "--session-id", "s2")
    assert foreign["lock_held"] is True
    assert foreign["matches_me"] is False

    # Release works and leaves the lock file in place.
    rc5, released, _ = session_env.run_json("release", "--session-id", "s1", "--wait-seconds", "20")
    assert rc5 == 0, released
    assert released["status"] == "released"
    assert session_env.lock_file().is_file()

    rc6, after = session_env.run_json("status", "--session-id", "s1")
    assert after["lock_held"] is False
    assert after["matches_me"] is False
    assert session_env.owner()["state"] == "released"
    # The archived owner history kept the previous record.
    history = (session_env.lock_dir / "owner_history.jsonl").read_text(encoding="utf-8")
    assert "s1" in history


def test_release_refuses_while_a_session_job_is_alive(session_env):
    rc, payload = session_env.acquire("s1")
    assert rc == 0, payload
    pid = session_env.spawn_sleep(60)
    session_env.register_job("job-1", "s1", pid)

    rc2, blocked, proc = session_env.run_json("release", "--session-id", "s1", "--wait-seconds", "10")
    assert rc2 == 4, (blocked, proc.stderr)
    assert blocked["code"] == "E_LOCK_JOBS_RUNNING"
    assert blocked["jobs"][0]["job_id"] == "job-1"
    assert blocked["jobs"][0]["pid_alive"] is True

    # The lock is still held by this session; nothing was abandoned.
    rc3, status, _ = session_env.run_json("status", "--session-id", "s1")
    assert status["lock_held"] is True
    assert status["matches_me"] is True


def test_release_without_ownership_is_refused(session_env):
    rc, payload = session_env.acquire("s1")
    assert rc == 0
    rc2, refused, _ = session_env.run_json("release", "--session-id", "other", "--wait-seconds", "5")
    assert rc2 == 4
    assert refused["code"] == "E_LOCK_NOT_HELD"


# ---------------------------------------------------------------------------
# stale records: report-only unless the user explicitly accepts
# ---------------------------------------------------------------------------


def _stale_owner(session_id: str = "old") -> dict:
    return {
        "state": "active",
        "session_id": session_id,
        "instance_id": "7",
        "controller": {},
        "acquired_at": time.time() - 7200,
        "heartbeat_at": time.time() - 7200,
        "heartbeat_seconds": 30.0,
        "ttl_seconds": 300.0,
        "buffer_seconds": 60.0,
        "daemon_pid": None,
    }


def test_stale_record_requires_explicit_accept_stale(session_env):
    session_env.write_owner(_stale_owner("old"))

    # Without --accept-stale: reported, nothing done (exit 4, E_LOCK_STALE).
    rc, payload = session_env.acquire("new")
    assert rc == 4, payload
    assert payload["code"] == "E_LOCK_STALE"
    assert payload["owner"]["session_id"] == "old"
    assert payload["expired"] is True

    # Status only reports expiry; it never takes over.
    rc2, status, _ = session_env.run_json("status", "--session-id", "new")
    assert status["lock_held"] is False
    assert status["matches_me"] is False
    assert "note" in status  # heartbeat expired note
    assert session_env.owner()["session_id"] == "old"

    # With the explicit flag (no live jobs): takeover works.
    rc3, taken = session_env.acquire("new", "--accept-stale")
    assert rc3 == 0, taken
    assert taken["status"] == "acquired"
    assert taken["takeover"] is True
    assert session_env.owner()["session_id"] == "new"
    assert session_env.owner()["takeover"] is True
    # The old record was archived, never silently overwritten.
    history = (session_env.lock_dir / "owner_history.jsonl").read_text(encoding="utf-8")
    assert '"old"' in history


def test_stale_takeover_refused_while_old_jobs_are_alive(session_env):
    session_env.write_owner(_stale_owner("old"))
    pid = session_env.spawn_sleep(60)
    session_env.register_job("job-old", "old", pid)

    rc, payload = session_env.acquire("new", "--accept-stale")
    assert rc == 4, payload
    assert payload["code"] == "E_LOCK_JOBS_RUNNING"
    assert payload["jobs"][0]["job_id"] == "job-old"
    # No daemon was spawned: the lock is still not held by anyone.
    rc2, status, _ = session_env.run_json("status", "--session-id", "new")
    assert status["lock_held"] is False


def test_takeover_refused_while_a_live_daemon_holds_the_lock(session_env):
    rc, payload = session_env.acquire("s1")
    assert rc == 0, payload
    rc2, conflict = session_env.acquire("s3", "--accept-stale")
    assert rc2 == 4
    assert conflict["code"] == "E_LOCK_CONFLICT"
    assert session_env.owner()["session_id"] == "s1"


# ---------------------------------------------------------------------------
# the lock file is never deleted by any code path
# ---------------------------------------------------------------------------


def test_no_code_path_deletes_the_lock_file():
    session_src = VC_SESSION.read_text(encoding="utf-8")
    common_src = VC_COMMON.read_text(encoding="utf-8")

    forbidden = [
        r"LOCK_FILE\s*\.unlink",  # never unlink the flock file
        r"os\.remove\(",  # never remove anything
        r"os\.unlink\(",
        r"shutil\.rmtree",
        r"rm\s+.*owner\.lock",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, session_src), f"vc_session.py must not match {pattern}"
        assert not re.search(pattern, common_src), f"vc_common.py must not match {pattern}"

    # The lock file is only ever opened with O_CREAT (create-if-missing) and
    # used through flock; release goes through release.request, not deletion.
    assert "O_CREAT" in common_src
    assert "release.request" in session_src
