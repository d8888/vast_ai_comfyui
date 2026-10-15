"""SPEC 17 items 9-11, 13 and 18: session lock + prepare flow (fake runner)."""

from __future__ import annotations

import pytest

from mocks.fake_ssh import FakeRunner
from vast_comfy import locks, prepare, remote
from vast_comfy.errors import VastComfyError


# ---------------------------------------------------------------------------
# locks.acquire / assert_owns
# ---------------------------------------------------------------------------


def test_locks_acquire_sends_the_configured_heartbeat_values(cfg):
    runner = FakeRunner()
    runner.add_json("vc_session.py", {"ok": True, "status": "acquired", "owner": {"session_id": "sess-1"}})

    payload = locks.acquire(runner, cfg, instance_id=4242, session_id="sess-1")

    assert payload["session_id"] == "sess-1"
    call = runner.calls_containing("vc_session.py")[0]
    argv = call.argv
    for flag, value in (
        ("--heartbeat-seconds", "30"),
        ("--ttl-seconds", "300"),
        ("--buffer-seconds", "60"),
    ):
        assert flag in argv, f"{flag} missing from the acquire command"
        assert argv[argv.index(flag) + 1] == value
    # The values match the default configuration exactly.
    assert int(cfg.require("lock.heartbeat_seconds")) == 30
    assert int(cfg.require("lock.heartbeat_ttl_seconds")) == 300
    assert int(cfg.require("lock.heartbeat_buffer_seconds")) == 60
    assert argv[argv.index("--instance-id") + 1] == "4242"
    assert "--accept-stale" not in argv  # never automatic


def test_locks_acquire_respects_configured_values(tmp_path):
    from vast_comfy.config import load_config

    cfg = load_config(
        None,
        {
            "lock.heartbeat_seconds": 15,
            "lock.heartbeat_ttl_seconds": 120,
            "lock.heartbeat_buffer_seconds": 20,
        },
    )
    runner = FakeRunner()
    runner.add_json("vc_session.py", {"ok": True, "status": "acquired", "owner": {}})
    locks.acquire(runner, cfg, instance_id=1, session_id="s")
    argv = runner.calls_containing("vc_session.py")[0].argv
    assert argv[argv.index("--heartbeat-seconds") + 1] == "15"
    assert argv[argv.index("--ttl-seconds") + 1] == "120"
    assert argv[argv.index("--buffer-seconds") + 1] == "20"


def test_locks_recover_sets_accept_stale(cfg):
    runner = FakeRunner()
    runner.add_json("vc_session.py", {"ok": True, "status": "acquired", "owner": {}})
    locks.recover(runner, cfg, instance_id=1, session_id="s")
    assert "--accept-stale" in runner.calls_containing("vc_session.py")[0].argv


@pytest.mark.parametrize(
    "status_payload",
    [
        {"ok": True, "lock_held": False, "matches_me": False},
        {"ok": True, "lock_held": True, "matches_me": False},
        {"ok": True, "lock_held": False, "matches_me": True},
    ],
)
def test_assert_owns_raises_when_lock_is_not_ours(cfg, status_payload):
    runner = FakeRunner()
    runner.add_json("vc_session.py", status_payload)
    with pytest.raises(VastComfyError) as excinfo:
        locks.assert_owns(runner, cfg, "sess-1")
    error = excinfo.value
    assert error.code == "E_LOCK_NOT_HELD"
    assert error.details["session_id"] == "sess-1"
    assert "acquire" in " ".join(error.hints).lower()


def test_assert_owns_passes_when_the_lock_is_mine(cfg):
    runner = FakeRunner()
    runner.add_json("vc_session.py", {"ok": True, "lock_held": True, "matches_me": True, "owner": {"session_id": "sess-1"}})
    payload = locks.assert_owns(runner, cfg, "sess-1")
    assert payload["matches_me"] is True


# ---------------------------------------------------------------------------
# prepare.prepare_existing
# ---------------------------------------------------------------------------


def _prepare_runner(job_state: str, summary: dict | None) -> FakeRunner:
    runner = FakeRunner()
    # deploy_package: dirs, package verification (already present), symlink.
    runner.add_rule("mkdir", rc=0)
    runner.add_json("verify-package", {"ok": True, "checked": 21, "problems": []})
    runner.add_rule("ln -sfn", rc=0)
    runner.add_json("vc_session.py acquire", {"ok": True, "status": "acquired", "owner": {"session_id": "sess-1"}})
    runner.add_json("vc_session.py release", {"ok": True, "status": "released", "released_at": 111.0})
    runner.add_json(
        "vc_jobs.py start",
        {"ok": True, "job_id": "job-123", "name": "prepare", "timeout_seconds": 9900.0},
    )
    runner.add_json(
        "vc_jobs.py status",
        {"ok": True, "job_id": "job-123", "state": job_state, "exit_code": 0 if job_state == "succeeded" else 5},
    )
    if summary is not None:
        runner.add_rule("cat ", stdout=__import__("json").dumps(summary) + "\n")
    else:
        # No usable summary on the instance: `cat` returns garbage so
        # read_remote_json raises a structured error internally.
        runner.add_rule("cat ", stdout="no summary was written\n")
    return runner


def test_prepare_existing_failure_keeps_the_lock_and_never_releases(cfg):
    runner = _prepare_runner(
        "failed",
        {"ok": False, "verdict": "FAILED", "run_id": "run-1", "cases": [{"id": "base_t2i", "status": "failed"}]},
    )

    with pytest.raises(VastComfyError) as excinfo:
        prepare.prepare_existing(runner, cfg, instance_id=4242, session_id="sess-1", run_id="run-1")

    error = excinfo.value
    assert error.code == "E_SMOKE_FAILED"
    assert error.stage == "prepare"
    assert "still held" in error.details["lock_state"]
    assert "continues to run" in error.details["instance_billing"]
    assert error.details["summary"]["verdict"] == "FAILED"
    assert error.details["session_id"] == "sess-1"
    assert error.details["job_id"] == "job-123"

    # No release was attempted: the lock is kept for the human to decide.
    assert not any("release" in call.joined for call in runner.calls)
    assert any("no cleanup" in hint.lower() for hint in error.hints)


def test_prepare_existing_failure_without_summary_is_job_failed(cfg):
    runner = _prepare_runner("failed", None)
    with pytest.raises(VastComfyError) as excinfo:
        prepare.prepare_existing(runner, cfg, instance_id=4242, session_id="sess-1", run_id="run-2")
    assert excinfo.value.code == "E_JOB_FAILED"
    assert not any("release" in call.joined for call in runner.calls)


def test_prepare_existing_success_releases_and_reports_ready(cfg):
    runner = _prepare_runner(
        "succeeded",
        {"ok": True, "verdict": "READY", "run_id": "run-3", "cases": [{"id": "base_t2i", "status": "passed"}]},
    )

    result = prepare.prepare_existing(
        runner, cfg, instance_id=4242, session_id="sess-1", run_id="run-3", release_on_success=True
    )

    assert result["verdict"] == "READY"
    assert result["instance_id"] == 4242
    assert result["session_id"] == "sess-1"
    assert result["release"]["status"] == "released"
    assert result["release_error"] is None
    assert any("vc_session.py release" in call.joined for call in runner.calls)
    # The remote prepare job ran through the deployed package entrypoint.
    start_call = runner.calls_containing("vc_jobs.py start")[0]
    assert "/workspace/.vast_comfy/current/remote/vc_prepare.py" in start_call.joined


def test_prepare_existing_success_can_keep_the_session(cfg):
    runner = _prepare_runner("succeeded", {"ok": True, "verdict": "READY", "run_id": "run-4"})
    result = prepare.prepare_existing(
        runner, cfg, instance_id=4242, session_id="sess-1", run_id="run-4", release_on_success=False
    )
    assert result["verdict"] == "READY"
    assert result["release"] is None
    assert not any("release" in call.joined for call in runner.calls)


def test_prepare_existing_reads_lock_settings_from_config(cfg):
    runner = _prepare_runner("succeeded", {"ok": True, "verdict": "READY"})
    prepare.prepare_existing(runner, cfg, instance_id=4242, session_id="sess-1", run_id="run-5")
    acquire_call = next(call for call in runner.calls if "vc_session.py acquire" in call.joined)
    assert acquire_call.argv[acquire_call.argv.index("--heartbeat-seconds") + 1] == "30"


# ---------------------------------------------------------------------------
# deploy_package (fake runner): idempotence and verification
# ---------------------------------------------------------------------------


def test_deploy_package_skips_upload_when_remote_is_verified(cfg):
    runner = FakeRunner()
    runner.add_rule("mkdir", rc=0)
    runner.add_json("verify-package", {"ok": True, "checked": 21, "problems": []})
    runner.add_rule("ln -sfn", rc=0)

    result = remote.deploy_package(runner, cfg)

    assert result["verified"] is True
    assert result["already_present"] is True
    assert result["uploaded"] == 0
    assert runner.uploads == []


def test_deploy_package_uploads_and_verifies_when_missing(cfg):
    runner = FakeRunner()
    runner.add_rule("mkdir", rc=0)
    # Probe against the *current* link fails -> the package is uploaded.
    runner.add_rule("current/remote/vc_common.py", stdout="probe-not-json\n", rc=1)
    runner.add_rule("ln -sfn", rc=0)
    runner.add_json("vc_common.py", {"ok": True, "checked": 21, "problems": []})

    result = remote.deploy_package(runner, cfg)

    assert result["uploaded"] > 0
    assert result["verified"] is True
    uploaded_rels = {remote_path.split("/pkg/")[-1].split("/", 1)[1] for _, remote_path in runner.uploads}
    assert "remote/vc_smoke.py" in uploaded_rels
    assert "manifest.sha256" in uploaded_rels


def test_deploy_package_fails_when_remote_verification_fails(cfg):
    runner = FakeRunner()
    runner.add_rule("mkdir", rc=0)
    runner.add_rule("current/remote/vc_common.py", stdout="probe-not-json\n", rc=1)
    runner.add_json("vc_common.py", {"ok": False, "checked": 21, "problems": ["sha256 mismatch: remote/vc_smoke.py"]})

    with pytest.raises(VastComfyError) as excinfo:
        remote.deploy_package(runner, cfg)
    assert excinfo.value.code == "E_DEPLOY"
    assert "not trusted" in excinfo.value.message
