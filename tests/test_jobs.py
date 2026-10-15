"""SPEC 17 item 12: background jobs / status / logs controller logic."""

from __future__ import annotations

import pytest

from mocks.fake_ssh import FakeRunner
from vast_comfy import jobs, remote
from vast_comfy.errors import JobTimeoutError, VastComfyError


def fake_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def sleep(seconds):
        state["t"] += float(seconds)

    return clock, sleep, state


# ---------------------------------------------------------------------------
# jobs.wait
# ---------------------------------------------------------------------------


def test_wait_returns_on_terminal_state(cfg):
    runner = FakeRunner()
    runner.add_json("vc_jobs.py status", {"ok": True, "job_id": "job-1", "state": "succeeded", "exit_code": 0})

    final = jobs.wait(runner, cfg, "job-1", timeout_seconds=60, sleep=lambda _s: None)

    assert final["state"] == "succeeded"
    assert len(runner.calls_containing("vc_jobs.py status")) == 1


@pytest.mark.parametrize("state", ["succeeded", "failed", "timeout", "crashed"])
def test_wait_returns_on_every_terminal_state(cfg, state):
    runner = FakeRunner()
    runner.add_json("vc_jobs.py status", {"ok": True, "job_id": "j", "state": state})
    final = jobs.wait(runner, cfg, "j", timeout_seconds=60, sleep=lambda _s: None)
    assert final["state"] == state


def test_wait_timeout_reports_the_job_may_still_be_running(cfg):
    runner = FakeRunner()
    runner.add_json("vc_jobs.py status", {"ok": True, "job_id": "job-9", "state": "running", "pid": 123})
    clock, sleep, state = fake_clock()

    with pytest.raises(JobTimeoutError) as excinfo:
        jobs.wait(runner, cfg, "job-9", timeout_seconds=30, poll_seconds=10, clock=clock, sleep=sleep)

    error = excinfo.value
    assert error.code == "E_JOB_TIMEOUT"
    assert error.details["job_may_still_be_running"] is True
    assert error.details["job_id"] == "job-9"
    assert error.details["last_status"]["state"] == "running"
    # The timeout is bounded and the job was NOT stopped.
    assert state["t"] == 30.0
    assert not any("stop" in call.joined or "cancel" in call.joined for call in runner.calls)
    assert any("does not stop" in hint.lower() for hint in error.hints)


def test_wait_or_fail_raises_with_log_tail(cfg):
    runner = FakeRunner()
    runner.add_json("vc_jobs.py status", {"ok": True, "job_id": "jf", "state": "failed", "exit_code": 1})
    runner.add_json("vc_jobs.py logs", {"ok": True, "job_id": "jf", "state": "failed", "log": "boom\n"})
    with pytest.raises(VastComfyError) as excinfo:
        jobs.wait_or_fail(runner, cfg, "jf", timeout_seconds=30, sleep=lambda _s: None)
    assert excinfo.value.code == "E_JOB_FAILED"
    assert excinfo.value.details["log_tail"] == "boom\n"


# ---------------------------------------------------------------------------
# jobs.final_json_from_log
# ---------------------------------------------------------------------------


def test_final_json_from_log_extracts_the_last_ok_object():
    log = (
        "[vast-comfy] starting\n"
        "{\"ok\": true, \"note\": \"early\"}\n"
        "noise\n"
        "{\"ok\": false, \"code\": \"E_LOCK_CONFLICT\", \"message\": \"held\"}\n"
        "{\"unrelated\": 1}\n"
        "{\"ok\": true, \"verdict\": \"READY\", \"cases\": []}\n"
        "trailing junk\n"
    )
    payload = jobs.final_json_from_log(log)
    assert payload == {"ok": True, "verdict": "READY", "cases": []}


def test_final_json_from_log_returns_none_without_a_json_object():
    assert jobs.final_json_from_log("plain log\nnothing to see\n") is None
    assert jobs.final_json_from_log("") is None
    assert jobs.final_json_from_log('{"not_json": true\n') is None


# ---------------------------------------------------------------------------
# remote.invoke turns remote failures into structured errors
# ---------------------------------------------------------------------------


def test_invoke_raises_remote_code_on_ok_false(cfg):
    runner = FakeRunner()
    runner.add_rule(
        "vc_session.py",
        stdout='{"ok": false, "code": "E_LOCK_CONFLICT", "message": "another session holds it", "owner": {"session_id": "other"}}\n',
        rc=4,
    )
    with pytest.raises(VastComfyError) as excinfo:
        remote.invoke(runner, cfg, "vc_session.py", ["status", "--session-id", ""], timeout_seconds=30)
    error = excinfo.value
    assert error.code == "E_LOCK_CONFLICT"
    assert error.stage == "remote"
    assert error.details["owner"] == {"session_id": "other"}
    assert error.details["remote_tool"] == "vc_session.py"


def test_invoke_raises_internal_when_no_json_result(cfg):
    runner = FakeRunner()
    runner.add_rule("vc_jobs.py", stdout="not json at all\n", rc=1)
    with pytest.raises(VastComfyError) as excinfo:
        remote.invoke(runner, cfg, "vc_jobs.py", ["status"])
    assert excinfo.value.code == "E_INTERNAL"


def test_invoke_rejects_tool_name_paths(cfg):
    runner = FakeRunner()
    with pytest.raises(VastComfyError) as excinfo:
        remote.invoke(runner, cfg, "../vc_jobs.py", ["status"])
    assert excinfo.value.code == "E_INTERNAL"
    assert runner.calls == []


def test_prepare_command_uses_the_deployed_current_link(cfg):
    argv = jobs.prepare_command(cfg, "vc_smoke.py", ["run"])
    assert argv[0] == "/venv/main/bin/python"
    assert argv[1] == "/workspace/.vast_comfy/current/remote/vc_smoke.py"
    assert argv[2:] == ["run"]


def test_jobs_start_passes_timeout_and_env(cfg):
    runner = FakeRunner()
    runner.add_json("vc_jobs.py start", {"ok": True, "job_id": "job-77", "pid": 4242})
    payload = jobs.start(
        runner,
        cfg,
        "sess-1",
        "smoke",
        ["/venv/main/bin/python", "x.py"],
        timeout_seconds=120,
        env=["A=1"],
    )
    assert payload["job_id"] == "job-77"
    call = runner.calls_containing("vc_jobs.py start")[0]
    assert call.argv[call.argv.index("--session-id") + 1] == "sess-1"
    assert call.argv[call.argv.index("--timeout-seconds") + 1] == "120.0"
    assert "--env" in call.argv and "A=1" in call.argv
    assert call.argv[-2:] == ["/venv/main/bin/python", "x.py"]
