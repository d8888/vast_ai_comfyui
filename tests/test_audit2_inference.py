"""Second audit: HTTP lifetime is not the management process lifetime. No sockets."""
from __future__ import annotations

import importlib
import io
import json
import os
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_smoke_runner import smoke_env, run_suite, report_of


class WatcherStopped(BaseException):
    """Only a test interruption; never an inference-completion signal."""


@pytest.mark.parametrize("value", ["run-1", "a.b_c-1", "a" * 128])
def test_identifier_accepts_portable_single_components(remote_agent, value):
    assert remote_agent.C.validate_identifier(value) == value


@pytest.mark.parametrize("value", ["", ".", "..", "../a", "/a", "a/b", "a\\b", "C:a", "a.",
                                  "a ", "é", "_a", "a" * 129, "CON", "nul.txt", "COM9.png", "lpt1"])
def test_identifier_rejects_unsafe_values(remote_agent, value):
    with pytest.raises(ValueError):
        remote_agent.C.validate_identifier(value)


@pytest.mark.parametrize("value", ["../a", "/a", "a//b", "a/./b", "a/../b", "a\\b", "C:a", "a/", ""])
def test_confined_path_rejects_escapes(remote_agent, value):
    with pytest.raises(ValueError):
        remote_agent.C.confined_path(remote_agent.runtime_dir, value)


def test_confined_path_checks_symlinks(remote_agent, tmp_path):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError):
        remote_agent.C.confined_path(root, "link/file")


@pytest.mark.parametrize("value", ["http://example.com/a", "https://u:p@example.com/a", "https://u@example.com/a",
                                  "https://example.com/a?token=x", "https://example.com/a#x",
                                  "https://example.com/%61", "https://example.com/a%3Ftoken=x",
                                  "https://example.com:444/a", "https://example.com/a/../b",
                                  "https://example.com/a//b", "https://example.com/a b", "https://example.com/é",
                                  "https://example.com\\@evil.test/a"])
def test_public_url_rejects_credentials_and_ambiguity(remote_agent, value):
    with pytest.raises(ValueError):
        remote_agent.C.validate_public_url(value)


def test_public_url_and_redaction(remote_agent):
    url = "https://huggingface.co:443/owner/repo/resolve/abc/model-v1.safetensors"
    assert remote_agent.C.validate_public_url(url) == url
    text = remote_agent.C.redact("https://u:private@example.com/f?%74oken=private1&api%5Fkey=private2&plain=ok "
                                "https://example.com/f%3Faccess%255Ftoken%3Dprivate3%26plain%3Dok "
                                "https://example.com/f%253Ftoken%253Dprivate4")
    assert all(secret not in text for secret in ("private", "private1", "private2", "private3", "private4"))
    assert "plain=ok" in text


def test_run_id_claim_is_exclusive_and_preserves_old_evidence(remote_agent):
    directory = remote_agent.C.reserve_run_directory("run-1", stage="smoke")
    (directory / "report.json").write_text("old evidence")
    with pytest.raises(remote_agent.C.RemoteFail):
        remote_agent.C.reserve_run_directory("run-1", stage="smoke")
    assert (directory / "report.json").read_text() == "old evidence"
    with pytest.raises(remote_agent.C.RemoteFail):
        remote_agent.C.reserve_run_directory("../escape", stage="smoke")


def test_prepare_and_smoke_share_one_reserved_run(remote_agent, monkeypatch):
    monkeypatch.setenv("VAST_COMFY_OPERATION_TOKEN", "a" * 32)
    directory = remote_agent.C.reserve_run_directory("prepared", stage="prepare")
    assert remote_agent.C.reserve_run_directory("prepared", stage="smoke", from_prepare=True) == directory
    with pytest.raises(remote_agent.C.RemoteFail):
        remote_agent.C.reserve_run_directory("prepared", stage="smoke", from_prepare=True)


def test_late_success_is_sticky_failure_and_stops_later_cases(smoke_env, monkeypatch):
    module = smoke_env.agent.SMOKE
    common = smoke_env.agent.C
    job_dir = common.JOBS_DIR / "test-job"; job_dir.mkdir(parents=True)
    monkeypatch.setenv("VAST_COMFY_JOB_ID", "test-job")
    observer = module.wait_for_history
    calls = []

    def wait(base, prompt_id, timeout):
        calls.append(prompt_id)
        if len(calls) == 1:
            raise module.CaseFailure("SMOKE_TIMEOUT", "deadline", {"prompt_id": prompt_id})
        assert common.pending_inferences()[0]["prompt_id"] == prompt_id
        assert common.read_json(job_dir / "progress.json")["state"] == "overdue"
        with pytest.raises(common.RemoteFail):
            common.assert_no_pending_inferences()
        return observer(base, prompt_id, timeout)

    monkeypatch.setattr(module, "wait_for_history", wait)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result, code = run_suite(smoke_env, cases=smoke_env.cases)
    assert code != 0 and result["verdict"] == "FAILED"
    assert len(smoke_env.fake.prompts) == 1
    cases = report_of(smoke_env)["cases"]
    assert [case["status"] for case in cases] == ["failed", "not_run", "not_run", "not_run"]
    assert cases[0]["reason"] == "SMOKE_TIMEOUT" and cases[0]["prompt_id"] == "prompt-1"
    assert not common.pending_inferences()
    assert common.read_json(job_dir / "progress.json")["state"] == "overdue"


@pytest.mark.parametrize("failure", [ConnectionError("lost history"), ValueError("bad JSON")])
def test_history_loss_retains_pending_until_positive_completion(remote_agent, monkeypatch, failure):
    module = remote_agent.SMOKE
    monkeypatch.setattr(module, "submit_prompt", lambda *a, **k: "prompt-1")
    outcomes = iter([failure, {"status": {"completed": True, "status_str": "success"}}])

    def observer(*args):
        item = next(outcomes)
        if isinstance(item, Exception):
            raise item
        assert remote_agent.C.pending_inferences()
        return item

    monkeypatch.setattr(module, "wait_for_history", observer)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(module.CaseFailure, match="Cannot establish"):
        module.submit_and_watch("unused", {}, 1, session_id="S", run_id="r", case_id="base_t2i")
    assert not remote_agent.C.pending_inferences()


def test_late_http_reply_cannot_evade_deadline(remote_agent, monkeypatch):
    module = remote_agent.SMOKE
    clock = {"now": 0.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(module, "submit_prompt", lambda *a, **k: "prompt-1")

    def slow_reply(*args):
        clock["now"] = 2.0
        return {"status": {"completed": True, "status_str": "success"}}

    monkeypatch.setattr(module, "wait_for_history", slow_reply)
    with pytest.raises(module.CaseFailure) as error:
        module.submit_and_watch("unused", {}, 1, session_id="S", run_id="r", case_id="base_t2i")
    assert error.value.reason == "SMOKE_TIMEOUT" and error.value.details["inference_terminated"]
    assert not remote_agent.C.pending_inferences()


@pytest.mark.parametrize("response", [None, {}, {"prompt_id": "../unsafe"}, {"prompt_id": 123}])
def test_unknown_post_never_retries_and_immediately_reports_overdue(remote_agent, monkeypatch, response):
    module = remote_agent.SMOKE
    common = remote_agent.C
    monkeypatch.setenv("VAST_COMFY_JOB_ID", "job-1")
    requests = []

    def http(url, **kwargs):
        assert common.pending_inferences()[0]["state"] == "submitting"
        requests.append((url, kwargs))
        return response

    monkeypatch.setattr(module.CM, "http_json", http)
    monkeypatch.setattr(module.time, "sleep", Mock(side_effect=WatcherStopped))
    with pytest.raises(WatcherStopped):
        module.submit_and_watch("unused", {}, 1, session_id="S", run_id="r", case_id="base_t2i")
    assert len(requests) == 1 and requests[0][1]["method"] == "POST"
    pending = common.pending_inferences()
    assert pending[0]["state"] == "submission_unknown"
    assert common.read_json(common.JOBS_DIR / "job-1/progress.json")["state"] == "overdue"
    with pytest.raises(common.RemoteFail):
        common.assert_no_pending_inferences()


@pytest.mark.parametrize("code,rejected", [(400, True), (500, False)])
def test_only_proven_validation_rejection_clears_intent(remote_agent, monkeypatch, code, rejected):
    module = remote_agent.SMOKE
    error = urllib.error.HTTPError("unused", code, "test", {}, io.BytesIO(b'{"error": {"type": "invalid_prompt"}}'))
    request = Mock(side_effect=error)
    monkeypatch.setattr(module.CM, "http_json", request)
    monkeypatch.setattr(module.time, "sleep", Mock(side_effect=WatcherStopped))
    with pytest.raises(module.CaseFailure if rejected else WatcherStopped):
        module.submit_and_watch("unused", {}, 1, session_id="S", run_id="r", case_id="base_t2i")
    assert bool(remote_agent.C.pending_inferences()) is not rejected
    assert request.call_count == 1


def test_post_response_loss_after_acceptance_does_not_resubmit(remote_agent, monkeypatch):
    module = remote_agent.SMOKE
    requests = []

    def accepted_but_response_lost(url, **kwargs):
        assert remote_agent.C.pending_inferences()
        requests.append(kwargs)
        raise urllib.error.URLError("synthetic accepted POST lost its reply")

    monkeypatch.setattr(module.CM, "http_json", accepted_but_response_lost)
    monkeypatch.setattr(module.time, "sleep", Mock(side_effect=WatcherStopped))
    with pytest.raises(WatcherStopped):
        module.submit_and_watch("unused", {}, 1, session_id="S", run_id="r", case_id="base_t2i")
    assert len(requests) == 1
    assert remote_agent.C.pending_inferences()[0]["state"] == "submission_unknown"


def test_dead_watcher_pending_blocks_all_mutations_and_reports_both_holds(remote_agent, monkeypatch):
    common = remote_agent.C
    common.write_inference({"intent_id": "intent", "state": "running", "session_id": "old", "watcher_pid": 999999})
    monkeypatch.setattr(common, "pid_alive", lambda _: False)
    monkeypatch.setattr(common, "operation_held", lambda: False)
    monkeypatch.setattr(common, "probe_lock_held", lambda: False)
    session = importlib.import_module("vc_session")
    for function in (session.cmd_acquire, session.cmd_release, session.cmd_begin_operation, session.cmd_end_operation):
        with pytest.raises(common.RemoteFail) as error:
            function(SimpleNamespace(session_id="new", accept_stale=True))
        assert error.value.details["persistent_hold"] is True
    with pytest.raises(common.RemoteFail):
        with common.operation("new"):
            pytest.fail("must not enter")
    output = Mock(); monkeypatch.setattr(common, "emit", output)
    session.cmd_status(SimpleNamespace(session_id="new"))
    payload = output.call_args.args[0]
    assert payload["persistent_hold"] and not payload["operation_kernel_held"] and not payload["lock_held"]


def test_corrupt_intent_is_fail_closed_and_cannot_finish_operation(remote_agent):
    common = remote_agent.C
    common.INFERENCE_DIR.mkdir(parents=True)
    (common.INFERENCE_DIR / "broken.json").write_text("not json")
    common.atomic_write_json(common.OPERATION_OWNER, {"token": "a", "state": "active"})
    assert not common.finish_operation("a")
    assert common.read_json(common.OPERATION_OWNER)["state"] == "active"
    assert common.pending_inferences(session_id="any")[0]["record_unreadable"]


def test_incomplete_terminal_record_is_not_completion_evidence(remote_agent):
    common = remote_agent.C
    common.write_inference({"intent_id": "bad-terminal", "state": "terminal"})
    assert common.pending_inferences(job_id="any")[0]["record_unreadable"]


def test_runner_keeps_pending_child_failure_nonterminal(remote_agent, monkeypatch):
    jobs = importlib.import_module("vc_jobs")
    common = remote_agent.C
    directory = common.JOBS_DIR / "job-1"; directory.mkdir(parents=True)
    common.write_inference({"intent_id": "intent", "state": "running", "job_id": "job-1"})
    common.atomic_write_json(common.OPERATION_OWNER, {"token": "token", "state": "active"})
    monkeypatch.setenv("VAST_COMFY_OPERATION_TOKEN", "token")
    monkeypatch.setattr(common, "child_process_options", lambda: {})
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=99, wait=lambda **k: 1))
    monkeypatch.setattr(jobs.time, "sleep", Mock(side_effect=WatcherStopped))
    with pytest.raises(WatcherStopped):
        jobs.cmd_run(SimpleNamespace(job_dir=str(directory), timeout=1, argv=["unused"]))
    status = common.read_json(directory / "status.json")
    assert status["state"] == "overdue" and "ended_at" not in status
    assert common.read_json(common.OPERATION_OWNER)["state"] == "active"


def test_job_status_exposes_case_overdue_without_a_second_status_writer(remote_agent, monkeypatch):
    common = remote_agent.C
    jobs = importlib.import_module("vc_jobs")
    directory = common.JOBS_DIR / "job-1"
    common.atomic_write_json(directory / "meta.json", {"job_id": "job-1", "pid": 2})
    initial = {"state": "running", "pid": 3, "runner_pid": 2}
    common.atomic_write_json(directory / "status.json", initial)
    monkeypatch.setenv("VAST_COMFY_JOB_ID", "job-1")
    common.write_job_progress({"state": "overdue", "prompt_id": "p"})
    monkeypatch.setattr(common, "pid_alive", lambda _: True)
    monkeypatch.setattr(common, "operation_held", lambda: True)
    output = Mock(); monkeypatch.setattr(common, "emit", output)
    jobs.cmd_status(SimpleNamespace(job_id="job-1"))
    assert output.call_args.args[0]["state"] == "overdue"
    assert common.read_json(directory / "status.json") == initial
    common.atomic_write_json(directory / "status.json", {"state": "failed"})
    jobs.cmd_status(SimpleNamespace(job_id="job-1"))
    assert output.call_args.args[0]["state"] == "failed"  # stale progress cannot revive terminal work


def test_status_deadline_is_visible_even_during_blocked_http_call(remote_agent, monkeypatch):
    common = remote_agent.C
    jobs = importlib.import_module("vc_jobs")
    directory = common.JOBS_DIR / "job-1"
    common.atomic_write_json(directory / "progress.json", {"state": "running", "deadline_at": 1})
    common.write_inference({"intent_id": "intent", "state": "submitting", "job_id": "job-1"})
    monkeypatch.setattr(common, "pid_alive", lambda _: True)
    state, pending, progress = jobs._job_state(directory, {}, {"state": "running", "runner_pid": 1})
    assert state == "overdue" and pending and progress["reason"] == "SMOKE_TIMEOUT"


def test_parent_never_overwrites_fast_child_terminal_status(remote_agent, monkeypatch):
    jobs = importlib.import_module("vc_jobs")
    common = remote_agent.C
    monkeypatch.setattr(common, "ensure_owns", lambda _: {})
    monkeypatch.setattr(common, "child_process_options", lambda: {})
    monkeypatch.setattr(jobs, "_new_job_id", lambda: "fast-job")
    monkeypatch.setattr(common, "emit", Mock())
    terminal = {"state": "succeeded", "pid": 5, "exit_code": 0, "ended_at": 1}

    def spawn(*args, **kwargs):
        common.atomic_write_json(common.JOBS_DIR / "fast-job/status.json", terminal)
        return SimpleNamespace(pid=4)

    monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    jobs._start_job(SimpleNamespace(session_id="S", name="fast", argv=["unused"],
                                    timeout_seconds=1, env=[]), SimpleNamespace(fd=1, handoff=lambda: None))
    assert common.read_json(common.JOBS_DIR / "fast-job/status.json") == terminal


def test_nested_job_start_is_rejected_before_spawn(remote_agent, monkeypatch):
    jobs = importlib.import_module("vc_jobs")
    monkeypatch.setenv("VAST_COMFY_OPERATION_TOKEN", "a" * 32)
    spawn = Mock(); monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    with pytest.raises(remote_agent.C.RemoteFail, match="nested"):
        jobs.cmd_start(SimpleNamespace(session_id="S", name="nested"))
    spawn.assert_not_called()


def test_smoke_main_requests_operation_guard(remote_agent, monkeypatch):
    dispatch = Mock(); monkeypatch.setattr(remote_agent.C, "run_cli", dispatch)
    remote_agent.SMOKE.main(["run", "--session-id", "S", "--cases-b64", "W10=",
                             "--workflows-dir", "unused", "--assets-dir", "unused"])
    assert dispatch.call_args.kwargs["guard"] is True


def test_resolve_inference_requires_confirmation_and_clears_only_after_it(remote_agent, monkeypatch):
    common = remote_agent.C
    session = importlib.import_module("vc_session")
    common.write_inference({"intent_id": "intent-1", "state": "submission_unknown", "job_id": "job-1"})
    with pytest.raises(common.RemoteFail) as error:
        session.cmd_resolve_inference(SimpleNamespace(intent_id="intent-1", confirm_verified=False, note=""))
    assert error.value.code == "E_APPROVAL_REQUIRED"
    assert common.pending_inferences()
    output = Mock(); monkeypatch.setattr(common, "emit", output)
    session.cmd_resolve_inference(SimpleNamespace(intent_id="intent-1", confirm_verified=True, note="checked queue"))
    assert output.call_args.args[0]["resolved"] is True
    record = common.read_json(common.INFERENCE_DIR / "intent-1.json")
    assert record["state"] == "rejected" and record["rejection_confirmed"] is True
    assert record["resolved_by"] == "operator" and record["note"] == "checked queue"
    assert not common.pending_inferences()


def test_resolve_inference_never_overwrites_terminal_evidence_or_unsafe_ids(remote_agent, monkeypatch):
    common = remote_agent.C
    session = importlib.import_module("vc_session")
    common.write_inference({"intent_id": "done", "state": "terminal", "prompt_id": "p",
                            "terminal_status": {"completed": True, "status_str": "success"}})
    output = Mock(); monkeypatch.setattr(common, "emit", output)
    session.cmd_resolve_inference(SimpleNamespace(intent_id="done", confirm_verified=True, note=""))
    assert output.call_args.args[0]["already_resolved"] is True
    assert common.read_json(common.INFERENCE_DIR / "done.json")["state"] == "terminal"
    for bad in ("../escape", "missing-id"):
        with pytest.raises(common.RemoteFail):
            session.cmd_resolve_inference(SimpleNamespace(intent_id=bad, confirm_verified=True, note=""))
    assert not common.pending_inferences()


def test_cli_resolve_inference_requires_flag_and_passes_runner(cfg, monkeypatch):
    from vast_comfy import cli, locks

    backend = Mock(return_value={"ok": True, "resolved": True})
    monkeypatch.setattr(locks, "resolve_inference", backend)
    runner = Mock()
    ctx = SimpleNamespace(cfg=cfg, instance_and_runner=lambda *_: ({}, runner), state=Mock())
    args = cli.build_parser().parse_args(
        ["session", "resolve-inference", "--instance", "1", "--intent-id", "i1"])
    with pytest.raises(cli.UsageError):
        cli.cmd_session_resolve_inference(ctx, args)
    backend.assert_not_called()
    args = cli.build_parser().parse_args(
        ["session", "resolve-inference", "--instance", "1", "--intent-id", "i1", "--confirm-verified"])
    cli.cmd_session_resolve_inference(ctx, args)
    backend.assert_called_once_with(runner, cfg, "i1", confirm_verified=True)
