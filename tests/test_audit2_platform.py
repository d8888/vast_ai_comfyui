"""Second-audit regression evidence: all platform/SSH/service IO is mocked."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from offer_factory import make_offer
from vast_comfy import api, cli, comfy, prepare, ssh
from vast_comfy.config import load_config
from vast_comfy.errors import VastComfyError
from vast_comfy.vast import rental


@pytest.mark.parametrize("payload", [None, {}, [], {"success": "false", "new_contract": 42},
    {"success": 1, "new_contract": 42}, {"success": False, "new_contract": 42},
    *({"success": True, "new_contract": value} for value in
      [None, "broken", True, False, 0, -1, 1.2, "1.2", "-1", "0", float("nan")])])
def test_ambiguous_create_shapes_never_retried(cfg, monkeypatch, payload):
    monkeypatch.setattr(rental, "rental_preflight", lambda *_: {})
    client = Mock(); client.create_instance.return_value = payload
    with pytest.raises(VastComfyError) as error:
        rental.create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)
    assert error.value.code == "E_CREATE_UNCERTAIN"
    assert error.value.details["instance_may_be_billing"] is True
    assert error.value.details["label"] == "audit"
    assert error.value.details["since_epoch"] > 0
    client.create_instance.assert_called_once()


@pytest.mark.parametrize("instance_id", [42, "42"])
def test_exact_create_success_accepts_positive_id(cfg, monkeypatch, instance_id):
    monkeypatch.setattr(rental, "rental_preflight", lambda *_: {})
    client = Mock(); client.create_instance.return_value = {"success": True, "new_contract": instance_id}
    assert rental.create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)["instance_id"] == 42


def test_unexpected_client_failure_after_dispatch_is_uncertain(cfg, monkeypatch):
    monkeypatch.setattr(rental, "rental_preflight", lambda *_: {})
    client = Mock(); client.create_instance.side_effect = ValueError("synthetic-sensitive-value")
    with pytest.raises(VastComfyError) as error:
        rental.create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)
    assert error.value.code == "E_CREATE_UNCERTAIN"
    assert "synthetic-sensitive-value" not in str(error.value.to_dict())
    client.create_instance.assert_called_once()


@pytest.mark.parametrize("error_code", ["E_SSH_CONNECT", "E_SSH_AUTH", "E_SSH_HOSTKEY"])
def test_api_paid_failure_has_resource_and_billing(cfg, monkeypatch, error_code):
    monkeypatch.setattr(api, "verify_offer", lambda *_: make_offer())
    monkeypatch.setattr(api, "create_instance", lambda *a, **k: {"instance_id": 777, "label": "audit"})
    monkeypatch.setattr(rental, "wait_for_running", lambda *a, **k: {"id": 777, "actual_status": "running"})
    runner = Mock()
    monkeypatch.setattr("vast_comfy.instances.runner_for", lambda *_: runner)
    failure = VastComfyError(error_code, "not ready")
    wait = Mock(side_effect=failure); monkeypatch.setattr(rental, "wait_for_ssh", wait)
    with pytest.raises(VastComfyError) as error:
        api.rent_create(cfg, alias="rtx_4060_ti_16gb", offer_id=1001,
                        label="audit", confirm_paid=True, client=Mock())
    assert error.value is failure
    assert error.value.details["instance_id"] == 777
    assert error.value.details["instance_may_be_billing"] is True
    wait.assert_called_once_with(runner, cfg)


def test_cli_paid_state_write_failure_keeps_instance_reference(cfg, monkeypatch):
    monkeypatch.setattr(cli, "verify_offer", lambda *_: make_offer())
    monkeypatch.setattr(cli, "create_instance", lambda *a, **k: {"instance_id": 777, "label": "audit"})
    state = Mock(); state.record_instance.side_effect = OSError("disk full")
    ctx = SimpleNamespace(cfg=cfg, client=Mock(), state=state)
    args = SimpleNamespace(alias="rtx_4060_ti_16gb", offer_id=1001, label="audit", confirm_paid=True, wait=True)
    with pytest.raises(VastComfyError) as error:
        cli.cmd_rent_create(ctx, args)
    assert error.value.code == "E_INTERNAL"
    assert error.value.details["instance_id"] == 777
    assert error.value.details["instance_may_be_billing"] is True


def test_agent_only_registered_identity_is_usable(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(ssh.shutil, "which", lambda name: name)
    monkeypatch.setattr(ssh.Path, "home", lambda: tmp_path)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="ssh-ed25519 AAAAagent comment"))
    monkeypatch.setattr(ssh.subprocess, "run", run)
    client = Mock(); client.list_ssh_keys.return_value = [{"ssh_key": "ssh-ed25519 AAAAagent"}]
    assert ssh.rental_preflight(client, cfg)["registered_identity_confirmed"]
    assert run.call_args.args[0] == ["ssh-add", "-L"]
    assert run.call_count == 1


def test_explicit_identity_does_not_fall_back_to_unrelated_agent(cfg, monkeypatch, tmp_path):
    key = tmp_path / "key"; key.write_text("synthetic")
    cfg = load_config(set_overrides={"ssh.key_path": str(key)})
    monkeypatch.setattr(ssh.shutil, "which", lambda name: name)
    monkeypatch.setattr(ssh.subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=0 if argv[0] == "ssh-add" else 1, stdout="ssh-ed25519 AAAAagent"))
    with pytest.raises(VastComfyError) as error:
        ssh.rental_preflight(Mock(), cfg)
    assert error.value.code == "E_SSH_AUTH"


def test_ssh_wait_is_bounded_and_retries_only_connect_errors(cfg):
    time = [0.0]
    def sleep(seconds): time[0] += seconds
    runner = Mock(); runner.check_login.side_effect = [VastComfyError("E_SSH_CONNECT", "starting"), {"ssh_ok": True}]
    assert ssh.wait_for_ssh(runner, cfg, clock=lambda: time[0], sleep=sleep)["ssh_ok"]
    assert runner.check_login.call_count == 2
    runner = Mock(); runner.check_login.side_effect = VastComfyError("E_SSH_AUTH", "rejected")
    with pytest.raises(VastComfyError):
        ssh.wait_for_ssh(runner, cfg, clock=lambda: time[0], sleep=sleep)
    assert runner.check_login.call_count == 1


@pytest.mark.parametrize("payload", [None, {}, [], "idle", {"queue_running": []},
    {"queue_running": None, "queue_pending": []}, {"queue_running": [], "queue_pending": {}},
    {"queue_running": "", "queue_pending": []}])
def test_unknown_queue_never_permits_restart(remote_agent, monkeypatch, payload):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    monkeypatch.setattr(module, "http_json", lambda *a, **k: payload)
    monkeypatch.setattr(module, "service_state", lambda _: {"running": True, "raw": "comfyui RUNNING"})
    restart = Mock(); monkeypatch.setattr(module.subprocess, "call", restart)
    with pytest.raises(module.C.RemoteFail) as error:
        module.restart_service(SimpleNamespace(session_id="S", service="comfyui", comfy_url="unused", wait_seconds=1))
    assert error.value.code == "E_COMFY_UNAVAILABLE"
    restart.assert_not_called()


@pytest.mark.parametrize("command", ["status", "restart", "logs"])
def test_service_configuration_is_forwarded(cfg, monkeypatch, command):
    cfg = load_config(set_overrides={"image.supervisor_service": "my-comfy"})
    invoke = Mock(return_value={"ok": True}); monkeypatch.setattr(comfy, "invoke", invoke)
    getattr(comfy, command)(Mock(), cfg, "S")
    argv = invoke.call_args.args[3]
    assert argv[argv.index("--service") + 1] == "my-comfy"


def test_cli_release_uses_config_unless_explicit_keep(cfg, monkeypatch):
    cfg = load_config(set_overrides={"session.release_on_success": False})
    ctx = SimpleNamespace(cfg=cfg, instance_and_runner=lambda *_: ({}, Mock()), state=Mock())
    backend = Mock(return_value={"session_id": "S", "job_id": "J"})
    monkeypatch.setattr(prepare, "prepare_existing", backend)
    args = cli.build_parser().parse_args(["flow", "prepare", "--instance", "123"])
    cli.cmd_flow_prepare(ctx, args)
    assert backend.call_args.kwargs["release_on_success"] is None
    args.keep_session = True
    cli.cmd_flow_prepare(ctx, args)
    assert backend.call_args.kwargs["release_on_success"] is False


def test_prepare_does_not_release_when_config_says_keep(cfg, monkeypatch):
    cfg = load_config(set_overrides={"session.release_on_success": False})
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "S"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "J"})
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(prepare.remote, "read_remote_json", lambda *a: {"verdict": "READY"})
    release = Mock(); monkeypatch.setattr(prepare.locks, "release", release)
    assert prepare.prepare_existing(Mock(), cfg, instance_id=123)["ok"]
    release.assert_not_called()


def test_interrupted_paid_request_is_uncertain_and_never_retried(cfg, monkeypatch):
    monkeypatch.setattr(rental, "rental_preflight", lambda *_: {})
    client = Mock(); client.create_instance.side_effect = KeyboardInterrupt
    with pytest.raises(VastComfyError) as error:
        rental.create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)
    assert error.value.code == "E_CREATE_UNCERTAIN"
    assert error.value.details["instance_may_be_billing"] is True
    assert error.value.details["cause"] == {"error_type": "KeyboardInterrupt"}
    client.create_instance.assert_called_once()


def test_interrupted_post_create_keeps_billing_handoff():
    from vast_comfy.vast.rental import created_resource

    with pytest.raises(VastComfyError) as error:
        with created_resource({"instance_id": 777, "label": "audit"}):
            raise KeyboardInterrupt
    details = error.value.details
    assert details["instance_id"] == 777
    assert details["instance_may_be_billing"] is True
    assert "NOT stopped or destroyed" in details["instance_billing"]


def test_comfy_health_uses_the_configured_health_timeout(cfg, monkeypatch):
    cfg = load_config(set_overrides={"comfy.health_timeout_seconds": 45})
    invoke = Mock(return_value={"ok": True}); monkeypatch.setattr(comfy, "invoke", invoke)
    comfy.health(Mock(), cfg, "S")
    assert invoke.call_args.kwargs["timeout_seconds"] == 45


def test_prepare_failure_after_lock_carries_handoff_details(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "auto-sid"})

    def boom(*a, **k):
        raise VastComfyError("E_SSH_COMMAND", "deploy failed")

    monkeypatch.setattr(prepare.remote, "deploy_package", boom)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=4242, run_id="run-handoff")
    details = error.value.details
    assert error.value.stage == "prepare"
    assert details["session_id"] == "auto-sid"
    assert details["run_id"] == "run-handoff"
    assert "still held" in details["lock_state"]
    assert "continues to run" in details["instance_billing"]


def test_prepare_wait_failure_carries_the_job_id(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "auto-sid"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "job-77"})

    def timeout(*a, **k):
        raise VastComfyError("E_NETWORK", "status poll failed")

    monkeypatch.setattr(prepare.jobs, "wait", timeout)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=4242, run_id="run-wait")
    assert error.value.details["job_id"] == "job-77"
    assert error.value.details["session_id"] == "auto-sid"


def test_explicit_release_failure_is_not_reported_as_overall_success(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "S"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "J"})
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(prepare.remote, "read_remote_json", lambda *a: {"verdict": "READY"})

    def boom(*a, **k):
        raise VastComfyError("E_LOCK_JOBS_RUNNING", "jobs still alive")

    monkeypatch.setattr(prepare.locks, "release", boom)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=123, release_on_success=True)
    details = error.value.details
    assert error.value.code == "E_LOCK_JOBS_RUNNING"
    assert details["verdict"] == "READY"
    assert details["release_error"]["code"] == "E_LOCK_JOBS_RUNNING"
    assert details["release_outcome"] == "not_released"
    assert "NOT released" in details["lock_state"]


def test_unknown_release_outcome_is_not_reported_as_still_held(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "S"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "J"})
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(prepare.remote, "read_remote_json", lambda *a: {"verdict": "READY"})

    def lost_reply(*a, **k):
        raise VastComfyError("E_NETWORK", "release reply lost after completion")

    monkeypatch.setattr(prepare.locks, "release", lost_reply)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=123, release_on_success=True)
    details = error.value.details
    assert details["release_outcome"] == "unknown"
    assert "UNKNOWN" in details["lock_state"]
    assert details["verdict"] == "READY"
    assert any("session status" in hint for hint in error.value.hints)


def test_release_not_held_is_reported_distinctly(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "S"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "J"})
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(prepare.remote, "read_remote_json", lambda *a: {"verdict": "READY"})

    def other_owner(*a, **k):
        raise VastComfyError("E_LOCK_NOT_HELD", "This session does not own the remote lock",
                             details={"owner": {"session_id": "other"}})

    monkeypatch.setattr(prepare.locks, "release", other_owner)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=123, release_on_success=True)
    details = error.value.details
    assert details["release_outcome"] == "not_held"
    assert "no longer owns" in details["lock_state"]
    assert details["owner"] == {"session_id": "other"}


def test_prepare_summary_read_failure_keeps_handoff_details(cfg, monkeypatch):
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "auto-sid"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    monkeypatch.setattr(prepare.jobs, "start", lambda *a, **k: {"job_id": "job-77"})
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})

    def lost(*a, **k):
        raise VastComfyError("E_SSH_CONNECT", "summary response lost")

    monkeypatch.setattr(prepare.remote, "read_remote_json", lost)
    with pytest.raises(VastComfyError) as error:
        prepare.prepare_existing(Mock(), cfg, instance_id=123, run_id="run-summary")
    details = error.value.details
    assert error.value.code == "E_SSH_CONNECT"
    assert details["session_id"] == "auto-sid"
    assert details["job_id"] == "job-77"
    assert details["run_id"] == "run-summary"
    assert "still held" in details["lock_state"]
    assert "continues to run" in details["instance_billing"]


def test_prepare_dispatches_the_configured_install_stage_timeout(cfg, monkeypatch):
    cfg = load_config(set_overrides={"jobs.install_timeout_minutes": 2})
    monkeypatch.setattr(prepare.locks, "acquire", lambda *a, **k: {"session_id": "S"})
    monkeypatch.setattr(prepare.remote, "deploy_package", lambda *a, **k: {})
    start = Mock(return_value={"job_id": "J"})
    monkeypatch.setattr(prepare.jobs, "start", start)
    monkeypatch.setattr(prepare.jobs, "wait", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(prepare.remote, "read_remote_json", lambda *a: {"verdict": "READY"})

    prepare.prepare_existing(Mock(), cfg, instance_id=123, release_on_success=False,
                             run_id="run-install-cap")
    command = start.call_args.args[4]
    assert float(command[command.index("--install-timeout-seconds") + 1]) == 120.0
