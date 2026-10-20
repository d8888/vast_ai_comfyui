"""Regression evidence for each safety finding; no paid API or real SSH."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from offer_factory import make_offer
from mocks.fake_ssh import FakeRunner
from vast_comfy import lifecycle, locks, prepare, remote
from vast_comfy.config import load_config
from vast_comfy.errors import VastComfyError
from vast_comfy.ssh import rental_preflight
from vast_comfy.vast.client import VastClient
from vast_comfy.vast.offers import evaluate_offer
from vast_comfy.vast.rental import create_instance


@pytest.mark.parametrize("field", ["dph_total", "inet_up", "inet_down", "gpu_ram", "reliability", "gpu_frac", "disk_space"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_offer_never_qualifies(cfg, mapping, field, value):
    reasons = evaluate_offer(make_offer(**{field: value}), cfg, mapping.resolve("rtx_4060_ti_16gb"))
    assert reasons


@pytest.mark.parametrize("field", ["pricing.max_dph_total_usd_per_hour", "network.inet_up_min", "gpu.vram_gb_min"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_configuration_rejected(field, value):
    with pytest.raises(VastComfyError, match="finite"):
        load_config(set_overrides={field: value})


@pytest.mark.parametrize("changes", [{"num_gpus": 1.9}, {"rentable": None}, {"rented": None}, {"disk_space": 299}])
def test_incomplete_or_fractional_hard_conditions_rejected(cfg, mapping, changes):
    assert evaluate_offer(make_offer(**changes), cfg, mapping.resolve("rtx_4060_ti_16gb"))


def test_common_real_cli_entrypoint(repo_root):
    process = subprocess.run([sys.executable, "-B", str(repo_root / "src/vast_comfy/remote_agent/vc_common.py"),
                              "instance-time"], capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["ok"] is True


@pytest.mark.parametrize("action", ["stop", "destroy"])
def test_lifecycle_conflict_prevents_platform_mutation(cfg, monkeypatch, action):
    client = Mock()
    monkeypatch.setattr(lifecycle, "load_runner", lambda *_: ({"actual_status": "running"}, Mock()))
    monkeypatch.setattr(lifecycle, "live_instance", lambda *_: {"actual_status": "running"})
    @contextlib.contextmanager
    def denied(*a, **kw):
        raise VastComfyError("E_LOCK_CONFLICT", "another operation")
        yield
    monkeypatch.setattr(locks, "operation", denied)
    with pytest.raises(VastComfyError) as exc:
        if action == "stop":
            lifecycle.stop(client, cfg, 123, "S")
        else:
            lifecycle.destroy(client, cfg, 123, "S", confirm_destroy=True)
    assert exc.value.code == "E_LOCK_CONFLICT"
    client.stop_instance.assert_not_called()
    client.destroy_instance.assert_not_called()


@pytest.mark.parametrize("action", ["stop", "destroy"])
@pytest.mark.parametrize("payload", [{"success": False}, {}, None])
def test_rejected_platform_mutation_is_not_success(action, payload):
    response = Mock(status_code=200, headers={"Content-Type": "application/json"})
    response.json.return_value = payload
    session = Mock(); session.request.return_value = response
    client = VastClient("http://unused.invalid", "synthetic", session=session)
    with pytest.raises(VastComfyError) as exc:
        getattr(client, action + "_instance")(123)
    assert exc.value.code == "E_UPSTREAM"


def test_missing_private_key_fails_before_paid_call(cfg, tmp_path, monkeypatch):
    cfg = load_config(set_overrides={"ssh.key_path": str(tmp_path / "missing")})
    monkeypatch.setattr("vast_comfy.ssh.shutil.which", lambda x: x)
    client = Mock()
    with pytest.raises(VastComfyError) as exc:
        create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)
    assert exc.value.code == "E_KEY_MISSING"
    client.create_instance.assert_not_called()


def test_unregistered_identity_fails_before_paid_call(tmp_path, monkeypatch):
    key = tmp_path / "id_rsa"; key.write_text("placeholder")
    cfg = load_config(set_overrides={"ssh.key_path": str(key)})
    monkeypatch.setattr("vast_comfy.ssh.shutil.which", lambda x: x)
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", lambda argv, **kw:
                        SimpleNamespace(returncode=1, stdout="") if argv[0] == "ssh-add" else
                        SimpleNamespace(returncode=0, stdout="ssh-rsa AAAAexamplepublic comment"))
    client = Mock(); client.list_ssh_keys.return_value = [{"ssh_key": "ssh-rsa OTHER"}]
    with pytest.raises(VastComfyError) as exc:
        create_instance(client, cfg, make_offer(), "audit", paid_confirmed=True)
    assert exc.value.code == "E_SSH_AUTH"
    client.create_instance.assert_not_called()


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_error_create_is_uncertain_without_retry(cfg, monkeypatch, status):
    monkeypatch.setattr("vast_comfy.vast.rental.rental_preflight", lambda *_: {})
    response = Mock(status_code=status, headers={"Content-Type": "application/json"}, text="server error")
    session = Mock(); session.request.return_value = response
    client = VastClient("http://unused.invalid", "synthetic", session=session)
    with pytest.raises(VastComfyError) as exc:
        create_instance(client, cfg, make_offer(), "audit-label", paid_confirmed=True)
    assert exc.value.code == "E_CREATE_UNCERTAIN"
    assert exc.value.details["label"] == "audit-label"
    assert exc.value.details["instance_may_be_billing"]
    assert session.request.call_count == 1


def test_conflict_does_not_deploy_full_package(cfg, monkeypatch):
    deployed = Mock(); monkeypatch.setattr(remote, "deploy_package", deployed)
    def conflict(*a, **kw):
        raise VastComfyError("E_LOCK_CONFLICT", "owned")
    monkeypatch.setattr(locks, "acquire", conflict)
    with pytest.raises(VastComfyError):
        prepare.prepare_existing(Mock(), cfg, instance_id=123)
    deployed.assert_not_called()


def test_windows_remote_paths_remain_posix(cfg, monkeypatch):
    monkeypatch.setattr(locks, "operation", lambda *a, **kw: contextlib.nullcontext())
    runner = FakeRunner()
    runner.add_rule("mkdir")
    runner.add_rule("current/remote/vc_common.py", stdout="not present", rc=1)
    runner.add_json("vc_common.py", {"ok": True, "checked": len(remote.build_package().records),
                                     "manifest_sha256": hashlib.sha256(
                                         remote.build_package().manifest_text.encode("utf-8")).hexdigest()})
    runner.add_rule("ln -s")
    runner.add_rule("mv -T")
    remote.deploy_package(runner, cfg, session_id="S")
    assert runner.uploads
    for call in runner.calls_containing("mkdir"):
        assert not any("\\" in argument for argument in call.argv)


def test_bootstrap_uploads_only_lock_runtime(cfg):
    class BootstrapRunner:
        def __init__(self): self.calls, self.uploads = [], []
        def run_remote(self, argv, **kw):
            self.calls.append(argv)
            return SimpleNamespace(rc=1 if len(self.calls) == 1 else 0)
        def upload(self, source, target): self.uploads.append((source, target))
    runner = BootstrapRunner()
    remote.ensure_bootstrap(runner, cfg)
    assert {Path(str(src)).name for src, _ in runner.uploads} == {"vc_common.py", "vc_session.py"}
    assert all("/bootstrap/" in target for _, target in runner.uploads)
    assert not any("current" in str(call) for call in runner.calls)


def test_bootstrap_whitelist_does_not_glob_user_files(monkeypatch):
    monkeypatch.setattr(Path, "glob", Mock(side_effect=AssertionError("must use explicit filenames")))
    monkeypatch.setattr(Path, "iterdir", Mock(side_effect=AssertionError("must not scan arbitrary assets")))
    assert len(remote.package_files()) == 19


def test_installer_never_logs_authorization(remote_agent, monkeypatch, capsys):
    installer = importlib.import_module("vc_install")
    monkeypatch.setenv("AUDIT_TOKEN", "synthetic-sensitive-token")
    monkeypatch.setattr(installer.subprocess, "call", lambda *a, **kw: 0)
    installer._run_checked(["curl", "-H", *installer._curl_headers({"auth_env": "AUDIT_TOKEN"}), "https://unused.invalid"])
    assert "synthetic-sensitive-token" not in capsys.readouterr().out


@pytest.mark.parametrize("state,allowed", [("comfyui STOPPED", True), ("comfyui RUNNING", False), ("unavailable", False)])
def test_cold_start_is_safe_and_running_unknown_queue_is_refused(remote_agent, monkeypatch, state, allowed):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    monkeypatch.setattr(module, "service_state", lambda _: {"running": "RUNNING" in state, "raw": state})
    monkeypatch.setattr(module, "queue_snapshot", Mock(side_effect=ConnectionRefusedError()))
    restart = Mock(return_value=0); monkeypatch.setattr(module.subprocess, "call", restart)
    monkeypatch.setattr(module, "wait_api_up", lambda *a: {"api_ok": True})
    args = SimpleNamespace(session_id="S", comfy_url="http://localhost:18188", service="comfyui", wait_seconds=1)
    if allowed:
        assert module.restart_service(args)["restarted"]
        restart.assert_called_once()
    else:
        with pytest.raises(module.C.RemoteFail): module.restart_service(args)
        restart.assert_not_called()


def test_service_logs_parser_provides_service(remote_agent):
    args = remote_agent.CM.build_parser().parse_args(["logs", "--session-id", "S"])
    assert args.service == "comfyui"
