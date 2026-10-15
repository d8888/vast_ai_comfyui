"""Audit regressions through real CLI parsing/handlers, with all external I/O mocked."""

from __future__ import annotations

import json
import shlex
import contextlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, mock_open

import pytest

from vast_comfy import cli, download, jobs, locks, remote, smoke, tunnel
from vast_comfy.config import load_config
from vast_comfy.errors import LockConflictError
from vast_comfy.prepare import b64_json
from vast_comfy.ssh import SSHContext, SSHRunner
from vast_comfy.state import StateStore


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    for target in (
        "requests.sessions.Session.request",
        "socket.create_connection",
        "subprocess.run",
        "subprocess.Popen",
        "vast_comfy.cli.VastClient",
        "vast_comfy.ssh.SSHRunner._execute",
        "vast_comfy.ssh.SSHRunner._scp",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError(f"Unexpected external I/O: {target}")))


@pytest.fixture
def cli_env(monkeypatch):
    cfg = load_config()
    records = {}
    # Keep real record_session/get/put logic, replacing only filesystem persistence.
    state = StateStore.__new__(StateStore)
    state.root = Path("mock-state")
    monkeypatch.setattr(state, "_load", lambda name: deepcopy(records.get(name, {})))
    monkeypatch.setattr(state, "_save", lambda name, data: records.__setitem__(name, deepcopy(data)))
    runner = SSHRunner(SSHContext(host="ssh5.vast.ai", port=41234, known_hosts_path=Path("mock-known-hosts")))
    lookup = Mock(return_value=({"id": 123, "actual_status": "running"}, runner))
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(cli, "setup_logging", Mock())
    monkeypatch.setattr(cli.Ctx, "instance_and_runner", lookup)
    monkeypatch.setattr(cli.Ctx, "state", property(lambda self: state))
    @contextlib.contextmanager
    def owned_operation(runner, cfg, sid, **kwargs):
        locks.assert_owns(runner, cfg, sid)
        yield
    monkeypatch.setattr(locks, "operation", owned_operation)
    return SimpleNamespace(cfg=cfg, state=state, records=records, runner=runner, lookup=lookup)


def read_payload(capsys):
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    assert captured.err == ""
    return json.loads(lines[0])


def session_options(env, explicit):
    env.state.record_session("local-session", instance_id="123", state="active")
    return (["--session-id", "explicit-session"], "explicit-session") if explicit else ([], "local-session")


JOB_OPTIONS = ["jobs", "start", "--instance", "123", "--name", "audit", "--timeout-minutes", "2.5"]


def test_jobs_start_parser_keeps_subcommand_separate_from_remote_argv():
    args = cli.build_parser().parse_args([*JOB_OPTIONS, "--", "python", "-c", "print('a b')"])
    assert args.command == "start"
    assert args.job_command == ["--", "python", "-c", "print('a b')"]
    assert args.func is cli.cmd_jobs_start


@pytest.mark.parametrize("separator", [[], ["--"]], ids=["no-separator", "separator"])
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-session", "local-session"])
def test_jobs_start_main_preserves_command_and_records_job(cli_env, monkeypatch, capsys, separator, explicit):
    options, session_id = session_options(cli_env, explicit)
    start = Mock(return_value={"job_id": "job-audit", "state": "running"})
    monkeypatch.setattr(jobs, "start", start)
    command = ["python", "-c", "print('a b')", "--json"]

    rc = cli.main(["--json", *JOB_OPTIONS, *options, *separator, *command])

    assert rc == 0
    payload = read_payload(capsys)
    assert payload["ok"] is True
    assert payload["command"] == "jobs start"
    start.assert_called_once_with(cli_env.runner, cli_env.cfg, session_id, "audit", command, timeout_seconds=150.0)
    record = cli_env.state.get("jobs", "job-audit")
    assert record["instance_id"] == "123"
    assert record["session_id"] == session_id
    assert record["name"] == "audit"


@pytest.mark.parametrize("remainder", [[], ["--"]])
def test_jobs_start_main_without_command_is_structured_usage_error(cli_env, monkeypatch, capsys, remainder):
    start = Mock()
    monkeypatch.setattr(jobs, "start", start)

    rc = cli.main(["--json", *JOB_OPTIONS, "--session-id", "explicit-session", *remainder])

    assert rc == 2
    payload = read_payload(capsys)
    assert payload["command"] == "jobs start"
    assert payload["code"] == "E_USAGE"
    start.assert_not_called()
    assert cli_env.records == {}


REPORT_CASES = [
    pytest.param(
        ["smoke", "report", "--run-id", "run-audit"], smoke, "show_report", ("run-audit",), {},
        id="smoke-report",
    ),
    pytest.param(
        ["download", "report", "--run-id", "run-audit", "--dest", "mock-download", "--overwrite"],
        download, "download_report", (), {"run_id": "run-audit", "dest_dir": "mock-download", "overwrite": True},
        id="download-report",
    ),
    pytest.param(
        ["download", "images", "--run-id", "run-audit", "--dest", "mock-download"],
        download, "download_smoke_images", (), {"run_id": "run-audit", "dest_dir": "mock-download", "overwrite": False},
        id="download-images",
    ),
    pytest.param(
        ["download", "outputs", "--dest", "mock-download", "--pattern", "*.webp", "--limit", "7", "--overwrite"],
        download, "download_outputs", (), {"dest_dir": "mock-download", "pattern": "*.webp", "limit": 7, "overwrite": True},
        id="download-outputs",
    ),
]


@pytest.mark.parametrize("argv,module,function,positional,keywords", REPORT_CASES)
@pytest.mark.parametrize("access", ["explicit", "inferred", "missing", "denied"])
def test_report_and_download_main_use_runner_and_owned_session(
    cli_env, monkeypatch, capsys, argv, module, function, positional, keywords, access,
):
    backend = Mock(return_value={"result": "mocked"})
    owns = Mock()
    monkeypatch.setattr(module, function, backend)
    monkeypatch.setattr(locks, "assert_owns", owns)
    options, session_id = [], None
    if access != "missing":
        options, session_id = session_options(cli_env, access != "inferred")
    else:
        cli_env.state.record_session("released-session", instance_id="123", state="released")
        cli_env.state.record_session("other-instance", instance_id="456", state="active")
    if access == "denied":
        owns.side_effect = LockConflictError("Not the current owner")

    rc = cli.main(["--json", *argv, "--instance", "123", *options])

    payload = read_payload(capsys)
    assert payload["command"] == " ".join(argv[:2])
    cli_env.lookup.assert_called_once_with("123")
    if access == "missing":
        assert rc == 2
        assert payload["code"] == "E_USAGE"
        owns.assert_not_called()
        backend.assert_not_called()
    else:
        owns.assert_called_once_with(cli_env.runner, cli_env.cfg, session_id)
        if access == "denied":
            assert rc == 4
            assert payload["code"] == "E_LOCK_CONFLICT"
            backend.assert_not_called()
        else:
            assert rc == 0
            assert payload["ok"] is True
            backend.assert_called_once_with(cli_env.runner, cli_env.cfg, *positional, **keywords)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("verify_shas", [True, False])
def test_install_verify_main_passes_session_to_remote_parser(cli_env, monkeypatch, capsys, explicit, verify_shas):
    options, session_id = session_options(cli_env, explicit)
    manifest = {"models": [], "plugins": []}
    monkeypatch.setattr(cli, "build_install_manifest", Mock(return_value=manifest))
    owns = Mock()
    invoke = Mock(return_value={"verified": True})
    monkeypatch.setattr(locks, "assert_owns", owns)
    monkeypatch.setattr(remote, "invoke", invoke)
    flags = ["--verify-shas"] if verify_shas else []

    rc = cli.main(["--json", "install", "verify", "--instance", "123", *options, *flags])

    assert rc == 0
    assert read_payload(capsys)["verified"] is True
    owns.assert_called_once_with(cli_env.runner, cli_env.cfg, session_id)
    invoke.assert_called_once_with(
        cli_env.runner, cli_env.cfg, "vc_install.py",
        ["verify", "--session-id", session_id, "--manifest-b64", b64_json(manifest), *flags],
        timeout_seconds=1800,
    )


def test_record_session_persists_identity_and_preserves_fields_on_update(cli_env):
    cli_env.state.record_session("session-audit", instance_id="123", state="active", owner="mock-owner")
    saved = cli_env.records["sessions"]["session-audit"]
    assert saved["session_id"] == "session-audit"
    assert cli_env.state.list_sessions_for_instance(123) == [saved]

    cli_env.state.record_session("session-audit", state="released")

    saved = cli_env.records["sessions"]["session-audit"]
    assert saved["session_id"] == "session-audit"
    assert saved["instance_id"] == "123"
    assert saved["owner"] == "mock-owner"
    assert saved["state"] == "released"


def test_record_session_update_repairs_legacy_record_without_identity(cli_env):
    cli_env.records["sessions"] = {"legacy": {"instance_id": "123", "state": "active"}}
    cli_env.state.record_session("legacy", owner="mock-owner")
    assert cli_env.state.get_session("legacy")["session_id"] == "legacy"


@pytest.fixture
def tunnel_io(cli_env, monkeypatch):
    log_path = Path("mock-logs") / "tunnel-123-9999.log"
    logs_dir = MagicMock(spec=Path)
    logs_dir.__truediv__.return_value = log_path
    monkeypatch.setattr(cli_env.cfg, "resolved_paths", Mock(return_value={"logs_dir": logs_dir}))
    opened = mock_open()
    monkeypatch.setattr(tunnel, "open", opened, raising=False)
    sleep = Mock()
    monkeypatch.setattr(tunnel.time, "sleep", sleep)
    proc = Mock(pid=4321)
    proc.poll.return_value = None
    popen = Mock(return_value=proc)
    monkeypatch.setattr(tunnel.subprocess, "Popen", popen)
    return SimpleNamespace(log_path=log_path, logs_dir=logs_dir, opened=opened, proc=proc, popen=popen, sleep=sleep)


@pytest.mark.parametrize("port", [1, 8188, 65535])
def test_tunnel_plan_main_returns_executable_safe_argv_without_spawning(cli_env, tunnel_io, capsys, port):
    rc = cli.main(["--json", "ssh", "tunnel", "--instance", "123", "--local-port", str(port)])

    assert rc == 0
    payload = read_payload(capsys)
    argv = payload["argv"]
    assert argv[0] == "ssh"
    assert argv[1:3] == ["-o", "ExitOnForwardFailure=yes"]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert argv[argv.index("-L") + 1] == f"127.0.0.1:{port}:localhost:18188"
    assert payload["command"] == shlex.join(argv)
    assert payload["url"] == f"http://127.0.0.1:{port}/"
    assert payload["local_port"] == port
    assert payload["remote_port"] == 18188
    tunnel_io.popen.assert_not_called()
    tunnel_io.opened.assert_not_called()
    tunnel_io.logs_dir.mkdir.assert_not_called()
    assert cli_env.records == {}


@pytest.mark.parametrize("port", [-1, 0, 65536])
@pytest.mark.parametrize("start", [False, True])
def test_tunnel_main_rejects_out_of_range_port_before_start(cli_env, tunnel_io, capsys, port, start):
    flags = ["--start"] if start else []
    rc = cli.main(["--json", "ssh", "tunnel", "--instance", "123", "--local-port", str(port), *flags])

    assert rc == 2
    payload = read_payload(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "E_USAGE"
    assert "65535" in payload["message"]
    tunnel_io.popen.assert_not_called()
    tunnel_io.opened.assert_not_called()
    tunnel_io.logs_dir.mkdir.assert_not_called()
    assert cli_env.records == {}


def test_tunnel_start_main_spawns_exact_planned_argv_and_records_success(cli_env, tunnel_io, capsys):
    rc = cli.main(["--json", "ssh", "tunnel", "--instance", "123", "--local-port", "9999", "--start"])

    assert rc == 0
    payload = read_payload(capsys)
    assert payload["started"] is True
    assert payload["pid"] == 4321
    assert tunnel_io.popen.call_args.args == (payload["argv"],)
    assert "ExitOnForwardFailure=yes" in payload["argv"]
    assert tunnel_io.popen.call_args.kwargs["stdout"] is tunnel_io.opened.return_value
    tunnel_io.opened.assert_called_once_with(tunnel_io.log_path, "ab")
    tunnel_io.opened.return_value.__exit__.assert_called_once()
    tunnel_io.sleep.assert_called_once_with(1.0)
    record = cli_env.state.get("tunnels", "123")
    assert record["pid"] == 4321
    assert record["local_port"] == 9999
    assert record["url"] == "http://127.0.0.1:9999/"


@pytest.mark.parametrize("returncode", [0, 255])
def test_tunnel_start_main_never_records_an_exited_process_as_success(cli_env, tunnel_io, capsys, returncode):
    tunnel_io.proc.poll.return_value = returncode

    rc = cli.main(["--json", "ssh", "tunnel", "--instance", "123", "--local-port", "9999", "--start"])

    assert rc == 1
    payload = read_payload(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "E_SSH_COMMAND"
    assert payload["stage"] == "tunnel"
    assert "started" not in payload
    assert cli_env.records == {}
    tunnel_io.opened.return_value.__exit__.assert_called_once()


def test_tunnel_start_main_reports_spawn_failure_as_json(cli_env, tunnel_io, capsys):
    tunnel_io.popen.side_effect = FileNotFoundError("mock ssh executable missing")

    rc = cli.main(["--json", "ssh", "tunnel", "--instance", "123", "--local-port", "9999", "--start"])

    assert rc == 1
    payload = read_payload(capsys)
    assert payload["ok"] is False
    assert payload["code"] == "E_SSH_COMMAND"
    assert payload["details"]["log"] == str(tunnel_io.log_path)
    assert cli_env.records == {}
    tunnel_io.sleep.assert_not_called()
    tunnel_io.opened.return_value.__exit__.assert_called_once()
