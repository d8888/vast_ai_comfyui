"""SPEC 17 item 8: proxy-only SSH, host keys, tunnels, failure classification."""

from __future__ import annotations

import subprocess

import pytest

from vast_comfy.config import load_config
from vast_comfy.errors import VastComfyError
from vast_comfy.ssh import (
    SSHContext,
    SSHRunner,
    classify_ssh_failure,
    context_from_instance,
    scp_base_args,
    ssh_base_args,
)
from vast_comfy.vast.client import scrub


def make_ctx(tmp_path, *, key_path=None) -> SSHContext:
    return SSHContext(
        host="ssh5.vast.ai",
        port=41234,
        user="root",
        key_path=key_path,
        known_hosts_path=tmp_path / "known_hosts",
        connect_timeout_seconds=15,
        command_timeout_seconds=120,
    )


# ---------------------------------------------------------------------------
# classify_ssh_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr, code",
    [
        ("Host key verification failed.", "E_SSH_HOSTKEY"),
        ("@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@", "E_SSH_HOSTKEY"),
        ("root@ssh5.vast.ai: Permission denied (publickey).", "E_SSH_AUTH"),
        ("Received disconnect: Too many authentication failures", "E_SSH_AUTH"),
        ("ssh: connect to host ssh5.vast.ai port 1: Connection refused", "E_SSH_CONNECT"),
        ("ssh: connect to host x port 1: Connection timed out", "E_SSH_CONNECT"),
        ("Could not resolve hostname ssh5.vast.ai", "E_SSH_CONNECT"),
        ("some totally different failure", "E_SSH_COMMAND"),
    ],
)
def test_classify_ssh_failure(stderr, code):
    assert classify_ssh_failure(stderr, 255) == code


# ---------------------------------------------------------------------------
# Base argv: proxy endpoint only, dedicated known_hosts, TOFU
# ---------------------------------------------------------------------------


def test_ssh_base_args_without_key(tmp_path):
    ctx = make_ctx(tmp_path)
    args = ssh_base_args(ctx)
    assert args[0] == "ssh"
    assert args[args.index("-p") + 1] == "41234"
    assert f"UserKnownHostsFile={ctx.known_hosts_path}" in args
    assert "StrictHostKeyChecking=accept-new" in args  # TOFU, never disabled
    assert "BatchMode=yes" in args  # keys / agent only, no prompts
    assert "ConnectTimeout=15" in args
    assert args[-1] == "root@ssh5.vast.ai"
    assert "-i" not in args
    assert "IdentitiesOnly=yes" not in args
    # Proxy-only: no direct port or alternative host is ever mentioned.
    assert "22" not in args
    assert "41235" not in args


def test_ssh_base_args_with_key_locks_identity(tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("PRIVATE KEY", encoding="utf-8")
    ctx = make_ctx(tmp_path, key_path=key)
    args = ssh_base_args(ctx)
    assert args[args.index("-i") + 1] == str(key)
    assert "IdentitiesOnly=yes" in args

    scp_args = scp_base_args(ctx)
    assert scp_args[0] == "scp"
    assert scp_args[scp_args.index("-P") + 1] == "41234"
    assert f"UserKnownHostsFile={ctx.known_hosts_path}" in scp_args


def test_ssh_context_describe_has_no_key_material(tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("PRIVATE KEY", encoding="utf-8")
    ctx = make_ctx(tmp_path, key_path=key)
    described = ctx.describe()
    assert described["strategy"] == "proxy_only"
    assert "PRIVATE KEY" not in str(described)


# ---------------------------------------------------------------------------
# context_from_instance
# ---------------------------------------------------------------------------


def test_context_from_instance_requires_proxy_endpoint(tmp_path):
    cfg = load_config(None, {"paths.root": str(tmp_path)})
    with pytest.raises(VastComfyError) as excinfo:
        context_from_instance({"id": 7}, cfg)
    assert excinfo.value.code == "E_SSH_INFO_MISSING"
    assert "proxy-only" in str(excinfo.value.hints).lower() or "ssh_proxy" in str(
        excinfo.value.hints
    )

    with pytest.raises(VastComfyError) as excinfo2:
        context_from_instance({"id": 7, "ssh_host": "ssh5.vast.ai"}, cfg)
    assert excinfo2.value.code == "E_SSH_INFO_MISSING"


def test_context_from_instance_resolves_paths(tmp_path):
    cfg = load_config(None, {"paths.root": str(tmp_path)})
    ctx = context_from_instance(
        {"id": 7, "ssh_host": "ssh5.vast.ai", "ssh_port": 41234, "public_ipaddr": "1.2.3.4"}, cfg
    )
    assert ctx.host == "ssh5.vast.ai"
    assert ctx.port == 41234
    assert ctx.user == "root"
    assert ctx.known_hosts_path == tmp_path / "known_hosts"
    assert ctx.known_hosts_path.parent.is_dir()  # prepared for TOFU
    assert ctx.key_path is None
    assert ctx.connect_timeout_seconds == 15


def test_context_from_instance_missing_key_file_fails(tmp_path):
    cfg = load_config(None, {"paths.root": str(tmp_path), "ssh.key_path": str(tmp_path / "nope")})
    with pytest.raises(VastComfyError) as excinfo:
        context_from_instance({"id": 7, "ssh_host": "ssh5.vast.ai", "ssh_port": 1}, cfg)
    assert excinfo.value.code == "E_KEY_MISSING"


# ---------------------------------------------------------------------------
# tunnel / execution failures
# ---------------------------------------------------------------------------


def test_tunnel_binds_localhost_only(tmp_path):
    ctx = make_ctx(tmp_path)
    argv = SSHRunner(ctx).tunnel_argv(8188, 18188)
    assert argv[0] == "ssh"
    assert "-N" in argv
    assert "-L" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:8188:localhost:18188"
    # OpenSSH treats everything after the destination as a remote command, so
    # the destination host must come last for the forward to actually happen.
    assert argv[-1] == "root@ssh5.vast.ai"


def test_execute_timeout_reports_command_may_still_be_running(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)

    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", fake_run)
    with pytest.raises(VastComfyError) as excinfo:
        runner.run_remote(["true"], timeout_seconds=5)
    error = excinfo.value
    assert error.code == "E_SSH_COMMAND"
    assert "may still be running" in " ".join(error.hints).lower()
    assert error.details["timeout_seconds"] == 5


def test_execute_nonzero_rc_classified(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)

    class Completed:
        returncode = 255
        stdout = b""
        stderr = b"Permission denied (publickey)."

    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", lambda *a, **k: Completed())
    with pytest.raises(VastComfyError) as excinfo:
        runner.run_remote(["true"])
    assert excinfo.value.code == "E_SSH_AUTH"
    assert excinfo.value.details["endpoint"] == "root@ssh5.vast.ai:41234"


def test_execute_dry_run_does_not_spawn(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx, dry_run=True)

    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", boom)
    result = runner.run_remote(["echo", "hi"])
    assert result.rc == 0


# ---------------------------------------------------------------------------
# scp protocol selection: never hand shell-metacharacter paths to the remote shell
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, rc: int, stderr: bytes | str = b"") -> None:
        self.returncode = rc
        self.stdout = b""
        self.stderr = stderr


_SFTP_FAILURE = b"subsystem request failed on channel 0\nscp: sftp unavailable"
_MODERN_PROBE = ("usage: scp [-346ABCOpqRrsTv] [-c cipher] [-D sftp_server_path] "
                 "[-F ssh_config]\n           [-i identity_file] source ... target\n")
_OLD_PROBE = ("scp: unknown option -- O\n"
              "usage: scp [-12346BCEpqrv] [-c cipher] [-F ssh_config] source ... target\n")


def _scp_fake(probe_output: str, replies: list[_Completed]):
    """subprocess.run stand-in: answers the `scp -O` probe, then scripted transfers."""
    calls: list[list[str]] = []
    pending = list(replies)

    def fake_run(argv, **kwargs):
        argv = list(argv)
        calls.append(argv)
        if argv[:2] == ["scp", "-O"]:
            return _Completed(1, stderr=probe_output)  # text mode, like the real call
        assert pending, "unexpected extra scp attempt"
        return pending.pop(0)

    def scp_calls() -> list[list[str]]:
        return [call for call in calls if call[:1] == ["scp"] and call[:2] != ["scp", "-O"]]

    return fake_run, calls, scp_calls


def test_modern_scp_falls_back_to_legacy_only_for_safe_paths(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)
    fake_run, calls, scp_calls = _scp_fake(
        _MODERN_PROBE, [_Completed(1, stderr=_SFTP_FAILURE), _Completed(0)]
    )
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", fake_run)
    runner.download("/workspace/ComfyUI/output/vast_probe.png", str(tmp_path / "local.bin"))
    scp = scp_calls()
    assert len(scp) == 2
    assert "-O" not in scp[0] and "-O" in scp[1]
    assert "root@ssh5.vast.ai:/workspace/ComfyUI/output/vast_probe.png" in scp[1]


def test_modern_scp_refuses_metacharacter_paths_before_the_legacy_retry(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)
    fake_run, calls, scp_calls = _scp_fake(_MODERN_PROBE, [_Completed(1, stderr=_SFTP_FAILURE)])
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", fake_run)
    with pytest.raises(VastComfyError) as excinfo:
        runner.download("/workspace/ComfyUI/output/probe;id.png", str(tmp_path / "local.bin"))
    error = excinfo.value
    assert error.code == "E_SSH_COMMAND"
    assert "legacy" in error.message.lower()
    assert error.details["scp_uses_sftp_by_default"] is True
    # The SFTP attempt may run (its path handling is safe), but -O is refused.
    assert len(scp_calls()) == 1
    assert all("-O" not in call for call in scp_calls())


@pytest.mark.parametrize("probe", [_OLD_PROBE, "unexpected noise without any marker"])
def test_legacy_scp_never_sends_metacharacter_paths_at_all(tmp_path, monkeypatch, probe):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)
    fake_run, calls, scp_calls = _scp_fake(probe, [])
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", fake_run)
    with pytest.raises(VastComfyError) as excinfo:
        runner.download("/workspace/ComfyUI/output/probe;id.png", str(tmp_path / "local.bin"))
    assert excinfo.value.code == "E_SSH_COMMAND"
    assert excinfo.value.details["scp_uses_sftp_by_default"] is False
    assert scp_calls() == []  # refused before any scp process was spawned


def test_legacy_scp_transfers_safe_paths_without_the_legacy_flag(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx)
    fake_run, calls, scp_calls = _scp_fake(_OLD_PROBE, [_Completed(0)])
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run", fake_run)
    runner.download("/workspace/ComfyUI/output/vast_probe.png", str(tmp_path / "local.bin"))
    scp = scp_calls()
    assert len(scp) == 1 and "-O" not in scp[0]


def test_dry_run_scp_never_probes_the_scp_binary(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    runner = SSHRunner(ctx, dry_run=True)
    calls: list[list[str]] = []
    monkeypatch.setattr("vast_comfy.ssh.subprocess.run",
                        lambda argv, **kw: calls.append(list(argv)) or _Completed(0))
    runner.download("/workspace/ComfyUI/output/probe;id.png", str(tmp_path / "local.bin"))
    assert calls == []


# ---------------------------------------------------------------------------
# Secret scrubbing
# ---------------------------------------------------------------------------


def test_scrub_hf_token():
    token = "hf_" + "abcdefghij1234567890"
    scrubbed = scrub(f"downloading with token {token} from huggingface")
    assert token not in scrubbed
    assert "<redacted>" in scrubbed


def test_scrub_empty_text_is_safe():
    assert scrub("") == ""
    assert scrub(None) is None
