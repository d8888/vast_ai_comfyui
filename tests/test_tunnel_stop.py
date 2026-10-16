"""Tunnel stop verifies the recorded PID before signalling; no real signals are sent."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vast_comfy import tunnel
from vast_comfy.config import load_config
from vast_comfy.errors import VastComfyError
from vast_comfy.ssh import SSHContext, SSHRunner


def _record(**overrides):
    record = {
        "instance_id": "123",
        "pid": 4321,
        "local_port": 8188,
        "remote_port": 18188,
        "forward": "127.0.0.1:8188:localhost:18188",
        "endpoint": "root@ssh5.vast.ai",
        "url": "http://127.0.0.1:8188/",
        "log": "unused.log",
    }
    record.update(overrides)
    return record


def _state(record):
    return SimpleNamespace(get=lambda *_: record, remove=Mock())


@pytest.mark.parametrize("pid", [0, -1, 1, "4321", 3.5, None, True])
def test_stop_refuses_unsafe_pids_without_any_signal(monkeypatch, pid):
    state = _state(_record(pid=pid))
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    with pytest.raises(VastComfyError) as error:
        tunnel.stop_tunnel("123", state)
    assert error.value.stage == "tunnel"
    kill.assert_not_called()
    state.remove.assert_not_called()


def test_stop_refuses_its_own_pid(monkeypatch):
    state = _state(_record(pid=os.getpid()))
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    with pytest.raises(VastComfyError):
        tunnel.stop_tunnel("123", state)
    kill.assert_not_called()
    state.remove.assert_not_called()


def test_stop_refuses_legacy_records_without_identity_fields(monkeypatch):
    state = _state({"instance_id": "123", "pid": 4321, "local_port": 8188})
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    with pytest.raises(VastComfyError):
        tunnel.stop_tunnel("123", state)
    kill.assert_not_called()
    state.remove.assert_not_called()


def test_stop_signals_only_a_verified_tunnel_process(monkeypatch):
    state = _state(_record())
    monkeypatch.setattr(
        tunnel, "_process_command_text",
        lambda pid: ("ssh -p 41234 -o UserKnownHostsFile=x -N "
                     "-L 127.0.0.1:8188:localhost:18188 root@ssh5.vast.ai", True),
    )
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    result = tunnel.stop_tunnel("123", state)
    kill.assert_called_once_with(4321, tunnel.signal.SIGTERM)
    state.remove.assert_called_once_with("tunnels", "123")
    assert result["stopped"] is True and result["pid"] == 4321


@pytest.mark.parametrize("text", [
    "python unrelated-tool.py",  # not ssh at all
    "ssh -N -L 127.0.0.1:9999:localhost:1 root@ssh5.vast.ai",  # a different tunnel
    "ssh -N -L 127.0.0.1:8188:localhost:18188 root@other.host",  # another endpoint
])
def test_stop_refuses_a_reused_pid_even_when_it_exists(monkeypatch, text):
    state = _state(_record())
    monkeypatch.setattr(tunnel, "_process_command_text", lambda pid: (text, True))
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    with pytest.raises(VastComfyError) as error:
        tunnel.stop_tunnel("123", state)
    assert "no longer runs this tunnel" in error.value.message
    kill.assert_not_called()
    state.remove.assert_not_called()


def test_stop_reports_an_already_exited_tunnel_without_signalling(monkeypatch):
    state = _state(_record())
    monkeypatch.setattr(tunnel, "_process_command_text", lambda pid: (None, False))
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    result = tunnel.stop_tunnel("123", state)
    assert result["already_exited"] is True and result["stopped"] is False
    kill.assert_not_called()
    state.remove.assert_called_once_with("tunnels", "123")


def test_stop_refuses_when_identity_cannot_be_read(monkeypatch):
    state = _state(_record())
    monkeypatch.setattr(tunnel, "_process_command_text", lambda pid: (None, True))
    kill = Mock()
    monkeypatch.setattr(tunnel.os, "kill", kill)
    with pytest.raises(VastComfyError) as error:
        tunnel.stop_tunnel("123", state)
    assert "cannot be read" in error.value.message
    kill.assert_not_called()
    state.remove.assert_not_called()


def test_plan_and_start_record_the_identity_fields(tmp_path, monkeypatch):
    cfg = load_config(set_overrides={"paths.root": str(tmp_path)})
    runner = SSHRunner(SSHContext(host="ssh5.vast.ai", port=41234, known_hosts_path=Path("unused")))
    plan = tunnel.tunnel_plan(cfg, runner, local_port=8188)
    assert plan["forward"] == "127.0.0.1:8188:localhost:18188"
    assert plan["endpoint"] == "root@ssh5.vast.ai"

    records: dict = {}
    state = SimpleNamespace(
        put=lambda name, key, record: records.__setitem__((name, key), dict(record)),
    )
    proc = SimpleNamespace(pid=202, poll=lambda: None)
    monkeypatch.setattr(tunnel.time, "sleep", lambda _: None)
    monkeypatch.setattr(tunnel.subprocess, "Popen", lambda *a, **k: proc)
    tunnel.start_tunnel(cfg, runner, instance_id="123", state=state, local_port=8188)
    record = records[("tunnels", "123")]
    assert record["pid"] == 202
    assert record["forward"] == plan["forward"]
    assert record["endpoint"] == plan["endpoint"]
    assert record["remote_port"] == 18188


@pytest.mark.skipif(sys.platform != "linux", reason="reads the real command line via /proc")
def test_process_command_text_reads_a_real_process_via_proc():
    text, exists = tunnel._process_command_text(os.getpid())
    assert exists is True
    assert text and "python" in text.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="reads the real Windows process command line")
def test_process_command_text_reads_a_real_process_on_windows():
    text, exists = tunnel._process_command_text(os.getpid())
    assert exists is True
    assert text and "python" in text.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="reads the real Windows process command line")
def test_process_command_text_reports_a_missing_pid_on_windows():
    _, exists = tunnel._process_command_text(999999991)
    assert exists is False
