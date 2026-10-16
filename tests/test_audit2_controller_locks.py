"""Per-thread delegation and uncertain platform mutations; no real SSH/API."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vast_comfy import lifecycle, locks, remote
from vast_comfy.errors import VastComfyError


@pytest.fixture
def protocol(monkeypatch):
    invoke = Mock(return_value={"ok": True})
    monkeypatch.setattr(locks, "ensure_bootstrap", Mock())
    monkeypatch.setattr(locks, "invoke", invoke)
    monkeypatch.setattr(locks, "assert_owns", Mock(return_value={}))
    return invoke


def test_shared_runner_cannot_overlap_across_threads(cfg, protocol):
    runner = SimpleNamespace()
    errors, entered, environments = [], [], []
    with locks.operation(runner, cfg, "S") as lease:
        assert locks.operation_token(runner) == lease.token
        def other():
            environments.append(remote.remote_environment(cfg, runner))
            try:
                with locks.operation(runner, cfg, "S"):
                    entered.append(True)
            except VastComfyError as exc:
                errors.append(exc.code)
        thread = threading.Thread(target=other)
        thread.start(); thread.join(5)
        assert not thread.is_alive()
        assert entered == [] and errors == ["E_LOCK_CONFLICT"]
        assert not any("VAST_COMFY_OPERATION_TOKEN=" in item for item in environments[0])
        assert any(f"VAST_COMFY_OPERATION_TOKEN={lease.token}" == item
                   for item in remote.remote_environment(cfg, runner))
    assert [c.args[3][0] for c in protocol.call_args_list] == ["begin-operation", "end-operation"]
    assert locks.operation_token(runner) is None


@pytest.mark.parametrize("retain", [True, False])
def test_true_nested_context_uses_shared_handle(cfg, protocol, retain):
    runner = SimpleNamespace()
    with locks.operation(runner, cfg, "S") as outer:
        with locks.operation(runner, cfg, "S") as inner:
            assert inner is outer
            if retain: inner.retain()
    expected = ["begin-operation"] if retain else ["begin-operation", "end-operation"]
    assert [c.args[3][0] for c in protocol.call_args_list] == expected
    assert locks.operation_token(runner) is None


def test_nested_different_session_is_not_delegated(cfg, protocol):
    runner = SimpleNamespace()
    with locks.operation(runner, cfg, "S"):
        with pytest.raises(VastComfyError) as error:
            with locks.operation(runner, cfg, "OTHER"):
                pytest.fail("different session must not enter")
        assert error.value.code == "E_LOCK_NOT_HELD"


def test_failed_begin_releases_local_guard_for_diagnosis(cfg, protocol):
    runner = SimpleNamespace()
    protocol.side_effect = [VastComfyError("E_LOCK_CONFLICT", "busy"), {"ok": True}, {"ok": True}]
    with pytest.raises(VastComfyError):
        with locks.operation(runner, cfg, "S"): pass
    with locks.operation(runner, cfg, "S"): pass


@pytest.mark.parametrize("action", ["stop", "destroy"])
@pytest.mark.parametrize("failure", [
    VastComfyError("E_NETWORK", "timeout after acceptance"),
    VastComfyError("E_UPSTREAM", "server error", details={"status": 503}),
    VastComfyError("E_UPSTREAM", "invalid response"),
    KeyboardInterrupt(),
    ValueError("unexpected response parser error"),
])
def test_uncertain_platform_mutation_never_ends_operation(cfg, protocol, monkeypatch, action, failure):
    runner = SimpleNamespace()
    monkeypatch.setattr(lifecycle, "load_runner", lambda *_: ({}, runner))
    client = Mock()
    getattr(client, action + "_instance").side_effect = failure
    with pytest.raises(VastComfyError) as error:
        if action == "stop": lifecycle.stop(client, cfg, 123, "S")
        else: lifecycle.destroy(client, cfg, 123, "S", confirm_destroy=True)
    assert [c.args[3][0] for c in protocol.call_args_list] == ["begin-operation"]
    assert error.value.details["instance_id"] == 123
    assert error.value.details["session_id"] == "S"
    assert error.value.details["action_outcome"] == "unknown"
    assert "retained" in error.value.details["lock_state"]
    getattr(client, action + "_instance").assert_called_once_with(123)


@pytest.mark.parametrize("action", ["stop", "destroy"])
def test_successful_mutation_also_retains_exclusion(cfg, protocol, monkeypatch, action):
    monkeypatch.setattr(lifecycle, "load_runner", lambda *_: ({}, SimpleNamespace()))
    client = Mock()
    if action == "stop": result = lifecycle.stop(client, cfg, 123, "S")
    else: result = lifecycle.destroy(client, cfg, 123, "S", confirm_destroy=True)
    assert result[action + "_requested"] is True
    assert [c.args[3][0] for c in protocol.call_args_list] == ["begin-operation"]
