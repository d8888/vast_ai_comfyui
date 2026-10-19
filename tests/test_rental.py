"""SPEC 17 items 6-7: paid creation, startup waiting and offer re-verification."""

from __future__ import annotations

import pytest

from offer_factory import make_offer
from vast_comfy.config import load_config
from vast_comfy.errors import E_NETWORK, ApprovalRequiredError, VastComfyError
from vast_comfy.vast.rental import (
    build_create_body,
    create_instance,
    image_ref,
    verify_offer,
    wait_for_running,
)


@pytest.fixture(autouse=True)
def isolate_local_key_preflight(monkeypatch):
    # This module tests request outcomes; the real local-key preflight rejection
    # is exercised separately in test_audit_safety_regressions (zero paid calls).
    monkeypatch.setattr("vast_comfy.vast.rental.rental_preflight", lambda *_: {})


class StubClient:
    """Minimal VastClient stand-in that records calls."""

    def __init__(self, *, create_result=None, create_error=None, offers=None, instances=None):
        self.create_result = create_result
        self.create_error = create_error
        self.offers = offers if offers is not None else []
        self.instances = instances
        self.calls: list[tuple[str, tuple, dict]] = []
        self.create_calls = 0

    def search_offers(self, query):
        self.calls.append(("search_offers", (query,), {}))
        return list(self.offers)

    def create_instance(self, ask_id, body):
        self.create_calls += 1
        self.calls.append(("create_instance", (ask_id, body), {}))
        if self.create_error is not None:
            raise self.create_error
        return self.create_result

    def show_instance(self, instance_id):
        self.calls.append(("show_instance", (instance_id,), {}))
        if callable(self.instances):
            return self.instances(instance_id, len(self.calls))
        if isinstance(self.instances, list):
            if len(self.instances) > 1:
                return self.instances.pop(0)
            return self.instances[0]  # repeat the last scripted state
        return self.instances


def fake_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def sleep(seconds):
        state["t"] += float(seconds)

    return clock, sleep, state


# ---------------------------------------------------------------------------
# Paid confirmation
# ---------------------------------------------------------------------------


def test_create_without_paid_confirmation_never_calls_the_client():
    client = StubClient(create_result={"success": True, "new_contract": 1})
    cfg = load_config()
    with pytest.raises(ApprovalRequiredError) as excinfo:
        create_instance(client, cfg, make_offer(), "label-x", paid_confirmed=False)
    error = excinfo.value
    assert error.code == "E_APPROVAL_REQUIRED"
    assert error.details["approval_required"] is True
    assert error.details["required_flag"] == "--confirm-paid"
    assert error.details["offer_id"] == 1001
    assert client.create_calls == 0  # no paid request was made


def test_create_network_failure_is_uncertain_and_never_retried():
    client = StubClient(create_error=VastComfyError(code=E_NETWORK, message="connection reset"))
    cfg = load_config()
    with pytest.raises(VastComfyError) as excinfo:
        create_instance(client, cfg, make_offer(id=555), "label-x", paid_confirmed=True)
    error = excinfo.value
    assert error.code == "E_CREATE_UNCERTAIN"
    assert "NOT retried" in error.message
    assert error.details["offer_id"] == 555
    assert client.create_calls == 1  # exactly one attempt — never resent


def test_create_success_returns_new_contract_instance_id():
    client = StubClient(create_result={"success": True, "new_contract": 4242})
    cfg = load_config()
    created = create_instance(client, cfg, make_offer(id=77), "label-x", paid_confirmed=True)
    assert created["instance_id"] == 4242
    assert created["offer_id"] == 77
    assert client.create_calls == 1


def test_create_missing_new_contract_is_uncertain():
    client = StubClient(create_result={"success": True})
    cfg = load_config()
    with pytest.raises(VastComfyError) as excinfo:
        create_instance(client, cfg, make_offer(), "label-x", paid_confirmed=True)
    assert excinfo.value.code == "E_CREATE_UNCERTAIN"
    assert excinfo.value.details["response_keys"] == ["success"]
    assert client.create_calls == 1


def test_create_rejected_by_api_is_upstream_error():
    client = StubClient(create_result={"success": False, "msg": "card declined"})
    cfg = load_config()
    with pytest.raises(VastComfyError) as excinfo:
        create_instance(client, cfg, make_offer(), "label-x", paid_confirmed=True)
    assert excinfo.value.code == "E_UPSTREAM"
    assert "card declined" in excinfo.value.message


# ---------------------------------------------------------------------------
# The create body (wire contract for the official image)
# ---------------------------------------------------------------------------


def test_build_create_body():
    cfg = load_config()
    body = build_create_body(cfg, "vast-comfy-test")
    assert body["image"] == "vastai/comfy:v0.40.0-cuda-12.9-py312"
    assert image_ref(cfg) == body["image"]
    assert body["runtype"] == "ssh_proxy"  # v1 is proxy-only
    assert body["target_state"] == "running"
    assert body["cancel_unavail"] is True
    assert body["disk"] == 300.0
    assert body["label"] == "vast-comfy-test"
    env = body["env"]
    assert env["-p 1111:1111"] == "1"
    assert env["-p 8188:8188"] == "1"
    assert "--port 18188" in env["COMFYUI_ARGS"]
    assert env["COMFYUI_API_BASE"] == "http://localhost:18188"
    assert "ComfyUI" in env["PORTAL_CONFIG"]


# ---------------------------------------------------------------------------
# wait_for_running
# ---------------------------------------------------------------------------


def test_wait_for_running_returns_when_running():
    client = StubClient(instances=[{"id": 1, "actual_status": "loading"}, {"id": 1, "actual_status": "running"}])
    cfg = load_config()
    clock, sleep, _ = fake_clock()
    instance = wait_for_running(client, 1, cfg, sleep=sleep, clock=clock)
    assert instance["actual_status"] == "running"
    assert len(client.calls) == 2


@pytest.mark.parametrize("status", ["exited", "unknown", "offline", "stopped", "frozen"])
def test_wait_for_running_terminal_status_fails_without_destroying(status):
    client = StubClient(instances=[{"id": 5, "actual_status": status, "status_msg": "boom"}])
    cfg = load_config()
    clock, sleep, _ = fake_clock()
    with pytest.raises(VastComfyError) as excinfo:
        wait_for_running(client, 5, cfg, sleep=sleep, clock=clock)
    error = excinfo.value
    assert error.code == "E_START_FAILED"
    assert status in error.message
    combined = (error.message + " " + " ".join(error.hints)).lower()
    assert "not stopped/destroyed" in combined
    assert len(client.calls) == 1  # terminal state is final; no polling


def test_wait_for_running_disappearing_instance_fails():
    client = StubClient(instances=[None])
    cfg = load_config()
    clock, sleep, _ = fake_clock()
    with pytest.raises(VastComfyError) as excinfo:
        wait_for_running(client, 5, cfg, sleep=sleep, clock=clock)
    error = excinfo.value
    assert error.code == "E_START_FAILED"
    assert "destroyed" in " ".join(error.hints).lower()


def test_wait_for_running_timeout_uses_bounded_clock():
    client = StubClient(instances=[{"id": 5, "actual_status": "loading"}])
    cfg = load_config(None, {"launch.start_wait_minutes": 1, "launch.poll_interval_seconds": 10})
    clock, sleep, state = fake_clock()
    with pytest.raises(VastComfyError) as excinfo:
        wait_for_running(client, 5, cfg, sleep=sleep, clock=clock)
    error = excinfo.value
    assert error.code == "E_START_TIMEOUT"
    assert error.details["last_status"] == "loading"
    assert error.details["instance_id"] == 5
    # Bounded polling: the loop stopped as soon as the deadline was reached.
    assert state["t"] == 60.0
    assert len(client.calls) == 7


def test_wait_for_running_polls_with_configured_interval():
    client = StubClient(instances=[{"id": 1, "actual_status": "loading"}, {"id": 1, "actual_status": "running"}])
    cfg = load_config(None, {"launch.poll_interval_seconds": 7})
    clock, sleep, state = fake_clock()
    on_poll = []
    wait_for_running(client, 1, cfg, sleep=sleep, clock=clock, on_poll=on_poll.append)
    assert state["t"] == 7.0
    assert len(on_poll) == 2


# ---------------------------------------------------------------------------
# verify_offer: re-check before paying; never auto-switch machines
# ---------------------------------------------------------------------------


def test_verify_offer_gone_raises_offer_changed(mapping):
    client = StubClient(offers=[])
    cfg = load_config()
    spec = mapping.resolve("rtx_4060_ti_16gb")
    with pytest.raises(VastComfyError) as excinfo:
        verify_offer(client, cfg, spec, 1001)
    assert excinfo.value.code == "E_OFFER_CHANGED"
    assert excinfo.value.details["offer_id"] == 1001


def test_verify_offer_no_longer_qualifying_raises(mapping):
    stale = make_offer(id=1001, reliability=0.5)
    client = StubClient(offers=[stale])
    cfg = load_config()
    spec = mapping.resolve("rtx_4060_ti_16gb")
    with pytest.raises(VastComfyError) as excinfo:
        verify_offer(client, cfg, spec, 1001)
    error = excinfo.value
    assert error.code == "E_OFFER_CHANGED"
    assert any("reliability" in r for r in error.details["reasons"])


def test_verify_offer_rechecks_with_an_id_filter_on_a_fresh_search(mapping):
    fresh_offer = make_offer(id=1001)
    client = StubClient(offers=[fresh_offer])
    cfg = load_config()
    spec = mapping.resolve("rtx_4060_ti_16gb")
    verified = verify_offer(client, cfg, spec, 1001)
    assert verified is fresh_offer
    query = client.calls[0][1][0]
    assert query["ask_contract_id"] == {"eq": 1001}
    assert query["allocated_storage"] == 300.0  # still priced for the real disk
