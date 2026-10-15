"""SPEC 17 items 6, 12 and 19: the Vast.ai REST wire contract (local mock server).

The mock server lives in ``tests/mocks/vast_api_server.py``. No real network
traffic leaves the machine and the real Vast.ai API is never contacted.
"""

from __future__ import annotations

import json

import pytest
import requests

from mocks.vast_api_server import mock_vast_server
from vast_comfy.errors import E_NETWORK, VastComfyError
from vast_comfy.vast.client import VastClient, scrub

API_KEY = "test-api-key-123"
SLOW_BACKOFF = [0, 0, 0]


@pytest.fixture()
def server():
    with mock_vast_server() as mock:
        yield mock


def make_client(server, *, retries: int = 0, with_session: requests.Session | None = None) -> VastClient:
    return VastClient(
        server.url,
        API_KEY,
        timeout_seconds=5,
        transient_retries=retries,
        retry_backoff_seconds=SLOW_BACKOFF,
        session=with_session,
        sleep=lambda _seconds: None,
    )


# ---------------------------------------------------------------------------
# Search offers: POST body is the query object itself
# ---------------------------------------------------------------------------


def test_search_offers_sends_bearer_header_and_raw_query_body(server):
    server.state.offers = [{"id": 1, "dph_total": 0.4}, {"id": 2, "dph_total": 0.5}]
    client = make_client(server)
    query = {"gpu_name": {"eq": "RTX 4060 Ti"}, "num_gpus": {"eq": 1}, "order": [["dph_total", "asc"]]}

    offers = client.search_offers(query)

    assert [o["id"] for o in offers] == [1, 2]
    record = server.state.requests[-1]
    assert record["method"] == "POST"
    assert record["path"] == "/api/v0/bundles/"
    assert record["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert record["body"] == query  # the query object itself, NOT wrapped in "q"
    assert "q" not in record["body"]


def test_search_offers_payload_without_offers_list_is_upstream_error(server):
    server.state.script("POST", "/api/v0/bundles/", [(200, {"weird": True})])
    client = make_client(server)
    with pytest.raises(VastComfyError) as excinfo:
        client.search_offers({"x": 1})
    assert excinfo.value.code == "E_UPSTREAM"
    assert excinfo.value.stage == "vast_api"


# ---------------------------------------------------------------------------
# Create: PUT /asks/{id}/, exactly once, never retried
# ---------------------------------------------------------------------------


def test_create_uses_put_and_is_never_retried_on_http_500(server):
    ask_body = {"client_id": "me", "runtype": "ssh_proxy"}
    server.state.script(
        "PUT",
        "/api/v0/asks/77/",
        [(500, {"error": "internal failure"}), (200, {"success": True, "new_contract": 4242})],
    )
    client = make_client(server, retries=3)

    with pytest.raises(VastComfyError) as excinfo:
        client.create_instance(77, ask_body)

    error = excinfo.value
    assert error.code == "E_UPSTREAM"
    assert error.details["status"] == 500
    assert error.details["endpoint"] == "/api/v0/asks/77/"

    calls = server.state.requests_for("PUT", "/api/v0/asks/77/")
    assert len(calls) == 1, "a create request must never be retried automatically"
    assert calls[0]["body"] == ask_body


def test_create_success_returns_new_contract(server):
    client = make_client(server)
    payload = client.create_instance(88, {"label": "x"})
    assert payload["success"] is True
    assert payload["new_contract"] == 4242
    assert server.state.created_instances[-1]["offer_id"] == 88


# ---------------------------------------------------------------------------
# Reads: transient retry works; persistent errors surface with status
# ---------------------------------------------------------------------------


def test_get_retries_on_500_then_succeeds(server):
    page = {"instances": [{"id": 1}, {"id": 2}], "next_token": None}
    server.state.script("GET", "/api/v1/instances/", [(500, {"error": "flaky"}), (200, page)])
    client = make_client(server, retries=2)

    rows = client.list_instances()

    assert [row["id"] for row in rows] == [1, 2]
    assert len(server.state.requests_for("GET", "/api/v1/instances/")) == 2


def test_get_gives_up_after_configured_retries(server):
    server.state.script("GET", "/api/v1/instances/", [(500, {"error": "always down"})])
    client = make_client(server, retries=2)
    with pytest.raises(VastComfyError) as excinfo:
        client.list_instances()
    assert excinfo.value.code == "E_UPSTREAM"
    assert excinfo.value.details["status"] == 500
    assert len(server.state.requests_for("GET", "/api/v1/instances/")) == 3  # 1 + 2 retries


def test_error_body_surfaces_as_upstream_with_status(server):
    server.state.script("GET", "/api/v0/instances/999/", [(400, {"error": "no such instance"})])
    client = make_client(server)
    with pytest.raises(VastComfyError) as excinfo:
        client.show_instance(999)
    error = excinfo.value
    assert error.code == "E_UPSTREAM"
    assert error.details["status"] == 400
    assert error.details["api_error"]["error"] == "no such instance"


def test_connection_error_surfaces_as_network(server, monkeypatch):
    client = make_client(server)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(client.session, "request", boom)
    with pytest.raises(VastComfyError) as excinfo:
        client.list_instances()
    error = excinfo.value
    assert error.code == "E_NETWORK"
    assert error.details["method"] == "GET"
    assert error.details["endpoint"] == "/api/v1/instances/"


def test_non_json_response_is_upstream(server):
    server.state.script("GET", "/api/v0/instances/321/", [(200, "<html>not json</html>")])
    client = make_client(server)
    with pytest.raises(VastComfyError) as excinfo:
        client.show_instance(321)
    assert excinfo.value.code == "E_UPSTREAM"
    assert "Invalid JSON" in excinfo.value.message


# ---------------------------------------------------------------------------
# Instance GET / pagination / stop / destroy
# ---------------------------------------------------------------------------


def test_show_instance_uses_owner_me(server):
    server.state.default_instance = {"id": 4242, "actual_status": "running", "ssh_host": "ssh5.vast.ai", "ssh_port": 12345}
    client = make_client(server)
    instance = client.show_instance(4242)
    assert instance["id"] == 4242
    record = server.state.requests[-1]
    assert record["path"] == "/api/v0/instances/4242/"
    assert record["query"] == {"owner": ["me"]}


def test_list_instances_follows_after_token_pagination(server):
    server.state.instance_pages = [
        {"instances": [{"id": 1}], "next_token": "tok-2"},
        {"instances": [{"id": 2}], "next_token": None},
    ]
    client = make_client(server)
    rows = client.list_instances()
    assert [row["id"] for row in rows] == [1, 2]

    calls = server.state.requests_for("GET", "/api/v1/instances/")
    assert len(calls) == 2
    assert "after_token" not in calls[0]["query"]
    assert calls[1]["query"]["after_token"] == ["tok-2"]
    assert json.loads(calls[0]["query"]["select_filters"][0]) == {}


def test_stop_and_destroy_use_put_and_delete(server):
    client = make_client(server)
    assert client.stop_instance(4242)["success"] is True
    assert client.destroy_instance(4242)["success"] is True
    assert server.state.stopped == ["4242"]
    assert server.state.destroyed == ["4242"]
    stop_call = server.state.requests_for("PUT", "/api/v0/instances/4242/")[0]
    assert stop_call["body"] == {"state": "stopped"}
    destroy_call = server.state.requests_for("DELETE", "/api/v0/instances/4242/")[0]
    assert destroy_call["body"] == {}


# ---------------------------------------------------------------------------
# SSH keys and logs
# ---------------------------------------------------------------------------


def test_ssh_key_register_and_list(server):
    client = make_client(server)
    client.register_ssh_key("ssh-ed25519 AAAA new-key")
    post = server.state.requests_for("POST", "/api/v0/ssh/")[0]
    assert post["body"] == {"ssh_key": "ssh-ed25519 AAAA new-key"}

    keys = client.list_ssh_keys()
    assert keys and keys[0]["ssh_key"].startswith("ssh-ed25519")


def test_request_logs_returns_text_from_result_url(server):
    server.state.log_text = "line one\nline two\n"
    client = make_client(server)
    text = client.request_logs(4242, tail=50)
    assert text == "line one\nline two\n"
    request_call = server.state.requests_for("PUT", "/api/v0/instances/request_logs/4242/")[0]
    assert request_call["body"] == {"tail": 50}


# ---------------------------------------------------------------------------
# Secrets are scrubbed on the way into error payloads
# ---------------------------------------------------------------------------


def test_scrub_redacts_tokens():
    token = "hf_" + "abcdefghij1234567890"
    assert scrub(f"download failed for {token}") == "download failed for <redacted>"
    assert scrub("Authorization: Bearer abcdefghijklmnop") == "Authorization: <redacted>"
    assert scrub("api_key=supersecretvalue123") == "<redacted>"


def test_upstream_error_message_scrubs_secret_body(server):
    token = "hf_" + "abcdefghij1234567890"
    server.state.script("POST", "/api/v0/bundles/", [(500, {"error": f"token {token} invalid"})])
    client = make_client(server)
    with pytest.raises(VastComfyError) as excinfo:
        client.search_offers({"x": 1})
    assert token not in json.dumps(excinfo.value.details)


def test_client_requires_api_key():
    with pytest.raises(VastComfyError) as excinfo:
        VastClient("http://127.0.0.1:1", "")
    assert excinfo.value.code == "E_UPSTREAM"
