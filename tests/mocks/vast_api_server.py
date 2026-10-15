"""Local mock of the Vast.ai REST API for wire-contract integration tests.

The server binds ``127.0.0.1`` on an ephemeral port and implements exactly the
endpoints listed below (the real Vast.ai API is never contacted):

* ``POST /api/v0/bundles/``  -> ``{"offers": [...]}`` (echoes the received query)
* ``PUT /api/v0/asks/{id}/`` -> ``{"success": true, "new_contract": 4242}``
* ``GET /api/v0/instances/{id}/?owner=me`` -> ``{"instances": {...}}``
* ``GET /api/v1/instances/`` -> keyset pagination via ``after_token``
* ``PUT /api/v0/instances/{id}/`` (stop) / ``DELETE`` (destroy)
* ``POST/GET /api/v0/ssh/``
* ``PUT /api/v0/instances/request_logs/{id}/`` -> ``{"result_url": "..."}``
  plus the returned ``/logs/abc`` page (plain text).

Responses can be scripted per route (e.g. return HTTP 500 once and then 200) to
test the client's retry policy.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

_ASKS_RE = re.compile(r"^/api/v0/asks/(\d+)/$")
_INSTANCE_RE = re.compile(r"^/api/v0/instances/(\d+)/$")
_REQ_LOGS_RE = re.compile(r"^/api/v0/instances/request_logs/(\d+)/$")


class MockVastState:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.offers: list[dict[str, Any]] = []
        self.instances: dict[str, dict[str, Any]] = {}
        self.default_instance: dict[str, Any] = {
            "id": 4242,
            "label": "vast-comfy-test",
            "actual_status": "running",
            "ssh_host": "ssh5.vast.ai",
            "ssh_port": 41234,
        }
        self.instance_pages: list[dict[str, Any]] = []
        self.page_index = 0
        self.ssh_keys: list[dict[str, Any]] = [{"id": 1, "ssh_key": "ssh-ed25519 AAAA test"}]
        self.log_text = "mock container log line\n"
        self.created_instances: list[dict[str, Any]] = []
        self.stopped: list[str] = []
        self.destroyed: list[str] = []
        self._overrides: list[dict[str, Any]] = []

    # -- test scripting ----------------------------------------------------

    def script(
        self,
        method: str,
        path: str,
        responses: list[tuple[int, Any]],
        *,
        match: str = "exact",
    ) -> None:
        """Queue responses for a route; the last entry repeats forever."""
        self._overrides.append(
            {
                "method": method.upper(),
                "path": path,
                "responses": list(responses),
                "index": 0,
                "match": match,
            }
        )

    def _override_for(self, method: str, path: str) -> tuple[int, Any] | None:
        for override in self._overrides:
            if override["method"] != method.upper():
                continue
            if override["match"] == "exact" and override["path"] != path:
                continue
            if override["match"] == "prefix" and not path.startswith(override["path"]):
                continue
            index = min(override["index"], len(override["responses"]) - 1)
            override["index"] += 1
            return override["responses"][index]
        return None

    # -- request bookkeeping ----------------------------------------------

    def requests_for(self, method: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
        rows = self.requests
        if method is not None:
            rows = [r for r in rows if r["method"] == method.upper()]
        if path is not None:
            rows = [r for r in rows if r["path"] == path]
        return rows


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence pytest output
        pass

    # -- helpers -----------------------------------------------------------

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw.decode("utf-8", "replace")

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        if isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- dispatch ----------------------------------------------------------

    def _dispatch(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()
        state: MockVastState = self.server.state  # type: ignore[attr-defined]
        state.requests.append(
            {
                "method": self.command,
                "path": path,
                "query": parse_qs(parsed.query),
                "headers": dict(self.headers),
                "body": body,
            }
        )

        scripted = state._override_for(self.command, path)
        if scripted is not None:
            status, payload = scripted
            self._send(status, payload)
            return

        if self.command == "POST" and path == "/api/v0/bundles/":
            self._send(200, {"offers": state.offers, "query_received": body})
            return

        match = _ASKS_RE.match(path)
        if self.command == "PUT" and match:
            instance = {"id": 4242, "offer_id": int(match.group(1)), "body": body}
            state.created_instances.append(instance)
            self._send(200, {"success": True, "new_contract": 4242})
            return

        match = _REQ_LOGS_RE.match(path)
        if self.command == "PUT" and match:
            self._send(200, {"result_url": f"http://127.0.0.1:{self.server.server_address[1]}/logs/abc"})
            return

        if self.command == "GET" and path == "/logs/abc":
            self._send(200, state.log_text, content_type="text/plain")
            return

        match = _INSTANCE_RE.match(path)
        if match:
            instance_id = match.group(1)
            if self.command == "GET":
                instance = state.instances.get(instance_id, state.default_instance)
                self._send(200, {"instances": instance})
                return
            if self.command == "PUT":
                state.stopped.append(instance_id)
                self._send(200, {"success": True})
                return
            if self.command == "DELETE":
                state.destroyed.append(instance_id)
                self._send(200, {"success": True})
                return

        if path == "/api/v1/instances/" and self.command == "GET":
            if state.instance_pages:
                index = min(state.page_index, len(state.instance_pages) - 1)
                state.page_index += 1
                self._send(200, state.instance_pages[index])
            else:
                self._send(200, {"instances": [], "next_token": None})
            return

        if path == "/api/v0/ssh/":
            if self.command == "GET":
                self._send(200, {"ssh_keys": state.ssh_keys})
                return
            if self.command == "POST":
                self._send(200, {"success": True})
                return

        self._send(404, {"error": f"no mock route for {self.command} {path}"})

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()


class MockVastServer:
    def __init__(self) -> None:
        self.state = MockVastState()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.state = self.state  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@contextlib.contextmanager
def mock_vast_server() -> Iterator[MockVastServer]:
    server = MockVastServer()
    try:
        yield server
    finally:
        server.stop()
