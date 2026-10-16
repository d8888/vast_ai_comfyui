"""Thin Vast.ai REST client with an explicit, conservative retry policy.

Wire contracts implemented here (verified against the official OpenAPI docs and
the official ``vastai`` CLI source):

* Search offers: ``POST /api/v0/bundles/`` with the query object as the JSON body.
* Create instance: ``PUT /api/v0/asks/{ask_id}/`` — never retried automatically;
  the response carries the new instance id in ``new_contract``.
* Show instance: ``GET /api/v0/instances/{id}/?owner=me`` -> ``{"instances": {...}}``
* List instances: ``GET /api/v1/instances/`` (keyset pagination via ``after_token``).
* Stop: ``PUT /api/v0/instances/{id}/`` ``{"state": "stopped"}``
* Destroy: ``DELETE /api/v0/instances/{id}/``
* SSH keys: ``POST /api/v0/ssh/`` ``{"ssh_key": "<public key>"}`` / ``GET /api/v0/ssh/``
* Logs: ``PUT /api/v0/instances/request_logs/{id}/`` then fetch ``result_url``.

Only idempotent reads (GET / search POST) are retried on transient failures.
The API key is sent only to the configured API base URL and is never logged.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import requests

from ..errors import (
    E_NETWORK,
    E_UPSTREAM,
    VastComfyError,
)

_USER_AGENT = "vast-comfy/0.1 (+https://github.com/; contact: local user)"

_SECRET_PATTERNS = [
    re.compile(r"(hf_[A-Za-z0-9]{10,})"),
    re.compile(r"(?i)(bearer\s+[A-Za-z0-9._\-]{10,})"),
    re.compile(r"(?i)(api[-_]?key[\"'\s:=]+[A-Za-z0-9._\-]{10,})"),
]


def scrub(text: str) -> str:
    """Best-effort secret scrubbing for error messages and logs."""
    if not text:
        return text
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("<redacted>", out)
    return out


class VastClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 30,
        transient_retries: int = 3,
        retry_backoff_seconds: list[float] | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise VastComfyError(
                code=E_UPSTREAM,
                message="Vast.ai API key is missing",
                hints=["Set VAST_API_KEY (see .env.template) or pass --env-file .env."],
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transient_retries = max(0, int(transient_retries))
        self.retry_backoff_seconds = list(retry_backoff_seconds or [1, 3, 8])
        self.session = session or requests.Session()
        self._sleep = sleep

    # ------------------------------------------------------------------
    # low level
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }

    def _do(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        allow_retry: bool = True,
        timeout: float | None = None,
    ) -> Any:
        attempts = (self.transient_retries + 1) if allow_retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self.session.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json_body,
                    params=params,
                    timeout=timeout or self.timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleep(self._backoff_for(attempt))
                    continue
                raise VastComfyError(
                    code=E_NETWORK,
                    message=f"Network error calling Vast.ai API {method} {path}: {exc.__class__.__name__}",
                    details={"endpoint": path, "method": method},
                    hints=(["Check connectivity and retry the read-only query."] if allow_retry else
                           ["The request may have been accepted. Verify its outcome before retrying."]),
                ) from exc
            except requests.RequestException as exc:
                raise VastComfyError(
                    code=E_NETWORK, message=f"Transport failed during Vast.ai API {method} {path}",
                    details={"endpoint": path, "method": method, "error_type": type(exc).__name__},
                    hints=["The outcome may be unknown; verify before retrying any mutation."],
                ) from exc

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt + 1 < attempts:
                    self._sleep(self._backoff_for(attempt))
                    continue
                raise VastComfyError(
                    code=E_UPSTREAM,
                    message=f"Vast.ai API returned HTTP {resp.status_code} for {method} {path}",
                    stage="vast_api",
                    details={
                        "status": resp.status_code,
                        "endpoint": path,
                        "api_message": scrub(_body_excerpt(resp)),
                    },
                )

            if resp.status_code >= 400:
                raise VastComfyError(
                    code=E_UPSTREAM,
                    message=f"Vast.ai API rejected {method} {path} with HTTP {resp.status_code}",
                    stage="vast_api",
                    details={
                        "status": resp.status_code,
                        "endpoint": path,
                        "api_error": _api_error_fields(resp),
                    },
                )

            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise VastComfyError(
                    code=E_UPSTREAM,
                    message=f"Unexpected non-JSON response from {method} {path} (Content-Type: {content_type!r})",
                    stage="vast_api",
                    details={"endpoint": path},
                )
            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                raise VastComfyError(
                    code=E_UPSTREAM,
                    message=f"Invalid JSON from {method} {path}",
                    stage="vast_api",
                    details={"endpoint": path},
                ) from exc

        raise AssertionError("unreachable")  # pragma: no cover

    def _backoff_for(self, attempt: int) -> float:
        if not self.retry_backoff_seconds:
            return 0.0
        if attempt < len(self.retry_backoff_seconds):
            return float(self.retry_backoff_seconds[attempt])
        return float(self.retry_backoff_seconds[-1])

    # ------------------------------------------------------------------
    # offers
    # ------------------------------------------------------------------

    def search_offers(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self._do("POST", "/api/v0/bundles/", json_body=query, allow_retry=True)
        offers = payload.get("offers") if isinstance(payload, dict) else None
        if not isinstance(offers, list):
            raise VastComfyError(
                code=E_UPSTREAM,
                message="Offer search response did not contain an 'offers' list",
                stage="vast_api",
                details={"keys": sorted(payload.keys()) if isinstance(payload, dict) else None},
            )
        return offers

    # ------------------------------------------------------------------
    # instances
    # ------------------------------------------------------------------

    def create_instance(self, ask_id: int, body: dict[str, Any]) -> dict[str, Any]:
        # Deliberately never retried: a retried create can double-charge.
        return self._do("PUT", f"/api/v0/asks/{ask_id}/", json_body=body, allow_retry=False)

    def show_instance(self, instance_id: int | str) -> dict[str, Any] | None:
        payload = self._do(
            "GET", f"/api/v0/instances/{instance_id}/", params={"owner": "me"}, allow_retry=True
        )
        if not isinstance(payload, dict):
            raise VastComfyError(code=E_UPSTREAM, message="Unexpected show-instance payload", stage="vast_api")
        return payload.get("instances")

    def list_instances(self, *, max_pages: int = 40) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        after_token: str | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {
                "select_filters": json.dumps({}),
                "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
                "limit": 25,
            }
            if after_token:
                params["after_token"] = after_token
            payload = self._do("GET", "/api/v1/instances/", params=params, allow_retry=True)
            if not isinstance(payload, dict):
                raise VastComfyError(code=E_UPSTREAM, message="Unexpected list-instances payload", stage="vast_api")
            page = payload.get("instances") or []
            if not isinstance(page, list):
                raise VastComfyError(code=E_UPSTREAM, message="Unexpected list-instances page", stage="vast_api")
            rows.extend(page)
            after_token = payload.get("next_token")
            if not after_token:
                return rows
        raise VastComfyError(
            code=E_UPSTREAM,
            message=f"Instance listing exceeded {max_pages} pages; refusing to loop forever",
            stage="vast_api",
        )

    def stop_instance(self, instance_id: int | str) -> dict[str, Any]:
        return self._action_result(
            self._do(
                "PUT", f"/api/v0/instances/{instance_id}/", json_body={"state": "stopped"}, allow_retry=False
            ),
            "stop",
            instance_id,
        )

    def destroy_instance(self, instance_id: int | str) -> dict[str, Any]:
        return self._action_result(
            self._do("DELETE", f"/api/v0/instances/{instance_id}/", json_body={}, allow_retry=False),
            "destroy",
            instance_id,
        )

    @staticmethod
    def _action_result(payload: Any, action: str, instance_id: int | str) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise VastComfyError(
                code=E_UPSTREAM,
                message=f"Vast.ai did not confirm {action}; the actual resource state must be checked.",
                details={"instance_id": instance_id, "action": action, "api_message": scrub(str(payload))[:300],
                         "action_outcome": "rejected" if isinstance(payload, dict) and payload.get("success") is False else "unknown"},
            )
        return payload

    # ------------------------------------------------------------------
    # ssh keys
    # ------------------------------------------------------------------

    def list_ssh_keys(self) -> list[dict[str, Any]]:
        payload = self._do("GET", "/api/v0/ssh/", allow_retry=True)
        if isinstance(payload, dict):
            keys = payload.get("ssh_keys", payload.get("keys", []))
        elif isinstance(payload, list):
            keys = payload
        else:
            keys = []
        return [k for k in keys if isinstance(k, dict)]

    def register_ssh_key(self, public_key: str) -> dict[str, Any]:
        payload = self._do(
            "POST", "/api/v0/ssh/", json_body={"ssh_key": public_key.strip()}, allow_retry=False
        )
        if isinstance(payload, dict) and payload.get("success") is False:
            raise VastComfyError(
                code=E_UPSTREAM,
                message="Vast.ai rejected the SSH public key",
                stage="vast_api",
                details={"api_message": scrub(str(payload.get("msg", "")))},
            )
        return payload if isinstance(payload, dict) else {"success": True}

    # ------------------------------------------------------------------
    # logs (best effort, bounded)
    # ------------------------------------------------------------------

    def request_logs(self, instance_id: int | str, *, tail: int = 200, attempts: int = 30) -> str:
        payload = self._do(
            "PUT",
            f"/api/v0/instances/request_logs/{instance_id}/",
            json_body={"tail": int(tail)},
            allow_retry=False,
        )
        if not isinstance(payload, dict) or not payload.get("result_url"):
            raise VastComfyError(
                code=E_UPSTREAM,
                message="Log request did not return a result_url",
                stage="vast_api",
                details={"api_message": scrub(str(payload))},
            )
        url = payload["result_url"]
        for _ in range(attempts):
            self._sleep(0.5)
            try:
                resp = requests.get(url, timeout=self.timeout_seconds)  # noqa: S113 (bounded by timeout)
            except requests.RequestException:
                continue
            if resp.status_code == 200:
                return resp.text
        raise VastComfyError(
            code=E_UPSTREAM,
            message="Instance logs were not available within the bounded wait",
            stage="vast_api",
            hints=["Retry `instance logs` later."],
        )


def _body_excerpt(resp: requests.Response, limit: int = 300) -> str:
    try:
        return scrub(resp.text[:limit])
    except Exception:  # pragma: no cover - defensive
        return "<unreadable>"


def _api_error_fields(resp: requests.Response) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        data = resp.json()
        if isinstance(data, dict):
            for key in ("error", "msg", "detail"):
                if key in data:
                    fields[key] = scrub(str(data[key])[:300])
    except Exception:
        fields["body_excerpt"] = _body_excerpt(resp)
    return fields
