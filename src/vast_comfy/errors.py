"""Stable error codes, exit codes and the exception type used across vast-comfy.

Contract (SPEC.md sections 3.2 and 14):

* Every failure has a stable English ``code`` (``E_*``) that agents can branch on.
* CLI failures exit non-zero; the exit code groups the failure class.
* Irrecoverable errors are reported explicitly and never trigger implicit
  cleanup, retries, re-renting, or lock manipulation.
* Error payloads must not contain secret values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Stable error codes
# ---------------------------------------------------------------------------

E_USAGE = "E_USAGE"
E_CONFIG = "E_CONFIG"
E_NOT_IMPLEMENTED = "E_NOT_IMPLEMENTED"

# GPU / offers / pricing
E_UNKNOWN_ALIAS = "E_UNKNOWN_ALIAS"
E_GPU_INCOMPATIBLE = "E_GPU_INCOMPATIBLE"
E_NO_OFFERS = "E_NO_OFFERS"
E_PRICE_LIMIT = "E_PRICE_LIMIT"
E_OFFER_CHANGED = "E_OFFER_CHANGED"

# Payment / creation
E_APPROVAL_REQUIRED = "E_APPROVAL_REQUIRED"
E_CREATE_UNCERTAIN = "E_CREATE_UNCERTAIN"
E_START_FAILED = "E_START_FAILED"
E_START_TIMEOUT = "E_START_TIMEOUT"

# SSH / keys
E_SSH_INFO_MISSING = "E_SSH_INFO_MISSING"
E_SSH_CONNECT = "E_SSH_CONNECT"
E_SSH_AUTH = "E_SSH_AUTH"
E_SSH_HOSTKEY = "E_SSH_HOSTKEY"
E_SSH_COMMAND = "E_SSH_COMMAND"
E_KEY_EXISTS = "E_KEY_EXISTS"
E_KEY_MISSING = "E_KEY_MISSING"

# Lock / session / jobs
E_LOCK_CONFLICT = "E_LOCK_CONFLICT"
E_LOCK_STALE = "E_LOCK_STALE"
E_LOCK_NOT_HELD = "E_LOCK_NOT_HELD"
E_LOCK_JOBS_RUNNING = "E_LOCK_JOBS_RUNNING"
E_JOB_FAILED = "E_JOB_FAILED"
E_JOB_TIMEOUT = "E_JOB_TIMEOUT"
E_JOB_NOT_FOUND = "E_JOB_NOT_FOUND"

# Remote package / install / service / smoke
E_DEPLOY = "E_DEPLOY"
E_DOWNLOAD = "E_DOWNLOAD"
E_INSTALL = "E_INSTALL"
E_INSTALL_HASH_MISMATCH = "E_INSTALL_HASH_MISMATCH"
E_INSTALL_DIRTY = "E_INSTALL_DIRTY"
E_INSTALL_PATH_ESCAPE = "E_INSTALL_PATH_ESCAPE"
E_COMFY_BUSY = "E_COMFY_BUSY"
E_COMFY_UNAVAILABLE = "E_COMFY_UNAVAILABLE"
E_SMOKE_FAILED = "E_SMOKE_FAILED"
E_INSTANCE_STATE = "E_INSTANCE_STATE"

# Transport level
E_UPSTREAM = "E_UPSTREAM"
E_NETWORK = "E_NETWORK"
E_INTERNAL = "E_INTERNAL"

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_APPROVAL_REQUIRED = 3
EXIT_LOCK_CONFLICT = 4
EXIT_VERIFICATION_FAILED = 5

_CODE_EXIT: dict[str, int] = {
    E_USAGE: EXIT_USAGE,
    E_APPROVAL_REQUIRED: EXIT_APPROVAL_REQUIRED,
    E_LOCK_CONFLICT: EXIT_LOCK_CONFLICT,
    E_LOCK_STALE: EXIT_LOCK_CONFLICT,
    E_LOCK_JOBS_RUNNING: EXIT_LOCK_CONFLICT,
    E_SMOKE_FAILED: EXIT_VERIFICATION_FAILED,
}


def exit_code_for(code: str | None) -> int:
    """Map an error code to the process exit code."""
    if code is None:
        return EXIT_OK
    return _CODE_EXIT.get(code, EXIT_ERROR)


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


@dataclass
class VastComfyError(Exception):
    """Structured failure carrying a stable code, stage and diagnostics.

    ``details`` must never contain secrets (tokens, private keys, .env values).
    ``hints`` are plain-language next steps for the human operator.
    """

    code: str
    message: str
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [f"[{self.code}] {self.message}"]
        if self.stage:
            parts.append(f"(stage: {self.stage})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.stage:
            payload["stage"] = self.stage
        if self.details:
            payload["details"] = self.details
        if self.hints:
            payload["hints"] = self.hints
        return payload


class UsageError(VastComfyError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(code=E_USAGE, message=message, **kw)


class ConfigError(VastComfyError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(code=E_CONFIG, message=message, **kw)


class UnknownAliasError(VastComfyError):
    def __init__(self, alias: str, candidates: list[str] | None = None, **kw: Any) -> None:
        details = dict(kw.pop("details", {}))
        details["alias"] = alias
        if candidates:
            details["candidates"] = candidates
        hints = list(kw.pop("hints", []))
        hints.append("Aliases are exact; no fuzzy auto-correction is performed. Run `vast-comfy gpu aliases` to list valid aliases.")
        super().__init__(
            code=E_UNKNOWN_ALIAS,
            message=f"Unknown GPU alias: {alias!r}",
            details=details,
            hints=hints,
            **kw,
        )


class ApprovalRequiredError(VastComfyError):
    def __init__(self, message: str, **kw: Any) -> None:
        details = dict(kw.pop("details", {}))
        details.setdefault("approval_required", True)
        super().__init__(code=E_APPROVAL_REQUIRED, message=message, details=details, **kw)


class LockConflictError(VastComfyError):
    def __init__(self, message: str, owner: dict[str, Any] | None = None, **kw: Any) -> None:
        details = dict(kw.pop("details", {}))
        if owner:
            details["owner"] = owner
        hints = list(kw.pop("hints", []))
        hints.extend(
            [
                "Only the lock owner may operate on this instance. No action was taken.",
                "To inspect: vast-comfy session status --instance <ID>",
                "Killing a local agent does NOT stop its remote work; confirm remote work has stopped before user-driven recovery.",
            ]
        )
        super().__init__(
            code=E_LOCK_CONFLICT, message=message, details=details, hints=hints, **kw
        )


class JobTimeoutError(VastComfyError):
    def __init__(self, job_id: str, timeout_seconds: float, **kw: Any) -> None:
        details = dict(kw.pop("details", {}))
        details.update({"job_id": job_id, "timeout_seconds": timeout_seconds, "job_may_still_be_running": True})
        hints = list(kw.pop("hints", []))
        hints.append("A wait timeout does NOT stop the remote job. Inspect with `vast-comfy jobs status` / `jobs logs`.")
        super().__init__(
            code=E_JOB_TIMEOUT,
            message=f"Timed out waiting for job {job_id} after {timeout_seconds:.0f}s (job may still be running)",
            details=details,
            hints=hints,
            **kw,
        )
