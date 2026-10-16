"""Explicit platform mutations, protected by the live remote operation lock."""
from __future__ import annotations

from . import locks
from .errors import ApprovalRequiredError, VastComfyError
from .instances import load_runner


def _mutate(client, cfg, instance_id, session_id, action):
    _, runner = load_runner(client, cfg, instance_id)
    with locks.operation(runner, cfg, session_id, label=f"instance-{action}") as lease:
        # Must precede dispatch: timeout/5xx/invalid JSON/interrupt may happen
        # AFTER acceptance. Even explicit rejection is left for human inspection.
        lease.retain()
        try:
            getattr(client, action + "_instance")(instance_id)
        except (Exception, KeyboardInterrupt) as exc:
            error = exc if isinstance(exc, VastComfyError) else VastComfyError(
                "E_INSTANCE_STATE", "Platform mutation outcome is unknown; verify it before retrying.",
                stage="lifecycle", details={"error_type": type(exc).__name__})
            error.details.update(instance_id=instance_id, session_id=session_id,
                                 action=action, lock_state="retained; manual inspection required",
                                 instance_may_be_billing=True)
            error.details.setdefault("action_outcome", "unknown")
            error.hints.append("The request was NOT retried and the operation lock was NOT released. Inspect the Vast.ai console.")
            if error is exc:
                raise
            raise error from exc


def stop(client, cfg, instance_id, session_id):
    _mutate(client, cfg, instance_id, session_id, "stop")
    return {"instance_id": instance_id, "stop_requested": True,
            "lock_state": "retained pending platform transition",
            "billing_note": "Storage billing continues. Verify the actual state in the Vast.ai console."}


def destroy(client, cfg, instance_id, session_id, *, confirm_destroy=False):
    if not confirm_destroy:
        raise ApprovalRequiredError("Destroy requires explicit --confirm-destroy confirmation.")
    _mutate(client, cfg, instance_id, session_id, "destroy")
    return {"instance_id": instance_id, "destroy_requested": True,
            "lock_state": "retained pending platform transition",
            "note": "The request was accepted; verify completion in the Vast.ai console."}
