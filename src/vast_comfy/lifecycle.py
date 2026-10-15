"""Explicit platform mutations, protected by the live remote operation lock."""
from __future__ import annotations

from . import locks
from .errors import ApprovalRequiredError
from .instances import load_runner


def stop(client, cfg, instance_id, session_id):
    _, runner = load_runner(client, cfg, instance_id)
    with locks.operation(runner, cfg, session_id, label="instance-stop") as lease:
        client.stop_instance(instance_id)
        # SSH may disappear immediately. Do not pretend the transition is already
        # complete, or unlock before the platform has actually stopped the work.
        lease.retain()
    return {"instance_id": instance_id, "stop_requested": True,
            "lock_state": "retained pending platform transition",
            "billing_note": "Storage billing continues. Verify the actual state in the Vast.ai console."}


def destroy(client, cfg, instance_id, session_id, *, confirm_destroy=False):
    if not confirm_destroy:
        raise ApprovalRequiredError("Destroy requires explicit --confirm-destroy confirmation.")
    _, runner = load_runner(client, cfg, instance_id)
    with locks.operation(runner, cfg, session_id, label="instance-destroy") as lease:
        client.destroy_instance(instance_id)
        lease.retain()
    return {"instance_id": instance_id, "destroy_requested": True,
            "lock_state": "retained pending platform transition",
            "note": "The request was accepted; verify completion in the Vast.ai console."}
