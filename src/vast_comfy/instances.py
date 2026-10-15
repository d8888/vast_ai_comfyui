"""Instance helpers: live lookup, state guards and SSH runner construction."""

from __future__ import annotations

from .config import Config
from .errors import E_INSTANCE_STATE, VastComfyError
from .ssh import SSHContext, SSHRunner, context_from_instance
from .vast.client import VastClient


def live_instance(client: VastClient, instance_id: int | str) -> dict:
    instance = client.show_instance(instance_id)
    if instance is None:
        raise VastComfyError(
            code=E_INSTANCE_STATE,
            message=f"Instance {instance_id} was not found on this account",
            stage="instance",
        )
    return instance


def ensure_running(instance: dict) -> None:
    status = instance.get("actual_status")
    if status != "running":
        raise VastComfyError(
            code=E_INSTANCE_STATE,
            message=(
                f"Instance {instance.get('id')} is not running (actual_status={status!r}); "
                "modifying actions are refused."
            ),
            stage="instance",
            details={"instance_id": instance.get("id"), "actual_status": status, "status_msg": instance.get("status_msg")},
            hints=[
                "Stopped/offline instances are handled by the user in the Vast.ai console; "
                "once it is running again, retry the toolkit.",
            ],
        )


def runner_for(instance: dict, cfg: Config) -> SSHRunner:
    context: SSHContext = context_from_instance(instance, cfg)
    return SSHRunner(context)


def load_runner(
    client: VastClient, cfg: Config, instance_id: int | str, *, require_running: bool = True
) -> tuple[dict, SSHRunner]:
    instance = live_instance(client, instance_id)
    if require_running:
        ensure_running(instance)
    return instance, runner_for(instance, cfg)
