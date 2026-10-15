"""Paid instance creation and startup verification (SPEC.md section 5.4).

Guarantees:
* Creating a paid resource requires an explicit paid-confirmation flag.
* The create request is never retried automatically; an uncertain outcome is
  reported as uncertain and must be verified, never blindly resent.
* Terminal startup states fail loudly; nothing is auto-destroyed or re-rented.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..config import Config
from ..errors import (
    E_NETWORK,
    E_OFFER_CHANGED,
    E_START_FAILED,
    E_START_TIMEOUT,
    E_UPSTREAM,
    ApprovalRequiredError,
    VastComfyError,
)
from ..gpu import AliasSpec
from ..ssh import rental_preflight
from .client import VastClient, scrub
from .offers import build_search_query, evaluate_offer, quote_from_offer

_TERMINAL_BAD_STATUSES = {"exited", "unknown", "offline", "stopped", "frozen"}


def image_ref(cfg: Config) -> str:
    return f"{cfg.require('image.repository')}:{cfg.require('image.tag')}"


def build_create_body(cfg: Config, label: str) -> dict[str, Any]:
    internal_port = int(cfg.require("image.internal_port"))
    comfy_ext = int(cfg.require("image.comfy_external_port"))
    portal_ext = int(cfg.require("image.portal_external_port"))
    comfyui_args = cfg.require("image.comfyui_args")
    portal_config = f"localhost:{portal_ext}:11111:/:Instance Portal|localhost:{comfy_ext}:{internal_port}:/:ComfyUI"

    env: dict[str, str] = {
        f"-p {portal_ext}:{portal_ext}": "1",
        f"-p {comfy_ext}:{comfy_ext}": "1",
        "COMFYUI_ARGS": comfyui_args,
        "COMFYUI_API_BASE": f"http://localhost:{internal_port}",
        "PORTAL_CONFIG": portal_config,
        "OPEN_BUTTON_PORT": str(portal_ext),
        "DATA_DIRECTORY": "/workspace/",
    }

    return {
        "client_id": "me",
        "image": image_ref(cfg),
        "env": env,
        "disk": float(cfg.require("storage.disk_gb")),
        "label": label,
        "runtype": "ssh_proxy",  # v1: proxy-only SSH
        "target_state": "running",
        # Do not leave a half-created instance behind when the machine cannot start.
        "cancel_unavail": True,
    }


def verify_offer(
    client: VastClient, cfg: Config, spec: AliasSpec, offer_id: int
) -> dict[str, Any]:
    """Re-check the exact offer right before creation (SPEC.md 5.3).

    Re-runs the full hard-condition evaluation on a fresh search; a changed,
    gone, or non-qualifying offer fails instead of silently switching machines.
    """
    query = build_search_query(cfg, spec)
    query.pop("limit", None)
    query["id"] = {"eq": int(offer_id)}
    query["limit"] = 5
    offers = client.search_offers(query)
    matches = [o for o in offers if int(o.get("id", -1)) == int(offer_id)]
    if not matches:
        raise VastComfyError(
            code=E_OFFER_CHANGED,
            message=f"Offer {offer_id} is no longer available; nothing was rented.",
            stage="rent",
            details={"offer_id": offer_id},
            hints=["Nothing is re-selected automatically. Re-run the search if you want a new offer."],
        )
    offer = matches[0]
    reasons = evaluate_offer(offer, cfg, spec)
    if reasons:
        raise VastComfyError(
            code=E_OFFER_CHANGED,
            message=f"Offer {offer_id} no longer satisfies every hard condition; nothing was rented.",
            stage="rent",
            details={"offer_id": offer_id, "reasons": reasons},
        )
    return offer


def create_instance(
    client: VastClient,
    cfg: Config,
    offer: dict[str, Any],
    label: str,
    *,
    paid_confirmed: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Create the instance for the given freshly verified offer.

    Requires ``paid_confirmed=True`` (explicit flag from the operator/agent).
    """
    if not paid_confirmed:
        quote = quote_from_offer(offer)
        raise ApprovalRequiredError(
            "Creating a paid instance requires explicit confirmation",
            stage="rent",
            details={
                "required_flag": "--confirm-paid",
                "offer_id": quote.get("offer_id"),
                "dph_total_usd_per_hour": quote.get("dph_total_usd_per_hour"),
                "note": "The hourly cap is not a total-spend cap; traffic is billed separately.",
            },
            hints=[
                "Re-run with --confirm-paid only after reviewing the quote.",
            ],
        )

    rental_preflight(client, cfg)
    body = build_create_body(cfg, label)
    try:
        response = client.create_instance(int(offer["id"]), body)
    except VastComfyError as exc:
        status_code = exc.details.get("status")
        ambiguous = exc.code == E_NETWORK or (
            exc.code == E_UPSTREAM and (status_code is None or status_code >= 500)
        )
        if ambiguous:
            raise VastComfyError(
                code="E_CREATE_UNCERTAIN",
                message=(
                    "The create request outcome is UNKNOWN (network or server failure). "
                    "It was NOT retried; the instance may or may not exist."
                ),
                stage="rent",
                details={"offer_id": int(offer["id"]), "label": label, "cause": exc.to_dict(),
                         "since_epoch": time.time(), "instance_may_be_billing": True},
                hints=[
                    f"Verify before doing anything else: vast-comfy rent verify --label {label!r}",
                    "Do NOT resend the create request blindly — that can double-charge.",
                ],
            ) from exc
        raise

    if not isinstance(response, dict) or not response.get("success"):
        api_message = ""
        if isinstance(response, dict):
            api_message = scrub(str(response.get("msg", response.get("error", ""))))
        raise VastComfyError(
            code=E_UPSTREAM,
            message=f"Vast.ai did not accept the create request: {api_message or 'unknown error'}",
            stage="rent",
            details={"offer_id": int(offer["id"]), "label": label, "api_message": api_message},
        )

    new_contract = response.get("new_contract")
    if not new_contract:
        raise VastComfyError(
            code="E_CREATE_UNCERTAIN",
            message=(
                "The create response did not include a new_contract id; the outcome is UNCERTAIN "
                "and was NOT retried."
            ),
            stage="rent",
            details={"offer_id": int(offer["id"]), "label": label, "response_keys": sorted(response.keys())},
            hints=[f"Verify with: vast-comfy rent verify --label {label!r}"],
        )

    return {"instance_id": int(new_contract), "label": label, "offer_id": int(offer["id"])}


def verify_creation(
    client: VastClient, label: str, since_epoch: float, *, clock: Callable[[], float] = time.time
) -> list[dict[str, Any]]:
    """Find instances created by this tool with the given label around ``since_epoch``."""
    rows = client.list_instances()
    found: list[dict[str, Any]] = []
    for row in rows:
        if row.get("label") != label:
            continue
        start = row.get("start_date")
        if isinstance(start, (int, float)) and start >= since_epoch - 600:
            found.append(
                {
                    "instance_id": row.get("id"),
                    "actual_status": row.get("actual_status"),
                    "created": start,
                    "gpu_name": row.get("gpu_name"),
                    "dph_total": row.get("dph_total"),
                }
            )
    return found


def wait_for_running(
    client: VastClient,
    instance_id: int | str,
    cfg: Config,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    on_poll: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Wait (bounded) until the instance is running; any terminal state fails."""
    timeout_seconds = float(cfg.require("launch.start_wait_minutes")) * 60.0
    poll_seconds = float(cfg.require("launch.poll_interval_seconds"))
    deadline = clock() + timeout_seconds
    last_status: Any = None

    while True:
        instance = client.show_instance(instance_id)
        if instance is None:
            raise VastComfyError(
                code=E_START_FAILED,
                message=(
                    f"Instance {instance_id} disappeared while waiting for it to start "
                    "(e.g. cancelled because the machine could not run it)."
                ),
                stage="start",
                details={"instance_id": instance_id, "last_status": last_status},
                hints=["No instance was destroyed by this tool. Choose an offer and retry explicitly."],
            )
        last_status = instance.get("actual_status")
        if on_poll:
            on_poll(instance)
        if last_status == "running":
            return instance
        if last_status in _TERMINAL_BAD_STATUSES:
            raise VastComfyError(
                code=E_START_FAILED,
                message=f"Instance {instance_id} reached terminal status {last_status!r} and will not start",
                stage="start",
                details={
                    "instance_id": instance_id,
                    "actual_status": last_status,
                    "status_msg": instance.get("status_msg"),
                },
                hints=[
                    "The instance was NOT stopped/destroyed automatically; billing may continue.",
                    f"Inspect with `vast-comfy instance show --instance {instance_id}` and decide explicitly.",
                ],
            )
        if clock() >= deadline:
            raise VastComfyError(
                code=E_START_TIMEOUT,
                message=(
                    f"Instance {instance_id} did not reach 'running' within {timeout_seconds/60:.0f} minutes "
                    f"(last status: {last_status!r})"
                ),
                stage="start",
                details={"instance_id": instance_id, "last_status": last_status},
                hints=["The instance may still start later; billing may already be running. Inspect and decide explicitly."],
            )
        sleep(poll_seconds)
