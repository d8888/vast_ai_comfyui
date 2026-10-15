"""Create-and-prepare flow (SPEC.md section 16.2) and offer quoting."""

from __future__ import annotations

import time
from typing import Any

from . import prepare
from .config import Config
from .errors import E_NO_OFFERS, VastComfyError
from .gpu import load_mapping
from .instances import runner_for
from .ssh import rental_preflight, wait_for_ssh
from .vast.client import VastClient
from .vast.offers import (
    build_search_query,
    filter_offers,
    quote_from_offer,
    raise_no_eligible,
    select_best,
)
from .vast.rental import create_instance, verify_offer, wait_for_running

DEFAULT_LABEL_PREFIX = "vast-comfy"


def _reject_non_cuda_hardware(spec) -> None:
    """Known non-NVIDIA models fail distinctly from 'no qualifying inventory'."""
    if spec.gpu_name.lower().startswith("instinct"):
        raise VastComfyError(
            code="E_GPU_INCOMPATIBLE",
            message=(
                f"Alias {spec.alias!r} maps to {spec.gpu_name!r}, which is not an NVIDIA CUDA GPU; "
                "v1 requires NVIDIA CUDA and x86-64."
            ),
            stage="offers",
            details={"alias": spec.alias, "gpu_name": spec.gpu_name},
            hints=["Pick an NVIDIA alias (`vast-comfy gpu aliases`)."],
        )


def quote_for_alias(client: VastClient, cfg: Config, alias: str, *, top: int = 10) -> dict:
    """Search + hard-filter + cheapest selection. Never creates anything."""
    spec = load_mapping().resolve(alias)
    _reject_non_cuda_hardware(spec)
    query = build_search_query(cfg, spec)
    offers = client.search_offers(query)
    if not offers:
        raise_no_eligible([], [], cfg, spec)
    eligible, rejected = filter_offers(offers, cfg, spec)
    if not eligible:
        raise_no_eligible(offers, rejected, cfg, spec)
    best = select_best(eligible)
    ranked = sorted(eligible, key=lambda o: (float(o["dph_total"]), -float(o["reliability"]), int(o["id"])))
    return {
        "ok": True,
        "alias": alias,
        "gpu_name": spec.gpu_name,
        "alias_constraints": {
            "gpu_ram_mb_min": spec.gpu_ram_mb_min,
            "gpu_ram_mb_max": spec.gpu_ram_mb_max,
        },
        "eligible_count": len(eligible),
        "rejected_sample": rejected[:5],
        "best": quote_from_offer(best),
        "offers": [quote_from_offer(o) for o in ranked[:top]],
        "price_cap_usd_per_hour": cfg.require("pricing.max_dph_total_usd_per_hour"),
        "note": (
            "The hourly cap applies to the integrated dph_total for the configured disk size. "
            "Traffic fees are billed separately and are not covered by the cap."
        ),
    }


def default_label(alias: str) -> str:
    return f"{DEFAULT_LABEL_PREFIX}-{alias}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"


def create_and_prepare(
    client: VastClient,
    cfg: Config,
    *,
    alias: str,
    label: str | None = None,
    confirm_paid: bool,
    accept_stale: bool = False,
    release_on_success: bool | None = None,
    cases: str | None = None,
    run_id: str | None = None,
    verify_model_shas: bool = False,
    on_step: Any = None,
) -> dict:
    """Full flow: search -> verify -> create (explicit payment) -> wait -> prepare."""

    def step(name: str, payload: Any = None) -> None:
        if on_step:
            on_step(name, payload)

    # Refuse before paying if the configured install would need remote credentials.
    from .prepare import assert_no_remote_credentials, build_install_manifest

    assert_no_remote_credentials(build_install_manifest())
    if confirm_paid:
        rental_preflight(client, cfg)

    spec = load_mapping().resolve(alias)
    quote = quote_for_alias(client, cfg, alias)
    step("quote", quote["best"])

    best_id = int(quote["best"]["offer_id"])
    fresh = verify_offer(client, cfg, spec, best_id)
    step("verified_offer", quote_from_offer(fresh))

    final_label = label or default_label(alias)
    created = create_instance(client, cfg, fresh, final_label, paid_confirmed=confirm_paid)
    step("created", created)

    try:
        instance = wait_for_running(client, created["instance_id"], cfg, on_poll=lambda inst: step("waiting", inst))
        step("running", {"instance_id": instance.get("id"), "gpu_name": instance.get("gpu_name")})
        runner = runner_for(instance, cfg)
        wait_for_ssh(runner, cfg)
        step("ssh_ok", {"ssh_host": instance.get("ssh_host"), "ssh_port": instance.get("ssh_port")})
        result = prepare.prepare_existing(
            runner,
            cfg,
            instance_id=created["instance_id"],
            session_id=None,
            release_on_success=release_on_success,
            cases=cases,
            run_id=run_id,
            verify_model_shas=verify_model_shas,
        )
    except VastComfyError as exc:
        # Never mask the original error; add the instance reference and billing state.
        exc.details.setdefault("instance_id", created["instance_id"])
        exc.details.setdefault("label", final_label)
        exc.details.setdefault(
            "instance_billing",
            "the instance was created and may still be billing; it was NOT stopped or destroyed",
        )
        raise
    result["label"] = final_label
    result["created"] = created
    return result
