"""Offer search, hard-condition evaluation and cheapest selection (SPEC.md section 5).

Unit contracts (verified against the official OpenAPI schema and CLI source):

* ``gpu_ram`` / ``cpu_ram`` / ``gpu_total_ram`` are **MB** in the REST search API
  (the CLI multiplies user GB values by 1000 before sending).
* ``inet_up`` / ``inet_down`` carry the platform-reported number; it is compared
  as-is with the same thresholds the official Vast.ai ComfyUI template uses
  (e.g. ``inet_down gte 500``). ``network.unit_contract`` pins this in config.
* ``dph_total`` is the integrated ``$/hour`` for the requested
  ``allocated_storage``; it is the only price used for the budget cap. Traffic
  costs (``inet_up_cost`` / ``inet_down_cost``, $/GB) are surfaced separately.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import Config
from ..errors import ConfigError, E_NO_OFFERS, E_PRICE_LIMIT, VastComfyError
from ..gpu import AliasSpec

# Numeric fields that must be present, correctly typed, for an offer to be
# evaluable at all. Missing fields are never guessed (SPEC.md 5.2).
REQUIRED_NUMERIC_FIELDS = (
    "id",
    "num_gpus",
    "gpu_ram",
    "cpu_ram",
    "cpu_cores_effective",
    "inet_up",
    "inet_down",
    "reliability",
    "dph_total",
    "compute_cap",
    "cuda_max_good",
    "gpu_frac",
    "disk_space",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_search_query(cfg: Config, spec: AliasSpec) -> dict[str, Any]:
    count = int(cfg.require("gpu.count"))
    query: dict[str, Any] = {
        # Mirror the CLI defaults so semantics match the console.
        "verified": {"eq": True},
        "external": {"eq": bool(cfg.get("host.external", False))},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        # Hard conditions from config + alias.
        "gpu_name": {"eq": spec.gpu_name},
        "num_gpus": {"eq": count},
        "gpu_arch": {"eq": cfg.require("gpu.gpu_arch")},
        "cpu_arch": {"eq": cfg.require("gpu.cpu_arch")},
        "reliability": {"gte": float(cfg.require("host.reliability_min"))},
        "cpu_cores_effective": {"gte": float(cfg.require("host.cpu_cores_effective_min"))},
        "cpu_ram": {"gte": int(float(cfg.require("host.cpu_ram_gb_min")) * 1000)},  # REST contract: MB
        "inet_up": {"gte": float(cfg.require("network.inet_up_min"))},
        "inet_down": {"gte": float(cfg.require("network.inet_down_min"))},
        "compute_cap": {"gte": int(cfg.require("gpu.compute_cap_min"))},
        "cuda_max_good": {"gte": float(cfg.require("gpu.cuda_max_good_min"))},
        "gpu_frac": {"eq": 1.0},  # a full GPU, never a fractional slice
        # Ordering / bounding / rental type.
        "order": [["dph_total", "asc"]],
        "type": cfg.require("pricing.rental_type"),
        "limit": int(cfg.require("offers.result_limit")),
        "allocated_storage": float(cfg.require("storage.disk_gb")),
    }

    ram_min: int | None = spec.gpu_ram_mb_min
    vram_floor_gb = cfg.get("gpu.vram_gb_min")
    if vram_floor_gb is not None:
        floor_mb = int(float(vram_floor_gb) * 1024)
        ram_min = max(ram_min or 0, floor_mb)
    ram_max: int | None = spec.gpu_ram_mb_max
    if ram_min is not None and ram_max is not None and ram_min > ram_max:
        raise ConfigError("Configured VRAM minimum contradicts the GPU alias maximum")
    if ram_min is not None or ram_max is not None:
        constraint: dict[str, Any] = {}
        if ram_min is not None:
            constraint["gte"] = ram_min
        if ram_max is not None:
            constraint["lte"] = ram_max
        query["gpu_ram"] = constraint

    allow = cfg.get("host.geolocation_allow") or []
    exclude = cfg.require("host.geolocation_exclude") or []
    if allow:
        query["geolocation"] = {"in": list(allow)}
    elif exclude:
        query["geolocation"] = {"notin": list(exclude)}

    return query


def _country_from_geolocation(geolocation: Any) -> str | None:
    if not isinstance(geolocation, str):
        return None
    parts = [p.strip() for p in geolocation.replace(",", " ").split()]
    for part in reversed(parts):
        if len(part) == 2 and part.isalpha():
            return part.upper()
    return None


def evaluate_offer(offer: dict[str, Any], cfg: Config, spec: AliasSpec) -> list[str]:
    """Return the list of hard-condition violations (empty = eligible)."""
    reasons: list[str] = []

    for field_name in REQUIRED_NUMERIC_FIELDS:
        if field_name not in offer or offer[field_name] is None:
            reasons.append(f"missing required field: {field_name}")
        elif not _is_number(offer[field_name]):
            reasons.append(f"field {field_name} is not numeric")

    if any(r.startswith("missing required field: id") for r in reasons):
        return reasons  # cannot even reference this offer

    if reasons:
        # Continue only if the numeric fields we compare below are usable.
        numeric_ok = {
            f
            for f in REQUIRED_NUMERIC_FIELDS
            if f in offer and _is_number(offer.get(f))
        }
    else:
        numeric_ok = set(REQUIRED_NUMERIC_FIELDS)

    if offer.get("gpu_name") != spec.gpu_name:
        reasons.append(f"gpu_name {offer.get('gpu_name')!r} != alias gpu_name {spec.gpu_name!r}")

    if "num_gpus" in numeric_ok and offer["num_gpus"] != int(cfg.require("gpu.count")):
        reasons.append(f"num_gpus {offer['num_gpus']} != required {cfg.require('gpu.count')} (exact match required)")

    if "gpu_frac" in numeric_ok and abs(float(offer["gpu_frac"]) - 1.0) > 1e-9:
        reasons.append(f"gpu_frac {offer['gpu_frac']} is a fractional GPU slice (full GPU required)")

    if "id" in numeric_ok and (offer["id"] <= 0 or int(offer["id"]) != offer["id"]):
        reasons.append("offer id must be a positive integer")

    if "disk_space" in numeric_ok and offer["disk_space"] < cfg.require("storage.disk_gb"):
        reasons.append("disk_space is below the requested allocated storage")

    if "gpu_ram" in numeric_ok:
        ram = float(offer["gpu_ram"])
        if spec.gpu_ram_mb_min is not None and ram < spec.gpu_ram_mb_min:
            reasons.append(
                f"gpu_ram {int(ram)}MB below alias minimum {spec.gpu_ram_mb_min}MB"
            )
        if spec.gpu_ram_mb_max is not None and ram > spec.gpu_ram_mb_max:
            reasons.append(
                f"gpu_ram {int(ram)}MB above alias maximum {spec.gpu_ram_mb_max}MB"
            )
        vram_floor_gb = cfg.get("gpu.vram_gb_min")
        if vram_floor_gb is not None and ram < float(vram_floor_gb) * 1024:
            reasons.append(f"gpu_ram {int(ram)}MB below configured floor {vram_floor_gb}GB")

    if "cpu_ram" in numeric_ok:
        required_mb = float(cfg.require("host.cpu_ram_gb_min")) * 1000
        if float(offer["cpu_ram"]) < required_mb:
            reasons.append(f"cpu_ram {offer['cpu_ram']}MB below required {int(required_mb)}MB")

    if "cpu_cores_effective" in numeric_ok:
        required_cores = float(cfg.require("host.cpu_cores_effective_min"))
        if float(offer["cpu_cores_effective"]) < required_cores:
            reasons.append(
                f"cpu_cores_effective {offer['cpu_cores_effective']} below required {required_cores}"
            )

    if "inet_up" in numeric_ok:
        required_up = float(cfg.require("network.inet_up_min"))
        if float(offer["inet_up"]) < required_up:
            reasons.append(f"inet_up {offer['inet_up']} below required {required_up}")

    if "inet_down" in numeric_ok:
        required_down = float(cfg.require("network.inet_down_min"))
        if float(offer["inet_down"]) < required_down:
            reasons.append(f"inet_down {offer['inet_down']} below required {required_down}")

    if "reliability" in numeric_ok:
        required_rel = float(cfg.require("host.reliability_min"))
        if float(offer["reliability"]) < required_rel:
            reasons.append(f"reliability {offer['reliability']} below required {required_rel}")

    if "compute_cap" in numeric_ok:
        required_cc = int(cfg.require("gpu.compute_cap_min"))
        if int(offer["compute_cap"]) < required_cc:
            reasons.append(
                f"compute_cap {offer['compute_cap']} below required {required_cc} "
                "(image/GPU compatibility requirement)"
            )

    if "cuda_max_good" in numeric_ok:
        required_cuda = float(cfg.require("gpu.cuda_max_good_min"))
        if float(offer["cuda_max_good"]) < required_cuda:
            reasons.append(
                f"cuda_max_good {offer['cuda_max_good']} below required {required_cuda} "
                "(host driver cannot run the pinned CUDA image)"
            )

    if offer.get("gpu_arch") != cfg.require("gpu.gpu_arch"):
        reasons.append(f"gpu_arch {offer.get('gpu_arch')!r} != required {cfg.require('gpu.gpu_arch')!r}")

    if offer.get("cpu_arch") not in (cfg.require("gpu.cpu_arch"), "x86_64"):
        reasons.append(f"cpu_arch {offer.get('cpu_arch')!r} != required {cfg.require('gpu.cpu_arch')!r}")

    if cfg.require("host.verified"):
        verified = offer.get("verified")
        verification = offer.get("verification")
        if not (verified is True or verification == "verified"):
            reasons.append("host is not verified")

    if offer.get("rentable") is not True:
        reasons.append("offer is not rentable or its rentable state is unknown")

    if offer.get("rented") is not False:
        reasons.append("offer is currently rented or its rented state is unknown")

    country = _country_from_geolocation(offer.get("geolocation"))
    allow = cfg.get("host.geolocation_allow") or []
    exclude = cfg.require("host.geolocation_exclude") or []
    if allow and (country is None or country not in allow):
        reasons.append(f"geolocation {offer.get('geolocation')!r} not in allow list")
    if exclude and country is not None and country in exclude:
        reasons.append(f"geolocation {offer.get('geolocation')!r} is excluded")

    if "dph_total" in numeric_ok:
        cap = float(cfg.require("pricing.max_dph_total_usd_per_hour"))
        if offer["dph_total"] < 0:
            reasons.append("dph_total must not be negative")
        elif float(offer["dph_total"]) > cap:
            reasons.append(
                f"dph_total ${offer['dph_total']}/h exceeds cap ${cap}/h (integrated price incl. disk)"
            )

    return reasons


def filter_offers(
    offers: list[dict[str, Any]], cfg: Config, spec: AliasSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split offers into (eligible, rejected-with-reasons)."""
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for offer in offers:
        reasons = evaluate_offer(offer, cfg, spec)
        if reasons:
            rejected.append({"offer_id": offer.get("id"), "reasons": reasons})
        else:
            eligible.append(offer)
    return eligible, rejected


def select_best(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    """Cheapest integrated hourly price wins; ties break deterministically."""
    if not eligible:
        raise AssertionError("select_best requires a non-empty eligible list")

    def key(offer: dict[str, Any]) -> tuple[float, float, int]:
        return (float(offer["dph_total"]), -float(offer["reliability"]), int(offer["id"]))

    return sorted(eligible, key=key)[0]


def raise_no_eligible(
    offers: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    cfg: Config,
    spec: AliasSpec,
) -> None:
    """Raise the precise no-result error (price cap vs no qualifying offer)."""
    cap = float(cfg.require("pricing.max_dph_total_usd_per_hour"))
    price_only_rejections = [
        r
        for r in rejected
        if r["reasons"] and all("exceeds cap" in reason for reason in r["reasons"])
    ]
    if not offers:
        raise VastComfyError(
            code=E_NO_OFFERS,
            message=(
                f"No offers at all matched the hard filters for alias {spec.alias!r} "
                f"({spec.gpu_name!r}); nothing was relaxed, replaced or re-rented."
            ),
            stage="offers",
            details={"alias": spec.alias, "gpu_name": spec.gpu_name},
            hints=[
                "The model may have no current inventory. Demand is not relaxed: pick another alias or retry later.",
            ],
        )
    if price_only_rejections:
        cheapest = min(offers, key=lambda o: float(o.get("dph_total", float("inf"))))
        raise VastComfyError(
            code=E_PRICE_LIMIT,
            message=(
                f"Qualifying offers exist for {spec.gpu_name!r} but the cheapest integrated price "
                f"${cheapest.get('dph_total')}/h exceeds the cap ${cap}/h"
            ),
            stage="offers",
            details={
                "alias": spec.alias,
                "cheapest_dph_total": cheapest.get("dph_total"),
                "cheapest_offer_id": cheapest.get("id"),
                "max_dph_total_usd_per_hour": cap,
            },
            hints=["Raise pricing.max_dph_total_usd_per_hour explicitly if you accept the cost."],
        )
    sample = rejected[:5]
    raise VastComfyError(
        code=E_NO_OFFERS,
        message=(
            f"{len(offers)} offer(s) were found for {spec.gpu_name!r} but none satisfied every hard condition; "
            "nothing was relaxed."
        ),
        stage="offers",
        details={"alias": spec.alias, "gpu_name": spec.gpu_name, "sample_rejections": sample},
    )


def quote_from_offer(offer: dict[str, Any]) -> dict[str, Any]:
    """A stable, JSON-safe quote record. Prices are never double-counted."""
    gpu_ram_mb = offer.get("gpu_ram")
    search_breakdown = offer.get("search")
    disk_hour = None
    if isinstance(search_breakdown, dict):
        disk_hour = search_breakdown.get("diskHour")
    return {
        "offer_id": offer.get("id"),
        "gpu_name": offer.get("gpu_name"),
        "num_gpus": offer.get("num_gpus"),
        "gpu_ram_mb": gpu_ram_mb,
        "gpu_ram_gb": round(gpu_ram_mb / 1024.0, 1) if _is_number(gpu_ram_mb) else None,
        "cpu_cores_effective": offer.get("cpu_cores_effective"),
        "cpu_ram_gb": round(offer.get("cpu_ram", 0) / 1000.0, 1) if _is_number(offer.get("cpu_ram")) else None,
        "inet_up": offer.get("inet_up"),
        "inet_down": offer.get("inet_down"),
        "inet_up_cost_usd_per_gb": offer.get("inet_up_cost"),
        "inet_down_cost_usd_per_gb": offer.get("inet_down_cost"),
        "dph_total_usd_per_hour": offer.get("dph_total"),
        "dph_base_usd_per_hour": offer.get("dph_base"),
        "disk_cost_usd_per_hour": disk_hour,
        "storage_cost_usd_per_gb_month": offer.get("storage_cost"),
        "reliability": offer.get("reliability"),
        "verification": offer.get("verification"),
        "geolocation": offer.get("geolocation"),
        "machine_id": offer.get("machine_id"),
        "host_id": offer.get("host_id"),
        "cuda_max_good": offer.get("cuda_max_good"),
        "compute_cap": offer.get("compute_cap"),
        "driver_version": offer.get("driver_version"),
        "duration_seconds": offer.get("duration"),
        "static_ip": offer.get("static_ip"),
        "unit_contracts": {
            "gpu_ram": "MB (REST API contract; CLI converts GB -> MB)",
            "cpu_ram": "MB",
            "network": "platform-reported value compared as-is (see network.unit_contract)",
            "dph_total": "USD/hour, integrated for the requested allocated_storage; traffic is billed separately",
        },
        "raw_min": {k: offer.get(k) for k in ("id", "ask_contract_id", "machine_id", "host_id")},
    }
