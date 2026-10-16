"""SPEC 17 items 3-5: hard-condition offer evaluation, filtering and selection."""

from __future__ import annotations

import pytest

from offer_factory import MISSING, make_offer
from vast_comfy.config import load_config
from vast_comfy.errors import VastComfyError
from vast_comfy.vast.offers import (
    filter_offers,
    quote_from_offer,
    raise_no_eligible,
    select_best,
)

ALIAS16 = "rtx_4060_ti_16gb"
ALIAS8 = "rtx_4060_ti_8gb"


def evaluate(mapping, offer, alias=ALIAS16, **overrides):
    from vast_comfy.vast.offers import evaluate_offer

    cfg = load_config(None, overrides or None)
    spec = mapping.resolve(alias)
    return evaluate_offer(offer, cfg, spec)


# ---------------------------------------------------------------------------
# VRAM: the 16GB alias must never accept an 8GB card and vice versa
# ---------------------------------------------------------------------------


def test_16gb_alias_rejects_8gb_offer(mapping):
    reasons = evaluate(mapping, make_offer(gpu_ram=8188))
    assert any("gpu_ram" in r and "below alias minimum" in r for r in reasons)


def test_8gb_alias_rejects_16gb_offer(mapping):
    reasons = evaluate(mapping, make_offer(gpu_ram=16376), alias=ALIAS8)
    assert any("gpu_ram" in r and "above alias maximum" in r for r in reasons)


def test_vram_floor_from_config_applies(mapping):
    reasons = evaluate(mapping, make_offer(gpu_ram=12000), **{"gpu.vram_gb_min": 16})
    assert any("below configured floor" in r for r in reasons)


# ---------------------------------------------------------------------------
# Every hard condition is enforced; nothing is guessed
# ---------------------------------------------------------------------------


def test_fully_qualifying_offer_is_accepted(mapping):
    assert evaluate(mapping, make_offer()) == []


@pytest.mark.parametrize(
    "overrides, needle",
    [
        (dict(gpu_frac=0.5), "fractional GPU"),
        (dict(gpu_frac=0.99), "fractional GPU"),
        (dict(num_gpus=2), "exact match required"),
        (dict(inet_up=499.0), "inet_up"),
        (dict(inet_down=499.0), "inet_down"),
        (dict(cpu_cores_effective=3.5), "cpu_cores_effective"),
        (dict(cpu_ram=15999), "cpu_ram"),
        (dict(reliability=0.989), "reliability"),
        (dict(verified=False, verification=None), "not verified"),
        (dict(verified=MISSING, verification=MISSING), "not verified"),
        (dict(gpu_arch="amd"), "gpu_arch"),
        (dict(cpu_arch="arm64"), "cpu_arch"),
        (dict(compute_cap=700), "compute_cap"),
        (dict(cuda_max_good=12.8), "cuda_max_good"),
        (dict(dph_total=2.01), "exceeds cap"),
        (dict(gpu_name="RTX 3060"), "gpu_name"),
        (dict(rentable=False), "not rentable"),
        (dict(rented=True), "currently rented"),
        (dict(geolocation="United States, US"), "not in allow list"),
    ],
)
def test_hard_conditions_never_relaxed(mapping, overrides, needle):
    if "not in allow list" in needle:
        offer = make_offer(**overrides)
        reasons = evaluate(mapping, offer, **{"host.geolocation_allow": ["JP"]})
    else:
        reasons = evaluate(mapping, make_offer(**overrides))
    assert reasons, f"expected rejection for {overrides}"
    assert any(needle in r for r in reasons), reasons


def test_verification_string_is_accepted_when_verified_flag_missing(mapping):
    offer = make_offer(verified=MISSING, verification="verified")
    assert evaluate(mapping, offer) == []


def test_geolocation_exclude(mapping):
    reasons = evaluate(mapping, make_offer(geolocation="China, CN"), **{"host.geolocation_exclude": ["CN"]})
    assert any("excluded" in r for r in reasons)
    ok = evaluate(mapping, make_offer(geolocation="Japan, JP"), **{"host.geolocation_exclude": ["CN"]})
    assert ok == []


def test_geolocation_exclude_fails_closed_on_unknown_locations(mapping):
    reasons = evaluate(mapping, make_offer(geolocation=MISSING), **{"host.geolocation_exclude": ["CN"]})
    assert any("could not be determined" in r for r in reasons)
    reasons = evaluate(mapping, make_offer(geolocation="Moon"), **{"host.geolocation_exclude": ["CN"]})
    assert any("could not be determined" in r for r in reasons)


# ---------------------------------------------------------------------------
# Missing fields are never guessed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_missing_required_field_is_rejected(mapping, field):
    reasons = evaluate(mapping, make_offer(**{field: MISSING}))
    assert any(f"missing required field: {field}" in r for r in reasons), reasons


def test_non_numeric_required_field_is_rejected(mapping):
    reasons = evaluate(mapping, make_offer(reliability="high"))
    assert any("field reliability is not numeric" in r for r in reasons)


def test_offer_without_id_cannot_be_referenced(mapping):
    reasons = evaluate(mapping, make_offer(id=MISSING))
    assert reasons == ["missing required field: id"]


# ---------------------------------------------------------------------------
# filter_offers / select_best / raise_no_eligible
# ---------------------------------------------------------------------------


def test_filter_offers_splits_eligible_and_rejected(mapping):
    cfg = load_config()
    spec = mapping.resolve(ALIAS16)
    good1 = make_offer(id=11, dph_total=0.4)
    good2 = make_offer(id=12, dph_total=0.5)
    bad_ram = make_offer(id=13, gpu_ram=8188)
    bad_rel = make_offer(id=14, reliability=0.5)

    eligible, rejected = filter_offers([good1, bad_ram, good2, bad_rel], cfg, spec)
    assert [o["id"] for o in eligible] == [11, 12]
    assert [r["offer_id"] for r in rejected] == [13, 14]
    assert all(r["reasons"] for r in rejected)


def test_select_best_returns_cheapest_and_breaks_ties_deterministically(mapping):
    cheapest = select_best([make_offer(id=5, dph_total=0.9), make_offer(id=2, dph_total=0.3), make_offer(id=9, dph_total=0.6)])
    assert cheapest["id"] == 2

    # Same price: higher reliability wins, then the lower id.
    best_rel = select_best(
        [
            make_offer(id=3, dph_total=0.5, reliability=0.991),
            make_offer(id=7, dph_total=0.5, reliability=0.999),
        ]
    )
    assert best_rel["id"] == 7

    best_id = select_best(
        [
            make_offer(id=8, dph_total=0.5, reliability=0.995),
            make_offer(id=4, dph_total=0.5, reliability=0.995),
        ]
    )
    assert best_id["id"] == 4


def test_select_best_requires_eligible_offers():
    with pytest.raises(AssertionError):
        select_best([])


def test_raise_no_eligible_no_offers_at_all(mapping):
    cfg = load_config()
    spec = mapping.resolve(ALIAS16)
    with pytest.raises(VastComfyError) as excinfo:
        raise_no_eligible([], [], cfg, spec)
    assert excinfo.value.code == "E_NO_OFFERS"


def test_raise_no_eligible_offers_fail_non_price_conditions(mapping):
    cfg = load_config()
    spec = mapping.resolve(ALIAS16)
    offers = [make_offer(id=21, gpu_ram=8188)]
    eligible, rejected = filter_offers(offers, cfg, spec)
    assert eligible == []
    with pytest.raises(VastComfyError) as excinfo:
        raise_no_eligible(offers, rejected, cfg, spec)
    error = excinfo.value
    assert error.code == "E_NO_OFFERS"
    assert error.details["sample_rejections"] == rejected[:5]


def test_raise_no_eligible_offers_only_fail_the_price_cap(mapping):
    cfg = load_config()
    spec = mapping.resolve(ALIAS16)
    offers = [make_offer(id=31, dph_total=2.75), make_offer(id=32, dph_total=3.5)]
    eligible, rejected = filter_offers(offers, cfg, spec)
    assert eligible == []
    with pytest.raises(VastComfyError) as excinfo:
        raise_no_eligible(offers, rejected, cfg, spec)
    error = excinfo.value
    assert error.code == "E_PRICE_LIMIT"
    assert error.details["cheapest_dph_total"] == 2.75
    assert error.details["cheapest_offer_id"] == 31
    assert error.details["max_dph_total_usd_per_hour"] == 2.0


def test_quote_from_offer_uses_integrated_price_and_lists_traffic_costs(mapping):
    offer = make_offer(dph_total=0.5, dph_base=0.4, inet_up_cost=0.05, inet_down_cost=0.02, search={"diskHour": 0.1})
    quote = quote_from_offer(offer)
    assert quote["dph_total_usd_per_hour"] == 0.5  # integrated, never re-summed
    assert quote["dph_base_usd_per_hour"] == 0.4
    assert quote["disk_cost_usd_per_hour"] == 0.1
    assert quote["inet_up_cost_usd_per_gb"] == 0.05
    assert quote["inet_down_cost_usd_per_gb"] == 0.02
    assert quote["gpu_ram_mb"] == 16376
    assert quote["gpu_ram_gb"] == 16.0
    assert quote["cpu_ram_gb"] == 64.0
    assert "USD/hour" in quote["unit_contracts"]["dph_total"]
