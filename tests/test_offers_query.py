"""SPEC 17 item 3: the numeric/unit contract of the offer search query."""

from __future__ import annotations

from vast_comfy.config import load_config
from vast_comfy.vast.offers import build_search_query


def query_for(mapping, *, alias="rtx_4060_ti_16gb", **overrides):
    cfg = load_config(None, overrides or None)
    spec = mapping.resolve(alias)
    return build_search_query(cfg, spec), cfg, spec


def test_exact_gpu_count_verified_and_rental_type(mapping):
    query, _, _ = query_for(mapping)
    assert query["num_gpus"] == {"eq": 1}  # exact, never "at least"
    assert query["verified"] == {"eq": True}
    assert query["rentable"] == {"eq": True}
    assert query["rented"] == {"eq": False}
    assert query["external"] == {"eq": False}
    assert query["type"] == "ondemand"
    assert query["gpu_frac"] == {"eq": 1.0}  # a full GPU, never a fractional slice


def test_storage_disk_and_ordering(mapping):
    query, _, _ = query_for(mapping)
    assert query["allocated_storage"] == 300.0
    assert query["order"] == [["dph_total", "asc"]]
    assert query["limit"] == 100


def test_gpu_ram_bounds_from_alias(mapping):
    query, _, _ = query_for(mapping)
    # Bounds come with the alias (RTX 4060 Ti's official name covers 8/16GB).
    assert query["gpu_ram"] == {"gte": 15500, "lte": 17000}

    query8, _, _ = query_for(mapping, alias="rtx_4060_ti_8gb")
    assert query8["gpu_ram"] == {"gte": 7500, "lte": 8600}

    plain, _, _ = query_for(mapping, alias="rtx_4090")
    assert "gpu_ram" not in plain


def test_gpu_ram_floor_override_widens_the_lower_bound(mapping):
    query, _, _ = query_for(mapping, **{"gpu.vram_gb_min": 17})
    assert query["gpu_ram"]["lte"] == 17000
    assert query["gpu_ram"]["gte"] == 17 * 1024


def test_cpu_ram_is_converted_gb_to_mb_with_1000(mapping):
    query, _, _ = query_for(mapping)
    assert query["cpu_ram"] == {"gte": 16000}  # 16 GB * 1000, not 1024

    query_override, _, _ = query_for(mapping, **{"host.cpu_ram_gb_min": 48})
    assert query_override["cpu_ram"] == {"gte": 48000}


def test_network_inet_thresholds_and_unit_contract(mapping):
    query, cfg, _ = query_for(mapping)
    assert query["inet_up"] == {"gte": 500.0}
    assert query["inet_down"] == {"gte": 500.0}
    assert cfg.require("network.unit_contract") == "mbps_platform_reported"

    query2, _, _ = query_for(mapping, **{"network.inet_up_min": 800, "network.inet_down_min": 900})
    assert query2["inet_up"] == {"gte": 800.0}
    assert query2["inet_down"] == {"gte": 900.0}


def test_host_and_image_compatibility_thresholds(mapping):
    query, _, _ = query_for(mapping)
    assert query["reliability"] == {"gte": 0.99}
    assert query["cpu_cores_effective"] == {"gte": 4.0}
    assert query["compute_cap"] == {"gte": 750}
    assert query["cuda_max_good"] == {"gte": 12.9}
    assert query["gpu_arch"] == {"eq": "nvidia"}
    assert query["cpu_arch"] == {"eq": "amd64"}


def test_geolocation_allow_exclude(mapping):
    query, _, _ = query_for(mapping)
    assert "geolocation" not in query  # default: no restriction

    allowed, _, _ = query_for(mapping, **{"host.geolocation_allow": ["JP", "TW"]})
    assert allowed["geolocation"] == {"in": ["JP", "TW"]}

    excluded, _, _ = query_for(mapping, **{"host.geolocation_exclude": ["CN"]})
    assert excluded["geolocation"] == {"notin": ["CN"]}

    both, _, _ = query_for(mapping, **{"host.geolocation_allow": ["JP"], "host.geolocation_exclude": ["CN"]})
    assert both["geolocation"] == {"in": ["JP"]}  # allow wins; exclude is still enforced locally


def test_query_is_not_wrapped(mapping):
    query, _, _ = query_for(mapping)
    assert "q" not in query
    assert query["gpu_name"] == {"eq": "RTX 4060 Ti"}
