"""Deterministic offer fixtures shared by the offer/rental tests."""

from __future__ import annotations

from typing import Any

_MISSING = object()


def make_offer(**overrides: Any) -> dict[str, Any]:
    """A fully qualifying offer for ``rtx_4060_ti_16gb``.

    Pass ``field=_MISSING`` to delete a field (missing-field tests).
    """
    offer: dict[str, Any] = {
        "id": 1001,
        "gpu_name": "RTX 4060 Ti",
        "num_gpus": 1,
        "gpu_ram": 16376,
        "cpu_ram": 64000,
        "cpu_cores_effective": 8.0,
        "inet_up": 1000.0,
        "inet_down": 1000.0,
        "reliability": 0.995,
        "dph_total": 0.35,
        "compute_cap": 890,
        "cuda_max_good": 12.9,
        "gpu_frac": 1.0,
        "disk_space": 500.0,
        "gpu_arch": "nvidia",
        "cpu_arch": "amd64",
        "verified": True,
        "rentable": True,
        "rented": False,
        "geolocation": "Japan, JP",
        "machine_id": 5001,
        "host_id": 9001,
    }
    for key, value in overrides.items():
        if value is _MISSING:
            offer.pop(key, None)
        else:
            offer[key] = value
    return offer


MISSING = _MISSING
