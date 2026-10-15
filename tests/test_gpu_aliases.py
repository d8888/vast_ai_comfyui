"""SPEC 17 items 1 and 2: GPU alias mapping behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vast_comfy.errors import ConfigError, UnknownAliasError
from vast_comfy.gpu import (
    AliasSpec,
    add_custom_alias,
    build_official_aliases,
    load_mapping,
    slugify_gpu_name,
    update_mapping_file,
    validate_mapping_data,
)


def _base_mapping() -> dict:
    return {
        "schema_version": 1,
        "source": "https://console.vast.ai/api/v0/gpu_names/unique/",
        "fetched_at_utc": None,
        "official_names": ["RTX 4090", "RTX 4060 Ti"],
        "aliases": {
            "rtx_4090": {"gpu_name": "RTX 4090"},
            "rtx_4060_ti": {"gpu_name": "RTX 4060 Ti"},
        },
        "custom": {"my_private_rig": {"gpu_name": "Private Rig 9000", "notes": "keep me"}},
    }


def _write_mapping(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "gpu_aliases.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Committed mapping: resolve + unknown alias
# ---------------------------------------------------------------------------


def test_resolve_rtx_4060_ti_16gb_vram_bounds(mapping):
    spec: AliasSpec = mapping.resolve("rtx_4060_ti_16gb")
    assert spec.gpu_name == "RTX 4060 Ti"
    assert spec.gpu_ram_mb_min == 15500
    assert spec.gpu_ram_mb_max == 17000
    assert spec.custom is False
    # The 8GB variant is a distinct alias (same official name, different VRAM).
    spec8 = mapping.resolve("rtx_4060_ti_8gb")
    assert spec8.gpu_name == "RTX 4060 Ti"
    assert (spec8.gpu_ram_mb_min, spec8.gpu_ram_mb_max) == (7500, 8600)


def test_unknown_alias_raises_with_candidates_and_no_autocorrection(mapping):
    with pytest.raises(UnknownAliasError) as excinfo:
        mapping.resolve("rtx_4060_ti_16g")  # one character short
    error = excinfo.value
    assert error.code == "E_UNKNOWN_ALIAS"
    assert error.details["alias"] == "rtx_4060_ti_16g"
    assert error.details.get("candidates"), "candidates should be offered but never applied"
    assert "rtx_4060_ti_16gb" in error.details["candidates"]
    assert any("no fuzzy auto-correction" in hint.lower() for hint in error.hints)

    # A completely unknown alias fails too (no substitution).
    with pytest.raises(UnknownAliasError) as excinfo2:
        mapping.resolve("totally_nope")
    assert excinfo2.value.code == "E_UNKNOWN_ALIAS"


def test_mapping_is_committed_and_self_consistent(mapping):
    assert mapping.official_names, "official name snapshot must not be empty"
    assert validate_mapping_data(mapping.data) == []
    assert mapping.fetched_at_utc  # snapshot timestamp recorded


# ---------------------------------------------------------------------------
# Custom aliases
# ---------------------------------------------------------------------------


def test_add_custom_alias_and_resolve(tmp_path):
    path = _write_mapping(tmp_path, _base_mapping())
    before = json.loads(path.read_text(encoding="utf-8"))

    result = add_custom_alias(path, "rig_9000", "Private Rig 9000", gpu_ram_mb_min=24000, gpu_ram_mb_max=26000, notes="lab")
    assert result["alias"] == "rig_9000"

    loaded = load_mapping(path)
    spec = loaded.resolve("rig_9000")
    assert spec.gpu_name == "Private Rig 9000"
    assert spec.custom is True
    assert (spec.gpu_ram_mb_min, spec.gpu_ram_mb_max) == (24000, 26000)
    assert spec.notes == "lab"

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["official_names"] == before["official_names"]
    assert after["aliases"] == before["aliases"]
    assert "my_private_rig" in after["custom"]  # existing custom alias preserved


def test_add_custom_alias_collision_with_official_alias_fails(tmp_path):
    path = _write_mapping(tmp_path, _base_mapping())
    with pytest.raises(ConfigError, match="rtx_4090"):
        add_custom_alias(path, "rtx_4090", "RTX 4090")
    # Nothing was written.
    assert "custom" in json.loads(path.read_text(encoding="utf-8"))
    assert "rtx_4090" not in json.loads(path.read_text(encoding="utf-8"))["custom"]


@pytest.mark.parametrize("bad", ["BadAlias", "bad-alias", "-leading", "with space", "UPPER"])
def test_add_custom_alias_invalid_regex_fails(tmp_path, bad):
    path = _write_mapping(tmp_path, _base_mapping())
    with pytest.raises(ConfigError, match="must match"):
        add_custom_alias(path, bad, "Some GPU")
    with pytest.raises(ConfigError, match="non-empty gpu_name"):
        add_custom_alias(path, "ok_alias", "")


def test_load_mapping_rejects_invalid_alias_key(tmp_path):
    data = _base_mapping()
    data["aliases"]["Bad-Alias"] = {"gpu_name": "RTX 4090"}
    path = _write_mapping(tmp_path, data)
    with pytest.raises(ConfigError, match="alias must match"):
        load_mapping(path)


def test_load_mapping_rejects_unknown_official_gpu_name(tmp_path):
    data = _base_mapping()
    data["aliases"]["not_official"] = {"gpu_name": "Not In Snapshot"}
    path = _write_mapping(tmp_path, data)
    with pytest.raises(ConfigError, match="is not in official_names"):
        load_mapping(path)


def test_load_mapping_rejects_custom_official_overlap(tmp_path):
    data = _base_mapping()
    data["custom"]["rtx_4090"] = {"gpu_name": "RTX 4090"}
    path = _write_mapping(tmp_path, data)
    with pytest.raises(ConfigError, match="collides|share keys"):
        load_mapping(path)


def test_load_mapping_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="mapping not found"):
        load_mapping(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Update tool: union names, never drops official names, preserves custom
# ---------------------------------------------------------------------------


def test_update_mapping_file_unions_and_preserves(tmp_path):
    path = _write_mapping(tmp_path, _base_mapping())
    result = update_mapping_file(
        path,
        fetch=lambda: ["RTX 4060 Ti", "RTX 5090"],  # 4090 is "out of stock" and absent
        now_utc="2026-10-15T00:00:00Z",
    )

    assert result["new_names"] == ["RTX 5090"]
    assert result["fetched_names"] == 2
    assert result["official_names_total"] == 3
    assert result["custom_total"] == 1

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["official_names"] == ["RTX 4060 Ti", "RTX 4090", "RTX 5090"]
    assert updated["fetched_at_utc"] == "2026-10-15T00:00:00Z"
    # Old aliases survive even though the model was not in the fetch.
    assert updated["aliases"]["rtx_4090"] == {"gpu_name": "RTX 4090"}
    # VRAM variants are regenerated for models that need them.
    assert updated["aliases"]["rtx_4060_ti_16gb"]["gpu_ram_mb_min"] == 15500
    # Custom aliases are preserved verbatim.
    assert updated["custom"] == {"my_private_rig": {"gpu_name": "Private Rig 9000", "notes": "keep me"}}
    # And the file is still a valid mapping.
    assert load_mapping(path).resolve("my_private_rig").custom is True


def test_update_mapping_file_never_invents_aliases_for_missing_names(tmp_path):
    path = _write_mapping(tmp_path, _base_mapping())
    update_mapping_file(path, fetch=lambda: ["RTX 5090"], now_utc="2026-10-15T00:00:00Z")
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert "RTX 4060 Ti" in updated["official_names"]
    assert "rtx_4060_ti_16gb" in updated["aliases"]


def test_update_mapping_file_keeps_existing_notes(tmp_path):
    data = _base_mapping()
    data["aliases"]["rtx_4090"]["notes"] = "hand-annotated"
    path = _write_mapping(tmp_path, data)
    update_mapping_file(path, fetch=lambda: ["RTX 4090"], now_utc="2026-10-15T00:00:00Z")
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["aliases"]["rtx_4090"]["notes"] == "hand-annotated"


def test_build_official_aliases_disambiguates_duplicate_slugs():
    names = ["RTX 4090", "RTX-4090"]
    aliases = build_official_aliases(names)
    assert set(aliases) == {"rtx_4090", "rtx_4090_2"}
    assert slugify_gpu_name("RTX 4090") == "rtx_4090"
