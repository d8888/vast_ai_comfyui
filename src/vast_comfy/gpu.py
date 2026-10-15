"""GPU name directory, alias mapping, resolution and updates (SPEC.md section 4).

The alias mapping lives in ``config/gpu_aliases.json``:

* ``official_names``: union of GPU names fetched from the Vast.ai directory.
  The update tool never removes names (a model missing from current inventory
  must not lose its alias).
* ``aliases``: generated aliases for official names (plus curated VRAM variants
  for models whose official name does not distinguish VRAM sizes).
* ``custom``: user-added aliases; preserved verbatim by updates.

Unknown aliases fail immediately; there is no fuzzy auto-correction.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import ConfigError, UnknownAliasError
from .paths import config_dir, find_repo_root

MAPPING_FILENAME = "gpu_aliases.json"
SCHEMA_VERSION = 1
SOURCE_URL = "https://console.vast.ai/api/v0/gpu_names/unique/"

ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

# Models whose official Vast.ai name does not distinguish VRAM size.
# Values: (alias suffix, gpu_ram_mb_min, gpu_ram_mb_max) — bounds cover the
# advertised VRAM with a small margin, they are always applied together with
# the exact gpu_name filter.
VRAM_VARIANTS: dict[str, list[tuple[str, int, int]]] = {
    "RTX 4060 Ti": [("8gb", 7500, 8600), ("16gb", 15500, 17000)],
    "RTX 5060 Ti": [("8gb", 7500, 8600), ("16gb", 15500, 17000)],
    "RTX 3060": [("8gb", 7500, 8600), ("12gb", 11500, 12600)],
    "RTX 3080": [("10gb", 9500, 10600), ("12gb", 11500, 12600)],
}


def slugify_gpu_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise ConfigError(f"GPU name cannot be slugified: {name!r}")
    if slug[0].isdigit():
        slug = f"gpu_{slug}"
    return slug


@dataclass
class AliasSpec:
    alias: str
    gpu_name: str
    gpu_ram_mb_min: int | None = None
    gpu_ram_mb_max: int | None = None
    custom: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"gpu_name": self.gpu_name}
        if self.gpu_ram_mb_min is not None:
            payload["gpu_ram_mb_min"] = self.gpu_ram_mb_min
        if self.gpu_ram_mb_max is not None:
            payload["gpu_ram_mb_max"] = self.gpu_ram_mb_max
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass
class GpuMapping:
    path: Path
    data: dict[str, Any]
    _aliases: dict[str, AliasSpec] = field(default_factory=dict)

    @property
    def official_names(self) -> list[str]:
        return list(self.data.get("official_names", []))

    @property
    def fetched_at_utc(self) -> str | None:
        return self.data.get("fetched_at_utc")

    def aliases(self) -> dict[str, AliasSpec]:
        return dict(self._aliases)

    def resolve(self, alias: str) -> AliasSpec:
        spec = self._aliases.get(alias)
        if spec is None:
            candidates = difflib.get_close_matches(alias, list(self._aliases.keys()), n=5, cutoff=0.6)
            raise UnknownAliasError(alias, candidates=candidates)
        return spec


def default_mapping_path(root: Path | None = None) -> Path:
    return config_dir(root or find_repo_root()) / MAPPING_FILENAME


def load_mapping(path: Path | str | None = None) -> GpuMapping:
    mapping_path = Path(path) if path else default_mapping_path()
    if not mapping_path.exists():
        raise ConfigError(
            f"GPU alias mapping not found: {mapping_path}. "
            "Run `vast-comfy gpu update-aliases` to create it from the Vast.ai GPU directory."
        )
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read GPU alias mapping {mapping_path}: {exc}") from exc

    errors = validate_mapping_data(data)
    if errors:
        raise ConfigError(
            "GPU alias mapping is invalid: " + "; ".join(errors),
            details={"path": str(mapping_path)},
        )

    alias_map: dict[str, AliasSpec] = {}
    for alias, entry in data.get("aliases", {}).items():
        alias_map[alias] = _spec_from_entry(alias, entry, custom=False)
    for alias, entry in data.get("custom", {}).items():
        if alias in alias_map:
            raise ConfigError(
                f"Custom alias {alias!r} collides with an official alias. "
                "Rename the custom alias; conflicts are never resolved automatically."
            )
        alias_map[alias] = _spec_from_entry(alias, entry, custom=True)
    return GpuMapping(path=mapping_path, data=data, _aliases=alias_map)


def _spec_from_entry(alias: str, entry: dict[str, Any], *, custom: bool) -> AliasSpec:
    return AliasSpec(
        alias=alias,
        gpu_name=str(entry["gpu_name"]),
        gpu_ram_mb_min=entry.get("gpu_ram_mb_min"),
        gpu_ram_mb_max=entry.get("gpu_ram_mb_max"),
        custom=custom or bool(entry.get("custom")),
        notes=entry.get("notes"),
    )


def validate_mapping_data(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    names = data.get("official_names")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        errors.append("official_names must be a non-empty list of strings")
        names = []
    name_set = set(names)

    for section, require_official in (("aliases", True), ("custom", False)):
        entries = data.get(section, {})
        if not isinstance(entries, dict):
            errors.append(f"{section} must be a mapping")
            continue
        for alias, entry in entries.items():
            where = f"{section}.{alias}"
            if not ALIAS_RE.match(alias):
                errors.append(f"{where}: alias must match {ALIAS_RE.pattern!r}")
            if not isinstance(entry, dict):
                errors.append(f"{where}: entry must be a mapping")
                continue
            gpu_name = entry.get("gpu_name")
            if not isinstance(gpu_name, str) or not gpu_name:
                errors.append(f"{where}: gpu_name is required")
            elif require_official and gpu_name not in name_set:
                errors.append(f"{where}: gpu_name {gpu_name!r} is not in official_names")
            lo, hi = entry.get("gpu_ram_mb_min"), entry.get("gpu_ram_mb_max")
            for key, val in (("gpu_ram_mb_min", lo), ("gpu_ram_mb_max", hi)):
                if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val <= 0):
                    errors.append(f"{where}: {key} must be a positive integer")
            if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
                errors.append(f"{where}: gpu_ram_mb_min > gpu_ram_mb_max")

    overlap = set(data.get("aliases", {})) & set(data.get("custom", {}))
    if overlap:
        errors.append(f"aliases and custom share keys: {sorted(overlap)}")
    return errors


# ---------------------------------------------------------------------------
# Snapshot generation / update
# ---------------------------------------------------------------------------


def fetch_official_names(url: str = SOURCE_URL, timeout_seconds: float = 30) -> list[str]:
    """Read the public GPU name directory (no API key required, read-only)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vast-comfy/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310 (fixed public URL)
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surfaced as structured error
        raise ConfigError(f"Cannot fetch GPU name directory from {url}: {exc}") from exc
    if isinstance(payload, dict):
        for key in ("gpu_names", "names", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(n, str) for n in payload):
        raise ConfigError(f"Unexpected GPU name directory payload from {url}: {type(payload).__name__}")
    return sorted({n for n in payload if n})


def build_official_aliases(
    names: list[str],
    existing_aliases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    existing_aliases = existing_aliases or {}
    aliases: dict[str, dict[str, Any]] = {}
    used: dict[str, str] = {}

    for name in names:
        slug = slugify_gpu_name(name)
        if slug in used and used[slug] != name:
            # Deterministic disambiguation; recorded so nothing is silent.
            n = 2
            while f"{slug}_{n}" in used:
                n += 1
            slug = f"{slug}_{n}"
        used[slug] = name
        entry: dict[str, Any] = {"gpu_name": name}
        prior = existing_aliases.get(slug)
        if prior and prior.get("notes"):
            entry["notes"] = prior["notes"]
        aliases[slug] = entry

        for suffix, lo, hi in VRAM_VARIANTS.get(name, []):
            alias = f"{slug}_{suffix}"
            variant = {"gpu_name": name, "gpu_ram_mb_min": lo, "gpu_ram_mb_max": hi}
            prior_v = existing_aliases.get(alias)
            if prior_v and prior_v.get("notes"):
                variant["notes"] = prior_v["notes"]
            aliases[alias] = variant

    return dict(sorted(aliases.items()))


def update_mapping_file(
    path: Path | str | None = None,
    *,
    fetch: Callable[[], list[str]] = fetch_official_names,
    now_utc: str | None = None,
) -> dict[str, Any]:
    """Refresh the official alias set; never deletes names or custom aliases."""
    mapping_path = Path(path) if path else default_mapping_path()
    existing: dict[str, Any] = {}
    if mapping_path.exists():
        try:
            existing = json.loads(mapping_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Existing mapping {mapping_path} is not valid JSON: {exc}") from exc

    fetched = fetch()
    old_names = existing.get("official_names", [])
    if not isinstance(old_names, list):
        old_names = []
    union_names = sorted(set(old_names) | set(fetched))

    custom = existing.get("custom", {})
    if not isinstance(custom, dict):
        raise ConfigError(f"Existing mapping {mapping_path}: 'custom' must be a mapping")

    aliases = build_official_aliases(union_names, existing.get("aliases", {}))
    custom_errors = validate_mapping_data(
        {
            "schema_version": SCHEMA_VERSION,
            "official_names": union_names,
            "aliases": {},
            "custom": custom,
        }
    )
    if custom_errors:
        raise ConfigError("Custom aliases are invalid: " + "; ".join(custom_errors))

    data = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_URL,
        "fetched_at_utc": now_utc or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "official_names": union_names,
        "aliases": aliases,
        "custom": custom,
    }
    errors = validate_mapping_data(data)
    if errors:
        raise ConfigError("Refusing to write an invalid mapping: " + "; ".join(errors))

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mapping_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(mapping_path)

    return {
        "path": str(mapping_path),
        "fetched_names": len(fetched),
        "official_names_total": len(union_names),
        "aliases_total": len(aliases),
        "custom_total": len(custom),
        "new_names": sorted(set(fetched) - set(old_names)),
    }


def add_custom_alias(
    path: Path | str | None,
    alias: str,
    gpu_name: str,
    gpu_ram_mb_min: int | None = None,
    gpu_ram_mb_max: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add (or update) a custom alias, preserving everything else."""
    if not ALIAS_RE.match(alias):
        raise ConfigError(f"Custom alias must match {ALIAS_RE.pattern!r}: {alias!r}")
    if not gpu_name:
        raise ConfigError("Custom alias requires a non-empty gpu_name")
    mapping_path = Path(path) if path else default_mapping_path()
    data = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_URL,
        "fetched_at_utc": None,
        "official_names": [],
        "aliases": {},
        "custom": {},
    }
    entry: dict[str, Any] = {"gpu_name": gpu_name}
    if gpu_ram_mb_min is not None:
        entry["gpu_ram_mb_min"] = int(gpu_ram_mb_min)
    if gpu_ram_mb_max is not None:
        entry["gpu_ram_mb_max"] = int(gpu_ram_mb_max)
    if notes:
        entry["notes"] = notes
    custom = dict(data.get("custom", {}))
    custom[alias] = entry
    data["custom"] = dict(sorted(custom.items()))
    errors = validate_mapping_data(data)
    if errors:
        raise ConfigError("Refusing to write an invalid mapping: " + "; ".join(errors))
    tmp = mapping_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(mapping_path)
    return {"alias": alias, **entry}
