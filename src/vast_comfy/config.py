"""Configuration loading, merging and strict validation (SPEC.md section 3.3).

Precedence: built-in defaults -> YAML (``--config``) -> explicit CLI options.

Strictness rules:
* unknown fields fail,
* duplicate YAML keys fail,
* type / range / cross-field violations fail,
* all failures happen before anything with side effects runs.
"""

from __future__ import annotations

import copy
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .paths import config_dir, find_repo_root

# ---------------------------------------------------------------------------
# Built-in defaults (kept in sync with config/default.yaml by a unit test)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "vast": {
        "api_base_url": "https://console.vast.ai",
        "api_key_env": "VAST_API_KEY",
        "request_timeout_seconds": 30,
        "transient_retries": 3,
        "retry_backoff_seconds": [1, 3, 8],
    },
    "gpu": {
        "alias": None,
        "count": 1,
        "compute_cap_min": 750,
        "cuda_max_good_min": 12.9,
        "gpu_arch": "nvidia",
        "cpu_arch": "amd64",
        "vram_gb_min": None,
    },
    "image": {
        "repository": "vastai/comfy",
        "tag": "v0.40.0-cuda-12.9-py312",
        "amd64_digest": "sha256:63ece702ee45b0b6748b28046b65005340eae4fa48db3bdba664cca1dacb948a",
        "comfyui_dir": "/workspace/ComfyUI",
        "venv_python": "/venv/main/bin/python",
        "supervisor_service": "comfyui",
        "internal_port": 18188,
        "comfy_external_port": 8188,
        "portal_external_port": 1111,
        "runtime_dir": "/workspace/.vast_comfy",
        "comfyui_args": "--disable-auto-launch --enable-cors-header --port 18188",
    },
    "storage": {"disk_gb": 300},
    "network": {
        "inet_down_min": 500,
        "inet_up_min": 500,
        "unit_contract": "mbps_platform_reported",
    },
    "host": {
        "verified": True,
        "reliability_min": 0.99,
        "cpu_cores_effective_min": 4,
        "cpu_ram_gb_min": 16,
        "geolocation_allow": [],
        "geolocation_exclude": [],
        "external": False,
        "rentable": True,
    },
    "offers": {"result_limit": 100},
    "pricing": {
        "rental_type": "ondemand",
        "max_dph_total_usd_per_hour": 2.0,
    },
    "ssh": {
        "strategy": "proxy_only",
        "user": "root",
        "key_path": None,
        "known_hosts_path": None,
        "connect_timeout_seconds": 15,
        "command_timeout_seconds": 120,
        "server_alive_interval_seconds": 20,
        "server_alive_count_max": 3,
        "scp_timeout_seconds": 3600,
    },
    "launch": {
        "start_wait_minutes": 45,
        "ssh_ready_wait_minutes": 10,
        "poll_interval_seconds": 10,
    },
    "lock": {
        "heartbeat_seconds": 30,
        "heartbeat_ttl_seconds": 300,
        "heartbeat_buffer_seconds": 60,
    },
    "jobs": {
        "install_timeout_minutes": 90,
        "smoke_case_timeout_minutes": 15,
        "wait_poll_seconds": 10,
        "log_tail_lines": 200,
    },
    "comfy": {"health_timeout_seconds": 180, "restart_wait_seconds": 240},
    "smoke": {"seed": 12345, "unique_seed_per_run": True},
    "session": {"release_on_success": True},
    "paths": {"root": None, "state_dir": None, "logs_dir": None},
    "download": {"default_dest": None},
}

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# YAML loading with duplicate-key detection
# ---------------------------------------------------------------------------


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ConfigError(
                    f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1} "
                    f"(column {key_node.start_mark.column + 1})"
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


# ---------------------------------------------------------------------------
# Minimal .env parsing (explicitly requested via --env-file)
# ---------------------------------------------------------------------------

_ENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a mapping. Values are NOT logged anywhere."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"Cannot read env file {path}: {exc}") from exc
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            raise ConfigError(f"Invalid line {lineno} in env file {path}")
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Merging / validation
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any], path: str = "") -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        key_path = f"{path}.{key}" if path else str(key)
        if key not in result:
            raise ConfigError(f"Unknown config field: {key_path}")
        if isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], dict(value), key_path)
        else:
            result[key] = value
    return result


def _type_name(expected: Any) -> str:
    if expected is None:
        return "null"
    return type(expected).__name__


def _validate_types(defaults: Any, data: Any, path: str) -> None:
    if isinstance(defaults, dict):
        if not isinstance(data, dict):
            raise ConfigError(f"Config field {path or '<root>'} must be a mapping")
        for key, value in data.items():
            if key not in defaults:
                raise ConfigError(f"Unknown config field: {path + '.' if path else ''}{key}")
            _validate_types(defaults[key], value, f"{path}.{key}" if path else str(key))
        return

    if isinstance(defaults, list):
        if not isinstance(data, list):
            raise ConfigError(f"Config field {path} must be a list")
        for i, item in enumerate(data):
            if defaults and isinstance(defaults[0], (int, float)) and not isinstance(defaults[0], bool):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ConfigError(f"Config field {path}[{i}] must be a number")
            elif defaults and isinstance(defaults[0], str) and not isinstance(item, str):
                raise ConfigError(f"Config field {path}[{i}] must be a string")
        return

    if isinstance(defaults, bool):
        if not isinstance(data, bool):
            raise ConfigError(f"Config field {path} must be a boolean")
        return
    if isinstance(defaults, int) and not isinstance(defaults, bool):
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"Config field {path} must be a number")
        return
    if isinstance(defaults, float):
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"Config field {path} must be a number")
        return
    if isinstance(defaults, str):
        if not isinstance(data, str):
            raise ConfigError(f"Config field {path} must be a string")
        return
    if defaults is None:
        # Nullable field: allow null or any scalar/list (validated semantically below)
        return
    raise ConfigError(f"Internal schema error for field {path}")  # pragma: no cover


def _require_number(data: dict[str, Any], path: str, *, minimum: float | None = None, maximum: float | None = None, integral: bool = False) -> float:
    value = data
    if value is None:
        raise ConfigError(f"Config field {path} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Config field {path} must be a number")
    if not math.isfinite(value):
        raise ConfigError(f"Config field {path} must be finite (NaN and infinity are rejected)")
    if integral and int(value) != value:
        raise ConfigError(f"Config field {path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"Config field {path} must be >= {minimum} (got {value})")
    if maximum is not None and value > maximum:
        raise ConfigError(f"Config field {path} must be <= {maximum} (got {value})")
    return value


def _validate_semantics(data: dict[str, Any]) -> None:
    if data.get("version") != 1:
        raise ConfigError(f"Unsupported config version: {data.get('version')!r} (expected 1)")

    vast = data["vast"]
    if not str(vast["api_base_url"]).startswith(("http://", "https://")):
        raise ConfigError("vast.api_base_url must be an http(s) URL")
    _require_number(vast["request_timeout_seconds"], "vast.request_timeout_seconds", minimum=1)
    _require_number(vast["transient_retries"], "vast.transient_retries", minimum=0, integral=True)
    for i, backoff in enumerate(vast["retry_backoff_seconds"]):
        _require_number(backoff, f"vast.retry_backoff_seconds[{i}]", minimum=0)

    gpu = data["gpu"]
    alias = gpu.get("alias")
    if alias is not None:
        if not isinstance(alias, str) or not _ALIAS_RE.match(alias):
            raise ConfigError(
                f"gpu.alias must match {_ALIAS_RE.pattern!r} (got {alias!r}); "
                "aliases are exact and case-sensitive"
            )
    _require_number(gpu["count"], "gpu.count", minimum=1, integral=True)
    _require_number(gpu["compute_cap_min"], "gpu.compute_cap_min", minimum=50, maximum=2000, integral=True)
    _require_number(gpu["cuda_max_good_min"], "gpu.cuda_max_good_min", minimum=0)
    if gpu["vram_gb_min"] is not None:
        _require_number(gpu["vram_gb_min"], "gpu.vram_gb_min", minimum=1)
    if gpu["gpu_arch"] not in ("nvidia",):
        raise ConfigError("gpu.gpu_arch must be 'nvidia' for v1 (NVIDIA CUDA only)")
    if gpu["cpu_arch"] not in ("amd64", "x86_64"):
        raise ConfigError("gpu.cpu_arch must be 'amd64' for v1")

    image = data["image"]
    if not image["repository"] or not image["tag"]:
        raise ConfigError("image.repository and image.tag must be set")
    if image["tag"] == "latest":
        raise ConfigError("image.tag must be a fixed version; 'latest' is rejected (SPEC.md section 6)")
    for port_key in ("internal_port", "comfy_external_port", "portal_external_port"):
        _require_number(image[port_key], f"image.{port_key}", minimum=1, maximum=65535, integral=True)
    for dir_key in ("comfyui_dir", "venv_python", "runtime_dir"):
        value = image[dir_key]
        if not isinstance(value, str) or not value.startswith("/"):
            raise ConfigError(f"image.{dir_key} must be an absolute Linux path")
    if f"--port {image['internal_port']}" not in str(image["comfyui_args"]):
        raise ConfigError(
            "image.comfyui_args must pass the pinned internal port "
            f"(expected '--port {image['internal_port']}')"
        )

    _require_number(data["storage"]["disk_gb"], "storage.disk_gb", minimum=10)

    network = data["network"]
    _require_number(network["inet_down_min"], "network.inet_down_min", minimum=0)
    _require_number(network["inet_up_min"], "network.inet_up_min", minimum=0)
    if network["unit_contract"] != "mbps_platform_reported":
        raise ConfigError(
            "network.unit_contract must be 'mbps_platform_reported' "
            "(values are compared with the platform-reported number, matching the official template)"
        )

    host = data["host"]
    if host["verified"] is not True:
        raise ConfigError("host.verified must be true (verified hosts are a hard requirement)")
    _require_number(host["reliability_min"], "host.reliability_min", minimum=0, maximum=1)
    _require_number(host["cpu_cores_effective_min"], "host.cpu_cores_effective_min", minimum=1)
    _require_number(host["cpu_ram_gb_min"], "host.cpu_ram_gb_min", minimum=1)
    for key in ("geolocation_allow", "geolocation_exclude"):
        value = host[key]
        if not isinstance(value, list) or any(not isinstance(x, str) or len(x) != 2 for x in value):
            raise ConfigError(f"host.{key} must be a list of two-letter country codes")
    if host["external"] not in (True, False):
        raise ConfigError("host.external must be a boolean")

    _require_number(data["offers"]["result_limit"], "offers.result_limit", minimum=1, maximum=1000, integral=True)

    pricing = data["pricing"]
    if pricing["rental_type"] != "ondemand":
        raise ConfigError("pricing.rental_type must be 'ondemand' for v1")
    _require_number(pricing["max_dph_total_usd_per_hour"], "pricing.max_dph_total_usd_per_hour", minimum=0.001)

    ssh = data["ssh"]
    if ssh["strategy"] != "proxy_only":
        raise ConfigError(
            "ssh.strategy must be 'proxy_only' for v1. "
            "'proxy_first' and 'direct_only' are explicit out-of-scope extensions (SPEC.md section 7.1)."
        )
    if ssh["user"] != "root":
        raise ConfigError("ssh.user must be 'root' for Vast.ai Docker instances")
    for key in ("key_path", "known_hosts_path"):
        if ssh[key] is not None and (not isinstance(ssh[key], str) or not ssh[key]):
            raise ConfigError(f"ssh.{key} must be a non-empty path string or null")
    _require_number(ssh["connect_timeout_seconds"], "ssh.connect_timeout_seconds", minimum=1)
    _require_number(ssh["command_timeout_seconds"], "ssh.command_timeout_seconds", minimum=1)
    _require_number(ssh["server_alive_interval_seconds"], "ssh.server_alive_interval_seconds", minimum=0)
    _require_number(ssh["server_alive_count_max"], "ssh.server_alive_count_max", minimum=1)
    _require_number(ssh["scp_timeout_seconds"], "ssh.scp_timeout_seconds", minimum=1)

    launch = data["launch"]
    _require_number(launch["start_wait_minutes"], "launch.start_wait_minutes", minimum=1)
    _require_number(launch["ssh_ready_wait_minutes"], "launch.ssh_ready_wait_minutes", minimum=1)
    _require_number(launch["poll_interval_seconds"], "launch.poll_interval_seconds", minimum=1)

    lock = data["lock"]
    heartbeat = _require_number(lock["heartbeat_seconds"], "lock.heartbeat_seconds", minimum=1)
    ttl = _require_number(lock["heartbeat_ttl_seconds"], "lock.heartbeat_ttl_seconds", minimum=1)
    buffer = _require_number(lock["heartbeat_buffer_seconds"], "lock.heartbeat_buffer_seconds", minimum=0)
    if heartbeat >= ttl:
        raise ConfigError("lock.heartbeat_seconds must be smaller than lock.heartbeat_ttl_seconds")
    if buffer > ttl:
        raise ConfigError("lock.heartbeat_buffer_seconds must not exceed lock.heartbeat_ttl_seconds")

    jobs = data["jobs"]
    _require_number(jobs["install_timeout_minutes"], "jobs.install_timeout_minutes", minimum=1)
    _require_number(jobs["smoke_case_timeout_minutes"], "jobs.smoke_case_timeout_minutes", minimum=1)
    _require_number(jobs["wait_poll_seconds"], "jobs.wait_poll_seconds", minimum=1)
    _require_number(jobs["log_tail_lines"], "jobs.log_tail_lines", minimum=1, integral=True)

    _require_number(data["comfy"]["health_timeout_seconds"], "comfy.health_timeout_seconds", minimum=10)
    _require_number(data["comfy"]["restart_wait_seconds"], "comfy.restart_wait_seconds", minimum=10)

    smoke = data["smoke"]
    _require_number(smoke["seed"], "smoke.seed", minimum=0, integral=True)
    if smoke["unique_seed_per_run"] not in (True, False):
        raise ConfigError("smoke.unique_seed_per_run must be a boolean")

    if data["session"]["release_on_success"] not in (True, False):
        raise ConfigError("session.release_on_success must be a boolean")

    for key in ("root", "state_dir", "logs_dir"):
        value = data["paths"][key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigError(f"paths.{key} must be a non-empty path string or null")


def set_dotted(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value by dotted path, validating the path exists in the schema."""
    keys = dotted_path.split(".")
    cursor: Any = data
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or key not in cursor or not isinstance(cursor[key], dict):
            raise ConfigError(f"Unknown config field: {dotted_path}")
        cursor = cursor[key]
    last = keys[-1]
    if not isinstance(cursor, dict) or last not in cursor:
        raise ConfigError(f"Unknown config field: {dotted_path}")
    if isinstance(cursor[last], bool) and not isinstance(value, bool):
        raise ConfigError(f"Config field {dotted_path} must be a boolean")
    cursor[last] = value


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


class Config:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def get(self, dotted_path: str, default: Any = None) -> Any:
        cursor: Any = self.data
        for key in dotted_path.split("."):
            if not isinstance(cursor, dict) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor

    def require(self, dotted_path: str) -> Any:
        sentinel = object()
        value = self.get(dotted_path, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required config field: {dotted_path}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def non_secret_summary(self) -> dict[str, Any]:
        """A JSON-safe summary that never contains secret values (only env var names)."""
        data = self.as_dict()
        return data

    # -- resolved paths -------------------------------------------------

    def resolved_paths(self) -> dict[str, Path]:
        root_cfg = self.get("paths.root")
        root = Path(root_cfg).expanduser() if root_cfg else Path.home() / ".vast_comfy"
        state = self.get("paths.state_dir")
        logs = self.get("paths.logs_dir")
        known_hosts = self.get("ssh.known_hosts_path")
        return {
            "root": root,
            "state_dir": Path(state).expanduser() if state else root / "state",
            "logs_dir": Path(logs).expanduser() if logs else root / "logs",
            "known_hosts_path": Path(known_hosts).expanduser() if known_hosts else root / "known_hosts",
            "key_path": Path(self.get("ssh.key_path")).expanduser() if self.get("ssh.key_path") else None,
        }


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    set_overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load and validate configuration.

    ``config_path``: optional user YAML (validated strictly against the schema).
    ``set_overrides``: dotted-path -> value overrides coming from explicit CLI flags.
    """
    data = copy.deepcopy(DEFAULTS)
    if config_path:
        yaml_data = _load_yaml(Path(config_path))
        data = _deep_merge(data, yaml_data)
    if set_overrides:
        # Apply CLI overrides on top of defaults->yaml, then re-validate everything.
        for dotted, value in set_overrides.items():
            set_dotted(data, dotted, value)
    _validate_types(DEFAULTS, {k: v for k, v in data.items() if k != "version"} | {"version": data["version"]}, "")
    _validate_semantics(data)
    return Config(data)


def load_config_from_defaults() -> Config:
    """Config built purely from built-in defaults (used by tests)."""
    return load_config()


def default_config_file() -> Path:
    return config_dir(find_repo_root()) / "default.yaml"


def builtin_defaults_match_reference_file() -> bool:
    """Test helper: verify config/default.yaml equals the built-in defaults."""
    try:
        reference = _load_yaml(default_config_file())
    except ConfigError:
        return False
    return reference == DEFAULTS


def apply_env_file(path: str | os.PathLike[str]) -> None:
    """Explicitly load a .env file into the process environment.

    Explicit user action: listed values override the current process environment.
    """
    values = load_env_file(Path(path))
    for key, value in values.items():
        os.environ[key] = value


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Required environment variable {name} is not set. "
            "Set it in your shell or pass --env-file .env (see .env.template)."
        )
    return value


def _main() -> int:  # pragma: no cover - manual smoke check
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    print(yaml.safe_dump(cfg.as_dict(), sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
