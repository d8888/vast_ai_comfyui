"""SPEC 17 items 1 and 2 (config part): strict config validation & precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from vast_comfy.config import (
    DEFAULTS,
    builtin_defaults_match_reference_file,
    load_config,
    set_dotted,
)
from vast_comfy.errors import ConfigError


def write_yaml(tmp_path: Path, text: str, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Duplicate keys / unknown fields / wrong types
# ---------------------------------------------------------------------------


def test_duplicate_yaml_keys_fail(tmp_path):
    path = write_yaml(
        tmp_path,
        "version: 1\nlock:\n  heartbeat_seconds: 30\n  heartbeat_seconds: 40\n",
    )
    with pytest.raises(ConfigError, match="Duplicate YAML key"):
        load_config(path)


def test_unknown_field_fails(tmp_path):
    path = write_yaml(tmp_path, "version: 1\ngpu:\n  alias: rtx_4090\n  bogus_field: 1\n")
    with pytest.raises(ConfigError, match=r"Unknown config field: gpu\.bogus_field"):
        load_config(path)


def test_unknown_top_level_field_fails(tmp_path):
    path = write_yaml(tmp_path, "version: 1\nmystery: {}\n")
    with pytest.raises(ConfigError, match=r"Unknown config field: mystery"):
        load_config(path)


def test_wrong_type_fails(tmp_path):
    path = write_yaml(tmp_path, "version: 1\nstorage:\n  disk_gb: \"big\"\n")
    with pytest.raises(ConfigError, match=r"storage\.disk_gb must be a number"):
        load_config(path)


def test_non_mapping_top_level_fails(tmp_path):
    path = write_yaml(tmp_path, "- just\n- a list\n")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(path)


def test_invalid_yaml_syntax_fails(tmp_path):
    path = write_yaml(tmp_path, "version: 1\nimage: [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_missing_file_fails(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Contradictory / invalid values (fail before any side effect)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, match",
    [
        ("ssh:\n  strategy: proxy_first\n", "ssh.strategy must be 'proxy_only'"),
        ("ssh:\n  strategy: direct_only\n", "ssh.strategy must be 'proxy_only'"),
        ("image:\n  tag: latest\n", "image.tag must be a fixed version"),
        ("lock:\n  heartbeat_seconds: 300\n", "heartbeat_seconds must be smaller"),
        ("lock:\n  heartbeat_seconds: 400\n", "heartbeat_seconds must be smaller"),
        ("lock:\n  heartbeat_buffer_seconds: 301\n", "must not exceed"),
        ("host:\n  verified: false\n", "host.verified must be true"),
        ("pricing:\n  rental_type: reserved\n", "pricing.rental_type must be 'ondemand'"),
        ("gpu:\n  count: 0\n", "gpu.count must be >= 1"),
        ("gpu:\n  gpu_arch: amd\n", "gpu.gpu_arch must be 'nvidia'"),
        ("gpu:\n  cpu_arch: arm64\n", "gpu.cpu_arch must be 'amd64'"),
        ("network:\n  unit_contract: MBps\n", "network.unit_contract must be 'mbps_platform_reported'"),
        ("image:\n  comfyui_args: \"--disable-auto-launch\"\n", "--port 18188"),
        ("image:\n  internal_port: 99999\n", "image.internal_port"),
        ("storage:\n  disk_gb: 5\n", "storage.disk_gb must be >= 10"),
        ("vast:\n  api_base_url: \"ftp://x\"\n", "must be an http"),
        ("smoke:\n  seed: -1\n", "smoke.seed must be >= 0"),
        ("host:\n  geolocation_allow: [\"JPN\"]\n", "two-letter country codes"),
        ("gpu:\n  alias: \"Bad-Alias\"\n", "gpu.alias must match"),
    ],
)
def test_invalid_values_fail(tmp_path, body, match):
    path = write_yaml(tmp_path, "version: 1\n" + body)
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_unsupported_version_fails(tmp_path):
    path = write_yaml(tmp_path, "version: 2\n")
    with pytest.raises(ConfigError, match="Unsupported config version"):
        load_config(path)


def test_unknown_set_path_fails():
    with pytest.raises(ConfigError, match="Unknown config field: nope.deep"):
        load_config(None, {"nope.deep": 1})
    with pytest.raises(ConfigError, match=r"Unknown config field: gpu\.title"):
        load_config(None, {"gpu.title": "x"})


def test_set_override_validates_types_and_semantics():
    with pytest.raises(ConfigError, match="must be a boolean"):
        load_config(None, {"host.verified": "yes"})
    with pytest.raises(ConfigError, match="image.tag"):
        load_config(None, {"image.tag": "latest"})
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(None, {"storage.disk_gb": "many"})


def test_set_dotted_rejects_unknown_paths():
    data = {"gpu": {"count": 1}}
    with pytest.raises(ConfigError, match="Unknown config field"):
        set_dotted(data, "gpu.nope", 1)
    with pytest.raises(ConfigError, match="Unknown config field"):
        set_dotted(data, "image.tag", "x")


# ---------------------------------------------------------------------------
# Precedence: built-in defaults -> YAML -> CLI --set
# ---------------------------------------------------------------------------


def test_precedence_defaults_yaml_cli(tmp_path):
    assert load_config().require("pricing.max_dph_total_usd_per_hour") == 2.0

    path = write_yaml(tmp_path, "version: 1\npricing:\n  max_dph_total_usd_per_hour: 1.5\n")
    assert load_config(path).require("pricing.max_dph_total_usd_per_hour") == 1.5

    merged = load_config(path, {"pricing.max_dph_total_usd_per_hour": 1.75})
    assert merged.require("pricing.max_dph_total_usd_per_hour") == 1.75
    # untouched fields still come from the defaults
    assert merged.require("storage.disk_gb") == 300


def test_defaults_reference_file_is_in_sync():
    assert builtin_defaults_match_reference_file() is True
    assert set(DEFAULTS) >= {"vast", "gpu", "image", "lock", "pricing", "smoke"}


def test_resolved_paths_default_under_home():
    config = load_config()
    paths = config.resolved_paths()
    root = Path.home() / ".vast_comfy"
    assert paths["root"] == root
    assert paths["state_dir"] == root / "state"
    assert paths["logs_dir"] == root / "logs"
    assert paths["known_hosts_path"] == root / "known_hosts"
    assert paths["key_path"] is None


def test_resolved_paths_respect_overrides(tmp_path):
    config = load_config(
        None,
        {
            "paths.root": str(tmp_path / "root"),
            "paths.logs_dir": str(tmp_path / "logz"),
            "ssh.key_path": str(tmp_path / "id_ed25519"),
            "ssh.known_hosts_path": str(tmp_path / "kh"),
        },
    )
    paths = config.resolved_paths()
    assert paths["root"] == tmp_path / "root"
    assert paths["state_dir"] == tmp_path / "root" / "state"
    assert paths["logs_dir"] == tmp_path / "logz"
    assert paths["key_path"] == tmp_path / "id_ed25519"
    assert paths["known_hosts_path"] == tmp_path / "kh"


def test_config_get_require_and_summary():
    config = load_config()
    assert config.get("does.not.exist", "fallback") == "fallback"
    with pytest.raises(ConfigError, match="Missing required config field"):
        config.require("does.not.exist")
    summary = config.non_secret_summary()
    # The summary only names the env var, never a secret value.
    assert summary["vast"]["api_key_env"] == "VAST_API_KEY"
    assert "sk-" not in str(summary)
