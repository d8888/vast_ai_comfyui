"""SPEC 17 item 19: JSON/CLI contract — stable codes, exit codes, clean stdout."""

from __future__ import annotations

import json

import pytest

from vast_comfy import cli
from vast_comfy.errors import (
    E_APPROVAL_REQUIRED,
    E_LOCK_CONFLICT,
    E_SMOKE_FAILED,
    exit_code_for,
)


def test_config_validate_prints_exactly_one_json_object(capsys):
    rc = cli.main(["--json", "config", "validate"])

    captured = capsys.readouterr()
    assert rc == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1, f"stdout must carry exactly one JSON line, got: {captured.out!r}"
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["command"] == "config validate"
    assert payload["valid"] is True
    assert payload["checks"]["smoke_cases"] == ["base_t2i", "lllite_lineart", "lllite_inpaint", "turbo_t2i"]
    assert payload["checks"]["models"] == 6
    assert payload["checks"]["plugins"] == 1


def test_config_show_json_is_machine_readable(capsys):
    rc = cli.main(["--json", "config", "show"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out.strip())
    assert payload["config"]["image"]["tag"] == "v0.40.0-cuda-12.9-py312"
    assert payload["config"]["vast"]["api_key_env"] == "VAST_API_KEY"


def test_unknown_alias_json_error_and_exit_code(capsys):
    rc = cli.main(["--json", "gpu", "resolve", "--alias", "nope"])

    captured = capsys.readouterr()
    assert rc == 1  # non-zero, grouped as a generic failure
    payload = json.loads(captured.out.strip())
    assert payload["ok"] is False
    assert payload["code"] == "E_UNKNOWN_ALIAS"
    assert payload["details"]["alias"] == "nope"


def test_known_alias_resolves_to_one_json_object(capsys):
    rc = cli.main(["--json", "gpu", "resolve", "--alias", "rtx_4060_ti_16gb"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out.strip())
    assert payload["ok"] is True
    assert payload["gpu_ram_mb_min"] == 15500
    assert payload["gpu_ram_mb_max"] == 17000


def test_usage_error_exits_2(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--json", "gpu", "resolve"])  # --alias is required
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "alias" in captured.err


def test_usage_error_via_handler_exits_2(capsys):
    # `instance destroy` without --confirm-destroy is a UsageError -> exit 2.
    rc = cli.main(["--json", "instance", "destroy", "--instance", "1"])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["code"] == "E_USAGE"


@pytest.mark.parametrize(
    "code, expected",
    [
        (E_APPROVAL_REQUIRED, 3),
        (E_LOCK_CONFLICT, 4),
        (E_SMOKE_FAILED, 5),
        ("E_LOCK_STALE", 4),
        (None, 0),
        ("E_CONFIG", 1),
        ("E_NO_OFFERS", 1),
        ("E_UNKNOWN_ALIAS", 1),
    ],
)
def test_exit_code_mapping(code, expected):
    assert exit_code_for(code) == expected


def test_json_error_keeps_stdout_clean(capsys):
    rc = cli.main(["--json", "config", "validate", "--set", "image.tag=latest"])
    captured = capsys.readouterr()
    assert rc == 1
    payload = json.loads(captured.out.strip())  # exactly one object
    assert payload["ok"] is False
    assert payload["code"] == "E_CONFIG"
    assert "latest" in payload["message"]


def test_human_mode_writes_errors_to_stderr_only(capsys):
    rc = cli.main(["gpu", "resolve", "--alias", "nope"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "E_UNKNOWN_ALIAS" in captured.err
