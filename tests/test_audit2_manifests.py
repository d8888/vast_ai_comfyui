"""Strict manifest validation must fail before upload, payment or remote install."""
import copy
from unittest.mock import Mock

import pytest

from vast_comfy import create_flow, prepare
from vast_comfy.errors import VastComfyError


@pytest.mark.parametrize("mutation", [
    lambda d: d["models"][0].update(id="../../escape"),
    lambda d: d["models"][0].update(id="/absolute"),
    lambda d: d["models"][0].update(id="C:\\escape"),
    lambda d: d["models"][0]["source"].update(revision="main"),
    lambda d: d["plugins"][0].update(commit="main"),
    lambda d: d["models"][0].update(size_bytes=1.9),
    lambda d: d["models"][0].update(size_bytes=True),
    lambda d: d["models"][0].update(size_bytes=-1),
    lambda d: d["models"][0].update(unknown="value"),
    lambda d: d["models"][0]["source"].update(unknown="value"),
    lambda d: d["models"].append(copy.deepcopy(d["models"][0])),
    lambda d: d["models"][1].update(dest=d["models"][0]["dest"]),
    lambda d: d["plugins"][0].update(dest="custom_nodes/.."),
    lambda d: d["plugins"][0].update(required_nodes="Node"),
    lambda d: d.update(schema_version=True),
    lambda d: d.update(unknown="value"),
])
def test_invalid_manifest_fails_before_any_remote_operation(monkeypatch, mutation):
    doc = copy.deepcopy(prepare.load_models_config()); mutation(doc)
    monkeypatch.setattr(prepare, "load_models_config", lambda: doc)
    with pytest.raises(VastComfyError):
        prepare.build_install_manifest()


@pytest.mark.parametrize("where,value", [
    ("path", "file.safetensors?token=synthetic-sensitive-value"),
    ("path", "file.safetensors%3ftoken=synthetic-sensitive-value"),
    ("repo", "user:synthetic-sensitive-value@huggingface.co/repo"),
    ("plugin", "https://user:synthetic-sensitive-value@github.com/org/plugin"),
    ("plugin", "https://github.com/org/plugin?token=synthetic-sensitive-value"),
    ("plugin", "file:///private/synthetic-sensitive-value"),
])
def test_url_credentials_fail_before_payment_and_are_not_echoed(cfg, monkeypatch, where, value):
    doc = copy.deepcopy(prepare.load_models_config())
    if where == "plugin": doc["plugins"][0]["repo_url"] = value
    else: doc["models"][0]["source"][where] = value
    monkeypatch.setattr(prepare, "load_models_config", lambda: doc)
    client = Mock()
    with pytest.raises(VastComfyError) as error:
        create_flow.create_and_prepare(client, cfg, alias="rtx_4060_ti_16gb", confirm_paid=True)
    assert "synthetic-sensitive-value" not in str(error.value.to_dict())
    client.create_instance.assert_not_called()
    client.search_offers.assert_not_called()


@pytest.mark.parametrize("text", [
    "schema_version: 1\nschema_version: 1\nmodels: []\nplugins: []\n",
    "schema_version: 1\nmodels: []\nmodels: []\nplugins: []\n",
])
def test_manifest_duplicate_yaml_keys_rejected(tmp_path, monkeypatch, text):
    path = tmp_path / "models.yaml"; path.write_text(text)
    monkeypatch.setattr(prepare, "_models_config_path", lambda: path)
    with pytest.raises(VastComfyError, match="Duplicate YAML key"):
        prepare.load_models_config()


@pytest.mark.parametrize("run_id", ["../outside", "/absolute", "C:\\outside", "..", "", "CON"])
def test_bad_run_id_fails_before_lock_and_payment(cfg, monkeypatch, run_id):
    lock = Mock(); monkeypatch.setattr(prepare.locks, "acquire", lock)
    with pytest.raises(VastComfyError):
        prepare.prepare_existing(Mock(), cfg, instance_id=1, run_id=run_id)
    lock.assert_not_called()
    client = Mock()
    with pytest.raises(VastComfyError):
        create_flow.create_and_prepare(client, cfg, alias="rtx_4060_ti_16gb", confirm_paid=True, run_id=run_id)
    client.create_instance.assert_not_called()
    client.search_offers.assert_not_called()


@pytest.mark.parametrize("cases", ["base_t2i,typo", "base_t2i,base_t2i", "", "turbo_t2i,"])
def test_invalid_case_selection_fails_before_lock(cfg, monkeypatch, cases):
    lock = Mock(); monkeypatch.setattr(prepare.locks, "acquire", lock)
    with pytest.raises(VastComfyError):
        prepare.prepare_existing(Mock(), cfg, instance_id=1, cases=cases)
    lock.assert_not_called()


@pytest.mark.parametrize("name", ["show_report", "download_report", "download_smoke_images"])
def test_api_bad_report_id_never_loads_runner_or_acquires_lock(cfg, monkeypatch, name):
    from vast_comfy import api
    lookup = Mock(); monkeypatch.setattr(api, "_runner", lookup)
    args = (cfg, 1, "../outside") if name == "show_report" else (cfg, 1, "../outside", ".")
    with pytest.raises(VastComfyError):
        getattr(api, name)(*args, session_id="S")
    lookup.assert_not_called()


def test_report_display_refuses_remote_symlink_escape(cfg, monkeypatch):
    from types import SimpleNamespace
    from vast_comfy import smoke
    runner = Mock()
    runner.run_remote.return_value = SimpleNamespace(rc=0,
        stdout="/workspace/.vast_comfy\n/workspace/user-secrets/report.json\n")
    read = Mock(); monkeypatch.setattr(smoke.remote, "read_remote_json", read)
    with pytest.raises(VastComfyError):
        smoke.show_report(runner, cfg, "safe-id")
    read.assert_not_called()


def test_unsupported_fixed_seed_configuration_fails_before_payment():
    from vast_comfy.config import load_config
    with pytest.raises(VastComfyError, match="fixed-seed"):
        load_config(set_overrides={"smoke.unique_seed_per_run": False})


def test_configured_smoke_case_timeout_reaches_the_dispatched_cases():
    from vast_comfy.config import load_config

    tuned = load_config(set_overrides={"jobs.smoke_case_timeout_minutes": 3})
    _, _, doc = prepare.smoke_request(tuned, run_id="timeout-check")
    assert {case["timeout_minutes"] for case in doc["cases"]} == {3}
    _, _, default_doc = prepare.smoke_request(load_config(), run_id="timeout-default")
    assert {case["timeout_minutes"] for case in default_doc["cases"]} == {15}
