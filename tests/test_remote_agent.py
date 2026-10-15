"""SPEC 17 items 2, 11 and 16: remote-agent pure logic + install manifest building.

The remote scripts are imported exactly as they are deployed (top-level modules
with ``import vc_common as C``); see the ``remote_agent`` fixture in conftest.
"""

from __future__ import annotations

import base64
import copy
import json

import pytest
import yaml

from vast_comfy.errors import ConfigError
from vast_comfy.prepare import build_install_manifest, build_models_payload, load_models_config, load_smoke_cases
from vast_comfy.remote import _validate_tool_name
from vast_comfy.errors import VastComfyError


def b64(payload) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------------------
# Path validation (vc_common)
# ---------------------------------------------------------------------------


def test_validate_model_relpath_accepts_allowed_paths(remote_agent):
    validate = remote_agent.C.validate_model_relpath
    assert validate("diffusion_models/x.safetensors") == "diffusion_models/x.safetensors"
    assert validate("controlnet/sub/dir/w.safetensors") == "controlnet/sub/dir/w.safetensors"
    assert validate("vae/qwen_image_vae.safetensors") == "vae/qwen_image_vae.safetensors"


@pytest.mark.parametrize(
    "candidate",
    [
        "/abs/x.safetensors",
        "../x.safetensors",
        "diffusion_models/../x.safetensors",
        "foo/bar.safetensors",
        "diffusion_models",
        "diffusion_models/",
        "models\\x.safetensors",
        "~/.cache/x.safetensors",
        "",
    ],
)
def test_validate_model_relpath_rejects_unsafe_paths(remote_agent, candidate):
    with pytest.raises(ValueError):
        remote_agent.C.validate_model_relpath(candidate)


def test_validate_plugin_relpath(remote_agent):
    validate = remote_agent.C.validate_plugin_relpath
    assert validate("custom_nodes/Something") == "custom_nodes/Something"
    assert validate("custom_nodes/ComfyUI-Anima-LLLite") == "custom_nodes/ComfyUI-Anima-LLLite"
    for bad in ("custom_nodes/../x", "/custom_nodes/x", "custom_nodes/", "custom_nodes/a/b", "custom_nodes/.hidden", "other/x"):
        with pytest.raises(ValueError):
            validate(bad)


def test_pid_alive(remote_agent):
    import os
    import sys

    assert remote_agent.C.pid_alive(os.getpid()) is True
    assert remote_agent.C.pid_alive(None) is False
    assert remote_agent.C.pid_alive(0) is False
    assert remote_agent.C.pid_alive(-5) is False
    if sys.platform == "linux":
        # On Linux a dead pid raises ProcessLookupError, which is mapped to False.
        assert remote_agent.C.pid_alive(999999999) is False


# ---------------------------------------------------------------------------
# vc_smoke: seeds, prompt building, case payload validation
# ---------------------------------------------------------------------------


def test_compute_seed_is_unique_per_case_and_run(remote_agent):
    compute = remote_agent.SMOKE.compute_seed
    assert compute(1000, 7, 0, True) != compute(1000, 7, 1, True)  # per case
    assert compute(1000, 7, 0, True) != compute(1000, 8, 0, True)  # per run
    assert compute(1000, 7, 0, True) != compute(1001, 7, 0, True)
    # Fixed seed mode is identical for every case/run.
    assert compute(12345, 7, 0, False) == 12345
    assert compute(12345, 999, 3, False) == 12345


def test_build_prompt_replaces_run_seed(remote_agent, repo_root):
    workflow = repo_root / "workflows" / "api" / "anima_base_t2i.api.json"
    original = workflow.read_text(encoding="utf-8")
    prompt = remote_agent.SMOKE.build_prompt(workflow, {"6": {"seed": "$run_seed"}}, 987654)
    assert prompt["6"]["inputs"]["seed"] == 987654
    # The committed workflow on disk is never modified.
    assert workflow.read_text(encoding="utf-8") == original


def test_build_prompt_merges_overrides_and_keeps_other_inputs(remote_agent, repo_root):
    workflow = repo_root / "workflows" / "api" / "anima_base_t2i.api.json"
    prompt = remote_agent.SMOKE.build_prompt(workflow, {"6": {"seed": "$run_seed", "steps": 9}}, 5)
    assert prompt["6"]["inputs"]["steps"] == 9
    assert prompt["6"]["inputs"]["cfg"] == 4.0
    assert prompt["7"]["inputs"]["width"] == 1024


def test_build_prompt_missing_node_raises_case_failure(remote_agent, repo_root):
    workflow = repo_root / "workflows" / "api" / "anima_base_t2i.api.json"
    with pytest.raises(remote_agent.SMOKE.CaseFailure) as excinfo:
        remote_agent.SMOKE.build_prompt(workflow, {"9999": {"seed": 1}}, 5)
    assert excinfo.value.reason == "SMOKE_WORKFLOW"


def test_build_prompt_missing_workflow_raises_case_failure(remote_agent, tmp_path):
    with pytest.raises(remote_agent.SMOKE.CaseFailure) as excinfo:
        remote_agent.SMOKE.build_prompt(tmp_path / "nope.json", {}, 5)
    assert excinfo.value.reason == "SMOKE_WORKFLOW"


def test_load_cases_accepts_the_committed_asset(remote_agent, repo_root):
    doc = yaml.safe_load((repo_root / "assets" / "smoke" / "smoke_cases.yaml").read_text(encoding="utf-8"))
    cases = remote_agent.SMOKE.load_cases(b64(doc))
    assert [c["id"] for c in cases] == ["base_t2i", "lllite_lineart", "lllite_inpaint", "turbo_t2i"]


@pytest.mark.parametrize(
    "payload",
    [
        "not-base64!!!",
        b64({"cases": []}),
        b64({"cases": [{"id": "x"}]}),
        b64([{"id": "x", "workflow": "w.json", "timeout_minutes": 1, "overrides": {}, "expect": {}, "inputs": []}, "junk"]),
    ],
)
def test_load_cases_rejects_malformed_payloads(remote_agent, payload):
    with pytest.raises(remote_agent.C.RemoteFail) as excinfo:
        remote_agent.SMOKE.load_cases(payload)
    assert excinfo.value.code == "E_USAGE"


def test_fail_helpers_build_remote_fail(remote_agent):
    with pytest.raises(remote_agent.C.RemoteFail) as excinfo:
        remote_agent.C.fail("E_TEST", "boom", exit_code=4, extra="x")
    assert excinfo.value.code == "E_TEST"
    assert excinfo.value.exit_code == 4
    assert excinfo.value.details == {"extra": "x"}


# ---------------------------------------------------------------------------
# Controller-side: build_install_manifest (URL building + rejection)
# ---------------------------------------------------------------------------


def _manifest_doc() -> dict:
    return {
        "schema_version": 1,
        "models": [
            {
                "id": "hf-model",
                "source": {
                    "type": "huggingface",
                    "repo": "circlestone-labs/Anima",
                    "revision": "f973fc41ec7545364ac9776c2440285f43ff2a30",
                    "path": "split_files/vae/qwen_image_vae.safetensors",
                },
                "dest": "vae/qwen_image_vae.safetensors",
                "sha256": "a" * 64,
                "size_bytes": 123,
            },
            {
                "id": "civitai-model",
                "source": {"type": "civitai", "version_id": 98765, "auth_env": "CIVITAI_TOKEN"},
                "dest": "loras/fun.safetensors",
                "sha256": "b" * 64,
            },
        ],
        "plugins": [
            {
                "id": "lllite",
                "repo_url": "https://github.com/kohya-ss/ComfyUI-Anima-LLLite",
                "commit": "b7495bd8eb876e334509976896702484ed19cdbb",
                "dest": "custom_nodes/ComfyUI-Anima-LLLite",
                "required_nodes": ["AnimaLLLiteApply_sdscripts"],
            }
        ],
    }


def _with_doc(monkeypatch, doc):
    monkeypatch.setattr("vast_comfy.prepare.load_models_config", lambda: doc)


def test_build_install_manifest_builds_huggingface_and_civitai_urls(monkeypatch):
    _with_doc(monkeypatch, _manifest_doc())
    manifest = build_install_manifest()

    hf = manifest["models"][0]
    assert hf["url"] == (
        "https://huggingface.co/circlestone-labs/Anima/resolve/"
        "f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/vae/qwen_image_vae.safetensors"
    )
    assert hf["dest"] == "vae/qwen_image_vae.safetensors"
    assert hf["sha256"] == "a" * 64 and hf["size_bytes"] == 123

    civitai = manifest["models"][1]
    assert civitai["url"] == "https://civitai.com/api/download/models/98765"
    assert civitai["auth_env"] == "CIVITAI_TOKEN"

    plugin = manifest["plugins"][0]
    assert plugin["dest"] == "custom_nodes/ComfyUI-Anima-LLLite"
    assert plugin["required_nodes"] == ["AnimaLLLiteApply_sdscripts"]

    payload = build_models_payload(manifest)
    assert payload[0]["id"] == "hf-model"
    assert payload[0]["sha256"] == "a" * 64
    assert payload[0]["size_bytes"] == 123


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["models"][0].update(sha256="zz"), "sha256 must be 64 hex"),
        (lambda d: d["models"][0].update(sha256="a" * 63), "sha256 must be 64 hex"),
        (lambda d: d["models"][0].update(dest="../escape/x.safetensors"), "unsafe model dest"),
        (lambda d: d["models"][0].update(dest="/abs/x.safetensors"), "unsafe model dest"),
        (lambda d: d["models"][0].update(dest="notallowed/x.safetensors"), "allowed directory"),
        (lambda d: d["models"][0].update(dest="vae"), "unsafe model dest"),
        (lambda d: d["models"][0].update(source={"type": "ftp"}), "unsupported source type"),
        (lambda d: d["models"][0]["source"].pop("revision"), "huggingface source requires"),
        (lambda d: d["models"][1]["source"].pop("version_id"), "civitai source requires"),
        (lambda d: d["plugins"][0].update(dest="custom_nodes/../x"), "unsafe dest"),
        (lambda d: d["plugins"][0].update(commit=None), "missing required fields"),
        (lambda d: d["plugins"][0].update(repo_url=""), "missing required fields"),
    ],
)
def test_build_install_manifest_rejects_invalid_entries(monkeypatch, mutate, match):
    doc = copy.deepcopy(_manifest_doc())
    mutate(doc)
    _with_doc(monkeypatch, doc)
    with pytest.raises(ConfigError, match=match):
        build_install_manifest()


def test_committed_models_yaml_builds_a_valid_manifest():
    doc = load_models_config()
    assert doc["schema_version"] == 1
    manifest = build_install_manifest()

    ids = [m["id"] for m in manifest["models"]]
    assert ids == [
        "anima-base-v1.0",
        "anima-turbo-v1.1",
        "qwen-3-0.6b-text-encoder",
        "qwen-image-vae",
        "anima-lllite-any-test-like-v2",
        "anima-lllite-inpainting-v2",
    ]
    for model in manifest["models"]:
        assert model["url"].startswith("https://")
        assert len(model["sha256"]) == 64
        assert int(model["size_bytes"]) > 0

    plugin = manifest["plugins"][0]
    assert plugin["repo_url"] == "https://github.com/kohya-ss/ComfyUI-Anima-LLLite"
    assert plugin["required_nodes"] == ["AnimaLLLiteApply_sdscripts"]
    assert len(plugin["commit"]) == 40


def test_committed_smoke_cases_have_the_four_required_cases(repo_root):
    doc = load_smoke_cases()
    cases = doc["cases"]
    assert [c["id"] for c in cases] == ["base_t2i", "lllite_lineart", "lllite_inpaint", "turbo_t2i"]
    for case in cases:
        assert case["expect"] == {"width": 1024, "height": 1024}
        assert case["overrides"]["6"]["seed"] == "$run_seed"
    # Turbo runs last with the fast 8-step / CFG 1 contract.
    turbo = json.loads(
        (repo_root / "workflows" / "api" / "anima_turbo_t2i.api.json").read_text(encoding="utf-8")
    )
    assert turbo["6"]["inputs"]["steps"] == 8
    assert turbo["6"]["inputs"]["cfg"] == 1.0
    # The two LLLite cases use the plugin node and distinct weights.
    lineart = json.loads(
        (repo_root / "workflows" / "api" / "anima_base_lllite_lineart.api.json").read_text(encoding="utf-8")
    )
    inpaint = json.loads(
        (repo_root / "workflows" / "api" / "anima_base_lllite_inpaint.api.json").read_text(encoding="utf-8")
    )
    assert lineart["11"]["class_type"] == "AnimaLLLiteApply_sdscripts"
    assert inpaint["11"]["class_type"] == "AnimaLLLiteApply_sdscripts"
    assert "lllite-any-test-like" in lineart["11"]["inputs"]["lllite_name"]
    assert "lllite-inpainting" in inpaint["11"]["inputs"]["lllite_name"]
    assert "mask" in inpaint["11"]["inputs"]  # inpainting gets a real mask input


def test_validate_tool_name_rejects_path_traversal():
    _validate_tool_name("vc_smoke.py")
    for bad in ("../vc_smoke.py", "vc_smoke", "vc_smoke.sh", "/bin/sh"):
        with pytest.raises(VastComfyError):
            _validate_tool_name(bad)
