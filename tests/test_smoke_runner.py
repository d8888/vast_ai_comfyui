"""SPEC 17 items 16-17: the remote smoke runner (READY / FAILED / PARTIAL / busy).

The ComfyUI HTTP layer is replaced with ``tests/mocks/fake_comfy.py``; the real
committed workflows and the real smoke cases YAML are used throughout. Every
output PNG is a real file written into the tmp ComfyUI output directory.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mocks.fake_comfy import (
    FakeComfyUI,
    flat_png,
    gpu_check_stub,
    noise_png,
    old_png,
    undecodable_png,
    wrong_size_png,
)
from vast_comfy.prepare import load_smoke_cases

BASE_URL = "http://localhost:18188"
RUN_TOKEN = 7777
BASE_SEED = 12345


def workflow_node_names(workflows_dir):
    names = set()
    for path in workflows_dir.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for node in doc.values():
            if isinstance(node, dict) and node.get("class_type"):
                names.add(node["class_type"])
    return names


@pytest.fixture()
def smoke_env(remote_agent, repo_root, monkeypatch):
    workflows_dir = repo_root / "workflows" / "api"
    assets_dir = repo_root / "assets" / "smoke"
    cases = load_smoke_cases()["cases"]
    nodes = workflow_node_names(workflows_dir) | {"AnimaLLLiteApply_sdscripts"}

    fake = FakeComfyUI(remote_agent.comfy_dir, object_info={name: {} for name in sorted(nodes)})
    monkeypatch.setattr(remote_agent.CM, "http_json", fake.http_json)
    monkeypatch.setattr(remote_agent.CM, "gpu_check", gpu_check_stub)

    env = SimpleNamespace(
        agent=remote_agent,
        fake=fake,
        cases=cases,
        workflows_dir=workflows_dir,
        assets_dir=assets_dir,
        nodes=nodes,
    )
    env.set_fake = lambda new_fake: monkeypatch.setattr(remote_agent.CM, "http_json", new_fake.http_json)
    return env


def run_suite(
    env,
    *,
    cases,
    run_id="run-0001",
    base_seed=BASE_SEED,
    run_token=RUN_TOKEN,
    unique_seed=True,
    require_plugin_node=True,
):
    result, exit_code = env.agent.SMOKE.run_suite(
        session_id="sess-1",
        run_id=run_id,
        cases=cases,
        workflows_dir=env.workflows_dir,
        assets_dir=env.assets_dir,
        base_url=BASE_URL,
        comfy_dir=env.agent.comfy_dir,
        instance_id="4242",
        image_ref="vastai/comfy:v0.40.0-cuda-12.9-py312",
        base_seed=base_seed,
        unique_seed=unique_seed,
        run_token=run_token,
        models=None,
        verify_model_shas=False,
        require_plugin_node=require_plugin_node,
    )
    return result, exit_code


def report_of(env, run_id="run-0001"):
    return json.loads((env.agent.C.RUNS_DIR / run_id / "report.json").read_text(encoding="utf-8"))


def summary_of(env, run_id="run-0001"):
    return json.loads((env.agent.C.RUNS_DIR / run_id / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_full_four_case_pass_reports_ready(smoke_env):
    result, exit_code = run_suite(smoke_env, cases=smoke_env.cases)

    assert exit_code == 0
    assert result["ok"] is True
    assert result["verdict"] == "READY"
    assert [c["status"] for c in result["cases"]] == ["passed"] * 4

    report = report_of(smoke_env)
    assert report["verdict"] == "READY"
    assert [c["status"] for c in report["cases"]] == ["passed"] * 4
    assert report["instance_id"] == "4242"
    assert report["image"] == "vastai/comfy:v0.40.0-cuda-12.9-py312"
    assert [c["seed"] for c in report["cases"]] == [
        prompt["6"]["inputs"]["seed"] for prompt in smoke_env.fake.prompts
    ]
    assert all(0 <= c["seed"] < 2**64 for c in report["cases"])
    for case in report["cases"]:
        assert case["images"] and case["images"][0]["width"] == 1024
        assert case["images"][0]["height"] == 1024
        assert case["images"][0]["bytes"] > 0

    # A passing run leaves report.json + report.md + summary.json with the seed.
    run_dir = smoke_env.agent.C.RUNS_DIR / "run-0001"
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "summary.json").is_file()

    summary = summary_of(smoke_env)
    assert summary["verdict"] == "READY"
    assert summary["cases"][0]["seed"] == report["cases"][0]["seed"]
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "READY" in markdown and "base_t2i" in markdown

    # Input assets were really copied into the ComfyUI input directory.
    assert (smoke_env.agent.comfy_dir / "input" / "vast_comfy_smoke_control.png").is_file()
    assert (smoke_env.agent.comfy_dir / "input" / "vast_comfy_smoke_inpaint_src.png").is_file()
    # All four prompts were submitted with unique seeds.
    seeds = [p["6"]["inputs"]["seed"] for p in smoke_env.fake.prompts]
    assert len(set(seeds)) == 4


def test_fixed_seed_mode_refuses_possible_cached_inference(smoke_env):
    result, exit_code = run_suite(smoke_env, cases=smoke_env.cases, unique_seed=False)
    assert exit_code != 0
    assert result["code"] == "E_CONFIG"
    assert result["reason"] == "SMOKE_CACHE"
    assert smoke_env.fake.prompts == []


def test_subset_run_reports_partial(smoke_env):
    subset = [c for c in smoke_env.cases if c["id"] in ("base_t2i", "turbo_t2i")]
    result, exit_code = run_suite(smoke_env, cases=subset)

    assert exit_code == 0
    assert result["ok"] is True
    assert result["verdict"] == "PARTIAL"  # never READY for a subset
    assert [c["status"] for c in result["cases"]] == ["passed", "passed"]
    assert report_of(smoke_env)["verdict"] == "PARTIAL"


def test_partial_run_can_never_be_called_ready(smoke_env):
    subset = [smoke_env.cases[0]]
    result, _ = run_suite(smoke_env, cases=subset)
    assert result["verdict"] == "PARTIAL"


# ---------------------------------------------------------------------------
# Failures stop the flow
# ---------------------------------------------------------------------------


def test_single_failed_case_stops_later_cases(smoke_env):
    smoke_env.fake.fail_prefixes = {"vast_comfy_smoke_lllite_lineart"}
    result, exit_code = run_suite(smoke_env, cases=smoke_env.cases)

    assert exit_code == 5
    assert result["ok"] is False
    assert result["verdict"] == "FAILED"

    report = report_of(smoke_env)
    statuses = [c["status"] for c in report["cases"]]
    assert statuses == ["passed", "failed", "not_run", "not_run"]
    assert report["cases"][1]["reason"] == "SMOKE_EXEC_ERROR"
    assert "previous case failed" in report["cases"][2]["reason"]
    # No later prompt was submitted.
    assert len(smoke_env.fake.prompts) == 2
    assert report["verdict"] == "FAILED"
    # summary.json is still written for the failed run.
    assert summary_of(smoke_env)["verdict"] == "FAILED"


def test_busy_queue_fails_fast_and_writes_summary(smoke_env):
    smoke_env.fake.busy = True
    result, exit_code = run_suite(smoke_env, cases=smoke_env.cases)

    assert exit_code == 13
    assert result["code"] == "E_COMFY_BUSY"
    assert result["ok"] is False
    assert smoke_env.fake.prompts == []  # the queue is never touched
    summary = summary_of(smoke_env)
    assert summary["code"] == "E_COMFY_BUSY"
    assert summary["queue"]["busy"] is True
    # No report is written when the run never started.
    assert not (smoke_env.agent.C.RUNS_DIR / "run-0001" / "report.json").exists()


def test_missing_plugin_node_fails_before_running(smoke_env):
    smoke_env.set_fake(FakeComfyUI(smoke_env.agent.comfy_dir, object_info={}))
    result, exit_code = run_suite(smoke_env, cases=smoke_env.cases)

    assert exit_code == 1
    assert result["code"] == "E_COMFY_UNAVAILABLE"
    assert "AnimaLLLiteApply_sdscripts" in result["missing"]
    assert smoke_env.fake.prompts == []  # check happens before any prompt
    assert summary_of(smoke_env)["code"] == "E_COMFY_UNAVAILABLE"


def test_missing_plugin_node_can_be_explicitly_skipped(smoke_env):
    nodes_without_plugin = {n: {} for n in smoke_env.nodes - {"AnimaLLLiteApply_sdscripts"}}
    smoke_env.set_fake(FakeComfyUI(smoke_env.agent.comfy_dir, object_info=nodes_without_plugin))
    result, _ = run_suite(smoke_env, cases=[smoke_env.cases[0]], require_plugin_node=False)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Image validation: old / flat / wrong size / undecodable are never READY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, reason",
    [
        (old_png, "SMOKE_IMAGE_OLD"),
        (flat_png, "SMOKE_IMAGE_SINGLE_COLOR"),
        (wrong_size_png, "SMOKE_IMAGE_SIZE"),
        (undecodable_png, "SMOKE_IMAGE_DECODE"),
    ],
)
def test_bad_output_image_fails_the_case(smoke_env, builder, reason):
    smoke_env.set_fake(FakeComfyUI(smoke_env.agent.comfy_dir, object_info=dict(smoke_env.fake.object_info), image_builder=builder))
    result, exit_code = run_suite(smoke_env, cases=[smoke_env.cases[0]])

    assert exit_code == 5
    assert result["verdict"] == "FAILED"
    report = report_of(smoke_env)
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["reason"] == reason
    assert report["verdict"] == "FAILED"


def test_image_missing_from_output_is_rejected(smoke_env):
    def vanishing(path):
        noise_png(path)
        path.unlink()

    fake = FakeComfyUI(
        smoke_env.agent.comfy_dir,
        object_info=dict(smoke_env.fake.object_info),
        image_builder=vanishing,
    )
    smoke_env.set_fake(fake)
    result, _ = run_suite(smoke_env, cases=[smoke_env.cases[0]])
    assert result["verdict"] == "FAILED"
    assert report_of(smoke_env)["cases"][0]["reason"] == "SMOKE_IMAGE_MISSING"


def test_zero_output_images_is_rejected(smoke_env):
    class NoImages(FakeComfyUI):
        def http_json(self, url, **kwargs):
            payload = super().http_json(url, **kwargs)
            if "/history/" in url:
                prompt_id = next(iter(payload))
                payload[prompt_id]["outputs"] = {}
            return payload

    smoke_env.set_fake(NoImages(smoke_env.agent.comfy_dir, object_info=dict(smoke_env.fake.object_info)))
    result, _ = run_suite(smoke_env, cases=[smoke_env.cases[0]])
    assert result["verdict"] == "FAILED"
    assert report_of(smoke_env)["cases"][0]["reason"] == "SMOKE_IMAGE_MISSING"


def test_health_requires_a_cuda_device(smoke_env, monkeypatch):
    fake = FakeComfyUI(
        smoke_env.agent.comfy_dir,
        object_info=dict(smoke_env.fake.object_info),
        system_stats={"system": {}, "devices": [{"name": "cpu", "type": "cpu"}]},
    )
    smoke_env.set_fake(fake)
    with pytest.raises(smoke_env.agent.C.RemoteFail) as excinfo:
        smoke_env.agent.SMOKE.run_suite(
            session_id="sess-1",
            run_id="run-0001",
            cases=[smoke_env.cases[0]],
            workflows_dir=smoke_env.workflows_dir,
            assets_dir=smoke_env.assets_dir,
            base_url=BASE_URL,
            comfy_dir=smoke_env.agent.comfy_dir,
            instance_id="4242",
            image_ref="x",
            base_seed=BASE_SEED,
            unique_seed=True,
            run_token=RUN_TOKEN,
            models=None,
            verify_model_shas=False,
        )
    assert excinfo.value.code == "E_COMFY_UNAVAILABLE"
