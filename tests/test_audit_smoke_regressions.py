"""SPEC 13 audit regressions: real graphs/PNG files, mocked HTTP, no GPU/network."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import shutil
from types import SimpleNamespace

import pytest
from PIL import Image

from mocks.fake_comfy import FakeComfyUI, gpu_check_stub, noise_png
from vast_comfy.prepare import load_smoke_cases


def encoded(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


@pytest.fixture()
def audit(remote_agent, anima_profile, repo_root, tmp_path, monkeypatch):
    workflows = tmp_path / "workflows"
    assets = tmp_path / "assets"
    shutil.copytree(repo_root / "workflows" / "anima" / "api", workflows)
    shutil.copytree(repo_root / "assets" / "smoke" / "anima", assets)
    cases = copy.deepcopy(load_smoke_cases()["cases"])
    nodes = remote_agent.SMOKE.required_nodes_from_workflows(
        workflows, [case["workflow"] for case in cases]
    )
    fake = FakeComfyUI(remote_agent.comfy_dir, object_info={node: {} for node in nodes})
    monkeypatch.setattr(remote_agent.CM, "http_json", fake.http_json)
    monkeypatch.setattr(remote_agent.CM, "gpu_check", gpu_check_stub)
    return SimpleNamespace(
        agent=remote_agent, workflows=workflows, assets=assets, cases=cases,
        fake=fake, monkeypatch=monkeypatch,
    )


def run(audit, cases=None, run_id="audit", **kwargs):
    options = dict(
        session_id="audit-session", run_id=run_id,
        cases=audit.cases if cases is None else cases,
        workflows_dir=audit.workflows, assets_dir=audit.assets,
        base_url="http://mock.invalid:18188", comfy_dir=audit.agent.comfy_dir,
        instance_id="4242", image_ref="test-image", base_seed=12345,
        unique_seed=True, run_token=7777,
        # READY is measured against the full declared suite, so a subset run is
        # never reported READY even when everything it ran passed.
        expected_case_ids=tuple(c["id"] for c in audit.cases),
    )
    options.update(kwargs)
    return audit.agent.SMOKE.run_suite(**options)


def report(audit, run_id="audit"):
    return json.loads((audit.agent.C.RUNS_DIR / run_id / "report.json").read_text())


def replace_fake(audit, cls=FakeComfyUI, **kwargs):
    fake = cls(audit.agent.comfy_dir, object_info=audit.fake.object_info, **kwargs)
    audit.monkeypatch.setattr(audit.agent.CM, "http_json", fake.http_json)
    audit.fake = fake
    return fake


def mutate_workflow(audit, case, mutate):
    path = audit.workflows / case["workflow"]
    graph = json.loads(path.read_text())
    mutate(graph)
    path.write_text(json.dumps(graph), encoding="utf-8")


@pytest.mark.parametrize("direct", [False, True])
def test_four_copies_of_base_cannot_impersonate_full_suite(audit, direct):
    cases = [copy.deepcopy(audit.cases[0]) for _ in range(4)]
    if direct:
        result, code = run(audit, cases)
        assert code != 0 and result["code"] == "E_CONFIG"
        assert result.get("verdict") != "READY"
        assert audit.fake.prompts == []
    else:
        with pytest.raises(audit.agent.C.RemoteFail, match="duplicate"):
            audit.agent.SMOKE.load_cases(encoded({"schema_version": 1, "cases": cases}))


def test_full_suite_in_wrong_order_is_rejected_before_submission(audit):
    result, code = run(audit, list(reversed(audit.cases)))
    assert code != 0 and result["code"] == "E_CONFIG"
    assert "order" in result["message"]
    assert audit.fake.prompts == []


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_payload_schema_version_is_checked(audit, version):
    with pytest.raises(audit.agent.C.RemoteFail) as error:
        audit.agent.SMOKE.load_cases(encoded({"schema_version": version, "cases": audit.cases}))
    assert error.value.code == "E_USAGE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""), ("id", []), ("unexpected", 1), ("title", []),
        ("timeout_minutes", 0), ("timeout_minutes", -1), ("timeout_minutes", True),
        ("timeout_minutes", "15"), ("timeout_minutes", float("nan")),
        ("timeout_minutes", float("inf")), ("timeout_minutes", float("-inf")),
        ("timeout_minutes", 1e308), ("expect", {}),
        ("expect", {"width": True, "height": 1024}),
        ("expect", {"width": 0, "height": 1024}),
        ("expect", {"width": 1.5, "height": 1024}),
        ("expect", {"width": 1024, "height": float("nan")}),
        ("overrides", []), ("overrides", {"6": []}),
        ("overrides", {"6": {"steps": 0}}),
        ("overrides", {"6": {"cfg": float("nan")}}),
        ("overrides", {"6": {"seed": -1}}),
        ("inputs", "not-a-list"), ("inputs", ["same.png", "same.png"]),
    ],
)
def test_malformed_case_parameters_fail_before_http(audit, field, value):
    case = copy.deepcopy(audit.cases[0])
    case[field] = value
    with pytest.raises(audit.agent.C.RemoteFail) as error:
        audit.agent.SMOKE.load_cases(encoded([case]))
    assert error.value.code == "E_USAGE"
    result, code = run(audit, [case])
    assert code != 0 and result["code"] == "E_CONFIG"
    assert audit.fake.prompts == []


@pytest.mark.parametrize(
    "path", ["../escape.json", "/escape.json", "C:/escape.json", "C:\\escape.json",
             "sub/../escape.json", "sub//escape.json", "~user/escape.json", "wrong.txt"],
)
def test_workflow_paths_cannot_escape(audit, path):
    case = copy.deepcopy(audit.cases[0])
    case["workflow"] = path
    with pytest.raises(audit.agent.C.RemoteFail):
        audit.agent.SMOKE.load_cases(encoded([case]))
    result, code = run(audit, [case])
    assert code != 0 and result.get("verdict") != "READY"
    assert audit.fake.prompts == []


@pytest.mark.parametrize("case_index", [1, 2, 3])
def test_required_names_with_base_graph_are_not_accepted(audit, case_index):
    case = copy.deepcopy(audit.cases[case_index])
    case["workflow"] = audit.cases[0]["workflow"]
    result, code = run(audit, [case])
    assert code != 0 and result["reason"] == "SMOKE_WORKFLOW"
    assert audit.fake.prompts == []


@pytest.mark.parametrize(
    "case_index,mutate",
    [
        (0, lambda g: g["1"]["inputs"].update(unet_name="anima-turbo-v1.1.safetensors")),
        (3, lambda g: g["1"]["inputs"].update(unet_name="anima-base-v1.0.safetensors")),
        (0, lambda g: g["3"]["inputs"].update(vae_name="other.safetensors")),
        (0, lambda g: g["2"]["inputs"].update(clip_name="other.safetensors")),
        (0, lambda g: g["2"]["inputs"].update(device="cpu")),
        (0, lambda g: g["6"]["inputs"].update(steps=1)),
        (3, lambda g: g["6"]["inputs"].update(cfg=4.0)),
        (0, lambda g: g["7"]["inputs"].update(batch_size=2)),
        (0, lambda g: g["9"]["inputs"].update(images=["7", 0])),
        (1, lambda g: g["11"].update(class_type="ControlNetApply")),
        (1, lambda g: g["6"]["inputs"].update(model=["1", 0])),
        (1, lambda g: g["11"]["inputs"].update(lllite_name="anima-lllite-inpainting-v2.safetensors")),
        (2, lambda g: g["11"]["inputs"].update(lllite_name="anima-lllite-any-test-like-v2.safetensors")),
        (1, lambda g: g["11"]["inputs"].update(strength=0)),
        (1, lambda g: g["11"]["inputs"].update(start_percent=1, end_percent=1)),
        (1, lambda g: g["10"]["inputs"].update(image="user-image.png")),
        (2, lambda g: g["11"]["inputs"].pop("mask")),
        (2, lambda g: g["11"]["inputs"].update(mask=["10", 0])),
        (2, lambda g: g["11"]["inputs"].update(mask=["7", 1])),
        (2, lambda g: g["11"]["inputs"].update(mask=None)),
    ],
)
def test_actual_active_graph_contract_is_enforced(audit, case_index, mutate):
    case = audit.cases[case_index]
    mutate_workflow(audit, case, mutate)
    result, code = run(audit, [case])
    assert code != 0 and result["reason"] == "SMOKE_WORKFLOW"
    assert audit.fake.prompts == []


def test_overrides_cannot_bypass_lllite_after_workflow_validation(audit):
    case = copy.deepcopy(audit.cases[1])
    case["overrides"]["6"]["model"] = ["1", 0]
    result, code = run(audit, [case])
    assert code != 0 and result["reason"] == "SMOKE_WORKFLOW"
    assert audit.fake.prompts == []


@pytest.mark.parametrize("mode,alpha", [("RGB", None), ("RGBA", 255), ("RGBA", 0)])
def test_inpaint_source_requires_a_real_partial_alpha_mask(audit, mode, alpha):
    path = audit.assets / audit.cases[2]["inputs"][0]
    color = (50, 100, 150) if alpha is None else (50, 100, 150, alpha)
    Image.new(mode, (1024, 1024), color).save(path)
    result, code = run(audit, [audit.cases[2]])
    assert code != 0 and result["reason"] == "SMOKE_WORKFLOW"
    assert "mask" in result["message"]
    assert audit.fake.prompts == []


def test_registered_plugin_does_not_make_a_dangling_plugin_valid(audit):
    case = audit.cases[1]
    mutate_workflow(audit, case, lambda g: g["6"]["inputs"].update(model=["1", 0]))
    assert "AnimaLLLiteApply_sdscripts" in audit.fake.object_info
    result, code = run(audit, [case], require_plugin_node=False)
    assert code != 0 and result["reason"] == "SMOKE_WORKFLOW"


def test_fixed_seed_refuses_without_http_or_restart(audit):
    def forbidden(*args, **kwargs):
        pytest.fail("fixed seed must fail before HTTP/GPU/restart")

    audit.monkeypatch.setattr(audit.agent.CM, "http_json", forbidden)
    audit.monkeypatch.setattr(audit.agent.CM, "gpu_check", forbidden)
    audit.monkeypatch.setattr(audit.agent.CM, "restart_service", forbidden)
    result, code = run(audit, unique_seed=False)
    assert code != 0 and result["code"] == "E_CONFIG" and result["reason"] == "SMOKE_CACHE"
    assert "unsupported" in result["message"]


@pytest.mark.parametrize("seed_override", [12345, None])
def test_static_sampler_seed_cannot_evade_unique_seed_mode(audit, seed_override):
    case = copy.deepcopy(audit.cases[0])
    case["overrides"] = {} if seed_override is None else {"6": {"seed": seed_override}}
    result, code = run(audit, [case])
    assert code != 0 and result["reason"] == "SMOKE_CACHE"
    assert audit.fake.prompts == []


def test_secrets_entropy_workflow_digest_and_effective_prompt_are_recorded(audit):
    entropy = iter(0xF000000000000000 + i * 1_000_000_000 for i in range(8))
    requested_bits = []

    def randbits(bits):
        requested_bits.append(bits)
        return next(entropy)

    audit.monkeypatch.setattr(audit.agent.SMOKE.secrets, "randbits", randbits)
    all_seeds = []
    for run_id in ("first", "second"):
        result, code = run(audit, run_id=run_id)
        assert code == 0 and result["verdict"] == "READY"
        doc = report(audit, run_id)
        assert doc["seed_base"] == 12345 and doc["run_token"] == 7777
        for index, case in enumerate(doc["cases"]):
            raw = (audit.workflows / audit.cases[index]["workflow"]).read_bytes()
            assert case["workflow_sha256"] == hashlib.sha256(raw).hexdigest()
            assert case["effective_prompt"] == audit.fake.prompts[len(all_seeds)]
            assert case["seed"] == case["effective_prompt"]["6"]["inputs"]["seed"]
            assert case["effective_settings"]["seed"] == case["seed"]
            assert case["execution_messages"] == []
            assert 10**12 < case["seed"] < 2**64
            all_seeds.append(case["seed"])
    assert requested_bits == [64] * 8 and len(set(all_seeds)) == 8


def test_explicit_parameters_are_hard_requirements_and_recorded(audit):
    case = copy.deepcopy(audit.cases[0])
    case["overrides"]["6"].update(steps=12, cfg=2.5)
    case["overrides"]["7"] = {"width": 512, "height": 512}
    case["expect"] = {"width": 512, "height": 512}
    replace_fake(audit, image_builder=lambda path: noise_png(path, (512, 512)))
    result, code = run(audit, [case])
    assert code == 0 and result["verdict"] == "PARTIAL"
    settings = report(audit)["cases"][0]["effective_settings"]
    assert (settings["steps"], settings["cfg"], settings["width"], settings["height"]) == (12, 2.5, 512, 512)


def test_configured_batch_count_must_actually_be_produced(audit):
    case = copy.deepcopy(audit.cases[0])
    case["overrides"]["7"] = {"batch_size": 2}
    result, code = run(audit, [case])
    assert code != 0 and result["verdict"] == "FAILED"
    assert report(audit)["cases"][0]["reason"] == "SMOKE_IMAGE_COUNT"


@pytest.mark.parametrize("cached_nodes", [["6"], [6], ["KSampler"]])
def test_cached_sampler_fails_even_when_saveimage_writes_a_fresh_png(audit, cached_nodes):
    class Cached(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if "/history/" in url:
                entry = next(iter(value.values()))
                entry["status"]["messages"] = [
                    ["execution_start", {"prompt_id": "prompt-1"}],
                    ["execution_cached", {"nodes": cached_nodes}],
                    ["execution_success", {}],
                ]
            return value

    replace_fake(audit, Cached)
    result, code = run(audit)
    assert code != 0 and result["verdict"] == "FAILED"
    cases = report(audit)["cases"]
    assert cases[0]["reason"] == "SMOKE_CACHE"
    assert cases[0]["execution_messages"][1] == ["execution_cached", {"nodes": cached_nodes}]
    assert [case["status"] for case in cases] == ["failed", "not_run", "not_run", "not_run"]
    assert list((audit.agent.comfy_dir / "output").glob("*.png"))
    assert len(audit.fake.prompts) == 1


def test_cached_non_sampler_nodes_do_not_block_real_sampling(audit):
    class CachedLoaders(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if "/history/" in url:
                next(iter(value.values()))["status"]["messages"] = [
                    ["execution_cached", {"nodes": ["1", "2", "3"]}],
                ]
            return value

    replace_fake(audit, CachedLoaders)
    result, code = run(audit)
    assert code == 0 and result["verdict"] == "READY"


@pytest.mark.parametrize(
    "status_str,completed,event",
    [("error", False, "execution_error"), ("error", False, "progress"),
     ("success", False, "execution_error"), ("success", True, "execution_error"),
     ("success", False, "execution_interrupted")],
)
def test_terminal_errors_fail_immediately_and_preserve_all_messages(audit, status_str, completed, event):
    messages = [["progress", {"step": i}] for i in range(20)] + [[event, {"exception_message": "OOM"}]]

    class ErrorHistory(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if "/history/" in url:
                next(iter(value.values()))["status"] = {
                    "status_str": status_str, "completed": completed, "messages": messages,
                }
            return value

    replace_fake(audit, ErrorHistory)
    audit.monkeypatch.setattr(audit.agent.SMOKE.time, "sleep", lambda _: pytest.fail("must not poll an error"))
    result, code = run(audit)
    assert code != 0 and result["verdict"] == "FAILED"
    case = report(audit)["cases"][0]
    assert case["reason"] == "SMOKE_EXEC_ERROR" and case["execution_messages"] == messages
    assert len(audit.fake.history_calls) == 1


def test_timeout_preserves_last_execution_messages(audit):
    messages = [["execution_start", {}], ["progress", {"value": 1}]]
    audit.monkeypatch.setattr(audit.agent.CM, "http_json", lambda *a, **k: {
        "pending": {"status": {"completed": False, "status_str": "running", "messages": messages}},
    })
    clock = {"now": 0.0}
    audit.monkeypatch.setattr(audit.agent.SMOKE.time, "monotonic", lambda: clock["now"])
    audit.monkeypatch.setattr(audit.agent.SMOKE.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    with pytest.raises(audit.agent.SMOKE.CaseFailure) as error:
        audit.agent.SMOKE.wait_for_history("http://mock.invalid", "pending", 1.0)
    assert error.value.reason == "SMOKE_TIMEOUT" and error.value.details["messages"] == messages


@pytest.mark.parametrize("busy_call,submitted", [(2, 0), (3, 1), (5, 3)])
def test_queue_is_rechecked_immediately_before_every_submit(audit, busy_call, submitted):
    class BecomesBusy(FakeComfyUI):
        queue_calls = 0

        def http_json(self, url, **kwargs):
            if url.endswith("/queue"):
                self.queue_calls += 1
                self.busy = self.queue_calls >= busy_call
            return super().http_json(url, **kwargs)

    replace_fake(audit, BecomesBusy)
    result, code = run(audit)
    assert code == 13 and result["code"] == "E_COMFY_BUSY"
    assert result["verdict"] == "FAILED" and len(audit.fake.prompts) == submitted
    assert report(audit)["cases"][submitted]["reason"] == "E_COMFY_BUSY"


@pytest.mark.parametrize("age", [0.000001, 0.1, 1.0, 4.999])
def test_no_five_second_mtime_grace_is_allowed(audit, age):
    path = audit.agent.comfy_dir / "output" / "stale.png"
    noise_png(path)
    record = {"filename": path.name, "subfolder": "", "path": path}
    with pytest.raises(audit.agent.SMOKE.CaseFailure) as error:
        audit.agent.SMOKE.validate_image(record, audit.cases[0]["expect"], path.stat().st_mtime + age)
    assert error.value.reason == "SMOKE_IMAGE_OLD"


def test_image_at_exact_case_start_is_accepted(audit):
    path = audit.agent.comfy_dir / "output" / "new.png"
    noise_png(path)
    record = {"filename": path.name, "subfolder": "", "path": path}
    assert audit.agent.SMOKE.validate_image(record, audit.cases[0]["expect"], path.stat().st_mtime)["sha256"]


def test_jpeg_disguised_as_png_is_rejected(audit):
    def jpeg(path):
        noise_png(path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        rgb.save(path, format="JPEG")

    replace_fake(audit, image_builder=jpeg)
    result, code = run(audit)
    assert code != 0 and result["verdict"] == "FAILED"
    assert report(audit)["cases"][0]["reason"] == "SMOKE_IMAGE_FORMAT"


def test_png_with_non_png_extension_is_rejected(audit):
    path = audit.agent.comfy_dir / "output" / "not-png.jpg"
    noise_png(path)
    record = {"filename": path.name, "subfolder": "", "path": path}
    with pytest.raises(audit.agent.SMOKE.CaseFailure) as error:
        audit.agent.SMOKE.validate_image(record, audit.cases[0]["expect"], path.stat().st_mtime)
    assert error.value.reason == "SMOKE_IMAGE_FORMAT"


def test_unrelated_output_cannot_substitute_for_active_saveimage(audit):
    class WrongOutput(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if "/history/" in url:
                outputs = next(iter(value.values()))["outputs"]
                outputs["unrelated"] = outputs.pop("9")
            return value

    replace_fake(audit, WrongOutput)
    result, code = run(audit)
    assert code != 0 and report(audit)["cases"][0]["reason"] == "SMOKE_IMAGE_MISSING"


@pytest.mark.parametrize("subfolder,filename", [("..", "outside.png"), ("", "../outside.png"), ("", "C:/outside.png")])
def test_history_output_paths_are_contained(audit, subfolder, filename):
    class EscapingOutput(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if "/history/" in url:
                image = next(iter(value.values()))["outputs"]["9"]["images"][0]
                image.update(subfolder=subfolder, filename=filename)
            return value

    replace_fake(audit, EscapingOutput)
    result, code = run(audit)
    assert code != 0 and report(audit)["cases"][0]["reason"] == "SMOKE_PATH"


def test_existing_different_input_is_never_overwritten(audit):
    name = audit.cases[1]["inputs"][0]
    user_file = audit.agent.comfy_dir / "input" / name
    user_file.parent.mkdir(parents=True)
    original = b"user-owned file, not the bundled smoke asset"
    user_file.write_bytes(original)
    before = user_file.stat().st_mtime_ns
    result, code = run(audit)
    assert code != 0 and result["code"] == "SMOKE_ASSET_CONFLICT"
    assert user_file.read_bytes() == original and user_file.stat().st_mtime_ns == before
    assert audit.fake.prompts == []


def test_identical_existing_input_is_reused_without_writing(audit):
    name = audit.cases[1]["inputs"][0]
    ensure = audit.agent.SMOKE.ensure_input_assets
    first = ensure(audit.assets, audit.agent.comfy_dir, [name])
    path = audit.agent.comfy_dir / "input" / name
    os.utime(path, (1234567890, 1234567890))
    before = path.stat().st_mtime_ns
    second = ensure(audit.assets, audit.agent.comfy_dir, [name])
    assert first[0]["action"] == "copied" and second[0]["action"] == "already_present"
    assert first[0]["sha256"] == second[0]["sha256"] and path.stat().st_mtime_ns == before


def test_input_changed_between_cases_is_rechecked_without_overwrite(audit):
    name = audit.cases[2]["inputs"][0]
    destination = audit.agent.comfy_dir / "input" / name

    class UploadedDuringRun(FakeComfyUI):
        def http_json(self, url, **kwargs):
            value = super().http_json(url, **kwargs)
            if url.endswith("/history/prompt-1"):
                destination.write_bytes(b"new user upload during smoke")
            return value

    replace_fake(audit, UploadedDuringRun)
    result, code = run(audit)
    assert code != 0 and result["verdict"] == "FAILED"
    cases = report(audit)["cases"]
    assert [case["status"] for case in cases] == ["passed", "passed", "failed", "not_run"]
    assert cases[2]["reason"] == "SMOKE_ASSET_CONFLICT"
    assert destination.read_bytes() == b"new user upload during smoke"


@pytest.mark.parametrize("name", [
    "../user.png", "/user.png", "C:/user.png", "..\\user.png",
    "user.png [output]", "user.png [temp]", "[output]/user.png",
])
def test_input_asset_paths_cannot_escape(audit, name):
    with pytest.raises(audit.agent.SMOKE.CaseFailure) as error:
        audit.agent.SMOKE.ensure_input_assets(audit.assets, audit.agent.comfy_dir, [name])
    assert error.value.reason == "SMOKE_PATH"
    case = copy.deepcopy(audit.cases[1])
    case["inputs"] = [name]
    with pytest.raises(audit.agent.C.RemoteFail):
        audit.agent.SMOKE.load_cases(encoded([case]))


def test_input_appearing_during_copy_is_not_overwritten(audit, monkeypatch):
    name = audit.cases[1]["inputs"][0]
    destination = audit.agent.comfy_dir / "input" / name
    original_open = type(destination).open

    def race(path, mode="r", *args, **kwargs):
        if path == destination and mode == "xb":
            with original_open(path, "wb") as user_file:
                user_file.write(b"concurrent user upload")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(type(destination), "open", race)
    with pytest.raises(audit.agent.SMOKE.CaseFailure) as error:
        audit.agent.SMOKE.ensure_input_assets(audit.assets, audit.agent.comfy_dir, [name])
    assert error.value.reason == "SMOKE_ASSET_CONFLICT"
    assert destination.read_bytes() == b"concurrent user upload"


def test_workflow_symlink_cannot_escape_workflows_directory(audit, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes((audit.workflows / audit.cases[0]["workflow"]).read_bytes())
    link = audit.workflows / "escape.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    case = copy.deepcopy(audit.cases[0])
    case["workflow"] = link.name
    result, code = run(audit, [case])
    assert code != 0 and result["reason"] == "SMOKE_PATH" and audit.fake.prompts == []
