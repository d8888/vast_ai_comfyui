"""LoRA track: plan building, pins, and the Hugging Face push path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vast_comfy import lora
from vast_comfy.errors import ConfigError, VastComfyError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(**kw) -> lora.LoraSpec:
    base = dict(
        name="noobai_concept",
        base_checkpoint="NoobAI-XL-v1.1.safetensors",
        train_images=5,
        steps=500,
        dim=16,
        alpha=8,
        learning_rate=1e-4,
        resolution=1024,
        tag_threshold=0.35,
    )
    base.update(kw)
    return lora.LoraSpec(**base)


def test_pins_are_exact_versions_not_floating():
    pin = lora.pins()
    # Every value identifies a commit or an exact version. "latest", a range or
    # a bare major version would make the run unreproducible.
    assert len(pin["sd_scripts_commit"]) == 40
    int(pin["sd_scripts_commit"], 16)  # hex
    for key in ("torch", "torchvision", "transformers", "wd14_revision"):
        assert pin[key] and not any(c in pin[key] for c in "<>=*,^~ ")
    assert pin["torch_index_url"].startswith("https://")


def test_committed_track_config_loads_and_matches_committed_assets():
    spec = lora.load_spec(REPO_ROOT / "config" / "lora" / "noobai_concept.yaml")
    assert spec.base_checkpoint == "NoobAI-XL-v1.1.safetensors"
    assert spec.train_images == 5

    train_dir = REPO_ROOT / "assets" / "lora" / "noobai_concept" / "train"
    pngs = sorted(train_dir.glob("*.png"))
    caps = sorted(train_dir.glob("*.txt"))
    # The config claims 5 images: the committed material must actually be 5,
    # and every image must have a caption (WD14 output is committed with it).
    assert len(pngs) == spec.train_images
    assert len(caps) == len(pngs)


def test_build_plan_orders_tag_then_train_then_verify():
    plan = lora.build_plan(
        _spec(), dataset_dir="/workspace/lora_smoke/img", work_dir="/workspace/lora_smoke", seed=4242
    )
    assert plan["kind"] == "lora_train"
    assert plan["seed"] == 4242
    assert plan["dataset"]["images"] == 5
    assert plan["dataset"]["tagger_revision"] == lora.PINNED_WD14_REVISION
    assert plan["train"]["steps"] == 500
    # Verification is mandatory and must be able to fail.
    assert plan["verify"]["min_mean_abs_diff"] > 0
    assert plan["verify"]["min_pixel_change_ratio"] > 0
    assert plan["verify"]["fixed_seed"] == plan["seed"]
    json.dumps(plan)  # must be plain JSON for the remote runner


def test_plan_seed_differs_per_run_but_verify_uses_it():
    a = lora.build_plan(_spec(), dataset_dir="d", work_dir="w", seed=1)
    b = lora.build_plan(_spec(), dataset_dir="d", work_dir="w", seed=2)
    assert a["train"]["seed"] != b["train"]["seed"]
    assert a["verify"]["fixed_seed"] != b["verify"]["fixed_seed"]


@pytest.mark.parametrize(
    "kw",
    [
        {"name": ""},
        {"name": "a/b"},
        {"name": ".hidden"},
        {"train_images": 0},
        {"steps": 0},
        {"dim": 0},
        {"alpha": 0},
        {"tag_threshold": 0},
        {"tag_threshold": 1},
    ],
)
def test_spec_rejects_bad_values(kw):
    with pytest.raises(ConfigError):
        _spec(**kw).validate()


def test_load_spec_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        lora.load_spec(tmp_path / "nope.yaml")


# --- Hugging Face push: exercised entirely through a mock transport ---------


def _fake_transport(status=200, body=None, record=None):
    def transport(url, headers, payload):
        if record is not None:
            record.append({"url": url, "headers": headers, "size": len(payload)})
        return status, body if body is not None else json.dumps({"commitOid": "abc123"})

    return transport


def test_push_dry_run_needs_no_token_and_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    record: list = []
    weights = tmp_path / "x.safetensors"
    weights.write_bytes(b"w" * 16)
    pub = lora.HfPublisher(transport=_fake_transport(record=record))
    result = pub.push(repo_id="user/repo", local_path=weights, path_in_repo="x.safetensors", dry_run=True)
    assert result.dry_run is True
    assert result.commit_id is None
    assert record == []  # nothing left the process


def test_push_uploads_bytes_and_reports_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    record: list = []
    weights = tmp_path / "x.safetensors"
    weights.write_bytes(b"payload!")
    pub = lora.HfPublisher(transport=_fake_transport(record=record))
    result = pub.push(repo_id="user/repo", local_path=weights, path_in_repo="x.safetensors")
    assert result.commit_id == "abc123"
    assert len(record) == 1
    assert record[0]["size"] == len(b"payload!")
    assert record[0]["url"] == "https://huggingface.co/api/models/user/repo/upload/x.safetensors"
    assert record[0]["headers"]["Authorization"] == "Bearer hf_fake_token_for_tests"


def test_push_without_token_fails_closed_and_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    record: list = []
    weights = tmp_path / "x.safetensors"
    weights.write_bytes(b"payload")
    pub = lora.HfPublisher(transport=_fake_transport(record=record))
    with pytest.raises(VastComfyError) as exc:
        pub.push(repo_id="user/repo", local_path=weights, path_in_repo="x.safetensors")
    assert exc.value.code == lora.E_HF_PUSH_FAILED
    assert record == []


def test_push_rejects_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    weights = tmp_path / "x.safetensors"
    weights.write_bytes(b"payload")
    pub = lora.HfPublisher(transport=_fake_transport(status=403, body="denied"))
    with pytest.raises(VastComfyError) as exc:
        pub.push(repo_id="user/repo", local_path=weights, path_in_repo="x.safetensors")
    assert "403" in str(exc.value)


def test_push_rejects_bad_repo_id_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    weights = tmp_path / "x.safetensors"
    weights.write_bytes(b"p")
    pub = lora.HfPublisher(transport=_fake_transport())
    with pytest.raises(VastComfyError):
        pub.push(repo_id="no-slash", local_path=weights, path_in_repo="x")
    with pytest.raises(VastComfyError):
        pub.push(repo_id="user/repo", local_path=tmp_path / "missing", path_in_repo="x")


def test_publisher_never_stores_the_token():
    pub = lora.HfPublisher(transport=_fake_transport())
    assert "hf_fake_token_for_tests" not in repr(vars(pub))


def test_summarize_marks_failed_verification_as_not_passed():
    plan = lora.build_plan(_spec(), dataset_dir="d", work_dir="w", seed=7)
    summary = lora.summarize(
        plan,
        {"saved": True, "weights_bytes": 123, "verify": {"mean_abs_diff": 0.0, "pixel_change_ratio": 0.0}},
    )
    assert summary["passed"] is False
    assert summary["saved"] is True  # a saved file is not a passing run
