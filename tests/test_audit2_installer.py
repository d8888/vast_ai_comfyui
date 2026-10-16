"""Remote manifest defense in depth and atomic no-clobber model publication."""
import copy
import errno
import hashlib
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vast_comfy.prepare import build_install_manifest


@pytest.fixture
def installer(remote_agent):
    module = importlib.import_module("vc_install")
    return module


@pytest.mark.parametrize("mutation", [
    lambda d: d["models"][-1].update(id="../../escape"),
    lambda d: d["models"][-1].update(id="/absolute"),
    lambda d: d["models"][-1].update(url="https://huggingface.co/a/b/resolve/main/file"),
    lambda d: d["models"][-1].update(url=d["models"][-1]["url"] + "?token=synthetic-secret"),
    lambda d: d["models"][-1].update(sha256="broken"),
    lambda d: d["models"][-1].update(size_bytes=1.5),
    lambda d: d["models"][-1].update(auth_env="TOKEN"),
    lambda d: d["plugins"][0].update(commit="main"),
    lambda d: d["plugins"][0].update(repo_url="https://user:secret@github.com/org/repo"),
    lambda d: d["plugins"][0].update(dest="custom_nodes/.."),
    lambda d: d["models"].append(copy.deepcopy(d["models"][0])),
])
def test_whole_remote_manifest_validated_before_first_mutation(installer, monkeypatch, mutation):
    manifest = copy.deepcopy(build_install_manifest()); mutation(manifest)
    download = Mock(); plugin = Mock()
    monkeypatch.setattr(installer, "_download", download)
    monkeypatch.setattr(installer, "_install_plugin", plugin)
    with pytest.raises(installer.C.RemoteFail):
        installer._install_manifest(manifest)
    download.assert_not_called(); plugin.assert_not_called()


def test_model_publication_never_overwrites_racing_destination(installer, tmp_path):
    partial = tmp_path / "model.part"; partial.write_bytes(b"verified model")
    dest = tmp_path / "model.safetensors"; dest.write_bytes(b"user file")
    with pytest.raises(installer.C.RemoteFail):
        installer._promote_without_overwrite(partial, dest, hashlib.sha256(partial.read_bytes()).hexdigest())
    assert dest.read_bytes() == b"user file"
    assert partial.read_bytes() == b"verified model"


def test_cross_filesystem_publication_uses_verified_local_stage(installer, tmp_path, monkeypatch):
    partial = tmp_path / "model.part"; partial.write_bytes(b"verified model")
    dest = tmp_path / "model.safetensors"
    original = installer.os.link
    calls = []
    def link(source, target, **kwargs):
        calls.append((source, target))
        if len(calls) == 1: raise OSError(errno.EXDEV, "different filesystem")
        return original(source, target, **kwargs)
    monkeypatch.setattr(installer.os, "link", link)
    installer._promote_without_overwrite(partial, dest, hashlib.sha256(partial.read_bytes()).hexdigest())
    assert dest.read_bytes() == b"verified model"
    assert not partial.exists()
    assert len(list(tmp_path.iterdir())) == 1


def test_staging_collision_does_not_delete_unknown_file(installer, tmp_path, monkeypatch):
    partial = tmp_path / "model.part"; partial.write_bytes(b"verified model")
    dest = tmp_path / "model.safetensors"
    collision = dest.with_name("." + dest.name + ".vast-comfy-collision.part")
    collision.write_bytes(b"not ours")
    monkeypatch.setattr(installer.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    monkeypatch.setattr(installer.os, "link", Mock(side_effect=OSError(errno.EXDEV, "different filesystem")))
    with pytest.raises(installer.C.RemoteFail):
        installer._promote_without_overwrite(partial, dest, "a" * 64)
    assert collision.read_bytes() == b"not ours"
    assert partial.exists() and not dest.exists()


def test_model_destination_symlink_escape_refused(installer, tmp_path, monkeypatch):
    comfy = tmp_path / "ComfyUI"; (comfy / "models").mkdir(parents=True)
    outside = tmp_path / "outside"; outside.mkdir()
    link = comfy / "models" / "diffusion_models"
    try: link.symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip("symlink privilege unavailable")
    monkeypatch.setattr(installer.C, "COMFY_DIR", comfy)
    with pytest.raises(installer.C.RemoteFail):
        installer._validate_model_entry(build_install_manifest()["models"][0])
    assert list(outside.iterdir()) == []


def test_remote_verify_package_refuses_an_empty_manifest(installer, tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("")
    output = Mock()
    monkeypatch.setattr(installer.C, "emit", output)
    installer.C.cmd_verify_package(SimpleNamespace(directory=str(tmp_path), manifest=str(manifest)))
    payload = output.call_args.args[0]
    assert output.call_args.kwargs["exit_code"] == 1
    assert payload["ok"] is False
    assert any("no entries" in problem for problem in payload["problems"])
    assert payload["checked"] == 0


def test_remote_verify_package_refuses_duplicate_entries(installer, tmp_path, monkeypatch):
    target = tmp_path / "good.txt"
    target.write_bytes(b"data")
    sha = hashlib.sha256(b"data").hexdigest()
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(f"{sha}  good.txt\n{sha}  good.txt\n")
    output = Mock()
    monkeypatch.setattr(installer.C, "emit", output)
    installer.C.cmd_verify_package(SimpleNamespace(directory=str(tmp_path), manifest=str(manifest)))
    payload = output.call_args.args[0]
    assert output.call_args.kwargs["exit_code"] == 1
    assert payload["ok"] is False
    assert any("duplicate" in problem for problem in payload["problems"])
    assert payload["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_install_manifest_stops_after_the_stage_deadline(installer, monkeypatch):
    manifest = build_install_manifest()
    download = Mock()
    monkeypatch.setattr(installer, "_download", download)
    plugin = Mock()
    monkeypatch.setattr(installer, "_install_plugin", plugin)
    monkeypatch.setattr(installer.C, "now_epoch", lambda: 100.0)

    with pytest.raises(installer.C.RemoteFail) as error:
        installer._install_manifest(manifest, deadline=99.0)
    assert error.value.code == "E_JOB_TIMEOUT"
    download.assert_not_called()
    plugin.assert_not_called()


def test_install_manifest_runs_while_the_stage_deadline_holds(installer, monkeypatch):
    manifest = build_install_manifest()
    monkeypatch.setattr(installer, "_download",
                        lambda entry: {"id": entry["id"], "action": "skipped"})
    monkeypatch.setattr(installer, "_install_plugin",
                        lambda entry: {"id": entry["id"], "action": "skipped"})
    monkeypatch.setattr(installer.C, "now_epoch", lambda: 100.0)

    result = installer._install_manifest(manifest, deadline=101.0)
    assert result["ok"] is True and result["changed"] is False


def test_install_stage_fails_when_the_last_item_overruns(installer, monkeypatch):
    manifest = build_install_manifest()
    clock = {"now": 0.0}
    monkeypatch.setattr(installer.C, "now_epoch", lambda: clock["now"])
    monkeypatch.setattr(installer, "_download",
                        lambda entry: {"id": entry["id"], "action": "skipped"})

    def slow_plugin(entry):
        clock["now"] = 120.0  # the item itself overran the 60s budget
        return {"id": entry["id"], "action": "installed"}

    monkeypatch.setattr(installer, "_install_plugin", slow_plugin)
    with pytest.raises(installer.C.RemoteFail) as error:
        installer._install_manifest(manifest, deadline=60.0)
    assert error.value.code == "E_JOB_TIMEOUT"


def test_prepare_accepts_the_install_stage_timeout_flag(remote_agent):
    module = importlib.import_module("vc_prepare")
    args = module.build_parser().parse_args([
        "run", "--session-id", "S", "--install-manifest-b64", "W10=", "--cases-b64", "W10=",
        "--workflows-dir", "w", "--assets-dir", "a", "--run-id", "r",
        "--install-timeout-seconds", "120"])
    assert args.install_timeout_seconds == 120.0


@pytest.mark.parametrize("text", [
    "curl https://host/file?token=synthetic-sensitive-value",
    "curl https://host/file?api_key=synthetic-sensitive-value&x=1",
    "git https://user:synthetic-sensitive-value@host/repo",
])
def test_url_credentials_are_redacted_in_defense_in_depth(installer, text):
    assert "synthetic-sensitive-value" not in installer.C.redact(text)
