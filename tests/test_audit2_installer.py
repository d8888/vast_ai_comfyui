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


@pytest.mark.parametrize("text", [
    "curl https://host/file?token=synthetic-sensitive-value",
    "curl https://host/file?api_key=synthetic-sensitive-value&x=1",
    "git https://user:synthetic-sensitive-value@host/repo",
])
def test_url_credentials_are_redacted_in_defense_in_depth(installer, text):
    assert "synthetic-sensitive-value" not in installer.C.redact(text)
