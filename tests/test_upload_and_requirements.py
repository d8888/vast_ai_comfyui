"""Image upload into ComfyUI input, and plugin requirement installation.

Both paths exist because a plugin that cannot import its dependencies and a
workflow that cannot read a reference photo are equally useless, yet neither
may compromise the instance's working torch build or overwrite user files.
"""
import hashlib
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def installer(remote_agent):
    return importlib.import_module("vc_install")


def _png(path, payload=b"synthetic png bytes"):
    path.write_bytes(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(remote_agent, **kwargs):
    defaults = dict(session_id="S", name="probe.png", staged_path="", sha256="")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _emit_payload(capsys, func, args):
    with pytest.raises(SystemExit) as exit_info:
        func(args)
    assert exit_info.value.code == 0
    return json.loads(capsys.readouterr().out)


def test_upload_publishes_verified_png_into_comfy_input(remote_agent, monkeypatch, tmp_path, capsys):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    staged = tmp_path / "staged.png"
    digest = _png(staged)

    payload = _emit_payload(
        capsys, module.cmd_upload_image, _args(remote_agent, staged_path=str(staged), sha256=digest)
    )

    assert payload["action"] == "installed" and payload["sha256"] == digest
    dest = remote_agent.comfy_dir / "input" / "probe.png"
    assert dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest


def test_upload_never_overwrites_a_different_existing_file(remote_agent, monkeypatch, tmp_path):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    input_dir = remote_agent.comfy_dir / "input"
    input_dir.mkdir(parents=True)
    existing = input_dir / "probe.png"
    existing.write_bytes(b"the user's own file")
    staged = tmp_path / "staged.png"
    digest = _png(staged)

    with pytest.raises(module.C.RemoteFail) as error:
        module.cmd_upload_image(_args(remote_agent, staged_path=str(staged), sha256=digest))

    assert error.value.code == "E_UPLOAD_CONFLICT"
    assert existing.read_bytes() == b"the user's own file"


def test_upload_is_idempotent_for_identical_content(remote_agent, monkeypatch, tmp_path, capsys):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    staged = tmp_path / "staged.png"
    digest = _png(staged)
    args = _args(remote_agent, staged_path=str(staged), sha256=digest)

    _emit_payload(capsys, module.cmd_upload_image, args)
    second = _emit_payload(capsys, module.cmd_upload_image, args)

    assert second["action"] == "already_present"


def test_upload_rejects_corrupted_transfer(remote_agent, monkeypatch, tmp_path):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    staged = tmp_path / "staged.png"
    _png(staged)

    with pytest.raises(module.C.RemoteFail) as error:
        module.cmd_upload_image(
            _args(remote_agent, staged_path=str(staged), sha256="0" * 64)
        )

    assert error.value.code == "E_USAGE"
    assert not (remote_agent.comfy_dir / "input" / "probe.png").exists()


@pytest.mark.parametrize("name", ["../escape.png", "sub/dir.png", "", ".hidden.png"])
def test_upload_rejects_unsafe_names(remote_agent, monkeypatch, tmp_path, name):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    staged = tmp_path / "staged.png"
    digest = _png(staged)

    with pytest.raises(module.C.RemoteFail) as error:
        module.cmd_upload_image(
            _args(remote_agent, name=name, staged_path=str(staged), sha256=digest)
        )

    assert error.value.code == "E_USAGE"


def test_upload_rejects_non_png(remote_agent, monkeypatch, tmp_path):
    module = remote_agent.CM
    monkeypatch.setattr(module.C, "ensure_owns", lambda _: {})
    staged = tmp_path / "staged.jpg"
    digest = _png(staged)

    with pytest.raises(module.C.RemoteFail) as error:
        module.cmd_upload_image(
            _args(remote_agent, name="probe.jpg", staged_path=str(staged), sha256=digest)
        )

    assert error.value.code == "E_USAGE"


def test_requirements_install_drops_torch_and_keeps_the_rest(installer, monkeypatch, tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "requirements.txt").write_text(
        "# comment\npackaging>=20\ntorch==0.0.1\ntorchvision\ntorchmetrics\nnumpy\n", encoding="utf-8"
    )
    seen = {}

    def fake_checked(argv):
        seen["argv"] = argv
        seen["content"] = open(argv[argv.index("-r") + 1]).read()
        if "-c" in argv:
            # The constraints file is removed after the call; read it now.
            seen["constraints"] = open(argv[argv.index("-c") + 1]).read()
        return 0

    monkeypatch.setattr(installer, "_run_checked", fake_checked)
    monkeypatch.setattr(
        installer.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="Version: 2.10.0+cu130\n"),
    )

    result = installer._install_requirements("probe", plugin_dir, "requirements.txt")

    assert result["skipped"] == ["torch==0.0.1", "torchvision"]
    # torchmetrics is a different package and must survive the filter.
    assert "torchmetrics" in result["installed"]
    assert "torch==0.0.1" not in seen["content"]
    assert "torch==2.10.0+cu130" in seen["constraints"]


def test_requirements_failure_is_not_reported_as_success(installer, monkeypatch, tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    monkeypatch.setattr(installer, "_run_checked", lambda argv: 2)
    monkeypatch.setattr(
        installer.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )

    with pytest.raises(installer.C.RemoteFail) as error:
        installer._install_requirements("probe", plugin_dir, "requirements.txt")

    assert error.value.code == "E_INSTALL"


def test_missing_requirements_file_is_refused(installer, tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()

    with pytest.raises(installer.C.RemoteFail) as error:
        installer._install_requirements("probe", plugin_dir, "requirements.txt")

    assert error.value.code == "E_INSTALL"


def test_requirements_path_cannot_escape_the_plugin(installer, tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("numpy\n", encoding="utf-8")

    # confined_path rejects the traversal before any file is read or pip runs.
    with pytest.raises(ValueError):
        installer._install_requirements("probe", plugin_dir, "../outside.txt")

