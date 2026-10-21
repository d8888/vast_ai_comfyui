"""SPEC 17 item 14: the deployment whitelist and deterministic packaging."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from vast_comfy import remote
from vast_comfy.errors import VastComfyError

FORBIDDEN_REL = [
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.pub)?$"),
    re.compile(r"(^|/)known_hosts$"),
    re.compile(r"\.(pem|ppk|key|p12|pfx)$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# package_files(): the whitelist itself
# ---------------------------------------------------------------------------


def test_package_files_are_the_expected_whitelist():
    files = remote.package_files()
    rels = [rel for _, rel in files]
    assert len(files) == len(set(rels)), "duplicate remote paths in the package"
    assert "remote/vc_common.py" in rels
    assert "remote/vc_smoke.py" in rels
    assert sum(1 for rel in rels if rel.startswith("workflows/anima/api/")) == 4
    assert sum(1 for rel in rels if rel.startswith("workflows/anima/gui/")) == 4
    assert "assets/smoke/anima/vast_comfy_smoke_control.png" in rels
    assert "assets/smoke/anima/smoke_cases.yaml" in rels
    for path, rel in files:
        assert path.is_file(), rel
        assert "\\" not in rel


def test_package_files_never_include_secret_like_files():
    for _, rel in remote.package_files():
        for pattern in FORBIDDEN_REL:
            assert not pattern.search(rel), f"forbidden file would be uploaded: {rel}"
        assert ".env" not in rel
        assert "id_rsa" not in rel
        assert "known_hosts" not in rel
        assert not rel.endswith(".pem")


def test_package_files_does_not_walk_user_directories():
    # The whitelist is derived from explicit roots only; a stray file in an
    # unlisted directory (e.g. a model cache) can never enter the package.
    allowed_prefixes = ("remote/", "workflows/", "assets/smoke/")
    for _, rel in remote.package_files():
        assert rel.startswith(allowed_prefixes), rel


# ---------------------------------------------------------------------------
# assert_upload_safe()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [".env", "id_rsa", "id_ed25519", "known_hosts", "foo.pem", "deploy.key", "bundle.p12"])
def test_assert_upload_safe_rejects_secret_like_names(tmp_path, name):
    path = tmp_path / name
    path.write_text("x", encoding="utf-8")
    with pytest.raises(VastComfyError) as excinfo:
        remote.assert_upload_safe([(path, name)])
    assert excinfo.value.code == "E_DEPLOY"
    assert "secret-like" in excinfo.value.message


def test_assert_upload_safe_rejects_nested_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("VAST_API_KEY=abc", encoding="utf-8")
    with pytest.raises(VastComfyError) as excinfo:
        remote.assert_upload_safe([(path, "remote/config/.env.local")])
    assert excinfo.value.code == "E_DEPLOY"


def test_assert_upload_safe_rejects_oversized_file(tmp_path):
    big = tmp_path / "big.bin"
    with big.open("wb") as fh:
        fh.seek(remote.MAX_FILE_BYTES)
        fh.write(b"x")
    assert big.stat().st_size == remote.MAX_FILE_BYTES + 1
    with pytest.raises(VastComfyError) as excinfo:
        remote.assert_upload_safe([(big, "remote/big.bin")])
    assert excinfo.value.code == "E_DEPLOY"
    assert "too large" in excinfo.value.message


def test_assert_upload_safe_rejects_missing_file(tmp_path):
    with pytest.raises(VastComfyError) as excinfo:
        remote.assert_upload_safe([(tmp_path / "gone.py", "remote/gone.py")])
    assert excinfo.value.code == "E_DEPLOY"
    assert "missing" in excinfo.value.message


def test_assert_upload_safe_rejects_too_many_files(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x", encoding="utf-8")
    files = [(path, f"remote/{i}.py") for i in range(remote.MAX_FILES + 1)]
    with pytest.raises(VastComfyError) as excinfo:
        remote.assert_upload_safe(files)
    assert excinfo.value.code == "E_DEPLOY"


def test_build_package_refuses_an_oversized_whitelist_entry(tmp_path, monkeypatch):
    big = tmp_path / "big.bin"
    with big.open("wb") as fh:
        fh.seek(remote.MAX_FILE_BYTES)
        fh.write(b"x")
    monkeypatch.setattr(remote, "package_files", lambda: [(big, "remote/big.bin")])
    with pytest.raises(VastComfyError) as excinfo:
        remote.build_package()
    assert excinfo.value.code == "E_DEPLOY"


# ---------------------------------------------------------------------------
# build_package(): deterministic content-addressing + manifest format
# ---------------------------------------------------------------------------


def test_build_package_is_deterministic_and_manifest_is_sha256_lines():
    first = remote.build_package()
    second = remote.build_package()

    assert first.package_hash == second.package_hash
    assert first.manifest_text == second.manifest_text
    assert first.package_hash == hashlib.sha256(first.manifest_text.encode("utf-8")).hexdigest()[:16]

    lines = first.manifest_text.splitlines()
    assert len(lines) == len(first.records)
    for line in lines:
        sha, sep, rel = line.partition("  ")
        assert sep == "  ", f"manifest lines must be '<sha256>  <rel>': {line!r}"
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
        assert rel and not rel.startswith("/")
    # Records agree with the manifest.
    for record in first.records:
        assert record["size"] == Path(record["local"]).stat().st_size
        assert f"{record['sha256']}  {record['rel']}" in first.manifest_text
    assert first.remote_dir.endswith(f"/pkg/{first.package_hash}")


def test_build_package_rejects_env_file_via_monkeypatched_whitelist(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("VAST_API_KEY=super-secret", encoding="utf-8")
    monkeypatch.setattr(remote, "package_files", lambda: [(env, ".env")])
    with pytest.raises(VastComfyError) as excinfo:
        remote.build_package()
    assert excinfo.value.code == "E_DEPLOY"
