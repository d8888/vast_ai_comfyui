"""Offline regressions for untrusted download metadata and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import pytest

from mocks.fake_ssh import FakeRunner
from vast_comfy import download as dl
from vast_comfy.errors import ConfigError, UsageError, VastComfyError
from vast_comfy.safety import validate_identifier, validate_path_component, validate_posix_path

ROOT = "/workspace/ComfyUI/output"
RUNTIME = "/workspace/.vast_comfy"
RUN = "audit-2"
REPORT = f"{RUNTIME}/runs/{RUN}/report.json"
CONTENT = b"verified download bytes"
SHA = hashlib.sha256(CONTENT).hexdigest()


def _allow_remote(runner, root, path, *, canonical_root=None, canonical_path=None):
    runner.add_rule(
        f"realpath -e -- {root} {path}",
        stdout=f"{canonical_root or root}\n{canonical_path or path}\n",
    )


def _outputs(paths, *, canonical=None):
    runner = FakeRunner()
    runner.add_rule("find ", stdout="".join(f"{100 - index}.5 {path}\n" for index, path in enumerate(paths)))
    runner.add_rule("sha256sum ", stdout=f"{SHA}  file\n")
    for path in paths:
        resolved = (canonical or {}).get(path, path)
        _allow_remote(runner, ROOT, path, canonical_path=resolved)
        runner.remote_files[resolved] = CONTENT
    return runner


def _smoke(report):
    runner = FakeRunner()
    _allow_remote(runner, RUNTIME, REPORT)
    runner.add_rule("cat ", stdout=json.dumps(report))
    return runner


def _symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"Creating symlinks requires OS permission: {exc}")


INVALID_IDS = [
    "", ".", "..", "../escape", "a/b", "a\\b", "C:foo", "a:b", "/abs", "\\abs",
    "-a", "_a", ".a", " a", "a ", "a.", "a\n", "a\x00", "é", "a" * 129,
    None, 1, True, "CON", "con.txt", "Aux", "NUL.log", "COM1", "com9.ext", "LPT1", "lPt9.txt",
]


@pytest.mark.parametrize("value", INVALID_IDS)
def test_identifier_rejects_nonportable_single_components(value):
    with pytest.raises(UsageError) as error:
        validate_identifier(value, label="run_id")
    assert error.value.code == "E_USAGE"
    assert "run_id" in error.value.message


@pytest.mark.parametrize("value", [
    "a", "A0", "run-1", "20261016T0000Z-abc", "a.b_c-d", "a..b", "conifer",
    "COM0", "COM10", "LPT0", "LPT10", "a" * 128,
])
def test_identifier_preserves_safe_values(value):
    assert validate_identifier(value) == value


@pytest.mark.parametrize("method", [dl.download_report, dl.download_smoke_images])
@pytest.mark.parametrize("run_id", ["../escape", "C:escape", "a\\b", "CON", "a" * 129])
def test_download_rejects_run_id_before_any_remote_or_local_effect(method, run_id, cfg, tmp_path):
    runner = FakeRunner()
    dest = tmp_path / "not-created"
    with pytest.raises(UsageError):
        method(runner, cfg, run_id=run_id, dest_dir=dest)
    assert runner.calls == []
    assert runner.downloads == []
    assert not dest.exists()


@pytest.mark.parametrize("value", [
    "", "relative/file", "C:/file", "//server/file", "/a//b", "/a/./b", "/a/../b",
    "/a/", "/a\\b", "/a/C:/b", "/a\x00b", "/a\nb", "/a\rb", None, 12,
])
def test_posix_validator_checks_raw_components_before_normalization(value):
    with pytest.raises(ConfigError) as error:
        validate_posix_path(value)
    assert error.value.code == "E_CONFIG"


@pytest.mark.parametrize("value", ["/", "/workspace/ComfyUI/output", "/with spaces/image.png"])
def test_posix_validator_returns_pure_posix_path(value):
    assert validate_posix_path(value) == PurePosixPath(value)


@pytest.mark.parametrize("value", ["CON.png", "lpt1.txt", "COM¹.png", "trailing.", "trailing ", "a:b", "a?b"])
def test_portable_filename_validation_rejects_windows_aliases(value):
    with pytest.raises(ConfigError):
        validate_path_component(value)


BAD_PATHS = [
    f"{ROOT}/../secret.png", f"{ROOT}/sub/../../secret.png", f"{ROOT}/./image.png",
    f"{ROOT}//image.png", f"{ROOT}/image.png/", f"{ROOT}/sub\\escape.png",
    f"{ROOT}/C:/escape.png", f"{ROOT}/C:escape.png", f"{ROOT}/NUL.png",
    f"{ROOT}/trailing.", f"{ROOT}/trailing ", f"{ROOT}/image\x00.png", f"{ROOT}/image\r.png",
    f"{ROOT}/image\n.png", f"{ROOT}-other/image.png", ROOT, "/etc/passwd",
    "relative.png", "C:\\escape.png", "//server/share/image.png", "",
]


@pytest.mark.parametrize("remote", BAD_PATHS)
@pytest.mark.parametrize("source", ["outputs", "smoke"])
def test_all_metadata_paths_are_prevalidated_not_silently_skipped(remote, source, cfg, tmp_path):
    good = f"{ROOT}/good.png"
    if source == "outputs":
        runner = _outputs([good, remote])
        invoke = lambda: dl.download_outputs(runner, cfg, dest_dir=tmp_path / "dest")
    else:
        runner = _smoke({"cases": [{"images": [{"path": good}, {"path": remote}]}]})
        invoke = lambda: dl.download_smoke_images(runner, cfg, run_id=RUN, dest_dir=tmp_path / "dest")
    with pytest.raises(VastComfyError) as error:
        invoke()
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("report", [
    [], {}, {"cases": None}, {"cases": {}}, {"cases": "bad"}, {"cases": [None]}, {"cases": [[]]},
    {"cases": [{"images": None}]}, {"cases": [{"images": {}}]}, {"cases": [{"images": "bad"}]},
    {"cases": [{"images": [None]}]}, {"cases": [{"images": [{}]}]},
    {"cases": [{"images": [{"path": None}]}]}, {"cases": [{"images": [{"path": 42}]}]},
])
def test_invalid_smoke_metadata_has_stable_download_error(report, cfg, tmp_path):
    runner = _smoke(report)
    with pytest.raises(VastComfyError) as error:
        dl.download_smoke_images(runner, cfg, run_id=RUN, dest_dir=tmp_path / "dest")
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


def test_invalid_smoke_json_has_download_error(cfg, tmp_path):
    runner = FakeRunner()
    _allow_remote(runner, RUNTIME, REPORT)
    runner.add_rule("cat ", stdout="{invalid JSON")
    with pytest.raises(VastComfyError) as error:
        dl.download_smoke_images(runner, cfg, run_id=RUN, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


@pytest.mark.parametrize("listing", [
    "bad metadata", "\n", f"nan {ROOT}/file.png\n", f"1 {ROOT}/file.png\r\n",
    f"1 {ROOT}/file.png\n\n", f"1 {ROOT}/file.png\nsecond-half\n",
])
def test_malformed_find_metadata_is_not_ignored(listing, cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("find ", stdout=listing)
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "2", None])
def test_limit_must_be_a_positive_integer_before_remote_calls(limit, cfg, tmp_path):
    runner = FakeRunner()
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, limit=limit)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.calls == []


def test_untrusted_listing_cannot_exceed_limit(cfg, tmp_path):
    runner = _outputs([f"{ROOT}/a.png", f"{ROOT}/b.png"])
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, limit=1)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


@pytest.mark.parametrize("names", [("a.png", "a.png"), ("A.png", "a.png")])
def test_duplicate_or_case_alias_destinations_reject_entire_batch(names, cfg, tmp_path):
    runner = _outputs([f"{ROOT}/{name}" for name in names])
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=True)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


def test_existing_later_destination_prevents_earlier_download(cfg, tmp_path):
    runner = _outputs([f"{ROOT}/a.png", f"{ROOT}/b.png"])
    (tmp_path / "b.png").write_bytes(b"old")
    with pytest.raises(VastComfyError):
        dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert runner.downloads == []
    assert not (tmp_path / "a.png").exists()
    assert (tmp_path / "b.png").read_bytes() == b"old"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("kind", ["parent", "file", "dangling"])
def test_local_symlink_escape_is_rejected_without_touching_external_files(kind, overwrite, cfg, tmp_path):
    dest, outside = tmp_path / "dest", tmp_path / "outside"
    dest.mkdir()
    outside.mkdir()
    victim = outside / "file.png"
    victim.write_bytes(b"outside must survive")
    if kind == "parent":
        _symlink(dest / "nested", outside, directory=True)
        remote = f"{ROOT}/nested/file.png"
    else:
        _symlink(dest / "file.png", victim if kind == "file" else outside / "missing.png")
        remote = f"{ROOT}/file.png"
    runner = _outputs([remote])
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=dest, overwrite=overwrite)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert victim.read_bytes() == b"outside must survive"
    assert sorted(path.name for path in outside.iterdir()) == ["file.png"]


def test_final_symlink_is_not_followed_even_when_it_points_inside_destination(cfg, tmp_path):
    (tmp_path / "old.png").write_bytes(b"old")
    _symlink(tmp_path / "file.png", tmp_path / "old.png")
    runner = _outputs([f"{ROOT}/file.png"])
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=True)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert (tmp_path / "old.png").read_bytes() == b"old"


def test_smoke_run_directory_cannot_symlink_out_of_explicit_destination(cfg, tmp_path):
    dest, outside = tmp_path / "dest", tmp_path / "outside"
    dest.mkdir()
    outside.mkdir()
    _symlink(dest / RUN, outside, directory=True)
    runner = _smoke({"cases": [{"images": [{"path": f"{ROOT}/file.png"}]}]})
    with pytest.raises(VastComfyError) as error:
        dl.download_smoke_images(runner, cfg, run_id=RUN, dest_dir=dest)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("resolved", ["/etc/passwd", f"{ROOT}-other/file.png", ROOT])
def test_later_remote_symlink_escape_rejects_whole_batch(resolved, cfg, tmp_path):
    first, second = f"{ROOT}/first.png", f"{ROOT}/second.png"
    runner = _outputs([first, second], canonical={second: resolved})
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert list(tmp_path.iterdir()) == []


def test_remote_symlink_inside_root_transfers_canonical_path_not_original_alias(cfg, tmp_path):
    alias, real = f"{ROOT}/alias.png", f"{ROOT}/nested/real.png"
    runner = _outputs([alias], canonical={alias: real})
    result = dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert runner.downloads[0][0] == real
    assert result["downloaded"][0]["remote"] == real
    assert (tmp_path / "alias.png").read_bytes() == CONTENT
    assert any(call.argv == ["realpath", "-e", "--", ROOT, alias] for call in runner.calls)


def test_configured_remote_output_root_may_resolve_to_another_storage_mount(cfg, tmp_path):
    runner = FakeRunner()
    alias, canonical_root = f"{ROOT}/file.png", "/mnt/output"
    real = f"{canonical_root}/file.png"
    runner.add_rule("find ", stdout=f"1 {alias}\n")
    _allow_remote(runner, ROOT, alias, canonical_root=canonical_root, canonical_path=real)
    runner.add_rule("sha256sum ", stdout=f"{SHA}  file\n")
    runner.remote_files[real] = CONTENT
    dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert runner.downloads[0][0] == real
    assert (tmp_path / "file.png").read_bytes() == CONTENT


@pytest.mark.parametrize("method", [dl.download_report, dl.download_smoke_images])
@pytest.mark.parametrize("resolved", ["/etc/passwd", f"{RUNTIME}/secret", f"{RUNTIME}/runs/other/report.json"])
def test_report_remote_symlink_escape_is_checked_before_transfer_or_json_read(method, resolved, cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("test -f", rc=0)
    _allow_remote(runner, RUNTIME, REPORT, canonical_path=resolved)
    with pytest.raises(VastComfyError) as error:
        method(runner, cfg, run_id=RUN, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert not any(call.argv[0] == "cat" for call in runner.calls)


@pytest.mark.parametrize("stdout,rc", [
    ("", 127), ("", 0), (f"{ROOT}\n", 0), (f"{ROOT}\n{ROOT}/file.png\nextra\n", 0),
    (f"{ROOT}\n{ROOT}/../file.png\n", 0), (f"{ROOT}\n{ROOT}/file.png\r\n", 0),
])
def test_unverifiable_remote_realpath_fails_closed(stdout, rc, cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("find ", stdout=f"1 {ROOT}/file.png\n")
    runner.add_rule("realpath -e --", stdout=stdout, rc=rc)
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []


def test_local_resolved_containment_check_is_platform_independent(cfg, tmp_path, monkeypatch):
    """Exercise the escape guard even where the OS forbids creating symlinks."""
    dest = tmp_path / "dest"
    victim = dest / "file.png"
    original_resolve = Path.resolve

    def redirected(path, *args, **kwargs):
        if path == victim:
            return tmp_path / "outside.png"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    runner = _outputs([f"{ROOT}/file.png"])
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=dest, overwrite=True)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert not dest.exists()


@pytest.mark.parametrize("stdout,rc", [("", 0), ("not-a-hash  file\n", 0), ("a" * 63, 0), (SHA, 1)])
def test_invalid_or_unavailable_remote_hash_does_not_touch_destination(stdout, rc, cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("find ", stdout=f"1 {ROOT}/file.png\n")
    _allow_remote(runner, ROOT, f"{ROOT}/file.png")
    runner.add_rule("sha256sum ", stdout=stdout, rc=rc)
    final = tmp_path / "file.png"
    final.write_bytes(b"old")
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=True)
    assert error.value.code == "E_DOWNLOAD"
    assert runner.downloads == []
    assert final.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [final]


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_hash_preserves_existing_files_and_cleans_only_own_temporary(overwrite, cfg, tmp_path):
    remote = f"{ROOT}/file.png"
    runner = _outputs([remote])
    runner.download_bytes[remote] = b"corrupt"
    final = tmp_path / "file.png"
    if overwrite:
        final.write_bytes(b"old")
    sentinel = tmp_path / ".vast-comfy-user-owned.tmp"
    sentinel.write_bytes(b"keep")
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=overwrite)
    assert error.value.code == "E_DOWNLOAD"
    assert error.value.details["expected_sha256"] == SHA
    assert (final.read_bytes() == b"old") if overwrite else not final.exists()
    assert sentinel.read_bytes() == b"keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [sentinel.name] + ([final.name] if overwrite else [])
    )
    temporary = Path(runner.downloads[0][1])
    assert temporary.parent == final.parent
    assert temporary != final
    assert not temporary.exists()


@pytest.mark.parametrize("failure", [OSError("partial transfer"), VastComfyError("E_SSH_COMMAND", "transfer failed")])
def test_failed_transfer_preserves_old_file_and_removes_partial_temporary(failure, cfg, tmp_path):
    runner = _outputs([f"{ROOT}/file.png"])
    final = tmp_path / "file.png"
    final.write_bytes(b"old")
    original_download = runner.download

    def partial(remote, local):
        original_download(remote, local)
        Path(local).write_bytes(b"partial")
        raise failure

    runner.download = partial
    with pytest.raises(VastComfyError):
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=True)
    assert final.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [final]


@pytest.mark.parametrize("overwrite", [False, True])
def test_publication_failure_preserves_old_destination_and_cleans_temp(overwrite, cfg, tmp_path, monkeypatch):
    runner = _outputs([f"{ROOT}/file.png"])
    final = tmp_path / "file.png"
    if overwrite:
        final.write_bytes(b"old")

    def denied(*args, **kwargs):
        raise PermissionError("publication denied")

    monkeypatch.setattr(dl.os, "replace" if overwrite else "link", denied)
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=overwrite)
    assert error.value.code == "E_DOWNLOAD"
    assert (final.read_bytes() == b"old") if overwrite else not final.exists()
    assert list(tmp_path.iterdir()) == ([final] if overwrite else [])


def test_no_overwrite_is_atomic_against_destination_created_at_publication(cfg, tmp_path, monkeypatch):
    runner = _outputs([f"{ROOT}/file.png"])
    original_link = os.link
    final = tmp_path / "file.png"

    def competing_writer(source, destination, **kwargs):
        Path(destination).write_bytes(b"competing writer")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(dl.os, "link", competing_writer)
    with pytest.raises(VastComfyError) as error:
        dl.download_outputs(runner, cfg, dest_dir=tmp_path)
    assert error.value.code == "E_DOWNLOAD"
    assert final.read_bytes() == b"competing writer"
    assert list(tmp_path.iterdir()) == [final]


def test_success_uses_unique_same_directory_temporaries_and_only_explicit_overwrite(cfg, tmp_path, monkeypatch):
    runner = _outputs([f"{ROOT}/nested/a.png", f"{ROOT}/nested/b.png"])
    parent = tmp_path / "nested"
    parent.mkdir()
    old = parent / "a.png"
    old.write_bytes(b"old")
    unrelated = parent / "keep.png"
    unrelated.write_bytes(b"keep")
    original_hash = dl._sha256_local
    original_replace = os.replace
    verified = []
    promoted = []

    def verified_hash(path):
        assert old.read_bytes() == (CONTENT if promoted else b"old")
        verified.append(path)
        return original_hash(path)

    def checked_replace(source, destination):
        assert source in verified
        assert Path(destination) in (parent / "a.png", parent / "b.png")
        promoted.append(destination)
        return original_replace(source, destination)

    monkeypatch.setattr(dl, "_sha256_local", verified_hash)
    monkeypatch.setattr(dl.os, "replace", checked_replace)
    result = dl.download_outputs(runner, cfg, dest_dir=tmp_path, overwrite=True)
    temporary_paths = [Path(local) for _, local in runner.downloads]
    assert len(set(temporary_paths)) == 2
    assert all(path.parent == parent and not path.exists() for path in temporary_paths)
    assert [record["sha256"] for record in result["downloaded"]] == [SHA, SHA]
    assert (parent / "a.png").read_bytes() == CONTENT
    assert (parent / "b.png").read_bytes() == CONTENT
    assert unrelated.read_bytes() == b"keep"
    assert sorted(path.name for path in parent.iterdir()) == ["a.png", "b.png", "keep.png"]
