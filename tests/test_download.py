"""SPEC 17 item 13 (download side): explicit downloads, overwrite refusal, sha256."""

from __future__ import annotations

import hashlib
import json as jsonlib
from pathlib import Path

import pytest

from mocks.fake_ssh import FakeRunner
from vast_comfy import download as dl
from vast_comfy.errors import VastComfyError

RUN_ID = "run-1"
REPORT_NAMES = ("report.json", "report.md", "prepare_summary.json", "summary.json")
REPORT_CONTENT = b"report-bytes-for-sha256"
REPORT_SHA = hashlib.sha256(REPORT_CONTENT).hexdigest()
OUTPUT_ROOT = "/workspace/ComfyUI/output"


def _report_runner() -> FakeRunner:
    runner = FakeRunner()
    runner.add_rule("test -f", rc=0)  # every candidate report file exists remotely
    runner.add_rule("sha256sum ", stdout=f"{REPORT_SHA}  remote-file\n", rc=0)
    for name in REPORT_NAMES:
        remote_path = f"/workspace/.vast_comfy/runs/{RUN_ID}/{name}"
        runner.remote_files[remote_path] = REPORT_CONTENT
    return runner


# ---------------------------------------------------------------------------
# download_report
# ---------------------------------------------------------------------------


def test_download_report_refuses_to_overwrite_without_flag(cfg, tmp_path):
    runner = _report_runner()
    dest = tmp_path / "out"
    dest.mkdir()
    local = dest / f"{RUN_ID}-report.json"
    local.write_text("already here", encoding="utf-8")

    with pytest.raises(VastComfyError) as excinfo:
        dl.download_report(runner, cfg, run_id=RUN_ID, dest_dir=dest)

    error = excinfo.value
    assert error.code == "E_DOWNLOAD"
    assert "refusing to overwrite" in error.message
    assert error.details["local"] == str(local)
    assert local.read_text(encoding="utf-8") == "already here"
    assert runner.downloads == []  # nothing was transferred


def test_download_report_success_verifies_sha256(cfg, tmp_path):
    runner = _report_runner()
    dest = tmp_path / "out"

    result = dl.download_report(runner, cfg, run_id=RUN_ID, dest_dir=dest)

    assert result["ok"] is True
    assert len(result["downloaded"]) == 4
    for record in result["downloaded"]:
        assert record["sha256"] == REPORT_SHA
        assert record["size_bytes"] == len(REPORT_CONTENT)
        local = Path(record["local"])
        assert local.is_file()
        assert local.read_bytes() == REPORT_CONTENT
        assert local.name.startswith(f"{RUN_ID}-")

    names = sorted(Path(record["local"]).name.replace(f"{RUN_ID}-", "") for record in result["downloaded"])
    assert names == sorted(REPORT_NAMES)
    # The sha256sum output from the "instance" was really used.
    assert any("sha256sum" in call.joined for call in runner.calls)


def test_download_report_with_overwrite_replaces_the_file(cfg, tmp_path):
    runner = _report_runner()
    dest = tmp_path / "out"
    dest.mkdir()
    local = dest / f"{RUN_ID}-report.json"
    local.write_text("stale", encoding="utf-8")

    result = dl.download_report(runner, cfg, run_id=RUN_ID, dest_dir=dest, overwrite=True)

    assert local.read_bytes() == REPORT_CONTENT
    assert len(result["downloaded"]) == 4


def test_download_report_tampered_sha_deletes_the_local_file(cfg, tmp_path):
    runner = _report_runner()
    # The instance reports the honest sha but the transfer is corrupted.
    for name in REPORT_NAMES:
        remote_path = f"/workspace/.vast_comfy/runs/{RUN_ID}/{name}"
        runner.download_bytes[remote_path] = b"tampered content"
    dest = tmp_path / "out"

    with pytest.raises(VastComfyError) as excinfo:
        dl.download_report(runner, cfg, run_id=RUN_ID, dest_dir=dest)

    error = excinfo.value
    assert error.code == "E_DOWNLOAD"
    assert error.details["expected_sha256"] == REPORT_SHA
    assert error.details["actual_sha256"] == hashlib.sha256(b"tampered content").hexdigest()
    assert not (dest / f"{RUN_ID}-report.json").exists()  # removed, never trusted


def test_download_report_missing_remote_files_is_download_error(cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("test -f", rc=1)  # no report files exist remotely
    with pytest.raises(VastComfyError) as excinfo:
        dl.download_report(runner, cfg, run_id=RUN_ID, dest_dir=tmp_path / "out")
    assert excinfo.value.code == "E_DOWNLOAD"
    assert "No report files exist" in excinfo.value.message
    assert runner.downloads == []


# ---------------------------------------------------------------------------
# download_outputs
# ---------------------------------------------------------------------------


def test_download_outputs_parses_find_output_newest_first(cfg, tmp_path):
    runner = FakeRunner()
    newest = f"{OUTPUT_ROOT}/newest.png"
    older = f"{OUTPUT_ROOT}/older.png"
    content = b"png-bytes-x"
    sha = hashlib.sha256(content).hexdigest()
    find_output = (
        f"1700000100.5 {newest}\n"
        f"1700000000.25 {older}\n"
        f"1699999999.0 /tmp/outside.png\n"  # outside the output root: ignored
    )
    runner.add_rule("find ", stdout=find_output, rc=0)
    runner.add_rule("sha256sum ", stdout=f"{sha}  f\n", rc=0)
    runner.remote_files[newest] = content
    runner.remote_files[older] = content

    result = dl.download_outputs(runner, cfg, dest_dir=tmp_path / "out", pattern="*.png", limit=3)

    assert result["matched"] == 2
    assert [record["remote"] for record in result["downloaded"]] == [newest, older]
    assert [remote for remote, _ in runner.downloads] == [newest, older]  # newest first

    script = runner.calls[0].argv[0]
    assert "find /workspace/ComfyUI/output" in script
    assert "-printf" in script
    assert "sort -rn" in script
    assert "head -n 3" in script

    assert (tmp_path / "out" / "newest.png").read_bytes() == content
    assert not (tmp_path / "out" / "outside.png").exists()


def test_download_outputs_respects_limit_in_the_remote_command(cfg, tmp_path):
    runner = FakeRunner()
    runner.add_rule("find ", stdout="", rc=0)
    result = dl.download_outputs(runner, cfg, dest_dir=tmp_path / "out", limit=7)
    assert result["matched"] == 0
    assert "head -n 7" in runner.calls[0].argv[0]


def test_download_smoke_images_uses_the_report_index(cfg, tmp_path):
    runner = FakeRunner()
    image_path = f"{OUTPUT_ROOT}/vast_comfy_smoke_base_t2i_00001_.png"
    report = {
        "cases": [
            {"id": "base_t2i", "images": [{"path": image_path}, {"path": "/etc/passwd"}]},
        ]
    }
    content = b"png-bytes"
    sha = hashlib.sha256(content).hexdigest()
    runner.add_rule("cat ", stdout=jsonlib.dumps(report) + "\n", rc=0)
    runner.add_rule("sha256sum ", stdout=f"{sha}  f\n", rc=0)
    runner.remote_files[image_path] = content

    result = dl.download_smoke_images(runner, cfg, run_id=RUN_ID, dest_dir=tmp_path / "out")

    assert result["count"] == 1
    assert result["downloaded"][0]["remote"] == image_path
    assert (tmp_path / "out" / RUN_ID / "vast_comfy_smoke_base_t2i_00001_.png").read_bytes() == content
