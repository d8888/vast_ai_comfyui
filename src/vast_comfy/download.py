"""Explicit downloads from the instance (SPEC.md 12.3).

Nothing is fetched automatically: every download is an explicit command.
Local files are never silently overwritten (``--overwrite`` is required), and
remote sha256 values are verified before files are accepted locally.
"""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from .config import Config
from .errors import E_DOWNLOAD, VastComfyError
from .remote import read_remote_json, runtime_root
from .ssh import SSHRunner


def _sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_sha256(runner: SSHRunner, remote_path: str) -> str | None:
    result = runner.run_remote(["sha256sum", remote_path], check=False)
    if result.rc != 0:
        return None
    return result.stdout.strip().split()[0] if result.stdout.strip() else None


def _download_one(
    runner: SSHRunner, remote_path: str, local_path: Path, *, overwrite: bool
) -> dict:
    if local_path.exists() and not overwrite:
        raise VastComfyError(
            code=E_DOWNLOAD,
            message=f"Local file already exists (refusing to overwrite): {local_path}",
            stage="download",
            details={"remote": remote_path, "local": str(local_path)},
            hints=["Re-run with --overwrite after checking the local file."],
        )
    expected = _remote_sha256(runner, remote_path)
    if expected is None:
        raise VastComfyError(
            code=E_DOWNLOAD,
            message=f"Remote file is unavailable: {remote_path}",
            stage="download",
        )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    runner.download(remote_path, str(local_path))
    actual = _sha256_local(local_path)
    if actual != expected:
        local_path.unlink(missing_ok=True)
        raise VastComfyError(
            code=E_DOWNLOAD,
            message="Downloaded file failed sha256 verification and was removed",
            stage="download",
            details={"remote": remote_path, "expected_sha256": expected, "actual_sha256": actual},
        )
    return {
        "remote": remote_path,
        "local": str(local_path),
        "sha256": actual,
        "size_bytes": local_path.stat().st_size,
    }


def download_report(
    runner: SSHRunner, cfg: Config, *, run_id: str, dest_dir: Path | str, overwrite: bool = False
) -> dict:
    base = f"{runtime_root(cfg)}/runs/{run_id}"
    dest = Path(dest_dir)
    downloaded = []
    for name in ("report.json", "report.md", "prepare_summary.json", "summary.json"):
        remote_path = f"{base}/{name}"
        exists = runner.run_remote(["test", "-f", remote_path], check=False).rc == 0
        if not exists:
            continue
        downloaded.append(
            _download_one(runner, remote_path, dest / f"{run_id}-{name}", overwrite=overwrite)
        )
    if not downloaded:
        raise VastComfyError(
            code=E_DOWNLOAD,
            message=f"No report files exist for run {run_id!r}",
            stage="download",
            details={"run_id": run_id, "remote_dir": base},
        )
    return {"ok": True, "run_id": run_id, "dest_dir": str(dest), "downloaded": downloaded}


def download_smoke_images(
    runner: SSHRunner,
    cfg: Config,
    *,
    run_id: str,
    dest_dir: Path | str,
    overwrite: bool = False,
) -> dict:
    report = read_remote_json(runner, cfg, f"{runtime_root(cfg)}/runs/{run_id}/report.json")
    output_root = f"{cfg.require('image.comfyui_dir').rstrip('/')}/output"
    dest = Path(dest_dir) / run_id
    downloaded = []
    for case in report.get("cases", []):
        for image in case.get("images", []) or []:
            remote_path = str(image.get("path", ""))
            if not remote_path.startswith(output_root + "/"):
                continue
            rel = remote_path[len(output_root) + 1 :]
            downloaded.append(
                _download_one(runner, remote_path, dest / rel, overwrite=overwrite)
            )
    return {"ok": True, "run_id": run_id, "dest_dir": str(dest), "count": len(downloaded), "downloaded": downloaded}


def download_outputs(
    runner: SSHRunner,
    cfg: Config,
    *,
    dest_dir: Path | str,
    pattern: str = "*.png",
    limit: int = 100,
    overwrite: bool = False,
) -> dict:
    output_root = f"{cfg.require('image.comfyui_dir').rstrip('/')}/output"
    script = (
        f"find {shlex.quote(output_root)} -type f -name {shlex.quote(pattern)} "
        f"-printf '%T@ %p\\n' | sort -rn | head -n {int(limit)}"
    )
    result = runner.run_script(script, check=True)
    dest = Path(dest_dir)
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        _, _, remote_path = line.partition(" ")
        if not remote_path.startswith(output_root + "/"):
            continue
        rel = remote_path[len(output_root) + 1 :]
        entries.append((remote_path, dest / rel))
    downloaded = [_download_one(runner, remote, local, overwrite=overwrite) for remote, local in entries]
    return {
        "ok": True,
        "dest_dir": str(dest),
        "matched": len(entries),
        "downloaded": downloaded,
        "note": "This is an explicit one-shot download; nothing is synchronized continuously.",
    }
