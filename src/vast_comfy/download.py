"""Explicit downloads from the instance (SPEC.md 12.3).

Nothing is fetched automatically: every download is an explicit command.
Local files are never silently overwritten (``--overwrite`` is required), and
remote sha256 values are verified before files are accepted locally.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import stat
import tempfile
from pathlib import Path, PurePosixPath

from .config import Config
from .errors import E_DOWNLOAD, E_INTERNAL, VastComfyError
from .remote import read_remote_json
from .safety import validate_identifier, validate_path_component, validate_posix_path
from .ssh import SSHRunner


def _sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_sha256(runner: SSHRunner, remote_path: str) -> str | None:
    result = runner.run_remote(["sha256sum", "--", remote_path], check=False)
    if result.rc != 0:
        return None
    fields = result.stdout.split()
    if not fields or re.fullmatch(r"[A-Fa-f0-9]{64}", fields[0]) is None:
        return None
    return fields[0].lower()


def _error(message: str, **details) -> VastComfyError:
    return VastComfyError(code=E_DOWNLOAD, message=message, stage="download", details=details)


def _posix(value: str) -> PurePosixPath:
    try:
        return validate_posix_path(value, label="Remote download path")
    except VastComfyError as exc:
        raise _error(exc.message) from exc


def _relative_remote(remote_path: str, root: str) -> PurePosixPath:
    path, base = _posix(remote_path), _posix(root)
    if path == base or not path.is_relative_to(base):
        raise _error("Remote download path escapes its allowed root", remote=remote_path, root=root)
    return path.relative_to(base)


def _resolve_remote(
    runner: SSHRunner, root: str, remote_path: str, *, subdir: PurePosixPath = PurePosixPath()
) -> str:
    """Check remote symlink containment and transfer the checked canonical path.

    GNU realpath is already part of the Linux/coreutils toolchain used for
    sha256sum. Missing tools, missing files and malformed responses fail closed.
    """
    _relative_remote(remote_path, str(_posix(root) / subdir))
    result = runner.run_remote(["realpath", "-e", "--", root, remote_path], check=False)
    lines = result.stdout.removesuffix("\n").split("\n")
    if result.rc != 0 or len(lines) != 2:
        raise _error("Cannot safely resolve remote download path", remote=remote_path, root=root)
    resolved_root, resolved_path = lines
    _relative_remote(resolved_path, str(_posix(resolved_root) / subdir))
    return resolved_path


def _local_root(dest_dir: Path | str) -> Path:
    try:
        root = Path(dest_dir).resolve()
        if root.exists() and not root.is_dir():
            raise _error("Download destination must be a directory", local=str(root))
        return root
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise _error("Cannot resolve local download destination") from exc


def _check_local(root: Path, local_path: Path, *, overwrite: bool) -> None:
    try:
        resolved = local_path.resolve()
        if root.resolve() != root or resolved == root or not resolved.is_relative_to(root):
            raise _error("Local download path escapes its destination", local=str(local_path), root=str(root))
        # Never follow a final symlink, even for an explicitly allowed overwrite.
        if local_path.is_symlink():
            raise _error("Refusing a symlink download destination", local=str(local_path))
        if local_path.exists():
            if not local_path.is_file():
                raise _error("Download destination is not a regular file", local=str(local_path))
            if not overwrite:
                raise VastComfyError(
                    code=E_DOWNLOAD,
                    message=f"Local file already exists (refusing to overwrite): {local_path}",
                    stage="download",
                    details={"local": str(local_path)},
                    hints=["Re-run with --overwrite after checking the local file."],
                )
        for parent in local_path.parents:
            if parent.exists() and not parent.is_dir():
                raise _error("Download parent is not a directory", local=str(parent))
            if parent == root:
                break
    except (OSError, RuntimeError, ValueError) as exc:
        raise _error("Cannot validate local download destination", local=str(local_path)) from exc


def _plan_entry(remote: str, remote_root: str, local_root: Path, *, prefix: str = "") -> tuple[str, Path]:
    rel = _relative_remote(remote, remote_root)
    try:
        for component in rel.parts:
            validate_path_component(component)
    except VastComfyError as exc:
        raise _error(exc.message, remote=remote) from exc
    return remote, local_root.joinpath(prefix, *rel.parts)


def _prevalidate(
    runner: SSHRunner, entries: list[tuple[str, Path]], *, remote_root: str, local_root: Path, overwrite: bool,
    remote_subdir: PurePosixPath = PurePosixPath(),
) -> list[tuple[str, Path]]:
    seen = set()
    for remote, local in entries:
        _relative_remote(remote, remote_root)
        try:
            _check_local(local_root, local, overwrite=overwrite)
        except VastComfyError as exc:
            exc.details.setdefault("remote", remote)
            raise
        # Reject aliases on either supported controller platform before transfers.
        key = str(local.resolve()).casefold()
        if key in seen:
            raise _error("Duplicate local download destination in metadata", local=str(local))
        seen.add(key)
    return [
        (_resolve_remote(runner, remote_root, remote, subdir=remote_subdir), local)
        for remote, local in entries
    ]


def _download_one(
    runner: SSHRunner, remote_path: str, local_path: Path, *, overwrite: bool, local_root: Path | None = None
) -> dict:
    root = _local_root(local_path.parent) if local_root is None else local_root
    _check_local(root, local_path, overwrite=overwrite)
    expected = _remote_sha256(runner, remote_path)
    if expected is None:
        raise _error(f"Remote file or its sha256 is unavailable: {remote_path}")
    temporary = None
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _check_local(root, local_path, overwrite=overwrite)
        fd, name = tempfile.mkstemp(prefix=".vast-comfy-", suffix=".tmp", dir=local_path.parent)
        temporary = Path(name)
        os.close(fd)
        runner.download(remote_path, str(temporary))
        _check_local(root, local_path, overwrite=overwrite)
        if temporary.resolve().parent != local_path.parent.resolve() or not stat.S_ISREG(temporary.lstat().st_mode):
            raise _error("Downloaded temporary file is not a safe regular file")
        actual = _sha256_local(temporary)
        if actual != expected:
            raise _error(
                "Downloaded file failed sha256 verification; destination was not changed",
                remote=remote_path, expected_sha256=expected, actual_sha256=actual,
            )
        size = temporary.stat().st_size
        _check_local(root, local_path, overwrite=overwrite)
        if overwrite:
            # Atomic replacement: never unlink the old destination first.
            os.replace(temporary, local_path)
        else:
            # An atomic no-clobber publication, including when a competing writer
            # creates the destination after prevalidation. Fail closed if the
            # destination filesystem does not support hard links.
            os.link(temporary, local_path, follow_symlinks=False)
        return {"remote": remote_path, "local": str(local_path), "sha256": actual, "size_bytes": size}
    except OSError as exc:
        raise _error("Download transfer or publication failed; existing files were not removed", remote=remote_path,
                     local=str(local_path)) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise _error("Cannot remove download temporary file", temporary=str(temporary)) from exc


def download_report(
    runner: SSHRunner, cfg: Config, *, run_id: str, dest_dir: Path | str, overwrite: bool = False
) -> dict:
    validate_identifier(run_id, label="run_id")
    runtime = _posix(cfg.require("image.runtime_dir"))
    remote_root = str(runtime)
    base = str(runtime / "runs" / run_id)
    root = _local_root(dest_dir)
    dest = Path(dest_dir)
    entries = []
    for name in ("report.json", "report.md", "prepare_summary.json", "summary.json"):
        remote_path = f"{base}/{name}"
        exists = runner.run_remote(["test", "-f", remote_path], check=False).rc == 0
        if not exists:
            continue
        entries.append((remote_path, root / f"{run_id}-{name}"))
    if not entries:
        raise VastComfyError(
            code=E_DOWNLOAD,
            message=f"No report files exist for run {run_id!r}",
            stage="download",
            details={"run_id": run_id, "remote_dir": base},
        )
    entries = _prevalidate(
        runner, entries, remote_root=remote_root, local_root=root, overwrite=overwrite,
        remote_subdir=PurePosixPath("runs", run_id),
    )
    downloaded = [_download_one(runner, remote, local, overwrite=overwrite, local_root=root) for remote, local in entries]
    return {"ok": True, "run_id": run_id, "dest_dir": str(dest), "downloaded": downloaded}


def download_smoke_images(
    runner: SSHRunner,
    cfg: Config,
    *,
    run_id: str,
    dest_dir: Path | str,
    overwrite: bool = False,
) -> dict:
    validate_identifier(run_id, label="run_id")
    runtime = str(_posix(cfg.require("image.runtime_dir")))
    output_root = str(_posix(cfg.require("image.comfyui_dir")) / "output")
    root = _local_root(dest_dir)
    report_path = _resolve_remote(
        runner, runtime, str(PurePosixPath(runtime) / "runs" / run_id / "report.json"),
        subdir=PurePosixPath("runs", run_id),
    )
    try:
        report = read_remote_json(runner, cfg, report_path)
    except VastComfyError as exc:
        if exc.code == E_INTERNAL:
            raise _error("Smoke report contains invalid JSON metadata") from exc
        raise
    dest = Path(dest_dir) / run_id
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise _error("Smoke report cases must be a list")
    entries = []
    for case in report["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("images", []), list):
            raise _error("Smoke report cases must contain image lists")
        for image in case.get("images", []):
            if not isinstance(image, dict):
                raise _error("Smoke report image metadata must be an object")
            entries.append(_plan_entry(image.get("path"), output_root, root, prefix=run_id))
    entries = _prevalidate(runner, entries, remote_root=output_root, local_root=root, overwrite=overwrite)
    downloaded = [_download_one(runner, remote, local, overwrite=overwrite, local_root=root) for remote, local in entries]
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
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise _error("Download limit must be a positive integer")
    if not isinstance(pattern, str) or not pattern or any(ord(char) < 32 or ord(char) == 127 for char in pattern):
        raise _error("Download pattern must be a nonempty string without control characters")
    output_root = str(_posix(cfg.require("image.comfyui_dir")) / "output")
    root = _local_root(dest_dir)
    script = (
        f"find {shlex.quote(output_root)} -type f -name {shlex.quote(pattern)} "
        f"-printf '%T@ %p\\n' | sort -rn | head -n {limit}"
    )
    result = runner.run_script(script, check=True)
    dest = Path(dest_dir)
    entries = []
    lines = result.stdout.removesuffix("\n").split("\n") if result.stdout else []
    for line in lines:
        timestamp, separator, remote_path = line.partition(" ")
        if not separator or re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", timestamp) is None:
            raise _error("Output listing contains malformed metadata")
        entries.append(_plan_entry(remote_path, output_root, root))
    if len(entries) > limit:
        raise _error("Output listing exceeds the requested download limit")
    entries = _prevalidate(runner, entries, remote_root=output_root, local_root=root, overwrite=overwrite)
    downloaded = [_download_one(runner, remote, local, overwrite=overwrite, local_root=root) for remote, local in entries]
    return {
        "ok": True,
        "dest_dir": str(dest),
        "matched": len(entries),
        "downloaded": downloaded,
        "note": "This is an explicit one-shot download; nothing is synchronized continuously.",
    }
