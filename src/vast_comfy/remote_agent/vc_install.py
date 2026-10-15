#!/usr/bin/env python3
"""vast-comfy remote installer: models (resumable + sha256) and git plugins.

Safety rules (SPEC.md 11.3):

* Downloads go directly to the instance; the control machine is never a relay.
* Partial downloads live next to the target and are only promoted after the
  sha256 checksum matches.
* An existing destination whose checksum matches is skipped; a mismatch fails
  loudly and nothing is overwritten.
* Plugins are pinned to a commit; a dirty or mismatched checkout fails; there
  is no automatic reset/update.
* Destination paths are validated against the model/plugin allowlists.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402


def _run_checked(argv: list[str]) -> int:
    print(f"$ {C.redact(' '.join(argv))}", flush=True)
    return subprocess.call(argv, **C.child_process_options())


def _curl_headers(entry: dict) -> list[str]:
    auth_env = entry.get("auth_env")
    if not auth_env:
        return []
    token = os.environ.get(auth_env)
    if not token:
        C.fail(
            "E_APPROVAL_REQUIRED",
            f"This file requires a remote credential ({auth_env}) that was not provided.",
            exit_code=21,
            file_id=entry.get("id"),
            auth_env=auth_env,
            hint="Sensitive credentials are only sent after explicit user approval.",
        )
    # Token is passed via header; never via URL, never printed.
    return [f"Authorization: Bearer {token}"]


def _download(entry: dict, *, timeout_note: str = "") -> dict:
    file_id = str(entry.get("id", "file"))
    dest_rel = C.validate_model_relpath(str(entry["dest"]))
    dest = C.COMFY_DIR / "models" / dest_rel
    expected_sha = str(entry["sha256"]).lower()
    expected_size = entry.get("size_bytes")

    if dest.is_file():
        actual = C.sha256_file(dest)
        if actual == expected_sha:
            return {"id": file_id, "action": "skipped", "path": str(dest)}
        C.fail(
            "E_INSTALL_HASH_MISMATCH",
            f"Existing file {dest} does not match the expected checksum. Nothing was overwritten.",
            exit_code=1,
            id=file_id,
            path=str(dest),
            expected_sha256=expected_sha,
            actual_sha256=actual,
            hint="User data is never replaced automatically; resolve the conflict yourself.",
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    C.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = C.DOWNLOADS_DIR / f"{file_id}.part"

    headers = _curl_headers(entry)
    argv = [
        "curl",
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-o",
        str(tmp),
    ]
    if tmp.exists():
        # Resume where the previous attempt stopped (source permitting).
        argv += ["-C", "-"]
    for header in headers:
        argv += ["-H", header]
    url = str(entry["url"])
    argv.append(url)
    rc = _run_checked(argv)
    if rc != 0:
        C.fail(
            "E_INSTALL",
            f"Download failed for {file_id} (curl rc={rc}); partial file kept for resume.",
            exit_code=1,
            id=file_id,
            partial=str(tmp),
        )

    actual_size = tmp.stat().st_size
    if expected_size is not None and int(expected_size) != actual_size:
        C.fail(
            "E_INSTALL",
            f"Downloaded size mismatch for {file_id}: {actual_size} != {expected_size}. "
            "The partial file was kept; retry to resume.",
            exit_code=1,
            id=file_id,
            partial=str(tmp),
        )
    actual_sha = C.sha256_file(tmp)
    if actual_sha != expected_sha:
        C.fail(
            "E_INSTALL_HASH_MISMATCH",
            f"Checksum mismatch for {file_id}. The file was NOT promoted.",
            exit_code=1,
            id=file_id,
            partial=str(tmp),
            expected_sha256=expected_sha,
            actual_sha256=actual_sha,
        )

    if dest.exists():
        # Someone created the destination meanwhile: never overwrite.
        C.fail(
            "E_INSTALL",
            f"Destination appeared while downloading: {dest}. Nothing was overwritten.",
            exit_code=1,
            id=file_id,
            path=str(dest),
        )
    os.replace(tmp, dest)
    return {"id": file_id, "action": "downloaded", "path": str(dest), "size_bytes": actual_size}


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        **C.child_process_options(),
    )


def _install_plugin(entry: dict) -> dict:
    plugin_id = str(entry.get("id", "plugin"))
    rel = C.validate_plugin_relpath(str(entry["dest"]))
    dest = C.COMFY_DIR / rel
    commit = str(entry["commit"]).strip()
    repo = str(entry["repo_url"])

    if dest.exists():
        head = _git(["rev-parse", "HEAD"], cwd=dest)
        if head.returncode != 0:
            C.fail(
                "E_INSTALL",
                f"{dest} exists but is not a git checkout; refusing to touch it.",
                exit_code=1,
                id=plugin_id,
                path=str(dest),
            )
        dirty = _git(["status", "--porcelain"], cwd=dest)
        if dirty.stdout.strip():
            C.fail(
                "E_INSTALL_DIRTY",
                f"{dest} has local modifications; refusing to reset or overwrite them.",
                exit_code=1,
                id=plugin_id,
                path=str(dest),
                hint="Resolve or stash the changes yourself; vast-comfy never resets user work.",
            )
        if head.stdout.strip() != commit:
            C.fail(
                "E_INSTALL",
                f"{dest} is at commit {head.stdout.strip()[:12]} but {commit[:12]} is required; "
                "refusing to switch versions automatically.",
                exit_code=1,
                id=plugin_id,
                path=str(dest),
            )
        return {"id": plugin_id, "action": "skipped", "path": str(dest), "commit": commit}

    dest.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        ["init", "-q", str(dest)],
        ["-C", str(dest), "remote", "add", "origin", repo],
        ["-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit],
        ["-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD"],
    ]
    for step in steps:
        result = _git(step)
        if result.returncode != 0:
            C.fail(
                "E_INSTALL",
                f"git {' '.join(step[:2])} failed for {plugin_id}: {result.stderr.strip()[:300]}",
                exit_code=1,
                id=plugin_id,
                path=str(dest),
            )
    head = _git(["rev-parse", "HEAD"], cwd=dest)
    if head.returncode != 0 or head.stdout.strip() != commit:
        C.fail(
            "E_INSTALL",
            f"Plugin checkout verification failed for {plugin_id}.",
            exit_code=1,
            id=plugin_id,
            path=str(dest),
        )
    return {"id": plugin_id, "action": "installed", "path": str(dest), "commit": commit}


def _install_manifest(manifest: dict) -> dict:
    models = []
    plugins = []
    for entry in manifest.get("models", []):
        C.check_deadline()
        models.append(_download(entry))
    for entry in manifest.get("plugins", []):
        C.check_deadline()
        plugins.append(_install_plugin(entry))
    changed = any(m["action"] == "downloaded" for m in models) or any(
        p["action"] == "installed" for p in plugins
    )
    return {"ok": True, "models": models, "plugins": plugins, "changed": changed}


def cmd_run(args) -> None:
    C.ensure_dirs()
    C.ensure_owns(args.session_id)
    try:
        manifest = json.loads(base64.b64decode(args.manifest_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        C.fail("E_USAGE", f"invalid install manifest: {exc}")
    if not isinstance(manifest, dict):
        C.fail("E_USAGE", "install manifest must be a JSON object")
    result = _install_manifest(manifest)
    C.emit(result)


def cmd_verify(args) -> None:
    """Check installed models/plugins against the manifest without downloading."""
    C.ensure_dirs()
    C.ensure_owns(args.session_id)
    try:
        manifest = json.loads(base64.b64decode(args.manifest_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        C.fail("E_USAGE", f"invalid install manifest: {exc}")
    checks = []
    ok = True
    for entry in manifest.get("models", []):
        rel = C.validate_model_relpath(str(entry["dest"]))
        path = C.COMFY_DIR / "models" / rel
        record = {"kind": "model", "id": entry.get("id"), "path": str(path), "exists": path.is_file()}
        if record["exists"]:
            stat = path.stat()
            record["size_bytes"] = stat.st_size
            record["size_ok"] = entry.get("size_bytes") is None or int(entry["size_bytes"]) == stat.st_size
            if args.verify_shas or not record["size_ok"]:
                actual = C.sha256_file(path)
                record["sha256"] = actual
                record["sha256_ok"] = actual == str(entry["sha256"]).lower()
        else:
            record["size_ok"] = False
        ok = ok and record["exists"] and record.get("size_ok", False) and record.get("sha256_ok", True)
        checks.append(record)
    for entry in manifest.get("plugins", []):
        rel = C.validate_plugin_relpath(str(entry["dest"]))
        path = C.COMFY_DIR / rel
        record = {"kind": "plugin", "id": entry.get("id"), "path": str(path), "exists": path.is_dir()}
        if record["exists"]:
            head = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
            )
            record["commit"] = head.stdout.strip() if head.returncode == 0 else None
            record["commit_ok"] = record["commit"] == entry.get("commit")
            dirty = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True
            )
            record["dirty"] = bool(dirty.stdout.strip())
            ok = ok and record["commit_ok"] and not record["dirty"]
        else:
            ok = False
        checks.append(record)
    C.emit({"ok": ok, "checks": checks}, exit_code=0 if ok else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote installer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--session-id", required=True)
    p.add_argument("--manifest-b64", required=True)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("verify")
    p.add_argument("--session-id", required=True)
    p.add_argument("--manifest-b64", required=True)
    p.add_argument("--verify-shas", action="store_true")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    C.run_cli(args.func, args, guard=args.command == "run")


if __name__ == "__main__":
    main()
