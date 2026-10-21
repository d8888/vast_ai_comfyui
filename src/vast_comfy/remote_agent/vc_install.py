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
import errno
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import uuid
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
    _validate_model_entry(entry)
    file_id = C.validate_identifier(entry["id"], label="model id")
    dest_rel = C.validate_model_relpath(entry["dest"])
    dest = C.confined_path(C.confined_path(C.COMFY_DIR, "models"), dest_rel)
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
    downloads = C.confined_path(C.RUNTIME_DIR, "downloads")
    downloads.mkdir(parents=True, exist_ok=True)
    tmp = C.confined_path(downloads, f"{file_id}.part")
    if (downloads / f"{file_id}.part").is_symlink():
        C.fail("E_INSTALL_PATH_ESCAPE", "A download partial must not be a symlink.")

    headers = _curl_headers(entry)
    argv = [
        "curl",
        "-fL",
        "--proto", "=https",
        "--proto-redir", "=https",
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
    _promote_without_overwrite(tmp, dest, expected_sha)
    return {"id": file_id, "action": "downloaded", "path": str(dest), "size_bytes": actual_size}


def _promote_without_overwrite(tmp: Path, dest: Path, expected_sha: str) -> None:
    """Atomic no-clobber publication, including separately mounted model dirs."""
    staged = None
    staged_owned = False
    try:
        try:
            os.link(tmp, dest, follow_symlinks=False)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            staged = dest.with_name("." + dest.name + ".vast-comfy-" + uuid.uuid4().hex + ".part")
            with staged.open("xb") as target:
                staged_owned = True
                with tmp.open("rb") as source:
                    shutil.copyfileobj(source, target)
            if C.sha256_file(staged) != expected_sha:
                C.fail("E_INSTALL_HASH_MISMATCH", "Cross-filesystem staging failed checksum verification.")
            os.link(staged, dest, follow_symlinks=False)
    except FileExistsError as exc:
        C.fail("E_INSTALL", "Destination appeared during publication; it was NOT overwritten.", path=str(dest))
    finally:
        if staged_owned:
            staged.unlink(missing_ok=True)
    tmp.unlink()


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        **C.child_process_options(),
    )


def _install_plugin(entry: dict) -> dict:
    _validate_plugin_entry(entry)
    plugin_id = entry["id"]
    rel = C.validate_plugin_relpath(entry["dest"])
    dest = C.confined_path(C.COMFY_DIR, rel)
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
        if dirty.returncode != 0 or dirty.stdout.strip():
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
    try:
        dest.mkdir()  # Exclusive claim: never git-init an existing user directory.
    except FileExistsError:
        C.fail("E_INSTALL", "Plugin destination appeared; nothing was overwritten.")
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
                f"git {' '.join(step[:2])} failed for {plugin_id}: {C.redact(result.stderr.strip())[:300]}",
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
    result = {"id": plugin_id, "action": "installed", "path": str(dest), "commit": commit}
    if entry.get("requirements"):
        result["requirements"] = _install_requirements(plugin_id, dest, entry["requirements"])
    return result


# Packages that must never be touched by a plugin's requirements: replacing the
# image's CUDA-matched torch build breaks every other workflow on the instance.
_PROTECTED_PACKAGES = ("torch", "torchvision", "torchaudio", "xformers", "pytorch")
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _install_requirements(plugin_id: str, dest: Path, req_rel: str) -> dict:
    """Install a plugin's requirements without letting it rewrite torch.

    Many ComfyUI plugins pin torch/CUDA versions that would replace the image's
    working build and break every existing workflow. Those lines are dropped and
    reported; everything else installs normally, with the current torch pinned
    as a constraint so a transitive dependency cannot upgrade it either.
    """
    req_path = C.confined_path(dest, req_rel)
    if not req_path.is_file():
        C.fail(
            "E_INSTALL",
            f"Plugin {plugin_id} declares requirements {req_rel!r} but no such file was checked out.",
            exit_code=1,
            id=plugin_id,
            path=str(req_path),
        )
    keep, skipped = [], []
    for raw in req_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = _REQ_NAME.match(line)
        name = (match.group(1) if match else "").lower().replace("_", "-")
        if any(name == p or name.startswith(p + "-") for p in _PROTECTED_PACKAGES):
            skipped.append(line)
            continue
        keep.append(line)

    python = str(os.environ.get("VAST_COMFY_VENV_PYTHON") or "/venv/main/bin/python")
    if not keep:
        return {"action": "skipped", "installed": [], "skipped": skipped, "reason": "no non-protected requirements"}

    tmp_dir = C.RUNTIME_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    filtered = tmp_dir / f"req-{plugin_id}-{uuid.uuid4().hex}.txt"
    filtered.write_text("\n".join(keep) + "\n", encoding="utf-8")
    argv = [python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "-r", str(filtered)]
    constraints = None
    try:
        version = subprocess.run(
            [python, "-m", "pip", "show", "torch"], capture_output=True, text=True,
            **C.child_process_options(),
        )
        constraints = None
        if version.returncode == 0:
            found = re.search(r"^Version:\s*(\S+)", version.stdout, re.MULTILINE)
            if found:
                constraints = tmp_dir / f"con-{plugin_id}-{uuid.uuid4().hex}.txt"
                constraints.write_text(f"torch=={found.group(1)}\n", encoding="utf-8")
                argv += ["-c", str(constraints)]
        rc = _run_checked(argv)
        if rc != 0:
            # The checkout is valid but unusable without its dependencies; say so
            # instead of reporting a plugin that will fail to import at runtime.
            C.fail(
                "E_INSTALL",
                f"pip install failed for plugin {plugin_id} (rc={rc}); the checkout is kept for inspection.",
                exit_code=1,
                id=plugin_id,
                path=str(dest),
                skipped_requirements=skipped,
            )
    finally:
        filtered.unlink(missing_ok=True)
        if constraints is not None:
            constraints.unlink(missing_ok=True)
    return {"action": "installed", "installed": keep, "skipped": skipped}


def _validate_model_entry(entry: dict) -> None:
    try:
        if not isinstance(entry, dict) or set(entry) - {"id", "url", "dest", "sha256", "size_bytes", "auth_env"}:
            raise ValueError("invalid model fields")
        C.validate_identifier(entry["id"], label="model id")
        rel = C.validate_model_relpath(entry["dest"])
        C.confined_path(C.confined_path(C.COMFY_DIR, "models"), rel)
        url = C.validate_public_url(entry["url"])
        parsed = urllib.parse.urlsplit(url)
        parts = parsed.path.strip("/").split("/")
        pinned_hf = (parsed.hostname == "huggingface.co" and len(parts) >= 5 and parts[2] == "resolve"
                     and re.fullmatch(r"[0-9a-f]{40}", parts[3]))
        pinned_civitai = (parsed.hostname == "civitai.com"
                          and re.fullmatch(r"/api/download/models/[1-9][0-9]*", parsed.path))
        if not (pinned_hf or pinned_civitai) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("unpinned source/hash")
        size = entry.get("size_bytes")
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("invalid size")
    except (KeyError, TypeError, ValueError):
        C.fail("E_CONFIG", "Invalid or unsafe model manifest entry; no download was started.")
    if entry.get("auth_env"):
        C.fail("E_APPROVAL_REQUIRED", "Remote credentials are not supported in v1; no download was started.")


def _validate_plugin_entry(entry: dict) -> None:
    try:
        if not isinstance(entry, dict) or set(entry) - {"id", "repo_url", "commit", "dest", "required_nodes", "requirements"}:
            raise ValueError("invalid plugin fields")
        C.validate_identifier(entry["id"], label="plugin id")
        C.confined_path(C.COMFY_DIR, C.validate_plugin_relpath(entry["dest"]))
        C.validate_public_url(entry["repo_url"])
        if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
            raise ValueError("unpinned plugin commit")
        nodes = entry.get("required_nodes", [])
        if not isinstance(nodes, list) or any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError("invalid required_nodes")
        req = entry.get("requirements")
        if req is not None:
            if (not isinstance(req, str) or not req.endswith(".txt") or req.startswith("/")
                    or ".." in req.split("/") or "\\" in req):
                raise ValueError("invalid requirements path")
    except (KeyError, TypeError, ValueError):
        C.fail("E_CONFIG", "Invalid or unsafe plugin manifest entry; no checkout was started.")


def _validate_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or set(manifest) != {"models", "plugins"}
            or not isinstance(manifest["models"], list) or not isinstance(manifest["plugins"], list)):
        C.fail("E_CONFIG", "Invalid install manifest schema.")
    ids, destinations = set(), set()
    for kind, validator in (("models", _validate_model_entry), ("plugins", _validate_plugin_entry)):
        for entry in manifest[kind]:
            validator(entry)
            dest = ("models/" if kind == "models" else "") + entry["dest"]
            if entry["id"] in ids or dest in destinations:
                C.fail("E_CONFIG", "Duplicate install identifier or destination.")
            ids.add(entry["id"]); destinations.add(dest)


def _check_stage_deadline(deadline: float | None) -> None:
    if deadline is not None and C.now_epoch() >= deadline:
        C.fail("E_JOB_TIMEOUT", "The install time budget was exceeded; no further item was started.",
               job_may_still_be_running=False, takeover_required=True)


def _install_manifest(manifest: dict, *, deadline: float | None = None) -> dict:
    _validate_manifest(manifest)  # Validate the entire batch before its first mutation.
    models = []
    plugins = []
    for entry in manifest.get("models", []):
        C.check_deadline()
        _check_stage_deadline(deadline)
        models.append(_download(entry))
    for entry in manifest.get("plugins", []):
        C.check_deadline()
        _check_stage_deadline(deadline)
        plugins.append(_install_plugin(entry))
    # The last item's own duration also counts: an install stage that already
    # overran its budget must not hand off to restart/smoke as if on budget.
    _check_stage_deadline(deadline)
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
