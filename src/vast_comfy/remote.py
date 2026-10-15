"""Whitelisted deployment of the management package + remote tool invocation.

Only this project's own small management components are auto-uploaded (SPEC.md
12.1): the remote_agent scripts, workflow JSONs and the smoke test assets.
The whitelist is explicit; secrets (.env, keys), user files, model caches and
anything unknown can never be packaged. Management files are small, so they are
uploaded as-is with a sha256 manifest and verified remotely.

The package root is ``<runtime_dir>/pkg/<content-hash>/`` and ``current`` is a
symlink pointing at it, so switching versions is atomic and old versions remain
for diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import E_DEPLOY, E_INTERNAL, VastComfyError
from .logging_setup import get_logger
from .paths import find_repo_root
from .ssh import SSHRunner

logger = get_logger("remote")

MAX_FILE_BYTES = 1_500_000
MAX_FILES = 300

_FORBIDDEN_NAME_PATTERNS = [
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.pub)?$"),
    re.compile(r"(^|/)known_hosts$"),
    re.compile(r"\.(pem|ppk|key|p12|pfx)$", re.IGNORECASE),
]


def package_files() -> list[tuple[Path, str]]:
    """The explicit whitelist of auto-uploaded files: (local_path, remote_rel)."""
    root = find_repo_root()
    files: list[tuple[Path, str]] = []

    agent_dir = Path(__file__).resolve().parent / "remote_agent"
    for path in sorted(agent_dir.glob("*.py")):
        files.append((path, f"remote/{path.name}"))

    for sub in ("api", "gui"):
        workflow_dir = root / "workflows" / sub
        if not workflow_dir.is_dir():
            raise VastComfyError(
                code=E_DEPLOY,
                message=f"Missing workflow directory: {workflow_dir}",
                stage="deploy",
            )
        for path in sorted(workflow_dir.glob("*.json")):
            files.append((path, f"workflows/{sub}/{path.name}"))

    smoke_dir = root / "assets" / "smoke"
    if not smoke_dir.is_dir():
        raise VastComfyError(code=E_DEPLOY, message=f"Missing assets directory: {smoke_dir}", stage="deploy")
    for path in sorted(smoke_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".png", ".yaml", ".md"):
            files.append((path, f"assets/smoke/{path.name}"))

    return files


def assert_upload_safe(files: list[tuple[Path, str]]) -> None:
    if len(files) > MAX_FILES:
        raise VastComfyError(
            code=E_DEPLOY,
            message=f"Deployment package has {len(files)} files (> {MAX_FILES}); refusing the whitelist.",
            stage="deploy",
        )
    for path, rel in files:
        for pattern in _FORBIDDEN_NAME_PATTERNS:
            if pattern.search(rel):
                raise VastComfyError(
                    code=E_DEPLOY,
                    message=f"Refusing to package a secret-like file: {rel}",
                    stage="deploy",
                )
        if not path.is_file():
            raise VastComfyError(code=E_DEPLOY, message=f"Package file missing: {path}", stage="deploy")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise VastComfyError(
                code=E_DEPLOY,
                message=f"Package file too large ({size} bytes > {MAX_FILE_BYTES}): {rel}",
                stage="deploy",
                hints=["Only small management components are auto-uploaded; large files are downloaded on the instance."],
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PackageInfo:
    records: list[dict]
    manifest_text: str
    package_hash: str

    @property
    def remote_dir(self) -> str:
        return f"__RUNTIME__/pkg/{self.package_hash}"


def build_package() -> PackageInfo:
    files = package_files()
    assert_upload_safe(files)
    records = []
    for path, rel in files:
        records.append({"rel": rel, "sha256": _sha256(path), "size": path.stat().st_size, "local": str(path)})
    manifest_text = "".join(f"{r['sha256']}  {r['rel']}\n" for r in records)
    package_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()[:16]
    return PackageInfo(records=records, manifest_text=manifest_text, package_hash=package_hash)


def runtime_root(cfg: Config) -> str:
    return str(cfg.require("image.runtime_dir")).rstrip("/")


def package_dir(cfg: Config, package_hash: str) -> str:
    return f"{runtime_root(cfg)}/pkg/{package_hash}"


def current_link(cfg: Config) -> str:
    return f"{runtime_root(cfg)}/current"


def deploy_package(runner: SSHRunner, cfg: Config, *, force: bool = False) -> dict:
    """Upload (idempotently) and verify the management package."""
    info = build_package()
    remote_pkg = package_dir(cfg, info.package_hash)
    venv_python = str(cfg.require("image.venv_python"))
    runtime = runtime_root(cfg)

    runner.run_remote(["mkdir", "-p", remote_pkg, runtime], check=True)

    verified_already = False
    if not force:
        probe = runner.run_remote(
            [
                venv_python,
                f"{current_link(cfg)}/remote/vc_common.py",
                "verify-package",
                "--directory",
                remote_pkg,
                "--manifest",
                f"{remote_pkg}/manifest.sha256",
            ],
            check=False,
        )
        if probe.rc == 0:
            try:
                payload = json.loads(probe.stdout.strip().splitlines()[-1])
                verified_already = bool(payload.get("ok"))
            except (ValueError, IndexError):
                verified_already = False

    uploaded = 0
    if not verified_already:
        # Upload into the content-addressed directory, then verify.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="vast-comfy-pkg-") as tmp:
            tmp_dir = Path(tmp)
            for record in info.records:
                rel_parent = f"{remote_pkg}/{Path(record['rel']).parent}"
                runner.run_remote(["mkdir", "-p", rel_parent], check=True)
                runner.upload(record["local"], f"{remote_pkg}/{record['rel']}")
                uploaded += 1
            manifest_local = tmp_dir / "manifest.sha256"
            manifest_local.write_text(info.manifest_text, encoding="utf-8")
            runner.upload(str(manifest_local), f"{remote_pkg}/manifest.sha256")

        verify = runner.run_remote(
            [
                venv_python,
                f"{remote_pkg}/remote/vc_common.py",
                "verify-package",
                "--directory",
                remote_pkg,
                "--manifest",
                f"{remote_pkg}/manifest.sha256",
            ],
            check=False,
        )
        try:
            payload = json.loads(verify.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise VastComfyError(
                code=E_DEPLOY,
                message="Remote package verification did not return a parsable result",
                stage="deploy",
                details={"rc": verify.rc, "stdout_excerpt": verify.stdout[-300:].strip()},
            ) from exc
        if verify.rc != 0 or not payload.get("ok"):
            raise VastComfyError(
                code=E_DEPLOY,
                message="Remote package verification failed; the deployment is not trusted",
                stage="deploy",
                details={"rc": verify.rc, "verify": payload},
            )

    # Point `current` at the verified package (atomic symlink swap).
    runner.run_remote(["ln", "-sfn", remote_pkg, current_link(cfg)], check=True)

    return {
        "package_hash": info.package_hash,
        "remote_dir": remote_pkg,
        "current": current_link(cfg),
        "files": len(info.records),
        "uploaded": uploaded,
        "already_present": verified_already,
        "verified": True,
    }


def invoke(
    runner: SSHRunner,
    cfg: Config,
    tool: str,
    argv: list[str],
    *,
    timeout_seconds: int | None = None,
) -> dict:
    """Run a deployed remote tool and parse its single JSON result.

    A non-zero rc with a JSON ``{"ok": false, "code": ...}`` payload is turned
    back into a structured :class:`VastComfyError` with the remote code.
    """
    venv_python = str(cfg.require("image.venv_python"))
    tool_path = f"{current_link(cfg)}/remote/{tool}"
    _validate_tool_name(tool)
    result = runner.run_remote(
        [venv_python, tool_path, *argv], timeout_seconds=timeout_seconds, check=False
    )
    payload: dict | None = None
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise VastComfyError(
            code=E_INTERNAL if result.rc == 0 else E_INTERNAL,
            message=f"Remote tool {tool} did not return a JSON result (rc={result.rc})",
            stage="remote",
            details={
                "tool": tool,
                "stdout_excerpt": result.stdout[-400:].strip(),
                "stderr_excerpt": result.stderr[-400:].strip(),
            },
        )
    if result.rc != 0 or payload.get("ok") is False:
        code = str(payload.get("code", E_INTERNAL))
        message = str(payload.get("message", f"Remote tool {tool} failed (rc={result.rc})"))
        details = {k: v for k, v in payload.items() if k not in ("ok", "code", "message")}
        details.setdefault("remote_tool", tool)
        raise VastComfyError(code=code, message=message, stage="remote", details=details)
    return payload


def read_remote_json(runner: SSHRunner, cfg: Config, remote_path: str) -> dict:
    result = runner.run_remote(["cat", remote_path], check=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VastComfyError(
            code=E_INTERNAL,
            message=f"Remote JSON file is unreadable: {remote_path}",
            stage="remote",
            details={"excerpt": result.stdout[:300]},
        ) from exc
    if not isinstance(payload, dict):
        raise VastComfyError(code=E_INTERNAL, message=f"Remote JSON file has unexpected shape: {remote_path}")
    return payload


def _validate_tool_name(tool: str) -> None:
    if not re.fullmatch(r"vc_[a-z_]+\.py", tool):
        raise VastComfyError(code=E_INTERNAL, message=f"Invalid remote tool name: {tool!r}")
