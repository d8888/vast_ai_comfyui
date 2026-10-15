"""Prepare flow: install manifest building + remote end-to-end preparation.

The prepare flow (SPEC.md 16.1):

    deploy package -> acquire session -> install models/plugins ->
    service restart if needed (never over a busy queue) -> health/GPU checks ->
    full smoke suite -> remote report -> optional release on success

Any failure stops the remaining phases, keeps the lock, keeps the instance and
reports exactly where it stopped. Nothing is cleaned up implicitly.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import jobs, locks, remote
from .config import Config
from .errors import ConfigError, E_JOB_FAILED, E_SMOKE_FAILED, VastComfyError
from .paths import find_repo_root
from .ssh import SSHRunner

_ALLOWED_MODEL_DIRS = {
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "vae",
    "controlnet",
    "loras",
    "embeddings",
    "clip_vision",
    "upscale_models",
    "model_patches",
    "ipadapter",
    "style_models",
    "audio_encoders",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def new_run_id() -> str:
    return "vc-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def _models_config_path() -> Path:
    return find_repo_root() / "config" / "models.yaml"


def load_models_config() -> dict:
    path = _models_config_path()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read model manifest {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ConfigError(f"{path} must contain schema_version: 1")
    if not isinstance(data.get("models"), list) or not isinstance(data.get("plugins"), list):
        raise ConfigError(f"{path}: 'models' and 'plugins' must be lists")
    return data


def _validate_model_dest(dest: Any) -> str:
    if not isinstance(dest, str) or not dest:
        raise ConfigError(f"model dest must be a non-empty relative path, got {dest!r}")
    parts = [p for p in dest.split("/") if p]
    if dest.startswith("/") or "\\" in dest or ".." in parts or len(parts) < 2:
        raise ConfigError(f"unsafe model dest: {dest!r}")
    if parts[0] not in _ALLOWED_MODEL_DIRS:
        raise ConfigError(f"model dest must start with an allowed directory, got {parts[0]!r}")
    return "/".join(parts)


def build_install_manifest() -> dict:
    """Convert config/models.yaml into the remote install manifest."""
    doc = load_models_config()
    models = []
    for entry in doc["models"]:
        m_id = entry.get("id")
        source = entry.get("source") or {}
        sha = str(entry.get("sha256", "")).lower()
        if not m_id or not isinstance(source, dict):
            raise ConfigError(f"model entry is missing id/source: {entry!r}")
        if not _SHA256_RE.match(sha):
            raise ConfigError(f"model {m_id}: sha256 must be 64 hex characters")
        dest = _validate_model_dest(entry.get("dest"))
        source_type = source.get("type")
        if source_type == "huggingface":
            repo, revision, file_path = source.get("repo"), source.get("revision"), source.get("path")
            if not repo or not revision or not file_path:
                raise ConfigError(f"model {m_id}: huggingface source requires repo/revision/path")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{file_path}"
        elif source_type == "civitai":
            version_id = source.get("version_id")
            if not version_id:
                raise ConfigError(f"model {m_id}: civitai source requires version_id")
            url = f"https://civitai.com/api/download/models/{version_id}"
        else:
            raise ConfigError(f"model {m_id}: unsupported source type {source_type!r}")
        record = {"id": m_id, "url": url, "dest": dest, "sha256": sha}
        if entry.get("size_bytes") is not None:
            record["size_bytes"] = int(entry["size_bytes"])
        if source.get("auth_env"):
            record["auth_env"] = str(source["auth_env"])
        models.append(record)

    plugins = []
    for entry in doc["plugins"]:
        p_id = entry.get("id")
        repo_url, commit, dest = entry.get("repo_url"), entry.get("commit"), entry.get("dest")
        if not p_id or not repo_url or not commit or not dest:
            raise ConfigError(f"plugin entry is missing required fields: {entry!r}")
        if not re.fullmatch(r"custom_nodes/[A-Za-z0-9_.\-]+", str(dest)):
            raise ConfigError(f"plugin {p_id}: unsafe dest {dest!r}")
        plugins.append(
            {
                "id": p_id,
                "repo_url": str(repo_url),
                "commit": str(commit),
                "dest": str(dest),
                "required_nodes": list(entry.get("required_nodes") or []),
            }
        )
    return {"models": models, "plugins": plugins}


def build_models_payload(manifest: dict) -> list[dict]:
    return [
        {k: m[k] for k in ("id", "dest", "sha256") if k in m} | ({"size_bytes": m["size_bytes"]} if "size_bytes" in m else {})
        for m in manifest.get("models", [])
    ]


def load_smoke_cases() -> dict:
    path = find_repo_root() / "assets" / "smoke" / "smoke_cases.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read smoke cases {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ConfigError(f"{path} must contain schema_version: 1 and a 'cases' list")
    return data


def b64_json(payload: Any) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def assert_no_remote_credentials(manifest: dict) -> None:
    """v1 default-deny: no remote credentials are relayed to the instance.

    Any manifest entry that requires an ``auth_env`` fails *before* a paid
    instance is created or a long install starts (SPEC.md 7.3 / 12.2).
    """
    needs = [m for m in manifest.get("models", []) if m.get("auth_env")]
    if needs:
        from .errors import E_APPROVAL_REQUIRED

        raise VastComfyError(
            code=E_APPROVAL_REQUIRED,
            message=(
                "config/models.yaml contains files that would require a credential on the instance; "
                "vast-comfy v1 does not relay credentials to instances."
            ),
            stage="install",
            details={
                "entries_requiring_auth_env": [
                    {"id": m.get("id"), "auth_env": m.get("auth_env")} for m in needs
                ]
            },
            hints=[
                "Use a public source instead, or handle the private file yourself on the instance. "
                "No credential was sent and nothing was rented.",
            ],
        )


# ---------------------------------------------------------------------------
# Prepare flow
# ---------------------------------------------------------------------------


def prepare_existing(
    runner: SSHRunner,
    cfg: Config,
    *,
    instance_id: int | str,
    session_id: str | None = None,
    release_on_success: bool | None = None,
    cases: str | None = None,
    run_id: str | None = None,
    base_seed: int | None = None,
    verify_model_shas: bool = False,
) -> dict:
    if release_on_success is None:
        release_on_success = bool(cfg.get("session.release_on_success", True))

    manifest = build_install_manifest()
    assert_no_remote_credentials(manifest)
    models = build_models_payload(manifest)
    cases_doc = load_smoke_cases()
    lock = locks.acquire(runner, cfg, instance_id, session_id=session_id)
    sid = lock["session_id"]
    deploy = remote.deploy_package(runner, cfg, session_id=sid)
    actual_run_id = run_id or new_run_id()
    seed = int(base_seed if base_seed is not None else cfg.get("smoke.seed", 12345))

    timeout_seconds = (
        float(cfg.require("jobs.install_timeout_minutes")) * 60
        + sum(float(c.get("timeout_minutes", 15)) for c in cases_doc["cases"]) * 60
        + 900  # preflight + restarts + report
    )
    argv = [
        "run",
        "--session-id",
        sid,
        "--instance-id",
        str(instance_id),
        "--image-ref",
        f"{cfg.require('image.repository')}:{cfg.require('image.tag')}",
        "--install-manifest-b64",
        b64_json(manifest),
        "--models-b64",
        b64_json(models),
        "--cases-b64",
        b64_json(cases_doc),
        "--workflows-dir",
        f"{remote.current_link(cfg)}/workflows/api",
        "--assets-dir",
        f"{remote.current_link(cfg)}/assets/smoke",
        "--run-id",
        actual_run_id,
        "--base-seed",
        str(seed),
        "--comfy-url", f"http://localhost:{cfg.require('image.internal_port')}",
        "--service", str(cfg.require("image.supervisor_service")),
        "--wait-seconds", str(cfg.require("comfy.restart_wait_seconds")),
    ]
    if cases:
        argv += ["--cases", cases]
    if verify_model_shas:
        argv.append("--verify-model-shas")

    job = jobs.start(
        runner,
        cfg,
        sid,
        "prepare",
        jobs.prepare_command(cfg, "vc_prepare.py", argv),
        timeout_seconds=timeout_seconds,
    )
    job_id = job["job_id"]

    final: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    summary_path = f"{remote.runtime_root(cfg)}/runs/{actual_run_id}/prepare_summary.json"

    try:
        final = jobs.wait(runner, cfg, job_id, timeout_seconds=timeout_seconds + 120)
    except VastComfyError as exc:
        raise VastComfyError(
            code=exc.code,
            message=exc.message,
            stage="prepare",
            details={
                **exc.details,
                "session_id": sid,
                "run_id": actual_run_id,
                "lock_state": "still held by this session; NOT released automatically",
                "instance_billing": "the instance continues to run",
            },
            hints=list(exc.hints),
        ) from exc

    if final is None or final.get("state") != "succeeded":
        try:
            summary = remote.read_remote_json(runner, cfg, summary_path)
        except VastComfyError:
            summary = None
        raise VastComfyError(
            code=E_SMOKE_FAILED if (summary or {}).get("verdict") == "FAILED" else E_JOB_FAILED,
            message=(
                "The prepare flow did not complete successfully "
                f"(job state: {(final or {}).get('state')!r})"
            ),
            stage="prepare",
            details={
                "job_id": job_id,
                "run_id": actual_run_id,
                "summary_path": summary_path,
                "summary": summary,
                "session_id": sid,
                "lock_state": "still held by this session unless explicitly released",
                "instance_billing": "the instance continues to run",
                "log_hint": f"vast-comfy jobs logs --job-id {job_id}",
            },
            hints=[
                "No cleanup, no unlock, no instance stop happened automatically.",
                "Review the summary/log, then decide explicitly.",
            ],
        )

    summary = remote.read_remote_json(runner, cfg, summary_path)

    release_result: dict | None = None
    release_error: dict | None = None
    if release_on_success:
        try:
            release_result = locks.release(runner, cfg, sid)
        except VastComfyError as exc:
            release_error = exc.to_dict()

    return {
        "ok": True,
        "verdict": summary.get("verdict"),
        "instance_id": instance_id,
        "session_id": sid,
        "job_id": job_id,
        "run_id": actual_run_id,
        "deploy": deploy,
        "summary_path": summary_path,
        "summary": summary,
        "release": release_result,
        "release_error": release_error,
    }
