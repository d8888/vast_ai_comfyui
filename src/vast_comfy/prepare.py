"""Prepare flow: install manifest building + remote end-to-end preparation.

The prepare flow (SPEC.md 16.1):

    acquire session -> deploy package -> install models/plugins ->
    service restart if needed (never over a busy queue) -> health/GPU checks ->
    full smoke suite -> remote report -> optional release on success

Any failure stops the remaining phases, keeps the lock, keeps the instance and
reports exactly where it stopped. Nothing is cleaned up implicitly.
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import jobs, locks, remote
from .config import Config, _UniqueKeySafeLoader
from .errors import ConfigError, E_JOB_FAILED, E_SMOKE_FAILED, VastComfyError
from .paths import find_repo_root
from .ssh import SSHRunner
from .safety import validate_identifier

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
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except OSError as exc:
        raise ConfigError(f"Cannot read model manifest {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        raise ConfigError("Invalid YAML in model manifest; source values are omitted.",
                          details={"line": mark.line + 1 if mark else None}) from None
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        raise ConfigError(f"{path} must contain schema_version: 1")
    if set(data) != {"schema_version", "models", "plugins"}:
        raise ConfigError("model manifest has missing or unknown top-level fields")
    if not isinstance(data.get("models"), list) or not isinstance(data.get("plugins"), list):
        raise ConfigError(f"{path}: 'models' and 'plugins' must be lists")
    return data


def _validate_model_dest(dest: Any) -> str:
    if not isinstance(dest, str) or not dest:
        raise ConfigError("model dest must be a non-empty relative path")
    parts = dest.split("/")
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", dest) or len(parts) < 2
            or any(p in ("", ".", "..") or p.startswith(".") for p in parts)):
        raise ConfigError("unsafe model dest: only canonical relative components are allowed")
    if parts[0] not in _ALLOWED_MODEL_DIRS:
        raise ConfigError("model dest must start with an allowed directory")
    return "/".join(parts)


def validate_public_url(url: Any) -> str:
    """Only credential-free HTTPS sources; never echo a rejected URL."""
    if not isinstance(url, str) or any(c.isspace() for c in url) or "%" in url:
        raise ConfigError("source must be a public HTTPS URL without credentials or encoded parameters")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ConfigError("invalid public source URL") from None
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment or port not in (None, 443)
            or "\\" in url or not re.fullmatch(r"/[A-Za-z0-9_./-]+", parsed.path)
            or any(p in ("", ".", "..") for p in parsed.path[1:].split("/"))):
        raise ConfigError("source must be public HTTPS without userinfo, query, fragment or unsafe path")
    return url


def _entry_fields(entry: Any, allowed: set[str], kind: str) -> None:
    if not isinstance(entry, dict) or set(entry) - allowed:
        raise ConfigError(f"{kind} entry must be a mapping with only supported fields")
    if "purpose" in entry and not isinstance(entry["purpose"], str):
        raise ConfigError(f"{kind} purpose must be a string")


def _source_path(value: Any) -> bool:
    return (isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_./-]+", value))
            and all(p not in ("", ".", "..") for p in value.split("/")))


def build_install_manifest() -> dict:
    """Convert config/models.yaml into the remote install manifest."""
    doc = load_models_config()
    if (not isinstance(doc, dict) or type(doc.get("schema_version")) is not int
            or doc.get("schema_version") != 1 or set(doc) != {"schema_version", "models", "plugins"}
            or not isinstance(doc["models"], list) or not isinstance(doc["plugins"], list)):
        raise ConfigError("invalid model manifest schema")
    models = []
    ids, destinations = set(), set()
    for entry in doc["models"]:
        _entry_fields(entry, {"id", "purpose", "source", "dest", "sha256", "size_bytes"}, "model")
        m_id = entry.get("id")
        source = entry.get("source") or {}
        sha = str(entry.get("sha256", "")).lower()
        if not m_id or not isinstance(source, dict):
            raise ConfigError("model entry is missing id/source")
        validate_identifier(m_id, label="model id")
        if m_id in ids:
            raise ConfigError("duplicate model/plugin id")
        ids.add(m_id)
        if not _SHA256_RE.fullmatch(sha):
            raise ConfigError(f"model {m_id}: sha256 must be 64 hex characters")
        dest = _validate_model_dest(entry.get("dest"))
        if dest in destinations:
            raise ConfigError("duplicate model destination")
        destinations.add(dest)
        source_type = source.get("type")
        if source_type == "huggingface":
            repo, revision, file_path = source.get("repo"), source.get("revision"), source.get("path")
            if not repo or not revision or not file_path:
                raise ConfigError(f"model {m_id}: huggingface source requires repo/revision/path")
            if set(source) - {"type", "repo", "revision", "path", "auth_env"}:
                raise ConfigError(f"model {m_id}: unknown huggingface source field")
            if not (_source_path(repo) and len(repo.split("/")) == 2 and _source_path(file_path)):
                raise ConfigError(f"model {m_id}: unsafe huggingface repo/path")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ConfigError(f"model {m_id}: revision must be a fixed 40-character commit SHA")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{file_path}"
        elif source_type == "civitai":
            version_id = source.get("version_id")
            if not version_id:
                raise ConfigError(f"model {m_id}: civitai source requires version_id")
            if set(source) - {"type", "version_id", "auth_env"}:
                raise ConfigError(f"model {m_id}: unknown civitai source field")
            if type(version_id) is not int or version_id <= 0:
                raise ConfigError(f"model {m_id}: version_id must be a positive integer")
            url = f"https://civitai.com/api/download/models/{version_id}"
        else:
            raise ConfigError(f"model {m_id}: unsupported source type {source_type!r}")
        record = {"id": m_id, "url": validate_public_url(url), "dest": dest, "sha256": sha}
        if entry.get("size_bytes") is not None:
            if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
                raise ConfigError(f"model {m_id}: size_bytes must be a positive integer")
            record["size_bytes"] = entry["size_bytes"]
        if source.get("auth_env"):
            if not isinstance(source["auth_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source["auth_env"]):
                raise ConfigError(f"model {m_id}: auth_env must name an environment variable")
            record["auth_env"] = source["auth_env"]
        models.append(record)

    plugins = []
    for entry in doc["plugins"]:
        _entry_fields(entry, {"id", "purpose", "repo_url", "commit", "dest", "required_nodes"}, "plugin")
        p_id = entry.get("id")
        repo_url, commit, dest = entry.get("repo_url"), entry.get("commit"), entry.get("dest")
        if not p_id or not repo_url or not commit or not dest:
            raise ConfigError("plugin entry is missing required fields")
        validate_identifier(p_id, label="plugin id")
        if p_id in ids:
            raise ConfigError("duplicate model/plugin id")
        ids.add(p_id)
        if (not isinstance(dest, str) or not re.fullmatch(r"custom_nodes/[A-Za-z0-9][A-Za-z0-9_.\-]*", dest)
                or ".." in dest or dest in destinations):
            raise ConfigError(f"plugin {p_id}: unsafe dest")
        destinations.add(dest)
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ConfigError(f"plugin {p_id}: commit must be a fixed 40-character SHA")
        validate_public_url(repo_url)
        nodes = entry.get("required_nodes", [])
        if not isinstance(nodes, list) or any(not isinstance(n, str) or not n for n in nodes):
            raise ConfigError(f"plugin {p_id}: required_nodes must be a list of node names")
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
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except OSError as exc:
        raise ConfigError(f"Cannot read smoke cases {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ConfigError(f"{path} must contain schema_version: 1 and a 'cases' list")
    if type(data["schema_version"]) is not int or set(data) != {"schema_version", "cases"}:
        raise ConfigError("invalid smoke cases schema")
    ids = []
    for case in data["cases"]:
        if not isinstance(case, dict):
            raise ConfigError("smoke cases must be mappings")
        ids.append(case.get("id"))
        timeout = case.get("timeout_minutes")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or timeout <= 0 or not math.isfinite(timeout * 60)):
            raise ConfigError("case timeout_minutes must be finite and positive")
    if ids != ["base_t2i", "lllite_lineart", "lllite_inpaint", "turbo_t2i"]:
        raise ConfigError("the configured suite must contain the four required cases in order")
    return data


def smoke_request(cfg: Config, *, run_id=None, base_seed=None, cases=None):
    """Validate controller inputs before any lock, upload or paid request."""
    actual_run_id = validate_identifier(run_id if run_id is not None else new_run_id(), label="run id")
    seed = base_seed if base_seed is not None else cfg.require("smoke.seed")
    if (type(seed) not in (int, float) or not math.isfinite(seed)
            or not 0 <= seed < 2**64 or int(seed) != seed):
        raise ConfigError("smoke seed must be a uint64")
    doc = load_smoke_cases()
    if cases is not None:
        if not isinstance(cases, str) or not cases.strip():
            raise ConfigError("cases must be a non-empty comma-separated selection")
        requested = [item.strip() for item in cases.split(",")]
        known = {item["id"] for item in doc["cases"]}
        if len(set(requested)) != len(requested) or any(item not in known for item in requested):
            raise ConfigError("case selection contains duplicates or unknown cases")
    return actual_run_id, int(seed), doc


def b64_json(payload: Any) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def assert_no_remote_credentials(manifest: dict) -> None:
    """v1 default-deny: no remote credentials are relayed to the instance.

    Any manifest entry that requires an ``auth_env`` fails *before* a paid
    instance is created or a long install starts (SPEC.md 7.3 / 12.2).
    """
    needs = [m for m in manifest.get("models", []) if m.get("auth_env")]
    for entry in manifest.get("models", []):
        validate_public_url(entry.get("url"))
    for entry in manifest.get("plugins", []):
        validate_public_url(entry.get("repo_url"))
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

    actual_run_id, seed, cases_doc = smoke_request(cfg, run_id=run_id, base_seed=base_seed, cases=cases)

    manifest = build_install_manifest()
    assert_no_remote_credentials(manifest)
    models = build_models_payload(manifest)
    lock = locks.acquire(runner, cfg, instance_id, session_id=session_id)
    sid = lock["session_id"]
    deploy = remote.deploy_package(runner, cfg, session_id=sid)

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
