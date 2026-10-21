"""Smoke suite orchestration from the control side (SPEC.md section 13)."""

from __future__ import annotations

from typing import Any
from pathlib import PurePosixPath

from . import jobs, remote
from .config import Config
from .errors import E_JOB_FAILED, E_SMOKE_FAILED, VastComfyError
from .prepare import b64_json, build_models_payload, build_install_manifest, smoke_request
from .paths import resolve_profile
from .safety import validate_identifier
from .ssh import SSHRunner


def run_smoke(
    runner: SSHRunner,
    cfg: Config,
    *,
    session_id: str,
    instance_id: int | str,
    run_id: str | None = None,
    cases: str | None = None,
    base_seed: int | None = None,
    models: bool = True,
    verify_model_shas: bool = False,
    fixed_seed: bool = False,
    on_poll: Any = None,
) -> dict:
    """Run the smoke suite as a remote background job and return its summary."""
    profile = resolve_profile()
    actual_run_id, seed, cases_doc = smoke_request(cfg, run_id=run_id, base_seed=base_seed, cases=cases)
    models_payload = build_models_payload(build_install_manifest()) if models else []

    argv = [
        "run",
        "--session-id",
        session_id,
        "--cases-b64",
        b64_json(cases_doc),
        "--workflows-dir",
        f"{remote.current_link(cfg)}/workflows/{profile}/api",
        "--assets-dir",
        f"{remote.current_link(cfg)}/assets/smoke/{profile}",
        "--run-id",
        actual_run_id,
        "--base-seed",
        str(seed),
        "--instance-id",
        str(instance_id),
        "--image-ref",
        f"{cfg.require('image.repository')}:{cfg.require('image.tag')}",
        "--comfy-url", f"http://localhost:{cfg.require('image.internal_port')}",
    ]
    if cases:
        argv += ["--cases", cases]
    if models_payload:
        argv += ["--models-b64", b64_json(models_payload)]
    if verify_model_shas:
        argv.append("--verify-model-shas")
    if fixed_seed:
        argv.append("--fixed-seed")

    requested = [c for c in cases_doc["cases"] if not cases or c["id"] in {x.strip() for x in cases.split(",")}]
    timeout_seconds = sum(float(c.get("timeout_minutes", 15)) for c in requested) * 60 + 900

    # SPEC 13.2: an actual SSH command must succeed as part of the verification.
    ssh_check = runner.check_login()

    job = jobs.start(
        runner,
        cfg,
        session_id,
        "smoke",
        jobs.prepare_command(cfg, "vc_smoke.py", argv),
        timeout_seconds=timeout_seconds,
    )
    job_id = job["job_id"]
    summary_path = f"{remote.runtime_root(cfg)}/runs/{actual_run_id}/summary.json"

    try:
        final = jobs.wait(runner, cfg, job_id, timeout_seconds=timeout_seconds + 120, on_poll=on_poll)
    except VastComfyError as exc:
        raise VastComfyError(
            code=exc.code,
            message=exc.message,
            stage="smoke",
            details={**exc.details, "job_id": job_id, "run_id": actual_run_id, "summary_path": summary_path},
        ) from exc

    summary: dict | None = None
    try:
        summary = remote.read_remote_json(runner, cfg, summary_path)
    except VastComfyError:
        summary = None

    if final.get("state") != "succeeded":
        raise VastComfyError(
            code=E_SMOKE_FAILED if (summary or {}).get("verdict") == "FAILED" else E_JOB_FAILED,
            message=f"Smoke suite did not succeed (job state: {final.get('state')!r})",
            stage="smoke",
            details={
                "job_id": job_id,
                "run_id": actual_run_id,
                "summary": summary,
                "log_hint": f"vast-comfy jobs logs --job-id {job_id}",
            },
        )
    if summary is None:
        raise VastComfyError(
            code=E_JOB_FAILED,
            message="Smoke job succeeded but its summary file could not be read",
            stage="smoke",
            details={"job_id": job_id, "run_id": actual_run_id, "summary_path": summary_path},
        )
    summary["job_id"] = job_id
    summary["controller_ssh_check"] = ssh_check
    return summary


def show_report(
    runner: SSHRunner, cfg: Config, run_id: str, *, section: str = "all"
) -> dict:
    """Read the remote report (stays on the instance; no download happens here)."""
    from .download import _resolve_remote
    run_id = validate_identifier(run_id, label="run id")
    base = f"{remote.runtime_root(cfg)}/runs/{run_id}"
    path = _resolve_remote(runner, remote.runtime_root(cfg), f"{base}/report.json",
                           subdir=PurePosixPath("runs", run_id))
    report = remote.read_remote_json(runner, cfg, path)
    payload = {"ok": True, "run_id": run_id, "report_path": f"{base}/report.json", "report": report}
    if section in ("all", "summary"):
        payload["verdict"] = report.get("verdict")
        payload["cases"] = [
            {"id": c.get("id"), "status": c.get("status"), "seed": c.get("seed")}
            for c in report.get("cases", [])
        ]
    return payload
