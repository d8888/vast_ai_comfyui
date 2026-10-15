"""Python API facade (SPEC.md 3.2): the same logic the CLI calls.

Every function raises :class:`vast_comfy.errors.VastComfyError` (or a subclass)
on failure; results are plain JSON-safe dicts. Nothing here silently relaxes a
hard condition, pays for resources, or destroys data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import comfy as _comfy
from . import download as _download
from . import jobs as _jobs
from . import locks as _locks
from . import prepare as _prepare
from . import remote as _remote
from . import smoke as _smoke
from .config import Config, load_config, require_env
from .create_flow import create_and_prepare as _create_and_prepare
from .create_flow import quote_for_alias
from .errors import VastComfyError
from .gpu import AliasSpec, load_mapping
from .instances import live_instance, load_runner, runner_for
from .ssh import SSHRunner
from .state import StateStore
from .vast.client import VastClient
from .vast.rental import create_instance, verify_creation, verify_offer, wait_for_running

__all__ = [
    "Config",
    "VastComfyError",
    "make_client",
    "make_state",
    "quote_for_alias",
    "rent_and_prepare",
    "rent_create",
    "rent_verify",
    "prepare_existing",
    "smoke_run",
    "show_report",
    "session_acquire",
    "session_status",
    "session_release",
    "session_recover",
    "comfy_status",
    "comfy_health",
    "comfy_restart",
    "gpu_check",
    "deploy_package",
    "download_report",
    "download_smoke_images",
    "download_outputs",
    "wait_instance_ready",
]


def make_client(cfg: Config) -> VastClient:
    return VastClient(
        cfg.require("vast.api_base_url"),
        require_env(cfg.require("vast.api_key_env")),
        timeout_seconds=float(cfg.require("vast.request_timeout_seconds")),
        transient_retries=int(cfg.require("vast.transient_retries")),
        retry_backoff_seconds=list(cfg.require("vast.retry_backoff_seconds")),
    )


def make_state(cfg: Config) -> StateStore:
    return StateStore(cfg.resolved_paths()["state_dir"])


# ---------------------------------------------------------------------------
# Offers / rent
# ---------------------------------------------------------------------------


def quote_for_alias_api(cfg: Config, alias: str, *, client: VastClient | None = None, top: int = 10) -> dict:
    return quote_for_alias(client or make_client(cfg), cfg, alias, top=top)


def rent_create(
    cfg: Config,
    *,
    alias: str,
    offer_id: int,
    label: str,
    confirm_paid: bool,
    wait: bool = True,
    client: VastClient | None = None,
) -> dict:
    client = client or make_client(cfg)
    spec: AliasSpec = load_mapping().resolve(alias)
    fresh = verify_offer(client, cfg, spec, offer_id)
    created = create_instance(client, cfg, fresh, label, paid_confirmed=confirm_paid)
    if not wait:
        return created
    instance = wait_for_running(client, created["instance_id"], cfg)
    runner = runner_for(instance, cfg)
    ssh = runner.check_login()
    return {**created, "ssh": ssh}


def rent_verify(cfg: Config, *, label: str, since_epoch: float, client: VastClient | None = None) -> list[dict]:
    return verify_creation(client or make_client(cfg), label, since_epoch)


def rent_and_prepare(
    cfg: Config,
    *,
    alias: str,
    label: str | None,
    confirm_paid: bool,
    release_on_success: bool | None = None,
    cases: str | None = None,
    run_id: str | None = None,
    client: VastClient | None = None,
) -> dict:
    return _create_and_prepare(
        client or make_client(cfg),
        cfg,
        alias=alias,
        label=label,
        confirm_paid=confirm_paid,
        release_on_success=release_on_success,
        cases=cases,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Prepare / smoke / service
# ---------------------------------------------------------------------------


def _runner(cfg: Config, instance_id: int | str, client: VastClient | None = None) -> tuple[dict, SSHRunner]:
    return load_runner(client or make_client(cfg), cfg, instance_id)


def deploy_package(cfg: Config, instance_id: int | str, *, session_id: str, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _remote.deploy_package(runner, cfg, session_id=session_id)


def prepare_existing(
    cfg: Config,
    instance_id: int | str,
    *,
    session_id: str | None = None,
    release_on_success: bool | None = None,
    cases: str | None = None,
    run_id: str | None = None,
    verify_model_shas: bool = False,
    client: VastClient | None = None,
) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _prepare.prepare_existing(
        runner,
        cfg,
        instance_id=instance_id,
        session_id=session_id,
        release_on_success=release_on_success,
        cases=cases,
        run_id=run_id,
        verify_model_shas=verify_model_shas,
    )


def smoke_run(
    cfg: Config,
    instance_id: int | str,
    session_id: str,
    *,
    cases: str | None = None,
    run_id: str | None = None,
    client: VastClient | None = None,
) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _smoke.run_smoke(runner, cfg, session_id=session_id, instance_id=instance_id, cases=cases, run_id=run_id)


def show_report(cfg: Config, instance_id: int | str, run_id: str, *, session_id: str, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    with _locks.operation(runner, cfg, session_id, label="report"):
        return _smoke.show_report(runner, cfg, run_id)


def session_acquire(cfg: Config, instance_id: int | str, *, session_id: str | None = None, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _locks.acquire(runner, cfg, instance_id, session_id=session_id)


def session_status(cfg: Config, instance_id: int | str, *, session_id: str | None = None, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _locks.status(runner, cfg, session_id=session_id)


def session_release(cfg: Config, instance_id: int | str, session_id: str, *, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _locks.release(runner, cfg, session_id)


def session_recover(cfg: Config, instance_id: int | str, *, session_id: str | None = None, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _locks.recover(runner, cfg, instance_id, session_id=session_id)


def comfy_status(cfg: Config, instance_id: int | str, session_id: str, *, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _comfy.status(runner, cfg, session_id)


def comfy_health(cfg: Config, instance_id: int | str, session_id: str, *, gpu: bool = False, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _comfy.health(runner, cfg, session_id, gpu=gpu)


def comfy_restart(cfg: Config, instance_id: int | str, session_id: str, *, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _comfy.restart(runner, cfg, session_id)


def gpu_check(cfg: Config, instance_id: int | str, session_id: str, *, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    return _comfy.gpu_check(runner, cfg, session_id)


def wait_instance_ready(cfg: Config, instance_id: int | str, *, client: VastClient | None = None) -> dict:
    client = client or make_client(cfg)
    instance = wait_for_running(client, instance_id, cfg)
    runner = runner_for(instance, cfg)
    return {**runner.check_login(), "instance_id": instance.get("id")}


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def download_report(cfg: Config, instance_id: int | str, run_id: str, dest: Path | str, *, session_id: str, overwrite: bool = False, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    with _locks.operation(runner, cfg, session_id, label="download"):
        return _download.download_report(runner, cfg, run_id=run_id, dest_dir=dest, overwrite=overwrite)


def download_smoke_images(cfg: Config, instance_id: int | str, run_id: str, dest: Path | str, *, session_id: str, overwrite: bool = False, client: VastClient | None = None) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    with _locks.operation(runner, cfg, session_id, label="download"):
        return _download.download_smoke_images(runner, cfg, run_id=run_id, dest_dir=dest, overwrite=overwrite)


def download_outputs(
    cfg: Config,
    instance_id: int | str,
    dest: Path | str,
    *,
    session_id: str,
    pattern: str = "*.png",
    limit: int = 100,
    overwrite: bool = False,
    client: VastClient | None = None,
) -> dict:
    _, runner = _runner(cfg, instance_id, client)
    with _locks.operation(runner, cfg, session_id, label="download"):
        return _download.download_outputs(runner, cfg, dest_dir=dest, pattern=pattern, limit=limit, overwrite=overwrite)


def instance_show(cfg: Config, instance_id: int | str, *, client: VastClient | None = None) -> dict:
    return live_instance(client or make_client(cfg), instance_id)
