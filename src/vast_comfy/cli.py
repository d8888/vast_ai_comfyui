"""vast-comfy command-line interface.

Every command returns a JSON-safe result. ``--json`` writes exactly one JSON
object to stdout (logs go to stderr); failures exit non-zero with stable codes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import Config, apply_env_file, load_config, require_env, set_dotted
from .create_flow import create_and_prepare, default_label, quote_for_alias
from .errors import (
    E_KEY_EXISTS,
    E_USAGE,
    UsageError,
    VastComfyError,
    exit_code_for,
)
from .gpu import add_custom_alias, load_mapping, update_mapping_file
from .logging_setup import get_logger, setup_logging
from .prepare import (
    build_install_manifest,
    load_smoke_cases,
    new_run_id,
)
from .state import StateStore
from .safety import validate_identifier
from .vast.client import VastClient
from .vast.rental import (create_instance, image_ref, verify_creation, verify_offer,
                          wait_for_running, wait_created, created_resource)

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        set_overrides = {}
        for item in getattr(args, "set_overrides", []) or []:
            if "=" not in item:
                raise UsageError(f"--set expects PATH=VALUE, got {item!r}")
            path, _, raw = item.partition("=")
            try:
                value = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise UsageError(f"--set value is not valid YAML: {item!r}: {exc}") from exc
            set_overrides[path.strip()] = value
        # --profile is sugar for --set profile=NAME; the resolved config stays the
        # single source of truth, and explicit --set profile=X still wins.
        explicit_profile = getattr(args, "profile", None)
        if explicit_profile is not None and "profile" not in set_overrides:
            set_overrides["profile"] = explicit_profile
        self.cfg: Config = load_config(getattr(args, "config", None), set_overrides)
        # Publish the resolved profile for helpers that run outside a Ctx
        # (prepare/smoke/remote all resolve paths from it).
        os.environ["VAST_COMFY_PROFILE"] = str(self.cfg.require("profile"))
        self._client: VastClient | None = None
        self._state: StateStore | None = None

    @property
    def client(self) -> VastClient:
        if self._client is None:
            api_key = require_env(self.cfg.require("vast.api_key_env"))
            self._client = VastClient(
                self.cfg.require("vast.api_base_url"),
                api_key,
                timeout_seconds=float(self.cfg.require("vast.request_timeout_seconds")),
                transient_retries=int(self.cfg.require("vast.transient_retries")),
                retry_backoff_seconds=list(self.cfg.require("vast.retry_backoff_seconds")),
            )
        return self._client

    @property
    def state(self) -> StateStore:
        if self._state is None:
            self._state = StateStore(self.cfg.resolved_paths()["state_dir"])
        return self._state

    def instance_and_runner(self, instance_id: int | str, *, require_running: bool = True):
        from .instances import load_runner

        return load_runner(self.client, self.cfg, instance_id, require_running=require_running)

    def require_session(self, explicit: str | None, instance_id: int | str) -> str:
        if explicit:
            return explicit
        for record in self.state.list_sessions_for_instance(instance_id):
            if record.get("state") == "active" and record.get("session_id"):
                return str(record["session_id"])
        raise UsageError(
            f"--session-id is required (no active session is recorded locally for instance {instance_id})"
        )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_config_show(ctx: Ctx, args) -> dict:
    return {"config": ctx.cfg.as_dict()}


def cmd_config_validate(ctx: Ctx, args) -> dict:
    mapping = load_mapping()
    manifest = build_install_manifest()
    cases = load_smoke_cases()
    return {
        "valid": True,
        "checks": {
            "gpu_aliases": len(mapping.aliases()),
            "official_names": len(mapping.official_names),
            "models": len(manifest["models"]),
            "plugins": len(manifest["plugins"]),
            "smoke_cases": [c["id"] for c in cases["cases"]],
        },
    }


def cmd_gpu_aliases(ctx: Ctx, args) -> dict:
    mapping = load_mapping()
    items = []
    for alias, spec in sorted(mapping.aliases().items()):
        if args.custom_only and not spec.custom:
            continue
        items.append(
            {
                "alias": alias,
                "gpu_name": spec.gpu_name,
                "gpu_ram_mb_min": spec.gpu_ram_mb_min,
                "gpu_ram_mb_max": spec.gpu_ram_mb_max,
                "custom": spec.custom,
            }
        )
    return {
        "count": len(items),
        "fetched_at_utc": mapping.fetched_at_utc,
        "aliases": items,
    }


def cmd_gpu_resolve(ctx: Ctx, args) -> dict:
    spec = load_mapping().resolve(args.alias)
    return {
        "alias": spec.alias,
        "gpu_name": spec.gpu_name,
        "gpu_ram_mb_min": spec.gpu_ram_mb_min,
        "gpu_ram_mb_max": spec.gpu_ram_mb_max,
        "custom": spec.custom,
    }


def cmd_gpu_update(ctx: Ctx, args) -> dict:
    return update_mapping_file()


def cmd_gpu_add_alias(ctx: Ctx, args) -> dict:
    return add_custom_alias(
        None,
        args.alias,
        args.gpu_name,
        gpu_ram_mb_min=int(args.vram_gb_min * 1024) if args.vram_gb_min else None,
        gpu_ram_mb_max=int(args.vram_gb_max * 1024) if args.vram_gb_max else None,
        notes=args.notes,
    )


def cmd_offers_search(ctx: Ctx, args) -> dict:
    result = quote_for_alias(ctx.client, ctx.cfg, args.alias, top=args.top)
    if args.only_best:
        result.pop("offers", None)
        result["command_note"] = "cheapest qualifying offer only; nothing was rented"
    return result


def cmd_rent_create(ctx: Ctx, args) -> dict:
    spec = load_mapping().resolve(args.alias)
    fresh = verify_offer(ctx.client, ctx.cfg, spec, args.offer_id)
    created = create_instance(ctx.client, ctx.cfg, fresh, args.label, paid_confirmed=args.confirm_paid)
    instance_id = created["instance_id"]
    with created_resource(created):
        ctx.state.record_instance(
            instance_id,
            label=args.label,
            alias=args.alias,
            offer_id=args.offer_id,
            image=image_ref(ctx.cfg),
            dph_total=fresh.get("dph_total"),
            created_at=time.time(),
            state="created",
        )
    if not args.wait:
        created["note"] = "Instance created. Use `rent verify` and `instance wait-ready` next."
        return created
    instance, _, ssh = wait_created(ctx.client, ctx.cfg, created)
    with created_resource(created):
        ctx.state.record_instance(instance_id, state="running")
    return {
        "instance_id": instance_id,
        "label": args.label,
        "gpu_name": instance.get("gpu_name"),
        "actual_status": instance.get("actual_status"),
        "ssh": {
            **ssh,
            "user": ctx.cfg.require("ssh.user"),
            "key_path": str(ctx.cfg.resolved_paths()["key_path"]) if ctx.cfg.resolved_paths()["key_path"] else None,
        },
        "next_steps": [
            "vast-comfy flow prepare --instance <ID>   # deploy models + run the smoke suite",
            "vast-comfy ssh tunnel --instance <ID> --start   # open ComfyUI locally",
        ],
        "billing_note": "The instance is running and billing; stopping/destroying is always an explicit call.",
    }


def cmd_rent_verify(ctx: Ctx, args) -> dict:
    since = args.since if args.since else time.time() - 3600
    found = verify_creation(ctx.client, args.label, float(since))
    payload = {"label": args.label, "found": found, "count": len(found)}
    if not found:
        payload["message"] = (
            "No instance with this label was found. The create outcome may have been 'not created'; "
            "nothing is resent automatically."
        )
    return payload


def cmd_instance_list(ctx: Ctx, args) -> dict:
    rows = ctx.client.list_instances()
    keys = (
        "id",
        "label",
        "actual_status",
        "cur_state",
        "gpu_name",
        "num_gpus",
        "dph_total",
        "disk_space",
        "image_uuid",
        "ssh_host",
        "ssh_port",
        "machine_id",
        "start_date",
    )
    return {"count": len(rows), "instances": [{k: row.get(k) for k in keys} for row in rows]}


def cmd_instance_show(ctx: Ctx, args) -> dict:
    from .instances import live_instance

    instance = live_instance(ctx.client, args.instance)
    payload = {
        "instance_id": instance.get("id"),
        "label": instance.get("label"),
        "actual_status": instance.get("actual_status"),
        "cur_state": instance.get("cur_state"),
        "status_msg": instance.get("status_msg"),
        "gpu_name": instance.get("gpu_name"),
        "num_gpus": instance.get("num_gpus"),
        "gpu_ram_mb": instance.get("gpu_ram"),
        "cpu_cores_effective": instance.get("cpu_cores_effective"),
        "cpu_ram_mb": instance.get("cpu_ram"),
        "disk_space": instance.get("disk_space"),
        "dph_total": instance.get("dph_total"),
        "image_uuid": instance.get("image_uuid"),
        "ssh_host": instance.get("ssh_host"),
        "ssh_port": instance.get("ssh_port"),
        "machine_id": instance.get("machine_id"),
        "start_date": instance.get("start_date"),
        "ports": instance.get("ports"),
        "public_ipaddr": instance.get("public_ipaddr"),
    }
    return payload


def _portal_info(instance: dict, cfg: Config) -> dict:
    portal_port = int(cfg.require("image.portal_external_port"))
    ports = instance.get("ports") or {}
    mapping = ports.get(f"{portal_port}/tcp")
    ip = instance.get("public_ipaddr")
    url = None
    if mapping and ip:
        try:
            url = f"http://{ip}:{mapping[0]['HostPort']}/"
        except (KeyError, IndexError, TypeError):
            url = None
    return {
        "open_button": "Vast.ai console -> instance card -> Open (sets the auth cookie automatically)",
        "portal_url_if_mapped": url,
        "token_note": (
            "The Portal requires OPEN_BUTTON_TOKEN (auto-generated). Prefer the console Open button "
            "or the local SSH tunnel; tokens are treated as secrets and never logged."
        ),
        "console_instances_url": "https://cloud.vast.ai/instances/",
    }


def cmd_instance_wait_ready(ctx: Ctx, args) -> dict:
    instance = wait_for_running(ctx.client, args.instance, ctx.cfg)
    from .instances import runner_for
    from .ssh import wait_for_ssh

    runner = runner_for(instance, ctx.cfg)
    ssh = wait_for_ssh(runner, ctx.cfg)
    return {
        "instance_id": instance.get("id"),
        "actual_status": instance.get("actual_status"),
        "ssh": ssh,
        "portal": _portal_info(instance, ctx.cfg),
        "note": "SSH readiness means an actual remote command succeeded, not just an open port.",
    }


def cmd_instance_stop(ctx: Ctx, args) -> dict:
    from .lifecycle import stop

    sid = ctx.require_session(args.session_id, args.instance)
    result = stop(ctx.client, ctx.cfg, args.instance, sid)
    ctx.state.record_instance(args.instance, state="stop_requested")
    return result


def cmd_instance_destroy(ctx: Ctx, args) -> dict:
    if not args.confirm_destroy:
        raise UsageError(
            "Destroying an instance requires the explicit --confirm-destroy flag "
            "(paid confirmation does not imply data-destruction confirmation)"
        )
    from .lifecycle import destroy

    sid = ctx.require_session(args.session_id, args.instance)
    result = destroy(ctx.client, ctx.cfg, args.instance, sid, confirm_destroy=True)
    ctx.state.record_instance(args.instance, state="destroy_requested")
    return result


def cmd_instance_logs(ctx: Ctx, args) -> dict:
    text = ctx.client.request_logs(args.instance, tail=args.tail)
    return {"instance_id": args.instance, "tail": args.tail, "log": text}


def cmd_ssh_info(ctx: Ctx, args) -> dict:
    instance, runner = ctx.instance_and_runner(args.instance)
    from .tunnel import tunnel_plan

    plan = tunnel_plan(ctx.cfg, runner, local_port=args.local_port)
    return {
        "instance_id": instance.get("id"),
        "strategy": "proxy_only",
        "endpoint": {
            "user": ctx.cfg.require("ssh.user"),
            "host": instance.get("ssh_host"),
            "port": instance.get("ssh_port"),
            "key_path": str(ctx.cfg.resolved_paths()["key_path"]) if ctx.cfg.resolved_paths()["key_path"] else None,
            "known_hosts": str(ctx.cfg.resolved_paths()["known_hosts_path"]),
        },
        "tunnel": plan,
        "portal": _portal_info(instance, ctx.cfg),
        "known_hosts_note": (
            "First connection records the host key (TOFU) in the dedicated known_hosts file; "
            "a later key change fails the connection and is never bypassed automatically."
        ),
    }


def cmd_ssh_test(ctx: Ctx, args) -> dict:
    instance, runner = ctx.instance_and_runner(args.instance)
    result = runner.check_login()
    return {"instance_id": instance.get("id"), **result}


def cmd_ssh_tunnel(ctx: Ctx, args) -> dict:
    from .tunnel import start_tunnel, stop_tunnel, tunnel_plan

    if args.stop:
        return stop_tunnel(args.instance, ctx.state)
    instance, runner = ctx.instance_and_runner(args.instance)
    if args.start:
        return start_tunnel(ctx.cfg, runner, instance_id=args.instance, state=ctx.state, local_port=args.local_port)
    return tunnel_plan(ctx.cfg, runner, local_port=args.local_port)


def cmd_keys_generate(ctx: Ctx, args) -> dict:
    key_path = Path(args.path).expanduser()
    pub_path = key_path.with_suffix(key_path.suffix + ".pub") if key_path.suffix else Path(str(key_path) + ".pub")
    if key_path.exists() or pub_path.exists():
        raise VastComfyError(
            code=E_KEY_EXISTS,
            message=f"Refusing to overwrite an existing key: {key_path}",
            stage="keys",
            hints=["Choose another path or keep the existing key."],
        )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", "vast-comfy"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VastComfyError(
            code=E_USAGE,
            message=f"ssh-keygen failed: {result.stderr.strip()[:300]}",
            stage="keys",
        )
    fingerprint = subprocess.run(
        ["ssh-keygen", "-lf", str(pub_path)], capture_output=True, text=True
    ).stdout.strip()
    return {
        "private_key_path": str(key_path),
        "public_key_path": str(pub_path),
        "fingerprint": fingerprint,
        "note": "Register the public key with Vast.ai BEFORE creating instances (keys register applies to new instances).",
    }


def cmd_keys_register(ctx: Ctx, args) -> dict:
    pub_path = Path(args.pubkey).expanduser()
    if not pub_path.is_file():
        raise VastComfyError(code=E_USAGE, message=f"Public key file not found: {pub_path}", stage="keys")
    public_key = pub_path.read_text(encoding="utf-8").strip()
    if not public_key.startswith(("ssh-", "ecdsa-")):
        raise VastComfyError(code=E_USAGE, message="File does not look like an OpenSSH public key", stage="keys")
    response = ctx.client.register_ssh_key(public_key)
    return {"registered": True, "public_key_path": str(pub_path), "api_response": response}


def cmd_keys_list(ctx: Ctx, args) -> dict:
    keys = ctx.client.list_ssh_keys()
    return {"count": len(keys), "ssh_keys": keys}


def cmd_deploy_package(ctx: Ctx, args) -> dict:
    from .remote import deploy_package

    _, runner = ctx.instance_and_runner(args.instance)
    sid = ctx.require_session(args.session_id, args.instance)
    return deploy_package(runner, ctx.cfg, session_id=sid)


def cmd_session_acquire(ctx: Ctx, args) -> dict:
    from .locks import acquire

    _, runner = ctx.instance_and_runner(args.instance)
    payload = acquire(runner, ctx.cfg, args.instance, session_id=args.session_id)
    ctx.state.record_session(
        payload["session_id"],
        instance_id=str(args.instance),
        state="active",
        acquired_at=time.time(),
        owner=payload.get("owner"),
    )
    return payload


def cmd_session_status(ctx: Ctx, args) -> dict:
    from .locks import status

    _, runner = ctx.instance_and_runner(args.instance)
    return status(runner, ctx.cfg, session_id=args.session_id)


def cmd_session_release(ctx: Ctx, args) -> dict:
    from .locks import release

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    payload = release(runner, ctx.cfg, session_id)
    ctx.state.record_session(session_id, state="released", released_at=time.time())
    return payload


def cmd_session_recover(ctx: Ctx, args) -> dict:
    from .locks import recover

    if not args.accept_stale:
        raise UsageError(
            "Recovery requires the explicit --accept-stale flag: it means you have personally confirmed "
            "the previous owner and its work are gone. vast-comfy never takes over a stale lock automatically."
        )
    _, runner = ctx.instance_and_runner(args.instance)
    payload = recover(runner, ctx.cfg, args.instance, session_id=args.session_id)
    ctx.state.record_session(
        payload["session_id"], instance_id=str(args.instance), state="active", recovered_at=time.time()
    )
    return payload


def cmd_session_resolve_inference(ctx: Ctx, args) -> dict:
    from .locks import resolve_inference
    from .safety import validate_identifier

    if not args.confirm_verified:
        raise UsageError(
            "Resolving an unresolved inference fence requires --confirm-verified: it declares that you "
            "personally checked the ComfyUI queue/history and the correlated work is no longer running."
        )
    intent_id = validate_identifier(args.intent_id, label="intent id")
    _, runner = ctx.instance_and_runner(args.instance)
    return resolve_inference(runner, ctx.cfg, intent_id, confirm_verified=True)


def cmd_jobs_start(ctx: Ctx, args) -> dict:
    from . import jobs

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    argv = args.job_command[1:] if args.job_command and args.job_command[0] == "--" else args.job_command
    if not argv:
        raise UsageError("jobs start requires the command after '--'")
    payload = jobs.start(
        runner,
        ctx.cfg,
        session_id,
        args.name,
        argv,
        timeout_seconds=float(args.timeout_minutes) * 60,
    )
    ctx.state.put(
        "jobs",
        payload["job_id"],
        {
            **payload,
            "instance_id": str(args.instance),
            "session_id": session_id,
            "name": args.name,
            "recorded_at": time.time(),
        },
    )
    return payload


def cmd_jobs_status(ctx: Ctx, args) -> dict:
    from . import jobs

    _, runner = ctx.instance_and_runner(args.instance)
    return jobs.status(runner, ctx.cfg, args.job_id)


def cmd_jobs_logs(ctx: Ctx, args) -> dict:
    from . import jobs

    _, runner = ctx.instance_and_runner(args.instance)
    return jobs.logs(runner, ctx.cfg, args.job_id, tail=args.tail)


def cmd_jobs_wait(ctx: Ctx, args) -> dict:
    from . import jobs

    _, runner = ctx.instance_and_runner(args.instance)
    return jobs.wait(runner, ctx.cfg, args.job_id, timeout_seconds=float(args.timeout_minutes) * 60)


def cmd_jobs_list(ctx: Ctx, args) -> dict:
    from . import jobs

    _, runner = ctx.instance_and_runner(args.instance)
    return jobs.list_jobs(runner, ctx.cfg, session_id=args.session_id)


def cmd_install_run(ctx: Ctx, args) -> dict:
    from . import jobs
    from .remote import current_link

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    manifest = build_install_manifest()
    from .prepare import assert_no_remote_credentials, b64_json
    assert_no_remote_credentials(manifest)

    argv = [
        "run",
        "--session-id",
        session_id,
        "--manifest-b64",
        b64_json(manifest),
    ]
    job = jobs.start(
        runner,
        ctx.cfg,
        session_id,
        "install",
        jobs.prepare_command(ctx.cfg, "vc_install.py", argv),
        timeout_seconds=float(ctx.cfg.require("jobs.install_timeout_minutes")) * 60,
    )
    ctx.state.put(
        "jobs",
        job["job_id"],
        {
            **job,
            "instance_id": str(args.instance),
            "session_id": session_id,
            "name": "install",
            "recorded_at": time.time(),
        },
    )
    final = jobs.wait(runner, ctx.cfg, job["job_id"], timeout_seconds=float(ctx.cfg.require("jobs.install_timeout_minutes")) * 60 + 120)
    log = jobs.logs(runner, ctx.cfg, job["job_id"], tail=2000).get("log", "")
    result = jobs.final_json_from_log(log)
    payload = {"job_id": job["job_id"], "state": final.get("state"), "exit_code": final.get("exit_code")}
    if result is not None:
        payload["result"] = result
    if final.get("state") != "succeeded":
        raise VastComfyError(
            code="E_INSTALL",
            message=f"Install job did not succeed (state={final.get('state')!r})",
            stage="install",
            details={**payload, "job_id": job["job_id"]},
            hints=[f"Inspect: vast-comfy jobs logs --job-id {job['job_id']} --instance {args.instance}"],
        )
    return payload


def cmd_install_verify(ctx: Ctx, args) -> dict:
    from .locks import assert_owns
    from .prepare import b64_json
    from .remote import invoke

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    assert_owns(runner, ctx.cfg, session_id)
    manifest = build_install_manifest()
    argv = ["verify", "--session-id", session_id, "--manifest-b64", b64_json(manifest)]
    if args.verify_shas:
        argv.append("--verify-shas")
    return invoke(runner, ctx.cfg, "vc_install.py", argv, timeout_seconds=1800)


def _session_guard(ctx: Ctx, instance_id, session_id_arg):
    from .locks import assert_owns

    _, runner = ctx.instance_and_runner(instance_id)
    session_id = ctx.require_session(session_id_arg, instance_id)
    assert_owns(runner, ctx.cfg, session_id)
    return runner, session_id


def cmd_comfy_status(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    return comfy.status(runner, ctx.cfg, session_id)


def cmd_comfy_health(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    return comfy.health(runner, ctx.cfg, session_id, gpu=args.gpu)


def cmd_comfy_restart(ctx: Ctx, args) -> dict:
    from . import comfy

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    return comfy.restart(runner, ctx.cfg, session_id)


def cmd_comfy_verify_nodes(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    else:
        # Derived from the active profile's own workflows: a hardcoded list can
        # only ever be right for one model family.
        from .prepare import required_nodes_for_profile

        nodes = required_nodes_for_profile()
    return comfy.verify_nodes(runner, ctx.cfg, session_id, nodes)


def cmd_comfy_gpu_check(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    return comfy.gpu_check(runner, ctx.cfg, session_id)


def cmd_comfy_logs(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    return comfy.logs(runner, ctx.cfg, session_id, tail=args.tail)


def cmd_comfy_upload_image(ctx: Ctx, args) -> dict:
    from . import comfy

    runner, session_id = _session_guard(ctx, args.instance, args.session_id)
    result = comfy.upload_image(
        runner, ctx.cfg, session_id, local_path=args.file, name=args.name
    )
    return result


def cmd_smoke_run(ctx: Ctx, args) -> dict:
    from . import smoke

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    result = smoke.run_smoke(
        runner,
        ctx.cfg,
        session_id=session_id,
        instance_id=args.instance,
        run_id=args.run_id,
        cases=args.cases,
        verify_model_shas=args.verify_model_shas,
        fixed_seed=args.fixed_seed,
    )
    if result.get("job_id"):
        ctx.state.put(
            "jobs",
            result["job_id"],
            {
                "job_id": result["job_id"],
                "instance_id": str(args.instance),
                "session_id": session_id,
                "name": "smoke",
                "run_id": result.get("run_id"),
                "recorded_at": time.time(),
            },
        )
    return result


def cmd_smoke_report(ctx: Ctx, args) -> dict:
    from . import locks, smoke

    validate_identifier(args.run_id, label="run id")
    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    with locks.operation(runner, ctx.cfg, session_id, label="report"):
        return smoke.show_report(runner, ctx.cfg, args.run_id)


def cmd_download_report(ctx: Ctx, args) -> dict:
    from . import download as dl
    from . import locks

    validate_identifier(args.run_id, label="run id")
    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    with locks.operation(runner, ctx.cfg, session_id, label="download"):
        return dl.download_report(runner, ctx.cfg, run_id=args.run_id, dest_dir=args.dest, overwrite=args.overwrite)


def cmd_download_images(ctx: Ctx, args) -> dict:
    from . import download as dl
    from . import locks

    validate_identifier(args.run_id, label="run id")
    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    with locks.operation(runner, ctx.cfg, session_id, label="download"):
        return dl.download_smoke_images(
            runner, ctx.cfg, run_id=args.run_id, dest_dir=args.dest, overwrite=args.overwrite
        )


def cmd_download_outputs(ctx: Ctx, args) -> dict:
    from . import download as dl
    from . import locks

    _, runner = ctx.instance_and_runner(args.instance)
    session_id = ctx.require_session(args.session_id, args.instance)
    with locks.operation(runner, ctx.cfg, session_id, label="download"):
        return dl.download_outputs(
            runner,
            ctx.cfg,
            dest_dir=args.dest,
            pattern=args.pattern,
            limit=args.limit,
            overwrite=args.overwrite,
        )


def cmd_flow_prepare(ctx: Ctx, args) -> dict:
    from .prepare import prepare_existing

    _, runner = ctx.instance_and_runner(args.instance)
    release = False if args.keep_session else None
    result = prepare_existing(
        runner,
        ctx.cfg,
        instance_id=args.instance,
        session_id=args.session_id,
        release_on_success=release,
        cases=args.cases,
        run_id=args.run_id,
        base_seed=args.base_seed,
        verify_model_shas=args.verify_model_shas,
    )
    ctx.state.record_session(
        result["session_id"],
        instance_id=str(args.instance),
        state="released" if result.get("release") else "active",
        updated_at=time.time(),
    )
    ctx.state.put(
        "jobs",
        result["job_id"],
        {
            "job_id": result["job_id"],
            "instance_id": str(args.instance),
            "session_id": result["session_id"],
            "name": "prepare",
            "run_id": result.get("run_id"),
            "recorded_at": time.time(),
        },
    )
    return result


def cmd_flow_create_and_prepare(ctx: Ctx, args) -> dict:
    def on_step(name: str, payload: Any) -> None:
        logger.info("step: %s", name)

    result = create_and_prepare(
        ctx.client,
        ctx.cfg,
        alias=args.alias,
        label=args.label,
        confirm_paid=args.confirm_paid,
        accept_stale=args.accept_stale,
        release_on_success=False if args.keep_session else None,
        cases=args.cases,
        run_id=args.run_id,
        verify_model_shas=args.verify_model_shas,
        on_step=on_step,
    )
    if result.get("session_id"):
        ctx.state.record_session(
            result["session_id"],
            instance_id=str(result.get("instance_id")),
            state="released" if result.get("release") else "active",
            updated_at=time.time(),
        )
    return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, metavar="PATH")
    common.add_argument("--set", dest="set_overrides", action="append", default=argparse.SUPPRESS, metavar="PATH=VALUE")
    common.add_argument("--env-file", dest="env_file", default=argparse.SUPPRESS, metavar="PATH")
    common.add_argument("--profile", default=argparse.SUPPRESS, metavar="NAME",
                        help="model profile under config/profiles (default: from config)")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--log-level", default=argparse.SUPPRESS, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="vast-comfy",
        description="Rent the cheapest qualifying Vast.ai GPU and prepare ComfyUI + Anima (see SPEC.md).",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"vast-comfy {__version__}")
    sub = parser.add_subparsers(dest="group", required=True)

    def add_group(name: str, help_text: str) -> argparse.ArgumentParser:
        group_parser = sub.add_parser(name, help=help_text, parents=[common])
        group_parser.required = True
        group_sub = group_parser.add_subparsers(dest="command", required=True)
        group_parser.sub = group_sub  # type: ignore[attr-defined]
        return group_parser

    # config
    g = add_group("config", "configuration helpers")
    p = g.sub.add_parser("show", help="show the effective config", parents=[common])
    p.set_defaults(func=cmd_config_show)
    p = g.sub.add_parser("validate", help="validate config, aliases, models, smoke cases", parents=[common])
    p.set_defaults(func=cmd_config_validate)

    # gpu
    g = add_group("gpu", "GPU alias mapping")
    p = g.sub.add_parser("aliases", help="list known GPU aliases", parents=[common])
    p.add_argument("--custom-only", action="store_true")
    p.set_defaults(func=cmd_gpu_aliases)
    p = g.sub.add_parser("resolve", help="resolve one alias", parents=[common])
    p.add_argument("--alias", required=True)
    p.set_defaults(func=cmd_gpu_resolve)
    p = g.sub.add_parser("update-aliases", help="refresh the GPU name directory (preserves custom aliases)", parents=[common])
    p.set_defaults(func=cmd_gpu_update)
    p = g.sub.add_parser("add-alias", help="add a custom alias", parents=[common])
    p.add_argument("--alias", required=True)
    p.add_argument("--gpu-name", required=True)
    p.add_argument("--vram-gb-min", type=float, default=None)
    p.add_argument("--vram-gb-max", type=float, default=None)
    p.add_argument("--notes", default=None)
    p.set_defaults(func=cmd_gpu_add_alias)

    # offers
    g = add_group("offers", "search & quote (never rents)")
    for name, only_best in (("search", False), ("best", True)):
        p = g.sub.add_parser(name, help="search qualifying offers" if not only_best else "show only the cheapest qualifying offer", parents=[common])
        p.add_argument("--alias", required=True)
        p.add_argument("--top", type=int, default=10)
        p.set_defaults(func=cmd_offers_search, only_best=only_best)

    # rent
    g = add_group("rent", "paid creation (requires --confirm-paid)")
    p = g.sub.add_parser("create", help="create the instance for a verified offer", parents=[common])
    p.add_argument("--alias", required=True)
    p.add_argument("--offer-id", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--confirm-paid", action="store_true")
    p.add_argument("--wait", action="store_true", help="wait until running and verify SSH")
    p.set_defaults(func=cmd_rent_create)
    p = g.sub.add_parser("verify", help="verify an uncertain create by label", parents=[common])
    p.add_argument("--label", required=True)
    p.add_argument("--since", type=float, default=None, help="epoch seconds (default: last hour)")
    p.set_defaults(func=cmd_rent_verify)

    # instance
    g = add_group("instance", "instance lifecycle & inspection")
    p = g.sub.add_parser("list", help="list instances on the account", parents=[common])
    p.set_defaults(func=cmd_instance_list)
    p = g.sub.add_parser("show", help="show one instance", parents=[common])
    p.add_argument("--instance", required=True)
    p.set_defaults(func=cmd_instance_show)
    p = g.sub.add_parser("wait-ready", help="wait for running + verify SSH login", parents=[common])
    p.add_argument("--instance", required=True)
    p.set_defaults(func=cmd_instance_wait_ready)
    p = g.sub.add_parser("stop", help="stop an instance (storage billing continues)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_instance_stop)
    p = g.sub.add_parser("destroy", help="destroy an instance (irreversible; needs --confirm-destroy)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--confirm-destroy", action="store_true")
    p.set_defaults(func=cmd_instance_destroy)
    p = g.sub.add_parser("logs", help="fetch container logs", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--tail", type=int, default=200)
    p.set_defaults(func=cmd_instance_logs)

    # ssh
    g = add_group("ssh", "proxy SSH, tunnels, portal info")
    p = g.sub.add_parser("info", help="connection info + tunnel/portal", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--local-port", type=int, default=8188)
    p.set_defaults(func=cmd_ssh_info)
    p = g.sub.add_parser("test", help="verify an actual SSH command", parents=[common])
    p.add_argument("--instance", required=True)
    p.set_defaults(func=cmd_ssh_test)
    p = g.sub.add_parser("tunnel", help="print/start/stop the local tunnel", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--local-port", type=int, default=8188)
    p.add_argument("--start", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.set_defaults(func=cmd_ssh_tunnel)

    # keys
    g = add_group("keys", "SSH keys")
    p = g.sub.add_parser("generate", help="create a new key pair (never overwrites)", parents=[common])
    p.add_argument("--path", default="~/.ssh/id_ed25519")
    p.set_defaults(func=cmd_keys_generate)
    p = g.sub.add_parser("register", help="register a public key with Vast.ai", parents=[common])
    p.add_argument("--pubkey", default="~/.ssh/id_ed25519.pub")
    p.set_defaults(func=cmd_keys_register)
    p = g.sub.add_parser("list", help="list account SSH keys", parents=[common])
    p.set_defaults(func=cmd_keys_list)

    # deploy
    g = add_group("deploy", "management package deployment")
    p = g.sub.add_parser("package", help="deploy + verify the whitelisted management package", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_deploy_package)

    # session
    g = add_group("session", "exclusive session lock")
    p = g.sub.add_parser("acquire", help="acquire the exclusive session", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_session_acquire)
    p = g.sub.add_parser("status", help="diagnose the lock (read-only)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_session_status)
    p = g.sub.add_parser("release", help="release the session (refuses while jobs run)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_session_release)
    p = g.sub.add_parser("recover", help="user-driven stale-lock recovery (needs --accept-stale)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--accept-stale", action="store_true")
    p.set_defaults(func=cmd_session_recover)
    p = g.sub.add_parser(
        "resolve-inference",
        help="human rescue: clear one unresolved inference fence after verifying it ended (needs --confirm-verified)",
        parents=[common],
    )
    p.add_argument("--instance", required=True)
    p.add_argument("--intent-id", required=True)
    p.add_argument("--confirm-verified", action="store_true")
    p.set_defaults(func=cmd_session_resolve_inference)

    # jobs
    g = add_group("jobs", "remote background jobs")
    p = g.sub.add_parser("start", help="start a remote background job", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--timeout-minutes", type=float, required=True)
    p.add_argument("job_command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_jobs_start)
    p = g.sub.add_parser("status", help="job status", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--job-id", required=True)
    p.set_defaults(func=cmd_jobs_status)
    p = g.sub.add_parser("logs", help="job log tail", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--tail", type=int, default=200)
    p.set_defaults(func=cmd_jobs_logs)
    p = g.sub.add_parser("wait", help="bounded wait (timeout does not stop the job)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--timeout-minutes", type=float, default=60)
    p.set_defaults(func=cmd_jobs_wait)
    p = g.sub.add_parser("list", help="list recent jobs", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_jobs_list)

    # install
    g = add_group("install", "models & plugins")
    p = g.sub.add_parser("run", help="install according to config/models.yaml", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_install_run)
    p = g.sub.add_parser("verify", help="verify installed models/plugins", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--verify-shas", action="store_true")
    p.set_defaults(func=cmd_install_verify)

    # comfy
    g = add_group("comfy", "ComfyUI service")
    p = g.sub.add_parser("status", help="service + API + queue status", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_comfy_status)
    p = g.sub.add_parser("health", help="API health (+ optional GPU)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--gpu", action="store_true")
    p.set_defaults(func=cmd_comfy_health)
    p = g.sub.add_parser("restart", help="restart (refuses when the queue is busy)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_comfy_restart)
    p = g.sub.add_parser("verify-nodes", help="check required nodes via /object_info", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--nodes", default=None)
    p.set_defaults(func=cmd_comfy_verify_nodes)
    p = g.sub.add_parser("gpu-check", help="nvidia-smi sanity check", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_comfy_gpu_check)
    p = g.sub.add_parser("logs", help="ComfyUI service log tail", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--tail", type=int, default=200)
    p.set_defaults(func=cmd_comfy_logs)
    p = g.sub.add_parser(
        "upload-image", help="copy a local PNG into ComfyUI's input directory", parents=[common]
    )
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--file", required=True, help="local .png path")
    p.add_argument("--name", default=None, help="name inside ComfyUI input (default: basename)")
    p.set_defaults(func=cmd_comfy_upload_image)

    # smoke
    g = add_group("smoke", "smoke tests & reports")
    p = g.sub.add_parser("run", help="run the smoke suite (READY only when all pass)", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--cases", default=None, help="comma-separated case ids (subset => verdict PARTIAL)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--fixed-seed", action="store_true")
    p.add_argument("--verify-model-shas", action="store_true")
    p.set_defaults(func=cmd_smoke_run)
    p = g.sub.add_parser("report", help="read a remote report", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_smoke_report)

    # download
    g = add_group("download", "explicit downloads (never automatic)")
    p = g.sub.add_parser("report", help="download report files for a run", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download_report)
    p = g.sub.add_parser("images", help="download smoke images for a run", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download_images)
    p = g.sub.add_parser("outputs", help="one-shot batch download of recent outputs", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--dest", required=True)
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download_outputs)

    # flow
    g = add_group("flow", "combined flows")
    p = g.sub.add_parser("prepare", help="prepare an existing instance end-to-end", parents=[common])
    p.add_argument("--instance", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--cases", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--base-seed", type=int, default=None)
    p.add_argument("--verify-model-shas", action="store_true")
    p.add_argument("--keep-session", action="store_true", help="do not release the lock on success")
    p.set_defaults(func=cmd_flow_prepare)
    p = g.sub.add_parser("create-and-prepare", help="search -> rent (paid) -> prepare", parents=[common])
    p.add_argument("--alias", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--confirm-paid", action="store_true")
    p.add_argument("--accept-stale", action="store_true")
    p.add_argument("--cases", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--verify-model-shas", action="store_true")
    p.add_argument("--keep-session", action="store_true")
    p.set_defaults(func=cmd_flow_create_and_prepare)

    return parser


# ---------------------------------------------------------------------------
# Output & entry point
# ---------------------------------------------------------------------------


def _human_print(command: str, result: dict) -> None:
    special = []
    if "best" in result and isinstance(result["best"], dict):
        best = result["best"]
        special.append(
            f"cheapest: offer {best.get('offer_id')} | {best.get('gpu_name')} | "
            f"${best.get('dph_total_usd_per_hour')}/h | {best.get('geolocation')}"
        )
    if "verdict" in result:
        special.append(f"verdict: {result['verdict']}")
    if "instance_id" in result:
        special.append(f"instance: {result['instance_id']}")
    if "session_id" in result:
        special.append(f"session: {result['session_id']}")
    if "job_id" in result:
        special.append(f"job: {result['job_id']}")
    if special:
        for line in special:
            print(line)
        print("")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(x for x in (getattr(args, "group", None), getattr(args, "command", None)) if x)
    json_mode = bool(getattr(args, "json", False))
    setup_logging(getattr(args, "log_level", "INFO"))

    try:
        if getattr(args, "env_file", None):
            apply_env_file(args.env_file)
        ctx = Ctx(args)
        result = args.func(ctx, args) or {}
        payload = {"ok": True, "command": command, **result}
        if json_mode:
            print(json.dumps(payload, ensure_ascii=False, default=str))
        else:
            _human_print(command, payload)
        return 0
    except VastComfyError as exc:
        payload = exc.to_dict()
        payload["command"] = command
        if json_mode:
            print(json.dumps(payload, ensure_ascii=False, default=str))
        else:
            print(f"ERROR {exc.code}: {exc.message}", file=sys.stderr)
            if exc.stage:
                print(f"stage: {exc.stage}", file=sys.stderr)
            for line in exc.hints:
                print(f"hint: {line}", file=sys.stderr)
            if exc.details:
                print(json.dumps(exc.details, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return exit_code_for(exc.code)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
