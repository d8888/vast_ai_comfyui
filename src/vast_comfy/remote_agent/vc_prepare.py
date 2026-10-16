#!/usr/bin/env python3
"""vast-comfy "prepare" orchestration for an existing instance.

One background job performs: install models/plugins -> service restart if the
installation changed files (never over a busy queue) -> health + GPU checks ->
full smoke suite -> remote report. Any failure stops the remaining steps and is
reported with the exact stage; nothing is cleaned up implicitly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402
import vc_comfy as CM  # noqa: E402
import vc_install as INS  # noqa: E402
import vc_smoke as SMOKE  # noqa: E402


def cmd_run(args) -> dict:
    C.ensure_dirs()
    C.ensure_owns(args.session_id)
    if not os.environ.get("VAST_COMFY_OPERATION_FD"):
        C.fail("E_LOCK_NOT_HELD", "Prepare requires an actual operation FD, not only a delegated token.", exit_code=4)
    run_dir = C.reserve_run_directory(args.run_id, stage="prepare")
    summary: dict = {
        "ok": False,
        "run_id": args.run_id,
        "instance_id": str(args.instance_id),
        "started_at": C.now_epoch(),
        "steps": {},
    }
    try:
        try:
            manifest = json.loads(base64.b64decode(args.install_manifest_b64).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            C.fail("E_USAGE", f"invalid install manifest: {exc}")
        install_result = INS._install_manifest(manifest)
        summary["steps"]["install"] = install_result
        C.check_deadline()

        base = CM.comfy_base(args)
        api_ok = True
        try:
            CM.check_health(base)
        except (C.RemoteFail, OSError) as exc:
            api_ok = False
            summary["steps"]["initial_health_error"] = {
                "code": getattr(exc, "code", "E_COMFY_UNAVAILABLE"),
                "message": C.redact(str(exc)),
            }

        if install_result.get("changed") or not api_ok:
            if api_ok:
                queue = CM.queue_snapshot(base)
                if queue["busy"]:
                    C.fail(
                        "E_COMFY_BUSY",
                        "The installation changed files and a service restart is required, "
                        "but ComfyUI has running or pending work. The queue is never cleared.",
                        exit_code=13,
                        queue=queue,
                    )
            else:
                summary["steps"]["restart_note"] = (
                    "ComfyUI API was unreachable; the queue could not be checked before restarting."
                )
            summary["steps"]["restart"] = CM.restart_service(args)

        summary["steps"]["health"] = CM.check_health(base)
        summary["steps"]["gpu"] = CM.gpu_check()

        cases = SMOKE.load_cases(args.cases_b64)
        if args.cases:
            wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
            cases = [c for c in cases if c["id"] in wanted]
            if not cases:
                C.fail("E_USAGE", f"no smoke cases match --cases {args.cases!r}")
        models: list[dict] = []
        if args.models_b64:
            models = json.loads(base64.b64decode(args.models_b64).decode("utf-8"))
        run_token = int(args.run_token) if args.run_token else int(C.now_epoch())

        smoke_result, smoke_exit = SMOKE.run_suite(
            session_id=args.session_id,
            run_id=args.run_id,
            cases=cases,
            workflows_dir=Path(args.workflows_dir),
            assets_dir=Path(args.assets_dir),
            base_url=base,
            comfy_dir=C.COMFY_DIR,
            instance_id=str(args.instance_id),
            image_ref=args.image_ref,
            base_seed=int(args.base_seed),
            unique_seed=not args.fixed_seed,
            run_token=run_token,
            models=models,
            verify_model_shas=args.verify_model_shas,
            _from_prepare=True,
        )
        summary["steps"]["smoke"] = smoke_result
        summary["ok"] = bool(smoke_result.get("ok"))
        summary["verdict"] = smoke_result.get("verdict")
        summary["ended_at"] = C.now_epoch()
        C.atomic_write_json(run_dir / "prepare_summary.json", summary)
        if smoke_exit != 0:
            C.emit(
                {
                    "ok": False,
                    "code": "E_SMOKE_FAILED",
                    "message": "the smoke suite did not reach READY",
                    "run_id": args.run_id,
                    "summary_path": str(run_dir / "prepare_summary.json"),
                    "smoke": smoke_result,
                },
                exit_code=smoke_exit,
            )
        return {
            "ok": True,
            "verdict": smoke_result.get("verdict"),
            "run_id": args.run_id,
            "summary_path": str(run_dir / "prepare_summary.json"),
            "install": {
                "changed": install_result.get("changed"),
                "models": [m["action"] for m in install_result.get("models", [])],
                "plugins": [p["action"] for p in install_result.get("plugins", [])],
            },
            "smoke": {
                "report_json": smoke_result.get("report_json"),
                "report_markdown": smoke_result.get("report_markdown"),
                "cases": smoke_result.get("cases"),
            },
        }
    except C.RemoteFail as exc:
        summary["error"] = {"code": exc.code, "message": exc.message, "details": exc.details}
        summary["ended_at"] = C.now_epoch()
        C.atomic_write_json(run_dir / "prepare_summary.json", summary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote prepare flow")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--session-id", required=True)
    p.add_argument("--instance-id", default="")
    p.add_argument("--image-ref", default="")
    p.add_argument("--install-manifest-b64", required=True)
    p.add_argument("--models-b64", default="")
    p.add_argument("--cases-b64", required=True)
    p.add_argument("--workflows-dir", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--comfy-url", default="http://localhost:18188")
    p.add_argument("--service", default="comfyui")
    p.add_argument("--wait-seconds", type=float, default=240.0)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-token", default="")
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--fixed-seed", action="store_true")
    p.add_argument("--verify-model-shas", action="store_true")
    p.add_argument("--cases", default="")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    C.run_cli(args.func, args, guard=True)


if __name__ == "__main__":
    main()
