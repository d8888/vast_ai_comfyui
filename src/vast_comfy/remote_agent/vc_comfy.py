#!/usr/bin/env python3
"""vast-comfy remote ComfyUI service helpers: health, queue, restart, nodes, GPU.

All modifying actions require a live session lock. Restart refuses while the
ComfyUI queue is busy: the queue is never cleared and the service is never
force-restarted over pending work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402


def http_json(url: str, *, timeout: float = 10.0, method: str = "GET", body: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 (localhost service)
        payload = resp.read().decode("utf-8")
    return json.loads(payload) if payload else None


def comfy_base(args) -> str:
    return args.comfy_url.rstrip("/")


def check_health(base_url: str) -> dict:
    stats = http_json(f"{base_url}/system_stats", timeout=10.0)
    devices = (stats or {}).get("devices") or []
    cuda_devices = [d for d in devices if str(d.get("type", "")).lower() == "cuda"]
    if not cuda_devices:
        C.fail(
            "E_COMFY_UNAVAILABLE",
            "ComfyUI is reachable but reports no CUDA device; refusing to claim the GPU is usable.",
            exit_code=1,
            devices=devices[:4],
        )
    system = (stats or {}).get("system") or {}
    return {
        "api_ok": True,
        "devices": [
            {
                "name": d.get("name"),
                "type": d.get("type"),
                "vram_total": d.get("vram_total"),
                "vram_free": d.get("vram_free"),
                "torch_version": d.get("torch_version"),
            }
            for d in cuda_devices
        ],
        "comfyui_version": system.get("comfyui_version"),
        "python_version": system.get("python_version"),
        "os": system.get("os"),
    }


def queue_snapshot(base_url: str) -> dict:
    payload = http_json(f"{base_url}/queue", timeout=10.0) or {}
    running = payload.get("queue_running") or []
    pending = payload.get("queue_pending") or []
    return {"running": len(running), "pending": len(pending), "busy": bool(running or pending)}


def service_state(service: str) -> dict:
    try:
        result = subprocess.run(
            ["supervisorctl", "status", service], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"running": False, "raw": f"supervisorctl failed: {exc}"}
    raw = (result.stdout or result.stderr or "").strip()
    return {"running": "RUNNING" in raw, "raw": raw}


def wait_api_up(base_url: str, timeout_seconds: float, poll_seconds: float = 3.0) -> dict:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            return check_health(base_url)
        except Exception as exc:  # noqa: BLE001 - retry until deadline
            last_error = str(exc)
            time.sleep(poll_seconds)
    C.fail(
        "E_COMFY_UNAVAILABLE",
        f"ComfyUI API did not become healthy within {timeout_seconds:.0f}s",
        exit_code=1,
        last_error=last_error,
    )


def restart_service(args) -> dict:
    C.ensure_owns(args.session_id)
    base = comfy_base(args)
    snapshot = queue_snapshot(base)
    if snapshot["busy"]:
        C.fail(
            "E_COMFY_BUSY",
            "ComfyUI has running or pending work; refusing to restart and never clearing the queue.",
            exit_code=13,
            queue=snapshot,
        )
    state_before = service_state(args.service)
    rc = subprocess.call(["supervisorctl", "restart", args.service])
    if rc != 0:
        C.fail(
            "E_COMFY_UNAVAILABLE",
            f"supervisorctl restart {args.service} failed (rc={rc})",
            exit_code=1,
            supervisor=state_before,
        )
    health = wait_api_up(base, args.wait_seconds)
    return {"restarted": True, "service": args.service, "health": health}


def verify_nodes(args) -> dict:
    base = comfy_base(args)
    object_info = http_json(f"{base}/object_info", timeout=30.0) or {}
    expected = [n.strip() for n in args.nodes.split(",") if n.strip()]
    missing = [n for n in expected if n not in object_info]
    if missing:
        C.fail(
            "E_COMFY_UNAVAILABLE",
            f"ComfyUI is missing required nodes: {', '.join(missing)}",
            exit_code=1,
            missing=missing,
        )
    return {"nodes_ok": True, "checked": expected}


def gpu_check() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        C.fail("E_GPU_INCOMPATIBLE", f"nvidia-smi is not usable: {exc}", exit_code=1)
    if result.returncode != 0:
        C.fail(
            "E_GPU_INCOMPATIBLE",
            "nvidia-smi failed; the instance does not expose a usable NVIDIA GPU.",
            exit_code=1,
            stderr=result.stderr.strip()[:300],
        )
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpus:
        C.fail("E_GPU_INCOMPATIBLE", "nvidia-smi reported no GPUs", exit_code=1)
    return {"gpus": gpus}


def cmd_status(args) -> None:
    C.ensure_owns(args.session_id)
    base = comfy_base(args)
    state = service_state(args.service)
    health = None
    queue = None
    api_error = None
    try:
        health = check_health(base)
        queue = queue_snapshot(base)
    except Exception as exc:  # noqa: BLE001
        api_error = str(exc)
    C.emit(
        {
            "ok": True,
            "service": state,
            "health": health,
            "queue": queue,
            "api_error": api_error,
        }
    )


def cmd_health(args) -> None:
    C.ensure_owns(args.session_id)
    payload = {"ok": True, "health": check_health(comfy_base(args)), "queue": queue_snapshot(comfy_base(args))}
    if args.gpu:
        payload["gpu"] = gpu_check()
    C.emit(payload)


def cmd_restart(args) -> None:
    C.emit({"ok": True, **restart_service(args)})


def cmd_verify_nodes(args) -> None:
    C.ensure_owns(args.session_id)
    C.emit({"ok": True, **verify_nodes(args)})


def cmd_gpu_check(args) -> None:
    C.ensure_owns(args.session_id)
    C.emit({"ok": True, **gpu_check()})


def cmd_queue(args) -> None:
    C.ensure_owns(args.session_id)
    C.emit({"ok": True, "queue": queue_snapshot(comfy_base(args))})


def cmd_logs(args) -> None:
    C.ensure_owns(args.session_id)
    candidates = ["/var/log/portal/comfyui.log", "/var/log/comfyui.log"]
    tail_text = None
    used = None
    for path in candidates:
        result = subprocess.run(
            ["tail", "-n", str(int(args.tail)), path], capture_output=True, text=True
        )
        if result.returncode == 0:
            tail_text = result.stdout
            used = path
            break
    state = service_state(args.service)
    payload = {"ok": True, "service": state, "log_path": used, "log": tail_text}
    if tail_text is None:
        payload["note"] = (
            "ComfyUI log files were not found; the service may not have started yet. "
            "Use `jobs logs` for management-job logs."
        )
    C.emit(payload)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comfy-url", default="http://localhost:18188")
    parser.add_argument("--service", default="comfyui")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote ComfyUI helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status")
    p.add_argument("--session-id", required=True)
    add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("health")
    p.add_argument("--session-id", required=True)
    p.add_argument("--gpu", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("restart")
    p.add_argument("--session-id", required=True)
    p.add_argument("--wait-seconds", type=float, default=240.0)
    add_common(p)
    p.set_defaults(func=cmd_restart)

    p = sub.add_parser("verify-nodes")
    p.add_argument("--session-id", required=True)
    p.add_argument("--nodes", required=True)
    add_common(p)
    p.set_defaults(func=cmd_verify_nodes)

    p = sub.add_parser("gpu-check")
    p.add_argument("--session-id", required=True)
    p.set_defaults(func=cmd_gpu_check)

    p = sub.add_parser("queue")
    p.add_argument("--session-id", required=True)
    add_common(p)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("logs")
    p.add_argument("--session-id", required=True)
    p.add_argument("--tail", type=int, default=200)
    p.set_defaults(func=cmd_logs)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    C.run_cli(args.func, args)


if __name__ == "__main__":
    main()
