#!/usr/bin/env python3
"""vast-comfy remote smoke test runner (SPEC.md section 13).

Runs the four required cases in order against the local ComfyUI API, proves the
GPU is really used, and only reports READY when every required case produced a
*new* image (correct size, decodable, not single-color) in that very run.

Anti-fake guarantees:

* Every prompt is submitted with a per-run unique seed (recorded in the report),
  so ComfyUI's node cache cannot satisfy the run with a previous result.
* Every output image must have been modified after the case started.
* A failure aborts the remaining cases and marks them ``not_run``; a partial or
  failed suite can never report READY.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vc_common as C  # noqa: E402
import vc_comfy as CM  # noqa: E402

try:  # PIL ships in the ComfyUI image venv; needed for real PNG validation.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only on broken images
    Image = None

SMOKE_CLIENT_ID = "vast-comfy-smoke"


class CaseFailure(Exception):
    def __init__(self, reason: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}


def load_cases(cases_b64: str) -> list[dict]:
    try:
        payload = json.loads(base64.b64decode(cases_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        C.fail("E_USAGE", f"invalid smoke cases payload: {exc}")
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        C.fail("E_USAGE", "smoke cases payload must contain a non-empty 'cases' list")
    for case in cases:
        if not isinstance(case, dict):
            C.fail("E_USAGE", f"smoke case entry must be a mapping: {case!r}")
        for key in ("id", "workflow", "timeout_minutes", "overrides", "expect", "inputs"):
            if key not in case:
                C.fail("E_USAGE", f"smoke case {case.get('id')!r} is missing key {key!r}")
    return cases


def compute_seed(base_seed: int, run_token: int, case_index: int, unique: bool) -> int:
    if not unique:
        return int(base_seed)
    return int(base_seed) + (int(run_token) + case_index) % 1_000_000


def build_prompt(workflow_path: Path, overrides: dict, seed: int) -> dict:
    if not workflow_path.is_file():
        raise CaseFailure("SMOKE_WORKFLOW", f"workflow file not found: {workflow_path}")
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    for node_id, inputs in (overrides or {}).items():
        node = prompt.get(str(node_id))
        if not isinstance(node, dict):
            raise CaseFailure(
                "SMOKE_WORKFLOW",
                f"override targets a missing node: {node_id!r} in {workflow_path.name}",
            )
        node.setdefault("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, str) and value == "$run_seed":
                value = int(seed)
            node["inputs"][key] = value
    return prompt


def submit_prompt(base_url: str, prompt: dict) -> str:
    try:
        response = CM.http_json(
            f"{base_url}/prompt",
            method="POST",
            body={"prompt": prompt, "client_id": SMOKE_CLIENT_ID},
            timeout=60.0,
        )
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            pass
        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            pass
        raise CaseFailure(
            "SMOKE_SUBMIT",
            f"ComfyUI rejected the prompt (HTTP {exc.code})",
            {"error": (parsed or {}).get("error"), "node_errors": (parsed or {}).get("node_errors")},
        ) from exc
    if not isinstance(response, dict) or not response.get("prompt_id"):
        raise CaseFailure("SMOKE_SUBMIT", "ComfyUI did not return a prompt_id", {"response": response})
    return str(response["prompt_id"])


def wait_for_history(base_url: str, prompt_id: str, timeout_seconds: float) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = CM.http_json(f"{base_url}/history/{prompt_id}", timeout=20.0) or {}
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("completed"):
                if status.get("status_str") == "success":
                    return entry
                raise CaseFailure(
                    "SMOKE_EXEC_ERROR",
                    "ComfyUI reported an execution error for this case",
                    {"status": status, "messages": status.get("messages", [])[-10:]},
                )
        time.sleep(2.0)
    raise CaseFailure(
        "SMOKE_TIMEOUT",
        f"case did not finish within {timeout_seconds:.0f}s",
        {"prompt_id": prompt_id, "note": "ComfyUI may still be executing; nothing was cancelled."},
    )


def collect_images(entry: dict, comfy_dir: Path) -> list[dict]:
    images: list[dict] = []
    outputs = entry.get("outputs") or {}
    for node_id, out in outputs.items():
        for image in (out or {}).get("images", []) or []:
            if image.get("type") != "output":
                continue
            subfolder = image.get("subfolder") or ""
            filename = image.get("filename") or ""
            if not filename:
                continue
            path = comfy_dir / "output" / subfolder / filename
            images.append(
                {"filename": filename, "subfolder": subfolder, "node_id": str(node_id), "path": path}
            )
    if not images:
        raise CaseFailure("SMOKE_IMAGE_MISSING", "the workflow produced no new output image")
    return images


def validate_image(record: dict, expect: dict, case_start_epoch: float) -> dict:
    path: Path = record["path"]
    if not path.is_file():
        raise CaseFailure("SMOKE_IMAGE_MISSING", f"expected output image is missing: {path}")
    stat = path.stat()
    if stat.st_mtime < case_start_epoch - 5.0:
        raise CaseFailure(
            "SMOKE_IMAGE_OLD",
            "the output image predates this run; it is NOT accepted as new evidence",
            {"path": str(path), "mtime": stat.st_mtime, "case_start": case_start_epoch},
        )
    if Image is None:
        raise CaseFailure(
            "SMOKE_PIL_MISSING",
            "Pillow is unavailable on the instance; cannot decode/verify the PNG",
        )
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            rgb = img.convert("RGB")
            colors = rgb.getcolors(maxcolors=64)
    except Exception as exc:  # noqa: BLE001
        raise CaseFailure("SMOKE_IMAGE_DECODE", f"output image cannot be decoded: {exc}") from exc

    expect_w = int(expect.get("width", 0))
    expect_h = int(expect.get("height", 0))
    if (expect_w and width != expect_w) or (expect_h and height != expect_h):
        raise CaseFailure(
            "SMOKE_IMAGE_SIZE",
            f"output image size {width}x{height} does not match expected {expect_w}x{expect_h}",
            {"path": str(path)},
        )
    if colors is not None and len(colors) == 1:
        raise CaseFailure(
            "SMOKE_IMAGE_SINGLE_COLOR",
            "output image is a single flat color",
            {"path": str(path)},
        )
    return {
        "filename": record["filename"],
        "subfolder": record["subfolder"],
        "path": str(path),
        "width": width,
        "height": height,
        "bytes": stat.st_size,
        "sha256": C.sha256_file(path),
        "mtime": stat.st_mtime,
        "distinct_colors": None if colors is None else len(colors),
    }


def ensure_input_assets(assets_dir: Path, comfy_dir: Path, case_inputs: list[str]) -> list[dict]:
    input_dir = comfy_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in case_inputs:
        src = assets_dir / name
        if not src.is_file():
            raise CaseFailure("SMOKE_ASSET_MISSING", f"bundled test asset not found: {src}")
        dst = input_dir / name
        src_sha = C.sha256_file(src)
        if dst.is_file() and C.sha256_file(dst) == src_sha:
            records.append({"name": name, "action": "already_present", "sha256": src_sha})
            continue
        shutil.copy2(src, dst)
        if C.sha256_file(dst) != src_sha:
            raise CaseFailure("SMOKE_ASSET", f"copied asset checksum mismatch: {name}")
        records.append({"name": name, "action": "copied", "sha256": src_sha})
    return records


def required_nodes_from_workflows(workflows_dir: Path, workflow_names: list[str]) -> list[str]:
    nodes: set[str] = set()
    for name in workflow_names:
        path = workflows_dir / name
        if not path.is_file():
            raise CaseFailure("SMOKE_WORKFLOW", f"workflow file not found: {path}")
        prompt = json.loads(path.read_text(encoding="utf-8"))
        for node in prompt.values():
            if isinstance(node, dict) and node.get("class_type"):
                nodes.add(str(node["class_type"]))
    return sorted(nodes)


def plugin_commit(comfy_dir: Path) -> str | None:
    import subprocess

    plugin_dir = comfy_dir / "custom_nodes" / "ComfyUI-Anima-LLLite"
    if not (plugin_dir / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(plugin_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def check_models(models: list[dict], comfy_dir: Path, verify_sha: bool) -> list[dict]:
    records = []
    for entry in models:
        rel = C.validate_model_relpath(str(entry["dest"]))
        path = comfy_dir / "models" / rel
        record = {"id": entry.get("id"), "path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            record["size_bytes"] = stat.st_size
            record["mtime"] = stat.st_mtime
            size_ok = entry.get("size_bytes") is None or int(entry["size_bytes"]) == stat.st_size
            record["size_ok"] = size_ok
            if verify_sha or not size_ok:
                actual = C.sha256_file(path)
                record["sha256"] = actual
                record["sha256_ok"] = actual == str(entry["sha256"]).lower()
                record["sha256_verified"] = True
            else:
                record["sha256_verified"] = False  # verified at install time
        else:
            record.update({"size_ok": False, "sha256_verified": False})
        records.append(record)
    bad = [r for r in records if not r["exists"] or not r["size_ok"] or r.get("sha256_ok") is False]
    if bad:
        raise CaseFailure(
            "SMOKE_MODELS",
            "required model files are missing or do not match the manifest",
            {"models": bad},
        )
    return records


def write_report(
    run_dir: Path,
    *,
    run_id: str,
    instance_id: str,
    image_ref: str,
    health: dict,
    gpu: dict,
    nodes_checked: list[str],
    plugin_sha: str | None,
    models: list[dict],
    inputs: list[dict],
    cases: list[dict],
    verdict: str,
    base_seed: int,
    run_token: int,
    started_at: float,
    ended_at: float,
) -> dict:
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "instance_id": instance_id,
        "image": image_ref,
        "verdict": verdict,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(ended_at - started_at, 1),
        "seed_base": base_seed,
        "run_token": run_token,
        "comfyui": health,
        "gpu": gpu,
        "required_nodes_checked": nodes_checked,
        "plugin_commit": plugin_sha,
        "models": models,
        "input_assets": inputs,
        "cases": cases,
        "notes": [
            "READY requires every required case to pass in this run.",
            "Images are validated as new files with the expected size, decodable and not single-color.",
            "Artistic quality and LLLite control strength are for human review and are NOT pass criteria.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_write_json(run_dir / "report.json", report)

    lines = [
        f"# vast-comfy smoke report — {run_id}",
        "",
        f"- Verdict: **{verdict}**",
        f"- Instance: {instance_id}",
        f"- Image: {image_ref}",
        f"- ComfyUI version: {health.get('comfyui_version')}",
        f"- Plugin commit: {plugin_sha}",
        f"- Started: {C.iso_utc(started_at)} / ended: {C.iso_utc(ended_at)} ({report['duration_seconds']}s)",
        "",
        "| # | case | status | duration | images | note |",
        "|---|------|--------|----------|--------|------|",
    ]
    for i, case in enumerate(cases, start=1):
        images = ", ".join(img["filename"] for img in case.get("images", [])) or "-"
        note = case.get("error", {}).get("message", "") if case.get("error") else ""
        lines.append(
            f"| {i} | {case.get('id')} | {case.get('status')} | {case.get('duration_seconds', '-')}s | {images} | {note} |"
        )
    lines += [
        "",
        "Full machine-readable report: report.json (same directory).",
        "Images stay in ComfyUI's output directory; download them explicitly if needed.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_suite(
    *,
    session_id: str,
    run_id: str,
    cases: list[dict],
    workflows_dir: Path,
    assets_dir: Path,
    base_url: str,
    comfy_dir: Path,
    instance_id: str,
    image_ref: str,
    base_seed: int,
    unique_seed: bool,
    run_token: int,
    models: list[dict] | None = None,
    verify_model_shas: bool = False,
    require_plugin_node: bool = True,
) -> tuple[dict, int]:
    started_at = C.now_epoch()
    run_dir = C.RUNS_DIR / run_id

    def early(result: dict, exit_code: int) -> tuple[dict, int]:
        C.atomic_write_json(run_dir / "summary.json", result)
        return result, exit_code

    # ---- preflight ----------------------------------------------------
    CM.check_health(base_url)
    gpu = CM.gpu_check()
    queue = CM.queue_snapshot(base_url)
    if queue["busy"]:
        return early(
            {
                "ok": False,
                "code": "E_COMFY_BUSY",
                "message": "ComfyUI already has running or pending work; refusing to insert smoke tests.",
                "queue": queue,
            },
            13,
        )
    health = CM.check_health(base_url)
    workflow_names = [str(c["workflow"]) for c in cases]
    nodes_checked = required_nodes_from_workflows(workflows_dir, workflow_names)
    object_info = CM.http_json(f"{base_url}/object_info", timeout=60.0) or {}
    missing = [n for n in nodes_checked if n not in object_info]
    if require_plugin_node and "AnimaLLLiteApply_sdscripts" not in object_info:
        missing.append("AnimaLLLiteApply_sdscripts")
    if missing:
        return early(
            {
                "ok": False,
                "code": "E_COMFY_UNAVAILABLE",
                "message": f"required nodes are missing from ComfyUI: {', '.join(sorted(set(missing)))}",
                "missing": sorted(set(missing)),
            },
            1,
        )

    model_records = check_models(models or [], comfy_dir, verify_model_shas) if models else []

    inputs_records: list[dict] = []
    for case in cases:
        try:
            inputs_records += ensure_input_assets(assets_dir, comfy_dir, list(case.get("inputs", [])))
        except CaseFailure as exc:
            return early(
                {"ok": False, "code": exc.reason, "message": exc.message, "details": exc.details}, 1
            )
    plugin_sha = plugin_commit(comfy_dir)

    # ---- run cases ------------------------------------------------------
    results: list[dict] = []
    overall_ok = True
    for index, case in enumerate(cases):
        if not overall_ok:
            results.append(
                {
                    "id": case["id"],
                    "status": "not_run",
                    "reason": "previous case failed; the flow stops explicitly and never skips ahead",
                }
            )
            continue
        case_start = C.now_epoch()
        seed = compute_seed(base_seed, run_token, index, unique_seed)
        record: dict = {"id": case["id"], "seed": seed, "started_at": case_start}
        try:
            prompt = build_prompt(workflows_dir / str(case["workflow"]), case.get("overrides", {}), seed)
            prompt_id = submit_prompt(base_url, prompt)
            record["prompt_id"] = prompt_id
            timeout_seconds = float(case.get("timeout_minutes", 15)) * 60.0
            entry = wait_for_history(base_url, prompt_id, timeout_seconds)
            images = collect_images(entry, comfy_dir)
            validated = [validate_image(img, case.get("expect", {}), case_start) for img in images]
            record.update(
                {
                    "status": "passed",
                    "images": validated,
                    "duration_seconds": round(C.now_epoch() - case_start, 1),
                }
            )
        except CaseFailure as exc:
            overall_ok = False
            record.update(
                {
                    "status": "failed",
                    "reason": exc.reason,
                    "error": {"message": exc.message, "details": exc.details},
                    "duration_seconds": round(C.now_epoch() - case_start, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            overall_ok = False
            record.update(
                {
                    "status": "failed",
                    "reason": "SMOKE_UNEXPECTED",
                    "error": {"message": str(exc), "details": {}},
                    "duration_seconds": round(C.now_epoch() - case_start, 1),
                }
            )
        results.append(record)

    # ---- verdict & report ----------------------------------------------
    requested_all = len(cases) == 4 and all(
        c.get("id") in {r["id"] for r in results} for c in cases
    )
    any_failed = any(r["status"] == "failed" for r in results)
    verdict = "READY" if (overall_ok and requested_all) else ("FAILED" if any_failed else "PARTIAL")

    ended_at = C.now_epoch()
    report = write_report(
        run_dir,
        run_id=run_id,
        instance_id=instance_id,
        image_ref=image_ref,
        health=health,
        gpu=gpu,
        nodes_checked=nodes_checked,
        plugin_sha=plugin_sha,
        models=model_records,
        inputs=inputs_records,
        cases=results,
        verdict=verdict,
        base_seed=base_seed,
        run_token=run_token,
        started_at=started_at,
        ended_at=ended_at,
    )
    result = {
        "ok": verdict != "FAILED",
        "verdict": verdict,
        "run_id": run_id,
        "report_dir": str(run_dir),
        "report_json": str(run_dir / "report.json"),
        "report_markdown": str(run_dir / "report.md"),
        "cases": [
            {"id": r["id"], "status": r["status"], "seed": r.get("seed")} for r in results
        ],
    }
    C.atomic_write_json(run_dir / "summary.json", result)
    return result, (0 if verdict in ("READY", "PARTIAL") else 5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    C.ensure_dirs()
    C.ensure_owns(args.session_id)
    cases = load_cases(args.cases_b64)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            C.fail("E_USAGE", f"no smoke cases match --cases {args.cases!r}")
    models: list[dict] = []
    if args.models_b64:
        try:
            models = json.loads(base64.b64decode(args.models_b64).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            C.fail("E_USAGE", f"invalid models payload: {exc}")

    run_token = int(args.run_token) if args.run_token else int(C.now_epoch())
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + f"{run_token % 4096:03x}"

    result, exit_code = run_suite(
        session_id=args.session_id,
        run_id=run_id,
        cases=cases,
        workflows_dir=Path(args.workflows_dir),
        assets_dir=Path(args.assets_dir),
        base_url=CM.comfy_base(args),
        comfy_dir=C.COMFY_DIR,
        instance_id=str(args.instance_id),
        image_ref=args.image_ref,
        base_seed=int(args.base_seed),
        unique_seed=not args.fixed_seed,
        run_token=run_token,
        models=models,
        verify_model_shas=args.verify_model_shas,
        require_plugin_node=not args.skip_plugin_node_check,
    )
    C.emit(result, exit_code=exit_code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vast-comfy remote smoke runner")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--session-id", required=True)
    p.add_argument("--cases-b64", required=True)
    p.add_argument("--models-b64", default="")
    p.add_argument("--workflows-dir", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--comfy-url", default="http://localhost:18188")
    p.add_argument("--cases", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--run-token", default="")
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--fixed-seed", action="store_true")
    p.add_argument("--verify-model-shas", action="store_true")
    p.add_argument("--skip-plugin-node-check", action="store_true")
    p.add_argument("--instance-id", default="")
    p.add_argument("--image-ref", default="")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    C.run_cli(args.func, args)


if __name__ == "__main__":
    main()
