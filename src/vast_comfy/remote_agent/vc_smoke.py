#!/usr/bin/env python3
"""vast-comfy remote smoke test runner (SPEC.md section 13).

Runs the four required cases in order against the local ComfyUI API, proves the
GPU is really used, and only reports READY when every required case produced a
*new* image (correct size, decodable, not single-color) in that very run.

Anti-fake guarantees:

* Every prompt uses a cryptographically random seed (recorded in the report),
  and history must not report a cached sampler. Fixed-seed runs fail closed.
* Case identities are checked against the actual output-to-sampler/model graph.
* Every output image must have been modified after the case started.
* A failure aborts the remaining cases and marks them ``not_run``; a partial or
  failed suite can never report READY.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import secrets
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
REQUIRED_CASE_IDS = ("base_t2i", "lllite_lineart", "lllite_inpaint", "turbo_t2i")
MAX_SEED = (1 << 64) - 1
LLLITE_NODE = "AnimaLLLiteApply_sdscripts"


class CaseFailure(Exception):
    def __init__(self, reason: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}


def _positive_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _relative_path(name, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or any(c in name for c in ("\\", ":", "\0"))
        or any(part in ("", ".", "..") or part.startswith("~") for part in name.split("/"))
    ):
        raise CaseFailure("SMOKE_PATH", f"unsafe {label}: {name!r}")
    return name


def _contained_path(root: Path, name: str, label: str) -> Path:
    path = root / _relative_path(name, label)
    if not path.resolve().is_relative_to(root.resolve()):
        raise CaseFailure("SMOKE_PATH", f"{label} escapes its directory: {name!r}")
    return path


def _input_asset_name(name) -> str:
    _relative_path(name, "input asset path")
    # LoadImage annotations can redirect a seemingly relative input to output
    # or temp. Validate/copy the same file that ComfyUI will actually read.
    if not name.lower().endswith(".png") or "[" in name or "]" in name:
        raise CaseFailure("SMOKE_PATH", f"input must be an unannotated relative PNG path: {name!r}")
    return name


def _validate_value(value) -> None:
    """Reject non-JSON/non-finite parameters, including nested override values."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CaseFailure("SMOKE_CONFIG", "parameter keys must be strings")
        for item in value.values():
            _validate_value(item)
    elif isinstance(value, list):
        for item in value:
            _validate_value(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise CaseFailure("SMOKE_CONFIG", "parameters must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        raise CaseFailure("SMOKE_CONFIG", "parameters must be JSON values")


def validate_cases(cases) -> list[dict]:
    if not isinstance(cases, list) or not cases:
        raise CaseFailure("SMOKE_CONFIG", "smoke cases must be a non-empty list")
    seen = set()
    required = {"id", "workflow", "timeout_minutes", "overrides", "expect", "inputs"}
    for case in cases:
        if not isinstance(case, dict):
            raise CaseFailure("SMOKE_CONFIG", "smoke case entry must be a mapping")
        if not required <= case.keys() or case.keys() - required - {"title"}:
            raise CaseFailure("SMOKE_CONFIG", "smoke case has missing or unknown fields")
        case_id = case["id"]
        if not isinstance(case_id, str) or case_id not in REQUIRED_CASE_IDS or case_id in seen:
            raise CaseFailure("SMOKE_CONFIG", f"unknown or duplicate smoke case id: {case_id!r}")
        seen.add(case_id)
        if "title" in case and not isinstance(case["title"], str):
            raise CaseFailure("SMOKE_CONFIG", "case title must be a string")
        workflow = _relative_path(case["workflow"], "workflow path")
        if not workflow.endswith(".json"):
            raise CaseFailure("SMOKE_CONFIG", "workflow must be an API JSON file")
        timeout = case["timeout_minutes"]
        if not _positive_number(timeout) or not _positive_number(timeout * 60):
            raise CaseFailure("SMOKE_CONFIG", "timeout_minutes must be finite and positive")
        expect = case["expect"]
        if not isinstance(expect, dict) or set(expect) != {"width", "height"} or not all(
            _positive_int(value) for value in expect.values()
        ):
            raise CaseFailure("SMOKE_CONFIG", "expect requires positive integer width and height")
        inputs = case["inputs"]
        if not isinstance(inputs, list):
            raise CaseFailure("SMOKE_CONFIG", "case inputs must be a list")
        for name in inputs:
            _input_asset_name(name)
        if len(set(inputs)) != len(inputs):
            raise CaseFailure("SMOKE_CONFIG", "input asset names must be unique")
        overrides = case["overrides"]
        if not isinstance(overrides, dict) or any(
            not isinstance(node_id, str) or not node_id or not isinstance(values, dict)
            for node_id, values in overrides.items()
        ):
            raise CaseFailure("SMOKE_CONFIG", "overrides must map node ids to input mappings")
        _validate_value(overrides)
        for values in overrides.values():
            for key, value in values.items():
                if key in {"width", "height", "batch_size", "steps"} and not _positive_int(value):
                    raise CaseFailure("SMOKE_CONFIG", f"{key} must be a positive integer")
                if key in {"cfg", "strength", "denoise"} and not _positive_number(value):
                    raise CaseFailure("SMOKE_CONFIG", f"{key} must be finite and positive")
                if key == "seed" and value != "$run_seed" and not (
                    type(value) is int and 0 <= value <= MAX_SEED
                ):
                    raise CaseFailure("SMOKE_CONFIG", "seed must be $run_seed or a uint64")
    if len(cases) == 4 and tuple(case["id"] for case in cases) != REQUIRED_CASE_IDS:
        raise CaseFailure("SMOKE_CONFIG", "the full smoke suite must use the required case order")
    return cases


def load_cases(cases_b64: str) -> list[dict]:
    try:
        payload = json.loads(base64.b64decode(cases_b64, validate=True).decode("utf-8"))
        if isinstance(payload, dict):
            if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
                raise CaseFailure("SMOKE_CONFIG", "smoke cases require schema_version: 1")
            if set(payload) != {"schema_version", "cases"}:
                raise CaseFailure("SMOKE_CONFIG", "unknown smoke cases payload fields")
            payload = payload["cases"]
        return validate_cases(payload)
    except (ValueError, TypeError, CaseFailure) as exc:
        C.fail("E_USAGE", f"invalid smoke cases payload: {exc}")


def compute_seed(base_seed: int, run_token: int, case_index: int, unique: bool) -> int:
    if not unique:
        return int(base_seed)
    # The token/clock alone is not entropy. Keep the configured base as an
    # explicit input, but use all 64 random bits rather than a small time bucket.
    return (secrets.randbits(64) + int(base_seed) + int(run_token) + case_index) & MAX_SEED


def build_prompt(
    workflow_path: Path, overrides: dict, seed: int, *, workflow_bytes: bytes | None = None
) -> dict:
    if not workflow_path.is_file():
        raise CaseFailure("SMOKE_WORKFLOW", f"workflow file not found: {workflow_path}")
    try:
        prompt = json.loads(workflow_bytes if workflow_bytes is not None else workflow_path.read_bytes())
    except (ValueError, OSError) as exc:
        raise CaseFailure("SMOKE_WORKFLOW", f"invalid workflow JSON: {workflow_path}: {exc}") from exc
    if not isinstance(prompt, dict) or not prompt or any(
        not isinstance(node, dict)
        or not isinstance(node.get("class_type"), str)
        or not isinstance(node.get("inputs"), dict)
        for node in prompt.values()
    ):
        raise CaseFailure("SMOKE_WORKFLOW", "workflow must be a non-empty ComfyUI API graph")
    _validate_value(prompt)
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


def validate_graph(case: dict, prompt: dict, seed: int, assets_dir: Path) -> dict:
    """Trace the *active* save/sampler/model path, not names or dangling nodes.

    v1 accepts the shipped UNET/CLIP/VAE/KSampler graph and its explicit input
    overrides, not arbitrary equivalent custom-node implementations.
    """
    def reject(message: str) -> None:
        raise CaseFailure("SMOKE_WORKFLOW", f"{case['id']}: {message}")

    def only(class_type: str) -> tuple[str, dict]:
        found = [(key, node) for key, node in prompt.items() if node["class_type"] == class_type]
        if len(found) != 1:
            reject(f"requires exactly one {class_type}")
        return found[0]

    def linked(inputs: dict, key: str, class_type: str, slot: int = 0) -> tuple[str, dict]:
        value = inputs.get(key)
        if not (
            isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and type(value[1]) is int and value[1] == slot
            and value[0] in prompt and prompt[value[0]]["class_type"] == class_type
        ):
            reject(f"{key} must connect to {class_type} output {slot}")
        return value[0], prompt[value[0]]["inputs"]

    output_id, output = only("SaveImage")
    _, decode = linked(output["inputs"], "images", "VAEDecode")
    sampler_id, _ = only("KSampler")
    connected_sampler, sampler = linked(decode, "samples", "KSampler")
    if connected_sampler != sampler_id:
        reject("SaveImage must use the required sampler")
    _, vae = linked(decode, "vae", "VAELoader")
    if vae.get("vae_name") != "qwen_image_vae.safetensors":
        reject("requires the Qwen Image VAE")
    for key in ("positive", "negative"):
        _, conditioning = linked(sampler, key, "CLIPTextEncode")
        _, clip = linked(conditioning, "clip", "CLIPLoader")
        if clip.get("clip_name") != "qwen_3_06b_base.safetensors" or clip.get("device") == "cpu":
            reject("requires the Qwen3 encoder on the default/GPU device")
        if not isinstance(conditioning.get("text"), str) or not conditioning["text"].strip():
            reject(f"{key} text must be non-empty")
    latent_id, latent = linked(sampler, "latent_image", "EmptyLatentImage")
    for key in ("width", "height"):
        if not _positive_int(latent.get(key)) or latent[key] != case["expect"][key]:
            reject(f"latent {key} must match the configured expectation")
    if not _positive_int(latent.get("batch_size")):
        reject("batch_size must be a positive integer")
    if not _positive_int(sampler.get("steps")) or not _positive_number(sampler.get("cfg")):
        reject("steps/CFG must be positive finite parameters")
    if not _positive_number(sampler.get("denoise")) or sampler["denoise"] > 1:
        reject("denoise must be in (0, 1]")
    turbo = case["id"] == "turbo_t2i"
    defaults = {"steps": 8 if turbo else 30, "cfg": 1.0 if turbo else 4.0, "denoise": 1.0}
    for key, default in defaults.items():
        if sampler[key] != case["overrides"].get(sampler_id, {}).get(key, default):
            reject(f"{key} differs from the default without an explicit override")
    if latent["batch_size"] != case["overrides"].get(latent_id, {}).get("batch_size", 1):
        reject("batch_size differs from the default without an explicit override")
    if type(sampler.get("seed")) is not int or sampler["seed"] != seed:
        raise CaseFailure("SMOKE_CACHE", "the active KSampler must use $run_seed, not a fixed seed")

    is_lllite = case["id"] in ("lllite_lineart", "lllite_inpaint")
    if is_lllite:
        plugin_id, _ = only(LLLITE_NODE)
        connected_plugin, apply = linked(sampler, "model", LLLITE_NODE)
        if plugin_id != connected_plugin:
            reject("the sampler must use the LLLite-patched model")
        weight = (
            "anima-lllite-inpainting-v2.safetensors" if case["id"] == "lllite_inpaint"
            else "anima-lllite-any-test-like-v2.safetensors"
        )
        if apply.get("lllite_name") != weight:
            reject(f"requires LLLite weight {weight}")
        if not _positive_number(apply.get("strength")):
            reject("LLLite strength must be finite and positive")
        start, end = apply.get("start_percent"), apply.get("end_percent")
        if type(start) not in (int, float) or not _positive_number(end) or not 0 <= start < end <= 1:
            reject("LLLite must be active for a non-empty sampling interval")
        image_id, image = linked(apply, "image", "LoadImage")
        name = image.get("image")
        if name not in case["inputs"]:
            reject("LLLite must use a declared bundled input image")
        source = _contained_path(assets_dir, name, "input asset path")
        if Image is None:
            raise CaseFailure("SMOKE_PIL_MISSING", "Pillow is required to verify input images/masks")
        try:
            with Image.open(source) as img:
                img.load()
                if img.format != "PNG":
                    reject("the bundled input must be PNG")
                if case["id"] == "lllite_inpaint":
                    mask_id, _ = linked(apply, "mask", "LoadImage", 1)
                    if mask_id != image_id or "A" not in img.getbands():
                        reject("inpainting mask must use the source image's alpha-mask output")
                    low, high = img.getchannel("A").getextrema()
                    if low == high:
                        reject("inpainting requires a non-empty, non-full alpha mask")
        except (OSError, ValueError) as exc:
            raise CaseFailure("SMOKE_ASSET", f"invalid bundled input image: {source}: {exc}") from exc
        _, model = linked(apply, "model", "UNETLoader")
    else:
        if any(node["class_type"] == LLLITE_NODE for node in prompt.values()):
            reject("text-to-image cases must use the unpatched checkpoint")
        _, model = linked(sampler, "model", "UNETLoader")
    checkpoint = (
        "anima-turbo-v1.1.safetensors" if case["id"] == "turbo_t2i"
        else "anima-base-v1.0.safetensors"
    )
    if model.get("unet_name") != checkpoint:
        reject(f"requires checkpoint {checkpoint}")
    return {
        "sampler_node_id": sampler_id,
        "output_node_id": output_id,
        "checkpoint": checkpoint,
        "steps": sampler["steps"],
        "cfg": sampler["cfg"],
        "seed": seed,
        **{key: latent[key] for key in ("width", "height", "batch_size")},
    }


def submit_prompt(base_url: str, prompt: dict, *, client_id: str = SMOKE_CLIENT_ID) -> str:
    try:
        response = CM.http_json(
            f"{base_url}/prompt",
            method="POST",
            body={"prompt": prompt, "client_id": client_id},
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
            {"error": parsed.get("error") if isinstance(parsed, dict) else None,
             "node_errors": parsed.get("node_errors") if isinstance(parsed, dict) else None,
             "submission_rejected": exc.code == 400 and isinstance(parsed, dict)
                                    and bool(parsed.get("error") or parsed.get("node_errors"))},
        ) from exc
    if not isinstance(response, dict) or not isinstance(response.get("prompt_id"), str):
        raise CaseFailure("SMOKE_SUBMIT", "ComfyUI did not return a prompt_id", {"response": response})
    try:
        return C.validate_identifier(response["prompt_id"], "prompt id")
    except ValueError as exc:
        raise CaseFailure("SMOKE_SUBMIT", "ComfyUI returned an unsafe prompt_id") from exc


def wait_for_history(base_url: str, prompt_id: str, timeout_seconds: float) -> dict:
    if not _positive_number(timeout_seconds):
        raise CaseFailure("SMOKE_CONFIG", "history timeout must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    last_status = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        history = CM.http_json(f"{base_url}/history/{prompt_id}", timeout=min(20.0, remaining)) or {}
        if not isinstance(history, dict):
            raise CaseFailure("SMOKE_HISTORY", "history must be a mapping")
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if not isinstance(status, dict):
                raise CaseFailure("SMOKE_HISTORY", "history status must be a mapping")
            last_status = status
            messages = status.get("messages") or []
            if not isinstance(messages, list):
                raise CaseFailure("SMOKE_HISTORY", "execution messages must be a list", {"status": status})
            execution_error = any(
                isinstance(message, (list, tuple)) and len(message) == 2
                and message[0] in ("execution_error", "execution_interrupted")
                for message in messages
            )
            if status.get("status_str") in ("error", "failed", "interrupted") or execution_error:
                raise CaseFailure(
                    "SMOKE_EXEC_ERROR",
                    "ComfyUI reported an execution error for this case",
                    {"status": status, "messages": messages},
                )
            if status.get("completed") is True:
                if status.get("status_str") == "success":
                    return entry
                raise CaseFailure(
                    "SMOKE_HISTORY", "ComfyUI returned an unknown completion status",
                    {"status": status, "messages": messages},
                )
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    raise CaseFailure(
        "SMOKE_TIMEOUT",
        f"case did not finish within {timeout_seconds:.0f}s",
        {
            "prompt_id": prompt_id, "status": last_status,
            "messages": last_status.get("messages") or [],
            "note": "ComfyUI may still be executing; nothing was cancelled.",
        },
    )


def _inference_progress(record: dict, *, state: str, reason: str | None = None, **details) -> None:
    record.update(state=state, updated_at=C.now_epoch(), **details)
    if reason:
        record["reason"] = reason
    C.write_inference(record)  # Commit the safety fence before reporting progress.
    C.write_job_progress({
        "deadline_at": record["deadline_at"],
        "state": "overdue" if record.get("overdue") else "running",
        "reason": record.get("reason"), "intent_id": record["intent_id"],
        "prompt_id": record.get("prompt_id"), "run_id": record["run_id"],
        "case_id": record["case_id"], "persistent_hold": state not in ("terminal", "rejected"),
        "work_state": state, "last_error": record.get("last_error"),
    })


def submit_and_watch(base_url: str, prompt: dict, timeout_seconds: float, *,
                     session_id: str, run_id: str, case_id: str) -> tuple[str, dict]:
    """The background smoke process is the long-lived, FD-holding watcher.

    Observer timeouts/errors are NOT completion. Unknown POST outcomes are never
    retried, even when no prompt ID was returned. The durable fence survives any
    process failure; the job runner also waits for unresolved work before exit.
    """
    C.assert_no_pending_inferences()
    if not _positive_number(timeout_seconds):
        raise CaseFailure("SMOKE_CONFIG", "inference timeout must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    intent_id = secrets.token_hex(16)
    record = {
        "intent_id": intent_id, "session_id": session_id, "run_id": run_id, "case_id": case_id,
        "operation_token": os.environ.get("VAST_COMFY_OPERATION_TOKEN"),
        "job_id": os.environ.get("VAST_COMFY_JOB_ID"), "watcher_pid": os.getpid(),
        "created_at": C.now_epoch(), "client_id": f"{SMOKE_CLIENT_ID}-{intent_id}",
    }
    record["deadline_at"] = C.now_epoch() + timeout_seconds
    _inference_progress(record, state="submitting")
    try:
        prompt_id = submit_prompt(base_url, prompt, client_id=record["client_id"])
    except Exception as exc:
        if isinstance(exc, CaseFailure) and exc.details.get("submission_rejected") is True:
            _inference_progress(record, state="rejected", rejection_confirmed=True, ended_at=C.now_epoch())
            raise
        _inference_progress(record, state="submission_unknown", reason="SMOKE_SUBMIT_UNCERTAIN",
                            overdue=True, last_error=C.redact(str(exc)))
        # No ID means no trustworthy correlation. Do not re-POST, infer completion
        # from an empty queue, or expire this hold. Explicit operator review is
        # required; the controller immediately observes overdue, not an endless wait.
        while True:
            time.sleep(2.0)

    _inference_progress(record, state="running", prompt_id=prompt_id)
    sticky: CaseFailure | None = None
    observer_timeout = max(0.001, deadline - time.monotonic())
    while True:
        try:
            entry = wait_for_history(base_url, prompt_id, observer_timeout)
        except CaseFailure as exc:
            if exc.reason == "SMOKE_EXEC_ERROR":
                _inference_progress(record, state="terminal", ended_at=C.now_epoch(),
                                    terminal_status=exc.details.get("status"))
                if sticky:
                    sticky.details.update(exc.details)
                    sticky.details["inference_terminated"] = True
                    raise sticky
                raise
            sticky = sticky or exc
        except Exception as exc:
            sticky = sticky or CaseFailure("SMOKE_HISTORY_UNCERTAIN", "Cannot establish HTTP inference completion",
                                           {"prompt_id": prompt_id, "last_error": C.redact(str(exc))})
        else:
            if sticky is None and time.monotonic() >= deadline:
                sticky = CaseFailure("SMOKE_TIMEOUT", f"case did not finish within {timeout_seconds:.0f}s",
                                     {"prompt_id": prompt_id})
                record.update(overdue=True, reason=sticky.reason)
            _inference_progress(record, state="terminal", ended_at=C.now_epoch(),
                                terminal_status=entry.get("status"))
            if sticky:
                sticky.details.update(inference_terminated=True, status=entry.get("status") or {},
                                      messages=(entry.get("status") or {}).get("messages") or [])
                raise sticky  # A late success must never turn a timeout into READY.
            return prompt_id, entry
        _inference_progress(record, state="completion_unknown", reason=sticky.reason,
                            overdue=True, last_error=C.redact(sticky.message))
        observer_timeout = 30.0
        time.sleep(2.0)


def reject_cached_samplers(entry: dict, prompt: dict) -> None:
    sampler_ids = {key for key, node in prompt.items() if node["class_type"] == "KSampler"}
    messages = (entry.get("status") or {}).get("messages") or []
    if not isinstance(messages, list):
        raise CaseFailure("SMOKE_HISTORY", "execution messages must be a list")
    for message in messages:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            raise CaseFailure("SMOKE_HISTORY", "invalid execution message")
        event, data = message
        if event != "execution_cached":
            continue
        nodes = data.get("nodes") if isinstance(data, dict) else None
        if not isinstance(nodes, list):
            raise CaseFailure("SMOKE_HISTORY", "execution_cached has no valid node list")
        cached = [str(node_id) for node_id in nodes if str(node_id) in sampler_ids or node_id == "KSampler"]
        if cached:
            raise CaseFailure(
                "SMOKE_CACHE", "ComfyUI reused a cached KSampler; new SaveImage output is not proof of inference",
                {"cached_samplers": cached, "messages": messages},
            )


def collect_images(
    entry: dict, comfy_dir: Path, output_node_ids: set[str] | None = None
) -> list[dict]:
    images: list[dict] = []
    outputs = entry.get("outputs") or {}
    for node_id, out in outputs.items():
        if output_node_ids is not None and str(node_id) not in output_node_ids:
            continue
        for image in (out or {}).get("images", []) or []:
            if image.get("type") != "output":
                continue
            subfolder = image.get("subfolder") or ""
            filename = image.get("filename") or ""
            if not filename:
                continue
            if "/" in filename or "\\" in filename:
                raise CaseFailure("SMOKE_PATH", "output filename must not contain directories")
            relative = f"{subfolder}/{filename}" if subfolder else filename
            output_dir = _contained_path(comfy_dir, "output", "ComfyUI output directory")
            path = _contained_path(output_dir, relative, "output image path")
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
    if Image is None:
        raise CaseFailure(
            "SMOKE_PIL_MISSING",
            "Pillow is unavailable on the instance; cannot decode/verify the PNG",
        )
    if path.suffix.lower() != ".png":
        raise CaseFailure("SMOKE_IMAGE_FORMAT", "the output must be a PNG file", {"path": str(path)})
    try:
        with Image.open(path) as img:
            if img.format != "PNG":
                raise CaseFailure("SMOKE_IMAGE_FORMAT", "the output is not PNG", {"path": str(path)})
            img.load()
            width, height = img.size
            rgb = img.convert("RGB")
            colors = rgb.getcolors(maxcolors=64)
    except CaseFailure:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CaseFailure("SMOKE_IMAGE_DECODE", f"output image cannot be decoded: {exc}") from exc

    if stat.st_mtime < case_start_epoch:
        raise CaseFailure(
            "SMOKE_IMAGE_OLD",
            "the output image predates this run; it is NOT accepted as new evidence",
            {"path": str(path), "mtime": stat.st_mtime, "case_start": case_start_epoch},
        )
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
    input_dir = _contained_path(comfy_dir, "input", "ComfyUI input directory")
    input_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in case_inputs:
        _input_asset_name(name)
        src = _contained_path(assets_dir, name, "bundled input path")
        if not src.is_file():
            raise CaseFailure("SMOKE_ASSET_MISSING", f"bundled test asset not found: {src}")
        dst = _contained_path(input_dir, name, "ComfyUI input path")
        src_sha = C.sha256_file(src)
        if dst.exists() or dst.is_symlink():
            if dst.is_file() and C.sha256_file(dst) == src_sha:
                records.append({"name": name, "action": "already_present", "sha256": src_sha})
                continue
            raise CaseFailure(
                "SMOKE_ASSET_CONFLICT", "an existing ComfyUI input differs; it was NOT overwritten",
                {"name": name, "path": str(dst), "expected_sha256": src_sha},
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation also protects a file appearing after the check.
            with dst.open("xb") as target, src.open("rb") as source:
                shutil.copyfileobj(source, target)
        except FileExistsError as exc:
            raise CaseFailure("SMOKE_ASSET_CONFLICT", f"input appeared during copy; not overwritten: {dst}") from exc
        if C.sha256_file(dst) != src_sha:
            raise CaseFailure("SMOKE_ASSET", f"copied asset checksum mismatch: {name}")
        records.append({"name": name, "action": "copied", "sha256": src_sha})
    return records


def required_nodes_from_workflows(workflows_dir: Path, workflow_names: list[str]) -> list[str]:
    nodes: set[str] = set()
    for name in workflow_names:
        path = _contained_path(workflows_dir, name, "workflow path")
        if not path.is_file():
            raise CaseFailure("SMOKE_WORKFLOW", f"workflow file not found: {path}")
        prompt = build_prompt(path, {}, 0)
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
            "READY requires exactly the four required cases, in order, to pass in this run.",
            "Seeds contain 64 bits of secrets entropy; cached KSamplers and fixed-seed runs are rejected.",
            "Images must be PNG, no older than their case start, the expected size and not single-color.",
            "Queue checks before submission cannot exclude simultaneous submissions outside this tool.",
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
    _from_prepare: bool = False,
) -> tuple[dict, int]:
    started_at = C.now_epoch()
    C.assert_no_pending_inferences()
    run_dir = C.reserve_run_directory(run_id, stage="smoke", from_prepare=_from_prepare)

    def early(result: dict, exit_code: int) -> tuple[dict, int]:
        C.atomic_write_json(run_dir / "summary.json", result)
        return result, exit_code

    # ---- preflight ----------------------------------------------------
    prepared = []
    try:
        validate_cases(cases)  # Direct Python callers must not bypass payload validation.
        if type(base_seed) is not int or not 0 <= base_seed <= MAX_SEED or type(run_token) is not int:
            raise CaseFailure("SMOKE_CONFIG", "base_seed must be uint64 and run_token must be an integer")
        if unique_seed is not True:
            raise CaseFailure(
                "SMOKE_CACHE",
                "fixed-seed smoke is unsupported until safe execution-cache invalidation is implemented; "
                "no inference was submitted and no service was restarted",
            )
        used_seeds = set()
        for index, case in enumerate(cases):
            seed = compute_seed(base_seed, run_token, index, True)
            if seed in used_seeds:
                raise CaseFailure("SMOKE_CACHE", "random seed collision; refusing to reuse a sampler seed")
            used_seeds.add(seed)
            path = _contained_path(workflows_dir, case["workflow"], "workflow path")
            if not path.is_file():
                raise CaseFailure("SMOKE_WORKFLOW", f"workflow file not found: {path}")
            # Hash and parse the same snapshot, then retain it until submission.
            raw = path.read_bytes()
            prompt = build_prompt(path, case["overrides"], seed, workflow_bytes=raw)
            settings = validate_graph(case, prompt, seed, assets_dir)
            prepared.append({
                "prompt": prompt, "seed": seed, "settings": settings,
                "workflow_sha256": hashlib.sha256(raw).hexdigest(),
            })
    except (CaseFailure, OSError) as exc:
        return early({
            "ok": False, "code": "E_CONFIG", "reason": getattr(exc, "reason", "SMOKE_WORKFLOW"),
            "message": str(exc), "details": getattr(exc, "details", {}),
        }, 1)

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
    nodes_checked = sorted({
        node["class_type"] for item in prepared for node in item["prompt"].values()
    })
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
    for case, item in zip(cases, prepared):
        if not overall_ok:
            results.append(
                {
                    "id": case["id"],
                    "status": "not_run",
                    "workflow": case["workflow"], "workflow_sha256": item["workflow_sha256"],
                    "effective_prompt": item["prompt"], "effective_settings": item["settings"],
                    "execution_messages": [],
                    "reason": "previous case failed; the flow stops explicitly and never skips ahead",
                }
            )
            continue
        case_start = C.now_epoch()
        seed = item["seed"]
        prompt = item["prompt"]
        record: dict = {
            "id": case["id"], "seed": seed, "started_at": case_start,
            "workflow": case["workflow"], "workflow_sha256": item["workflow_sha256"],
            "effective_prompt": prompt, "effective_settings": item["settings"],
            "timeout_minutes": case["timeout_minutes"], "expect": case["expect"],
            "execution_messages": [],
        }
        try:
            # A UI upload may have changed an input since suite preflight. Never
            # repair it by overwriting; recheck the actual image/mask being used.
            record["input_assets"] = ensure_input_assets(assets_dir, comfy_dir, case["inputs"])
            validate_graph(case, prompt, seed, comfy_dir / "input")
            queue = CM.queue_snapshot(base_url)
            if queue["busy"]:
                raise CaseFailure(
                    "E_COMFY_BUSY", "ComfyUI became busy; refusing to insert the next smoke case",
                    {"queue": queue},
                )
            timeout_seconds = float(case["timeout_minutes"]) * 60.0
            prompt_id, entry = submit_and_watch(
                base_url, prompt, timeout_seconds, session_id=session_id, run_id=run_id, case_id=case["id"],
            )
            record["prompt_id"] = prompt_id
            record["history_status"] = entry.get("status") or {}
            record["execution_messages"] = record["history_status"].get("messages") or []
            reject_cached_samplers(entry, prompt)
            images = collect_images(entry, comfy_dir, {item["settings"]["output_node_id"]})
            validated = [validate_image(img, case.get("expect", {}), case_start) for img in images]
            if len(validated) != item["settings"]["batch_size"]:
                raise CaseFailure("SMOKE_IMAGE_COUNT", "output count does not match the configured batch_size")
            record.update(
                {
                    "status": "passed",
                    "images": validated,
                    "duration_seconds": round(C.now_epoch() - case_start, 1),
                }
            )
        except CaseFailure as exc:
            overall_ok = False
            if exc.details.get("prompt_id"):
                record["prompt_id"] = exc.details["prompt_id"]
            if "messages" in exc.details:
                record["execution_messages"] = exc.details["messages"]
            if "status" in exc.details:
                record["history_status"] = exc.details["status"]
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
                    "error": {"message": C.redact(str(exc)), "details": {}},
                    "duration_seconds": round(C.now_epoch() - case_start, 1),
                }
            )
        results.append(record)

    # ---- verdict & report ----------------------------------------------
    requested_all = tuple(c["id"] for c in cases) == REQUIRED_CASE_IDS and tuple(
        r["id"] for r in results if r["status"] == "passed"
    ) == REQUIRED_CASE_IDS
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
    became_busy = any(r.get("reason") == "E_COMFY_BUSY" for r in results)
    if became_busy:
        result["code"] = "E_COMFY_BUSY"
    C.atomic_write_json(run_dir / "summary.json", result)
    return result, (13 if became_busy else (0 if verdict in ("READY", "PARTIAL") else 5))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    C.ensure_dirs()
    C.ensure_owns(args.session_id)
    if not os.environ.get("VAST_COMFY_OPERATION_FD"):
        C.fail("E_LOCK_NOT_HELD", "Smoke inference requires an actual inherited operation FD, not only a token.",
               exit_code=4)
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

    run_token = int(args.run_token) if args.run_token else secrets.randbits(64)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + f"{run_token:016x}"

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
    C.run_cli(args.func, args, guard=True)


if __name__ == "__main__":
    main()
