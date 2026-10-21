"""LoRA training track: pins, plan building, and the Hugging Face publisher.

The LoRA track (as opposed to the inference smoke suite) turns a handful of
generated images into a concept LoRA on the instance's own base checkpoint.

Nothing here talks to an instance. It builds the *plan* (a plain dict that the
remote runner executes) and owns the two things that make a run reproducible:

* ``pins`` records the exact git commit / package versions used, so a later run
  can prove it used the same training stack instead of "whatever was current".
* ``hf`` is the upload client, kept behind a tiny interface so the whole path
  can be exercised with a mock transport and no token.

See ``docs/lora-pipeline.md`` for the operating procedure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .errors import ConfigError, VastComfyError

E_LORA_FAILED = "E_LORA_FAILED"
E_HF_PUSH_FAILED = "E_HF_PUSH_FAILED"

PINNED_SD_SCRIPTS_COMMIT = "11d8cedd0c67b93a10f8b2d75f43554f6f12c794"
PINNED_TORCH = "2.10.0+cu130"
PINNED_TORCHVISION = "0.25.0+cu130"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
PINNED_TRANSFORMERS = "5.17.0"
PINNED_ONNXRUNTIME = "1.31.0"
PINNED_WD14_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
PINNED_WD14_REVISION = "dbd4dbe553ee51feb3bc745b614fb762080e3912"

DEFAULT_STEPS = 500
DEFAULT_DIM = 16
DEFAULT_ALPHA = 8
DEFAULT_LR = 1e-4
DEFAULT_RESOLUTION = 1024
DEFAULT_TAG_THRESHOLD = 0.35


def pins() -> dict[str, Any]:
    """The exact stack this repository knows how to reproduce."""
    return {
        "sd_scripts_commit": PINNED_SD_SCRIPTS_COMMIT,
        "torch": PINNED_TORCH,
        "torchvision": PINNED_TORCHVISION,
        "torch_index_url": PINNED_TORCH_INDEX,
        "transformers": PINNED_TRANSFORMERS,
        "onnxruntime": PINNED_ONNXRUNTIME,
        "wd14_repo": PINNED_WD14_REPO,
        "wd14_revision": PINNED_WD14_REVISION,
    }


@dataclass
class LoraSpec:
    """Everything a training run needs, resolved from the track config."""

    name: str
    base_checkpoint: str
    train_images: int
    steps: int = DEFAULT_STEPS
    dim: int = DEFAULT_DIM
    alpha: int = DEFAULT_ALPHA
    learning_rate: float = DEFAULT_LR
    resolution: int = DEFAULT_RESOLUTION
    tag_threshold: float = DEFAULT_TAG_THRESHOLD
    hf_repo: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ConfigError(f"invalid lora name: {self.name!r}")
        if self.train_images < 1:
            raise ConfigError("train_images must be at least 1")
        if self.steps < 1:
            raise ConfigError("steps must be at least 1")
        if self.dim < 1 or self.alpha < 1:
            raise ConfigError("dim and alpha must be positive")
        if not 0.0 < self.tag_threshold < 1.0:
            raise ConfigError("tag_threshold must be between 0 and 1")


def load_spec(path: Path) -> LoraSpec:
    """Read a track YAML (``config/lora/<name>.yaml``) into a LoraSpec."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"lora track not found: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"lora track must be a mapping: {path}")
    train = raw.get("train") or {}
    spec = LoraSpec(
        name=str(raw.get("name") or path.stem),
        base_checkpoint=str(raw.get("base_checkpoint") or ""),
        train_images=int(train.get("images") or 0),
        steps=int(train.get("steps") or DEFAULT_STEPS),
        dim=int(train.get("network_dim") or DEFAULT_DIM),
        alpha=int(train.get("network_alpha") or DEFAULT_ALPHA),
        learning_rate=float(train.get("learning_rate") or DEFAULT_LR),
        resolution=int(train.get("resolution") or DEFAULT_RESOLUTION),
        tag_threshold=float(raw.get("tag_threshold") or DEFAULT_TAG_THRESHOLD),
        hf_repo=(raw.get("hf_repo") or None),
        extra=dict(raw),
    )
    spec.validate()
    return spec


def build_plan(
    spec: LoraSpec,
    *,
    dataset_dir: str,
    work_dir: str,
    seed: int,
) -> dict[str, Any]:
    """Describe the remote run. Pure data, so it is cheap to test.

    The order matters and the runner enforces it: tag, then train, then verify.
    Verification is part of the plan because a LoRA file that exists but does
    not change the output is a failed run, not a successful one.
    """
    spec.validate()
    return {
        "kind": "lora_train",
        "name": spec.name,
        "pins": pins(),
        "seed": seed,
        "dataset": {
            "dir": dataset_dir,
            "images": spec.train_images,
            "caption_extension": ".txt",
            "tag_threshold": spec.tag_threshold,
            "tagger": PINNED_WD14_REPO,
            "tagger_revision": PINNED_WD14_REVISION,
        },
        "train": {
            "work_dir": work_dir,
            "base_checkpoint": spec.base_checkpoint,
            "output_name": spec.name,
            "steps": spec.steps,
            "network_dim": spec.dim,
            "network_alpha": spec.alpha,
            "learning_rate": spec.learning_rate,
            "resolution": spec.resolution,
            "seed": seed,
            "optimizer": "AdamW8bit",
            "mixed_precision": "fp16",
            "scheduler": "cosine",
            "warmup_steps": min(30, max(1, spec.steps // 10)),
        },
        "verify": {
            # Same seed and prompt, LoRA off then on. A run is only successful
            # when the two images measurably differ.
            "fixed_seed": seed,
            "strength": 1.0,
            "min_mean_abs_diff": 5.0,
            "min_pixel_change_ratio": 0.20,
        },
    }


@dataclass
class PushResult:
    repo_id: str
    path_in_repo: str
    commit_id: str | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "path_in_repo": self.path_in_repo,
            "commit_id": self.commit_id,
            "dry_run": self.dry_run,
        }


class HfPublisher:
    """Minimal Hugging Face uploader with an injectable transport.

    ``transport`` is any callable ``(url, headers, payload) -> (status, body)``.
    Production passes a requests-based one; tests pass a fake and never need a
    token or network. The token is read from the environment at call time and
    never stored on the object, so it cannot leak through a repr or a log.
    """

    def __init__(self, transport=None, *, token_env: str = "HF_TOKEN") -> None:
        self._transport = transport
        self._token_env = token_env

    def _default_transport(self, url: str, headers: dict[str, str], payload: Any):
        import requests  # imported lazily: the train path must not need it

        response = requests.put(url, headers=headers, data=payload, timeout=120)
        return response.status_code, response.text

    def _token(self, dry_run: bool) -> str | None:
        token = os.environ.get(self._token_env)
        if not token and not dry_run:
            raise VastComfyError(
                E_HF_PUSH_FAILED,
                f"{self._token_env} is not set; refusing to push",
                stage="hf_push",
            )
        return token

    def push(
        self,
        *,
        repo_id: str,
        local_path: Path,
        path_in_repo: str,
        commit_message: str = "Add LoRA weights",
        dry_run: bool = False,
    ) -> PushResult:
        if not repo_id or repo_id.count("/") != 1:
            raise VastComfyError(
                E_HF_PUSH_FAILED, f"invalid repo id: {repo_id!r}", stage="hf_push"
            )
        if not local_path.is_file():
            raise VastComfyError(
                E_HF_PUSH_FAILED, f"weights not found: {local_path}", stage="hf_push"
            )
        token = self._token(dry_run)
        if dry_run:
            return PushResult(repo_id=repo_id, path_in_repo=path_in_repo, dry_run=True)

        transport = self._transport or self._default_transport
        url = (
            f"https://huggingface.co/api/models/{repo_id}/upload/"
            f"{path_in_repo.lstrip('/')}"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        payload = local_path.read_bytes()
        status, body = transport(url, headers, payload)
        if status not in (200, 201):
            raise VastComfyError(
                E_HF_PUSH_FAILED,
                f"upload failed ({status}): {str(body)[:200]}",
                stage="hf_push",
            )
        return PushResult(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_id=_extract_commit(body),
        )


def _extract_commit(body: Any) -> str | None:
    import json

    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if not isinstance(body, str):
        return None
    try:
        doc = json.loads(body)
    except (ValueError, TypeError):
        return None
    if isinstance(doc, Mapping):
        for key in ("commitOid", "commit_id", "oid"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    return None


def summarize(plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a remote run report to the fields a reviewer actually checks."""
    verify = result.get("verify") or {}
    return {
        "name": plan.get("name"),
        "images": (plan.get("dataset") or {}).get("images"),
        "steps": (plan.get("train") or {}).get("steps"),
        "saved": bool(result.get("saved")),
        "weights_bytes": result.get("weights_bytes"),
        "mean_abs_diff": verify.get("mean_abs_diff"),
        "pixel_change_ratio": verify.get("pixel_change_ratio"),
        "passed": bool(result.get("passed")),
        "reason": result.get("reason"),
    }
