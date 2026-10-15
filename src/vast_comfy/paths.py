"""Repository / data-directory discovery.

vast-comfy v1 is used from the repo checkout (documented, editable install).
User-editable data lives at the repo root: ``config/``, ``workflows/``, ``assets/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ConfigError

_MARKER_FILES = ("pyproject.toml", "SPEC.md")
_MARKER_DIRS = ("config", "workflows")


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the vast-comfy repo root.

    Resolution order: ``VAST_COMFY_ROOT`` env var, then walking up from
    ``start`` (default: cwd) and from this file's location.
    """
    env = os.environ.get("VAST_COMFY_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if _looks_like_root(root):
            return root
        raise ConfigError(
            f"VAST_COMFY_ROOT={env!r} does not look like a vast-comfy repo root "
            f"(expected {_MARKER_FILES} and {_MARKER_DIRS})."
        )

    starts: list[Path] = []
    if start is not None:
        starts.append(Path(start))
    starts.append(Path.cwd())
    starts.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    for candidate_start in starts:
        for parent in [candidate_start, *candidate_start.resolve().parents]:
            if parent in seen:
                continue
            seen.add(parent)
            if _looks_like_root(parent):
                return parent
    raise ConfigError(
        "Could not locate the vast-comfy repo root. Run from the repo checkout "
        "or set VAST_COMFY_ROOT."
    )


def _looks_like_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / f).is_file() for f in _MARKER_FILES):
        return False
    return all((path / d).is_dir() for d in _MARKER_DIRS)


def config_dir(root: Path | None = None) -> Path:
    return (root or find_repo_root()) / "config"


def workflows_dir(root: Path | None = None) -> Path:
    return (root or find_repo_root()) / "workflows"


def assets_dir(root: Path | None = None) -> Path:
    return (root or find_repo_root()) / "assets"
