"""Local (control-machine) state persistence.

Stores non-secret bookkeeping only: instance references, session ownership
records, tunnel PIDs, job references. Never stores API keys or private keys.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .errors import VastComfyError, E_INTERNAL


class StateStore:
    _FILES = ("instances", "sessions", "tunnels")

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- low level -------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise VastComfyError(
                code=E_INTERNAL,
                message=f"Local state file {path} is unreadable: {exc}",
                hints=["Inspect or rename the state file, then retry."],
            ) from exc
        if not isinstance(data, dict):
            raise VastComfyError(code=E_INTERNAL, message=f"Local state file {path} has an unexpected shape")
        return data

    def _save(self, name: str, data: dict[str, Any]) -> None:
        path = self._path(name)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)

    # -- generic ---------------------------------------------------------

    def put(self, name: str, key: str, record: dict[str, Any]) -> None:
        data = self._load(name)
        record = dict(record)
        record.setdefault("updated_at", time.time())
        data[str(key)] = record
        self._save(name, data)

    def get(self, name: str, key: str) -> dict[str, Any] | None:
        return self._load(name).get(str(key))

    def remove(self, name: str, key: str) -> None:
        data = self._load(name)
        if str(key) in data:
            del data[str(key)]
            self._save(name, data)

    def all(self, name: str) -> dict[str, Any]:
        return self._load(name)

    # -- convenience -----------------------------------------------------

    def record_instance(self, instance_id: int | str, **fields: Any) -> None:
        existing = self.get("instances", str(instance_id)) or {}
        existing.update(fields)
        self.put("instances", str(instance_id), existing)

    def get_instance(self, instance_id: int | str) -> dict[str, Any] | None:
        return self.get("instances", str(instance_id))

    def record_session(self, session_id: str, **fields: Any) -> None:
        existing = self.get("sessions", session_id) or {}
        existing.update(fields)
        existing["session_id"] = str(session_id)
        self.put("sessions", session_id, existing)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.get("sessions", session_id)

    def list_sessions_for_instance(self, instance_id: int | str) -> list[dict[str, Any]]:
        return [
            rec
            for rec in self.all("sessions").values()
            if str(rec.get("instance_id")) == str(instance_id)
        ]
