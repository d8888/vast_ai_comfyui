"""Scripted fake ``SSHRunner`` used by controller-side tests.

Implements the same duck-typed surface as ``vast_comfy.ssh.SSHRunner``:

* ``run_remote`` / ``run_script`` record every call and answer from an ordered
  list of substring rules (each rule returns canned stdout/stderr/rc),
* ``upload`` / ``download`` record transfers,
* ``check_login`` / ``tunnel_argv`` return canned values.

No real SSH or subprocess is ever started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vast_comfy.ssh import CommandResult


@dataclass
class Call:
    kind: str
    argv: list[str]
    timeout_seconds: float | None = None
    check: bool = True

    @property
    def joined(self) -> str:
        return " ".join(self.argv)


@dataclass
class _Rule:
    contains: str
    stdout: str = ""
    stderr: str = ""
    rc: int = 0


@dataclass
class FakeRunner:
    """A deterministic stand-in for SSHRunner."""

    rules: list[_Rule] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    uploads: list[tuple[str, str]] = field(default_factory=list)
    downloads: list[tuple[str, str]] = field(default_factory=list)
    # remote path -> bytes that a real instance would hold (used by sha256sum)
    remote_files: dict[str, bytes] = field(default_factory=dict)
    # remote path -> bytes actually transferred by `download` (may differ to
    # simulate corruption); falls back to remote_files.
    download_bytes: dict[str, bytes] = field(default_factory=dict)
    tunnel: tuple[int, int] | None = None

    # -- scripting ---------------------------------------------------------

    def add_rule(self, contains: str, *, stdout: str = "", stderr: str = "", rc: int = 0) -> None:
        self.rules.append(_Rule(contains=contains, stdout=stdout, stderr=stderr, rc=rc))

    def add_json(self, contains: str, payload: dict[str, Any], *, rc: int = 0) -> None:
        self.add_rule(contains, stdout=json.dumps(payload) + "\n", rc=rc)

    # -- inspection --------------------------------------------------------

    def calls_containing(self, needle: str) -> list[Call]:
        return [call for call in self.calls if needle in call.joined]

    def last_call(self) -> Call:
        assert self.calls, "FakeRunner has not been called"
        return self.calls[-1]

    # -- SSHRunner surface -------------------------------------------------

    def _answer(self, joined: str, argv: Sequence[str]) -> CommandResult:
        for rule in self.rules:
            if rule.contains in joined:
                return CommandResult(
                    rc=rule.rc, stdout=rule.stdout, stderr=rule.stderr, argv=[str(a) for a in argv]
                )
        raise AssertionError(f"FakeRunner has no rule matching: {joined}")

    def run_remote(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int | float | None = None,
        check: bool = True,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        call = Call("run_remote", [str(a) for a in argv], timeout_seconds, check)
        self.calls.append(call)
        return self._answer(call.joined, argv)

    def run_script(
        self,
        script: str,
        *,
        timeout_seconds: int | float | None = None,
        check: bool = True,
    ) -> CommandResult:
        call = Call("run_script", [script], timeout_seconds, check)
        self.calls.append(call)
        return self._answer(script, [script])

    def build_remote_argv(self, argv: Sequence[str]) -> list[str]:
        return ["ssh", "fake", *map(str, argv)]

    def upload(self, local_path: Path | str, remote_path: str) -> None:
        self.uploads.append((str(local_path), remote_path))
        source = Path(local_path)
        if source.is_file():
            self.remote_files.setdefault(remote_path, source.read_bytes())

    def download(self, remote_path: str, local_path: Path | str) -> None:
        self.downloads.append((remote_path, str(local_path)))
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.download_bytes.get(remote_path, self.remote_files.get(remote_path, b"")))

    def check_login(self) -> dict[str, Any]:
        return {"ssh_ok": True, "endpoint": "root@fake:22"}

    def tunnel_argv(self, local_port: int, remote_port: int) -> list[str]:
        return ["ssh", "-L", f"127.0.0.1:{local_port}:localhost:{remote_port}"]
