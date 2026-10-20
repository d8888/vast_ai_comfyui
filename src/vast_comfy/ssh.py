"""SSH over the Vast.ai proxy (v1: proxy_only), OpenSSH based.

Design points (SPEC.md section 7):

* Only the Vast.ai proxy endpoint (``ssh_host`` / ``ssh_port``) is used.
  The direct-mapped port, if any, is deliberately ignored.
* OpenSSH on Windows/Linux; no PuTTY/PPK.
* A dedicated known_hosts file with TOFU (``StrictHostKeyChecking=accept-new``).
  A changed host key fails the connection; host verification is never disabled.
* ``BatchMode=yes``: keys / ssh-agent only, no interactive prompts.
* Readiness means an actual remote command succeeded, not just an open TCP port.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .errors import (
    E_KEY_MISSING,
    E_SSH_AUTH,
    E_SSH_COMMAND,
    E_SSH_CONNECT,
    E_SSH_HOSTKEY,
    E_SSH_INFO_MISSING,
    VastComfyError,
)
from .logging_setup import get_logger

logger = get_logger("ssh")

_HOSTKEY_PATTERNS = ("host key verification failed", "remote host identification has changed")
_AUTH_PATTERNS = ("permission denied", "too many authentication failures", "no supported authentication methods")
_CONNECT_PATTERNS = (
    "connection refused",
    "connection timed out",
    "no route to host",
    "could not resolve hostname",
    "connection reset",
    "connection closed",
)

# Shell-safe charset for the legacy scp protocol (-O): it hands the remote path
# to the server's shell unquoted, so any other character must never be passed
# to -O (metacharacters could execute commands on the instance).
_LEGACY_SCP_PATH = re.compile(r"[A-Za-z0-9._/+:@%,=-]+\Z")


@dataclass
class SSHContext:
    host: str
    port: int
    user: str = "root"
    key_path: Path | None = None
    known_hosts_path: Path | None = None
    connect_timeout_seconds: int = 15
    command_timeout_seconds: int = 120
    scp_timeout_seconds: int = 3600
    server_alive_interval_seconds: int = 20
    server_alive_count_max: int = 3
    strategy: str = "proxy_only"
    # Ordered extra endpoints tried only when the primary is unreachable at the
    # transport level (E_SSH_CONNECT). Never used for auth/host-key failures.
    fallbacks: list[tuple[str, int]] = field(default_factory=list)

    def endpoints(self) -> list[tuple[str, int]]:
        ordered = [(self.host, self.port)]
        for endpoint in self.fallbacks:
            if endpoint not in ordered:
                ordered.append(endpoint)
        return ordered

    def describe(self) -> dict[str, Any]:
        # Never include key material; paths are not secret.
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "key_path": str(self.key_path) if self.key_path else None,
            "known_hosts_path": str(self.known_hosts_path) if self.known_hosts_path else None,
            "strategy": self.strategy,
            "endpoints": [f"{self.user}@{h}:{p}" for h, p in self.endpoints()],
        }


def direct_endpoint(instance: dict[str, Any]) -> tuple[str, int] | None:
    """Direct-mapped SSH endpoint from the instance's published 22/tcp port.

    Vast.ai publishes a direct port mapping separately from the sshN.vast.ai
    proxy. The proxy is the documented default, but it can be unavailable while
    the direct mapping works; that case is handled by ``ssh.strategy``.
    """
    mapping = (instance.get("ports") or {}).get("22/tcp") or []
    ip = instance.get("public_ipaddr")
    if not ip or not mapping:
        return None
    try:
        host_port = int(mapping[0].get("HostPort"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not 1 <= host_port <= 65535:
        return None
    return (str(ip), host_port)


def context_from_instance(instance: dict[str, Any], cfg: Config) -> SSHContext:
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    strategy = str(cfg.require("ssh.strategy"))
    direct = direct_endpoint(instance)
    # direct_only uses the published 22/tcp mapping; proxy_then_direct prefers
    # the proxy but may fail over. Neither relaxes host-key verification.
    if (not host or not port) and direct is not None and strategy in ("direct_only", "proxy_then_direct"):
        host, port = direct
    if not host or not port:
        raise VastComfyError(
            code=E_SSH_INFO_MISSING,
            message=(
                "Instance has no usable SSH endpoint (no ssh_host/ssh_port and no direct 22/tcp mapping)."
            ),
            stage="ssh",
            details={"instance_id": instance.get("id"), "strategy": strategy},
            hints=[
                "The instance publishes neither an sshN.vast.ai proxy endpoint nor a direct 22/tcp mapping.",
                "Check ssh.strategy ('proxy_only', 'direct_only', 'proxy_then_direct') and recreate the instance "
                "with runtype 'ssh_proxy' if proxy-only SSH (ssh_proxy) is required.",
            ],
        )
    paths = cfg.resolved_paths()
    key_path = paths["key_path"]
    if key_path is not None and not Path(key_path).exists():
        raise VastComfyError(
            code=E_KEY_MISSING,
            message=f"SSH private key not found: {key_path}",
            stage="ssh",
            hints=["Set ssh.key_path in the config (OpenSSH format) or configure ssh-agent."],
        )
    known_hosts = paths["known_hosts_path"]
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    chosen = (str(host), int(port))
    fallbacks: list[tuple[str, int]] = []
    if strategy == "direct_only":
        if direct is None:
            raise VastComfyError(
                code=E_SSH_INFO_MISSING,
                message=(
                    "ssh.strategy is 'direct_only' but the instance publishes no direct 22/tcp mapping."
                ),
                stage="ssh",
                details={"instance_id": instance.get("id")},
                hints=["Switch ssh.strategy to 'proxy_only' or 'proxy_then_direct'."],
            )
        chosen = direct
    elif strategy == "proxy_then_direct" and direct is not None and direct != chosen:
        fallbacks = [direct]
    return SSHContext(
        host=chosen[0],
        port=chosen[1],
        user=str(cfg.require("ssh.user")),
        key_path=Path(key_path) if key_path else None,
        known_hosts_path=known_hosts,
        connect_timeout_seconds=int(cfg.require("ssh.connect_timeout_seconds")),
        command_timeout_seconds=int(cfg.require("ssh.command_timeout_seconds")),
        scp_timeout_seconds=int(cfg.require("ssh.scp_timeout_seconds")),
        server_alive_interval_seconds=int(cfg.require("ssh.server_alive_interval_seconds")),
        server_alive_count_max=int(cfg.require("ssh.server_alive_count_max")),
        strategy=strategy,
        fallbacks=fallbacks,
    )


def _common_opts(ctx: SSHContext) -> list[str]:
    opts = [
        "-o",
        f"UserKnownHostsFile={ctx.known_hosts_path}",
        "-o",
        "StrictHostKeyChecking=accept-new",  # TOFU for the dedicated known_hosts file
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={ctx.connect_timeout_seconds}",
        "-o",
        f"ServerAliveInterval={ctx.server_alive_interval_seconds}",
        "-o",
        f"ServerAliveCountMax={ctx.server_alive_count_max}",
    ]
    if ctx.key_path:
        opts += ["-i", str(ctx.key_path), "-o", "IdentitiesOnly=yes"]
    return opts


def ssh_args_for(ctx: SSHContext, host: str, port: int) -> list[str]:
    """SSH argv for one concrete endpoint (options precede the destination)."""
    return ["ssh", "-p", str(port), *_common_opts(ctx), f"{ctx.user}@{host}"]


def scp_args_for(ctx: SSHContext, host: str, port: int) -> list[str]:
    return ["scp", "-P", str(port), *_common_opts(ctx)]


def ssh_base_args(ctx: SSHContext) -> list[str]:
    return ssh_args_for(ctx, ctx.host, ctx.port)


def scp_base_args(ctx: SSHContext) -> list[str]:
    # Path-sensitive users (Windows scp needs no special flags; -O forces legacy protocol)
    return scp_args_for(ctx, ctx.host, ctx.port)


@dataclass
class CommandResult:
    rc: int
    stdout: str
    stderr: str
    argv: Sequence[str]


def classify_ssh_failure(stderr: str, rc: int) -> str:
    """Map an OpenSSH failure to a stable error code (exported for tests)."""
    lowered = (stderr or "").lower()
    if any(p in lowered for p in _HOSTKEY_PATTERNS):
        return E_SSH_HOSTKEY
    if any(p in lowered for p in _AUTH_PATTERNS):
        return E_SSH_AUTH
    if any(p in lowered for p in _CONNECT_PATTERNS):
        return E_SSH_CONNECT
    return E_SSH_COMMAND


def _public_identity(text: str) -> str | None:
    fields = text.strip().split()
    if len(fields) >= 2 and fields[0].startswith(("ssh-", "ecdsa-", "sk-")):
        return " ".join(fields[:2])
    return None


def rental_preflight(client, cfg: Config) -> dict:
    """Read-only local/key checks that must finish BEFORE paid creation.

    A missing/unusable private key or an unregistered public key must never be
    discovered only after the instance has already been billed.
    """
    for executable in ("ssh", "scp", "ssh-keygen"):
        if not shutil.which(executable):
            raise VastComfyError(
                code=E_SSH_COMMAND,
                message=f"Required local tool is missing: {executable}",
                stage="preflight",
            )
    configured = cfg.resolved_paths()["key_path"]
    if configured is not None and not configured.is_file():
        raise VastComfyError(
            code=E_KEY_MISSING,
            message=f"SSH private key not found: {configured}",
            stage="preflight",
        )
    agent_identities: set[str] = set()
    if shutil.which("ssh-add"):
        try:
            result = subprocess.run(["ssh-add", "-L"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                agent_identities = {i for line in result.stdout.splitlines() if (i := _public_identity(line))}
        except (OSError, subprocess.TimeoutExpired):
            pass
    # With no explicit identity selection, OpenSSH can authenticate entirely
    # through ssh-agent (including hardware-backed keys with no local private file).
    identities: set[str] = set(agent_identities) if configured is None else set()
    candidates = [configured] if configured else [Path.home() / ".ssh" / n for n in ("id_ed25519", "id_rsa")]
    for private in candidates:
        if not private.is_file():
            continue
        try:
            result = subprocess.run(
                ["ssh-keygen", "-y", "-P", "", "-f", str(private)],
                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VastComfyError(
                code=E_SSH_AUTH,
                message="Cannot validate the SSH identity before paying.",
                stage="preflight",
            ) from exc
        public = _public_identity(result.stdout) if result.returncode == 0 else None
        if public:
            identities.add(public)
        else:
            pubfile = Path(str(private) + ".pub")
            supplied = _public_identity(pubfile.read_text(encoding="utf-8")) if pubfile.is_file() else None
            if supplied in agent_identities:
                identities.add(supplied)
            elif configured:
                raise VastComfyError(
                    code=E_SSH_AUTH,
                    message="Private key is locked or invalid; load it in ssh-agent before renting.",
                    stage="preflight",
                )
    if not identities:
        raise VastComfyError(
            code=E_KEY_MISSING,
            message="No usable SSH identity is available; nothing was rented.",
            stage="preflight",
        )
    registered: set[str] = set()
    for key in client.list_ssh_keys():
        for field in ("ssh_key", "public_key", "key"):
            identity = _public_identity(str(key.get(field, "")))
            if identity:
                registered.add(identity)
    if not identities.intersection(registered):
        raise VastComfyError(
            code=E_SSH_AUTH,
            message="No usable SSH public key is registered on this Vast.ai account; nothing was rented.",
            stage="preflight",
            hints=["Register the public key BEFORE creating the instance."],
        )
    return {"ssh_preflight_ok": True, "registered_identity_confirmed": True}


def wait_for_ssh(runner, cfg: Config, *, clock=time.monotonic, sleep=time.sleep):
    """Bounded readiness wait: an actual remote command must succeed."""
    deadline = clock() + cfg.require("launch.ssh_ready_wait_minutes") * 60
    while True:
        try:
            return runner.check_login()
        except VastComfyError as exc:
            if exc.code != E_SSH_CONNECT or clock() >= deadline:
                raise
            sleep(min(cfg.require("launch.poll_interval_seconds"), max(0, deadline - clock())))


class SSHRunner:
    """Runs OpenSSH commands against one instance (proxy-only)."""

    def __init__(self, ctx: SSHContext, *, dry_run: bool = False):
        self.ctx = ctx
        self.dry_run = dry_run
        self._scp_sftp_default: bool | None = None
        self._active_endpoint: tuple[str, int] | None = None

    # -- endpoint selection -----------------------------------------------

    def active_endpoint(self) -> tuple[str, int]:
        """The endpoint that last worked (or the primary one before any attempt)."""
        return self._active_endpoint or (self.ctx.host, self.ctx.port)

    def _try_next(self, code: str, index: int) -> bool:
        """Only transport-level failures may fail over; auth/host-key never do."""
        return code == E_SSH_CONNECT and index + 1 < len(self.ctx.endpoints())

    # -- command execution ------------------------------------------------

    def build_remote_argv(self, argv: Sequence[str], *,
                          endpoint: tuple[str, int] | None = None) -> list[str]:
        host, port = endpoint or self.active_endpoint()
        return [*ssh_args_for(self.ctx, host, port), shlex.join(list(argv))]

    def run_remote(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int | None = None,
        check: bool = True,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        endpoints = self.ctx.endpoints()
        last_error: VastComfyError | None = None
        for index, endpoint in enumerate(endpoints):
            full = self.build_remote_argv(argv, endpoint=endpoint)
            try:
                result = self._execute(
                    full, timeout_seconds=timeout_seconds, check=check, input_bytes=input_bytes
                )
            except VastComfyError as exc:
                if self._try_next(exc.code, index):
                    logger.info("SSH endpoint %s:%s unreachable (%s); trying the next endpoint",
                                endpoint[0], endpoint[1], exc.code)
                    last_error = exc
                    continue
                raise
            self._active_endpoint = endpoint
            if check or result.rc == 0:
                return result
            if self._try_next(classify_ssh_failure(result.stderr, result.rc), index):
                logger.info("SSH endpoint %s:%s unreachable; trying the next endpoint",
                            endpoint[0], endpoint[1])
                continue
            return result
        if last_error is not None:
            raise last_error
        raise VastComfyError(
            code=E_SSH_CONNECT,
            message="No SSH endpoint for this instance could be reached.",
            stage="ssh",
        )

    def run_script(self, script: str, *, timeout_seconds: int | None = None, check: bool = True) -> CommandResult:
        return self.run_remote(["bash", "-lc", script], timeout_seconds=timeout_seconds, check=check)

    def _execute(
        self,
        full_argv: Sequence[str],
        *,
        timeout_seconds: int | None = None,
        check: bool = True,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        timeout = timeout_seconds or self.ctx.command_timeout_seconds
        if self.dry_run:
            logger.info("dry-run: %s", shlex.join(list(full_argv)))
            return CommandResult(rc=0, stdout="", stderr="", argv=list(full_argv))
        try:
            completed = subprocess.run(
                list(full_argv),
                capture_output=True,
                input=input_bytes,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise VastComfyError(
                code=E_SSH_COMMAND,
                message="OpenSSH client ('ssh') was not found on this machine",
                stage="ssh",
                hints=["Install the Windows OpenSSH client or the Linux openssh-client package."],
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VastComfyError(
                code=E_SSH_COMMAND,
                message=f"SSH command timed out after {timeout}s: {shlex.join(list(full_argv))}",
                stage="ssh",
                details={"timeout_seconds": timeout},
                hints=["The remote command may still be running. Check the instance before retrying."],
            ) from exc

        result = CommandResult(
            rc=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
            argv=list(full_argv),
        )
        if check and result.rc != 0:
            code = classify_ssh_failure(result.stderr, result.rc)
            message = f"SSH command failed (rc={result.rc})"
            hints: list[str] = []
            if code == E_SSH_HOSTKEY:
                message = "SSH host key verification failed (known host changed)"
                hints = [
                    "The dedicated known_hosts file recorded a different key for this host.",
                    "Host verification is never disabled automatically; investigate before removing any entry.",
                ]
            elif code == E_SSH_AUTH:
                message = "SSH authentication failed (public key rejected)"
                hints = [
                    "Ensure the public key is registered on the Vast.ai account BEFORE creating instances.",
                    "Existing instances only receive keys that were registered before their creation.",
                ]
            elif code == E_SSH_CONNECT:
                message = "SSH connection could not be established"
                hints = ["The instance may still be starting, stopped, or offline. Check its status and decide explicitly."]
            raise VastComfyError(
                code=code,
                message=message,
                stage="ssh",
                details={
                    "rc": result.rc,
                    "stderr_excerpt": _scrub_tail(result.stderr),
                    "endpoint": f"{self.ctx.user}@{self.ctx.host}:{self.ctx.port}",
                },
                hints=hints,
            )
        return result

    # -- readiness --------------------------------------------------------

    def check_login(self) -> dict[str, Any]:
        """Actual login + command execution proof (not a TCP port check)."""
        marker = "vast-comfy-ssh-ok"
        result = self.run_remote(["echo", marker], check=True)
        if marker not in result.stdout:
            raise VastComfyError(
                code=E_SSH_COMMAND,
                message="SSH login succeeded but the verification marker was not returned",
                stage="ssh",
                details={"stdout_excerpt": _scrub_tail(result.stdout)},
            )
        return {"ssh_ok": True, "endpoint": f"{self.ctx.user}@{self.ctx.host}:{self.ctx.port}"}

    # -- file transfer ----------------------------------------------------

    def upload(self, local_path: Path | str, remote_path: str) -> None:
        self._scp(str(local_path), remote_path, direction="upload")

    def download(self, remote_path: str, local_path: Path | str) -> None:
        self._scp(str(local_path), remote_path, direction="download")

    def _scp(self, local_path: str, remote_path: str, *, direction: str) -> None:
        endpoints = self.ctx.endpoints()
        last_error: VastComfyError | None = None
        for index, endpoint in enumerate(endpoints):
            try:
                self._scp_endpoint(endpoint, local_path, remote_path, direction=direction)
            except VastComfyError as exc:
                if self._try_next(exc.code, index):
                    logger.info("scp endpoint %s:%s unreachable (%s); trying the next endpoint",
                                endpoint[0], endpoint[1], exc.code)
                    last_error = exc
                    continue
                raise
            self._active_endpoint = endpoint
            return
        if last_error is not None:
            raise last_error

    def _scp_endpoint(self, endpoint: tuple[str, int], local_path: str, remote_path: str, *,
                      direction: str) -> None:
        host, port = endpoint
        remote = f"{self.ctx.user}@{host}:{remote_path}"
        operands = [local_path, remote] if direction == "upload" else [remote, local_path]

        def args(legacy: bool) -> list[str]:
            base = scp_args_for(self.ctx, host, port)
            if legacy:
                base.append("-O")
            return [*base, *operands]

        if self.dry_run:
            logger.info("dry-run scp: %s", shlex.join(args(False)))
            return
        # OpenSSH >= 9.0 uses SFTP by default; older scp always speaks the
        # legacy protocol, where the path must survive the remote shell.
        modern = self._scp_uses_sftp_by_default()
        attempts = [False, True] if modern else [False]
        for legacy in attempts:
            if (legacy or not modern) and not _LEGACY_SCP_PATH.fullmatch(remote_path):
                # Legacy scp sends the path to the remote shell unquoted.
                raise VastComfyError(
                    code=E_SSH_COMMAND,
                    message=(
                        "The remote path contains characters that are unsafe for the legacy scp "
                        "protocol, which passes it to the remote shell unquoted; the transfer was refused."
                    ),
                    stage="ssh",
                    details={"direction": direction, "path_excerpt": _scrub_tail(remote_path, 120),
                             "scp_uses_sftp_by_default": modern},
                    hints=[
                        "Rename the file to safe characters, keep the SFTP subsystem enabled, "
                        "or use OpenSSH 9.0+ where scp uses SFTP by default.",
                    ],
                )
            full = args(legacy)
            try:
                completed = subprocess.run(full, capture_output=True, timeout=self.ctx.scp_timeout_seconds)
            except FileNotFoundError as exc:
                raise VastComfyError(
                    code=E_SSH_COMMAND,
                    message="OpenSSH client ('scp') was not found on this machine",
                    stage="ssh",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise VastComfyError(
                    code=E_SSH_COMMAND,
                    message=f"scp {direction} timed out after {self.ctx.scp_timeout_seconds}s",
                    stage="ssh",
                ) from exc
            if completed.returncode == 0:
                return
            stderr = completed.stderr.decode("utf-8", errors="replace")
            # If the modern SFTP-based scp failed and we have not tried the legacy
            # protocol yet, retry once with -O (some sshd configs disable SFTP).
            if modern and not legacy and "sftp" in stderr.lower():
                logger.info("scp failed with %s; retrying with legacy protocol (-O)", _scrub_tail(stderr, 120))
                continue
            code = classify_ssh_failure(stderr, completed.returncode)
            raise VastComfyError(
                code=code if code != E_SSH_COMMAND else E_SSH_COMMAND,
                message=f"File {direction} failed (rc={completed.returncode}): {operands[-1]}",
                stage="ssh",
                details={"stderr_excerpt": _scrub_tail(stderr), "direction": direction},
            )

    def _scp_uses_sftp_by_default(self) -> bool:
        """Whether the local scp speaks SFTP by default (OpenSSH >= 9.0).

        Probes the *same* scp command that will perform the transfer by asking
        for the -O flag, which only the SFTP-capable versions know. An
        unparseable or unexpected answer counts as legacy (fail closed).
        Cached per runner.
        """
        if self._scp_sftp_default is None:
            modern = False
            try:
                result = subprocess.run(["scp", "-O"], capture_output=True, text=True,
                                        timeout=10, stdin=subprocess.DEVNULL)
                text = ((result.stdout or "") + (result.stderr or "")).lower()
                modern = "unknown option" not in text and "usage:" in text
            except (OSError, subprocess.TimeoutExpired):
                modern = False
            self._scp_sftp_default = modern
        return self._scp_sftp_default

    # -- tunnel -----------------------------------------------------------

    def tunnel_argv(self, local_port: int, remote_port: int) -> list[str]:
        """Local-only port forward (bound to 127.0.0.1); options precede the host.

        OpenSSH treats everything after the destination as a remote command, so
        ``-N``/``-L`` must come before ``user@host`` or no tunnel is created.
        """
        host, port = self.active_endpoint()
        return [
            "ssh",
            "-p",
            str(port),
            *_common_opts(self.ctx),
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:localhost:{remote_port}",
            f"{self.ctx.user}@{host}",
        ]


def _scrub_tail(text: str, limit: int = 400) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned[-limit:]
