"""Pure controller-side validation for portable identifiers and remote paths."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .errors import ConfigError, UsageError

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
}


def _reserved_windows_name(value: str) -> bool:
    return value.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICES


def validate_identifier(value: str, label: str = "identifier") -> str:
    """Return a Windows/Linux-safe single identifier, or raise ``UsageError``.

    Identifiers have 1–128 ASCII characters, start with an alphanumeric, and
    otherwise contain only alphanumerics, ``_``, ``.`` and ``-``. Windows device
    names (also with extensions) and trailing dots are never accepted.
    """
    if (
        not isinstance(value, str)
        or _IDENTIFIER.fullmatch(value) is None
        or value.endswith(".")
        or _reserved_windows_name(value)
    ):
        raise UsageError(
            f"{label} must be a portable 1-128 character identifier starting with an ASCII "
            "letter or digit, using only letters, digits, '_', '.' and '-' (no Windows device names)."
        )
    return value


def validate_posix_path(value: str, label: str = "path") -> PurePosixPath:
    """Validate an absolute POSIX path *before* pathlib can normalize it.

    Empty, dot and parent components, backslashes, colons and control characters
    are rejected. Only the root itself may end in ``/``. Raises ``ConfigError``.
    """
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or (value != "/" and any(part in ("", ".", "..") for part in value[1:].split("/")))
    ):
        raise ConfigError(f"{label} must be a canonical absolute POSIX path without unsafe components.")
    return PurePosixPath(value)


def validate_path_component(value: str, label: str = "path component") -> str:
    """Validate a portable local filename component, or raise ``ConfigError``."""
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value.endswith((".", " "))
        or any(char in '<>:"/\\|?*' or ord(char) < 32 or ord(char) == 127 for char in value)
        or _reserved_windows_name(value)
    ):
        raise ConfigError(f"{label} is not a portable Windows/Linux filename component.")
    return value
