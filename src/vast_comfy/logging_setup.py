"""Logging helpers: logs go to stderr, JSON results go to stdout.

SPEC.md section 3.2: the machine-readable JSON output must never be polluted by
progress messages, ANSI decoration or logs.
"""

from __future__ import annotations

import logging
import sys

_SECRET_HINTS = ("key", "token", "secret", "password", "authorization", "bearer")


class SecretMaskingFilter(logging.Filter):
    """Best-effort masking of obviously secret-bearing record arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        lowered = msg.lower()
        for hint in _SECRET_HINTS:
            if hint in lowered and "=" in lowered:
                record.msg = "<masked:message-with-secret-like-content>"
                record.args = ()
                break
        return True


def setup_logging(level: str = "INFO", stream=None) -> logging.Logger:
    """Configure the root ``vast_comfy`` logger to write to stderr."""
    logger = logging.getLogger("vast_comfy")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler_stream = stream if stream is not None else sys.stderr
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(handler_stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(SecretMaskingFilter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vast_comfy.{name}")
