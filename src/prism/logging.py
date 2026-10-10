"""Structured logging via structlog.

JSON in production (``LOG_FORMAT=json``, the default), human-readable console
output locally (``LOG_FORMAT=console``).

Secret safety: the ``_redact`` processor masks any log kwarg whose key contains
a secret-like substring (see :data:`REDACT_SUBSTRINGS`), recursively into
nested dicts/lists. Never pass a secret as the log *message* itself — the
redactor can only see structured keys.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import EventDict

_INITIALIZED = False

#: Substrings (case-insensitive) that mark a log key as secret-bearing.
#: Shared with ``prism.config.Settings.redacted()``.
REDACT_SUBSTRINGS: frozenset[str] = frozenset(
    {"secret", "token", "key", "password", "credential", "authorization"}
)

#: Replacement written for secret values.
REDACTED = "***REDACTED***"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(sub in lowered for sub in REDACT_SUBSTRINGS)


def _redact_value(value: Any) -> Any:
    """Recursively mask secret-bearing keys inside dicts and lists."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_secret_key(str(k)) else _redact_value(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


def _redact(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict.keys()):
        if _is_secret_key(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def setup_logging(
    level: str = "INFO", *, force: bool = False, log_format: str | None = None
) -> None:
    """Configure structlog. ``log_format``: ``"json"`` (default) or ``"console"``.

    Falls back to the ``LOG_FORMAT`` env var when not passed explicitly, so
    workers can wire it straight from :class:`prism.config.Settings`.
    """
    global _INITIALIZED
    if _INITIALIZED and not force:
        return
    _INITIALIZED = False

    fmt = (log_format or os.environ.get("LOG_FORMAT", "json")).lower()
    renderer: Any = (
        structlog.dev.ConsoleRenderer() if fmt == "console" else structlog.processors.JSONRenderer()
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    structlog.configure(
        processors=[
            _redact,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    _INITIALIZED = True


def get_logger(name: str = "prism") -> Any:
    setup_logging()
    return structlog.get_logger(name)
