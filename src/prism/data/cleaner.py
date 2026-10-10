"""
Cleaning stage: turn raw collected samples into training-ready records.

Filtering (each drop is counted by reason in the returned Counter):
  - bot comments (``[bot]`` logins, ``type == "Bot"``, known bot accounts),
  - comments not anchored to a diff line (no ``line``/``original_line``),
  - huge diffs (``patch`` longer than ``max_patch_chars``),
  - comment length guards (too short to be informative / too long to be a review),
  - files whose language cannot be detected from the filename,
  - empty patches.

Normalization:
  - unicode NFKC, strip zero-width / control characters,
  - secret-looking strings redacted to ``[REDACTED]`` (API keys, tokens,
    private keys, ``password = "..."`` assignments, ...),
  - language detection from the filename (extension map + known filenames).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from prism.data.models import CleanSample, RawSample
from prism.logging import get_logger

log = get_logger(__name__)

REDACTED = "[REDACTED]"

EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".r": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".tf": "terraform",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "rst",
    ".dockerfile": "dockerfile",
}

FILENAME_TO_LANGUAGE: dict[str, str] = {
    "makefile": "make",
    "gnumakefile": "make",
    "dockerfile": "dockerfile",
    "cmakelists.txt": "cmake",
    "rakefile": "ruby",
    "gemfile": "ruby",
    "vagrantfile": "ruby",
}

KNOWN_BOTS: frozenset[str] = frozenset(
    {
        "dependabot",
        "dependabot-preview",
        "renovate",
        "github-actions",
        "codecov",
        "codecov-io",
        "sonarcloud",
        "snyk-bot",
        "deepsource",
        "codacy",
        "lgtm-com",
        "allcontributors",
        "stale",
        "mergify",
        "bors",
    }
)


@dataclass
class CleanerConfig:
    max_patch_chars: int = 30_000
    min_comment_chars: int = 20
    max_comment_chars: int = 8_000
    drop_unknown_language: bool = True


def is_bot(author_login: str, author_type: str) -> bool:
    """True for GitHub Apps / bot accounts whose comments are not human reviews."""
    login = author_login.strip().lower()
    if not login:
        return True  # no author: cannot trust it as a human review
    if author_type.strip().lower() == "bot":
        return True
    if login.endswith("[bot]"):
        return True
    return login in KNOWN_BOTS


def detect_language(path: str) -> str:
    """Detect the code language from a repo-relative file path; "" if unknown."""
    name = path.rsplit("/", 1)[-1]
    lowered = name.lower()
    if lowered in FILENAME_TO_LANGUAGE:
        return FILENAME_TO_LANGUAGE[lowered]
    if "." in name:
        ext = "." + name.rsplit(".", 1)[-1].lower()
        return EXT_TO_LANGUAGE.get(ext, "")
    return ""


_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def normalize_text(text: str) -> str:
    """
    Unicode-normalize text for the dataset.

    NFKC folding, strip zero-width characters and control characters
    (keeping newlines/tabs so code structure survives).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return "".join(
        ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


# (pattern, replacement) — applied in order; all case-insensitive.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # GitHub / Stripe / AWS / Slack / Google tokens with recognizable prefixes.
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), REDACTED),
    (re.compile(r"\bsk-(live|test)-[A-Za-z0-9]{8,}\b"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b"), REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), REDACTED),
    # PEM private key blocks (may span lines in a patch).
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), REDACTED),
    # Named secret assignments: password = "...", api_key: '...', etc.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|secret|client[_-]?secret|password|passwd|pwd"
            r"|token|auth[_-]?token|access[_-]?token|private[_-]?key|deploy[_-]?key)"
            r"\s*[:=]\s*(['\"])[^'\"]{4,}\2"
        ),
        r"\1=[REDACTED]",
    ),
]


def redact_secrets(text: str) -> str:
    """Replace secret-looking strings with ``[REDACTED]``."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _resolve_line(raw: RawSample) -> int | None:
    if raw.comment_line is not None:
        return raw.comment_line
    return raw.comment_original_line


def clean_sample(raw: RawSample, config: CleanerConfig) -> tuple[CleanSample | None, str]:
    """
    Clean one raw sample.

    Returns ``(sample, "")`` on success, ``(None, reason)`` when dropped.
    """
    if is_bot(raw.comment_author, raw.comment_author_type):
        return None, "bot"
    line = _resolve_line(raw)
    if line is None or line <= 0:
        return None, "unanchored"
    patch = normalize_text(raw.patch)
    if not patch.strip():
        return None, "empty_patch"
    if len(patch) > config.max_patch_chars:
        return None, "huge_diff"
    body = normalize_text(raw.comment_body).strip()
    if len(body) < config.min_comment_chars:
        return None, "comment_too_short"
    if len(body) > config.max_comment_chars:
        return None, "comment_too_long"
    language = detect_language(raw.path)
    if not language and config.drop_unknown_language:
        return None, "unknown_language"
    return (
        CleanSample(
            sample_id=raw.sample_id,
            repo=raw.repo,
            pr_number=raw.pr_number,
            pr_title=raw.pr_title,
            path=raw.path,
            language=language,
            patch=redact_secrets(patch),
            comment_id=raw.comment_id,
            comment_author=raw.comment_author,
            comment_body=redact_secrets(body),
            comment_line=line,
            comment_side=raw.comment_side,
            finding=raw.finding,
        ),
        "",
    )


def clean_all(
    raws: list[RawSample], config: CleanerConfig | None = None
) -> tuple[list[CleanSample], Counter[str]]:
    """
    Clean a batch of raw samples.

    Returns ``(kept, drop_reasons)`` where ``drop_reasons`` counts samples
    dropped per reason (plus ``"kept"`` for survivors).
    """
    cfg = config or CleanerConfig()
    kept: list[CleanSample] = []
    reasons: Counter[str] = Counter()
    for raw in raws:
        sample, reason = clean_sample(raw, cfg)
        if sample is None:
            reasons[reason] += 1
        else:
            reasons["kept"] += 1
            kept.append(sample)
    log.info(
        "cleaning_done",
        kept=len(kept),
        dropped=sum(v for k, v in reasons.items() if k != "kept"),
        reasons=dict(reasons),
    )
    return kept, reasons


def describe_drop_reasons(reasons: Counter[str]) -> dict[str, Any]:
    """JSON-serializable summary of a cleaning run."""
    total = sum(reasons.values())
    return {"total": total, "reasons": dict(reasons)}
