"""
Shared pydantic models for the PRism data pipeline.

Record flow:
    RawSample   — one inline review comment joined with its file's unified diff,
                  as collected from the GitHub REST API (collector stage).
    CleanSample — filtered + normalized + language-tagged (cleaner stage).
                  Deduplicated (dedup stage), then split (splits stage).
    TrainingRecord — (system, user, assistant) SFT triple (format stage).
                  The assistant is strict JSON: {"findings": [Finding, ...]}
                  matching ``prism.review.schemas.Finding``.

The optional ``finding`` field on Raw/CleanSample is a handcrafted label
override used by the sample dataset and tests: when present, the format stage
validates it against the Finding schema and uses it directly instead of the
heuristic labeler.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawSample(BaseModel):
    """One inline review comment joined with its file's diff patch."""

    model_config = {"extra": "ignore"}

    sample_id: str = Field(description="Stable id: {repo}#{pr}#{path}#c{comment_id}")
    repo: str = Field(description="'owner/name'")
    pr_number: int
    pr_title: str = ""
    path: str = Field(description="Repo-relative file path")
    patch: str = Field(description="Unified diff for the file")
    comment_id: int
    comment_author: str
    comment_author_type: str = "User"
    comment_body: str
    comment_line: int | None = Field(
        default=None, description="New-file line the comment anchors to (may be null)"
    )
    comment_original_line: int | None = Field(
        default=None, description="Original diff line (for outdated comments)"
    )
    comment_side: str = "RIGHT"
    finding: dict[str, Any] | None = Field(
        default=None,
        description="Handcrafted Finding-schema label override (samples/tests only)",
    )


class CleanSample(BaseModel):
    """Filtered, normalized, language-tagged sample ready for dedup/split/format."""

    model_config = {"extra": "ignore"}

    sample_id: str
    repo: str
    pr_number: int
    pr_title: str = ""
    path: str
    language: str = Field(description="Detected language, e.g. 'python', 'javascript'")
    patch: str
    comment_id: int
    comment_author: str
    comment_body: str
    comment_line: int = Field(description="Resolved new-file anchor line")
    comment_side: str = "RIGHT"
    finding: dict[str, Any] | None = Field(
        default=None,
        description="Handcrafted Finding-schema label override (samples/tests only)",
    )


class TrainingRecord(BaseModel):
    """One SFT example: (system, user, assistant) triple.

    The reviewer comment is the *label source*, not model input — it will not
    exist at inference time, so ``user`` contains only the diff. The
    ``assistant`` is strict JSON: {"findings": [Finding, ...]}.
    """

    model_config = {"extra": "ignore"}

    system: str
    user: str
    assistant: str = Field(description='Strict JSON: {"findings": [Finding, ...]}')
    meta: dict[str, Any] = Field(default_factory=dict)
