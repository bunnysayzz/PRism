"""
Review engine: orchestrates the whole PR review.

Pipeline per PR:
  1. Fetch PR metadata (base/head SHAs) + changed files.
  2. Filter files (ignore globs, binary, size caps, max files).
  3. Per file (bounded concurrency): build prompt -> primary backend -> findings.
  4. Confidence gate per finding:
       confidence >= auto_post        -> keep as-is
       verify <= confidence < auto    -> second-pass verify via stronger model
       confidence < verify            -> drop
  5. Validate finding lines against commentable_lines() of the file's patch
     (drop off-diff lines; clamp to the nearest commentable line when close).
  6. Filter by severity threshold + disabled categories.
  7. Dedup against existing bot comments on the PR.
  8. Persist every kept finding to the feedback store.

The engine never posts to GitHub itself — it returns a ReviewResult and the
caller (worker) posts via github.reviews.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prism.config import RepoConfig, Settings, is_category_enabled
from prism.diff.fetcher import (
    ChangedFile,
    enrich_with_context,
    list_pr_files,
    should_review_file,
    to_changed_file,
)
from prism.diff.hunks import commentable_lines
from prism.feedback.store import FeedbackStore
from prism.github import reviews as gh_reviews
from prism.logging import get_logger
from prism.review import prompts
from prism.review.backends import BackendError, ReviewBackend, build_backend
from prism.review.dedup import dedupe_against_existing
from prism.review.schemas import Finding, ReviewResult, Severity

if TYPE_CHECKING:
    from prism.github.client import GitHubClient

log = get_logger(__name__)

VERIFY_SYSTEM_PROMPT = """\
You are a strict verification judge. You are given a code finding from a first-pass
reviewer and the surrounding diff. Answer ONLY with strict JSON:
{"verdict": "real" | "false_positive",
 "revised_confidence": <float 0.0-1.0>,
 "reason": "<one sentence>"}
A finding is "real" only if you can describe the concrete failure it causes.
When in doubt, say "false_positive".
"""


@dataclass
class EngineDeps:
    """Injectable dependencies (tests/eval swap backends without HTTP)."""

    primary: ReviewBackend | None = None
    verifier: ReviewBackend | None = None
    feedback_store: FeedbackStore | None = None


@dataclass
class _FilePlan:
    changed: ChangedFile
    commentable: set[int]


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


async def verify_finding(
    verifier: ReviewBackend,
    finding: Finding,
    hunk_context: str,
) -> tuple[bool, float]:
    """
    Second-pass verification with the stronger model.

    Returns (is_real, revised_confidence). Fail closed: backend errors or
    unparseable verdicts count as "not verified" and the finding is dropped.
    """
    user_prompt = (
        f"File: {finding.path}, line {finding.line}\n"
        f"Category: {finding.category} | Severity: {finding.severity.value} | "
        f"First-pass confidence: {finding.confidence:.2f}\n"
        f"Title: {finding.title}\nExplanation: {finding.explanation}\n"
        f"Suggestion: {finding.suggestion or '(none)'}\n\n"
        f"Diff context:\n```diff\n{hunk_context}\n```\n"
        "Is this a real issue? Respond with strict JSON only."
    )
    try:
        text, _ = await verifier.complete(VERIFY_SYSTEM_PROMPT, user_prompt)
        verdict = json.loads(_strip_fences(text))
        is_real = verdict.get("verdict") == "real"
        revised = float(verdict.get("revised_confidence", 0.0))
        revised = max(0.0, min(1.0, revised))
        log.info(
            "finding_verified" if is_real else "finding_rejected",
            path=finding.path,
            line=finding.line,
            revised_confidence=revised,
        )
        return is_real, revised
    except (BackendError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.warning("verify_failed_closed", error=str(exc)[:200])
        return False, 0.0


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


async def apply_confidence_gate(
    findings: list[Finding],
    verifier: ReviewBackend,
    hunk_contexts: dict[str, str],
    auto_post: float,
    verify_threshold: float,
) -> list[Finding]:
    """
    Route each finding through the confidence gate.

    - >= auto_post: kept, confidence unchanged.
    - [verify_threshold, auto_post): re-asked to the stronger verifier model.
      Kept only if the verifier says "real"; confidence replaced by the
      revised value (must still be >= verify_threshold or it is dropped).
    - < verify_threshold: dropped.
    """
    kept: list[Finding] = []
    for finding in findings:
        if finding.confidence >= auto_post:
            kept.append(finding)
        elif finding.confidence >= verify_threshold:
            context = hunk_contexts.get(finding.path, "")
            is_real, revised = await verify_finding(verifier, finding, context)
            if is_real and revised >= verify_threshold:
                kept.append(finding.model_copy(update={"confidence": revised}))
            else:
                log.info("finding_dropped_by_verifier", path=finding.path, line=finding.line)
        else:
            log.info(
                "finding_dropped_low_confidence",
                path=finding.path,
                line=finding.line,
                confidence=finding.confidence,
            )
    return kept


# ---------------------------------------------------------------------------
# Line validation
# ---------------------------------------------------------------------------


def validate_finding_lines(
    findings: list[Finding], commentable: dict[str, set[int]]
) -> list[Finding]:
    """
    Ensure every finding line is commentable (inside the diff, new-file side).

    - Line in commentable set -> keep.
    - Line within 3 lines of a commentable line -> clamp to the nearest one.
    - Otherwise -> drop (GitHub would reject the comment with 422).
    """
    valid: list[Finding] = []
    for finding in findings:
        lines = commentable.get(finding.path, set())
        if finding.line in lines:
            valid.append(finding)
            continue
        nearest = min(lines, key=lambda ln: abs(ln - finding.line), default=None)
        if nearest is not None and abs(nearest - finding.line) <= 3:
            log.info(
                "finding_line_clamped",
                path=finding.path,
                from_line=finding.line,
                to_line=nearest,
            )
            valid.append(finding.model_copy(update={"line": nearest}))
        else:
            log.info("finding_dropped_off_diff", path=finding.path, line=finding.line)
    return valid


def _meets_severity_threshold(finding: Finding, threshold: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return finding.severity.rank() >= order.get(threshold, 0)


# ---------------------------------------------------------------------------
# Per-file review
# ---------------------------------------------------------------------------


async def _review_one_file(
    plan: _FilePlan,
    primary: ReviewBackend,
    cfg: RepoConfig,
    settings: Settings,
    client: GitHubClient,
    owner: str,
    repo: str,
    base_sha: str,
    head_sha: str,
) -> list[Finding]:
    changed = plan.changed
    # The full patch is the primary diff signal (hunks carry line numbers);
    # extract_hunk_context() is used for targeted verification snippets.

    try:
        await enrich_with_context(client, owner, repo, base_sha, head_sha, changed)
    except Exception as exc:  # noqa: BLE001 — context enrichment is best-effort
        log.warning("context_enrich_failed", path=changed.filename, error=str(exc)[:200])

    user_prompt = prompts.build_user_prompt(
        path=changed.filename,
        language=prompts.guess_language(changed.filename),
        hunk_context=changed.patch or "",
        head_excerpt=changed.head_excerpt,
        base_excerpt=changed.base_excerpt,
        max_findings=int(cfg.get("max_findings_per_file", 10)),
    )
    try:
        findings, usage = await primary.analyze(prompts.SYSTEM_PROMPT, user_prompt)
    except BackendError as exc:
        log.error("file_review_failed", path=changed.filename, error=str(exc)[:200])
        return []
    log.info(
        "file_reviewed",
        path=changed.filename,
        findings=len(findings),
        usage={k: v for k, v in usage.items() if k in ("prompt_tokens", "completion_tokens")},
    )
    # Force the schema path to the actual file (models sometimes echo it wrong).
    return [f.model_copy(update={"path": changed.filename}) for f in findings]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


async def review_pull_request(
    settings: Settings,
    cfg: RepoConfig,
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    deps: EngineDeps | None = None,
) -> ReviewResult:
    """Run the full review pipeline for one PR. Never raises on reviewable PRs."""
    deps = deps or EngineDeps()
    primary = deps.primary or build_backend(settings)
    verifier = deps.verifier or build_backend(settings, for_verification=True)
    feedback = deps.feedback_store or FeedbackStore(settings.feedback_dir)

    pr = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    base_sha: str = pr["base"]["sha"]
    head_sha: str = pr["head"]["sha"]

    raw_files = await list_pr_files(client, owner, repo, pr_number)
    log.info("pr_files_listed", repo=f"{owner}/{repo}", pr=pr_number, count=len(raw_files))

    plans: list[_FilePlan] = []
    skipped: list[str] = []
    ignore_globs: list[str] = cfg.get("ignore_globs", [])
    for entry in raw_files[: settings.max_files_per_pr]:
        changed = to_changed_file(entry)
        patch = changed.patch or ""
        ok, reason = should_review_file(
            changed.filename,
            changed.status,
            changed.patch,
            ignore_globs=ignore_globs,
            max_file_bytes=settings.max_file_bytes,
            patch_bytes=len(patch.encode("utf-8")),
        )
        if not ok:
            skipped.append(f"{changed.filename} ({reason})")
            continue
        plans.append(_FilePlan(changed=changed, commentable=commentable_lines(changed.patch)))
    if len(raw_files) > settings.max_files_per_pr:
        skipped.append(f"... and {len(raw_files) - settings.max_files_per_pr} more files (cap)")

    # Bounded concurrency over files.
    semaphore = asyncio.Semaphore(settings.max_concurrent_files)

    async def _guarded(plan: _FilePlan) -> list[Finding]:
        async with semaphore:
            return await _review_one_file(
                plan, primary, cfg, settings, client, owner, repo, base_sha, head_sha
            )

    per_file = await asyncio.gather(*(_guarded(p) for p in plans))
    findings: list[Finding] = [f for group in per_file for f in group]

    # Confidence gate (needs hunk context per path for the verifier).
    hunk_contexts = {p.changed.filename: (p.changed.patch or "") for p in plans}
    findings = await apply_confidence_gate(
        findings,
        verifier,
        hunk_contexts,
        settings.confidence_auto_post,
        settings.confidence_verify,
    )

    # Line validation, severity threshold, disabled categories.
    findings = validate_finding_lines(findings, {p.changed.filename: p.commentable for p in plans})
    threshold = str(cfg.get("severity_threshold", "low"))
    findings = [f for f in findings if _meets_severity_threshold(f, threshold)]
    findings = [f for f in findings if is_category_enabled(cfg, f.category)]

    # Dedup against comments already posted by the bot.
    existing = await gh_reviews.fetch_existing_review_comments(client, owner, repo, pr_number)
    findings = dedupe_against_existing(findings, existing)

    # Sort: severity desc, then confidence desc.
    findings.sort(key=lambda f: (f.severity.rank(), f.confidence), reverse=True)

    result = ReviewResult(
        findings=findings,
        skipped_files=skipped,
        model=getattr(primary, "model", primary.name),
    )
    feedback.log_review(owner, repo, pr_number, result)
    log.info(
        "pr_review_complete",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(findings),
        skipped=len(skipped),
    )
    return result


def make_finding(
    path: str = "a.py",
    line: int = 1,
    severity: Severity = Severity.medium,
    category: str = "correctness",
    title: str = "Test finding",
    explanation: str = "This is a test finding with enough explanation text.",
    suggestion: str = "",
    confidence: float = 0.9,
) -> Finding:
    """Test helper: build a valid Finding quickly."""
    return Finding(
        path=path,
        line=line,
        severity=severity,
        category=category,
        title=title,
        explanation=explanation,
        suggestion=suggestion,
        confidence=confidence,
    )
