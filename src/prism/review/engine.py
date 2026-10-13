"""
Review engine: orchestrates the whole PR review.

Pipeline per PR:
  1. Fetch PR metadata (base/head SHAs) + changed files.
  2. Filter files (ignore globs, binary, size caps, max files).
  3. Per file (bounded concurrency): build prompt -> primary backend -> findings.
  4. Secret scan: deterministic scan of every file's patch for leaked
     credentials; findings enter the pipeline at confidence 1.0.
  5. Confidence gate per finding:
       confidence >= auto_post        -> keep as-is
       verify <= confidence < auto    -> second-pass verify via stronger model
       confidence < verify            -> drop
  6. Validate finding lines against commentable_lines() of the file's patch
     (drop off-diff lines; clamp to the nearest commentable line when close).
  7. Filter by severity threshold + disabled categories.
  8. Dedup against existing bot comments on the PR.
  9. One extra model call: terse PR walkthrough stored on ReviewResult.
  10. Persist every kept finding to the feedback store.

The engine never posts to GitHub itself - it returns a ReviewResult and the
caller (worker) posts via github.reviews.
"""

from __future__ import annotations

import asyncio
import fnmatch
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
from prism.review.secrets import scan_patches

if TYPE_CHECKING:
    from prism.github.client import GitHubClient

log = get_logger(__name__)

VERIFY_SYSTEM_PROMPT = """\
You are an adversarial defense engineer. Your job is to INVALIDATE the finding
by finding counter-evidence in the provided code. You are rewarded for correct
invalidations, not agreement. Return exactly one verdict:
- VALID: the issue is real; cite the triggering lines.
- INVALID: the finding is wrong; cite the specific file:line that disproves it.
- AMBIGUOUS: you cannot decide from the code alone; state the one question a
  human must answer.

Respond with strict JSON only:
{"verdict": "VALID" | "INVALID" | "AMBIGUOUS",
 "evidence": "<cited lines>",
 "revised_confidence": <float 0.0-1.0>}
Do not hedge.
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

    The verifier is an adversarial defense engineer: only a VALID verdict keeps
    the finding (confidence replaced by the revised value). INVALID and
    AMBIGUOUS verdicts drop it. Fail closed: backend errors or unparseable
    verdicts count as "not verified" and the finding is dropped.
    """
    user_prompt = (
        f"File: {finding.path}, line {finding.line}\n"
        f"Category: {finding.category} | Severity: {finding.severity.value} | "
        f"First-pass confidence: {finding.confidence:.2f}\n"
        f"Title: {finding.title}\nExplanation: {finding.explanation}\n"
        f"Suggestion: {finding.suggestion or '(none)'}\n\n"
        f"Diff context:\n```diff\n{hunk_context}\n```\n"
        "Invalidate this finding if you can. Respond with strict JSON only."
    )
    try:
        text, _ = await verifier.complete(VERIFY_SYSTEM_PROMPT, user_prompt)
        verdict = json.loads(_strip_fences(text))
        decision = str(verdict.get("verdict", "")).strip().upper()
        revised = float(verdict.get("revised_confidence", 0.0))
        revised = max(0.0, min(1.0, revised))
        if decision == "VALID":
            log.info(
                "finding_verified",
                path=finding.path,
                line=finding.line,
                revised_confidence=revised,
                evidence=str(verdict.get("evidence", ""))[:200],
            )
            return True, revised
        log.info(
            "finding_rejected",
            path=finding.path,
            line=finding.line,
            verdict=decision or "missing",
        )
        return False, 0.0
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
    - [verify_threshold, auto_post): re-asked to the stronger verifier model
      (batched via asyncio.gather). Kept only if the verifier returns VALID;
      confidence replaced by the revised value (must still be >=
      verify_threshold or it is dropped).
    - < verify_threshold: dropped.
    """
    kept: list[tuple[int, Finding]] = []
    to_verify: list[tuple[int, Finding]] = []
    for idx, finding in enumerate(findings):
        if finding.confidence >= auto_post:
            kept.append((idx, finding))
        elif finding.confidence >= verify_threshold:
            to_verify.append((idx, finding))
        else:
            log.info(
                "finding_dropped_low_confidence",
                path=finding.path,
                line=finding.line,
                confidence=finding.confidence,
            )

    async def _verify_one(finding: Finding) -> Finding | None:
        context = hunk_contexts.get(finding.path, "")
        is_real, revised = await verify_finding(verifier, finding, context)
        if is_real and revised >= verify_threshold:
            return finding.model_copy(update={"confidence": revised})
        log.info("finding_dropped_by_verifier", path=finding.path, line=finding.line)
        return None

    # Batch the verify-band calls concurrently; gather preserves input order.
    verified = await asyncio.gather(*(_verify_one(f) for _, f in to_verify))
    for (idx, _), result in zip(to_verify, verified, strict=True):
        if result is not None:
            kept.append((idx, result))

    kept.sort(key=lambda item: item[0])
    return [finding for _, finding in kept]


# ---------------------------------------------------------------------------
# PR summary
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """\
You are summarizing a pull request for reviewers. Output strict JSON:
{"summary": "<2-4 sentences: what changed and why, file overview>"}
No mermaid, no diagrams, no bullet bloat.
"""


async def generate_pr_summary(
    backend: ReviewBackend,
    file_summaries: list[str],
    max_chars: int = 4000,
) -> str | None:
    """
    One extra model call producing a terse PR walkthrough.

    file_summaries: "path: first 400 chars of patch" lines. Never raises and
    never breaks the review: on backend errors or unparseable output it logs
    and returns None (the summary is polish).
    """
    if not file_summaries:
        return None
    body = "\n".join(file_summaries)[:max_chars]
    try:
        text, _ = await backend.complete(
            SUMMARY_SYSTEM_PROMPT,
            "Summarize this pull request for reviewers.\n\nChanged files:\n" + body,
        )
        data = json.loads(_strip_fences(text))
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            log.warning("pr_summary_unparseable")
            return None
        return summary.strip()
    except (BackendError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.warning("pr_summary_failed", error=str(exc)[:200])
        return None


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


class _FileBackendError(Exception):
    """The model backend failed for one file; distinct from "no findings"."""

    def __init__(self, path: str, error: str) -> None:
        super().__init__(f"{path}: {error}")
        self.path = path
        self.error = error


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
    system_prompt: str | None = None,
) -> list[Finding]:
    changed = plan.changed
    # The full patch is the primary diff signal (hunks carry line numbers);
    # extract_hunk_context() is used for targeted verification snippets.

    try:
        await enrich_with_context(client, owner, repo, base_sha, head_sha, changed)
    except Exception as exc:  # noqa: BLE001 - context enrichment is best-effort
        log.warning("context_enrich_failed", path=changed.filename, error=str(exc)[:200])

    user_prompt = prompts.build_user_prompt(
        path=changed.filename,
        language=prompts.guess_language(changed.filename),
        hunk_context=changed.patch or "",
        head_excerpt=changed.head_excerpt,
        base_excerpt=changed.base_excerpt,
        max_findings=int(cfg.get("max_findings_per_file", 10)),
    )
    # Per-path repo instructions (glob -> text), appended verbatim.
    path_instructions = cfg.get("path_instructions", {}) or {}
    for glob, text in path_instructions.items():
        if fnmatch.fnmatch(changed.filename, glob):
            user_prompt += f"\n\n## Instructions for this file\n{text}"
    try:
        findings, usage = await primary.analyze(system_prompt or prompts.SYSTEM_PROMPT, user_prompt)
    except BackendError as exc:
        # Fail LOUD, never silent: an exception here means the model did not
        # run. Returning [] would post a fake "looks clean" review.
        log.error("file_review_failed", path=changed.filename, error=str(exc)[:200])
        raise _FileBackendError(changed.filename, str(exc)[:200]) from exc
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
    backend_errors: list[str] = []
    failed_paths: set[str] = set()

    # Repo-level review instructions thread into the system prompt; per-path
    # instructions are applied per file inside _review_one_file.
    instructions = str(cfg.get("instructions") or "")
    system_prompt = prompts.SYSTEM_PROMPT + (
        "\n\n## Repository review instructions\n" + instructions if instructions else ""
    )

    async def _guarded(plan: _FilePlan) -> list[Finding]:
        async with semaphore:
            try:
                return await _review_one_file(
                    plan,
                    primary,
                    cfg,
                    settings,
                    client,
                    owner,
                    repo,
                    base_sha,
                    head_sha,
                    system_prompt=system_prompt,
                )
            except _FileBackendError as exc:
                backend_errors.append(f"{exc.path}: {exc.error}")
                failed_paths.add(exc.path)
                return []

    per_file = await asyncio.gather(*(_guarded(p) for p in plans))
    findings: list[Finding] = [f for group in per_file for f in group]

    # Deterministic secret scan over every file's patch, before the
    # confidence gate. Findings carry confidence 1.0 so they auto-post, but
    # still go through line validation and dedup like everything else.
    secret_findings = scan_patches([(p.changed.filename, p.changed.patch or "") for p in plans])
    log.info("secrets_scanned", findings=len(secret_findings))
    findings.extend(secret_findings)

    # Per-PR confidence overrides fall back to the global settings.
    auto_post = float(cfg.get("confidence_auto_post") or settings.confidence_auto_post)
    verify_threshold = float(cfg.get("confidence_verify") or settings.confidence_verify)

    # Confidence gate (needs hunk context per path for the verifier).
    hunk_contexts = {p.changed.filename: (p.changed.patch or "") for p in plans}
    findings = await apply_confidence_gate(
        findings,
        verifier,
        hunk_contexts,
        auto_post,
        verify_threshold,
    )

    # Terse PR walkthrough (polish only): needs at least one successfully
    # reviewed file; a fully failed run is catastrophic, not summarizable.
    file_summaries = [
        f"{p.changed.filename}: {(p.changed.patch or '')[:400]}"
        for p in plans
        if p.changed.filename not in failed_paths
    ]
    pr_summary = await generate_pr_summary(primary, file_summaries) if file_summaries else None

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
        backend_errors=backend_errors,
        pr_summary=pr_summary,
    )
    feedback.log_review(owner, repo, pr_number, result)
    log.info(
        "pr_review_complete",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(findings),
        skipped=len(skipped),
        backend_errors=len(backend_errors),
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
