"""Tests for the review-pipeline upgrades: adversarial verification (item 4),
PR summary (item 9), repo/path instructions (item 5), secret scan (item 10),
and verifier provider rotation (item 4e)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from prism.config import RepoConfig, Settings
from prism.review import engine
from prism.review.backends import (
    BackendError,
    ChainedBackend,
    ReviewBackend,
    build_backend,
)
from prism.review.engine import (
    EngineDeps,
    apply_confidence_gate,
    generate_pr_summary,
    make_finding,
    review_pull_request,
    verify_finding,
)
from prism.review.schemas import Finding, ReviewResult, Severity

AUTO_POST = 0.90
VERIFY = 0.70


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _scrub_proxy_env(monkeypatch: pytest.MonkeyPatch):
    """httpx reads proxy env at client construction; the sandbox's are malformed."""
    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides: Any) -> Settings:
    base = {"confidence_auto_post": AUTO_POST, "confidence_verify": VERIFY}
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _VerdictBackend(ReviewBackend):
    """Verifier double speaking the new adversarial vocabulary."""

    name = "fake-verifier"

    def __init__(self, payload: str | Exception) -> None:
        self._payload = payload
        self.calls = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.calls += 1
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload, {"backend": self.name}


def _verdict(verdict: str, revised: float) -> str:
    return json.dumps({"verdict": verdict, "evidence": "a.py:3", "revised_confidence": revised})


class _ConcurrentVerifier(_VerdictBackend):
    """Tracks peak in-flight verify calls to prove gather concurrency."""

    def __init__(self, payload: str) -> None:
        super().__init__(payload)
        self.in_flight = 0
        self.peak_in_flight = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight -= 1
        return self._payload, {"backend": self.name}  # type: ignore[return-value]


class _FakePrimary(ReviewBackend):
    """Primary double: returns canned findings; answers summary calls too."""

    name = "fake-primary"

    def __init__(
        self, findings: list[Finding] | None = None, summary: str = "Summary text."
    ) -> None:
        self.findings = findings or []
        self.summary = summary
        self.seen_system_prompts: list[str] = []
        self.seen_user_prompts: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.seen_system_prompts.append(system_prompt)
        self.seen_user_prompts.append(user_prompt)
        if "summarizing a pull request" in system_prompt.lower():
            return json.dumps({"summary": self.summary}), {"backend": self.name}
        payload = {"findings": [f.model_dump(mode="json") for f in self.findings]}
        return json.dumps(payload), {"backend": self.name}


class _FakeClient:
    """Minimal GitHub client double for review_pull_request."""

    def __init__(self, files: list[dict[str, Any]]) -> None:
        self._files = files

    async def get(self, url: str) -> dict[str, Any]:
        if url.endswith("/pulls/1"):
            return {"base": {"sha": "base-sha"}, "head": {"sha": "head-sha"}}
        raise RuntimeError(f"unexpected GET {url}")

    async def paginate(self, url: str) -> list[dict[str, Any]]:
        if url.endswith("/pulls/1/files"):
            return self._files
        if url.endswith("/pulls/1/comments"):
            return []
        raise RuntimeError(f"unexpected paginate {url}")


class _NullFeedbackStore:
    def log_review(self, *args: Any) -> list[str]:
        return []


def _file_entry(filename: str, patch: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "status": "modified",
        "additions": 3,
        "deletions": 0,
        "patch": patch,
    }


async def _run_pipeline(
    *,
    files: list[dict[str, Any]],
    primary: ReviewBackend,
    verifier: ReviewBackend,
    cfg: RepoConfig,
    settings: Settings | None = None,
) -> ReviewResult:
    return await review_pull_request(
        settings or _settings(),
        cfg,
        _FakeClient(files),  # type: ignore[arg-type]
        "owner",
        "repo",
        1,
        deps=EngineDeps(primary=primary, verifier=verifier, feedback_store=_NullFeedbackStore()),  # type: ignore[arg-type]
    )


PATCH_A = "@@ -1,2 +1,4 @@\n def f():\n+    x = 1\n+    return x\n     return None\n"


# ---------------------------------------------------------------------------
# Item 4a/4b: adversarial verify_finding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_valid_keeps_with_revised_confidence():
    f = make_finding(title="Borderline", confidence=0.75)
    ok, revised = await verify_finding(_VerdictBackend(_verdict("VALID", 0.82)), f, "ctx")
    assert ok is True
    assert revised == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_verify_invalid_drops():
    f = make_finding(title="Borderline", confidence=0.75)
    ok, revised = await verify_finding(_VerdictBackend(_verdict("INVALID", 0.9)), f, "ctx")
    assert (ok, revised) == (False, 0.0)


@pytest.mark.asyncio
async def test_verify_ambiguous_drops():
    f = make_finding(title="Borderline", confidence=0.75)
    ok, revised = await verify_finding(_VerdictBackend(_verdict("AMBIGUOUS", 0.5)), f, "ctx")
    assert (ok, revised) == (False, 0.0)


@pytest.mark.asyncio
async def test_verify_garbage_json_fails_closed():
    f = make_finding(title="Borderline", confidence=0.75)
    ok, revised = await verify_finding(_VerdictBackend("this is not json {{{"), f, "ctx")
    assert (ok, revised) == (False, 0.0)


@pytest.mark.asyncio
async def test_verify_backend_error_fails_closed():
    f = make_finding(title="Borderline", confidence=0.75)
    ok, revised = await verify_finding(_VerdictBackend(BackendError("boom")), f, "ctx")
    assert (ok, revised) == (False, 0.0)


@pytest.mark.asyncio
async def test_verify_adversarial_prompt_asks_for_invalidation():
    assert "INVALIDATE" in engine.VERIFY_SYSTEM_PROMPT
    assert "VALID" in engine.VERIFY_SYSTEM_PROMPT
    assert "INVALID" in engine.VERIFY_SYSTEM_PROMPT
    assert "AMBIGUOUS" in engine.VERIFY_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Item 4d: batched verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_verifies_concurrently_not_sequentially():
    findings = [make_finding(title=f"Finding {i}", confidence=0.75) for i in range(3)]
    verifier = _ConcurrentVerifier(_verdict("VALID", 0.80))
    kept = await apply_confidence_gate(findings, verifier, {"a.py": "ctx"}, AUTO_POST, VERIFY)
    assert len(kept) == 3
    assert verifier.calls == 3
    assert verifier.peak_in_flight >= 2  # gather overlaps the calls


@pytest.mark.asyncio
async def test_gate_preserves_finding_order_with_gather():
    findings = [
        make_finding(title="First", confidence=0.95),
        make_finding(title="Second", confidence=0.75),
        make_finding(title="Third", confidence=0.75),
    ]
    verifier = _VerdictBackend(_verdict("VALID", 0.80))
    kept = await apply_confidence_gate(findings, verifier, {"a.py": "ctx"}, AUTO_POST, VERIFY)
    assert [f.title for f in kept] == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# Item 4e: verifier provider rotation
# ---------------------------------------------------------------------------


def _two_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PRISM_PROVIDERS",
        json.dumps(
            [
                {
                    "name": "first",
                    "base_url": "https://a.example/v1",
                    "api_key": "k1",
                    "model": "m1",
                },
                {
                    "name": "second",
                    "base_url": "https://b.example/v1",
                    "api_key": "k2",
                    "model": "m2",
                },
            ]
        ),
    )


def test_verifier_rotates_provider_chain(monkeypatch: pytest.MonkeyPatch):
    _two_provider_env(monkeypatch)
    backend = build_backend(object(), for_verification=True)  # settings unused here
    assert isinstance(backend, ChainedBackend)
    assert [b.model for b in backend.backends] == ["m2", "m1"]  # rotated by one


def test_reviewer_chain_keeps_original_order(monkeypatch: pytest.MonkeyPatch):
    _two_provider_env(monkeypatch)
    settings = _settings(review_backend="api")
    backend = build_backend(settings, for_verification=False)
    assert isinstance(backend, ChainedBackend)
    assert [b.model for b in backend.backends] == ["m1", "m2"]  # no rotation


def test_verifier_single_provider_not_rotated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "PRISM_PROVIDERS",
        json.dumps(
            [
                {
                    "name": "only",
                    "base_url": "https://a.example/v1",
                    "api_key": "k1",
                    "model": "m1",
                }
            ]
        ),
    )
    backend = build_backend(object(), for_verification=True)
    assert isinstance(backend, ChainedBackend)
    assert [b.model for b in backend.backends] == ["m1"]


# ---------------------------------------------------------------------------
# Item 9: PR summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_pr_summary_returns_summary():
    class _B(ReviewBackend):
        name = "b"

        async def complete(self, sp: str, up: str) -> tuple[str, dict]:
            assert "summarizing a pull request" in sp.lower()
            return json.dumps({"summary": "Adds retries. Touches net.py."}), {}

    assert (
        await generate_pr_summary(_B(), ["net.py: @@ -1 +1 @@ ..."])
        == "Adds retries. Touches net.py."
    )


@pytest.mark.asyncio
async def test_generate_pr_summary_none_on_backend_error():
    class _B(ReviewBackend):
        name = "b"

        async def complete(self, sp: str, up: str) -> tuple[str, dict]:
            raise BackendError("down")

    assert await generate_pr_summary(_B(), ["net.py: ..."]) is None


@pytest.mark.asyncio
async def test_generate_pr_summary_none_on_unparseable():
    class _B(ReviewBackend):
        name = "b"

        async def complete(self, sp: str, up: str) -> tuple[str, dict]:
            return "not json at all", {}

    assert await generate_pr_summary(_B(), ["net.py: ..."]) is None


@pytest.mark.asyncio
async def test_generate_pr_summary_none_when_no_files():
    class _Exploding(ReviewBackend):
        name = "b"

        async def complete(self, sp: str, up: str) -> tuple[str, dict]:
            raise AssertionError("must not call the model with no files")

    assert await generate_pr_summary(_Exploding(), []) is None


@pytest.mark.asyncio
async def test_review_pull_request_stores_pr_summary():
    primary = _FakePrimary(findings=[], summary="Refactors f for clarity.")
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert result.pr_summary == "Refactors f for clarity."


@pytest.mark.asyncio
async def test_review_pull_request_no_files_no_summary():
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    result = await _run_pipeline(
        files=[],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert result.pr_summary is None
    assert result.findings == []


# ---------------------------------------------------------------------------
# Item 5: repo + path instructions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repo_instructions_appended_to_system_prompt():
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    cfg = RepoConfig({"instructions": "Always flag raw SQL strings."})
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)], primary=primary, verifier=verifier, cfg=cfg
    )
    assert primary.seen_system_prompts, "primary was never called"
    system_prompt = primary.seen_system_prompts[0]
    assert "## Repository review instructions" in system_prompt
    assert "Always flag raw SQL strings." in system_prompt


@pytest.mark.asyncio
async def test_no_instructions_no_extra_section():
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert "## Repository review instructions" not in primary.seen_system_prompts[0]


@pytest.mark.asyncio
async def test_path_instructions_glob_matching():
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    cfg = RepoConfig(
        {
            "path_instructions": {
                "*.py": "Be extra strict about types.",
                "*.js": "Ignore this for python files.",
            }
        }
    )
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)], primary=primary, verifier=verifier, cfg=cfg
    )
    user_prompt = primary.seen_user_prompts[0]
    assert "## Instructions for this file" in user_prompt
    assert "Be extra strict about types." in user_prompt
    assert "Ignore this for python files." not in user_prompt


# ---------------------------------------------------------------------------
# Item 10: secret scan integration (real producer-side scanner)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_finding_survives_to_result():
    patch = '@@ -0,0 +1,3 @@\n+aws_key = "AKIAIOSFODNN7EXAMPLE"\n+x = 1\n+y = 2\n'
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    result = await _run_pipeline(
        files=[_file_entry("a.py", patch)], primary=primary, verifier=verifier, cfg=RepoConfig({})
    )
    secret = [f for f in result.findings if f.title == "Exposed secret: AWS access key ID"]
    assert len(secret) == 1
    assert secret[0].severity == Severity.critical
    assert secret[0].confidence == 1.0
    assert secret[0].category == "security"
    assert secret[0].path == "a.py"
    assert secret[0].line == 1  # correct new-file line number


@pytest.mark.asyncio
async def test_clean_patch_yields_no_secret_findings():
    primary = _FakePrimary(findings=[])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert result.findings == []


# ---------------------------------------------------------------------------
# Item 4c: per-PR confidence overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cfg_confidence_overrides_respected():
    """A 0.92 finding auto-posts under defaults but must be verified when the
    repo raises auto_post to 0.99 (revised confidence replaces the original)."""
    finding = make_finding(title="Borderline", confidence=0.92, line=2)
    verifier = _VerdictBackend(_verdict("VALID", 0.88))

    default_result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=_FakePrimary(findings=[finding]),
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert default_result.findings[0].confidence == pytest.approx(0.92)

    override_result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=_FakePrimary(findings=[finding]),
        verifier=_VerdictBackend(_verdict("VALID", 0.88)),
        cfg=RepoConfig({"confidence_auto_post": 0.99, "confidence_verify": 0.85}),
    )
    assert override_result.findings[0].confidence == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_cfg_verify_override_drops_below_threshold():
    """With verify raised above the finding's confidence, it is dropped
    without ever calling the verifier."""
    finding = make_finding(title="Weak", confidence=0.80, line=2)

    class _Exploding(_VerdictBackend):
        async def complete(self, sp: str, up: str) -> tuple[str, dict]:
            raise AssertionError("verifier must not be called")

    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_A)],
        primary=_FakePrimary(findings=[finding]),
        verifier=_Exploding(_verdict("VALID", 0.9)),
        cfg=RepoConfig({"confidence_verify": 0.85}),
    )
    assert result.findings == []
