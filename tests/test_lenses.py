"""Tests for the review-pipeline lenses: test-file scrutiny (item 1),
test-gap detection (item 2), parallel specialist passes (item 3), call-site
context for the verifier (item 4), and ticket context (item 5)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism.config import RepoConfig, Settings
from prism.review import prompts
from prism.review.backends import BackendError, ReviewBackend
from prism.review.engine import (
    EngineDeps,
    _changed_symbols,
    apply_confidence_gate,
    dedupe_cross_lens,
    fetch_call_sites,
    make_finding,
    review_pull_request,
    verify_finding,
)
from prism.review.schemas import ReviewResult, Severity

AUTO_POST = 0.90
VERIFY = 0.70


def _settings(**overrides: Any) -> Settings:
    base = {"confidence_auto_post": AUTO_POST, "confidence_verify": VERIFY}
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _LensAwarePrimary(ReviewBackend):
    """Primary double returning canned findings per pass type (general/lens)."""

    name = "fake-primary"

    def __init__(
        self,
        general: list | None = None,
        security: list | None = None,
        logic: list | None = None,
        test: list | None = None,
        summary: str = "Summary text.",
        compliance: str | None = None,
        fail_lenses: set[str] | None = None,
    ) -> None:
        self._by_lens = {
            "general": general or [],
            "security": security or [],
            "logic": logic or [],
            "test": test or [],
        }
        self.summary = summary
        self.compliance = compliance
        self.fail_lenses = fail_lenses or set()
        self.seen_system_prompts: list[str] = []
        self.seen_user_prompts: list[str] = []

    def _lens_of(self, system_prompt: str) -> str:
        if "summarizing a pull request" in system_prompt.lower():
            return "summary"
        if "reviewing TEST files" in system_prompt:
            return "test"
        if "security-focused reviewer" in system_prompt:
            return "security"
        if "correctness-focused reviewer" in system_prompt:
            return "logic"
        return "general"

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.seen_system_prompts.append(system_prompt)
        self.seen_user_prompts.append(user_prompt)
        lens = self._lens_of(system_prompt)
        if lens == "summary":
            payload: dict[str, Any] = {"summary": self.summary}
            if self.compliance is not None:
                payload["ticket_compliance"] = self.compliance
            return json.dumps(payload), {"backend": self.name}
        if lens in self.fail_lenses:
            raise BackendError(f"{lens} lens exploded")
        findings = self._by_lens[lens]
        return (
            json.dumps({"findings": [f.model_dump(mode="json") for f in findings]}),
            {"backend": self.name},
        )


class _VerdictBackend(ReviewBackend):
    """Verifier double speaking the adversarial vocabulary."""

    name = "fake-verifier"

    def __init__(self, payload: str | Exception) -> None:
        self._payload = payload
        self.calls = 0
        self.seen_user_prompts: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.calls += 1
        self.seen_user_prompts.append(user_prompt)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload, {"backend": self.name}


def _verdict(verdict: str, revised: float) -> str:
    return json.dumps({"verdict": verdict, "evidence": "a.py:3", "revised_confidence": revised})


class _FakeClient:
    """Minimal GitHub client double; optional code-search stub."""

    def __init__(
        self,
        files: list[dict[str, Any]],
        search_payload: dict[str, Any] | Exception | None = None,
    ) -> None:
        self._files = files
        self._search = search_payload
        self.search_calls: list[dict[str, Any]] = []

    async def get(self, url: str, **kwargs: Any) -> Any:
        if url.endswith("/pulls/1"):
            return {"base": {"sha": "base-sha"}, "head": {"sha": "head-sha"}}
        if url == "/search/code":
            self.search_calls.append(kwargs.get("params", {}))
            if isinstance(self._search, Exception):
                raise self._search
            return self._search if self._search is not None else {"items": []}
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
    client: _FakeClient | None = None,
    ticket_context: str | None = None,
) -> ReviewResult:
    return await review_pull_request(
        settings or _settings(),
        cfg,
        client or _FakeClient(files),  # type: ignore[arg-type]
        "owner",
        "repo",
        1,
        deps=EngineDeps(primary=primary, verifier=verifier, feedback_store=_NullFeedbackStore()),  # type: ignore[arg-type]
        ticket_context=ticket_context,
    )


def _finding(**overrides: Any) -> Any:
    base = {
        "path": "a.py",
        "line": 2,
        "severity": Severity.high,
        "category": "correctness",
        "title": "Test finding",
        "explanation": "This is a test finding with enough explanation text.",
        "suggestion": "",
        "confidence": 0.95,
    }
    base.update(overrides)
    return make_finding(**base)


# Small patch (< 200 bytes): general review only.
PATCH_SMALL = "@@ -1,2 +1,4 @@\n def f():\n+    x = 1\n+    return x\n     return None\n"

# Big patch (> 200 bytes): triggers the security + logic lens passes.
PATCH_BIG = (
    "@@ -1,6 +1,12 @@\n"
    " def process(items):\n"
    "+    total = 0\n"
    "+    for item in items:\n"
    "+        total += item.value\n"
    "+    result = compute(total)\n"
    "+    cache.store(result)\n"
    "+    notify_subscribers(result)\n"
    "     return None\n"
    " \n"
    " \n"
    " \n"
)
assert len(PATCH_BIG.encode("utf-8")) > 200


# ---------------------------------------------------------------------------
# Item 1: is_test_file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "test_foo.py",
        "tests/test_foo.py",
        "foo_test.py",
        "src/foo_test.go",
        "tests/unit/a.js",
        "test/a.js",
        "__tests__/a.js",
        "a.spec.ts",
        "src/components/a.spec.js",
        "TEST_FOO.PY",
    ],
)
def test_is_test_file_true(path: str):
    assert prompts.is_test_file(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "src/foo.py",
        "latest.py",
        "contest.py",
        "src/testing.py",
        "attest.py",
        "contest_spec.md",
        "",
    ],
)
def test_is_test_file_false(path: str):
    assert prompts.is_test_file(path) is False


def test_test_lens_prompt_mentions_specified_vs_implementation():
    prompt = prompts.TEST_LENS_SYSTEM_PROMPT
    assert "TEST files" in prompt
    assert "specified" in prompt.lower()
    assert "implementation" in prompt.lower()
    assert "could it pass while the code is wrong" in prompt.lower()


def test_security_and_logic_lens_prompts_scoped():
    assert "Report ONLY" in prompts.SECURITY_LENS_SYSTEM_PROMPT
    assert "SQL" in prompts.SECURITY_LENS_SYSTEM_PROMPT
    assert "Report ONLY" in prompts.LOGIC_LENS_SYSTEM_PROMPT
    assert "off-by-one" in prompts.LOGIC_LENS_SYSTEM_PROMPT
    assert "strict json" in prompts.SECURITY_LENS_SYSTEM_PROMPT.lower()
    assert "strict json" in prompts.LOGIC_LENS_SYSTEM_PROMPT.lower()


@pytest.mark.asyncio
async def test_test_file_gets_general_and_test_lens():
    """Test files run the SAME general review plus the bounded test lens."""
    general_f = _finding(title="General finding", line=2)
    test_f = _finding(
        title="Test asserts implementation detail",
        line=3,
        category="correctness",
        severity=Severity.medium,
    )
    primary = _LensAwarePrimary(general=[general_f], test=[test_f])
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("test_foo.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    kinds = {primary._lens_of(sp) for sp in primary.seen_system_prompts}
    assert "general" in kinds
    assert "test" in kinds
    # No deprioritization: the general review still runs on test files, and
    # the specialist lenses for non-test files do not.
    assert "security" not in kinds
    assert "logic" not in kinds
    assert {f.title for f in result.findings} == {
        "General finding",
        "Test asserts implementation detail",
    }
    # The general pass for the test file still uses the full system prompt.
    general_prompts = [
        sp for sp in primary.seen_system_prompts if primary._lens_of(sp) == "general"
    ]
    assert general_prompts and "senior staff engineer" in general_prompts[0]


@pytest.mark.asyncio
async def test_test_lens_capped_at_three_findings_in_prompt():
    primary = _LensAwarePrimary(test=[_finding(title=f"Test gap {i}", line=2) for i in range(5)])
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[_file_entry("test_foo.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    test_prompts = [
        up
        for sp, up in zip(primary.seen_system_prompts, primary.seen_user_prompts, strict=True)
        if primary._lens_of(sp) == "test"
    ]
    assert test_prompts and "at most 3 findings" in test_prompts[0]


# ---------------------------------------------------------------------------
# Item 2: test-gap detection (prompt-level, conservative)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_tests_note_appended_when_no_test_changes():
    primary = _LensAwarePrimary()
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_SMALL)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert primary.seen_user_prompts, "primary was never called"
    general_prompts = [
        up
        for sp, up in zip(primary.seen_system_prompts, primary.seen_user_prompts, strict=True)
        if primary._lens_of(sp) == "general"
    ]
    assert len(general_prompts) == 1
    note = general_prompts[0]
    assert "no test changes" in note
    assert "at most ONE low-severity" in note
    assert "confidence <= 0.6" in note
    assert "Do not report this if the change is trivial" in note


@pytest.mark.asyncio
async def test_missing_tests_note_absent_when_tests_present():
    primary = _LensAwarePrimary()
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[
            _file_entry("a.py", PATCH_SMALL),
            _file_entry("test_foo.py", PATCH_SMALL),
        ],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert primary.seen_user_prompts
    for up in primary.seen_user_prompts:
        assert "no test changes" not in up


@pytest.mark.asyncio
async def test_missing_tests_note_not_in_lens_prompts():
    primary = _LensAwarePrimary()
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    for sp, up in zip(primary.seen_system_prompts, primary.seen_user_prompts, strict=True):
        lens = primary._lens_of(sp)
        if lens in ("security", "logic"):
            assert "no test changes" not in up
        elif lens == "general":
            assert "no test changes" in up


# ---------------------------------------------------------------------------
# Item 3: parallel specialist passes + cross-lens dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_specialist_lenses_run_for_nontrivial_patch():
    primary = _LensAwarePrimary()
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    kinds = {primary._lens_of(sp) for sp in primary.seen_system_prompts}
    assert {"general", "security", "logic"} <= kinds
    # Lens passes are bounded: capped at 3 findings in the prompt.
    for sp, up in zip(primary.seen_system_prompts, primary.seen_user_prompts, strict=True):
        if primary._lens_of(sp) in ("security", "logic"):
            assert "at most 3 findings" in up


@pytest.mark.asyncio
async def test_small_patch_skips_specialist_lenses():
    primary = _LensAwarePrimary()
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    await _run_pipeline(
        files=[_file_entry("a.py", PATCH_SMALL)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    kinds = {primary._lens_of(sp) for sp in primary.seen_system_prompts}
    assert kinds <= {"general", "summary"}
    assert not ({"security", "logic", "test"} & kinds)


@pytest.mark.asyncio
async def test_cross_lens_dedup_keeps_highest_confidence():
    """Overlapping general + security findings collapse to one (max confidence)."""
    general_f = _finding(
        title="SQL injection via f-string",
        category="security",
        line=2,
        confidence=0.95,
    )
    security_f = _finding(
        title="SQL injection via f-string",  # identical: cross-lens duplicate
        category="security",
        line=2,
        confidence=0.98,
    )
    logic_f = _finding(
        title="Off-by-one in loop bound",
        category="correctness",
        line=4,
        severity=Severity.medium,
        confidence=0.95,
    )
    primary = _LensAwarePrimary(general=[general_f], security=[security_f], logic=[logic_f])
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert len(result.findings) == 2
    dup = next(f for f in result.findings if "SQL injection" in f.title)
    assert dup.confidence == pytest.approx(0.98)
    assert {f.title for f in result.findings} == {
        "SQL injection via f-string",
        "Off-by-one in loop bound",
    }


def test_dedupe_cross_lens_unit():
    a = _finding(title="Same title", line=2, confidence=0.80)
    b = _finding(title="same  TITLE", line=2, confidence=0.95)  # normalized match
    c = _finding(title="Same title", line=3, confidence=0.99)  # different line: kept
    merged = dedupe_cross_lens([a, b, c])
    assert len(merged) == 2
    assert merged[0].confidence == pytest.approx(0.95)
    assert merged[1].line == 3


@pytest.mark.asyncio
async def test_failing_lens_does_not_sink_general_review():
    """A dead lens pass is logged and skipped; the file is not marked failed."""
    primary = _LensAwarePrimary(
        general=[_finding(title="General finding")],
        fail_lenses={"security"},
    )
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_BIG)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert [f.title for f in result.findings] == ["General finding"]
    assert result.backend_errors == []


# ---------------------------------------------------------------------------
# Item 4: call-site context for the verifier
# ---------------------------------------------------------------------------


def test_changed_symbols_extracts_added_top_level_names():
    patch = (
        "@@@ -0,0 +1,12 @@\n"
        "+++ b/x.py\n"
        "+def create_user(name):\n"
        "+    pass\n"
        "+class UserStore:\n"
        "+    pass\n"
        "+const handler = (req) => {};\n"
        "+export function helper() {}\n"
        "+func main() {}\n"
        "+fn helper_fn() {}\n"
        "+    indented_def = 1\n"
    )
    assert _changed_symbols(patch, "x.py") == [
        "create_user",
        "UserStore",
        "handler",
        "helper",
        "main",
    ]  # max 5


def test_changed_symbols_dedupes_and_skips_headers():
    patch = "--- a/x.py\n+++ b/x.py\n+def foo():\n+def foo():\n+    def nested():\n"
    assert _changed_symbols(patch, "x.py") == ["foo"]


def test_changed_symbols_empty_patch():
    assert _changed_symbols("", "x.py") == []
    assert _changed_symbols(None, "x.py") == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_call_sites_success():
    client = _FakeClient([], search_payload={"items": [{"path": "a.py"}, {"path": "b.py"}]})
    result = await fetch_call_sites(client, "owner", "repo", "create_user")  # type: ignore[arg-type]
    assert result == "a.py\nb.py"
    assert client.search_calls == [{"q": "create_user repo:owner/repo", "per_page": 3}]


@pytest.mark.asyncio
async def test_fetch_call_sites_403_skips_gracefully():
    client = _FakeClient([], search_payload=RuntimeError("GitHub 403: rate limited"))
    assert await fetch_call_sites(client, "owner", "repo", "create_user") == ""  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_call_sites_empty_symbol():
    client = _FakeClient([])
    assert await fetch_call_sites(client, "owner", "repo", "") == ""  # type: ignore[arg-type]
    assert client.search_calls == []


@pytest.mark.asyncio
async def test_verify_finding_includes_call_sites_in_prompt():
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    f = _finding(confidence=0.75)
    ok, _ = await verify_finding(verifier, f, "ctx", call_sites="a.py\nb.py")
    assert ok is True
    assert "Known call sites / usages:" in verifier.seen_user_prompts[0]
    assert "a.py" in verifier.seen_user_prompts[0]


@pytest.mark.asyncio
async def test_verify_finding_omits_call_sites_section_when_empty():
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    f = _finding(confidence=0.75)
    await verify_finding(verifier, f, "ctx")
    assert "Known call sites" not in verifier.seen_user_prompts[0]


@pytest.mark.asyncio
async def test_confidence_gate_forwards_call_sites():
    """The fetcher runs for verify-band findings and its output reaches the verifier."""
    seen_titles: list[str] = []

    async def fetcher(f: Any) -> str:
        seen_titles.append(f.title)
        return "callers.py"

    f = _finding(title="Borderline", confidence=0.75, line=2)
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    kept = await apply_confidence_gate(
        [f], verifier, {"a.py": "ctx"}, AUTO_POST, VERIFY, call_site_fetcher=fetcher
    )
    assert len(kept) == 1
    assert seen_titles == ["Borderline"]
    assert "Known call sites / usages:" in verifier.seen_user_prompts[0]
    assert "callers.py" in verifier.seen_user_prompts[0]


@pytest.mark.asyncio
async def test_call_site_fetcher_failure_does_not_break_verification():
    async def bad_fetcher(f: Any) -> str:
        raise RuntimeError("search exploded")

    f = _finding(title="Borderline", confidence=0.75, line=2)
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    kept = await apply_confidence_gate(
        [f], verifier, {"a.py": "ctx"}, AUTO_POST, VERIFY, call_site_fetcher=bad_fetcher
    )
    assert len(kept) == 1  # verified anyway; the search failure was swallowed


@pytest.mark.asyncio
async def test_pipeline_fetches_call_sites_for_verify_band():
    """End to end: a verify-band finding triggers a bounded code-search call."""
    patch_with_symbol = (
        "@@ -0,0 +1,6 @@\n"
        "+def create_user(name):\n"
        "+    return name\n"
        "+x = 1\n"
        "+y = 2\n"
        "+z = 3\n"
        "+w = 4\n"
    )
    client = _FakeClient(
        [_file_entry("a.py", patch_with_symbol)],
        search_payload={"items": [{"path": "callers.py"}]},
    )
    primary = _LensAwarePrimary(general=[_finding(title="Borderline", line=2, confidence=0.75)])
    verifier = _VerdictBackend(_verdict("VALID", 0.8))
    result = await _run_pipeline(
        files=[_file_entry("a.py", patch_with_symbol)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
        client=client,
    )
    assert len(result.findings) == 1
    assert client.search_calls, "expected a code-search call for the changed symbol"
    assert client.search_calls[0]["q"] == "create_user repo:owner/repo"
    assert "Known call sites / usages:" in verifier.seen_user_prompts[0]


# ---------------------------------------------------------------------------
# Item 5: ticket context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticket_context_threads_into_system_prompt_and_summary():
    primary = _LensAwarePrimary(summary="Adds retries.", compliance="partially")
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_SMALL)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
        ticket_context="AC-1: retry failed uploads up to 3 times.",
    )
    system_prompt = primary.seen_system_prompts[0]
    assert "## Linked ticket / acceptance criteria" in system_prompt
    assert "AC-1: retry failed uploads up to 3 times." in system_prompt
    assert "Review the diff against these acceptance criteria" in system_prompt
    assert result.pr_summary is not None
    assert "Adds retries." in result.pr_summary
    assert "Ticket compliance: partially" in result.pr_summary


@pytest.mark.asyncio
async def test_ticket_context_invalid_compliance_normalized():
    primary = _LensAwarePrimary(summary="Adds retries.", compliance="definitely")
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_SMALL)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
        ticket_context="AC-1: something.",
    )
    assert result.pr_summary is not None
    assert "Ticket compliance: n/a" in result.pr_summary


@pytest.mark.asyncio
async def test_ticket_context_none_is_backward_compatible():
    primary = _LensAwarePrimary(summary="Adds retries.")
    verifier = _VerdictBackend(_verdict("VALID", 0.9))
    result = await _run_pipeline(
        files=[_file_entry("a.py", PATCH_SMALL)],
        primary=primary,
        verifier=verifier,
        cfg=RepoConfig({}),
    )
    assert "## Linked ticket" not in primary.seen_system_prompts[0]
    assert result.pr_summary == "Adds retries."
