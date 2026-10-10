"""Tests for the confidence gate (auto-post / verify / drop)."""

import pytest

from prism.eval.runner import VerdictStubBackend
from prism.review.engine import apply_confidence_gate, make_finding

AUTO_POST = 0.80
VERIFY = 0.55


def _verifier(**verdicts):
    return VerdictStubBackend(verdicts)


async def _gate(findings, verifier):
    return await apply_confidence_gate(
        findings, verifier, {"a.py": "@@ -1,2 +1,2 @@\n x\n+y\n"}, AUTO_POST, VERIFY
    )


@pytest.mark.asyncio
async def test_auto_post_kept_unchanged():
    f = make_finding(title="Confident issue", confidence=0.95)
    kept = await _gate([f], _verifier())
    assert len(kept) == 1
    assert kept[0].confidence == 0.95  # untouched, no verification call needed


@pytest.mark.asyncio
async def test_verify_upgraded_kept_with_revised_confidence():
    f = make_finding(title="Borderline real issue", confidence=0.65)
    kept = await _gate([f], _verifier(**{"Borderline real issue": (True, 0.72)}))
    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_verify_rejected_dropped():
    f = make_finding(title="Borderline false alarm", confidence=0.65)
    kept = await _gate([f], _verifier(**{"Borderline false alarm": (False, 0.2)}))
    assert kept == []


@pytest.mark.asyncio
async def test_verify_unknown_title_fails_closed():
    f = make_finding(title="Something the stub never heard of", confidence=0.65)
    kept = await _gate([f], _verifier())
    assert kept == []  # unknown -> false_positive -> dropped


@pytest.mark.asyncio
async def test_below_verify_dropped_without_calling_verifier():
    class ExplodingVerifier(_verifier().__class__):
        async def complete(self, *a, **k):
            raise AssertionError("verifier must not be called for low-confidence findings")

    f = make_finding(title="Weak suspicion", confidence=0.30)
    kept = await _gate([f], ExplodingVerifier())
    assert kept == []


@pytest.mark.asyncio
async def test_mixed_batch():
    findings = [
        make_finding(title="Auto post me", confidence=0.9),
        make_finding(title="Verify me ok", confidence=0.7),
        make_finding(title="Verify me bad", confidence=0.6),
        make_finding(title="Drop me", confidence=0.1),
    ]
    verifier = _verifier(**{"Verify me ok": (True, 0.75), "Verify me bad": (False, 0.0)})
    kept = await _gate(findings, verifier)
    assert [f.title for f in kept] == ["Auto post me", "Verify me ok"]
