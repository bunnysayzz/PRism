"""Tests for review prompt hardening (SYSTEM_PROMPT sections)."""

import re

from prism.review.prompts import SYSTEM_PROMPT

# The OUTPUT FORMAT section must stay byte-identical; any edit to it breaks
# the strict-JSON contract the engine depends on.
EXPECTED_OUTPUT_FORMAT_BLOCK = """\
OUTPUT FORMAT - strict JSON only, no prose, no markdown fences:
{
  "findings": [
    {
      "path": "<file path as given>",
      "line": <new-file line number, integer>,
      "severity": "critical" | "high" | "medium" | "low",
      "category": "security" | "correctness" | "concurrency" | "performance" | "resource" | "api",
      "title": "<short title>",
      "explanation": "<why this is a problem, with the concrete failure scenario>",
      "suggestion": "<minimal fixed code snippet, or empty string>",
      "confidence": <float 0.0-1.0>,
      "language_hint": "<code fence language, e.g. python>"
    }
  ]
}
If there are no real issues, return {"findings": []}."""


def test_output_format_block_byte_identical():
    assert EXPECTED_OUTPUT_FORMAT_BLOCK in SYSTEM_PROMPT


def _flat() -> str:
    """SYSTEM_PROMPT with all whitespace runs collapsed (prompt wraps lines)."""
    return re.sub(r"\s+", " ", SYSTEM_PROMPT)


def test_evidence_requirement_section_present():
    flat = _flat()
    assert "EVIDENCE REQUIREMENT" in flat
    assert "cite verifiable evidence" in flat
    assert "the exact lines that trigger the bug" in flat
    assert "a caller that supplies the bad input" in flat
    assert "the contract being violated" in flat
    assert "If you cannot point to the evidence, do not report the finding." in flat


def test_exclusion_rules_section_present():
    flat = _flat()
    assert "EXCLUSION RULES" in flat
    # Rule 1: only diff-introduced issues.
    assert "introduced or made worse by THIS diff" in flat
    assert "exists identically in the base version" in flat
    # Rule 2: linters own style.
    assert "linter, formatter, or type-checker" in flat
    assert "unused imports" in flat
    # Rule 3: established patterns are intentional.
    assert "pattern visible elsewhere in the file or repo" in flat
    assert "demonstrate a concrete failure it causes" in flat


def test_not_findings_negative_examples_present():
    flat = _flat()
    assert "NOT FINDINGS" in flat
    # Optional-chaining anti-example.
    assert "a?.b.c" in flat
    assert "short-circuits" in flat
    # Rules-of-Hooks anti-example.
    assert "Rules of Hooks" in flat
    assert "early return" in flat
    # Refactoring-opinion anti-example.
    assert "consider extracting this into a helper" in flat
    assert "never a finding" in flat


def test_system_prompt_has_no_emoji():
    # Emoji-ish ranges: emoticons, symbols, pictographs, dingbats, flags.
    emoji_re = re.compile(
        "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff"
        "\U0000fe00-\U0000fe0f\U0001f1e6-\U0001f1ff]"
    )
    assert not emoji_re.search(SYSTEM_PROMPT), "emoji found in SYSTEM_PROMPT"


def test_system_prompt_has_no_model_branding():
    lowered = SYSTEM_PROMPT.lower()
    for brand in ("gpt", "claude", "llama", "mistral", "gemini", "openai", "anthropic"):
        assert brand not in lowered, f"model branding '{brand}' in SYSTEM_PROMPT"
