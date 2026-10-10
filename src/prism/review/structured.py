"""
Structured-output serving for review backends.

ConstrainedReviewBackend wraps any ReviewBackend and enforces schema-valid
JSON output:

- The Finding JSON Schema is injected into the system prompt
  (schema-guided generation).
- Model output is validated per-finding; validation errors feed an
  escalating repair prompt with bounded retries (default 2).
- Confidence is clamped to [0, 1] and quantized; blank findings are dropped.
- Usage dicts are enriched with token counts and estimated cost when a
  price table is provided.
- Optional temperature/top_p are plumbed through to inner backends whose
  complete() accepts them as kwargs; otherwise they are recorded as
  metadata only.

Exhaustion policy (documented choice): best-effort by default. If retries
are exhausted and output is still invalid, analyze() returns whatever
validated (possibly []) instead of raising. Rationale: the review engine is
designed to never raise on reviewable PRs, so a total schema failure
degrades to "no findings" with a structured error log rather than killing
the whole PR review. Pass strict=True for fail-fast semantics (raises
BackendError) — useful in eval harnesses where silent degradation would
hide regressions.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from pydantic import ValidationError

from prism.logging import get_logger
from prism.review.backends import BackendError, ReviewBackend, _OpenAICompatibleBackend
from prism.review.schemas import Finding, _extract_json

log = get_logger(__name__)

_FINDING_SCHEMA_JSON = json.dumps(Finding.model_json_schema(), indent=1)

_SCHEMA_BLOCK = (
    "\n\nSTRICT OUTPUT SCHEMA — your entire response must be one JSON object "
    '{"findings": [...]}. Every finding must validate against this JSON Schema '
    "(no extra top-level keys):\n" + _FINDING_SCHEMA_JSON
)

# For backends without native JSON mode (no response_format support), the
# schema instruction alone is not enough — demand bare JSON explicitly.
_PROMPT_ONLY_SUFFIX = (
    "\nRespond with ONLY the JSON object. No prose before or after it, "
    "no markdown fences, no apologies."
)

_MAX_ERRORS_IN_REPAIR = 8
_BAD_OUTPUT_TRUNCATION = 4000


def _accepts_sampling_kwargs(fn: Callable[..., Any]) -> bool:
    """True if fn accepts temperature/top_p keyword arguments."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    has_named = "temperature" in params and "top_p" in params
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return has_named or has_varkw


def _as_int(value: Any) -> int:
    """Best-effort int coercion for token counts; garbage -> 0."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class ConstrainedReviewBackend(ReviewBackend):
    """
    Decorator around any ReviewBackend enforcing schema-valid output.

    Args:
        inner: the wrapped backend (vLLM, HF endpoint, API, stub, ...).
        max_retries: repair retries after a validation failure (default 2).
            Total completion calls are bounded by 1 + max_retries.
        retry_on_empty: also retry when zero findings parse cleanly. Default
            False — silence is a legitimate review outcome; enable only when
            the caller has independent signal that findings exist (e.g. an
            eval harness with seeded bugs).
        strict: if True, raise BackendError when retries are exhausted and
            output is still invalid. If False (default), return best-effort
            findings (possibly []) — see module docstring for rationale.
        temperature/top_p: forwarded to inner.complete() as kwargs when the
            inner backend accepts them; otherwise recorded in usage metadata.
        price_table: {model_name: (usd_per_1k_input_tokens,
            usd_per_1k_output_tokens)} for cost estimation. No default table
            is shipped — pass your provider's current pricing, e.g.
            {"gpt-4o": (0.005, 0.015)}.
    """

    def __init__(
        self,
        inner: ReviewBackend,
        *,
        max_retries: int = 2,
        retry_on_empty: bool = False,
        strict: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        price_table: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")
        self.inner = inner
        self.name = f"constrained:{inner.name}"
        # Delegated for ReviewResult.model — nicer than "constrained:vllm".
        self.model = str(getattr(inner, "model", inner.name))
        self._max_retries = max(0, int(max_retries))
        self._retry_on_empty = retry_on_empty
        self._strict = strict
        self._sampling: dict[str, float] = {}
        if temperature is not None:
            self._sampling["temperature"] = temperature
        if top_p is not None:
            self._sampling["top_p"] = top_p
        self._sampling_supported = bool(self._sampling) and _accepts_sampling_kwargs(inner.complete)
        if self._sampling and not self._sampling_supported:
            log.info("sampling_not_supported", backend=inner.name, sampling=self._sampling)
        self._price_table = dict(price_table) if price_table is not None else None

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Delegate to the inner backend, plumbing sampling kwargs when supported."""
        if self._sampling_supported:
            inner = cast(Any, self.inner)
            result: tuple[str, dict[str, Any]] = await inner.complete(
                system_prompt, user_prompt, **self._sampling
            )
            return result
        return await self.inner.complete(system_prompt, user_prompt)

    async def analyze(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[list[Finding], dict[str, Any]]:
        """
        Run the review with schema guidance + bounded repair retries.

        Returns (findings, usage). On persistent validation failure returns
        best-effort findings, or raises BackendError when strict=True.
        """
        guided_system = self._guided_system_prompt(system_prompt)
        text, usage = await self.complete(guided_system, user_prompt)
        findings, errors = self._validate(text)

        attempt = 0
        needs_retry = bool(errors) or (not findings and self._retry_on_empty)
        while needs_retry and attempt < self._max_retries:
            attempt += 1
            log.warning(
                "structured_retry",
                backend=self.inner.name,
                attempt=attempt,
                max_retries=self._max_retries,
                errors=errors[:_MAX_ERRORS_IN_REPAIR],
            )
            repair_prompt = self._repair_prompt(user_prompt, text, errors, attempt)
            text, usage = await self.complete(guided_system, repair_prompt)
            findings, errors = self._validate(text)
            needs_retry = bool(errors) or (not findings and self._retry_on_empty)

        findings = self._calibrate(findings)
        usage = self._account_usage(usage if isinstance(usage, dict) else {})

        if errors:
            log.error(
                "structured_output_invalid",
                backend=self.inner.name,
                attempts=attempt + 1,
                errors=errors[:_MAX_ERRORS_IN_REPAIR],
                kept_findings=len(findings),
            )
            if self._strict:
                raise BackendError(
                    f"{self.name}: schema validation failed after {attempt + 1} "
                    f"attempt(s): " + "; ".join(errors[:3])
                )
        log.info(
            "structured_analyze_ok",
            backend=self.name,
            findings=len(findings),
            attempts=attempt + 1,
            json_mode=self._json_mode(),
        )
        return findings, usage

    async def aclose(self) -> None:
        """Close the wrapped backend if it supports it."""
        close = getattr(self.inner, "aclose", None)
        if callable(close):
            await close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _json_mode(self) -> str:
        """
        JSON-mode negotiation: 'json_object' when the inner backend already
        sends response_format={"type": "json_object"} (OpenAI-compatible);
        otherwise 'prompt_only' — enforced via prompt + post-validation.
        """
        if isinstance(self.inner, _OpenAICompatibleBackend):
            return "json_object"
        return "prompt_only"

    def _guided_system_prompt(self, system_prompt: str) -> str:
        guided = system_prompt + _SCHEMA_BLOCK
        if self._json_mode() == "prompt_only":
            guided += _PROMPT_ONLY_SUFFIX
        return guided

    def _validate(self, text: str | None) -> tuple[list[Finding], list[str]]:
        """
        Validate raw model output.

        Returns (valid findings, error messages). Unlike parse_findings(),
        every rejection is captured as an error string so the repair prompt
        can show the model exactly what to fix.
        """
        raw = text or ""
        data = _extract_json(raw)
        # _extract_json() returns {} both for unparseable input and for a
        # literal "{}" — neither is valid review output (the schema requires
        # {"findings": [...]}), so both are validation errors that trigger
        # the repair path.
        if data == {}:
            return [], ["no parseable JSON found in model output"]
        items: Any = data.get("findings", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return [], ["top-level 'findings' must be a list"]
        findings: list[Finding] = []
        errors: list[str] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"finding[{i}]: not a JSON object")
                continue
            try:
                findings.append(Finding.model_validate(item))
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                errors.append(f"finding[{i}]: {details}")
        return findings, errors

    def _calibrate(self, findings: list[Finding]) -> list[Finding]:
        """
        Confidence calibration: clamp to [0, 1], quantize to 2 decimals.

        Drops findings with blank title/explanation (defense in depth — the
        schema already enforces min lengths, but this layer logs the drop).
        """
        kept: list[Finding] = []
        for finding in findings:
            if not finding.title.strip() or not finding.explanation.strip():
                log.warning("finding_dropped_blank", path=finding.path, line=finding.line)
                continue
            calibrated = round(min(1.0, max(0.0, finding.confidence)), 2)
            if calibrated != finding.confidence:
                finding = finding.model_copy(update={"confidence": calibrated})
            kept.append(finding)
        return kept

    def _repair_prompt(
        self, original_user_prompt: str, bad_output: str, errors: list[str], attempt: int
    ) -> str:
        """Escalating repair prompt: bad output + validation errors, JSON-only."""
        lines = [
            f"Your previous response (attempt {attempt}) was not valid review JSON.",
            "Fix it and respond with ONLY the corrected JSON object.",
            "",
            "Validation errors to fix:",
            *(f"- {e}" for e in errors[:_MAX_ERRORS_IN_REPAIR]),
            "",
            "Your previous output:",
            "```",
            (bad_output or "")[:_BAD_OUTPUT_TRUNCATION],
            "```",
            "",
            "Original review request:",
            original_user_prompt,
            "",
            'Respond with ONLY a JSON object like {"findings": [...]} matching the '
            "schema in the system prompt. No prose, no markdown fences.",
        ]
        if attempt >= self._max_retries:
            lines.append("This is the final attempt: unparseable output will be discarded.")
        return "\n".join(lines)

    def _account_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich usage with token counts and estimated cost.

        Token keys are always present (0 when the backend did not report
        them). estimated_cost_usd is added only when a price table covers
        the inner backend's model.
        """
        out = dict(usage)
        prompt_tokens = _as_int(out.get("prompt_tokens"))
        completion_tokens = _as_int(out.get("completion_tokens"))
        out["prompt_tokens"] = prompt_tokens
        out["completion_tokens"] = completion_tokens
        if self._price_table is not None:
            prices = self._price_table.get(self.model)
            if prices is not None:
                per_1k_in, per_1k_out = prices
                cost = prompt_tokens / 1000 * per_1k_in + completion_tokens / 1000 * per_1k_out
                out["estimated_cost_usd"] = round(cost, 6)
                out["cost_model"] = self.model
        if self._sampling:
            out["sampling"] = dict(self._sampling)
        return out
