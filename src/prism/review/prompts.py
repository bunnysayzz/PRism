"""
Review prompts.

The model must output strict JSON matching the Finding schema. The system
prompt carries the persona + rules; build_user_prompt assembles the per-file
payload: language, diff hunks, and surrounding file context.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a senior staff engineer performing a code review on a pull request diff.
You are precise, skeptical of your own first impressions, and allergic to noise.

RULES
1. Report ONLY real, defensible issues: bugs, security vulnerabilities,
   correctness problems, race conditions, resource leaks, and meaningful
   performance problems. When in doubt, stay silent.
2. NO style nitpicks, NO formatting opinions, NO "consider renaming" comments,
   NO praise, NO summaries of what the code does. Silence is better than noise.
3. Every finding MUST reference an exact new-file line number from the diff.
   Never invent line numbers. Never comment on unchanged lines.
4. Explain WHY the code is wrong, not just WHAT is wrong. A finding without a
   concrete failure scenario is not a finding.
5. If you suggest a fix, make it a minimal, compilable code snippet.
6. Calibrate confidence honestly: 0.95+ only when you can name the exact
   failure; 0.7-0.9 for likely issues; below 0.7 for suspicions (which you
   should usually not report at all).

OUTPUT FORMAT — strict JSON only, no prose, no markdown fences:
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
If there are no real issues, return {"findings": []}.
"""


def build_user_prompt(
    path: str,
    language: str,
    hunk_context: str,
    head_excerpt: str = "",
    base_excerpt: str = "",
    max_findings: int = 10,
) -> str:
    """
    Assemble the per-file review prompt.

    - hunk_context: the unified diff hunks for this file (primary signal).
    - head_excerpt: surrounding code from the new file version (context).
    - base_excerpt: surrounding code from the old version (for renames/refactors).
    """
    sections = [
        f"Review the following changes to `{path}` (language: {language or 'unknown'}).",
        f"Report at most {max_findings} findings, highest severity first.",
        "",
        "## Diff hunks",
        "```diff",
        hunk_context or "(no diff hunks available)",
        "```",
    ]
    if head_excerpt:
        sections += [
            "",
            "## Surrounding code (new version, line numbers shown)",
            "```",
            head_excerpt,
            "```",
        ]
    if base_excerpt:
        sections += [
            "",
            "## Surrounding code (old version, for reference)",
            "```",
            base_excerpt,
            "```",
        ]
    sections += [
        "",
        "Respond with strict JSON only, in the exact schema from the system prompt.",
        f'Use "path": "{path}" verbatim in every finding.',
    ]
    return "\n".join(sections)


_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
}


def guess_language(path: str) -> str:
    """Map file extension to a code-fence language hint."""
    lower = path.lower()
    for ext, lang in _LANGUAGE_BY_EXTENSION.items():
        if lower.endswith(ext):
            return lang
    return ""
