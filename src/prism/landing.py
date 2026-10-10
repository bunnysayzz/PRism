"""Landing page HTML for the PRism web service."""

from __future__ import annotations

from typing import Any

INSTALL_URL = "https://github.com/apps/prism-reviews/installations/new"
SOURCE_URL = "https://github.com/bunnysayzz/PRism"


def render_landing(checks: dict[str, Any]) -> str:
    """Render the marketing + status landing page."""
    ok = bool(checks.get("ok"))
    problems = checks.get("problems", [])
    badge = (
        '<span class="pill pill-ok">● Live</span>'
        if ok
        else '<span class="pill pill-warn">● Setup needed</span>'
    )
    setup_rows = "".join(f"<li>{p}</li>" for p in problems)
    setup_card = (
        ""
        if ok
        else f"""<section class="card"><h2>Setup checklist</h2>
<ul class="checklist">{setup_rows}</ul></section>"""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="PRism - autonomous AI code review bot for GitHub pull requests.">
<title>PRism - autonomous code review bot</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
background:#0d1117;color:#e6edf3;line-height:1.65}}
.hero{{text-align:center;padding:5rem 1.25rem 3.5rem;
background:radial-gradient(60rem 30rem at 50% -8rem,#1c2f4a 0%,#0d1117 60%)}}
.pill{{display:inline-block;font-size:.8rem;font-weight:600;padding:.3rem .8rem;
border-radius:999px;border:1px solid #30363d;margin-bottom:1.25rem}}
.pill-ok{{color:#3fb950;border-color:#3fb95055;background:#3fb95014}}
.pill-warn{{color:#d29922;border-color:#d2992255;background:#d2992214}}
.hero h1{{font-size:3rem;letter-spacing:-.02em;margin-bottom:.5rem}}
.hero p.sub{{color:#8b949e;font-size:1.15rem;max-width:34rem;margin:0 auto 2rem}}
.cta-row{{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap}}
.btn{{display:inline-block;font-weight:600;padding:.85rem 1.8rem;border-radius:.6rem;
text-decoration:none;font-size:1rem;transition:transform .08s ease}}
.btn:active{{transform:scale(.98)}}
.btn-primary{{background:#238636;color:#fff;border:1px solid #2ea043}}
.btn-primary:hover{{background:#2ea043}}
.btn-ghost{{background:transparent;color:#e6edf3;border:1px solid #30363d}}
.btn-ghost:hover{{border-color:#8b949e}}
main{{max-width:900px;margin:0 auto;padding:0 1.25rem 4rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
gap:1rem;margin:3rem 0}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:.75rem;
padding:1.4rem 1.3rem}}
.card h3{{font-size:1.02rem;margin-bottom:.45rem}}
.card p{{color:#8b949e;font-size:.92rem}}
.card .icon{{font-size:1.5rem}}
.steps{{counter-reset:step;display:grid;gap:1rem;margin:1rem 0 3rem}}
.step{{display:flex;gap:1rem;align-items:flex-start;background:#161b22;
border:1px solid #30363d;border-radius:.75rem;padding:1.2rem 1.3rem}}
.step::before{{counter-increment:step;content:counter(step);flex:0 0 2rem;
height:2rem;border-radius:50%;background:#1f6feb;color:#fff;font-weight:700;
display:flex;align-items:center;justify-content:center}}
.card h2{{font-size:1.15rem;margin-bottom:.6rem}}
.checklist{{color:#d29922;padding-left:1.2rem}}
.endpoints{{list-style:none}}
.endpoints li{{padding:.45rem 0;border-bottom:1px solid #21262d;font-size:.95rem}}
.endpoints li:last-child{{border-bottom:none}}
code{{background:#0d1117;border:1px solid #30363d;padding:.15rem .45rem;
border-radius:.35rem;font-size:.88em}}
a{{color:#58a6ff}}
footer{{text-align:center;color:#8b949e;font-size:.85rem;padding:2rem 1rem 3rem;
border-top:1px solid #21262d}}
</style></head>
<body>
<header class="hero">
{badge}
<h1>PRism</h1>
<p class="sub">An autonomous code review bot for GitHub. Every pull request
gets reviewed for bugs, vulnerabilities and code smells - with severity,
category and suggested fixes, right inline.</p>
<div class="cta-row">
<a class="btn btn-primary" href="{INSTALL_URL}">Install on GitHub</a>
<a class="btn btn-ghost" href="{SOURCE_URL}">View source</a>
</div>
</header>
<main>
<div class="grid">
<div class="card"><div class="icon">🔍</div><h3>Finds real bugs</h3>
<p>Fine-tuned LLM trained on thousands of real pull request reviews -
not regex linting.</p></div>
<div class="card"><div class="icon">🛡️</div><h3>Security first</h3>
<p>Injection flaws, auth gaps and unsafe patterns surfaced with
severity ratings.</p></div>
<div class="card"><div class="icon">🧾</div><h3>Structured output</h3>
<p>Every finding is schema-validated JSON: file, line, category,
confidence and a fix.</p></div>
<div class="card"><div class="icon">🔌</div><h3>Your model or ours</h3>
<p>Bring your own fine-tuned model, or plug in any OpenAI-compatible
API as backup.</p></div>
</div>
<section class="card"><h2>How it works</h2><div class="steps">
<div class="step"><div><strong>Install PRism</strong> on your repositories
with one click. No code changes, no CI config.</div></div>
<div class="step"><div><strong>Open a pull request.</strong> GitHub notifies
PRism automatically.</div></div>
<div class="step"><div><strong>Get the review.</strong> Inline comments with
severity, category and suggested fixes appear on your PR.</div></div>
</div></section>
<section class="card"><h2>Service endpoints</h2>
<ul class="endpoints">
<li><a href="/healthz">/healthz</a> - liveness probe</li>
<li><a href="/readyz">/readyz</a> - readiness report (JSON)</li>
<li><code>POST /webhooks/github</code> - GitHub webhook receiver</li>
</ul></section>
{setup_card}
</main>
<footer>PRism is open source -
<a href="{SOURCE_URL}">bunnysayzz/PRism</a></footer>
</body></html>"""
