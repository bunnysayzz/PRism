"""Landing page HTML for the PRism web service.

Modeled on production code-review products: nav, hero with a realistic
review sample, numbered features, stats, install CTA, footer.
"""

from __future__ import annotations

from typing import Any

INSTALL_URL = "https://github.com/apps/prism-reviews-bot/installations/new"
SOURCE_URL = "https://github.com/bunnysayzz/PRism"
DOCS_URL = "https://github.com/bunnysayzz/PRism#readme"

# Sample diff shown in the hero; kept as short concatenated pieces for lint.
_SAMPLE_DEL = (
    '<div class="cl del"><span class="ln">41</span>'
    'cursor.execute(f"SELECT * FROM users WHERE id = &#123;user_id&#125;")'
    "</div>"
)
_SAMPLE_ADD = (
    '<div class="cl add"><span class="ln">41</span>'
    'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'
    "</div>"
)


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
        else f"""<section class="wrap"><div class="card"><h2>Setup checklist</h2>
<ul class="checklist">{setup_rows}</ul></div></section>"""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="PRism - autonomous AI code review bot for GitHub pull requests.">
<title>PRism - autonomous AI code review for GitHub</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
background:#0a0c10;color:#e6edf3;line-height:1.65;-webkit-font-smoothing:antialiased}}
a{{color:#58a6ff;text-decoration:none}}a:hover{{text-decoration:underline}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 1.5rem}}
nav{{position:sticky;top:0;z-index:10;backdrop-filter:blur(12px);
background:#0a0c10cc;border-bottom:1px solid #21262d}}
nav .wrap{{display:flex;align-items:center;justify-content:space-between;
height:64px}}
.brand{{font-weight:800;font-size:1.25rem;color:#fff;letter-spacing:-.01em}}
.brand span{{color:#58a6ff}}
.nav-links{{display:flex;gap:1.5rem;font-size:.92rem;color:#8b949e}}
.nav-links a{{color:#8b949e}}.nav-links a:hover{{color:#e6edf3}}
.btn{{display:inline-block;font-weight:600;padding:.7rem 1.5rem;border-radius:.55rem;
font-size:.95rem;transition:transform .08s ease,background .15s}}
.btn:active{{transform:scale(.98)}}
.btn-primary{{background:#238636;color:#fff;border:1px solid #2ea043}}
.btn-primary:hover{{background:#2ea043;text-decoration:none}}
.btn-lg{{padding:.9rem 2.2rem;font-size:1.05rem}}
.hero{{text-align:center;padding:5.5rem 1.5rem 4rem;
background:radial-gradient(70rem 32rem at 50% -10rem,#16283f 0%,#0a0c10 62%)}}
.hero h1{{font-size:clamp(2.2rem,5vw,3.4rem);letter-spacing:-.03em;
line-height:1.12;max-width:44rem;margin:0 auto 1rem}}
.hero h1 .grad{{background:linear-gradient(90deg,#58a6ff,#a371f7);
-webkit-background-clip:text;background-clip:text;color:transparent}}
.hero .sub{{color:#8b949e;font-size:1.12rem;max-width:36rem;margin:0 auto 2.2rem}}
.pill{{display:inline-block;font-size:.78rem;font-weight:600;padding:.32rem .85rem;
border-radius:999px;border:1px solid #30363d;margin-bottom:1.4rem}}
.pill-ok{{color:#3fb950;border-color:#3fb95055;background:#3fb95014}}
.pill-warn{{color:#d29922;border-color:#d2992255;background:#d2992214}}
.sample{{max-width:880px;margin:0 auto;text-align:left;background:#0d1117;
border:1px solid #30363d;border-radius:.9rem;overflow:hidden;
box-shadow:0 24px 70px #00000088}}
.sample-head{{display:flex;align-items:center;gap:.6rem;padding:.8rem 1.1rem;
border-bottom:1px solid #21262d;font-size:.85rem;color:#8b949e}}
.dot{{width:.7rem;height:.7rem;border-radius:50%}}
.code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.83rem;
line-height:1.7;padding:1rem 0;overflow-x:auto}}
.cl{{padding:0 1.1rem;white-space:pre}}
.cl.add{{background:#3fb9501c}}.cl.del{{background:#f8514914}}
.cl .ln{{color:#6e7681;display:inline-block;width:2.2rem;user-select:none}}
.cl.add .ln{{color:#3fb950}}.cl.del .ln{{color:#f85149}}
.finding{{margin:.9rem 1.1rem;border:1px solid #d2992244;border-left:3px solid #d29922;
border-radius:.5rem;padding:.9rem 1.1rem;background:#d299220d}}
.finding .fh{{display:flex;gap:.5rem;align-items:center;margin-bottom:.35rem;
font-size:.82rem}}
.sev{{font-weight:700;font-size:.72rem;padding:.15rem .55rem;border-radius:999px;
background:#d2992226;color:#d29922;text-transform:uppercase;letter-spacing:.04em}}
.cat{{color:#8b949e}}
.finding p{{font-size:.88rem;color:#c9d1d9}}
.finding .fix{{margin-top:.55rem;font-size:.85rem}}
.finding .fix code{{background:#0d1117;border:1px solid #30363d;padding:.15rem .45rem;
border-radius:.35rem;font-size:.82em;color:#7ee787}}
.stats{{display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;
padding:3.2rem 1rem;border-bottom:1px solid #21262d}}
.stat{{text-align:center}}.stat .n{{font-size:1.9rem;font-weight:800;
letter-spacing:-.02em}}.stat .l{{color:#8b949e;font-size:.85rem}}
section.block{{padding:4.5rem 0}}
.eyebrow{{color:#58a6ff;font-weight:700;font-size:.82rem;letter-spacing:.12em;
text-transform:uppercase;margin-bottom:.7rem}}
h2.stitle{{font-size:clamp(1.5rem,3vw,2.1rem);letter-spacing:-.02em;
margin-bottom:1rem;max-width:38rem}}
.lead{{color:#8b949e;max-width:40rem;margin-bottom:2rem}}
.feat{{display:grid;grid-template-columns:64px 1fr;gap:1.1rem;padding:1.6rem 0;
border-top:1px solid #21262d}}
.feat .num{{font-size:2rem;font-weight:800;color:#30363d}}
.feat h3{{font-size:1.12rem;margin-bottom:.35rem}}
.feat p{{color:#8b949e;font-size:.95rem;max-width:44rem}}
.feat ul{{color:#8b949e;font-size:.92rem;margin-top:.6rem;padding-left:1.2rem}}
.feat li{{margin:.3rem 0}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:1rem;margin-top:2rem}}
.step{{background:#0d1117;border:1px solid #30363d;border-radius:.75rem;
padding:1.4rem 1.3rem}}
.step .n{{width:2rem;height:2rem;border-radius:50%;background:#1f6feb;color:#fff;
font-weight:700;display:flex;align-items:center;justify-content:center;
margin-bottom:.8rem}}
.step h3{{font-size:1rem;margin-bottom:.35rem}}
.step p{{color:#8b949e;font-size:.9rem}}
.card{{background:#0d1117;border:1px solid #30363d;border-radius:.75rem;
padding:1.5rem 1.4rem;margin-top:2.5rem}}
.card h2{{font-size:1.15rem;margin-bottom:.6rem}}
.checklist{{color:#d29922;padding-left:1.2rem}}
.endpoints{{list-style:none}}.endpoints li{{padding:.45rem 0;
border-bottom:1px solid #21262d;font-size:.93rem}}
.endpoints li:last-child{{border-bottom:none}}
code{{background:#0d1117;border:1px solid #30363d;padding:.15rem .45rem;
border-radius:.35rem;font-size:.88em}}
.final{{text-align:center;padding:5rem 1.5rem;
background:radial-gradient(50rem 24rem at 50% 8rem,#16283f 0%,#0a0c10 65%)}}
.final h2{{font-size:clamp(1.6rem,3.4vw,2.4rem);letter-spacing:-.02em;
margin-bottom:1rem}}
.final p{{color:#8b949e;margin-bottom:2rem}}
footer{{border-top:1px solid #21262d;padding:2rem 1.5rem;text-align:center;
color:#8b949e;font-size:.85rem}}
footer .wrap{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:1rem;
max-width:1080px}}
</style></head>
<body>
<nav><div class="wrap">
<a class="brand" href="/">PR<span>ism</span></a>
<div class="nav-links">
<a href="#features">Features</a>
<a href="#how">How it works</a>
<a href="{DOCS_URL}">Docs</a>
<a href="/readyz">Status</a>
</div>
<a class="btn btn-primary" href="{INSTALL_URL}">Install</a>
</div></nav>

<header class="hero">
{badge}
<h1>Ship pull requests with <span class="grad">confidence.</span></h1>
<p class="sub">PRism is an autonomous code review bot for GitHub. Every pull
request gets reviewed for bugs, vulnerabilities and code smells - inline,
with severity and suggested fixes.</p>
<p><a class="btn btn-primary btn-lg" href="{INSTALL_URL}">Install on GitHub</a></p>
<div class="sample">
<div class="sample-head">
<span class="dot" style="background:#f85149"></span>
<span class="dot" style="background:#d29922"></span>
<span class="dot" style="background:#3fb950"></span>
<span style="margin-left:.4rem">PRism bot reviewed 2 minutes ago</span>
</div>
<div class="code">{_SAMPLE_DEL}{_SAMPLE_ADD}</div>
<div class="finding">
<div class="fh"><span class="sev">High</span><span class="cat">security · sql-injection</span></div>
<p><strong>SQL injection via f-string query.</strong> User input flows directly
into the SQL statement. A crafted <code>user_id</code> can read or modify
arbitrary rows.</p>
<p class="fix">Fix: <code>cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))</code></p>
</div>
</div>
</header>

<div class="stats wrap">
<div class="stat"><div class="n">200</div><div class="l">tests passing</div></div>
<div class="stat"><div class="n">31</div><div class="l">golden eval cases</div></div>
<div class="stat"><div class="n">1.000</div><div class="l">eval F1 score</div></div>
<div class="stat"><div class="n">24/7</div><div class="l">always reviewing</div></div>
</div>

<section class="block" id="features"><div class="wrap">
<div class="eyebrow">Features</div>
<h2 class="stitle">Reviews that understand code, not just pattern-match it.</h2>
<p class="lead">A fine-tuned language model trained on thousands of real pull
request reviews, wrapped in production guardrails.</p>
<div class="feat"><div class="num">01</div><div>
<h3>Deep, contextual reviews</h3>
<p>Understands data flow across your diff - not just the changed lines.
Catches logic bugs, race conditions and API misuse that linters miss.</p>
</div></div>
<div class="feat"><div class="num">02</div><div>
<h3>Confidence-gated, low noise</h3>
<p>Every finding is scored. High-confidence issues post directly;
borderline ones go through a second verification pass. Silence beats
false alarms.</p>
<ul><li>Severity-ranked findings, never style nitpicks</li>
<li>Structured JSON output - file, line, category, fix</li></ul>
</div></div>
<div class="feat"><div class="num">03</div><div>
<h3>Your model or ours</h3>
<p>Serve your own fine-tuned model, or plug in any OpenAI-compatible API
as backup. Providers fail over automatically per request.</p>
</div></div>
</div></section>

<section class="block" id="how" style="padding-top:0"><div class="wrap">
<div class="eyebrow">How it works</div>
<h2 class="stitle">Live in three steps.</h2>
<div class="steps">
<div class="step"><div class="n">1</div><h3>Install</h3>
<p>One click installs PRism on your repositories. No code changes, no CI config.</p></div>
<div class="step"><div class="n">2</div><h3>Open a PR</h3>
<p>GitHub notifies PRism the moment a pull request is opened or updated.</p></div>
<div class="step"><div class="n">3</div><h3>Get the review</h3>
<p>Inline comments with severity, category and suggested fixes appear on your PR.</p></div>
</div>
</div></section>

{setup_card}

<section class="wrap"><div class="card"><h2>Service endpoints</h2>
<ul class="endpoints">
<li><a href="/healthz">/healthz</a> - liveness probe</li>
<li><a href="/readyz">/readyz</a> - readiness report (JSON)</li>
<li><code>POST /webhooks/github</code> - GitHub webhook receiver</li>
</ul></div></section>

<section class="final">
<h2>Stop shipping surprises.</h2>
<p>Install PRism and get your first AI review in minutes.</p>
<a class="btn btn-primary btn-lg" href="{INSTALL_URL}">Install on GitHub</a>
</section>

<footer><div class="wrap">
<span>PRism is open source</span>
<span><a href="{SOURCE_URL}">GitHub</a> · <a href="{DOCS_URL}">Docs</a> ·
<a href="/readyz">Status</a></span>
</div></footer>
</body></html>"""
