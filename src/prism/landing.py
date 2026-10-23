"""Landing page HTML for the PRism web service.

Premium dark developer-tool design (Linear/Raycast/CodeRabbit grammar):
near-black + grid + single glow, Inter type, one accent, the product mock
as the hero asset. Single inline stylesheet, no build step.
"""

from __future__ import annotations

from typing import Any

from prism.logo_data import LOGO_DATA_URI

INSTALL_URL = "https://github.com/apps/prism-reviews-bot/installations/new"
SOURCE_URL = "https://github.com/bunnysayzz/PRism"
DOCS_URL = "https://github.com/bunnysayzz/PRism#readme"
_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:'
    'wght@400;500;600&display=swap" rel="stylesheet">'
)

_CSS = """
:root{
--bg:#0A0A0F;--surface:#0D0D12;--border:rgba(255,255,255,.07);
--text:#F5F5F0;--muted:#A0A0A8;--faint:#6B6B73;
--green:#238636;--green-hi:#2ea043;--blue:#58a6ff;
--red:#f85149;--amber:#d29922;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,
sans-serif;background:var(--bg);color:var(--text);line-height:1.65;
-webkit-font-smoothing:antialiased}
::selection{background:#23863655}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1080px;margin:0 auto;padding:0 1.5rem}
/* backdrop */
.bg{position:fixed;inset:0;z-index:-2;
background-image:linear-gradient(rgba(255,255,255,.033) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.033) 1px,transparent 1px);
background-size:48px 48px;
-webkit-mask-image:radial-gradient(ellipse 90% 62% at 50% 0%,black 25%,
transparent 78%);
mask-image:radial-gradient(ellipse 90% 62% at 50% 0%,black 25%,
transparent 78%)}
.glow{position:fixed;left:50%;top:-12rem;transform:translateX(-50%);
width:56rem;height:26rem;z-index:-1;pointer-events:none;
background:radial-gradient(closest-side,#23863630,transparent);
filter:blur(70px)}
/* nav */
nav{position:sticky;top:0;z-index:20;
background:rgba(10,10,15,.72);
-webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px);
border-bottom:1px solid rgba(255,255,255,.06)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;
height:64px}
.brand{display:flex;align-items:center;gap:.55rem;font-weight:700;
font-size:1.15rem;letter-spacing:-.02em;color:var(--text)}
.brand img{width:30px;height:30px;border-radius:7px;display:block}
.brand:hover{text-decoration:none}
.brand em{font-style:normal;color:var(--green-hi)}
.nav-links{display:flex;gap:1.6rem;font-size:.9rem}
.nav-links a{color:var(--muted)}
.nav-links a:hover{color:var(--text);text-decoration:none}
/* buttons */
.btn{display:inline-block;font-weight:600;border-radius:8px;
font-size:.95rem;transition:background .15s ease,border-color .15s ease;
text-decoration:none}
.btn:hover{text-decoration:none}
.btn-primary{background:var(--green);color:#fff;border:1px solid var(--green-hi)}
.btn-primary:hover{background:var(--green-hi)}
.btn-nav{padding:.55rem 1.15rem;font-size:.88rem}
.btn-lg{padding:.85rem 2.1rem;font-size:1rem}
/* hero */
.hero{text-align:center;padding:6rem 1.5rem 0}
.hero-logo{display:block;width:104px;height:104px;border-radius:24px;
margin:0 auto 1.8rem;
box-shadow:0 0 0 1px rgba(255,255,255,.08),0 18px 50px rgba(0,0,0,.5)}
.pill{display:inline-block;font-size:.78rem;font-weight:500;color:var(--muted);
border:1px solid rgba(255,255,255,.12);border-radius:999px;
padding:.35rem 1rem;margin-bottom:1.6rem}
.pill b{color:var(--text);font-weight:600}
.hero h1{font-size:clamp(2.6rem,6vw,4.2rem);font-weight:600;
letter-spacing:-.03em;line-height:1.06;max-width:46rem;margin:0 auto 1.2rem}
.grad{background:linear-gradient(92deg,#7ee787,#58a6ff);
-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .sub{color:var(--muted);font-size:1.12rem;max-width:37rem;
margin:0 auto 2.2rem}
.micro{margin-top:1rem;font-size:.83rem;color:var(--faint)}
/* product mock */
.mock{max-width:860px;margin:3.5rem auto 0;text-align:left;
background:var(--surface);border:1px solid var(--border);border-radius:14px;
overflow:hidden;box-shadow:0 0 0 1px rgba(255,255,255,.04),
0 30px 80px rgba(0,0,0,.55)}
.mock-bar{display:flex;align-items:center;gap:.5rem;
padding:.8rem 1.1rem;border-bottom:1px solid var(--border)}
.dot{width:.7rem;height:.7rem;border-radius:50%}
.mock-title{margin-left:.6rem;font-family:var(--mono);font-size:.8rem;
color:var(--faint)}
.mock-code{font-family:var(--mono);font-size:.83rem;line-height:1.75;
padding:1rem 0;overflow-x:auto;border-bottom:1px solid var(--border)}
.ml{padding:0 1.2rem;white-space:pre;color:#c9d1d9}
.ml .ln{color:var(--faint);display:inline-block;width:2rem;user-select:none}
.ml.add{background:#3fb95014}
.ml.add .tx{color:#7ee787}
.mock-finding{padding:1.2rem 1.3rem}
.mf-head{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem}
.sev{font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
padding:.2rem .6rem;border-radius:999px}
.sev-high{background:#f8514922;color:#ff7b72;border:1px solid #f8514944}
.mf-cat{font-size:.8rem;color:var(--faint)}
.mf-title{font-weight:600;font-size:.95rem;margin-bottom:.35rem}
.mf-title code{font-family:var(--mono);font-size:.85em;background:#ffffff10;
padding:.1rem .35rem;border-radius:.3rem}
.mf-body{font-size:.88rem;color:var(--muted);max-width:44rem}
.mf-body code{font-family:var(--mono);font-size:.85em;background:#ffffff10;
padding:.1rem .35rem;border-radius:.3rem}
.mf-fix{margin-top:.8rem;font-size:.85rem;display:flex;gap:.6rem;
align-items:baseline}
.mf-fix .k{font-size:.72rem;font-weight:700;letter-spacing:.06em;
text-transform:uppercase;color:var(--green-hi)}
.mf-fix code{font-family:var(--mono);font-size:.82em;color:#7ee787;
background:#3fb95010;border:1px solid #3fb95030;padding:.25rem .6rem;
border-radius:.4rem}
/* stats */
.stats{display:flex;justify-content:center;gap:3.5rem;flex-wrap:wrap;
padding:4.5rem 1.5rem 1rem}
.stat{text-align:center}
.stat .n{font-family:var(--mono);font-size:2.1rem;font-weight:700;
letter-spacing:-.02em}
.stat .l{color:var(--faint);font-size:.83rem;margin-top:.25rem}
/* sections */
section.block{padding:6rem 0}
.eyebrow{font-size:.78rem;font-weight:600;letter-spacing:.1em;
text-transform:uppercase;color:var(--faint);margin-bottom:.9rem}
h2.stitle{font-size:clamp(1.7rem,3.4vw,2.4rem);font-weight:600;
letter-spacing:-.02em;line-height:1.15;margin-bottom:1rem;max-width:40rem}
.lead{color:var(--muted);max-width:42rem;font-size:1.02rem}
.center{text-align:center}
.center h2.stitle,.center .lead{margin-left:auto;margin-right:auto}
.feat{display:grid;grid-template-columns:5.5rem 1fr;gap:1.4rem;
padding:2.2rem 0;border-top:1px solid var(--border)}
.feat:last-child{border-bottom:1px solid var(--border)}
.feat .num{font-family:var(--mono);font-size:1rem;color:var(--faint);
padding-top:.35rem}
.feat h3{font-size:1.2rem;font-weight:600;letter-spacing:-.01em;
margin-bottom:.45rem}
.feat p{color:var(--muted);font-size:.96rem;max-width:46rem}
/* steps */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
gap:1rem;margin-top:2.5rem}
.step{background:var(--surface);border:1px solid var(--border);
border-radius:12px;padding:1.6rem 1.4rem}
.step .sn{width:2rem;height:2rem;border-radius:50%;background:#1f6feb;
color:#fff;font-weight:700;font-size:.9rem;display:flex;align-items:center;
justify-content:center;margin-bottom:1rem}
.step h3{font-size:1rem;margin-bottom:.4rem}
.step p{color:var(--muted);font-size:.9rem}
/* final */
.final{text-align:center;padding:6rem 1.5rem;position:relative}
.final h2{font-size:clamp(1.8rem,3.6vw,2.6rem);font-weight:600;
letter-spacing:-.02em;margin-bottom:1rem}
.final p{color:var(--muted);margin-bottom:2.2rem}
/* footer */
footer{border-top:1px solid var(--border);padding:2.5rem 1.5rem 2rem;
font-size:.85rem;color:var(--faint)}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;
gap:1.2rem;max-width:1080px}
footer .cols{display:flex;gap:3rem}
footer .col h4{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;
margin-bottom:.7rem;color:var(--muted);font-weight:600}
footer .col a{display:block;color:var(--faint);margin:.35rem 0}
footer .col a:hover{color:var(--text)}
.status-dot{display:inline-block;width:.45rem;height:.45rem;border-radius:50%;
margin-right:.4rem;vertical-align:baseline}
.ok{background:#3fb950}.warn{background:#d29922}
@media(max-width:640px){.nav-links{display:none}.feat{grid-template-columns:1fr
1.2rem}}
"""


def _mock_html() -> str:
    d1 = (
        '<div class="ml"><span class="ln">1</span>'
        '<span class="tx">+ export function addPoints(a: any, b: any) {</span></div>'
    )
    d2 = (
        '<div class="ml"><span class="ln">2</span>'
        '<span class="tx">+   var total = a + b;</span></div>'
    )
    d3 = (
        '<div class="ml"><span class="ln">3</span>'
        '<span class="tx">+   if (total == 0) {</span></div>'
    )
    return (
        '<div class="mock">'
        '<div class="mock-bar"><span class="dot" style="background:#f85149"></span>'
        '<span class="dot" style="background:#d29922"></span>'
        '<span class="dot" style="background:#3fb950"></span>'
        '<span class="mock-title">bunnysayzz/airplane · '
        "__prism_e2e__.ts</span></div>"
        '<div class="mock-code">' + d1 + d2 + d3 + "</div>"
        '<div class="mock-finding">'
        '<div class="mf-head"><span class="sev sev-high">High</span>'
        '<span class="mf-cat">correctness</span></div>'
        '<p class="mf-title">Unexpected string concatenation from '
        "<code>any</code> parameters</p>"
        '<p class="mf-body">With <code>a</code> and <code>b</code> typed as '
        "<code>any</code>, <code>a + b</code> silently concatenates strings "
        'instead of adding numbers: <code>addPoints("1", 2)</code> returns '
        '<code>"12"</code>.</p>'
        '<div class="mf-fix"><span class="k">Fix</span>'
        "<code>export function addPoints(a: number, b: number)</code></div>"
        "</div></div>"
    )


def render_landing(checks: dict[str, Any]) -> str:
    """Render the marketing landing page (status lives in the footer only)."""
    ok = bool(checks.get("ok"))
    dot = '<span class="status-dot ok"></span>' if ok else '<span class="status-dot warn"></span>'
    status_word = "operational" if ok else "setup needed"
    mock = _mock_html()
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="PRism - autonomous AI code review for GitHub pull requests.">
<title>PRism - AI code review for GitHub</title>
{_FONT_LINK}
<style>{_CSS}</style></head>
<body>
<div class="bg"></div><div class="glow"></div>
<nav><div class="wrap">
<a class="brand" href="/"><img src="{LOGO_DATA_URI}" alt="PRism logo">PR<em>ism</em></a>
<div class="nav-links">
<a href="#features">Features</a>
<a href="#how">How it works</a>
<a href="{DOCS_URL}">Docs</a>
<a href="/readyz">Status</a>
</div>
<a class="btn btn-primary btn-nav" href="{INSTALL_URL}">Install</a>
</div></nav>
<header class="hero">
<img class="hero-logo" src="{LOGO_DATA_URI}" alt="PRism logo">
<span class="pill"><b>PRism Reviews Bot</b> for GitHub</span>
<h1>Ship pull requests with <span class="grad">confidence.</span></h1>
<p class="sub">An autonomous code review bot. Every pull request gets checked
for bugs, vulnerabilities and code smells - inline, with severity and a fix.</p>
<p><a class="btn btn-primary btn-lg" href="{INSTALL_URL}">Install on GitHub</a></p>
<p class="micro">Free for public repos &middot; 2-minute setup</p>
{mock}
</header>
<div class="stats wrap">
<div class="stat"><div class="n">454</div><div class="l">tests passing</div></div>
<div class="stat"><div class="n">31</div><div class="l">golden eval cases</div></div>
<div class="stat"><div class="n">1.000</div><div class="l">eval F1 score</div></div>
<div class="stat"><div class="n">&lt;60s</div><div class="l">review turnaround</div></div>
</div>
<section class="block"><div class="wrap center">
<div class="eyebrow">The problem</div>
<h2 class="stitle">Unreviewed code ships bugs.</h2>
<p class="lead">Review bottlenecks slow every team down, and tired reviewers
rubber-stamp. PRism reviews every pull request in under a minute - including
the ones humans would skim at 2am.</p>
</div></section>
<section class="block" id="features" style="padding-top:0"><div class="wrap">
<div class="eyebrow">Features</div>
<h2 class="stitle">Reviews that read the diff.</h2>
<p class="lead" style="margin-bottom:2rem">A fine-tuned model trained on real
pull request reviews, wrapped in production guardrails.</p>
<div class="feat"><div class="num">01</div><div>
<h3>Understands code, not just lines</h3>
<p>Follows data flow across your diff to catch logic bugs, race conditions
and API misuse that linters miss.</p></div></div>
<div class="feat"><div class="num">02</div><div>
<h3>Silence over noise</h3>
<p>Every finding is confidence-scored. High-confidence issues post directly;
borderline ones get a second verification pass. No style nitpicks, ever.</p></div></div>
<div class="feat"><div class="num">03</div><div>
<h3>Your model or ours</h3>
<p>Serve your own fine-tuned model, or plug in any OpenAI-compatible API.
Providers fail over automatically, per request.</p></div></div>
<div class="feat"><div class="num">04</div><div>
<h3>Honest when it fails</h3>
<p>If the model backend goes down, PRism says so - it never posts a fake
clean bill of health.</p></div></div>
</div></section>
<section class="block" id="how" style="padding-top:0"><div class="wrap">
<div class="eyebrow">How it works</div>
<h2 class="stitle">Live in three steps.</h2>
<div class="steps">
<div class="step"><div class="sn">1</div><h3>Install</h3>
<p>One click installs PRism on your repositories. No code changes, no CI config.</p></div>
<div class="step"><div class="sn">2</div><h3>Open a PR</h3>
<p>GitHub notifies PRism the moment a pull request is opened or updated.</p></div>
<div class="step"><div class="sn">3</div><h3>Get the review</h3>
<p>Inline findings with severity, category and suggested fixes appear on your PR.</p></div>
</div>
</div></section>
<section class="final">
<h2>Stop shipping surprises.</h2>
<p>Install PRism and get your first AI review in minutes.</p>
<a class="btn btn-primary btn-lg" href="{INSTALL_URL}">Install on GitHub</a>
</section>
<footer><div class="wrap">
<div class="cols">
<div class="col"><h4>Product</h4>
<a href="#features">Features</a><a href="#how">How it works</a>
<a href="/readyz">Status</a></div>
<div class="col"><h4>Resources</h4>
<a href="{DOCS_URL}">Documentation</a><a href="{SOURCE_URL}">Source code</a>
<a href="/healthz">Health</a></div>
</div>
<div>{dot}Status: {status_word} &middot; PRism is open source</div>
</div></footer>
</body></html>"""
