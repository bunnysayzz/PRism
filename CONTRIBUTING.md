# Contributing to PRism

Thanks for your interest in contributing. PRism is a FastAPI GitHub App that
posts AI code reviews on pull requests. This guide covers the dev setup, the
code standards every change is held to, and what we expect in a pull request.

## Development setup

Requirements: Python 3.11+.

```bash
git clone https://github.com/bunnysayzz/PRism.git
cd PRism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

For a pure code change you do not need a GitHub App, a model, or Redis: the
test suite runs against stub backends and mocked GitHub responses.

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -q        # full suite
.venv/bin/python -m pytest tests/test_reviews.py -q   # single file
```

Every behavior change needs a test. If you fix a bug, add a regression test
that fails without the fix.

## Running locally

Dev-inline mode runs the whole review in-process (no Redis, no worker):

```bash
# leave REDIS_URL unset in .env
uvicorn prism.main:app --reload
```

To exercise the real webhook path you need a GitHub App installed on a test
repo and a public tunnel to your machine:

1. Create a GitHub App (see "GitHub App setup" in the README) with Pull
   requests (read and write), Contents (read), and Metadata (read)
   permissions, subscribed to Pull request events.
2. Point its webhook URL at `https://<your-tunnel>/webhooks/github`
   ([smee.io](https://smee.io) or ngrok both work).
3. Install the App on a scratch repo, open a PR, and watch the review land.

With `REVIEW_BACKEND=stub` the pipeline runs end to end without any model,
which is enough for most webhook, parsing, and posting work.

## Code standards

All four checks must pass before a PR is reviewable:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy src                # strict mode
```

Conventions worth knowing:

- **No secrets in code, logs, or tests.** Settings go through
  `prism.config.Settings`; anything secret-shaped is redacted from logs.
- **No model or provider branding in user-facing output.** Reviews say
  "PRism reviewed this pull request", never which model produced them.
- **No emoji in review output or headings.** Severity is plain text
  (`**CRITICAL**`, `**MEDIUM**`).
- **No em dashes in docs or comments.** Use a colon, comma, or parentheses.
- **Fail loud, never fake clean.** If a backend call fails, the review must
  say the review is incomplete rather than claim no issues were found.
- **Never log webhook secrets, tokens, or private keys**, not even digests.

## Pull request expectations

- One logical change per PR. Split unrelated work.
- Describe what changed and why; link the issue if there is one.
- Tests, lint, format, and mypy green (CI runs the same four checks, plus
  the eval gate: review quality must not regress below the F1 floor).
- Update the README or `docs/` when you change behavior, config keys, or
  the `.prism.yml` schema.
- Keep the change scoped: no drive-by refactors in a bug-fix PR.

## Reporting issues

Use the issue templates (bug report / feature request). For review-quality
problems, the most useful report includes the PR diff (or a minimal repro),
the finding PRism posted, and what it should have said instead. Those cases
often become golden eval cases.

## License

By contributing, you agree your contributions are licensed under the MIT
License (see [LICENSE](LICENSE)).
