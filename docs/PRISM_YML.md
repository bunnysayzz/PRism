# `.prism.yml` - per-repo review config

Drop a `.prism.yml` file in your repo root and PRism picks it up at review
time. It is fetched with the installation token (so private repos work),
then merged over the server-side config. Anything it cannot parse is
silently ignored: a missing file, bad YAML, or a non-mapping all fall back
to the server defaults. The file contents are never logged.

## Quick start

```yaml
# .prism.yml
review_profile: chill          # "chill" | "assertive": preset strictness
rules:
  - "Always use structured logging; never print() in production code."
  - "New database migrations must be backward compatible (expand before contract)."
ignore_globs:
  - "**/generated/**"
```

## Full reference

| Key | Type | Default | What it does |
|---|---|---|---|
| `review_profile` | `chill` \| `assertive` | unset | Preset for the strictness knobs below. `chill`: post only at confidence >= 0.90, verify at >= 0.75, max 5 findings/file, threshold `medium`. `assertive`: post at >= 0.70, verify at >= 0.50, max 15 findings/file, threshold `low`. Explicit keys in your file always beat the preset. Unknown values are ignored with a warning. |
| `severity_threshold` | `low` \| `medium` \| `high` \| `critical` | `low` | Findings below this severity are dropped. |
| `max_findings_per_file` | int | `10` | Cap on findings posted per file. |
| `instructions` | string | empty | Free-form review instructions for this repo, injected into the reviewer prompt as a "Repository review instructions" section. |
| `rules` | list of strings | `[]` | Plain-English team standards. Each entry becomes a binding review rule ("- ..." bullet) in the prompt. Unioned with the server-side rules, deduped, order preserved. |
| `ignore_globs` | list of strings | `[]` | Extra paths PRism never reviews. Added on top of the server defaults (lockfiles, `dist/`, `node_modules/`, generated protobuf, etc.). |
| `categories` | map | `{}` | Enable/disable finding categories, e.g. `style: false` re-enables a category the server turned off. |
| `path_instructions` | map | `{}` | Glob -> instruction pairs, e.g. `"**/api/**": "This is a public API surface; flag any breaking change."` |
| `model_routing` | map | `{}` | Server-defined model routing overrides for this repo. |
| `confidence_auto_post` | float | server value | Findings at/above this confidence post directly without the verify pass. |
| `confidence_verify` | float | server value | Findings at/above this confidence get the second-pass verifier; below it they are dropped. |
| `tracker` | map | `{}` | Issue-tracker binding for ticket-scoped reviews (see below). Holds a token, so it is repo-wins outright when present and never merged or logged. |

## Ticket-bound review (`tracker`)

Link a Linear or Jira ticket so the bot checks the diff against the
ticket's acceptance criteria. PRism looks for ticket refs in the PR title,
body, and branch name: Jira/Linear-style keys (`PROJ-123`) and GitHub
issue refs (`#45`).

```yaml
tracker:
  provider: linear        # "linear" | "jira"
  api_token: ${{ secrets.PRISM_TRACKER_TOKEN }}  # do NOT hardcode tokens
```

For Jira you also need `base_url`:

```yaml
tracker:
  provider: jira
  base_url: https://your-company.atlassian.net
  api_token: ${{ secrets.PRISM_TRACKER_TOKEN }}
```

Notes:

- The token is only ever read at review time from the merged config; it is
  never logged, never appears in exceptions, and never leaks into review
  text. Only the provider name and ticket ref are logged.
- Store the token as a GitHub secret on the repo and reference it with
  `${{ secrets.* }}` if your workflow templates the file; PRism reads the
  value that lands in the file. Never commit a raw token.
- Without a `tracker` section, PRism still parses ticket refs but has no
  acceptance criteria to check against.

## Merge precedence

Highest wins:

1. Explicit keys in `.prism.yml` (scalars; `categories` and
   `path_instructions` are merged per-key with repo winning conflicts).
2. The `review_profile` preset, for keys the file does not set.
3. The server-side `config.yaml` / env defaults.

`ignore_globs` and `rules` are unioned across both levels instead of
replaced. `tracker` is never unioned: a present, non-empty repo `tracker`
replaces the server one outright.

## Example: strict security-focused repo

```yaml
# .prism.yml
review_profile: assertive
severity_threshold: medium
categories:
  style: false
rules:
  - "All user input must be validated at the boundary before use."
  - "No new dependencies without a security review note in the PR."
  - "Secrets and credentials must come from env/config, never code."
path_instructions:
  "**/auth/**": "Auth code: flag any logic change, even small."
ignore_globs:
  - "**/migrations/**"
tracker:
  provider: linear
  api_token: ${{ secrets.PRISM_TRACKER_TOKEN }}
```

## Where this lives in the code

- `src/prism/github/repoconfig.py` - fetches and parses `.prism.yml` (never raises).
- `src/prism/config.py` - `load_repo_config`, `merge_repo_config`, `build_team_standards`.
- `src/prism/github/tracker.py` - ticket context fetch (Linear/Jira).
- `tests/test_repoconfig.py` - merge and parse behavior.
