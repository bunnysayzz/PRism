# PRism sample dataset

`records.jsonl` — 12 handcrafted `RawSample` records (see
`src/prism/data/models.py`) used by the pipeline tests and as a smoke-test
input for the CLI:

    python scripts/collect_pr_data.py --input data/samples/records.jsonl \
        --processed /tmp/prism-smoke

Contents: realistic Python/JavaScript diffs with reviewer comments and
handcrafted `finding` labels in the `Finding` schema (`src/prism/review/schemas.py`).

- 4 repos (`acme/shop-api`, `acme/web-ui`, `beta/toolbox`, `gamma/cli`) so the
  repo-stratified split test has multiple groups.
- Severities span critical/high/medium/low; categories span
  security/correctness/performance/concurrency/style.
- Records `c3` (`config/settings.py`) and `c12` (`src/deploy.js`) contain
  hardcoded fake secrets so the cleaner's secret-redaction path is exercised
  end to end (`tok_test_…` / `dpl_…` are synthetic, not real credentials).

Keep this file small enough for git (it is test fixture data, not training data).
