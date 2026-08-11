# Task 4 implementation report: durable article-text checkpoints and resume

## Delivered

Limited article-text runs now checkpoint each selected work item in the separate
embedding catalog. The modality-ready `embedding_work_items` relation owns the
durable states `queued`, `in_progress`, `succeeded`, `retryable`, `terminal`,
and `interrupted`, keyed by article, modality, and embedding configuration.

The coordinator commits queued/recovered state before attempts and commits
`in_progress` before reading Markdown or invoking the adapter. It validates and
normalizes the returned vector before opening the success transaction. The
vector upsert, `succeeded` transition, and run-count update then commit together.
One article's committed success therefore survives a later item failure or
interruption.

On restart, an unchanged `succeeded` item is reused only when a matching vector
with the same input/configuration identity exists. `in_progress` work becomes
durably `interrupted` and is reattempted; `retryable` work is reattempted;
unchanged `terminal` work remains visible without an implicit adapter call.
Explicit `JinaHostedAdapterError.retryable` controls failure state. Failure
checkpoints retain only the stable code, numeric status/retry metadata, attempt
count, run ID, and timestamp—never exception text, Markdown, response bodies,
credentials, or vectors.

Run results and durable `runs` rows retain the compatible `articles` and
`embeddings` counts and add `reused`, `attempted`, `succeeded`, `retryable`,
`terminal`, and `interrupted`. CLI success JSON remains sorted and content-free;
classified failures retain their existing one-code error JSON.

## Schema and version changes

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `1` to `2`.
- Added `embedding_work_items` with an exact composite primary key and ordered
  lifecycle/provenance/failure metadata columns.
- Extended `runs` with six non-null content-free lifecycle counters.
- Kept the catalog index-free and updated exact table/column/constraint tests.
- Added explicit refusal coverage for version-1 output without mutation or
  migration. README and crash-course recovery guidance tells operators to move
  reproducible old derived output aside or select a fresh output root.
- Updated canonical article reads to carry the preprocessing Markdown SHA-256
  into checkpoint identity.
- Updated cross-artifact validation for lifecycle domains, failure metadata,
  and matching successful work/vector checkpoints.

## Red-green evidence

Exact schema RED:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py::test_embedding_catalog_has_exact_version_two_checkpoint_schema \
  tests/test_embeddings_pipeline.py::test_pipeline_refuses_version_one_embedding_catalog_without_migration
```

Observed `1 failed, 1 passed in 1.19s`: `embedding_work_items` did not exist.
After the schema-v2 creation/validation update, the same command reported
`2 passed in 6.50s`.

Public coordinator RED:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'replaying_unchanged or restart_recovers or checkpoint_survives or retryable_failure_can or terminal_failure_is'
```

Observed `7 failed, 39 deselected in 6.51s`. Existing code either invoked the
adapter before creating durable state or attempted one all-or-nothing run
transaction. After per-item checkpoints and recovery, the same selection
reported `7 passed, 39 deselected in 13.17s`.

Validation RED:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py::test_validator_reports_invalid_durable_work_state
```

Observed `1 failed in 3.11s`: invalid checkpoint state produced no validation
issue. After lifecycle/cross-artifact validation, that case plus fresh-output
and missing-vector cases reported `3 passed in 6.19s`.

Existing public CLI tests then went red in four expected places because hosted
and Markdown failures now intentionally leave durable content-free checkpoint
state rather than no output directory. Updated assertions require the stable
work state/metadata, zero vectors, owned-lock cleanup, unchanged one-code error
JSON, and absence of generated editorial text/secrets from catalog bytes. The
CLI file then reported `16 passed in 8.42s`.

## Generated interruption and failure coverage

- before request: injected adapter interrupts immediately after `in_progress`
  commits;
- after response: injected adapter obtains the generated fake response and
  interrupts before vector publication;
- before catalog commit: injected transaction failpoint rolls back vector and
  success together, and restart retries exactly once;
- after catalog commit: injected transaction failpoint raises after the vector
  and success commit, and restart reuses it with zero adapter calls;
- partial retryable failure: first article commits, second records a retryable
  429/rate-limit checkpoint, then a later run reuses the first and publishes one
  vector for the second;
- terminal failure: deterministic rejection remains visible and replay makes no
  adapter call;
- unchanged success replay: one vector remains and the second run records one
  reuse with zero attempts.

All fixtures use generated preprocessing catalogs/Markdown and injected fake
adapters, transports, or transaction failpoints. No network, credential,
licensed archive, real output, or full-corpus command was used.

## Final verification and exact output

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_cli.py tests/test_jina_hosted_adapter.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check .
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -c \
  'from wsj_embeddings.cli import main; raise SystemExit(main(["smoke"]))'
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings --help
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- coordinator/catalog: `47 passed in 44.82s`;
- CLI and hosted adapter: `37 passed in 8.37s`;
- Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live parser lists `{smoke,pilot,inventory,run}` and exact documented run roots,
  positive `--limit`, and `--authorize-hosted-processing` options;
- diff check clean; no tracked `data/`, `artifacts/`, or `outputs/` paths.

Per parent direction, the full repository test suite was intentionally not run;
verification stayed focused on every changed embedding boundary plus the hosted
adapter and generated smoke.

## Self-review

- The success transaction inserts/validates the vector before transitioning the
  work item, so no visible success can precede publication.
- An after-commit failure is proven by a real row and reused; a before-commit
  failure leaves `in_progress`, becomes `interrupted`, and reattempts.
- Earlier successes cannot be rolled back by later items because every attempt
  has bounded state and publication transactions under the existing
  descriptor-anchored exclusive lock.
- Retryable and terminal state comes only from the explicit hosted-adapter
  classification. Unexpected programmer exceptions still propagate and leave
  recoverable `in_progress` state.
- The primary key prevents duplicate research-facing vectors during retries.
- Limited selection and non-reconciliation outside the selected scope remain
  unchanged.
- Exact schema validation remains read-only before writable open and fails
  closed on old, extra-index, or malformed output.
- Documentation no longer claims that article-text resume is unavailable and
  identifies broader invalidation/full reconciliation as later boundaries.

## Concerns

None. The shared virtual environment still has no linked-worktree console-script
installation, so generated smoke and live help were verified directly against
the worktree source with `PYTHONPATH=src`; no machine configuration changed.
