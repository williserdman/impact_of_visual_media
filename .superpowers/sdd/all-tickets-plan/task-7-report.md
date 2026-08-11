# Task 7 implementation report: header-image eligibility and retries

## Delivered

Header-image work now has an explicit, content-free disposition for every
selected article and configuration. A true no-header article is
`not_applicable`; a readable header progresses through `queued`, `in_progress`,
and `succeeded`; hosted/transport failures remain `retryable`; deterministic
image-request rejection becomes `terminal` on its third attempt; and an
interrupted request remains recoverable. Changed exact bytes archive the prior
image generation with `input_changed`, invalidate only that header vector and
its same-article/configuration multimodal dependent, and preserve successful
text, sibling articles, and prior configurations.

A declared but unavailable local header is the only applicable work allowed to
have a null input hash. It is a narrowly validated `retryable` checkpoint with
the safe canonical relative path, `missing_header_image`, zero adapter attempts,
no status/retry metadata, no generation identity, and no active vector. It does
not fail or regenerate independently successful text. Creating the file later
queues and publishes only the image vector.

Image adapter failures no longer abort a completed text publication. Public
coordinator and CLI summaries include `header_absent` and `header_failed`, so a
valid no-header article is not conflated with missing, in-progress, retryable,
or terminal image work. Read-only validation emits stable content-free issue
codes for each incomplete/failed image disposition and remains silent for a
valid `not_applicable` image.

## Schema and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `5` to `6`.
- Added non-null integer `header_absent` and `header_failed` columns to `runs`.
- Updated exact schema creation/inspection, run initialization, counter
  whitelisting, result reads, coordinator writes, CLI JSON, tests, README, and
  architecture documentation together.
- Added an exact prior-v5 refusal fixture by generating a catalog, removing only
  the v6 run columns, and pinning metadata to version 5. It is refused read-only
  without migration, overwrite, repair, or deletion. Versions 1 through 4
  remain refused.
- No table stores Markdown, article excerpts, image bytes, vectors in failure
  rows, remote URLs, credentials, request bodies, or adapter responses.

## Red-green evidence

The initial generated lifecycle/schema selection was:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'exact_version_six or refuses_version_five or missing_header_is_retryable or retryable_header_failure or deterministic_header_rejection or changed_header_bytes or interrupted_header_attempt'
```

Observed RED: `6 failed, 1 passed, 82 deselected in 15.72s`. Missing input
aborted before output, image adapter failures escaped the coordinator,
deterministic rejection became terminal on the first attempt, image-byte
changes left the same-configuration multimodal vector active, interrupted image
work was invisible to validation, and the catalog remained schema v5. The one
pass was the first weak metadata-only refusal fixture.

That refusal fixture was strengthened to transform a generated current catalog
into the exact prior shape. Before the bump it observed the intended RED:
`1 failed, 88 deselected in 2.88s`, because version 5 remained accepted.

After the smallest schema/catalog/coordinator/validation implementation, the
same seven-case selection observed GREEN:
`7 passed, 82 deselected in 21.85s`.

The broader coordinator run exposed eight existing exact result assertions that
needed the new `header_absent` field. After updating those generated-fixture
expectations, the affected rerun was `8 passed, 81 deselected in 17.49s`.
The public CLI module, including generated absent-vs-missing nonzero summary
fixtures, then passed `19 passed in 16.22s`.

## Final focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings tests/test_embeddings_pipeline.py \
  tests/test_embeddings_cli.py
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings smoke
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- focused coordinator/catalog/validation/CLI tests:
  `108 passed in 193.46s`;
- changed Python Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live module help retained the three explicit roots, required positive
  `--limit`, article-text-only `--reprocess`, and explicit hosted authorization;
- `git diff --check` produced no output;
- current-facing README/crash-course searches found no schema-v5 or v1-v4
  refusal claims; and
- the tracked generated-artifact audit produced no matches (expected `rg`
  exit 1).

No full repository suite, licensed archive, live credential, network request,
real embedding output, or full-corpus operation was used.

## Self-review

- Missing input is classified before output mutation but becomes durable only
  under the coordinator lock. Unsafe, symlinked, hard-linked, and replaced
  source paths retain their existing fail-closed behavior and never become the
  missing exception.
- The missing exception is modality-, state-, error-, path-, hash-, attempt-,
  generation-, and vector-specific. General applicable work still requires an
  exact lowercase SHA-256 identity.
- Text success commits before image work. Image failure commits only stable
  codes and numeric metadata, increments a failed-header summary, and continues
  without a placeholder vector. Text adapter failures retain their existing
  fail-fast behavior.
- Provider-declared retryability is kept separate from the explicit
  deterministic image policy. Only `deterministic_request` receives the
  bounded three-attempt transition; other nonretryable failures remain
  terminal immediately.
- Exact-byte changes reset the affected image attempt lifecycle, archive the
  active image and multimodal generations, and never touch article text or a
  different configuration. Configuration changes retain independent old and
  new work keys.
- A success commit writes the vector, immutable `generation_run_id`, work state,
  and run counters together. Recovery reuses unchanged text and image successes
  and purchases only incomplete image work.
- Run summaries account for every selected header as absent, successful, or
  failed without storing identities or content in the summary.

## Concerns

None. The shared virtual environment's installed console script is not linked
to this worktree, so smoke and live help were verified against worktree source
with `PYTHONPATH=src`. No machine configuration changed.
