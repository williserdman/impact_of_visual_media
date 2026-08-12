# Task 12 implementation report: explicit full embedding reconciliation

## Delivered

`wsj-embeddings run` now requires exactly one positive `--limit` or explicit
`--full`; hosted-processing authorization remains independently mandatory.
Invalid or ambiguous scope is rejected by argument parsing before output or
adapter construction.

Full runs hold one validated read-only preprocessing snapshot and discover
canonical articles in stable lexical `article_id` keyset pages. Each bounded
page commits only article identity and its optional source-relative header
association to a durable downstream run inventory. The coordinator later
reopens and processes those identities through a bounded public page seam; it
never constructs a corpus-sized Python inventory or hands the entire durable
inventory to the existing coordinator at once.

Reconciliation is reachable only after discovery and every processing page
complete. Bounded reconciliation transactions remove active article vectors
for disappeared canonical articles across every retained configuration. Header
association disappearance leaves article text active, makes header work
`not_applicable`, and stales dependent multimodal work across configurations.
A failed reconciliation page rolls back only that page and leaves the run
non-complete; limited, discovery-failed, processing-failed, and interrupted
pre-reconciliation runs do not infer removal.

Generated image-rendition cleanup occurs only after the reconciliation catalog
state and successful run terminal state commit. It descriptor-scans only the
safe rendition namespace, does not follow symlinks, and unlinks only verified
unreferenced generated JPEGs. An interruption at that post-commit boundary
leaves a validator-visible harmless orphan and cannot turn an already completed
full reconciliation into an incomplete run. A later full run retries cleanup.
Licensed input and preprocessing output remain read-only.

## Schema and validation

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from 12 to 13.
- `runs` now records `scope`, terminal `status`, discovery and reconciliation
  completion flags, and nullable `finished_at`.
- New `full_run_articles` has primary key `(run_id, article_id)` and stores only
  the optional `header_image_path` association beside those identities.
- The catalog now has exactly twelve base tables and no indexes.
- Exact prior-v12 and malformed shapes are refused without mutation or
  migration.
- Offline validation rejects contradictory scope, terminal, completion, and
  durable-inventory state. Cross-configuration reconciliation run IDs remain
  valid supersession references.
- `stale_input` is an explicit non-published work state for canonical identity
  or modality input that disappeared during a completed full reconciliation.

## TDD evidence

The initial schema/CLI selection observed RED with `1 failed, 4 passed in
3.30s`: the durable full-run relation did not exist. Positive full CLI then
observed RED with `1 failed` because the coordinator still required a limit.
The exact CLI/schema selection turned GREEN with `6 passed in 5.54s`, and the
expanded exact schema/refusal selection passed `7 passed in 7.82s`.

The first full recovery selection observed `3 failed, 184 deselected in 3.22s`
because the coordinator had no full-page seam. Interrupted discovery,
cross-configuration article disappearance, and reconciliation-page rollback
then passed `3 passed, 184 deselected in 14.30s`.

Bounded discovery/processing, header disappearance across configurations, and
traversal failure first exposed one generated-fixture construction error; after
correcting fixture identity, the behavior passed `3 passed, 187 deselected in
17.44s`. Cleanup, header/traversal failure, and interruption recovery passed
`4 passed in 22.90s`. The final post-commit cleanup terminal-state node passed
`1 passed in 7.37s`.

The new run-lifecycle validator observed all six intended contradictions as
RED (`6 failed in 18.28s`) and then passed `6 passed in 16.79s`. The affected
CLI selection passed `9 passed, 15 deselected in 19.05s`; five metric defaults
were mechanically added to stale generated expected JSON payloads exposed by
that focused sweep.

Changed Python and test files pass Ruff, and `git diff --check` is clean. One
later four-node pytest wrapper lost its final summary after progress output, so
it is deliberately not claimed as verification evidence. No full test suite was
run.

## Self-review and safety

- Source-root validation and catalog traversal failure occur before any
  reconciliation transaction. Earlier published vectors remain intact.
- Full discovery and processing share one read-only snapshot; canonical rows
  cannot drift between durable discovery and page reopening.
- Full processing, inventory reads, and reconciliation all use positive bounded
  pages. No whole-corpus list, API request, or archive traversal was performed.
- Every reconciliation page is independently transactional. A failpoint inside
  the page proves rollback without false completion.
- Reconciliation invalidates active generations without deleting immutable
  supersession history. Header eligibility changes propagate to dependent
  multimodal state for every configuration.
- Cleanup queries committed rendition references, refuses unsafe namespace
  entries, and never mutates source images or preprocessing state.
- No network request, hosted credential, licensed archive read, real output,
  real `--full`, machine configuration change, or external issue/ledger update
  was used.

The durable inventory is intentionally retained as content-free run audit
state. It grows with completed and interrupted full runs; retention/compaction
policy is outside this ticket and should be designed explicitly rather than
silently deleting recovery evidence.
