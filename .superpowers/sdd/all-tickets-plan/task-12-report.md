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

## Correction round: atomic visibility and bounded cleanup

Review found that version-13 reconciliation committed physical removals page by
page. An interruption on page two therefore leaked page-one removals despite
the run remaining incomplete. Version 14 replaces that protocol with exact
run-scoped actions, private physical storage, and canonical public
`embeddings`/`embedding_work_items` views. Each bounded action row binds its
expected source path, input hash, state, generation run ID, and vector hash.
Staging is invisible; one run-marker commit exposes the complete overlay;
bounded compaction archives and materializes exact action rows without changing
the public result. Failed/interrupted invisible actions are discarded in
bounded pages before later work. The validator rejects action/base divergence
and treats visible uncompacted generations as logical run history.

The page-two regression first failed because the first article disappeared
from the public query (`1 failed in 8.99s`), then passed with atomic views (`1
passed in 5.20s`). Marker-before-compaction and compaction interruption/resume
passed `2 passed in 9.80s`. Exact schema-v14 and byte-preserving prior-v13
refusal passed `2 passed in 2.42s`. The action mismatch validator first lacked
the stable issue, then passed `1 passed in 3.25s`.

Header reconciliation now stages exact configuration/modality actions rather
than grouping by article. Generated `NULL`-to-new, path-A-to-path-B, and
new-to-`NULL` coexistence fixtures prove that the just-processed matching
configuration remains current while mismatched prior header/dependent
multimodal work becomes content-free `stale_input` or `not_applicable`.
Action staging and compaction callbacks both prove a page size of one even when
an article has multiple modalities/configurations. The stale-input validator
fixture first reported applicable-input hash/path errors, then passed `1 passed
in 4.59s` after adding its explicit generation-free disposition.

Rendition cleanup no longer calls `fetchall` or builds a set of every historical
path. It descriptor-scans one verified candidate at a time and asks the catalog
an exact `LIMIT 1` current/historical provenance question. The incremental
candidate fixture preserves a historical reference and removes only the orphan
(`1 passed in 1.48s`); the existing post-commit interruption/resume fixture also
remains green.

This correction used generated temporary catalogs and images only. It did not
read the licensed archive, call a hosted API, run a real full corpus, mutate
preprocessing output, or update an external issue/ledger.

Final correction verification passed the atomic staging, marker, compaction,
and expected-identity validator selection (`4 passed in 22.00s`); all header
association transitions (`3 passed in 22.55s`); schema/refusal, stale-input,
and cleanup selection (`5 passed in 11.13s`); and a final cross-slice selection
after binding vector source/input identity (`5 passed in 28.65s`). Changed-file
Ruff and `git diff --check` passed. The generated embedding smoke returned
`{"articles": 1, "embeddings": 1, "validation_ok": true}`, and live run help
showed the required `(--limit LIMIT | --full)` scope.
Pending visible article-text, header-image, and multimodal generations are also
treated as logical history until compaction; the generated multimodal marker
fixture validated cleanly (`1 passed in 6.29s`).

## Correction round 2: monotonic cursors and canonical-view integrity

Action staging and physical compaction now expose explicit composite cursors
through their bounded catalog page seams. Staging advances strictly by
`(article_id, modality, configuration_id)` and compaction by
`(run_id, article_id, modality, configuration_id)`, with the SQL `LIMIT`
applying to exact action rows. Neither loop restarts at the remaining
relation's beginning. The generated observer fixture first failed because the
catalog accepted no cursor (`1 failed in 13.98s`), then passed after both
coordinator loops forwarded each returned cursor (`1 passed in 7.07s`).

Schema 14 remains unchanged. Its exact validation now fingerprints normalized
behavior-bearing SQL returned by `duckdb_views()` for both canonical views.
The same-column storage bypass initially opened successfully (`1 failed in
3.63s`); the parametrized `embeddings` and `embedding_work_items` mutations
then refused byte-for-byte alongside the exact schema fixture (`3 passed in
1.75s`).

The `stale_input` validator now requires a header association equal to the
retained canonical article's current path, including null, and requires null
when the article disappeared. Generated retained-wrong-safe-path and
disappeared-nonnull fixtures initially produced no issue (`2 failed in
16.71s`), then passed with the valid content-free baseline (`3 passed in
16.12s`). README and crash-course operator contracts enumerate and define this
state.

Final focused lifecycle verification covered cursor progression, both altered
views, exact schema 14, all three stale-input cases, staging-page rollback,
atomic marker visibility, interrupted/resumed compaction, overlay/base identity
checking, and pending multimodal history (`12 passed in 49.30s`). Changed
Python files passed Ruff and `git diff --check`; the final cursor and both view
mutation cases passed `3 passed in 8.72s`. The generated smoke returned
`{"articles": 1, "embeddings": 1, "validation_ok": true}`, and live run help
showed `(--limit LIMIT | --full)`. All fixtures used temporary generated
catalogs and content; no hosted request, network access, licensed archive
traversal, real full run, or external issue/ledger update occurred.
