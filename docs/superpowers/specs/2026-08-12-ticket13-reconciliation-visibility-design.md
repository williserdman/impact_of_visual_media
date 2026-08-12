# Ticket 13 reconciliation visibility correction design

## Goal

A failed or interrupted full run must expose no removal inferred by that run.
Full reconciliation must remain bounded, and successful reconciliation must
become visible atomically without a corpus-sized transaction. Header
association changes must affect only configurations whose stored association
is stale. Rendition cleanup must not materialize all historical provenance
paths.

## Canonical read boundary

Schema version 14 separates physical storage from the canonical read
interface. Existing mutable embedding and work rows move to private physical
base tables. Public `embeddings` and `embedding_work_items` views expose the
current logical state. Every pipeline, validator, documented research query,
and catalog read uses those views. Production mutations target the base tables.

A new run-scoped reconciliation-action table stores content-free exact keys and
the complete replacement work disposition. An action is invisible while its
run is not reconciliation-visible. Public views anti-join vector-removal
actions from visible runs and replace affected work rows with their staged
disposition. Thus a single run marker is the visibility commit for the entire
action set.

## Staging and promotion

After successful full discovery and processing, the coordinator stages exact
reconciliation actions in lexical keyset pages. Staging does not mutate
physical active rows or generation history. Page transactions are independently
retryable and idempotent. Failure on page two after page one commits leaves the
run non-visible, so both public views remain byte-for-byte equivalent to their
pre-reconciliation state.

Once staging exhausts, one transaction validates the run and flips its
reconciliation-visible marker. All staged removals and work replacements then
appear through the public views at once. There is no corpus-sized promotion
transaction.

## Bounded physical compaction

After visibility commits, the coordinator drains committed actions in bounded
pages. Each page archives any superseded generation, removes the physical
vector, materializes the exact replacement work row, and deletes the drained
action. The public views preserve the same logical result before, during, and
after compaction: a remaining action overlays old storage; a drained action has
already materialized its replacement. Interruption is harmless, and the next
full run drains committed actions before new processing. Validation accepts
pending committed actions only when their overlay and base rows are mutually
consistent.

## Header association targeting

Reconciliation decisions are keyed by article, modality, and configuration.
For a retained article, the coordinator compares each configuration's stored
header source path with the newly discovered association. A matching path,
including matching `NULL`, is preserved. A differing path stages replacement
for that configuration's header and dependent multimodal work only. This keeps
the just-processed configuration active while making prior configurations
stale for `NULL` to new, path A to path B, and new to `NULL` transitions.

Disappeared articles stage removal/replacement for every modality and retained
configuration.

## Rendition cleanup

The rendition descriptor scanner yields one verified candidate at a time. For
each candidate, the catalog performs an exact indexed-equivalent membership
query with `LIMIT 1` against all current and historical image provenance. No
`fetchall`, corpus-sized set, or history-sized Python collection is built.
Only an unreferenced verified candidate is unlinked.

## Tests and safety

Generated fixtures exercise the public views and coordinator:

- page-two staging failure after page one commits leaves every visible vector
  and work row unchanged;
- the one-row visibility commit exposes all staged removals before compaction;
- interruption during compaction preserves the exact same public view, and a
  resumed run drains actions in bounded pages;
- header `NULL` to new, path A to path B, and new to `NULL` preserve the
  matching current configuration and stale only mismatched configurations;
- cleanup checks candidates incrementally and preserves a referenced
  historical rendition.

Exact schema tests describe version 14 base tables/views/actions and prove a
byte-for-byte refusal of version 13 output. No fixture reads the licensed
archive, performs a hosted request, or runs a real full corpus.
