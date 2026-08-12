# Task 13 implementation report: complete embedding corpus validation

## Delivered

The public `validate_embedding_corpus` interface and `wsj-embeddings validate`
command now open the exact embedding and preprocessing catalogs read-only and
return one deterministic, content-free result for a selected immutable
configuration. The CLI requires the same three explicit disjoint roots as
inventory/run and accepts `--configuration-id`; when multiple configurations
coexist, omitting that identity produces the existing stable selection issue.
An exact but empty embedding catalog is incomplete when canonical articles
exist and now reports `missing_embedding_configuration`.

Coverage accounts for canonical articles, text success/failure, header
present/absent/success/failure, multimodal success/unavailable/failure, stale
work, unresolved retryable work, and orphan generated rendition leaves. It is
computed in the same read-only embedding-catalog snapshot as integrity
validation. Orphan traversal descriptor-opens verified directories and never
follows a directory or leaf symlink.

The existing validator remains the integrity engine. Exact schema-14 tables,
views, columns, types, nullability, primary-key constraints, absence of
indexes, schema version, and behavior-bearing canonical-view SQL fingerprints
are checked by `EmbeddingCatalog.read_only` before relational or artifact
checks. The selected pass then validates vectors and current input/config
identity, multimodal recomputation, long-text parts/aggregates, image input and
rendition provenance, work/failure states, terminal runs, and visible pending
reconciliation actions/compaction semantics.

Generation history now has a global content-free validation pass. It rejects
unsupported modalities even outside the selected configuration, and every
history row must resolve an existing configuration, a generating run with that
configuration, and its durable `(article_id, modality, configuration_id)` work
identity. Hosted adapter usage normalization now persists the parsed finite
nonnegative numeric value instead of retaining a validated numeric string.

README and the architecture crash course document the five-command embedding
CLI, safe credential-free validation workflow, configuration selection, and
coverage categories. No catalog shape changed, so schema version 14 remains
correct.

## TDD evidence

- Public fresh-corpus CLI RED: `1 failed in 5.41s` because `validate` was not a
  registered subcommand. GREEN: `1 passed in 6.42s` with deterministic coverage,
  no adapter construction, and an unchanged source/preprocessing/embedding
  snapshot.
- Numeric usage RED: `1 failed in 0.47s` because `_safe_usage` returned
  `"11"`/`"11.5"`. GREEN: `1 passed in 0.31s` with `11.0`/`11.5`.
- Generation identity RED: `2 failed in 8.06s` because detached history rows
  produced no stable identity issue. GREEN: `2 passed in 9.20s`.
- Empty exact-catalog RED: `1 failed in 2.61s` because a canonical corpus with
  no configuration returned no issue. GREEN: `1 passed in 1.43s`.
- Fresh public CLI, generation references, and rendition behavior passed
  `5 passed in 17.44s`; explicit selection plus incomplete/stale/retryable/
  orphan coverage passed `2 passed in 10.64s`; tampered, malformed, and
  unsupported public fixtures passed `3 passed in 7.40s`.

## Focused verification

- Single-snapshot public coverage/refactor selection: `5 passed in 16.34s`.
- Malformed/unsupported schemas, history identity, and malformed history:
  `5 passed in 11.87s`.
- Supported history modalities plus numeric usage: `4 passed in 17.21s`.
- Visible pending multimodal reconciliation plus no-follow orphan scan:
  `2 passed in 8.84s`.
- Hosted adapter file: `34 passed in 1.17s`.
- Changed Python/test files: Ruff reported `All checks passed!`.
- `git diff --check` was clean.
- Live help showed `{smoke,pilot,inventory,validate,run}` and the validate roots
  plus optional configuration selector.
- Generated embedding smoke returned
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`.
- Generated preprocessing smoke returned two articles, three discovered
  candidates, one duplicate, zero failures, and `validation_ok: true`.
- Tracked generated-data audit found no paths below `data`, `artifacts`, or
  `outputs`; current README/crash-course relative links resolve in the tree.

A whole `tests/test_embeddings_cli.py` attempt exposed the already documented
ticket-15 clean-install boundary: three production-Jina hosted-failure tests
received `image_codec_unavailable` in the shared environment because its exact
Pillow/codec build is unavailable. The remaining isolated CLI nodes reported
five passes, and the ticket-14 public validation CLI node passes. This task did
not mutate the shared environment; ticket 15 owns the clean temporary install
and production Pillow verification. No full suite was run.

## Safety and self-review

- All fixtures use generated Markdown, images, catalogs, vectors, and orphan
  leaves below pytest temporary roots.
- Validation never constructs a Jina adapter, reads a credential, makes a
  network call, opens a writable catalog, or changes/removes an artifact.
- Malformed/unsupported catalog bytes remain unchanged after validation; no
  migration or repair path exists.
- Coverage and issues contain counts, stable identities, and stable codes only;
  no article excerpt, vector value, credential, source bytes, or path is
  emitted.
- No real archive traversal, live hosted call, real full run, generated-data
  commit, machine configuration change, or shared-environment mutation was
  performed.

## Review correction round 1

Review found that a no-header canonical article could still count an old
visible composite as multimodal success, producing success plus unavailable
and a negative failure count. A generated header-removal fixture reproduced
that exact result (`1 failed in 8.46s`). Coverage now counts multimodal
success/failure only among currently header-eligible canonical identities; the
remaining no-header identities are unavailable. The partition, incomplete
coverage, and fresh CLI cases pass `3 passed in 12.10s` with explicit total and
nonnegative assertions.

Every vector exposed by the canonical `embeddings` view now has a global
invariant requiring an exact public work key in `succeeded` state, matching
source path/input hash, a non-null generation run, and a same-configuration run
reference. Orphan-key and null-hash stale-work fixtures first failed to produce
the stable issue (`2 failed in 5.62s`), then passed together with visible
reconciliation compatibility (`3 passed in 8.63s`).

Catalog validation now opens embedding first, begins its read transaction
before exact schema validation, opens preprocessing second, begins its read
transaction before exact schema validation or article queries, and retains
both until the complete validation and coverage pass ends. These are
independently stable snapshots acquired in that order. Separate writers mean
their pinned versions need not ever have been simultaneously current;
cross-catalog relational inconsistencies are reported, but validation is not
an atomic cross-catalog snapshot. Filesystem state is also not claimed atomic:
content-free start/end tokens cover all canonical Markdown/current headers, all
referenced renditions, and every no-follow rendition namespace/leaf identity
and regular file hash. In-pass changes produce
`concurrent_validation_state_change`.

The public failpoint fixture first failed because the snapshot callback was not
invoked (`1 failed in 8.68s`), then passed (`1 passed in 2.76s`) while proving a
concurrent Markdown replacement, failed validation, and unchanged catalog
bytes. Concurrent Markdown and whole-namespace orphan addition pass together
(`2 passed in 5.46s`). The complete correction selection passed `10 passed in
27.53s`; exact schema/prior-schema refusal passed `5 passed in 4.40s`. The
single-use rendition forwarding helper was removed. All correction fixtures
remain generated, credential-free, offline, and read-only except for their
explicit temporary artifact mutation.

Final correction verification covered all new findings plus malformed
preprocessing and the public CLI (`8 passed in 21.90s`), representative fresh,
multimodal, long-text, vector/publication, and reconciliation validation (`6
passed in 23.64s`), image/history/orphan validation (`4 passed in 14.90s`), and
all run-metric/lifecycle parameter cases (`13 passed in 28.96s`).

The final read-only compatibility and correction selection passed `14 passed
in 31.99s`. It included the exact version-14 schema, version-13 refusal,
malformed canonical-view and preprocessing-catalog refusal, immutable malformed
catalog bytes, the public validation CLI, reconciliation-visible pending work,
both concurrent-artifact fixtures, the coverage partition, and both global
public-vector checkpoint corruptions. Ruff and `git diff --check` passed, live
embedding help still exposes `{smoke,pilot,inventory,validate,run}` and the
validate roots/configuration selector, and both generated smokes returned
`validation_ok: true`. The three previously observed environment-dependent CLI
failures are unchanged and were not broadly rerun: this correction does not
touch the hosted image-codec path, and ticket 15 still owns verification in its
clean temporary environment. No whole module or full suite is claimed green.

## Review correction round 2

Documentation and current validator docstrings no longer describe the two
independently pinned catalog states as a pair that necessarily existed at one
instant. The embedding snapshot is acquired first and preprocessing second;
because the catalogs have separate writers, the pinned versions need not ever
have been simultaneously current. Validation reports relational inconsistencies
across the pinned states but does not provide an atomic cross-catalog snapshot.
The filesystem guarantee remains start/end detection of in-pass changes across
referenced artifacts and the complete no-follow rendition namespace.

The tracked-Markdown/current-docstring stale-language search found no remaining
claim of jointly current catalog state at acquisition or one shared read-only
catalog snapshot. README and crash-course relative links resolved, Ruff passed
for the docstring-only Python edit, and `git diff --check` was clean. No
behavioral test was run because this round changes wording only and the
repository has no focused documentation wording assertion.
