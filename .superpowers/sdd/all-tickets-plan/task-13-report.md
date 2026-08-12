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
