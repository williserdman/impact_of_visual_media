# Task 9 implementation report: complete over-context article text

## Delivered

Canonical Markdown at or below the conservative 8,192-token ceiling retains
the existing one-input hosted path with `truncate=false`. Longer Markdown is
tokenized locally with the official `jinaai/jina-embeddings-v4`
`tokenizer.json` at immutable revision
`d1e5d70b7b34d927a8cddac458583c4fbe50a914`; the acquired bytes must match
SHA-256 `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa`
before loading. Automated fixtures inject an offset tokenizer and never acquire
an artifact or use the network.

Oversized Markdown is greedily packed as complete blank-line-delimited blocks.
One block that exceeds the ceiling is partitioned only at safe tokenizer
offsets, including byte-fallback cases where several tokens cover one Unicode
codepoint. Every part is an exact source substring; ordered concatenation is
the original Markdown with no dropped, paraphrased, overlapping, or reordered
content.

Each part has content-free input identity, token count, lifecycle/attempt
state, generation identity, float32-normalized vector hash, and operational
vector. Successful parts commit independently before the next hosted call.
Interrupted and retryable runs therefore buy only unproven parts. Once all
parts succeed, the coordinator publishes exactly one research-facing
`article_text` vector: the float32 L2-normalized token-count-weighted mean of
all normalized part vectors. Append-only aggregate provenance identifies every
part generation/hash/weight. Parts never enter the research `embeddings`
relation as observations.

Changed Markdown uses a new article-input identity; tokenizer revision,
ceiling, boundary rules, aggregation, or other profile changes use a new
configuration identity. Old parts remain content-free auditable generations
but cannot satisfy the new aggregate. Existing article-text invalidation also
archives/removes the same-configuration dependent multimodal generation before
replacement.

## Schema and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `7` to `8`.
- Added operational `long_text_parts`, keyed by article, configuration,
  complete article input hash, and part index.
- Added append-only `article_text_aggregation_provenance`, keyed by article,
  configuration, aggregate generation, and part index.
- The catalog now has exactly nine base tables, exact columns/constraints, and
  no indexes.
- Added an exact prior-v7 refusal fixture by removing only the two long-text
  tables and pinning metadata to version 7. It is refused without migration or
  mutation; versions 1 through 6 remain refused.
- No new relation contains Markdown, excerpts, image bytes, credentials,
  request bodies, adapter responses, or error text.

## Red-green evidence

The exact schema fixture first observed RED because `long_text_parts` did not
exist: `1 failed, 106 deselected in 7.19s`. The v7 refusal fixture separately
observed RED because the current catalog was still accepted:
`1 failed, 107 deselected in 2.92s`. After the minimal schema/version change,
their combined GREEN was `2 passed, 106 deselected in 1.51s`.

The first public-coordinator long-text fixture observed RED:
`1 failed, 108 deselected in 2.90s`; the adapter received one complete
over-context string rather than the expected two lossless block-packed inputs.
After durable part publication and weighted aggregation, the same selection
observed GREEN: `1 passed, 108 deselected in 3.69s`.

The immutable tokenizer tests observed RED at collection because the tokenizer
module did not exist, then at the adapter seam because tokenizer injection was
unsupported. Checksum verification, immutable resolver arguments, and local
offset injection observed GREEN: `3 passed, 22 deselected in 0.59s`.

The read-only validator fixtures observed RED:
`2 failed, 113 deselected in 9.93s`; deleting one part-provenance row and
self-consistently replacing a part vector/hash were both invisible. After
complete linkage and independent token-weighted recomputation, those fixtures
plus fresh-output validation observed GREEN:
`3 passed, 112 deselected in 11.55s`.

The overlapping byte-fallback-offset fixture observed RED with the stable
content-free `invalid_tokenizer_offsets` failure. Accepting overlapping token
spans while splitting only between non-crossing offset groups produced GREEN
with the ordinary oversized-block fixture:
`2 passed, 116 deselected in 8.93s`.

## Focused verification

No full repository suite was run. Fresh focused evidence before commit:

- all exact-boundary, multi-block, oversized-block, byte-fallback,
  interruption/failure/resume, validation, schema-v8, and v7-refusal fixtures:
  `9 passed, 109 deselected in 28.84s`;
- established direct-text, three-modality, configuration identity, and fresh
  validation fixtures: `4 passed, 114 deselected in 7.26s`;
- established text/multimodal invalidation for changed input and explicit
  reprocess: `2 passed in 28.78s`;
- established changed-configuration isolation: `1 passed in 3.85s`;
- complete hosted-adapter synthetic module, including fixed
  `truncate=false` request serialization and tokenizer verification:
  `25 passed in 0.58s`;
- affected authorized CLI fixtures: `2 passed, 17 deselected in 11.20s`;
- changed-file Ruff: `All checks passed!`;
- generated embedding smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live worktree-source `run --help` retained the three explicit roots,
  required positive `--limit`, bounded `--reprocess`, and explicit hosted
  authorization.

- `git diff --check` produced no output;
- the tracked generated-artifact audit produced no matches (expected `rg`
  exit 1);
- the current-facing schema/aggregation stale-claim audit produced no matches;
  and
- all relative links in the two current operator documents resolve.

No licensed archive, credential, hosted request, tokenizer download, real
embedding output, machine configuration change, or full-corpus operation was
used.

## Self-review

- The direct/split decision uses the same configured tokenizer and inclusive
  ceiling. Every hosted text call continues through the adapter contract that
  serializes `truncate=false` and `return_multivector=false`.
- Block separators remain attached to exact source substrings. Oversized-block
  candidates are re-tokenized beneath the ceiling and a boundary is accepted
  only when the next token does not cross it. Fixture concatenation checks make
  loss, overlap, reordering, and rewriting observable.
- Part primary keys contain configuration and the complete article-input hash;
  part rows additionally retain their own exact substring hash. A deterministic
  plan mismatch under the same identity fails closed.
- `in_progress` commits before each part call; normalized vector/hash/success
  commits immediately afterward. Resume trusts only a successful row with
  generation, hash, and vector, then re-verifies dimension, finiteness, norm,
  and hash before aggregation.
- The aggregate transaction publishes the canonical vector/work checkpoint and
  all part linkage together. The weighted mean uses every positive token count,
  is L2-normalized, and is float32-quantized before hashing/publication.
- Run observation counts remain research-modality counts: operational parts do
  not inflate published embeddings or become independent observations.
- Read-only validation needs neither tokenizer nor credential. It verifies
  operational identity/state/vector facts, exact provenance coverage and
  linkage, then independently recomputes the aggregate.
- Existing registration remains the single invalidation seam, so text changes
  archive/remove both the old text aggregate and its dependent multimodal
  vector while preserving header image, siblings, and other configurations.

## Concerns

The first real authorized run may need to acquire the pinned tokenizer artifact
into the Hugging Face cache before local token accounting; acquisition is by
immutable revision and checksum-verified, but it is still a separate network
dependency from hosted embedding inference. Automated and smoke verification
do not exercise that acquisition. The shared virtual environment's installed
console script is not linked to this worktree, so smoke/help use
`PYTHONPATH=src`.
