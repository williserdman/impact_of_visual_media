# Task 9 implementation report: complete over-context article text

## Delivered

Canonical Markdown at or below the conservative 8,000-token input ceiling retains
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

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `7` to `8`, then to `9`
  for the immutable-generation and exact-span review fixes.
- Added operational `long_text_parts`, keyed by article, configuration,
  complete article input hash, and part index.
- Added append-only `article_text_aggregation_provenance`, keyed by article,
  configuration, aggregate generation, and part index.
- Added append-only operational `long_text_part_generations`; current part
  checkpoints point to these immutable vector generations.
- Added exact character and UTF-8 byte spans to current and immutable part rows.
- Added tokenizer-engine and bounded-part-attempt configuration identity.
- The catalog now has exactly ten base tables, exact columns/constraints, and
  no indexes.
- Added exact prior-v7 and prior-v8 refusal fixtures. The v7 fixture removes the
  original two long-text
  tables and pinning metadata to version 7. It is refused without migration or
  mutation; versions 1 through 6 remain refused.
- No new relation contains Markdown, excerpts, image bytes, credentials,
  request bodies, adapter responses, or error text.

## Review-fix round 1

Explicit reprocess no longer destroys the part generation named by an archived
aggregate. Each successful part is inserted into append-only
`long_text_part_generations` before the mutable checkpoint points at it.
Aggregate provenance resolves against that immutable relation for both active
article-text vectors and superseded article-text history. Validation recomputes
every active or archived aggregate hash and also compares the active vector.

Each planned part now records content-free `[char_start, char_end)` and
`[byte_start, byte_end)` spans. When an aggregate input is still canonical,
validation reopens the Markdown through the safe descriptor path and proves
the ordered spans are gapless, non-overlapping, cover all characters and UTF-8
bytes, and hash to the recorded part identity. Older input-changed generations
cannot be reopened because bodies are intentionally not retained; their spans
must still be ordered/gapless and all immutable vector/hash links must resolve.

Retryable part requests become terminal at the profile limit of three committed
attempts. The parent article-text work item mirrors that terminal disposition,
so later replay does not purchase the part again. The profile and catalog store
the limit as configuration identity.

The production input ceiling is now 8,000 tokens, leaving 192 tokens of framing
headroom below the confirmed 8,192-token model context. The tokenizer engine is
pinned exactly as `tokenizers-0.21.4` in both dependency declaration and
configuration identity. After immutable-revision acquisition, checksum-verified
UTF-8 tokenizer bytes are passed directly to `Tokenizer.from_str`; the verified
path is never reopened for loading.

Review-fix RED/GREEN evidence:

- schema v9 and prior-v8 refusal: `2 failed, 117 deselected`, then
  `2 passed, 117 deselected`;
- immutable reprocess initially failed on the old mutable insert, then passed;
  archived-generation deletion was independently RED (`1 failed`) before the
  active/archive resolver and GREEN together with retention (`2 passed`);
- canonical gap/overlap/reorder coverage mutation check: `3 failed`, then
  `3 passed` with the slice gate restored;
- bounded retry: `1 failed` with attempt three still retryable, then `1 passed`;
- 8,000/engine identity plus in-memory tokenizer loading: `2 failed`, then
  `2 passed`.

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

Review-fix focused evidence (no full suite):

- direct boundary, oversized block, byte-fallback spans, and interruption:
  `5 passed, 120 deselected in 27.54s`;
- failure, bounded retry, missing provenance, and aggregate recomputation:
  `4 passed, 121 deselected in 26.62s`;
- immutable reprocess/archive resolution and gap/overlap/reorder coverage:
  `5 passed, 120 deselected in 47.48s`;
- changed input/configuration plus schema/configuration identity:
  `5 passed, 120 deselected in 32.71s`;
- direct publication and fresh cross-catalog validation:
  `3 passed, 122 deselected in 7.46s`;
- complete injected hosted-adapter module: `25 passed in 0.70s`;
- changed-file Ruff: `All checks passed!`;
- worktree-source smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- worktree-source `run --help` retained all three explicit roots, positive
  `--limit`, `--reprocess`, and hosted authorization;
- `git diff --check`, current-doc stale-claim audit, and tracked generated
  artifact audit produced no findings; both operator-doc link sets resolved.

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
- Reprocess invalidates the research aggregate through the existing archive
  seam, but only clears mutable part checkpoints. Immutable part generations
  remain addressable by the archived aggregate's append-only provenance.
- Character and UTF-8 byte positions are derived cumulatively from the exact
  planned substrings. Validation trusts neither the checkpoint nor immutable
  row alone: for current canonical input it reopens the file and hashes both
  recorded slices.
- The three-attempt ceiling is committed configuration identity. The inner part
  failure becomes terminal first; the outer article-text failure reads that
  state in the same coordinator path and mirrors it, preventing replay.
- The 8,000-token ceiling reserves 192 tokens beneath the measured 8,192 model
  context while retaining `truncate=false`. Tokenizer engine/version and
  artifact identity both participate in `configuration_id`.

## Concerns

The first real authorized run may need to acquire the pinned tokenizer artifact
into the Hugging Face cache before local token accounting; acquisition is by
immutable revision and checksum-verified, but it is still a separate network
dependency from hosted embedding inference. Automated and smoke verification
do not exercise that acquisition or a clean dependency install; ticket 15 owns
the clean temporary-install check. The shared virtual environment intentionally
does not contain `tokenizers` or `huggingface_hub`, so all tokenizer tests remain
injected/offline and smoke/help use worktree source through `PYTHONPATH=src`.
