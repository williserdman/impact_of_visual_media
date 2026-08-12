# WSJ Markdown catalog

The preprocessing pipeline in this repository turns a local, licensed WSJ
HTML/image archive into two small, research-ready outputs:

- one deterministic UTF-8 Markdown file for each canonical article; and
- one DuckDB catalog that indexes those files, publication time, provenance,
  the optional local header image, and remote editorial-image URLs.

The preprocessing pipeline does not modify the source archive. It also does not
embed, summarize, classify, download remote images, fetch market data,
construct graphs, label events, or run trading research. Those jobs belong
downstream.

The repository also contains a downstream text/header-image embedding slice.
`wsj-embeddings smoke` creates one generated canonical article in a temporary
directory, encodes it with a deterministic fake adapter, publishes a
2,048-dimensional normalized `article_text` vector to a separate versioned
DuckDB catalog, validates it, and removes the fixture. It never reads the
configured archive or calls a network service. `wsj-embeddings pilot` is a
separate, explicit hosted Jina v4 measurement command: it sends only fixed
internally generated text and PNG bytes, never reads the archive or creates
catalog/output state. `wsj-embeddings inventory` reads only canonical catalog
metadata, while `wsj-embeddings run` exposes explicitly authorized bounded or
full production Jina scopes for canonical article text and optional
local header images. Remote editorial-image URLs remain references only.

For the data model and maintenance guide, read the
[architecture crash course](docs/architecture-crash-course.md).

## Install and configure

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

The default configuration is `config/wsj_pipeline.toml`:

```toml
[paths]
source = "data/wsj_archive"
output = "data/processed/wsj"

[pipeline]
batch_size = 250
workers = 4
```

Relative paths resolve from the project root. Source and output roots must not
overlap. Put a custom configuration before the subcommand:

```bash
.venv/bin/wsj-pipeline --config config/other.toml inventory
```

## Preprocessing CLI

The installed `wsj-pipeline` CLI has exactly four subcommands. On its normal
result path, each command prints one deterministic JSON summary:

```text
inventory                 read-only counts for all discovered HTML/header images
run (--limit N | --full)  incrementally update Markdown and the catalog
     [--reprocess]        explicitly process stale extractor-version rows
     [--workers N]        override bounded source-preparation threads
validate                  check the existing catalog and generated files
smoke                     run the complete workflow on generated temporary fixtures
```

`run` requires exactly one of a positive `--limit` or `--full`. `validate`,
`run`, and `smoke` return a nonzero status when their final validation fails.
`smoke` ignores the configured archive and output paths.
Argument parsing, configuration, lock, schema, or unexpected pipeline errors
can exit before a JSON summary is produced.

The separate `wsj-embeddings` CLI has these five subcommands:

```text
smoke      publish and validate one generated article text embedding
pilot      send fixed generated text/image probes to hosted Jina v4
inventory  count canonical/text/header eligibility without credentials
validate   check integrity and content-free coverage without credentials
run        embed or reprocess a positive lexical article-id limit after authorization
```

It prints this deterministic, content-free result on success:

```json
{"articles": 1, "embeddings": 1, "validation_ok": true}
```

`pilot` accepts no source, output, article, text, or file selector. It requires
`JINA_API_KEY` in its environment, calls the fixed hosted adapter with
`truncate: false`, and emits one deterministic content-free JSON record. The
record includes the recorded OpenAPI observation, per-probe returned model,
dimensions, raw-vector norms, usage, safe rate-limit headers, retry count, and
the observed outcomes for nominal 8,192/32,768-unit text and 5/8 MB PNG
boundaries. It may report a boundary as rejected; that is a measurement, not a
silent truncation. Missing or rejected credentials return only a classified
content-free error and create no catalog, output, or log artifact.

Run a successful credentialed pilot before authorizing a real-corpus embedding
run. Automated tests never call the hosted API: they inject simulated HTTP
responses or a fake adapter at the same public seams.

`wsj-embeddings` does not accept the preprocessing TOML configuration and has
no implicit root defaults. `inventory`, `validate`, and `run` require explicit source,
preprocessing-output, and embedding-output roots; resolving them must produce
three pairwise-disjoint paths.

## Safe smoke-to-full workflow

First prove the installation without touching the licensed archive:

```bash
.venv/bin/wsj-pipeline smoke
.venv/bin/wsj-embeddings smoke
```

Both commands use generated temporary fixtures. The embedding smoke command
also snapshots its generated preprocessing inputs and fails if the downstream
coordinator changes them.

After smoke, an operator who has a valid Jina credential may measure the hosted
contract without transmitting licensed content:

```bash
.venv/bin/wsj-embeddings pilot
```

Keep `JINA_API_KEY` out of shell history, configuration files, command output,
and committed artifacts. Review the pilot's JSON result before allowing any
corpus scope.

Inventory canonical embedding eligibility without a credential, adapter, or
embedding-output mutation:

```bash
.venv/bin/wsj-embeddings inventory \
  --source-root data/wsj_archive \
  --preprocessing-output-root data/processed/wsj \
  --embedding-output-root data/embeddings/wsj
```

An embedding run requires exactly one of a strictly positive lexical
`article_id` limit or explicit `--full`, plus the separate
`--authorize-hosted-processing` assertion. A configured `JINA_API_KEY` alone is
insufficient:

```bash
.venv/bin/wsj-embeddings run \
  --source-root data/wsj_archive \
  --preprocessing-output-root data/processed/wsj \
  --embedding-output-root data/embeddings/wsj \
  --limit 5 \
  --authorize-hosted-processing
```

Use `--full` in place of `--limit 5` only after reviewing inventory, pilot,
authorization, cost, and throughput. Full mode keyset-pages canonical articles
from one read-only snapshot in lexical `article_id` order, records a durable
content-free inventory, and processes it in bounded pages. Only successful
exhaustion and processing enter bounded reconciliation. Limited, failed,
interrupted, or cancelled runs never infer deletion.

Add `--reprocess` only when article text in the selected scope must be
regenerated despite matching content and configuration. Independent unchanged
header-image successes remain reusable. Reprocess never expands beyond the
selected scope.

The run sends each selected canonical Markdown artifact in full, with hosted
truncation disabled. A static JPEG, PNG, or WebP at or below 5,000,000 bytes
and 20,000,000 pixels is decoded safely and sent as base64 of its exact source
bytes. An oversized but safely decodable image is EXIF-oriented, converted to
RGB, aspect-scaled only when it exceeds the pixel ceiling, and encoded with the
versioned Pillow 11.3.0 JPEG-85/4:2:0/Lanczos policy. The metadata-free
rendition is staged, reopened, decoded, hashed, and atomically installed below
the embedding output root. The full output-root/rendition/transform/leaf chain
is reanchored after decode and reverified again immediately before run state can
begin. This conservative
transform is implementation policy pending confirmation by a credentialed
synthetic pilot; changing it creates a new configuration identity. Unsupported,
animated, corrupt, unsafe, or still-oversized input is terminal and publishes
no placeholder vector. At runtime the transform identity additionally includes
the linked JPEG, libjpeg-turbo, zlib, and WebP build versions; an unavailable or
mismatched build fails before configuration or work publication. The production
Pillow path still requires the clean-install synthetic verification owned by
Task 14/ticket 15. Markdown at or below the conservative 8,000-token input
ceiling keeps the single hosted-input path, reserving 192 tokens below the
confirmed 8,192-token model context. Longer Markdown is greedily partitioned at
complete blank-line-delimited block boundaries; an individually oversized
block is partitioned only at tokenizer offsets. The exact substrings concatenate
back to the original Markdown without overlap or loss. The tokenizer is the
official `jinaai/jina-embeddings-v4` `tokenizer.json` at immutable revision
`d1e5d70b7b34d927a8cddac458583c4fbe50a914`; runtime acquisition verifies SHA-256
`9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa` before
loading its verified bytes in memory with pinned `tokenizers-0.21.4`.
Automated tests inject an offline tokenizer and never download the
artifact. The run never uses remote inline-image URLs. Run/configuration setup and each selected item's
queue/recovery registration use separate bounded transactions. Each validated
source vector and its successful work checkpoint commit before the next
modality is attempted. Once both sources succeed under the same configuration,
their float32 normalized vectors produce a 2,048-dimensional, L2-normalized
equal-weight midpoint in a separate bounded transaction. Replaying the same
limit reuses unchanged successes without another
hosted request; when input changes, the mismatched vector is removed atomically
with checkpoint invalidation before replacement is attempted. Any existing
same-article/configuration multimodal dependent is archived, removed, and
requeued in that transaction; the other source modality, other articles, and
other configurations remain unchanged. A successful source checkpoint whose
active vector is missing likewise invalidates its composite before recovery is
attempted. A missing declared local header records
a retryable, content-free image disposition without failing or regenerating
successful text. Provider/transport retryable failures and interrupted image
work are attempted again. Deterministic image-request rejection is attempted at
most three times before becoming terminal; terminal work stays visible and is
not retried implicitly. Limited runs never infer removal outside the selected
scope. Success JSON includes content-free `reused`, `attempted`, `succeeded`,
`retryable`, `terminal`, `interrupted`, `header_absent`, and `header_failed`
counts plus the deterministic, content-free `configuration_id` required for
research queries.

Validate the completed corpus without a Jina credential or hosted request:

```bash
.venv/bin/wsj-embeddings validate \
  --source-root data/wsj_archive \
  --preprocessing-output-root data/processed/wsj \
  --embedding-output-root data/embeddings/wsj \
  --configuration-id '<configuration_id>'
```

`--configuration-id` may be omitted only while the catalog contains zero or
one configuration. The result reports stable integrity issues and content-free
counts for canonical articles; text success/failure; header present/absent,
success/failure; a mutually exclusive multimodal success/unavailable/failure
partition; stale work;
unresolved retryable work; and orphan rendition artifacts. A nonempty issue
list returns a nonzero status. Validation opens both catalogs read-only, does
not construct an embedding adapter, and never repairs, migrates, deletes, or
downloads anything.

Validation opens the embedding catalog first and then the preprocessing
catalog, begins a read transaction before each catalog's first schema query,
and holds both through the complete pass. These are independently stable read
snapshots acquired in that order. Because the catalogs have separate writers,
their pinned versions need not ever have been simultaneously current.
Validation reports cross-catalog relational inconsistencies, but it is not an
atomic cross-catalog or filesystem snapshot. Generated files cannot be locked
by those transactions, so validation records content-free start/end tokens for
every canonical Markdown/current header, every referenced rendition, and the
complete no-follow rendition namespace. A detected in-pass filesystem change
reports `concurrent_validation_state_change` and makes validation unsuccessful.

Then inventory the configured archive without changing either source or
generated state:

```bash
.venv/bin/wsj-pipeline inventory
```

`inventory` and `run` require the configured source root to be an accessible
real directory. If the root is absent, is not a directory, or traversal fails,
the command stops; a full run does not reconcile unseen sources after such a
failure.

Create or update a bounded sample from the lexically first 20 source paths:

```bash
.venv/bin/wsj-pipeline run --limit 20
.venv/bin/wsj-pipeline validate
```

Source reading, hashing, and extraction use four threads by default. Override
that count for one run when the machine warrants it:

```bash
.venv/bin/wsj-pipeline run --limit 20 --workers 8
```

At most one configured batch is in flight. DuckDB changes, duplicate
selection, Markdown publication, cleanup, and validation remain serialized.

The bounded run writes to the configured output root, but it never treats
unseen source paths as deleted. Because selection is lexical, it is a safety
sample rather than a way to target the newest year. Limited discovery stops
after those paths instead of first inventorying the whole archive.

A corpus-wide run is deliberately explicit and may be long-running:

```bash
.venv/bin/wsj-pipeline run --full
```

Use a durable terminal such as `tmux`. Coding agents must not launch this
command, or `--full --reprocess`, without the user's authorization in the
current task. A completed full run is the only operation that reconciles
previously known source files that have disappeared.

## Generated layout

All generated paths are below the configured output root and ignored by Git:

```text
data/processed/wsj/
├── catalog.duckdb
├── text/
│   └── YYYY/
│       └── MM/
│           └── DD/
│               └── <sha256-of-article-id>.md
├── staging/
└── pipeline.lock
```

Dates in the path are derived from the authoritative publication timestamp in
`America/New_York`. Paths stored in DuckDB are relative to the source or output
root, so the archive and its generated output can be moved together.

## Query preprocessing by publication date

Open `data/processed/wsj/catalog.duckdb` with DuckDB and query the single
research-facing `articles` table:

```sql
SELECT
    article_id,
    cleaned_markdown_path,
    header_image_path,
    inline_image_urls,
    published_at_utc,
    publication_date_new_york
FROM articles
WHERE publication_date_new_york
      BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
ORDER BY published_at_utc, article_id;
```

Resolve `cleaned_markdown_path` against the output root and
`header_image_path` against the source root. Remote `inline_image_urls` are
references only; the pipeline never downloads them.

## Text, header-image, and multimodal embedding catalog contract

The downstream catalog is a separate schema-version-14 `catalog.duckdb` below a
root that must be disjoint from both the licensed source root and preprocessing
output root. The generated smoke and injected-adapter coordinator exercise the
same catalog contract as the explicitly rooted production CLI.
Versions 1 through 13 are refused without migration; move reproducible derived
output aside or choose a fresh embedding output root.

The catalog has exactly thirteen base tables, two canonical views, and no
indexes. The operational
`full_run_articles` relation stores only run/article identity and optional
source-relative header association, never Markdown or image bytes:

| Table | Ordered columns | Key |
|---|---|---|
| `metadata` | `key VARCHAR`, `value VARCHAR` | `key` |
| `embedding_configurations` | `configuration_id VARCHAR`, `model VARCHAR`, `observed_model VARCHAR`, `observed_api_version VARCHAR`, `task VARCHAR`, `dimensions INTEGER`, `output_type VARCHAR`, `normalization VARCHAR`, `tokenizer_revision VARCHAR`, `tokenizer_engine VARCHAR`, `context_token_limit INTEGER`, `context_rules VARCHAR`, `long_text_aggregation VARCHAR`, `long_text_part_attempt_limit INTEGER`, `batch_max_items INTEGER`, `batch_max_estimated_tokens INTEGER`, `batch_max_encoded_bytes BIGINT`, `batch_max_concurrency INTEGER`, `batch_max_attempts INTEGER`, `batch_initial_backoff_seconds DOUBLE`, `batch_max_backoff_seconds DOUBLE`, `image_input_rules VARCHAR`, `image_transform VARCHAR`, `multimodal_formula VARCHAR`, `client_configuration_version VARCHAR` | `configuration_id` |
| `runs` | `run_id VARCHAR`, `configuration_id VARCHAR`, `scope VARCHAR`, `status VARCHAR`, `discovery_complete BOOLEAN`, `reconciliation_complete BOOLEAN`, `articles INTEGER`, `embeddings INTEGER`, `reused INTEGER`, `attempted INTEGER`, `succeeded INTEGER`, `retryable INTEGER`, `terminal INTEGER`, `interrupted INTEGER`, `header_absent INTEGER`, `header_failed INTEGER`, `hosted_requests INTEGER`, `hosted_retries INTEGER`, `usage_json VARCHAR`, `throttles INTEGER`, `elapsed_seconds DOUBLE`, `started_at TIMESTAMPTZ`, `finished_at TIMESTAMPTZ` | `run_id` |
| `full_run_articles` | `run_id VARCHAR`, `article_id VARCHAR`, `header_image_path VARCHAR` | `(run_id, article_id)` |
| `embedding_work_storage` | physical columns exposed by `embedding_work_items` | `(article_id, modality, configuration_id)` |
| `embedding_storage` | physical columns exposed by `embeddings` | `(article_id, modality, configuration_id)` |
| `reconciliation_actions` | exact run/article/modality/configuration target disposition plus expected source/input/state/generation/vector identity and `staged_at` | `(run_id, article_id, modality, configuration_id)` |
| `embedding_work_items` (view) | canonical current work interface over storage and visible actions | — |
| `embeddings` (view) | canonical current vector interface over storage and visible actions | — |
| `embedding_generation_history` | `article_id VARCHAR`, `modality VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `source_relative_path VARCHAR`, `input_sha256 VARCHAR`, `stored_vector_sha256 VARCHAR`, `superseded_run_id VARCHAR`, `superseded_reason VARCHAR`, `superseded_at TIMESTAMPTZ` | `(article_id, modality, configuration_id, generation_run_id)` |
| `multimodal_embedding_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `text_generation_run_id VARCHAR`, `text_stored_vector_sha256 VARCHAR`, `header_image_generation_run_id VARCHAR`, `header_image_stored_vector_sha256 VARCHAR`, `formula_version VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id)` |
| `image_input_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `source_sha256 VARCHAR`, `source_format VARCHAR`, `source_bytes BIGINT`, `source_width INTEGER`, `source_height INTEGER`, `source_frames INTEGER`, `embedded_input_sha256 VARCHAR`, `embedded_format VARCHAR`, `embedded_bytes BIGINT`, `embedded_width INTEGER`, `embedded_height INTEGER`, `embedded_frames INTEGER`, `transform_id VARCHAR`, `rendition_relative_path VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id)` |
| `long_text_parts` | `article_id VARCHAR`, `configuration_id VARCHAR`, `article_input_sha256 VARCHAR`, `part_index INTEGER`, `part_count INTEGER`, `char_start BIGINT`, `char_end BIGINT`, `byte_start BIGINT`, `byte_end BIGINT`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `state VARCHAR`, `attempt_count INTEGER`, `error_code VARCHAR`, `status_code INTEGER`, `retry_after_seconds DOUBLE`, `last_run_id VARCHAR`, `generation_run_id VARCHAR`, `stored_vector_sha256 VARCHAR`, `vector FLOAT[2048]`, `updated_at TIMESTAMPTZ` | `(article_id, configuration_id, article_input_sha256, part_index)` |
| `long_text_part_generations` | `article_id VARCHAR`, `configuration_id VARCHAR`, `article_input_sha256 VARCHAR`, `part_index INTEGER`, `generation_run_id VARCHAR`, `part_count INTEGER`, `char_start BIGINT`, `char_end BIGINT`, `byte_start BIGINT`, `byte_end BIGINT`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `stored_vector_sha256 VARCHAR`, `vector FLOAT[2048]`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, article_input_sha256, part_index, generation_run_id)` |
| `article_text_aggregation_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `part_index INTEGER`, `article_input_sha256 VARCHAR`, `part_count INTEGER`, `part_generation_run_id VARCHAR`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `part_stored_vector_sha256 VARCHAR`, `aggregation_version VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id, part_index)` |

The source path is nullable for article text and absent images. The input hash
is nullable for content-free `stale_input` work, `not_applicable` image work,
and only one applicable case: a
safe-path `retryable` header with `missing_header_image`, zero adapter attempts,
and no active vector or generation. The three failure-detail columns retain
classified codes and numeric response/retry metadata, never exception text or
editorial content. Work states are `queued`, `in_progress`, `succeeded`,
`retryable`, `terminal`, `interrupted`, `not_applicable`, and `stale_input`.
`stale_input` is a generation-free completed-full-run disposition; its header
path equals the retained canonical association, or is null when the article
disappeared. `not_applicable` means no canonical header; header failures remain
distinct and count in `header_failed`. Changed exact image bytes archive the
prior image generation as `input_changed` and invalidate only that image plus
its same-configuration multimodal dependent. A changed configuration uses a
separate work key and
leaves the prior configuration queryable.

Production requests may mix article text, header images, and long-text parts.
Each request is bounded independently by item count, estimated text tokens,
total encoded input bytes, and configured concurrency. Response indexes are
validated independently and every item commits in its own transaction, so a
valid sibling survives a missing or malformed response item. Retryable
timeouts, connection errors, rate limits, and eligible server failures use
bounded exponential backoff with real bounded jitter (injectable in tests).
Numeric or RFC 9110 HTTP-date `Retry-After` and zero-remaining rate-reset
headers are honored up to the configured maximum wait; throttling halves later
concurrency without exceeding its configured maximum. Authentication,
authorization, and invalid request configuration stop the run. Summaries and
`runs` store only nonnegative counts, allowlisted numeric token usage,
retry/throttle observations, and whole-run elapsed seconds—never request
content. Each durable prior attempt reduces the remaining request budget, so
replay cannot purchase beyond the configured ceiling.
All completed exchanges in a concurrent wave are processed in stable submission
order before a request-wide fatal error is raised, preserving successful sibling
work. A long part that succeeds on an in-run retry no longer leaves its parent
marked failed and therefore participates in immediate complete-part aggregation.
Every successful image generation has content-free `image_input_provenance`:
source and exact embedded-input hashes, formats, byte counts, dimensions, frame counts,
transform identity, and an optional output-relative rendition path. It stores
no image bytes. Pass-through uses `exact-source-bytes-v1`; derived rows name
the pinned transform and exact JPEG rendition supplied to the adapter.
Read-only validation decodes current source and embedded bytes with the matching
profile codec, proves their recorded hashes/formats/byte counts/dimensions/frame
counts, enforces the safe source-decode ceiling for current and archived
generations, and proves the pass-through-versus-JPEG decision. Malformed
dimensions become stable validation issues rather than aborting validation.
Validation rejects unknown provenance configuration
references, and descriptor-scans the rendition namespace without following
symlinks. Unreferenced final and interrupted temporary renditions are reported
as harmless orphans rather than deleted.
The configured archive root is opened as an accessible, no-follow directory
before the preprocessing catalog, any item, or embedding output is touched. An
absent, non-directory, or inaccessible archive root is an operational
`source_root` failure; only an absent optional leaf below a proven root becomes
`missing_header_image`. If a formerly applicable canonical header becomes
absent, one transaction archives/removes both its image vector and its
same-configuration multimodal dependent before recording `not_applicable`.
`configuration_id` is the SHA-256 of compact,
key-sorted JSON covering model alias, observed hosted model/API metadata, task,
dimensions, output type, normalization, tokenizer artifact and engine,
context rules/ceiling, long-text aggregation and attempt limit, batching
payload/concurrency/retry policy, image input/transform rules, multimodal
formula, and client configuration version.
The tokenizer engine is exactly `tokenizers-0.21.4`; checksum-verified artifact
bytes are decoded and loaded directly from memory, so a path replacement after
verification cannot change the tokenizer. The current long-text rule is
`l2-token-count-weighted-mean-float32-v1`. The input ceiling is 8,000 tokens,
leaving 192 tokens of framing headroom below the confirmed 8,192-token model
context. Inputs through that ceiling receive one provider-pooled vector.
Longer inputs commit each normalized operational part vector independently,
then publish one canonical `article_text` vector as the float32 L2-normalized
token-count-weighted mean. Retryable part requests become terminal after three
committed attempts. The part, parent article work, and terminal run count change
in one transaction. If attempt three was checkpointed before an interruption,
replay terminalizes the work without another hosted call. Mutable checkpoints
remain terminal on ordinary replay; bounded explicit reprocess atomically resets
every part before making new hosted calls. Mutable checkpoints point to
append-only part-generation
rows; explicit reprocess clears only the checkpoint, never a generation named
by active or archived aggregate provenance. Part vectors and their append-only
aggregate linkage are audit/resume state, never additional research
observations in `embeddings`. `input_sha256` hashes the exact canonical Markdown or
local header-image bytes. For `multimodal_article`, it hashes the content-free
source-generation identity and formula. `stored_vector_sha256` hashes the
normalized vector encoded as little-endian float32 values. Current modalities
are `article_text`, `header_image`, and `multimodal_article`; vectors must be
finite, nonzero, L2-normalized, and exactly 2,048-dimensional. A composite is
published only while both successful sources belong to its exact configuration.
Its append-only provenance records both source generation IDs/vector hashes and
`l2-normalize-0.5-text-0.5-image-v1`; it stores no editorial content. Superseded
research-vector history stores only identities, hashes, run IDs, reason, and
time. Append-only operational part generations retain their vectors for audit
and aggregate reconstruction, but never Markdown, image bytes, or excerpts.
`last_run_id` records the latest observation or attempt, while nullable
`generation_run_id` is set only for a successful active vector and remains
unchanged across reuse. Supersession history therefore names the run that
actually generated the archived vector rather than a later replay.

Full reconciliation first stages exact identity-bound actions in bounded action
pages using strict monotonic `(article_id, modality, configuration_id)` cursors.
Failed staging is invisible. One run-marker commit exposes the complete action
set atomically through the canonical views, then bounded idempotent compaction
materializes the same state with a monotonic cursor that also includes
`run_id`. Exact schema validation fingerprints the normalized behavior-bearing
SQL of both canonical views. Header association changes preserve a
configuration already matching the new path and stale only mismatched header
and dependent multimodal work. Derived renditions are removed only after their
catalog references commit, with one exact history-reference query per scanned
candidate rather than a corpus-sized set. If cleanup is interrupted, offline
validation reports the unreferenced file as `orphan_image_rendition`; a later
completed full run safely retries cleanup without touching source or
preprocessing output.

A retained catalog can be queried without joining back to article bodies:

```sql
SELECT
    e.article_id,
    e.source_relative_path,
    e.published_at_utc,
    e.publication_date_new_york,
    c.model,
    c.task,
    e.vector
FROM embeddings AS e
JOIN embedding_configurations AS c USING (configuration_id)
WHERE e.modality IN ('article_text', 'header_image', 'multimodal_article')
  AND e.configuration_id = '<configuration_id>'
  AND e.publication_date_new_york
      BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
ORDER BY e.published_at_utc, e.article_id;
```

Embedding validation opens both catalogs read-only, checks exact schemas before
data, links each vector to a canonical `articles` row, rehashes canonical
Markdown and local header images through no-follow descriptors, checks
publication metadata and both hashes, and verifies that successful run counts
exactly match the active or superseded generations that name that run as their
immutable generator. It requires provenance for every active and superseded
composite, re-derives its content-free input identity, resolves both source
links, and recomputes each active equal-weight normalized midpoint. An
impossible midpoint is reported rather than skipped.
It also validates part identities, limits, bounded attempts,
lifecycle/vector checkpoints, and aggregate linkage. Every active and archived
long-text aggregate must resolve all of its immutable part generations and
recompute to the recorded aggregate hash; active vectors must also equal that
recomputation. For aggregates matching current canonical input, validation
reopens Markdown and proves the recorded character and UTF-8 byte spans are
ordered, gapless, non-overlapping, cover the whole file, and hash to each part.
Missing provenance or self-consistent part tampering is therefore reported
without loading a tokenizer or credential.
It emits stable issues without article text or vector values, including
distinct missing, queued, in-progress, interrupted, retryable, and terminal
header-image dispositions; a true no-header `not_applicable` row is valid and
silent. It also requires a header disposition for every article/configuration
selected by article-text work and reconciles the latest run's `header_absent`
and `header_failed` claims against durable header work. When more than one
configuration exists, callers must select an explicit `configuration_id`;
validation never silently mixes generations.
Every vector exposed by the canonical `embeddings` view must also join the
same public article/modality/configuration work key in `succeeded` state with
matching source path, input hash, and a same-configuration generating run.

## Incremental runs and reprocessing

Normal reruns skip matching source/image fingerprints. If file stats change,
the pipeline hashes HTML before deciding whether extraction is necessary.
New content, changed HTML, and changed selected header images are processed in
bounded transactions; unchanged content is not republished. Source preparation
may complete in worker order, while canonical ranking and the completed
research output remain deterministic.

Discovery and later source reads reject symlinked HTML or local header-image
leaves and reopen accepted paths through no-follow source-relative descriptors.
Malformed optional update timestamps become a stable warning and rank as
missing update metadata; publication timestamps remain strict.

An extractor-version change marks existing sources stale but does not silently
start corpus-wide work. Reprocess an explicit scope:

```bash
.venv/bin/wsj-pipeline run --limit 20 --reprocess
.venv/bin/wsj-pipeline run --full --reprocess
```

Review the JSON counters after every run. Extraction failures are isolated and
recorded without article text; a valid duplicate may be promoted in place of a
failed winner.

`validate` opens the catalog read-only and checks the exact supported tables,
columns, primary/unique constraints, and publication-date index before it
checks relational data or generated files. Malformed and unsupported catalogs
produce stable, content-free issues and are never repaired in place.

## Recovery

Rerun the same idempotent command after an interruption. Markdown is staged,
hashed, and atomically replaced before its catalog transaction commits, so a
retry repairs a file/catalog mismatch. A date or identity change publishes the
new path before switching the catalog and removes the old generated file only
after commit; an interrupted cleanup can leave a harmless orphan that
`validate` reports.

Mutating runs own `data/processed/wsj/pipeline.lock`. If a process dies before
cleanup, first verify that no pipeline process is active, remove only that
stale lock, then rerun. Never delete the source archive to recover. If legacy
or unsupported derived output is detected, move that generated directory
aside or choose a fresh output root; the pipeline will not migrate or delete
it automatically.

The catalog leaf must be a single-link regular file; symlinks, directories, and
hard links are refused before DuckDB mutation. A validated run remains
nonterminal until validation completes and its final content-free summary is
stored in the terminal transition.

The same leaf rule applies to the separate embedding catalog. An existing
embedding catalog is inspected through a read-only DuckDB connection before a
writable connection is opened, and a replacement between those phases is
refused. Canonical Markdown is reopened below the preprocessing output root
through no-follow directory descriptors; symlinked, non-regular, multiply
linked, or replaced leaves are rejected. Local header images use the same
containment and no-follow rules below the immutable source root. Eligible
images are sent to Jina as base64 of those exact bytes; oversized safe images
use the staged and verified deterministic rendition below the embedding output
root. Embedding mutation is confined below its disjoint output root. The
coordinator opens every output-root component without following links, then
creates its catalog and exclusive `pipeline.lock` relative to that anchored
directory. Lock cleanup compares file identity and will not remove a pathname
replaced by another process.

## Development checks

These checks use generated fixtures and do not read the licensed archive:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
.venv/bin/wsj-embeddings smoke
```
