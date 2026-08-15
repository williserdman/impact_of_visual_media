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
run        embed or reprocess exactly one positive limit or explicit full scope
           after hosted-processing authorization and a bound pilot observation
```

It prints this deterministic, content-free result on success:

```json
{"articles": 1, "embeddings": 1, "validation_ok": true}
```

`pilot` accepts no source, output, article, text, or file selector. It requires
`JINA_API_KEY` in its environment, calls the fixed hosted adapter with
`truncate: false`, and emits one deterministic content-free JSON record. The
record separates the client's requested model and implemented-against OpenAPI
contract version from live observations. Bounded requests to the fixed Jina
OpenAPI and model-catalogue endpoints report only allowlisted fields actually
returned; unavailable or malformed metadata is `not_observed`, and an absent
billing currency or embedding-response billing object is `not_returned` rather
than zero or an invented value. Per-probe results include returned model labels,
dimensions, raw-vector norms, usage, safe rate-limit headers, actual scheduler
requests/retries/throttles, and the observed outcomes for locally tokenized
8,192/32,768-target text and exact 5/8 MB PNG boundaries. A separate two-request
generated probe reports attempted scheduler fan-out and measured overlap; it is
not a throughput promise. Natural service failures may exercise bounded retry,
but the pilot never induces a billable failure to claim retry behavior.

`ready_for_operator_review` means only that normal text, image, and mixed probes
succeeded with the configured dimension, task, and `truncate: false` contract.
It does not authorize corpus transmission or enable `run`. A context or image
setting is confirmed only by its actual boundary probe outcome, not because a
catalogue or OpenAPI document names it. Missing or rejected credentials return
only a classified content-free error and create no catalog, output, or log
artifact.

Every hosted pilot attempt also passes through one shared, process-local
rolling quota window: at most 90 requests and 90,000 estimated input tokens in
60 seconds, with at most two concurrent requests and 45,000 estimated tokens
per request. It authenticates first with the generated 224x224 PNG probe, uses the
pinned tokenizer before every generated text probe,
waits once until enough request and token debt has expired for the whole next
wave. The pilot remains state-free; this pacing disappears when that pilot
process exits.

Run a successful credentialed pilot before authorizing a real-corpus embedding
run. Automated tests never call the hosted API: they inject simulated HTTP
responses or a fake adapter at the same public seams.

Production `run` and the generated-content `pilot` use the real hosted Jina API.
Before either command is used, the operator must be authorized to transmit the
selected licensed material and must confirm that the provider's current terms,
privacy controls, and data-handling policy are acceptable. The configured
`jina-embeddings-v4` name is a hosted mutable alias: each hosted generation
records the returned model, while the pilot separately reports current bounded
API observations. Exact hosted reproducibility is not guaranteed if the
provider changes behavior behind that alias. A pilot reports returned usage,
billing fields, rate metadata, and one small concurrency observation; absent
fields remain unknown, and the result is not a full-corpus cost or throughput
estimate.

`wsj-embeddings` does not accept the preprocessing TOML configuration and has
no implicit root defaults. `inventory`, `validate`, and `run` require explicit source,
preprocessing-output, and embedding-output roots; resolving them must produce
three pairwise-disjoint paths.

## Safe preprocessing workflow

First prove the preprocessing installation without touching the licensed
archive:

```bash
.venv/bin/wsj-pipeline smoke
```

The command uses generated temporary fixtures and ignores the configured
archive and output paths. The separate first-time embedding workflow appears
after the preprocessing layout and query guidance below.

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

### Preprocessing reruns and reprocessing

Normal reruns reuse matching source and header-image fingerprints; when file
stats change, the pipeline hashes HTML before deciding whether extraction is
needed. An extractor-version change marks existing sources stale but does not
start corpus-wide work. Reprocess an explicit scope:

```bash
.venv/bin/wsj-pipeline run --limit 20 --reprocess
.venv/bin/wsj-pipeline run --full --reprocess
```

`--reprocess` is not schema migration, and `--full --reprocess` needs the same
current-task authorization as any other real full run. Review JSON counters
after every run. Extraction failures are isolated without article text, and a
valid duplicate may be promoted in place of a failed winner. `validate` is
read-only: it checks the supported schema, constraints, and publication-date
index before relational data and generated files; unsupported catalogs are
reported, never repaired in place.

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

`article_text` is a hosted Jina v4 embedding of canonical Markdown.
`header_image` is a hosted Jina v4 embedding of the optional local canonical
header image. `multimodal_article` is the locally computed normalized midpoint
of matching text and image generations; no additional hosted request produces
it. All vectors are 2,048-dimensional normalized float32 research observations.

### First-time embedding workflow

Before processing any licensed material, complete these steps in order:

1. Confirm that you have the right to submit the licensed input to hosted
   processing.
2. Install the declared project dependencies.
3. Set `JINA_API_KEY` in the environment without recording its value in shell
   history, configuration, output, or committed files.
4. Run `wsj-embeddings smoke`; it uses generated fixtures and needs neither a
   key nor network access.
5. Manually run the generated-only `wsj-embeddings pilot`; unlike smoke, it
   makes a hosted request but sends only generated probes, never archive data.
6. Review the returned model and record its exact safe label for
   `--observed-model`.
7. Keep the source, preprocessing-output, and embedding-output roots pairwise
   disjoint.

Start with the generated smoke check:

```bash
wsj-embeddings smoke
```

Smoke also snapshots its generated preprocessing inputs and fails if the
downstream coordinator changes them.

Set the key only in the shell that will run the hosted command. This prompt
does not echo the value or place it in shell history:

```bash
read -rsp "Jina API key: " JINA_API_KEY; echo
export JINA_API_KEY
```

Do not paste the value into a command argument, repository file, issue, log,
or generated output. After exporting it, run the pilot manually and review its
content-free result before authorizing any archive scope:

```bash
wsj-embeddings pilot
```

Inventory canonical eligibility without a credential, hosted adapter, or
embedding-output mutation:

```bash
wsj-embeddings inventory \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output
```

Make the first real mutation deliberately bounded to one article. This command
sends canonical Markdown and any optional local canonical header image for the
selected article to Jina; replace the placeholder with the model returned by
the pilot:

```bash
wsj-embeddings run \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output \
  --limit 1 \
  --authorize-hosted-processing \
  --observed-model '<pilot-returned-model>'
```

A configured key alone is insufficient. Without the reviewed pilot model,
production stops with `pilot_observation_required` before output mutation; a
later response-model mismatch records content-free `configuration_drift` and
publishes no vector under the old configuration.

Production requests use a durable rolling quota recorded in the embedding
catalog: no more than 90 hosted attempts or 90,000 estimated input tokens in
any 60-second window. A request is limited to 45,000 estimated tokens and
concurrency remains two. Text and long-text parts use the pinned tokenizer;
header images use the documented patch-token estimate. Every retry consumes a
new reservation. Safe provider input usage may increase later quota debt but
can never reduce the conservative estimate; malformed, non-integral, or
implausibly large usage metadata is ignored for quota accounting. Expired
reservations are deleted in bounded pages during later admissions, so this
operational table does not become permanent request history.

Quota waits occur outside DuckDB transactions. A reservation commits
immediately before transport and is not refunded after failure or process
loss, so a crash before transmission can conservatively delay replay for at
most 60 seconds. Existing 429, `Retry-After`, rate-reset, jitter, and bounded
retry behavior remains active because the 10% margin is not a guarantee.
Reservations coordinate only runs sharing this embedding catalog/output; they
cannot observe another program or machine using the same Jina account.
An individual input that cannot fit the immutable 45,000-token or encoded-byte
request ceiling becomes a content-free terminal item without preventing a
valid sibling from completing.

Then validate the generated catalog offline. Validation is read-only and makes
no hosted request. When more than one configuration exists, provide its
explicit ID rather than allowing configurations to be mixed:

```bash
wsj-embeddings validate \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output \
  --configuration-id '<configuration-id>'
```

A nonempty issue list returns a nonzero status. The result contains only
integrity issues and content-free coverage counts; it never repairs, migrates,
deletes, or downloads anything. The preprocessing and embedding catalogs are
separately stable read snapshots rather than one atomic cross-catalog or
filesystem snapshot. See the crash course's
[validation contract](docs/architecture-crash-course.md#10-validation) for the
detailed consistency model.

### Maintainer reference

For the detailed contracts behind this reference, see the crash course's
[embedding schema](docs/architecture-crash-course.md#separate-text-header-image-and-multimodal-embedding-catalog),
[atomicity and recovery](docs/architecture-crash-course.md#9-atomicity-and-recovery),
[validation](docs/architecture-crash-course.md#10-validation),
[downstream boundary](docs/architecture-crash-course.md#13-downstream-boundaries),
and [glossary](docs/architecture-crash-course.md#15-glossary).

### Schema-v17 catalog reference

The downstream catalog is a separate schema-version-17 `catalog.duckdb` below
an embedding output root that must be disjoint from both the licensed source
root and preprocessing output root. Versions 1 through 16 are refused without
migration: move reproducible derived output aside or choose a fresh embedding
output root. The catalog has exactly fourteen base tables, two canonical views,
and no indexes. `full_run_articles` stores only run/article identity and an
optional source-relative header association, never Markdown or image bytes.

The operational table and view inventory is:

| Table | Ordered columns | Key |
|---|---|---|
| `metadata` | `key VARCHAR`, `value VARCHAR` | `key` |
| `embedding_configurations` | `configuration_id VARCHAR`, `model VARCHAR`, `observed_model VARCHAR`, `client_api_contract_version VARCHAR`, `task VARCHAR`, `dimensions INTEGER`, `output_type VARCHAR`, `normalization VARCHAR`, `tokenizer_revision VARCHAR`, `tokenizer_engine VARCHAR`, `context_token_limit INTEGER`, `context_rules VARCHAR`, `long_text_aggregation VARCHAR`, `long_text_part_attempt_limit INTEGER`, `batch_max_items INTEGER`, `batch_max_estimated_tokens INTEGER`, `batch_max_encoded_bytes BIGINT`, `batch_max_response_bytes BIGINT`, `batch_max_concurrency INTEGER`, `batch_max_attempts INTEGER`, `batch_initial_backoff_seconds DOUBLE`, `batch_max_backoff_seconds DOUBLE`, `quota_max_requests INTEGER`, `quota_max_estimated_tokens INTEGER`, `quota_window_seconds INTEGER`, `image_input_rules VARCHAR`, `image_transform VARCHAR`, `multimodal_formula VARCHAR`, `client_configuration_version VARCHAR` | `configuration_id` |
| `runs` | `run_id VARCHAR`, `configuration_id VARCHAR`, `scope VARCHAR`, `status VARCHAR`, `discovery_complete BOOLEAN`, `reconciliation_complete BOOLEAN`, `articles INTEGER`, `embeddings INTEGER`, `reused INTEGER`, `attempted INTEGER`, `succeeded INTEGER`, `retryable INTEGER`, `terminal INTEGER`, `interrupted INTEGER`, `header_absent INTEGER`, `header_failed INTEGER`, `hosted_requests INTEGER`, `hosted_retries INTEGER`, `usage_json VARCHAR`, `throttles INTEGER`, `elapsed_seconds DOUBLE`, `started_at TIMESTAMPTZ`, `finished_at TIMESTAMPTZ` | `run_id` |
| `full_run_articles` | `run_id VARCHAR`, `article_id VARCHAR`, `header_image_path VARCHAR` | `(run_id, article_id)` |
| `hosted_request_reservations` | `reservation_id VARCHAR`, `run_id VARCHAR`, `configuration_id VARCHAR`, `reserved_at TIMESTAMPTZ`, `estimated_tokens INTEGER`, `observed_input_tokens INTEGER` | `reservation_id` |
| `embedding_work_storage` | physical columns exposed by `embedding_work_items` | `(article_id, modality, configuration_id)` |
| `embedding_storage` | physical columns exposed by `embeddings` | `(article_id, modality, configuration_id)` |
| `reconciliation_actions` | exact run/article/modality/configuration target disposition plus expected source/input/state/generation/vector identity and `staged_at` | `(run_id, article_id, modality, configuration_id)` |
| `embedding_work_items` (view) | canonical current work interface over storage and visible actions | — |
| `embeddings` (view) | canonical current vector interface over storage and visible actions | — |
| `embedding_generation_history` | `article_id VARCHAR`, `modality VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `source_relative_path VARCHAR`, `input_sha256 VARCHAR`, `derivation_kind VARCHAR`, `raw_response_sha256 VARCHAR`, `response_model VARCHAR`, `stored_vector_sha256 VARCHAR`, `superseded_run_id VARCHAR`, `superseded_reason VARCHAR`, `superseded_at TIMESTAMPTZ` | `(article_id, modality, configuration_id, generation_run_id)` |
| `multimodal_embedding_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `text_generation_run_id VARCHAR`, `text_stored_vector_sha256 VARCHAR`, `header_image_generation_run_id VARCHAR`, `header_image_stored_vector_sha256 VARCHAR`, `formula_version VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id)` |
| `image_input_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `source_sha256 VARCHAR`, `source_format VARCHAR`, `source_bytes BIGINT`, `source_width INTEGER`, `source_height INTEGER`, `source_frames INTEGER`, `embedded_input_sha256 VARCHAR`, `embedded_format VARCHAR`, `embedded_bytes BIGINT`, `embedded_width INTEGER`, `embedded_height INTEGER`, `embedded_frames INTEGER`, `transform_id VARCHAR`, `rendition_relative_path VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id)` |
| `long_text_parts` | `article_id VARCHAR`, `configuration_id VARCHAR`, `article_input_sha256 VARCHAR`, `part_index INTEGER`, `part_count INTEGER`, `char_start BIGINT`, `char_end BIGINT`, `byte_start BIGINT`, `byte_end BIGINT`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `state VARCHAR`, `attempt_count INTEGER`, `error_code VARCHAR`, `status_code INTEGER`, `retry_after_seconds DOUBLE`, `last_run_id VARCHAR`, `generation_run_id VARCHAR`, `derivation_kind VARCHAR`, `raw_response_sha256 VARCHAR`, `response_model VARCHAR`, `stored_vector_sha256 VARCHAR`, `vector FLOAT[2048]`, `updated_at TIMESTAMPTZ` | `(article_id, configuration_id, article_input_sha256, part_index)` |
| `long_text_part_generations` | `article_id VARCHAR`, `configuration_id VARCHAR`, `article_input_sha256 VARCHAR`, `part_index INTEGER`, `generation_run_id VARCHAR`, `part_count INTEGER`, `char_start BIGINT`, `char_end BIGINT`, `byte_start BIGINT`, `byte_end BIGINT`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `derivation_kind VARCHAR`, `raw_response_sha256 VARCHAR`, `response_model VARCHAR`, `stored_vector_sha256 VARCHAR`, `vector FLOAT[2048]`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, article_input_sha256, part_index, generation_run_id)` |
| `article_text_aggregation_provenance` | `article_id VARCHAR`, `configuration_id VARCHAR`, `generation_run_id VARCHAR`, `part_index INTEGER`, `article_input_sha256 VARCHAR`, `part_count INTEGER`, `part_generation_run_id VARCHAR`, `part_input_sha256 VARCHAR`, `token_count INTEGER`, `part_stored_vector_sha256 VARCHAR`, `aggregation_version VARCHAR`, `created_at TIMESTAMPTZ` | `(article_id, configuration_id, generation_run_id, part_index)` |

`configuration_id` is the SHA-256 of compact, key-sorted configuration JSON,
including the requested model alias and exact observed hosted model. A changed
observed model creates a separate valid, queryable configuration; valid
configurations coexist, so select one explicitly for validation and research.

Work states are `eligible`, `queued`, `in_progress`, `succeeded`, `retryable`,
`terminal`, `interrupted`, `not_applicable`, `stale_input`, and
`stale_configuration`. Fresh or invalidated work begins durably `eligible`, is
admitted to `queued`, and becomes `in_progress` before an adapter attempt.
`not_applicable` means an article has no canonical header. `stale_input` and
`stale_configuration` are explicit reconciliation dispositions; coexistence of
valid configurations never creates `stale_configuration`.

The supported modalities are `article_text`, `header_image`, and
`multimodal_article`. All published vectors are finite, normalized float32
vectors with 2,048 dimensions. Canonical Markdown at or below the 8,000-token
direct ceiling is submitted once; longer text is processed with durable,
three-attempt part checkpoints and published as one `article_text` vector.
Header-image work also has a durable three-attempt policy for deterministic
request rejection. Operators need only inspect the content-free run counters
and retryable or terminal states. Unsupported, animated, corrupt, unsafe, or
still-oversized local images publish no placeholder vector, and remote inline
image URLs are never inputs. The batching, long-text, and image-rendition
implementation contracts are in the [crash course](docs/architecture-crash-course.md#separate-text-header-image-and-multimodal-embedding-catalog).

### Repeat runs, reprocessing, and recovery

Unchanged matching work is reused. A limited or interrupted run never infers
deletion; only a successfully completed explicit `--full` run may reconcile a
disappearance. `--reprocess` is deliberate, bounded regeneration of the
selected article-text scope, not schema migration; unchanged header-image
successes remain reusable, and reprocessing never expands the selected scope.
A real `--full` or `--full --reprocess` requires explicit authorization in the
current task. Never run a real full-corpus operation unattended: use a durable
terminal, monitor the content-free summaries and provider limits, and keep the
embedding output on recoverable storage so interruption can be replayed.

If a catalog is incompatible, move the generated embedding output aside or use
a fresh embedding output root; do not migrate, overwrite, or delete it in
place. After a process dies, verify that no embedding pipeline process is
active, remove only the exact stale `<embedding-output-root>/pipeline.lock`,
then rerun the same idempotent command. Validation reports harmless orphan
renditions; a later successfully completed full run safely retries their
cleanup. Never delete source data, a broad directory tree, a catalog, or an
active lock for recovery. See the crash course's
[recovery boundary](docs/architecture-crash-course.md#9-atomicity-and-recovery)
and [validation contract](docs/architecture-crash-course.md#10-validation).

### Maintainer queries

Select an explicit configuration before reading vectors:

```sql
SELECT
    e.article_id,
    e.modality,
    e.published_at_utc,
    e.stored_vector_sha256,
    e.vector
FROM embeddings AS e
WHERE e.configuration_id = '<configuration_id>'
  AND e.modality = 'article_text'
ORDER BY e.published_at_utc, e.article_id;
```

List the valid configuration identities and their requested and observed models:

```sql
SELECT configuration_id, model, observed_model, task, dimensions
FROM embedding_configurations
ORDER BY configuration_id;
```

Inspect recent content-free run summaries:

```sql
SELECT run_id, configuration_id, scope, status, reused, attempted, succeeded,
       retryable, terminal, interrupted, header_absent, header_failed,
       started_at, finished_at
FROM runs
ORDER BY started_at DESC;
```

Inspect durable coverage states for one configuration:

```sql
SELECT modality, state, count(*) AS items
FROM embedding_work_items
WHERE configuration_id = '<configuration_id>'
GROUP BY modality, state
ORDER BY modality, state;
```

## Recovery

Rerun the same idempotent command after an interruption. Markdown is staged,
hashed, and atomically replaced before its catalog transaction commits, so a
retry repairs a file/catalog mismatch. A date or identity change publishes the
new path before switching the catalog and removes the old generated file only
after commit; an interrupted cleanup can leave a harmless orphan that
`validate` reports.

Mutating preprocessing runs own `data/processed/wsj/pipeline.lock`. If a process
dies before cleanup, first verify that no preprocessing pipeline process is
active, remove only that stale lock, then rerun. Never delete the source archive
to recover. If legacy or unsupported derived output is detected, move that
generated directory aside or choose a fresh output root; the pipeline will not
migrate or delete it automatically.

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

Embedding runs separately own `<embedding-output-root>/pipeline.lock`. After an
embedding process dies, first verify that no embedding pipeline process is
active. Remove only that exact stale embedding lock, then rerun the same
idempotent `wsj-embeddings run` command. Do not remove the preprocessing lock as
part of embedding recovery, and do not remove any other file below either
output root.

## Development checks

These checks use generated fixtures and do not read the licensed archive:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
.venv/bin/wsj-embeddings smoke
```
