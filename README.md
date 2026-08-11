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

The repository also contains a downstream article-text embedding slice.
`wsj-embeddings smoke` creates one generated canonical article in a temporary
directory, encodes it with a deterministic fake adapter, publishes a
2,048-dimensional normalized `article_text` vector to a separate five-table
DuckDB catalog, validates it, and removes the fixture. It never reads the
configured archive or calls a network service. `wsj-embeddings pilot` is a
separate, explicit hosted Jina v4 measurement command: it sends only fixed
internally generated text and PNG bytes, never reads the archive or creates
catalog/output state. `wsj-embeddings inventory` reads only canonical catalog
metadata, while `wsj-embeddings run` exposes an explicitly authorized,
strictly limited production Jina article-text scope.

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

The separate `wsj-embeddings` CLI has these four subcommands:

```text
smoke      publish and validate one generated article text embedding
pilot      send fixed generated text/image probes to hosted Jina v4
inventory  count canonical/text/header eligibility without credentials
run        embed a positive lexical article-id limit after explicit authorization
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
no implicit root defaults. `inventory` and `run` require explicit source,
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

An article-text run requires both a strictly positive lexical `article_id`
limit and the separate `--authorize-hosted-processing` assertion. A configured
`JINA_API_KEY` alone is insufficient:

```bash
.venv/bin/wsj-embeddings run \
  --source-root data/wsj_archive \
  --preprocessing-output-root data/processed/wsj \
  --embedding-output-root data/embeddings/wsj \
  --limit 5 \
  --authorize-hosted-processing
```

The run sends each selected canonical Markdown artifact in full with hosted
truncation disabled. Run/configuration setup and each selected item's
queue/recovery registration use separate bounded transactions. Each validated
vector and its successful work checkpoint commit before the next article is
attempted. Replaying the same limit reuses unchanged successes without another
hosted request; when input changes, the mismatched vector is removed atomically
with checkpoint invalidation before replacement is attempted. Retryable
failures and interrupted in-progress work are attempted again, while terminal
failures stay visible and are not retried implicitly. Limited runs never infer
removal outside the selected scope. Success JSON includes content-free
`reused`, `attempted`, `succeeded`, `retryable`, `terminal`, and `interrupted`
counts.

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

## Article text embedding catalog contract

The downstream catalog is a separate schema-version-2 `catalog.duckdb` below a
root that must be disjoint from both the licensed source root and preprocessing
output root. The generated smoke and injected-adapter coordinator exercise the
same catalog contract as the explicitly rooted, limit-only production CLI.
Version 1 output is refused without migration; move reproducible derived output
aside or choose a fresh embedding output root.

The catalog has exactly five base tables and no indexes:

| Table | Ordered columns | Key |
|---|---|---|
| `metadata` | `key VARCHAR`, `value VARCHAR` | `key` |
| `embedding_configurations` | `configuration_id VARCHAR`, `model VARCHAR`, `task VARCHAR`, `dimensions INTEGER`, `output_type VARCHAR`, `normalization VARCHAR` | `configuration_id` |
| `runs` | `run_id VARCHAR`, `configuration_id VARCHAR`, `articles INTEGER`, `embeddings INTEGER`, `reused INTEGER`, `attempted INTEGER`, `succeeded INTEGER`, `retryable INTEGER`, `terminal INTEGER`, `interrupted INTEGER`, `started_at TIMESTAMPTZ` | `run_id` |
| `embedding_work_items` | `article_id VARCHAR`, `modality VARCHAR`, `configuration_id VARCHAR`, `input_sha256 VARCHAR`, `state VARCHAR`, `attempt_count INTEGER`, `error_code VARCHAR`, `status_code INTEGER`, `retry_after_seconds DOUBLE`, `last_run_id VARCHAR`, `updated_at TIMESTAMPTZ` | `(article_id, modality, configuration_id)` |
| `embeddings` | `article_id VARCHAR`, `modality VARCHAR`, `configuration_id VARCHAR`, `published_at_utc TIMESTAMPTZ`, `publication_date_new_york DATE`, `dimensions INTEGER`, `input_sha256 VARCHAR`, `stored_vector_sha256 VARCHAR`, `vector FLOAT[2048]` | `(article_id, modality, configuration_id)` |

Only the three failure-detail columns are nullable; they retain classified
codes and numeric response/retry metadata, never exception text or editorial
content. Work states are `queued`, `in_progress`, `succeeded`, `retryable`,
`terminal`, and `interrupted`. `configuration_id` is the SHA-256 of canonical
JSON covering every adapter-profile field. `input_sha256` hashes the exact
canonical Markdown bytes. `stored_vector_sha256` hashes the normalized vector
encoded as little-endian float32 values. The current modality is exactly
`article_text`; vectors must be finite, nonzero, L2-normalized, and exactly
2,048-dimensional.

A retained catalog can be queried without joining back to article bodies:

```sql
SELECT
    e.article_id,
    e.published_at_utc,
    e.publication_date_new_york,
    c.model,
    c.task,
    e.vector
FROM embeddings AS e
JOIN embedding_configurations AS c USING (configuration_id)
WHERE e.modality = 'article_text'
  AND e.publication_date_new_york
      BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
ORDER BY e.published_at_utc, e.article_id;
```

Embedding validation opens both catalogs read-only, checks exact schemas before
data, links each vector to a canonical `articles` row, rehashes canonical
Markdown through no-follow descriptors, checks publication metadata and both
hashes, and verifies that successful run counts still have published vectors.
It emits stable issues without article text or vector values.

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
linked, or replaced leaves are rejected. Embedding mutation is confined below
its disjoint output root. The coordinator opens every output-root component
without following links, then creates its catalog and exclusive `pipeline.lock`
relative to that anchored directory. Lock cleanup compares file identity and
will not remove a pathname replaced by another process.

## Development checks

These checks use generated fixtures and do not read the licensed archive:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
.venv/bin/wsj-embeddings smoke
```
