# WSJ Markdown catalog

This repository turns a local, licensed WSJ HTML/image archive into two small,
research-ready outputs:

- one deterministic UTF-8 Markdown file for each canonical article; and
- one DuckDB catalog that indexes those files, publication time, provenance,
  the optional local header image, and remote editorial-image URLs.

The pipeline does not modify the source archive. It also does not embed,
summarize, classify, download remote images, fetch market data, construct
graphs, label events, or run trading research. Those jobs belong downstream.

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
```

Relative paths resolve from the project root. Source and output roots must not
overlap. Put a custom configuration before the subcommand:

```bash
.venv/bin/wsj-pipeline --config config/other.toml inventory
```

## Four-command CLI

The installed CLI has exactly four subcommands and prints one deterministic
JSON summary per invocation:

```text
inventory                 read-only counts for all discovered HTML/header images
run (--limit N | --full)  incrementally update Markdown and the catalog
     [--reprocess]        explicitly process stale extractor-version rows
validate                  check the existing catalog and generated files
smoke                     run the complete workflow on generated temporary fixtures
```

`run` requires exactly one of a positive `--limit` or `--full`. `validate`,
`run`, and `smoke` return a nonzero status when their final validation fails.
`smoke` ignores the configured archive and output paths.

## Safe smoke-to-full workflow

First prove the installation without touching the licensed archive:

```bash
.venv/bin/wsj-pipeline smoke
```

Then inventory the configured archive without changing either source or
generated state:

```bash
.venv/bin/wsj-pipeline inventory
```

Create or update a bounded sample from the lexically first 20 source paths:

```bash
.venv/bin/wsj-pipeline run --limit 20
.venv/bin/wsj-pipeline validate
```

The bounded run writes to the configured output root, but it never treats
unseen source paths as deleted. Because selection is lexical, it is a safety
sample rather than a way to target the newest year.

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

## Query by publication date

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

## Incremental runs and reprocessing

Normal reruns skip matching source/image fingerprints. If file stats change,
the pipeline hashes HTML before deciding whether extraction is necessary.
New content, changed HTML, and changed selected header images are processed in
bounded transactions; unchanged content is not republished.

An extractor-version change marks existing sources stale but does not silently
start corpus-wide work. Reprocess an explicit scope:

```bash
.venv/bin/wsj-pipeline run --limit 20 --reprocess
.venv/bin/wsj-pipeline run --full --reprocess
```

Review the JSON counters after every run. Extraction failures are isolated and
recorded without article text; a valid duplicate may be promoted in place of a
failed winner.

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

## Development checks

These checks use generated fixtures and do not read the licensed archive:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
```
