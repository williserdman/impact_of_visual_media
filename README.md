# WSJ preprocessing and publication-date index

This repository turns the local licensed WSJ HTML/image archive into a
reproducible research dataset:

- normalized article records in monthly, New York-date Parquet partitions;
- deterministic duplicate selection with provenance and audit datasets;
- an incremental DuckDB state database;
- a compact DuckDB index for publication-date queries.

Embeddings, knowledge graphs, market labels, backtests, and trading strategies
are intentionally downstream. The raw archive is never modified.

## Current source archive

The archive currently contains 2016–2023 data under `data/wsj_archive`, with
both direct (`2016/*.html`) and nested (`2023/2023/*.html`) layouts. Future
2024–2026 folders can use either layout because discovery is recursive.

An HTML file is paired with a sibling named
`<html-stem>_main_image.{jpg,jpeg,png,webp}`. Missing images are valid and are
reported in the audit data. Hidden directories, `__MACOSX`, and the top-level
`backup` directory are excluded.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

If Ubuntu lacks `ensurepip`, install its matching `python3-venv` package or use
the official PyPA `virtualenv.pyz` to create `.venv`. Runtime and development
dependencies are declared in `pyproject.toml`.

The tracked default configuration is `config/wsj_pipeline.toml`:

```toml
[paths]
source = "data/wsj_archive"
output = "data/processed/wsj"

[pipeline]
batch_size = 250
```

Relative paths resolve from the project root. To use another configuration,
put the global option before the command:

```bash
.venv/bin/wsj-pipeline --config config/other.toml inventory --limit 20
```

## Safe first commands

The full end-to-end smoke test creates its own temporary archive and cannot
read the licensed corpus:

```bash
.venv/bin/wsj-pipeline smoke
```

Inspect a deterministic, lexically first sample of the real archive:

```bash
.venv/bin/wsj-pipeline inventory --limit 20
.venv/bin/wsj-pipeline run --limit 20
```

`process` and `run` reject an invocation that has neither `--limit` nor
`--full`. They also reject using both. Automated tests only use generated
fixtures.

## Incremental and full operation

The first corpus-wide run is deliberately explicit and will be long-running:

```bash
.venv/bin/wsj-pipeline run --full
```

Use a durable terminal such as `tmux` for that operator-authorized run. This
repository's implementation and verification did not launch it.

On later `--full` runs, the pipeline inventories all paths but parses only new
or changed sources. A completed full inventory also reconciles HTML files that
were removed since the prior full run; bounded `--limit` runs never infer
deletions from unseen paths. Add 2024–2026 folders under `data/wsj_archive`,
then use:

```bash
.venv/bin/wsj-pipeline inventory
.venv/bin/wsj-pipeline run --full
```

A small `--limit` always selects the lexically first source paths, so it is a
smoke tool—not a way to discover newly appended years near the end of the
archive.

When extraction behavior changes, bump `EXTRACTOR_VERSION`. Existing rows are
reported as stale and remain untouched until reprocessing is explicit:

```bash
.venv/bin/wsj-pipeline run --limit 20 --reprocess
.venv/bin/wsj-pipeline run --full --reprocess
```

Available stages are:

```bash
.venv/bin/wsj-pipeline inventory --limit 20
.venv/bin/wsj-pipeline process --limit 20
.venv/bin/wsj-pipeline publish
.venv/bin/wsj-pipeline index
.venv/bin/wsj-pipeline validate
.venv/bin/wsj-pipeline run --limit 20
```

`run` is the normal operation because it extracts, canonicalizes, publishes,
indexes, and validates under one exclusive lock.

## Generated layout

All outputs are ignored by Git:

```text
data/processed/wsj/
├── state/
│   └── pipeline.duckdb
├── parquet/
│   ├── articles/
│   │   └── publication_year_ny=YYYY/
│   │       └── publication_month_ny=MM/
│   │           └── articles.parquet
│   └── audit/
│       ├── duplicates.parquet
│       ├── failures.parquet
│       ├── missing_images.parquet
│       └── runs.parquet
├── index/
│   └── wsj.duckdb
└── staging/
```

The state database is operational and should not be queried as the research
dataset. Canonical Parquet is the portable data product. `index/wsj.duckdb`
contains views over that Parquet plus a materialized body-free
`publication_index`.

DuckDB view definitions contain the resolved Parquet location. If the output
directory moves, rebuild the small index:

```bash
.venv/bin/wsj-pipeline index
```

## Dates and look-ahead safety

`article.published` is the authoritative event time. Each canonical row stores:

- `published_at_utc`;
- `publication_date_utc`;
- `published_at_new_york`;
- `publication_date_new_york`;
- New York partition year and month.

Conversion uses the IANA `America/New_York` timezone and observes daylight
saving. Created, updated, and last-published timestamps remain separate. The
archive may contain article text updated after initial publication; the
pipeline preserves the update time but cannot recover revisions absent from
the source archive.

## Querying by publication date

Open the published index:

```bash
duckdb data/processed/wsj/index/wsj.duckdb
```

Exact New York publication date:

```sql
SELECT article_key, published_at_utc, source_path
FROM publication_index
WHERE publication_date_new_york = DATE '2023-08-24'
ORDER BY published_at_utc, article_key;
```

Date range joined to canonical article content:

```sql
SELECT
    i.publication_date_new_york,
    a.article_key,
    a.headline,
    a.body_text,
    a.image_path
FROM publication_index AS i
JOIN articles AS a USING (article_key)
WHERE i.publication_date_new_york
      BETWEEN DATE '2023-08-01' AND DATE '2023-08-31'
ORDER BY i.publication_date_new_york, i.published_at_utc, i.article_key;
```

Daily counts:

```sql
SELECT *
FROM daily_publication_counts
ORDER BY publication_date_new_york;
```

## Recovery and validation

Extraction commits bounded batches. Monthly Parquet and the query index are
written to staging, reopened for validation, and atomically replaced. An
interruption cannot partially overwrite a valid partition.

Committed extraction work is recorded in a durable pending-key queue.
Canonical changes similarly record durable dirty month partitions before
publication. If a run stops between stages, rerunning the same command drains
those queues even when every source file is otherwise unchanged.

```bash
.venv/bin/wsj-pipeline validate
```

Validation scans only compact provenance/date columns—not article bodies—and
checks schema versions, unique article keys, partition/date agreement, UTC/New
York derivations, duplicate-winner references, durable publication queues, and
full provenance/date equality between canonical Parquet and the materialized
date index.

Mutating commands use `data/processed/wsj/state/pipeline.lock`. If a process
dies without cleanup, first verify no pipeline process is active, then remove
that one stale lock file and rerun the command.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
```

Tests generate tiny HTML/image fixtures under temporary directories and do not
read `data/wsj_archive`. Parser changes require a regression fixture, an
extractor-version bump when outputs can change, and an update to the design
document when schema or canonicalization semantics change.

The detailed design is in
`docs/superpowers/specs/2026-07-29-wsj-preprocessing-index-design.md`.
