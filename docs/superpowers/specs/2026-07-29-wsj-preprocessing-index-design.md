# WSJ Preprocessing and Publication-Date Index Design

## Purpose

Build a reproducible, incremental pipeline that converts the licensed raw WSJ
archive into a clean, research-ready article dataset. The first version stops
at shared preprocessing and date indexing. Embeddings, temporal knowledge
graphs, market data, and trading strategies are downstream projects.

The pipeline must support the current 2016–2023 archive and later additions for
2024–2026 without rescanning or rewriting unaffected data. Development and
verification use generated fixtures and explicitly limited smoke samples; no
command run as part of development may process the full archive.

## Chosen Approach

Use a staged Python CLI:

1. inventory source HTML and companion images;
2. extract and normalize new or changed records;
3. select one canonical record per WSJ article and audit duplicates;
4. publish affected Parquet partitions atomically;
5. rebuild a small DuckDB query index over the published dataset.

This is preferable to a single streaming job because every stage is
restartable and inspectable. It is preferable to making DuckDB the only
published representation because Parquet remains portable to embedding,
graph, and distributed-compute tools.

## Repository and Runtime Layout

Tracked source:

```text
AGENTS.md
README.md
pyproject.toml
config/wsj_pipeline.toml
src/wsj_pipeline/
tests/
docs/superpowers/specs/
```

Ignored licensed and generated data:

```text
data/wsj_archive/                 # immutable source
data/processed/wsj/
  state/pipeline.duckdb           # operational manifest and extracted rows
  parquet/articles/
    publication_year_ny=YYYY/
      publication_month_ny=MM/
        articles.parquet
  parquet/audit/
    duplicates.parquet
    failures.parquet
    missing_images.parquet
    runs.parquet
  index/wsj.duckdb                # user-facing tables/views and date index
  staging/                        # atomic publication workspace
```

All roots are configurable. Stored paths are relative to the configured source
or output root where possible, so moving the project does not invalidate the
dataset.

## Components

### Source discovery and pairing

The inventory stage recursively finds `.html` files under the source root in
stable lexical order. It excludes hidden directories, `__MACOSX`, and the
top-level `backup` directory. A companion image is a file beside the HTML whose
stem is `<html-stem>_main_image` and whose extension is one of JPEG, PNG, or
WebP. No image is a valid condition recorded in the missing-image audit.
Multiple matching image files are an extraction warning resolved by stable
extension/path ordering.

Discovery records relative path, byte size, and nanosecond modification time.
That inexpensive stat tuple detects candidates for reprocessing. For each
candidate already being read, the processor records a SHA-256 of the HTML
bytes. Unchanged stat tuples are skipped; matching content hashes prevent a
metadata-only filesystem change from creating a new logical record.

The source archive is read-only. The pipeline never renames, moves, or modifies
raw HTML or images.

### HTML extraction

Extraction has a versioned interface and produces one normalized source record
or a structured failure. It uses, in order:

- WSJ `<meta>` fields for article ID, canonical URL, headline, summary,
  timestamps, author, section, type, access, keywords, and declared counts;
- JSON-LD as a structured fallback and for image caption, credit, dimensions,
  and creator;
- the article DOM for ordered body paragraphs, excluding ads, related-content
  cards, author biographies, captions, and dynamic insets;
- filename and filesystem metadata only as explicitly flagged fallbacks.

Body output includes both an ordered list of cleaned paragraphs and a
double-newline-joined `body_text`. Whitespace and Unicode are normalized without
lowercasing, stemming, sentence splitting, summarizing, or otherwise changing
editorial meaning. Raw metadata selected during parsing is retained as compact
JSON for forward compatibility. Raw full HTML is not copied into Parquet.

Every extracted row records the extractor version, source content hash,
extraction timestamp, extraction warnings, and source-relative paths.

### Identity and deterministic duplicates

`article_key` is derived from the WSJ article ID when present. Otherwise it is
a SHA-256 of the normalized canonical URL; if both are missing, it is a
SHA-256 of the source HTML.

When several source rows have the same article key, the canonical winner is
selected deterministically by:

1. valid publication timestamp;
2. nonempty body;
3. larger body word count;
4. greater metadata completeness;
5. later WSJ `updated_at`;
6. lexical source path as the final tie-breaker.

Nonwinners are written to the duplicate audit with their winner, rank, and
reason. They remain in operational state and can be re-evaluated after parser
changes. Because archived pages may contain post-publication edits, the
pipeline preserves `updated_at` and does not claim that body text represents
the first published revision.

### Dates

The authoritative event time is the parsed WSJ publication timestamp. It is
stored timezone-aware in UTC as `published_at_utc`. The pipeline derives:

- `publication_date_utc`;
- `published_at_new_york`;
- `publication_date_new_york`;
- `publication_year_ny`;
- `publication_month_ny`.

New York conversion uses the IANA `America/New_York` zone and therefore
observes daylight-saving transitions. Parquet partitions use New York year and
month. `updated_at_utc`, `created_at_utc`, and `last_published_at_utc` are
preserved separately and never substituted silently for publication time.

Rows without a valid publication timestamp are retained in failure/audit state
but are not published as canonical dated articles.

### Canonical article schema

The published article table contains:

- identity/provenance: `article_key`, WSJ ID, canonical URL, source path,
  source hash, extractor version;
- editorial fields: headline, original headline, summary, body paragraphs,
  body text, authors, keywords, section, page, article type, access;
- timestamps and derived dates described above;
- image fields: relative path, presence flag, media type, byte size, caption,
  credit, creator, alt text, declared width and height;
- quality fields: body word count, metadata completeness, warnings;
- compact selected raw metadata JSON.

Lists use native Parquet list types. Timestamps use Arrow timezone-aware
timestamps. Schema versioning is explicit and checked before publication.

### Operational state and incremental publication

`pipeline.duckdb` stores runs, the source manifest, extracted source records,
failures, and duplicate decisions. A run:

1. acquires an exclusive pipeline lock;
2. inventories sources and identifies new, changed, and optionally missing
   paths;
3. processes candidates in bounded batches and commits progress after each
   batch;
4. recomputes canonical winners for affected article keys;
5. determines affected old and new year/month partitions;
6. writes each affected partition to staging, validates it, then atomically
   replaces the published partition file;
7. publishes audit Parquet files and the query index;
8. records a run summary and releases the lock.

An interrupted extraction can resume from committed batches. An interrupted
publication leaves the previous partition intact. Re-running with no source or
extractor-version changes is a no-op apart from a run record.

A changed extractor/schema version does not trigger an accidental archive-wide
run. The CLI reports that records are stale and requires an explicit
`--reprocess` scope.

### DuckDB publication-date index

The published `wsj.duckdb` is rebuilt transactionally from canonical Parquet
and contains:

- an `articles` view over all article partitions;
- a materialized `publication_index` table ordered by
  `(publication_date_new_york, published_at_utc, article_key)`;
- a `daily_publication_counts` view;
- views over duplicate, failure, missing-image, and run audits.

The index stores article keys and dates rather than duplicating article bodies.
Typical queries by exact date or bounded date range are documented in the
README.

## CLI and Safety

The package exposes `wsj-pipeline` with:

- `inventory`: inspect candidate counts and source layout;
- `process`: extract a bounded set of new/changed records;
- `publish`: deduplicate and publish affected partitions;
- `index`: rebuild the DuckDB query index;
- `validate`: check state, Parquet, and index invariants;
- `run`: run the incremental stages in order;
- `smoke`: run the entire workflow on generated test fixtures.

Potentially large commands print source/output roots and an estimated candidate
count before processing. Development documentation uses `smoke` and
`--limit`. A full unbounded run requires an explicit `--full` flag; `run`
without `--full` or `--limit` exits before processing. This prevents an agent
or developer from accidentally launching a 100+ GB job.

## Errors and Audits

One malformed article does not stop a batch. Failures include stage, source
path, stable error code, concise message, extractor version, and run ID.
Warnings cover missing images, ambiguous image matches, metadata fallbacks,
empty bodies, declared/observed word-count discrepancies, and suspicious date
relationships.

Fatal conditions stop publication: incompatible schema, corrupt operational
state, lock contention, duplicate published article keys, partition/date
mismatch, invalid timezone handling, or an index whose article keys differ
from canonical Parquet.

Logs must not include article bodies, secrets, or licensed content excerpts.

## Validation and Testing

Unit tests use small synthetic HTML documents representing:

- complete metadata and body;
- missing image;
- missing ID with canonical-URL fallback;
- missing/invalid publication timestamp;
- duplicate candidates and deterministic tie-breaking;
- New York dates around UTC midnight and daylight-saving changes;
- extraction failures and recovery;
- changed and unchanged source fingerprints.

Integration tests run the complete pipeline in a temporary directory and
verify:

- a second unchanged run performs no article extraction;
- adding a new year processes only new files;
- changing a source replaces the correct affected partition;
- canonical article keys are unique;
- audit rows point to valid source/winner keys;
- Parquet and DuckDB contain identical canonical keys and date values;
- an interrupted staged publication cannot damage an existing partition;
- an unbounded `run` without `--full` is rejected.

Repository verification runs formatting/linting, static checks where
configured, unit tests, and the fixture-only smoke command. It may run a
real-archive sample only with an explicit small `--limit`; it must never start
the full archive job.

## Documentation for Future Work

`README.md` documents setup, commands, data layout, date semantics, sample
queries, incremental additions, recovery, and the boundary between this
pipeline and downstream research.

The repository `AGENTS.md` tells future agents:

- treat `data/wsj_archive` as immutable licensed input;
- never run an unbounded archive job without explicit user authorization;
- use generated fixtures or a very small explicit smoke limit;
- keep raw and derived data out of Git;
- preserve provenance, UTC/New York dates, deterministic ordering, and
  idempotence;
- update this design and schema documentation when changing extraction or
  canonicalization behavior.

## Out of Scope

- text, image, or multimodal embeddings;
- OCR, image caption generation, or image transformations;
- entity/event extraction and temporal knowledge graphs;
- market calendars, market data, labels, backtests, or trading strategies;
- distributed execution and cloud orchestration;
- recovering historical article revisions not present in the archive.
