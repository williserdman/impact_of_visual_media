# Repository guidance for coding agents

## Scope

This repository preprocesses the local licensed WSJ archive into canonical
Parquet and a publication-date DuckDB index. Text/image embeddings, temporal
knowledge graphs, market data, and trading strategies are separate downstream
projects.

Read `README.md` and
`docs/superpowers/specs/2026-07-29-wsj-preprocessing-index-design.md` before
changing pipeline behavior.

## Nonnegotiable data safety

- Treat `data/wsj_archive/` as immutable input. Never rename, rewrite, move, or
  delete raw HTML or companion images as part of pipeline work.
- Do not stage or commit anything under `data/`, generated Parquet, DuckDB
  databases, logs, archives, model artifacts, or licensed article excerpts.
- Never run `wsj-pipeline process --full`, `run --full`, or
  `--full --reprocess` unless the user explicitly authorizes that corpus-scale
  operation in the current task. A full run is long-running.
- Use `.venv/bin/wsj-pipeline smoke` for end-to-end verification. If real data
  is genuinely needed, use a small explicit `--limit`.
- Do not log article bodies or include licensed excerpts in exceptions,
  snapshots, fixtures, documentation, or commits.
- Missing header images are valid data and belong in the audit output, not a
  fatal error.

## Archive facts

- Current source years are 2016–2023; 2024–2026 will be added later.
- Layouts vary between `YEAR/*.html` and `YEAR/YEAR/*.html`; discovery must
  remain recursive.
- A companion image is a sibling named
  `<html-stem>_main_image.{jpg,jpeg,png,webp}`.
- Hidden directories, `__MACOSX`, and the top-level `backup` directory are
  excluded from discovery.
- Publication timestamps are sourced from WSJ metadata, not filename epochs,
  filesystem times, folder years, or `dateLastPubbed`.

## Architecture

- `discovery.py`: stable source enumeration and image pairing.
- `extract.py`: versioned, content-free-failure HTML/JSON-LD extraction.
- `state.py`: operational DuckDB manifest, batches, failures, and runs.
- `canonical.py`: article identity, winner ranking, and duplicate audit logic.
- `publish.py`: explicit Arrow schemas and atomic monthly/audit Parquet writes.
- `index.py`: transactional body-free date index plus Parquet-backed views.
- `validate.py`: cross-artifact invariants.
- `service.py`: locking and end-to-end orchestration.
- `cli.py`: safety gates and JSON command output.

Keep these boundaries focused. Avoid putting research-specific embeddings,
entity extraction, graph construction, or market logic into the shared
preprocessing package.

## Invariants to preserve

- Raw inputs are read-only; outputs are written only below the configured
  output root.
- `article_key` prefers WSJ ID, then normalized canonical URL hash, then HTML
  content hash.
- Canonical winner ordering is deterministic: valid publication time,
  nonempty body, body word count, metadata completeness, update time, lexical
  source path.
- Store exact timezone-aware UTC publication time and derived
  `America/New_York` time/date. Never silently substitute created, updated, or
  last-published time.
- Parquet partitions use New York publication year/month and contain unique
  canonical article keys.
- Unchanged runs do not re-extract or rewrite article partitions.
- Changed partitions and the DuckDB index are staged, validated, and atomically
  replaced.
- Duplicate, failure, missing-image, and run audits contain provenance but no
  article-body text.
- `process` and `run` must continue to require exactly one of `--limit` or
  `--full`.

## Development workflow

Use test-driven development for behavior changes:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
```

All automated tests must use generated temporary fixtures. Add a focused
fixture for every new WSJ HTML variation or regression.

When an extraction change can alter normalized records:

1. add and observe a failing regression test;
2. update the parser;
3. bump `EXTRACTOR_VERSION`;
4. verify stale rows are skipped without `--reprocess`;
5. update the design/schema documentation if fields or semantics changed.

When a schema changes, also update explicit Arrow schemas, state schema
versioning/migration behavior, DuckDB index creation, validation, README
examples, and tests.

## Operations and repository hygiene

- The default config is `config/wsj_pipeline.toml`.
- Canonical output is `data/processed/wsj/parquet/articles/`.
- The user-facing index is `data/processed/wsj/index/wsj.duckdb`.
- If outputs move, rebuild the index because its views use resolved Parquet
  paths.
- Before removing a stale pipeline lock, verify no pipeline process is active.
- Before committing, use `git status --short` and confirm no data artifact is
  tracked with `git ls-files | rg '^(data|artifacts|outputs)/'`.
- If machine configuration, services, Codex configuration, plugins, skills, or
  environment-variable names change, follow the global instruction to update
  `/home/willis/SETUP_REPLICATION.md` in the same task without recording
  secrets.
