# Repository guidance for coding agents

## Read first

The only user-facing documentation is:

1. `README.md` for installation, safe operation, recovery, and queries.
2. `docs/architecture-crash-course.md` for architecture and maintenance.

Keep `AGENTS.md` as control guidance, not a third tutorial. Files under
`docs/superpowers/` are decision and execution history, not current operator
documentation.

## Data and scale safety

- Treat `data/wsj_archive/` as immutable licensed input. Never rename, rewrite,
  move, or delete raw HTML or images.
- Write generated state only below the configured output root. Source and
  output roots must remain disjoint.
- Never stage or commit raw/derived data, DuckDB files, generated Markdown,
  logs, model artifacts, archives, or licensed article excerpts.
- Never put article bodies in errors, audit rows, snapshots, fixtures,
  documentation, or commit messages.
- Automated checks use generated temporary fixtures only. Use
  `.venv/bin/wsj-pipeline smoke` for end-to-end verification.
- Do not run `run --full` or `run --full --reprocess` against the real archive
  unless the user explicitly authorizes that corpus-scale action in the
  current task. A mutating real-data diagnostic must use an explicit small
  `--limit`.
- Before removing `pipeline.lock`, verify that no pipeline process is active.
  Remove only the stale lock and rerun the idempotent command.

## Invariants

- Extract complete editorial content in reading order: headline, distinct
  descriptive metadata, byline, paragraphs, headings, lists, tables, quotes,
  editorial links, figures, captions/credits/creator text, and relevant visible
  interactive text. Exclude chrome, ads, prompts, recommendations, sharing,
  biographies, scripts/styles, hidden duplicates, and repeated metadata.
- Normalization is deterministic and conservative. Do not summarize,
  paraphrase, translate, stem, lowercase, or otherwise change editorial
  meaning.
- Preserve resolved remote HTTP(S) editorial image URLs in Markdown and the
  ordered, deduplicated `inline_image_urls` list. Never download them. Keep the
  optional local companion header image as a separate source-relative path;
  a missing image is valid and warned, not fatal.
- Publish one canonical Markdown file and one `articles` row per winning
  `article_id`. Keep one DuckDB catalog for both the research interface and
  minimal operational state. Operational tables must not store cleaned bodies.
- Preserve identity fallback order: trimmed WSJ ID, normalized canonical-URL
  hash, then source-HTML hash. Preserve winner ordering: nonempty content,
  character count, block count, metadata completeness, later update time, then
  lexical source path.
- Use authoritative timezone-aware publication metadata, normalize it to UTC,
  and derive the indexed date with `America/New_York`. Never substitute created,
  updated, last-published, filename, directory, or filesystem time.
- Keep output paths deterministic:
  `text/YYYY/MM/DD/<sha256-of-namespaced-article-id>.md`, using the New York
  publication date.
- Discover in stable lexical order without inventorying the full corpus before
  a limited yield. Exclude hidden directories and `__MACOSX` at every depth,
  but exclude `backup` only when it is directly below the source root.
- `run` requires exactly one positive `--limit` or explicit `--full`.
  Limited/interrupted runs never infer deletion. Only a completed full run may
  mark unseen sources missing and promote/remove affected winners. Reject an
  absent, non-directory, or inaccessible source root before beginning a run;
  traversal failure must abort before reconciliation.
- Keep mutation under one exclusive lock and bounded transactions. Stage,
  reopen, hash, and atomically replace Markdown before committing its catalog
  state. Reconcile full-run removals in bounded pages. Delete obsolete generated
  files only after commit; validation must report harmless orphans left by
  interrupted cleanup without following directory symlinks.
- Keep worker threads source-only: stat, safe read, hash, and extract within one
  bounded batch. DuckDB access, ranking, Markdown publication, cleanup, and
  validation remain serialized on the coordinator.
- Fail closed on incompatible or malformed generated schemas. Never silently
  migrate, overwrite, or delete legacy derived output.

## Change workflow

Use test-driven development for behavior changes. Every new archive variation,
regression, interruption boundary, or invariant needs a focused generated
fixture before implementation.

If extraction can change Markdown, identity metadata, inline image URLs, or
ranking metrics:

1. add and observe a failing fixture test;
2. implement the smallest extraction change;
3. bump `EXTRACTOR_VERSION`;
4. verify stale rows are skipped without `--reprocess` and handled with an
   explicit bounded reprocess; and
5. update the crash-course contract if semantics changed.

If a table, column, type, constraint, or index changes:

1. update exact schema tests first;
2. update catalog creation/validation, access methods, pipeline writes,
   cross-artifact validation, and documented queries together;
3. bump `CATALOG_SCHEMA_VERSION`; and
4. test refusal of old/malformed output plus a complete fresh fixture run.

There is no automatic schema migration. A version change instructs operators
to move reproducible derived output aside or choose a fresh output root.

## Verification and hygiene

Run only fixture-safe checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/wsj-pipeline smoke
```

Verify documented CLI claims against live `--help`. Resolve relative Markdown
links and audit all tracked Markdown for stale current-facing architecture
claims before committing.

Use `git status --short` and confirm no generated artifacts are tracked:

```bash
git ls-files | rg '^(data|artifacts|outputs)/'
```

If machine configuration, services, Codex configuration, plugins, skills, or
environment-variable names change, follow the global instruction to update
`/home/willis/SETUP_REPLICATION.md` without recording secret values.
