# Simplified Markdown Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Parquet-oriented prototype with an incremental pipeline
that writes one canonical editorial Markdown file per unique WSJ article and
indexes the file, images, provenance, and publication date in one DuckDB
catalog.

**Architecture:** Discovery and extraction remain bounded, read-only stages.
The new catalog owns incremental manifests, candidate quality, duplicates,
failures, runs, and the research-facing `articles` table; the pipeline stages
and atomically publishes deterministic dated Markdown files before committing
catalog state. Validation streams the generated file tree and cross-checks it
against DuckDB without loading the corpus into Python memory.

**Tech Stack:** Python 3.11+, Beautiful Soup 4, lxml, DuckDB, standard-library
`zoneinfo`, pytest, Ruff, setuptools.

## Global Constraints

- Treat `data/wsj_archive/` as immutable input and write only below the
  configured output root.
- Never log or commit licensed article excerpts.
- Keep all raw and derived data ignored by Git.
- Require exactly one of a positive `--limit` or explicit `--full` for `run`.
- Never run a real-corpus `run --full` or `--full --reprocess` during
  implementation or verification.
- Automated tests and end-to-end verification use generated temporary fixtures
  only.
- Preserve all editorial text in reading order while excluding ads, chrome,
  prompts, recommendations, biographies, scripts, and repeated metadata.
- Store remote editorial image URLs but never download them.
- Use exact UTC publication time and derive the indexed date with
  `America/New_York`.
- Reconcile missing sources only after a completed `--full` inventory.
- Keep user-facing documentation to `README.md` and
  `docs/architecture-crash-course.md`; `AGENTS.md` remains an agent control
  file, and Superpowers files remain internal history.

---

## Target File Structure

### Production package

- `src/wsj_pipeline/config.py`: safe source/output paths and batch size.
- `src/wsj_pipeline/models.py`: immutable discovery/extraction/catalog result
  value objects.
- `src/wsj_pipeline/discovery.py`: recursive stable source enumeration and
  local header-image pairing.
- `src/wsj_pipeline/extract.py`: editorial DOM-to-Markdown extraction,
  identity metadata, dates, and inline image URLs.
- `src/wsj_pipeline/catalog.py`: versioned DuckDB schema and transactional
  state/catalog methods.
- `src/wsj_pipeline/pipeline.py`: identity, duplicate ranking, incremental
  orchestration, locking, and atomic Markdown publication.
- `src/wsj_pipeline/validate.py`: streaming catalog/filesystem/date/hash/orphan
  invariants.
- `src/wsj_pipeline/cli.py`: four-command CLI and deterministic JSON output.
- `src/wsj_pipeline/__main__.py`: module entry point.

### Removed production modules

- `src/wsj_pipeline/state.py`
- `src/wsj_pipeline/canonical.py`
- `src/wsj_pipeline/publish.py`
- `src/wsj_pipeline/index.py`
- `src/wsj_pipeline/service.py`

### Tests

- Retain and adapt: `tests/fixtures.py`, `tests/test_config.py`,
  `tests/test_discovery.py`, `tests/test_dates.py`, `tests/test_extract.py`,
  `tests/test_cli_safety.py`, `tests/test_validate.py`, `tests/test_smoke.py`.
- Create: `tests/test_catalog.py`, `tests/test_pipeline.py`,
  `tests/test_recovery.py`.
- Remove after equivalent coverage exists: `tests/test_state.py`,
  `tests/test_canonical.py`, `tests/test_incremental.py`, `tests/test_publish.py`,
  `tests/test_publish_atomicity.py`, `tests/test_index.py`,
  `tests/test_service.py`.

---

### Task 1: Simplify configuration and value objects

**Files:**
- Modify: `src/wsj_pipeline/config.py`
- Modify: `src/wsj_pipeline/models.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_models.py`

**Interfaces:**
- `PipelineConfig(source_root: Path, output_root: Path, batch_size: int = 250)`
- `PipelineConfig.catalog_db -> Path`
- `PipelineConfig.text_root -> Path`
- `PipelineConfig.staging_root -> Path`
- `PipelineConfig.lock_path -> Path`
- `SourceCandidate(relative_html_path, html_size, html_mtime_ns,
  relative_header_image_path, header_image_size, header_image_mtime_ns,
  warnings)`
- `ExtractedArticle(article_id, relative_html_path, source_html_sha256,
  extractor_version, canonical_url, published_at_utc,
  publication_date_new_york, updated_at_utc, markdown_text,
  editorial_character_count, editorial_block_count, metadata_completeness,
  relative_header_image_path, inline_image_urls, warnings)`

- [ ] **Step 1: Write failing configuration and model tests**

```python
def test_config_exposes_single_catalog_and_text_paths(tmp_path: Path) -> None:
    config = PipelineConfig(tmp_path / "archive", tmp_path / "processed")
    assert config.catalog_db == tmp_path / "processed" / "catalog.duckdb"
    assert config.text_root == tmp_path / "processed" / "text"
    assert config.staging_root == tmp_path / "processed" / "staging"
    assert config.lock_path == tmp_path / "processed" / "pipeline.lock"


def test_extracted_article_contract_is_immutable() -> None:
    assert ExtractedArticle.__dataclass_params__.frozen is True
```

- [ ] **Step 2: Run the focused tests and confirm they fail against legacy
  paths/models**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q
```

Expected: failures for missing `catalog_db`, `text_root`, and the new extraction
fields.

- [ ] **Step 3: Implement the new path properties and immutable dataclasses**

Remove `state_db`, `parquet_root`, and `index_db`. Keep overlap and positive
batch-size checks. Use tuples for warnings and inline URLs.

- [ ] **Step 4: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q
.venv/bin/ruff check src/wsj_pipeline/config.py src/wsj_pipeline/models.py tests/test_config.py tests/test_models.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/wsj_pipeline/config.py src/wsj_pipeline/models.py tests/test_config.py tests/test_models.py
git commit -m "refactor: define Markdown catalog contracts"
```

---

### Task 2: Preserve stable discovery and header-image pairing

**Files:**
- Modify: `src/wsj_pipeline/discovery.py`
- Modify: `tests/test_discovery.py`

**Interfaces:**
- Consumes: `SourceCandidate` from Task 1.
- Produces:
  `discover_sources(source_root: Path, limit: int | None = None) -> Iterator[SourceCandidate]`.

- [ ] **Step 1: Update discovery tests to the renamed header-image fields**

Retain assertions for recursive mixed layouts, lexical ordering, excluded
directories, missing images, extension priority, global limit ordering, stat
fingerprints, and one directory scan per source directory.

```python
assert candidate.relative_header_image_path == "2024/story_main_image.jpg"
assert candidate.header_image_size == len(image_bytes)
```

- [ ] **Step 2: Run discovery tests and observe contract failures**

```bash
.venv/bin/python -m pytest tests/test_discovery.py -q
```

Expected: failures reference legacy `relative_image_path` names.

- [ ] **Step 3: Rename discovery fields without changing archive traversal**

Keep `EXCLUDED_DIRECTORIES`, `IMAGE_EXTENSIONS`, stable global sorting, and the
directory image cache. Do not read article contents in discovery.

> **Task 12 repair clarification (2026-08-01):** Review later found that the
> inherited global sort and corpus-lived directory cache violated the bounded
> design. The final traversal preserves global lexical order with per-directory
> sorting, retains only the current directory's image index, stops after a
> limited yield, and excludes `backup` only at the source-root level.

- [ ] **Step 4: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_discovery.py -q
.venv/bin/ruff check src/wsj_pipeline/discovery.py tests/test_discovery.py
```

- [ ] **Step 5: Commit**

```bash
git add src/wsj_pipeline/discovery.py tests/test_discovery.py
git commit -m "refactor: name local header image explicitly"
```

---

### Task 3: Build editorial DOM-to-Markdown extraction

**Files:**
- Modify: `tests/fixtures.py`
- Replace: `src/wsj_pipeline/extract.py`
- Replace: `tests/test_extract.py`
- Retain/adapt: `tests/test_dates.py`

**Interfaces:**
- `EXTRACTOR_VERSION = "3"`
- `parse_wsj_timestamp(value: str | None) -> datetime | None`
- `derive_publication_date_new_york(value: datetime) -> date`
- `derive_article_id(wsj_id: str | None, canonical_url: str | None,
  html_sha256: str) -> str`
- `extract_article(candidate: SourceCandidate, source_root: Path) -> ExtractedArticle`
- Raises `ExtractionError(code: str, message: str)` with content-free messages.

- [ ] **Step 1: Expand the fixture writer for ordered editorial elements**

Add fixture parameters for dek, headings, lists, table rows, quotes, editorial
links, inline figures, captions, credits, interactive text, and excluded page
elements. Fixture prose must remain synthetic.

- [ ] **Step 2: Write failing extraction tests for complete reading order**

```python
def test_extracts_all_editorial_content_as_markdown(tmp_path: Path) -> None:
    article = extract_fixture_article(tmp_path)
    assert article.markdown_text.splitlines() == [
        "# Synthetic Headline",
        "",
        "Synthetic dek.",
        "",
        "By Test Reporter",
        "",
        "Opening paragraph with an [editorial link](https://example.com/source).",
        "",
        "## Synthetic section",
        "",
        "- First item",
        "- Second item",
        "",
        "> Synthetic pull quote",
    ]
```

Add separate tests proving excluded ads/prompts/related cards/biographies do not
appear, identical metadata is emitted once, Unicode is normalized, and empty
editorial content is retained with `empty_content` warning.

- [ ] **Step 3: Write failing inline-image and link tests**

```python
assert article.inline_image_urls == (
    "https://images.wsj.net/inline-one.jpg",
    "https://images.wsj.net/inline-two.png",
)
assert "![Synthetic alt](https://images.wsj.net/inline-one.jpg)" in article.markdown_text
assert "Synthetic caption. TEST CREDIT." in article.markdown_text
```

Cover duplicate image URLs, relative URL resolution against the canonical URL,
non-HTTP image rejection, and exclusion of images inside ads/related cards.

- [ ] **Step 4: Run extraction/date tests and confirm failures**

```bash
.venv/bin/python -m pytest tests/test_extract.py tests/test_dates.py -q
```

- [ ] **Step 5: Implement structured metadata extraction and conservative
  Markdown rendering**

Walk the article root in DOM order. Render supported block elements with small,
pure helpers for paragraphs, headings, lists, tables, quotes, links, figures,
and interactives. Deduplicate only exact normalized metadata blocks and exact
image URLs; do not globally deduplicate body paragraphs.

- [ ] **Step 6: Implement identity and date failure behavior**

Require an aware publication timestamp, normalize it to UTC, derive the New
York date with `zoneinfo`, and use the approved WSJ-ID/URL/hash identity chain.

- [ ] **Step 7: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_extract.py tests/test_dates.py -q
.venv/bin/ruff check src/wsj_pipeline/extract.py tests/fixtures.py tests/test_extract.py tests/test_dates.py
```

- [ ] **Step 8: Commit**

```bash
git add src/wsj_pipeline/extract.py tests/fixtures.py tests/test_extract.py tests/test_dates.py
git commit -m "feat: extract complete editorial Markdown"
```

---

### Task 4: Create the single versioned DuckDB catalog

**Files:**
- Create: `src/wsj_pipeline/catalog.py`
- Create: `tests/test_catalog.py`
- Remove after tests migrate: `src/wsj_pipeline/state.py`
- Remove after tests migrate: `tests/test_state.py`

**Interfaces:**
- `CATALOG_SCHEMA_VERSION = "1"`
- `CatalogError(RuntimeError)`
- `Catalog.open(path: Path) -> Catalog`
- Context-manager lifecycle closes the DuckDB connection.
- `begin_run(command: str, scope: str) -> str`
- `finish_run(run_id: str, status: str, summary: Mapping[str, object]) -> None`
- `classify(candidate: SourceCandidate, extractor_version: str) -> CandidateStatus`
- Transactional methods for manifest, candidate, duplicate, failure, and article
  replacement.

- [ ] **Step 1: Write failing catalog schema tests**

Assert the exact tables `metadata`, `runs`, `source_manifest`, `candidates`,
`duplicates`, `failures`, and `articles`; assert schema version `1`; inspect
`PRAGMA table_info('articles')` for the approved fields and types.

```python
assert article_columns == {
    "article_id",
    "source_html_path",
    "cleaned_markdown_path",
    "header_image_path",
    "inline_image_urls",
    "published_at_utc",
    "publication_date_new_york",
    "cleaned_markdown_sha256",
}
```

- [ ] **Step 2: Write lifecycle, incompatible-schema, and transaction tests**

Verify run JSON is content-free and deterministic, schema `99` is rejected,
failed transactions roll back all catalog changes, and the context manager
closes its connection.

- [ ] **Step 3: Run the tests and confirm `catalog.py` is missing**

```bash
.venv/bin/python -m pytest tests/test_catalog.py -q
```

- [ ] **Step 4: Implement explicit schema creation**

Use named columns, primary/unique constraints, `VARCHAR[]`, `TIMESTAMPTZ`, and
`DATE`. Create a date index over
`(publication_date_new_york, published_at_utc, article_id)`.

- [ ] **Step 5: Implement run lifecycle and transactional access methods**

Keep SQL inside `catalog.py`; callers pass typed model values rather than raw
SQL fragments.

- [ ] **Step 6: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_catalog.py -q
.venv/bin/ruff check src/wsj_pipeline/catalog.py tests/test_catalog.py
```

- [ ] **Step 7: Remove the legacy state module/tests and commit**

```bash
git rm src/wsj_pipeline/state.py tests/test_state.py
git add src/wsj_pipeline/catalog.py tests/test_catalog.py
git commit -m "feat: add unified WSJ catalog"
```

---

### Task 5: Implement deterministic winner selection and Markdown paths

**Files:**
- Create: `src/wsj_pipeline/pipeline.py`
- Create: `tests/test_pipeline.py`
- Remove after migration: `src/wsj_pipeline/canonical.py`
- Remove after migration: `tests/test_canonical.py`

**Interfaces:**
- `article_markdown_path(article_id: str, publication_date: date) -> Path`
- `candidate_rank(article: ExtractedArticle) -> tuple[object, ...]`
- `PipelineSummary` with discovered/new/changed/metadata-only/unchanged/stale/
  reprocessed/succeeded/failed/removed/article counts and validation status.

- [ ] **Step 1: Write failing deterministic path tests**

```python
path = article_markdown_path("wsj:ABC", date(2024, 1, 15))
assert path.parent.as_posix() == "text/2024/01/15"
assert path.name == f"{sha256(b'wsj:ABC').hexdigest()}.md"
```

- [ ] **Step 2: Write failing rank and duplicate tests**

Cover nonempty content, character count, block count, completeness, update time,
and lexical source path in exactly the approved order. Verify one `articles`
row/file, candidate rows for both sources, and a content-free duplicate row.

- [ ] **Step 3: Run focused tests and observe failures**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
```

- [ ] **Step 4: Implement path and ranking helpers plus winner recomputation**

Re-extract raw alternatives only when an affected identity's stored winner is
removed or invalid and no current extracted object is available. Never retain
nonwinner Markdown in DuckDB.

- [ ] **Step 5: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
.venv/bin/ruff check src/wsj_pipeline/pipeline.py tests/test_pipeline.py
```

- [ ] **Step 6: Remove legacy canonical code and commit**

```bash
git rm src/wsj_pipeline/canonical.py tests/test_canonical.py
git add src/wsj_pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: select canonical Markdown articles"
```

---

### Task 6: Publish Markdown atomically and process incrementally

**Files:**
- Modify: `src/wsj_pipeline/pipeline.py`
- Extend: `tests/test_pipeline.py`
- Create: `tests/test_recovery.py`
- Remove after migration: `src/wsj_pipeline/publish.py`
- Remove after migration: `tests/test_publish.py`
- Remove after migration: `tests/test_publish_atomicity.py`
- Remove after migration: `tests/test_incremental.py`

**Interfaces:**
- `pipeline_lock(lock_path: Path) -> Iterator[None]`
- `write_markdown_atomic(markdown: str, target: Path, staging_root: Path,
  run_id: str) -> str` returning SHA-256.
- `ensure_no_legacy_output(config: PipelineConfig) -> None`
- `run_pipeline(config: PipelineConfig, *, limit: int | None, full: bool,
  reprocess: bool = False) -> PipelineSummary`

- [ ] **Step 1: Write failing atomic-publication tests**

Verify staged bytes are reopened and hashed, `os.replace` preserves an existing
target when replacement fails, staging is cleaned, and only output-root paths
are accepted.

- [ ] **Step 2: Write failing incremental classification tests**

Cover initial extraction, unchanged rerun, mtime-only metadata update, HTML
change, header-image-only change, missing image, extraction failure isolation,
and stale extractor behavior with and without `--reprocess`.

- [ ] **Step 3: Write a failing file-before-catalog interruption test**

Monkeypatch the catalog commit to fail after atomic replacement. Assert the run
is failed, the source manifest was not advanced, and an unchanged retry repairs
the catalog and stored hash.

- [ ] **Step 4: Write a failing legacy-output refusal test**

Create each legacy marker in turn—`state/pipeline.duckdb`, `parquet/`, and
`index/wsj.duckdb`—without a new `catalog.duckdb`. Assert `run_pipeline` stops
with a concise instruction to move the derived output or choose a fresh output
root, and assert no legacy file or directory is removed.

- [ ] **Step 5: Implement the exclusive lock and bounded batch loop**

Each batch uses one DuckDB transaction for catalog changes. Article Markdown is
published before the transaction commits; manifest advancement is part of that
same commit.

- [ ] **Step 6: Implement legacy detection, changed-date moves, and post-commit
  orphan cleanup**

Before creating the new catalog, refuse legacy markers without deleting them.
For changed dates, write the new path before switching the catalog. Delete only
the exact old generated file after commit. If cleanup fails, retain a harmless
orphan for validation to report.

- [ ] **Step 7: Run pipeline/recovery tests and lint**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py tests/test_recovery.py -q
.venv/bin/ruff check src/wsj_pipeline/pipeline.py tests/test_pipeline.py tests/test_recovery.py
```

- [ ] **Step 8: Remove superseded legacy modules/tests and commit**

```bash
git rm src/wsj_pipeline/publish.py tests/test_publish.py tests/test_publish_atomicity.py tests/test_incremental.py
git add src/wsj_pipeline/pipeline.py tests/test_pipeline.py tests/test_recovery.py
git commit -m "feat: publish Markdown incrementally"
```

---

### Task 7: Reconcile full-run removals and promote duplicates

**Files:**
- Modify: `src/wsj_pipeline/catalog.py`
- Modify: `src/wsj_pipeline/pipeline.py`
- Extend: `tests/test_pipeline.py`
- Extend: `tests/test_recovery.py`

**Interfaces:**
- `Catalog.reconcile_missing(run_id: str, *, batch_size: int) -> int` reconciles
  one bounded source page; affected identities are then read in lexical pages
  with `Catalog.affected_missing_article_ids(...)` after all missing candidates
  have been removed.
- `recompute_article(article_id: str, ...)` promotes the best remaining source
  or removes the research record when none remain.

- [ ] **Step 1: Write failing full-removal tests**

Start with two unique articles, remove one raw HTML/header image, run `--full`,
and assert its article row and canonical Markdown disappear while the manifest
status becomes `missing`.

- [ ] **Step 2: Write a failing limited-run non-reconciliation test**

Run a one-source limit against a two-source catalog and assert the unseen source
and article remain unchanged.

- [ ] **Step 3: Write a failing duplicate-promotion test**

Remove the winning raw source during a full run and assert the remaining source
is re-extracted, promoted, written to the same identity-derived filename, and
recorded as the new winner.

- [ ] **Step 4: Implement last-seen tracking and full-only reconciliation**

Do not reconcile after an interrupted inventory. Perform it only after source
iteration completes successfully in explicit full mode. Page missing-source
updates, affected-identity recomputation, and cleanup by configured batch size.

- [ ] **Step 5: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py tests/test_recovery.py -q
.venv/bin/ruff check src/wsj_pipeline/catalog.py src/wsj_pipeline/pipeline.py tests/test_pipeline.py tests/test_recovery.py
```

- [ ] **Step 6: Commit**

```bash
git add src/wsj_pipeline/catalog.py src/wsj_pipeline/pipeline.py tests/test_pipeline.py tests/test_recovery.py
git commit -m "feat: reconcile removed WSJ sources"
```

---

### Task 8: Validate the catalog and generated file tree

**Files:**
- Replace: `src/wsj_pipeline/validate.py`
- Replace: `tests/test_validate.py`

**Interfaces:**
- `ValidationIssue(code: str, message: str)`
- `ValidationReport(issues: tuple[ValidationIssue, ...])`
- `validate_outputs(config: PipelineConfig) -> ValidationReport`

- [ ] **Step 1: Write a clean-output validation test**

Create a two-article fixture catalog through `run_pipeline` and assert
`report.ok` and no issues.

- [ ] **Step 2: Write one failing test per invariant family**

Cover missing Markdown, hash mismatch, missing header image, duplicate
article/path constraint attempts, UTC/New-York mismatch, date/path mismatch,
invalid inline URL, broken duplicate winner, inconsistent manifest winner,
running run, failed/missing winner, and orphan Markdown.

- [ ] **Step 3: Write a bounded-memory filesystem test**

Monkeypatch the file iterator to count yielded paths and assert validation
processes each file independently without building a list of file contents.

- [ ] **Step 4: Run validation tests and confirm failures**

```bash
.venv/bin/python -m pytest tests/test_validate.py -q
```

- [ ] **Step 5: Implement streaming checks with stable issue ordering**

Use DuckDB queries for catalog-wide relational invariants and a generator for
filesystem existence/hash/path checks. Messages contain paths/counts but never
Markdown contents.

- [ ] **Step 6: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/test_validate.py -q
.venv/bin/ruff check src/wsj_pipeline/validate.py tests/test_validate.py
```

- [ ] **Step 7: Commit**

```bash
git add src/wsj_pipeline/validate.py tests/test_validate.py
git commit -m "feat: validate Markdown catalog outputs"
```

---

### Task 9: Reduce the CLI to four safe commands

**Files:**
- Replace: `src/wsj_pipeline/cli.py`
- Modify: `src/wsj_pipeline/__main__.py`
- Replace: `tests/test_cli_safety.py`
- Replace: `tests/test_smoke.py`
- Remove: `src/wsj_pipeline/index.py`
- Remove: `src/wsj_pipeline/service.py`
- Remove: `tests/test_index.py`
- Remove: `tests/test_service.py`

**Interfaces:**
- Commands: `inventory`, `run`, `validate`, `smoke`.
- Global option: `--config PATH` before the subcommand.
- `run` requires `--limit N` xor `--full`; optional `--reprocess`.
- Every command prints one deterministic JSON object.

- [ ] **Step 1: Write failing parser safety tests**

Assert the exact command set, mutual exclusion, positive limits, reprocess scope,
and that removed `process`, `publish`, and `index` commands are rejected.

- [ ] **Step 2: Write failing generated-fixture smoke tests**

Assert two canonical articles from three sources, one duplicate, two dated
Markdown files, inline URLs in the catalog, no failures, and clean validation.
Patch default config loading to prove `smoke` never opens the configured archive.

- [ ] **Step 3: Run CLI/smoke tests and confirm failures**

```bash
.venv/bin/python -m pytest tests/test_cli_safety.py tests/test_smoke.py -q
```

- [ ] **Step 4: Implement the reduced parser and command dispatch**

Move fixture-smoke orchestration into `pipeline.py`; keep CLI presentation thin.
Return exit code `1` for failed run/validation summaries.

- [ ] **Step 5: Remove legacy service/index modules and tests**

```bash
git rm src/wsj_pipeline/index.py src/wsj_pipeline/service.py tests/test_index.py tests/test_service.py
```

- [ ] **Step 6: Run all fixture tests and lint**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

- [ ] **Step 7: Commit**

```bash
git add src/wsj_pipeline/cli.py src/wsj_pipeline/__main__.py src/wsj_pipeline/pipeline.py tests/test_cli_safety.py tests/test_smoke.py
git commit -m "refactor: expose the simplified cleaning CLI"
```

---

### Task 10: Remove Parquet dependencies and verify packaging

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/wsj_pipeline/__init__.py` if exports reference removed modules.

**Interfaces:**
- Runtime dependencies: `beautifulsoup4>=4.12,<5`, `duckdb>=1.2,<2`,
  `lxml>=5,<7`.
- Development dependencies remain pytest and Ruff.

- [ ] **Step 1: Add an import/package smoke assertion**

Add to `tests/test_smoke.py`:

```python
def test_legacy_parquet_modules_are_absent() -> None:
    assert importlib.util.find_spec("wsj_pipeline.publish") is None
    assert importlib.util.find_spec("wsj_pipeline.index") is None
```

- [ ] **Step 2: Remove PyArrow and pytz from `pyproject.toml`**

The implementation uses standard-library `zoneinfo`; confirm no remaining source
or test import references either dependency.

- [ ] **Step 3: Run dependency/reference audits**

```bash
! rg -n 'pyarrow|pytz|ARTICLE_SCHEMA|publication_index|parquet_root' src tests pyproject.toml
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

- [ ] **Step 4: Build a wheel in a temporary directory**

```bash
wheel_dir=$(mktemp -d)
.venv/bin/python -m pip wheel --no-deps . -w "$wheel_dir"
find "$wheel_dir" -maxdepth 1 -name 'wsj_preprocessing-*.whl' -print
```

Expected: exactly one wheel path and exit code zero.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/wsj_pipeline/__init__.py tests/test_smoke.py
git commit -m "build: remove legacy Parquet dependencies"
```

---

### Task 11: Rewrite the two user-facing documents and agent guidance

**Files:**
- Replace: `README.md`
- Create: `docs/architecture-crash-course.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/plans/2026-07-29-wsj-preprocessing-index.md`

**Interfaces:**
- README is the concise operator entry point.
- The crash course is the only deep user-facing guide.
- AGENTS contains safety/invariant/change workflow instructions, not duplicated
  tutorial prose.
- The old plan starts with a historical/superseded banner linking to the new
  design and crash course.

- [ ] **Step 1: Rewrite README around the smoke-to-full workflow**

Include purpose and non-goals, install, exact four-command CLI, safe smoke,
bounded real sample, full-run warning, generated layout, one date-range SQL
query, incremental/reprocess behavior, recovery, and the crash-course link.

- [ ] **Step 2: Write the comprehensive architecture crash course**

Include research mental model, end-to-end diagram, one article journey, module
map, Markdown contract, catalog schema, identity/ranking, time semantics,
incremental state transitions, atomicity/recovery cases, validation, test map,
safe extension recipes, downstream embedding/graph boundaries, code-reading
order, and glossary.

- [ ] **Step 3: Update AGENTS.md to the simplified invariants**

Remove Parquet/index instructions. Add rules for complete editorial extraction,
remote URL preservation without download, deterministic Markdown, one catalog,
full-only reconciliation, extractor/schema version changes, fixture-only tests,
and exactly two user-facing docs.

- [ ] **Step 4: Mark the old implementation plan as superseded history**

Add a top banner stating that it describes the removed Parquet prototype and
link to the 2026-08-01 plan, current design spec, README, and crash course. Do
not rewrite its completed historical steps.

- [ ] **Step 5: Verify every documented command against live CLI help**

```bash
.venv/bin/wsj-pipeline --help
.venv/bin/wsj-pipeline inventory --help
.venv/bin/wsj-pipeline run --help
.venv/bin/wsj-pipeline validate --help
.venv/bin/wsj-pipeline smoke --help
```

- [ ] **Step 6: Audit internal Markdown links and stale architecture terms**

Use a small read-only Python check to resolve every relative `.md` link. Then
run:

```bash
! rg -n 'publish --|index --|process --|articles\.parquet|publication_index' README.md AGENTS.md docs/architecture-crash-course.md
```

- [ ] **Step 7: Commit**

```bash
git add README.md AGENTS.md docs/architecture-crash-course.md docs/superpowers/plans/2026-07-29-wsj-preprocessing-index.md
git commit -m "docs: teach the Markdown catalog architecture"
```

---

### Task 12: Complete smoke-scale verification and review

**Files:**
- Modify only files required by issues found during verification.

**Interfaces:**
- No new interfaces. This task proves the approved design and documentation are
  implemented together.

- [ ] **Step 1: Run the complete fixture suite and lint from a clean command**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 2: Build the package from current source**

```bash
wheel_dir=$(mktemp -d)
.venv/bin/python -m pip wheel --no-deps . -w "$wheel_dir"
```

Expected: wheel build succeeds without restoring removed dependencies.

- [ ] **Step 3: Run only the generated end-to-end smoke command**

```bash
.venv/bin/wsj-pipeline smoke
```

Expected JSON: validation is true, two canonical articles exist, and the
duplicate count is one. Date-partitioned Markdown paths are intentionally
test-only evidence asserted inside the temporary smoke fixture before cleanup;
the content-free CLI summary exposes counts rather than ephemeral paths.

- [ ] **Step 4: Audit Git and the real archive boundary**

```bash
git diff --check
git status --short
if git ls-files | rg '^(data|artifacts|outputs)/'; then exit 1; fi
```

Expected: no whitespace errors, only intentional source/docs changes before the
final commit, and no tracked data artifacts. Do not inventory or process the
real archive for this check.

- [ ] **Step 5: Perform a requirement-by-requirement design audit**

Check the approved specification sections against current source, tests, CLI
help, README, crash course, and generated smoke output. Record and fix any
missing or contradictory requirement before claiming completion.

- [ ] **Step 6: Request independent code and documentation review**

Ask the reviewer to focus on raw-data safety, complete editorial extraction,
remote-image URL handling, catalog constraints, duplicate promotion,
interruption windows, full-only reconciliation, bounded memory, stale legacy
references, and command accuracy. Resolve all Critical and Important findings.

- [ ] **Step 7: Commit verification fixes if needed**

```bash
git add README.md AGENTS.md docs pyproject.toml src tests
git commit -m "fix: resolve Markdown catalog review findings"
```

- [ ] **Step 8: Invoke `superpowers:finishing-a-development-branch`**

Run fresh verification once more, then present merge, push/PR, or keep-branch
options without touching the 100 GB archive.
