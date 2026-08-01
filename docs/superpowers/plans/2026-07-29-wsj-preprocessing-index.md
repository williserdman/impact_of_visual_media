# WSJ Preprocessing and Publication-Date Index Implementation Plan

> [!WARNING]
> **Superseded history.** This completed plan describes the removed Parquet
> prototype and is not current operating or architecture guidance. Use the
> [2026-08-01 implementation plan](2026-08-01-simplified-markdown-catalog.md),
> [README](../../../README.md), and
> [architecture crash course](../../architecture-crash-course.md) instead.
> The [approved architecture record](../specs/2026-07-29-wsj-preprocessing-index-design.md)
> explains the design, but its `inventory [--limit N]` synopsis predates the
> implemented no-option `inventory` command; use live help and the two current
> user documents for CLI syntax.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, incremental Python pipeline that cleans WSJ HTML/image pairs into partitioned Parquet and publishes a DuckDB publication-date index.

**Architecture:** A staged CLI discovers raw files, extracts versioned normalized records into an operational DuckDB state database, deterministically selects canonical article records, atomically rewrites only affected monthly Parquet partitions, and rebuilds a small DuckDB query index. Raw data is immutable, all generated data is ignored, and corpus-scale processing requires an explicit `--full` flag.

**Tech Stack:** Python 3.11+, argparse, Beautiful Soup 4 with lxml, DuckDB, PyArrow, pytest, Ruff

## Global Constraints

- Treat `data/wsj_archive/` as immutable licensed source data.
- Keep all raw and derived datasets out of Git.
- Store exact UTC timestamps and derive dates using `America/New_York`.
- Partition canonical articles by New York publication year and month.
- Make unchanged reruns idempotent and changed-source publication atomic.
- Preserve deterministic ordering, identity, canonicalization, and audit reasons.
- Never run an unbounded archive-processing command during implementation or verification.
- Require `--full` for an unbounded `process` or `run`; otherwise require a positive `--limit`.
- Use only generated fixtures and explicitly bounded smoke samples in automated tests.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, CLI entry point, and test/lint configuration.
- `config/wsj_pipeline.toml`: tracked default source/output paths and batch settings.
- `src/wsj_pipeline/config.py`: validated configuration loading and path resolution.
- `src/wsj_pipeline/models.py`: shared immutable records and schema constants.
- `src/wsj_pipeline/discovery.py`: stable HTML discovery and companion-image pairing.
- `src/wsj_pipeline/extract.py`: versioned WSJ metadata, body, image, and date extraction.
- `src/wsj_pipeline/state.py`: DuckDB operational schema, manifests, runs, failures, and extracted payloads.
- `src/wsj_pipeline/canonical.py`: article identity and deterministic winner ranking.
- `src/wsj_pipeline/publish.py`: explicit Arrow schema and atomic affected-partition/audit publication.
- `src/wsj_pipeline/index.py`: transactional user-facing DuckDB index creation.
- `src/wsj_pipeline/validate.py`: cross-artifact invariants and structured validation report.
- `src/wsj_pipeline/service.py`: incremental stage orchestration independent of the CLI.
- `src/wsj_pipeline/cli.py`: argparse commands, safety gates, summaries, and exit codes.
- `src/wsj_pipeline/__main__.py`: `python -m wsj_pipeline` entry point.
- `tests/fixtures.py`: generated WSJ-like HTML and tiny image fixtures.
- `tests/test_*.py`: focused unit and integration coverage.
- `README.md`: installation, operations, date semantics, layouts, queries, and recovery.
- `AGENTS.md`: repository-specific safety and architecture guidance for future agents.

---

### Task 1: Package Skeleton, Configuration, and Large-Run Safety

**Files:**
- Create: `pyproject.toml`
- Create: `config/wsj_pipeline.toml`
- Create: `src/wsj_pipeline/__init__.py`
- Create: `src/wsj_pipeline/__main__.py`
- Create: `src/wsj_pipeline/config.py`
- Create: `src/wsj_pipeline/cli.py`
- Create: `tests/test_config.py`
- Create: `tests/test_cli_safety.py`

**Interfaces:**
- Produces: `PipelineConfig.from_toml(path: Path, *, project_root: Path | None = None) -> PipelineConfig`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: `require_processing_scope(limit: int | None, full: bool) -> None`

- [x] **Step 1: Write failing configuration tests**

```python
def test_config_resolves_relative_paths_from_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[paths]\nsource = "data/wsj_archive"\noutput = "data/processed/wsj"\n',
        encoding="utf-8",
    )
    config = PipelineConfig.from_toml(config_path, project_root=tmp_path)
    assert config.source_root == tmp_path / "data/wsj_archive"
    assert config.output_root == tmp_path / "data/processed/wsj"
```

- [x] **Step 2: Run the configuration test and confirm it fails**

Run: `python -m pytest tests/test_config.py -v`

Expected: import failure because `wsj_pipeline.config` does not exist.

- [x] **Step 3: Implement the package metadata and validated configuration**

Define a frozen `PipelineConfig` with `source_root`, `output_root`,
`state_db`, `parquet_root`, `index_db`, `staging_root`, and positive
`batch_size`. Load TOML with `tomllib`; resolve configured relative paths
against an explicit project root; reject identical or nested source/output
paths that could write into the raw archive.

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    source_root: Path
    output_root: Path
    batch_size: int = 250

    @classmethod
    def from_toml(
        cls, path: Path, *, project_root: Path | None = None
    ) -> "PipelineConfig": ...

    @property
    def state_db(self) -> Path:
        return self.output_root / "state" / "pipeline.duckdb"
```

- [x] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_config.py -v`

Expected: all tests pass.

- [x] **Step 5: Write failing CLI safety tests**

```python
@pytest.mark.parametrize("command", ["process", "run"])
def test_processing_requires_limit_or_full(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


def test_limit_and_full_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--limit", "3", "--full"])
    assert exc.value.code == 2
```

- [x] **Step 6: Implement the CLI parser and scope guard**

Add `inventory`, `process`, `publish`, `index`, `validate`, `run`, and `smoke`
subcommands. `process` and `run` must use a required mutually exclusive group
containing `--limit POSITIVE_INT` and `--full`. Until later tasks wire command
handlers, valid commands may print a concise “not implemented” status and
return a nonzero code.

```python
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


scope = command_parser.add_mutually_exclusive_group(required=True)
scope.add_argument("--limit", type=positive_int)
scope.add_argument("--full", action="store_true")
```

- [x] **Step 7: Run task tests and lint**

Run: `python -m pytest tests/test_config.py tests/test_cli_safety.py -v`

Run: `ruff check src tests`

Expected: all tests and lint checks pass.

- [x] **Step 8: Commit**

```bash
git add pyproject.toml config src tests
git commit -m "build: scaffold safe WSJ pipeline CLI"
```

### Task 2: Stable Source Discovery and Image Pairing

**Files:**
- Create: `src/wsj_pipeline/models.py`
- Create: `src/wsj_pipeline/discovery.py`
- Create: `tests/test_discovery.py`

**Interfaces:**
- Produces: `SourceCandidate(relative_html_path: str, html_size: int, html_mtime_ns: int, relative_image_path: str | None, image_size: int | None, image_mtime_ns: int | None, warnings: tuple[str, ...])`
- Produces: `discover_sources(source_root: Path, limit: int | None = None) -> Iterator[SourceCandidate]`
- Consumes: `PipelineConfig.source_root`

- [x] **Step 1: Write failing discovery tests**

Create a temporary archive with direct 2016 files, nested `2024/2024` files,
hidden directories, `__MACOSX`, `backup`, a missing image, and two competing
image extensions. Assert stable relative-path ordering, correct pairing,
missing-image representation, deterministic ambiguity resolution, and exact
limit behavior.

```python
candidates = list(discover_sources(archive))
assert [row.relative_html_path for row in candidates] == sorted(
    row.relative_html_path for row in candidates
)
assert candidates[0].relative_image_path == "2016/story_main_image.jpg"
assert candidates[1].relative_image_path is None
```

- [x] **Step 2: Run discovery tests and confirm they fail**

Run: `python -m pytest tests/test_discovery.py -v`

Expected: import failure for `wsj_pipeline.discovery`.

- [x] **Step 3: Implement focused data models and discovery**

Use `os.walk` with excluded directory names removed in-place, accept
case-insensitive `.html`, match sibling images against the exact
`<html-stem>_main_image` stem and allowed extensions, and yield frozen
`SourceCandidate` instances. Apply `limit` after stable global lexical
ordering so repeated smoke runs select identical sources.

```python
EXCLUDED_DIRECTORIES = frozenset({"__MACOSX", "backup"})
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def discover_sources(
    source_root: Path, limit: int | None = None
) -> Iterator[SourceCandidate]:
    html_paths = sorted(_walk_html(source_root), key=lambda path: path.as_posix())
    for html_path in html_paths[:limit]:
        yield _candidate_for(source_root, html_path)
```

- [x] **Step 4: Run discovery tests and lint**

Run: `python -m pytest tests/test_discovery.py -v`

Run: `ruff check src/wsj_pipeline/models.py src/wsj_pipeline/discovery.py tests/test_discovery.py`

Expected: all checks pass.

- [x] **Step 5: Commit**

```bash
git add src/wsj_pipeline/models.py src/wsj_pipeline/discovery.py tests/test_discovery.py
git commit -m "feat: discover and pair WSJ archive sources"
```

### Task 3: Versioned Article Extraction and Date Semantics

**Files:**
- Create: `src/wsj_pipeline/extract.py`
- Create: `tests/fixtures.py`
- Create: `tests/test_extract.py`
- Create: `tests/test_dates.py`

**Interfaces:**
- Produces: `EXTRACTOR_VERSION: str`
- Produces: `ExtractionError(code: str, message: str)`
- Produces: `extract_article(candidate: SourceCandidate, source_root: Path) -> ExtractedArticle`
- Produces: `parse_wsj_timestamp(value: str | None) -> datetime | None`
- Produces: `derive_publication_dates(published_at_utc: datetime) -> PublicationDates`
- Consumes: `SourceCandidate` from Task 2.

- [x] **Step 1: Generate representative fixture helpers**

Implement `write_article_fixture(...)` to create UTF-8 WSJ-like HTML with
meta tags, JSON-LD, an `<article>` containing `p[data-type="paragraph"]`,
optional excluded inset paragraphs, and an optional companion image made from
constant test bytes. Parameters control IDs, URLs, timestamps, metadata, body,
and image presence.

```python
def write_article_fixture(
    root: Path,
    relative_html_path: str,
    *,
    article_id: str | None = "WP-WSJ-TEST",
    published: str | None = "2024-01-02T15:30:00Z",
    paragraphs: tuple[str, ...] = ("First paragraph.", "Second paragraph."),
    with_image: bool = True,
) -> Path: ...
```

- [x] **Step 2: Write failing metadata/body extraction tests**

Assert extraction of article ID, normalized canonical URL, headlines, summary,
authors, keywords, section/type, ordered paragraphs, joined body text, source
SHA-256, JSON-LD caption/credit, relative image path, and warnings. Assert
author-bio, related-card, advertisement, caption, and dynamic-inset paragraphs
are excluded.

- [x] **Step 3: Run extraction tests and confirm they fail**

Run: `python -m pytest tests/test_extract.py -v`

Expected: import failure for `wsj_pipeline.extract`.

- [x] **Step 4: Implement HTML extraction**

Parse HTML bytes once with Beautiful Soup’s `lxml` parser while updating the
SHA-256. Prefer WSJ meta fields, use JSON-LD fallbacks, restrict body extraction
to the primary article container, and normalize Unicode with NFC plus collapsed
intra-paragraph whitespace. Serialize selected raw metadata with sorted JSON
keys. Do not include body excerpts in errors or logs.

```python
def extract_article(
    candidate: SourceCandidate, source_root: Path
) -> ExtractedArticle:
    html_bytes = (source_root / candidate.relative_html_path).read_bytes()
    html_sha256 = hashlib.sha256(html_bytes).hexdigest()
    soup = BeautifulSoup(html_bytes, "lxml")
    metadata = _extract_meta(soup)
    json_ld = _extract_json_ld(soup)
    paragraphs = _extract_body_paragraphs(soup)
    return _build_article(candidate, html_sha256, metadata, json_ld, paragraphs)
```

- [x] **Step 5: Run extraction tests**

Run: `python -m pytest tests/test_extract.py -v`

Expected: all extraction tests pass.

- [x] **Step 6: Write failing timestamp and timezone tests**

Cover `Z`, explicit offsets, invalid/missing timestamps, UTC/New York date
differences, 2024 spring-forward, and 2024 fall-back. Assert the authoritative
stored timestamp is UTC-aware and publication partitions use the New York
calendar date.

```python
dates = derive_publication_dates(
    datetime(2024, 1, 1, 2, 30, tzinfo=timezone.utc)
)
assert dates.publication_date_utc.isoformat() == "2024-01-01"
assert dates.publication_date_new_york.isoformat() == "2023-12-31"
```

- [x] **Step 7: Implement date parsing and derivation**

Use `datetime.fromisoformat` after translating terminal `Z` to `+00:00`,
normalize to `timezone.utc`, and convert with
`ZoneInfo("America/New_York")`. Missing or invalid publication timestamps
raise stable `ExtractionError` codes when creating the final extracted record;
other optional timestamps remain `None`.

```python
NEW_YORK = ZoneInfo("America/New_York")


def parse_wsj_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ExtractionError("naive_timestamp", "timestamp has no timezone")
    return parsed.astimezone(timezone.utc)
```

- [x] **Step 8: Run extraction/date tests and lint**

Run: `python -m pytest tests/test_extract.py tests/test_dates.py -v`

Run: `ruff check src/wsj_pipeline/extract.py tests/fixtures.py tests/test_extract.py tests/test_dates.py`

Expected: all checks pass.

- [x] **Step 9: Commit**

```bash
git add src/wsj_pipeline/extract.py tests/fixtures.py tests/test_extract.py tests/test_dates.py
git commit -m "feat: extract normalized WSJ article records"
```

### Task 4: Operational State and Incremental Processing

**Files:**
- Create: `src/wsj_pipeline/state.py`
- Create: `tests/test_state.py`
- Create: `tests/test_incremental.py`

**Interfaces:**
- Produces: `PipelineState.open(path: Path) -> PipelineState`
- Produces: `PipelineState.begin_run(command: str, scope: str) -> str`
- Produces: `PipelineState.classify(candidate: SourceCandidate, extractor_version: str) -> CandidateStatus`
- Produces: `PipelineState.store_success(run_id: str, candidate: SourceCandidate, article: ExtractedArticle) -> None`
- Produces: `PipelineState.store_failure(run_id: str, candidate: SourceCandidate, error: ExtractionError) -> None`
- Produces: `process_candidates(config: PipelineConfig, candidates: Iterable[SourceCandidate], run_id: str) -> ProcessSummary`

- [x] **Step 1: Write failing state-schema and run-lifecycle tests**

Assert first-open schema creation, UUID run IDs, timestamps, successful and
failed run completion, context-manager cleanup, and rejection of incompatible
schema versions.

- [x] **Step 2: Run state tests and confirm they fail**

Run: `python -m pytest tests/test_state.py -v`

Expected: import failure for `wsj_pipeline.state`.

- [x] **Step 3: Implement the DuckDB state schema**

Create `metadata`, `runs`, `source_manifest`, `extracted_sources`, `failures`,
`canonical_sources`, and `duplicates` tables. Store normalized article payloads
as deterministic JSON and duplicated ranking columns as typed SQL columns.
Use explicit transactions and parameterized statements.

```python
class PipelineState:
    @classmethod
    def open(cls, path: Path) -> "PipelineState":
        connection = duckdb.connect(str(path))
        state = cls(connection)
        state._ensure_schema()
        return state

    def store_success(
        self,
        run_id: str,
        candidate: SourceCandidate,
        article: ExtractedArticle,
    ) -> None: ...
```

- [x] **Step 4: Run state tests**

Run: `python -m pytest tests/test_state.py -v`

Expected: all state tests pass.

- [x] **Step 5: Write failing incremental-processing tests**

Test that a first run extracts all fixture candidates, an unchanged second run
extracts zero, an mtime-only change with identical HTML hash updates manifest
metadata without replacing the logical payload, a content change extracts one,
and one malformed source records a failure without preventing later candidates.

- [x] **Step 6: Implement classification and bounded batch processing**

Classify by relative path, stat tuple, and extractor version. For stat changes,
extract and compare the content hash inside one transaction. Commit after the
configured batch size. Return counts for discovered, new, changed, unchanged,
succeeded, and failed records. Keep error messages concise and content-free.

```python
def process_candidates(
    config: PipelineConfig,
    candidates: Iterable[SourceCandidate],
    run_id: str,
) -> ProcessSummary:
    for batch in batched(candidates, config.batch_size):
        with PipelineState.open(config.state_db) as state:
            for candidate in batch:
                _process_one(state, config.source_root, run_id, candidate)
    return _summarize_processing(config.state_db, run_id)
```

- [x] **Step 7: Run state/incremental tests and lint**

Run: `python -m pytest tests/test_state.py tests/test_incremental.py -v`

Run: `ruff check src/wsj_pipeline/state.py tests/test_state.py tests/test_incremental.py`

Expected: all checks pass.

- [x] **Step 8: Commit**

```bash
git add src/wsj_pipeline/state.py tests/test_state.py tests/test_incremental.py
git commit -m "feat: persist incremental WSJ processing state"
```

### Task 5: Deterministic Article Identity and Duplicate Audits

**Files:**
- Create: `src/wsj_pipeline/canonical.py`
- Create: `tests/test_canonical.py`

**Interfaces:**
- Produces: `derive_article_key(wsj_id: str | None, canonical_url: str | None, html_sha256: str) -> str`
- Produces: `rank_candidate(article: ExtractedArticle) -> tuple[object, ...]`
- Produces: `recompute_canonical(state: PipelineState, article_keys: Collection[str]) -> CanonicalSummary`
- Consumes: typed ranking and payload columns in `PipelineState`.

- [x] **Step 1: Write failing identity tests**

Assert prefixed WSJ IDs are stable, URL normalization removes fragments and
normalizes scheme/host/trailing slash, and HTML hash is the final fallback.

- [x] **Step 2: Write failing deterministic ranking tests**

Construct duplicates varying one criterion at a time and verify the exact
precedence: valid date, nonempty body, word count, metadata completeness,
`updated_at_utc`, then lexical source path. Assert nonwinners record winner,
rank, and a machine-readable reason.

- [x] **Step 3: Run canonicalization tests and confirm they fail**

Run: `python -m pytest tests/test_canonical.py -v`

Expected: import failure for `wsj_pipeline.canonical`.

- [x] **Step 4: Implement identity, ranking, and affected-key recomputation**

Make ranking pure and deterministic. Recompute only supplied article keys in a
transaction, replacing their prior canonical/duplicate rows. Store old and new
partition keys in `CanonicalSummary` so publication can rewrite both when a
winner moves dates.

```python
def rank_candidate(article: ExtractedArticle) -> tuple[object, ...]:
    return (
        article.published_at_utc is not None,
        bool(article.body_text),
        article.body_word_count,
        article.metadata_completeness,
        article.updated_at_utc or datetime.min.replace(tzinfo=timezone.utc),
        _reverse_lexical_key(article.source_path),
    )
```

- [x] **Step 5: Run canonicalization tests and lint**

Run: `python -m pytest tests/test_canonical.py -v`

Run: `ruff check src/wsj_pipeline/canonical.py tests/test_canonical.py`

Expected: all checks pass.

- [x] **Step 6: Commit**

```bash
git add src/wsj_pipeline/canonical.py tests/test_canonical.py
git commit -m "feat: canonicalize duplicate WSJ articles"
```

### Task 6: Explicit Parquet Schema and Atomic Partition Publication

**Files:**
- Create: `src/wsj_pipeline/publish.py`
- Create: `tests/test_publish.py`
- Create: `tests/test_publish_atomicity.py`

**Interfaces:**
- Produces: `ARTICLE_SCHEMA: pyarrow.Schema`
- Produces: `publish_partitions(state: PipelineState, config: PipelineConfig, partitions: Collection[PartitionKey], run_id: str) -> PublishSummary`
- Produces: `publish_audits(state: PipelineState, config: PipelineConfig, run_id: str) -> None`
- Consumes: canonical state and affected `PartitionKey(year: int, month: int)`.

- [x] **Step 1: Write failing schema tests**

Assert exact field names and types, including UTC timestamp timezone, New York
timestamp timezone, date32 fields, list-of-string authors/keywords/paragraphs,
nullable image fields, quality fields, and `schema_version`.

- [x] **Step 2: Write failing partition-publication tests**

Publish fixtures spanning two New York months. Assert Hive-style directory
names, one deterministically sorted `articles.parquet` per month, no duplicate
article keys, correct date/partition agreement, and native list columns.

- [x] **Step 3: Run publication tests and confirm they fail**

Run: `python -m pytest tests/test_publish.py -v`

Expected: import failure for `wsj_pipeline.publish`.

- [x] **Step 4: Implement explicit Arrow conversion and affected partitions**

Decode canonical payload JSON in bounded monthly groups, convert using
`ARTICLE_SCHEMA`, sort by `(published_at_utc, article_key)`, write compressed
Parquet to a run-specific staging directory, reopen and validate the file, then
replace the target file with `os.replace`.

```python
def publish_partitions(
    state: PipelineState,
    config: PipelineConfig,
    partitions: Collection[PartitionKey],
    run_id: str,
) -> PublishSummary:
    for partition in sorted(partitions):
        table = pa.Table.from_pylist(
            state.canonical_payloads(partition), schema=ARTICLE_SCHEMA
        ).sort_by([("published_at_utc", "ascending"), ("article_key", "ascending")])
        _stage_validate_replace(table, config, partition, run_id)
    return PublishSummary(partitions_written=len(partitions))
```

- [x] **Step 5: Implement audit Parquet publication**

Publish deterministic duplicates, failures, missing images, and runs datasets
with explicit schemas. Audit rows contain keys, paths, codes, counts, and
messages but no article-body content.

```python
AUDIT_EXPORTS = {
    "duplicates": DUPLICATE_SCHEMA,
    "failures": FAILURE_SCHEMA,
    "missing_images": MISSING_IMAGE_SCHEMA,
    "runs": RUN_SCHEMA,
}
```

- [x] **Step 6: Write and run atomicity tests**

Monkeypatch the replacement step to fail after staging. Assert the prior
published file retains its checksum and can still be read. Then rerun normally
and assert staging cleanup plus the expected replacement.

Run: `python -m pytest tests/test_publish.py tests/test_publish_atomicity.py -v`

Expected: all tests pass.

- [x] **Step 7: Run lint**

Run: `ruff check src/wsj_pipeline/publish.py tests/test_publish.py tests/test_publish_atomicity.py`

Expected: no findings.

- [x] **Step 8: Commit**

```bash
git add src/wsj_pipeline/publish.py tests/test_publish.py tests/test_publish_atomicity.py
git commit -m "feat: publish atomic partitioned Parquet datasets"
```

### Task 7: DuckDB Date Index and Cross-Artifact Validation

**Files:**
- Create: `src/wsj_pipeline/index.py`
- Create: `src/wsj_pipeline/validate.py`
- Create: `tests/test_index.py`
- Create: `tests/test_validate.py`

**Interfaces:**
- Produces: `build_index(config: PipelineConfig, run_id: str) -> IndexSummary`
- Produces: `validate_outputs(config: PipelineConfig) -> ValidationReport`
- Produces: `ValidationReport.ok: bool` and `ValidationReport.issues: tuple[ValidationIssue, ...]`

- [x] **Step 1: Write failing index tests**

From published fixture Parquet, assert `articles`, `publication_index`,
`daily_publication_counts`, `duplicates`, `failures`, `missing_images`, and
`runs` are queryable. Assert exact/range date queries return ordered keys and
the materialized index contains no body column.

- [x] **Step 2: Run index tests and confirm they fail**

Run: `python -m pytest tests/test_index.py -v`

Expected: import failure for `wsj_pipeline.index`.

- [x] **Step 3: Implement transactional index creation**

Build a run-specific temporary DuckDB file, create Parquet-backed views with
portable paths relative to the index location, materialize
`publication_index`, create date/key indexes supported by DuckDB, validate row
counts, close the connection, and atomically replace `wsj.duckdb`.

```sql
CREATE VIEW articles AS
SELECT * FROM read_parquet(
  '../parquet/articles/publication_year_ny=*/publication_month_ny=*/articles.parquet',
  hive_partitioning = true
);
CREATE TABLE publication_index AS
SELECT article_key, published_at_utc, publication_date_utc,
       publication_date_new_york
FROM articles
ORDER BY publication_date_new_york, published_at_utc, article_key;
```

- [x] **Step 4: Run index tests**

Run: `python -m pytest tests/test_index.py -v`

Expected: all tests pass.

- [x] **Step 5: Write failing validation tests**

Introduce duplicate article keys, mismatched partition dates, missing audit
winner references, and a Parquet/index key mismatch one at a time. Assert each
produces its documented stable issue code and a non-OK report.

- [x] **Step 6: Implement cross-artifact validation**

Check schema version, unique/non-null keys, partition/date consistency, audit
references, canonical Parquet versus index key equality, and UTC/New York date
derivation. Return all discovered issues without body content.

```python
@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues
```

- [x] **Step 7: Run index/validation tests and lint**

Run: `python -m pytest tests/test_index.py tests/test_validate.py -v`

Run: `ruff check src/wsj_pipeline/index.py src/wsj_pipeline/validate.py tests/test_index.py tests/test_validate.py`

Expected: all checks pass.

- [x] **Step 8: Commit**

```bash
git add src/wsj_pipeline/index.py src/wsj_pipeline/validate.py tests/test_index.py tests/test_validate.py
git commit -m "feat: build and validate publication-date index"
```

### Task 8: End-to-End Incremental Service, CLI, and Fixture Smoke Test

**Files:**
- Create: `src/wsj_pipeline/service.py`
- Modify: `src/wsj_pipeline/cli.py`
- Create: `tests/test_service.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces: `run_incremental(config: PipelineConfig, limit: int | None, full: bool) -> RunSummary`
- Produces: concrete CLI handlers for every subcommand.
- Consumes: discovery, extraction/state, canonicalization, publication, index, and validation interfaces from Tasks 2–7.

- [x] **Step 1: Write failing end-to-end service tests**

Run a fixture archive through the service and assert canonical/audit outputs.
Run it unchanged and assert zero extractions and identical article Parquet
checksums. Add a `2025/2025` folder and assert only those sources are processed.
Modify a duplicate winner and assert only its affected monthly partition
changes.

- [x] **Step 2: Run service tests and confirm they fail**

Run: `python -m pytest tests/test_service.py -v`

Expected: import failure for `wsj_pipeline.service`.

- [x] **Step 3: Implement orchestration and locking**

Add a filesystem lock created atomically under the output state directory.
Begin/finish runs in `PipelineState`, process candidates, recompute affected
article keys, publish affected partitions and audits, build the index, validate,
record summaries, and always release the lock. Mark failed runs without hiding
the original exception.

```python
def run_incremental(
    config: PipelineConfig, limit: int | None, full: bool
) -> RunSummary:
    require_processing_scope(limit, full)
    with pipeline_lock(config.output_root / "state" / "pipeline.lock"):
        candidates = discover_sources(config.source_root, limit=limit)
        return _run_stages(config, candidates, reconcile_missing=full)
```

- [x] **Step 4: Wire every CLI command**

Load configuration once, print resolved source/output roots and candidate scope,
emit compact JSON summaries, map validation failures to exit code 1 and usage
errors to exit code 2, and ensure `smoke` always builds/runs generated fixtures
in a temporary directory.

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = HANDLERS[args.command](args)
    print(json.dumps(asdict(summary), sort_keys=True, default=str))
    return 0
```

- [x] **Step 5: Write and run CLI smoke tests**

Invoke `main(["smoke"])` and assert success. Invoke bounded `inventory` and
`run --limit 2` on fixtures. Assert no command defaults to the real archive and
that `run` without a processing scope remains rejected.

Run: `python -m pytest tests/test_service.py tests/test_smoke.py tests/test_cli_safety.py -v`

Expected: all tests pass.

- [x] **Step 6: Run the complete fixture-only suite**

Run: `python -m pytest -q`

Run: `ruff check .`

Expected: all tests pass and Ruff reports no findings. Do not run an unbounded
command against `data/wsj_archive`.

- [x] **Step 7: Commit**

```bash
git add src/wsj_pipeline/service.py src/wsj_pipeline/cli.py tests/test_service.py tests/test_smoke.py
git commit -m "feat: orchestrate incremental WSJ pipeline"
```

### Task 9: Operations Documentation and Repository Agent Guidance

**Files:**
- Create: `README.md`
- Create: `AGENTS.md`
- Modify: `.gitignore`
- Modify: `.gitattributes`
- Modify: `docs/superpowers/specs/2026-07-29-wsj-preprocessing-index-design.md`

**Interfaces:**
- Documents: install, smoke, bounded sample, full run, incremental additions,
  layouts, date semantics, DuckDB queries, recovery, and parser limitations.
- Documents: mandatory safety/invariant rules for future coding agents.

- [x] **Step 1: Write README command examples and operational semantics**

Include commands for editable installation, `wsj-pipeline smoke`, bounded
inventory/processing, deliberately authorized `--full` execution, validation,
and DuckDB exact/range date queries. Explain that a real full run is expected
to be long-running and is not performed during implementation.

- [x] **Step 2: Write project `AGENTS.md`**

State that agents must keep the raw archive immutable, never run `--full`
without explicit current-task authorization, use fixture or small limited
smoke tests, keep all data ignored, preserve identity/date/idempotence
invariants, add regression fixtures for parser changes, update schema/design
docs when behavior changes, and avoid logging licensed article text.

- [x] **Step 3: Harden ignore and attribute rules**

Ensure `/data/`, `/artifacts/`, `/outputs/`, DuckDB, Parquet, Arrow, logs,
caches, and local environments are ignored. Preserve text LF rules and binary
image/data attributes. Do not stage any raw or generated dataset.

- [x] **Step 4: Reconcile design documentation with implementation**

Compare actual commands, schemas, paths, and failure behavior against the
design. Update concrete differences; do not weaken the raw-data, full-run,
date, atomicity, audit, or idempotence requirements.

- [x] **Step 5: Verify tracked-file and data safety**

Run: `git status --short`

Run: `git ls-files | rg '^(data|artifacts|outputs)/'`

Expected: the second command prints nothing; no generated database, Parquet,
archive, image, or HTML file is staged.

- [x] **Step 6: Run final fixture-only verification**

Run: `python -m pytest -q`

Run: `ruff check .`

Run: `wsj-pipeline smoke`

Expected: all commands pass without reading `data/wsj_archive`.

- [x] **Step 7: Commit**

```bash
git add README.md AGENTS.md .gitignore .gitattributes docs/superpowers/specs
git commit -m "docs: add WSJ pipeline operations guidance"
```
