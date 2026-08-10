# Synthetic Article-Text Smoke Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and validate one deterministic article text embedding from a generated canonical article through the new downstream coordinator without reading the licensed archive or calling Jina.

**Architecture:** Add a separate `wsj_embeddings` package whose public coordinator reads canonical winners from the preprocessing catalog, delegates text encoding through an injected adapter, normalizes and publishes vectors to a separate versioned DuckDB catalog, and returns a content-free result. A deterministic fake adapter and generated preprocessing fixture power the smoke command; read-only validation checks the published research artifact through its public catalog contract.

**Tech Stack:** Python 3.11+, DuckDB, `dataclasses`, `hashlib`, `argparse`, pytest, Ruff.

## Global Constraints

- Treat the configured archive and preprocessing outputs as read-only; source and embedding output roots remain disjoint.
- Publish only canonical `articles` rows and the complete UTF-8 canonical Markdown bytes they reference.
- Store no article bodies, image bytes, secrets, or licensed excerpts in embedding state, summaries, errors, fixtures, or documentation.
- The first slice uses a deterministic fake adapter only and makes no network call.
- Store one finite, nonzero, L2-normalized, 2,048-dimensional `article_text` vector with source and vector hashes.
- Fail closed on an incompatible or malformed existing embedding catalog; perform no migration or overwrite.
- Preserve the existing `wsj-pipeline` command surface.
- Test behavior through the public coordinator, read-only validator, and CLI seams using generated fixtures.

---

### Task 1: Publish one generated article text embedding

**Files:**
- Create: `src/wsj_embeddings/__init__.py`
- Create: `src/wsj_embeddings/models.py`
- Create: `src/wsj_embeddings/adapters.py`
- Create: `src/wsj_embeddings/config.py`
- Create: `src/wsj_embeddings/catalog.py`
- Create: `src/wsj_embeddings/pipeline.py`
- Test: `tests/test_embeddings_pipeline.py`

**Interfaces:**
- Consumes: the preprocessing DuckDB `articles` research relation and canonical Markdown paths resolved below the preprocessing output root.
- Produces: `EmbeddingPipelineConfig`, `EmbeddingProfile`, `EmbeddingAdapter`, `FakeEmbeddingAdapter`, `EmbeddingRunResult`, and `run_embedding_pipeline(config, adapter, *, limit)`.

- [ ] **Step 1: Write the failing coordinator test**

```python
def test_pipeline_publishes_one_normalized_article_text_embedding(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    before = snapshot_fixture_inputs(config)

    result = run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert result == EmbeddingRunResult(articles=1, embeddings=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        row = db.execute(
            """
            SELECT article_id, modality, published_at_utc,
                   publication_date_new_york, dimensions,
                   input_sha256, stored_vector_sha256, vector
            FROM embeddings
            """
        ).fetchone()
    assert row[:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        date(2024, 1, 2),
        2048,
    )
    assert row[7] == [1.0] + [0.0] * 2047
    assert snapshot_fixture_inputs(config) == before
```

- [ ] **Step 2: Run the focused test and verify red**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_pipeline.py::test_pipeline_publishes_one_normalized_article_text_embedding`

Expected: collection fails because `wsj_embeddings` does not exist.

- [ ] **Step 3: Add immutable public records and the fake adapter**

```python
@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    model: str
    task: str
    dimensions: int
    output_type: str = "float"
    normalization: str = "l2-client-v1"


class EmbeddingAdapter(Protocol):
    @property
    def profile(self) -> EmbeddingProfile: ...
    def embed_text(self, text: str) -> tuple[float, ...]: ...


class FakeEmbeddingAdapter:
    profile = EmbeddingProfile(
        model="fake-jina-embeddings-v4",
        task="retrieval.passage",
        dimensions=2048,
    )

    def embed_text(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("text input must not be empty")
        return (1.0, *(0.0 for _ in range(2047)))
```

- [ ] **Step 4: Add resolved, disjoint pipeline configuration**

```python
@dataclass(frozen=True, slots=True)
class EmbeddingPipelineConfig:
    source_root: Path
    preprocessing_output_root: Path
    embedding_output_root: Path

    @property
    def preprocessing_catalog(self) -> Path:
        return self.preprocessing_output_root / "catalog.duckdb"

    @property
    def embedding_catalog(self) -> Path:
        return self.embedding_output_root / "catalog.duckdb"
```

Resolve all roots in `__post_init__` and reject equality or containment between any two roots.

- [ ] **Step 5: Add the minimal exact embedding schema**

Create one schema version with `metadata`, `embedding_configurations`, `runs`, and `embeddings`. The research row is uniquely keyed by article identity, modality, and configuration identity; its vector type is a fixed 2,048-float array. Configuration identity is the SHA-256 of canonical JSON for every `EmbeddingProfile` field.

- [ ] **Step 6: Implement the minimal coordinator**

```python
def run_embedding_pipeline(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    *,
    limit: int,
) -> EmbeddingRunResult:
    if limit < 1:
        raise ValueError("limit must be positive")
    # Open preprocessing read-only, select canonical rows by article_id,
    # read exact UTF-8 Markdown bytes, hash them, call the adapter,
    # validate/normalize the vector, and commit one research row.
```

Use strict UTF-8 decoding, path containment under the preprocessing output root, `math.isfinite`, `math.sqrt(sum(value * value ...))`, canonical float32 packing for the stored-vector hash, and bounded DuckDB transactions. Error text contains identifiers and stable codes, never Markdown.

- [ ] **Step 7: Run the focused test and verify green**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_pipeline.py`

Expected: all tests in the file pass.

- [ ] **Step 8: Run focused lint**

Run: `/home/willis/projects/finance_wsj/.venv/bin/ruff check src/wsj_embeddings tests/test_embeddings_pipeline.py`

Expected: no findings.

### Task 2: Refuse incompatible output and validate the research artifact

**Files:**
- Modify: `src/wsj_embeddings/catalog.py`
- Create: `src/wsj_embeddings/validate.py`
- Modify: `tests/test_embeddings_pipeline.py`

**Interfaces:**
- Consumes: `EmbeddingPipelineConfig` and the schema/publication contract from Task 1.
- Produces: `EmbeddingValidationIssue`, `EmbeddingValidationResult`, and `validate_embedding_outputs(config)`.

- [ ] **Step 1: Write failing validation and refusal tests**

```python
def test_validator_accepts_fresh_synthetic_embedding_output(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    assert validate_embedding_outputs(config) == EmbeddingValidationResult(issues=())


def test_pipeline_refuses_unknown_existing_embedding_schema(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("CREATE TABLE legacy_vectors(value INTEGER)")
    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
```

- [ ] **Step 2: Run the focused tests and verify red**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_pipeline.py -k 'validator or refuses'`

Expected: failure because the public validator and exact-schema refusal do not exist.

- [ ] **Step 3: Implement exact schema inspection and fail-closed opening**

Validate table names, ordered columns/types/nullability, primary keys, and schema version before a writable connection is returned for an existing database. Never run `CREATE OR REPLACE`, migration SQL, or destructive recovery against an incompatible file.

- [ ] **Step 4: Implement read-only cross-artifact validation**

```python
@dataclass(frozen=True, slots=True)
class EmbeddingValidationIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EmbeddingValidationResult:
    issues: tuple[EmbeddingValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues
```

Check the exact schema first, then configuration references, text input hash against canonical Markdown bytes, modality, dimension, finiteness, nonzero/unit norm tolerance, float32 vector hash, canonical article identity, and publication metadata. Return stable content-free issues in deterministic order.

- [ ] **Step 5: Run focused tests and lint**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_pipeline.py`

Run: `/home/willis/projects/finance_wsj/.venv/bin/ruff check src/wsj_embeddings tests/test_embeddings_pipeline.py`

Expected: both commands pass.

### Task 3: Expose the fixture-safe smoke command

**Files:**
- Create: `src/wsj_embeddings/smoke.py`
- Create: `src/wsj_embeddings/cli.py`
- Create: `src/wsj_embeddings/__main__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_embeddings_cli.py`

**Interfaces:**
- Consumes: the coordinator, fake adapter, and validator from Tasks 1–2.
- Produces: `run_embedding_smoke()`, `build_parser()`, and `main(argv=None)` for the separate `wsj-embeddings` executable.

- [ ] **Step 1: Write the failing CLI smoke test**

```python
def test_embedding_smoke_is_fixture_safe_and_deterministic(capsys):
    exit_code = main(["smoke"])
    line = capsys.readouterr().out
    assert exit_code == 0
    assert json.loads(line) == {
        "articles": 1,
        "embeddings": 1,
        "validation_ok": True,
    }
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
```

- [ ] **Step 2: Run the focused test and verify red**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_cli.py`

Expected: collection fails because the embedding CLI does not exist.

- [ ] **Step 3: Implement the generated smoke workflow**

Create a temporary source root, preprocessing output root, and embedding output root. Build one generated canonical preprocessing row and Markdown artifact using existing catalog interfaces, snapshot its bytes, run the coordinator with `FakeEmbeddingAdapter`, run read-only validation, assert the snapshot is unchanged, and return a content-free result after the temporary directory closes.

- [ ] **Step 4: Implement the separate CLI**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wsj-embeddings")
    parser.add_subparsers(dest="command", required=True).add_parser("smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "smoke":
        raise AssertionError(f"unhandled command {args.command}")
    result = run_embedding_smoke()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.validation_ok else 1
```

Register `wsj-embeddings` without modifying the existing `wsj-pipeline` registration.

- [ ] **Step 5: Run focused tests and both smoke commands**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py`

Run: `PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m wsj_embeddings smoke`

Run: `PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m wsj_pipeline smoke`

Expected: focused tests pass; each smoke command emits one successful deterministic JSON object.

- [ ] **Step 6: Run focused lint**

Run: `/home/willis/projects/finance_wsj/.venv/bin/ruff check src/wsj_embeddings tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py`

Expected: no findings.

### Task 4: Verify, review, and commit the tracer slice

**Files:**
- Review all files created or modified by Tasks 1–3.

**Interfaces:**
- Consumes: the complete ticket-01 diff.
- Produces: one reviewed commit on `feature/jina-v4-embedding-corpus`.

- [ ] **Step 1: Run complete fixture-safe verification**

Run: `/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q`

Run: `/home/willis/projects/finance_wsj/.venv/bin/ruff check .`

Run: `PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m wsj_pipeline smoke`

Run: `PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m wsj_embeddings smoke`

Expected: all tests and lint pass; both smoke workflows report successful validation.

- [ ] **Step 2: Run two-axis review against the branch base**

Review the full working-tree diff against the branch point for repository standards and ticket-01 specification coverage. Fix every high-confidence issue, rerun affected focused checks, and repeat the complete verification after fixes.

- [ ] **Step 3: Confirm repository hygiene**

Run: `git status --short`

Run: `git diff --check`

Run: `git ls-files | rg '^(data|artifacts|outputs)/'`

Expected: only intentional source, test, plan, and package metadata changes are present; no generated or licensed artifacts are tracked.

- [ ] **Step 4: Commit the implementation**

```bash
git add pyproject.toml src/wsj_embeddings tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py docs/superpowers/plans/2026-08-09-synthetic-article-text-smoke.md
git commit -m "feat: add synthetic article embedding smoke path"
```
