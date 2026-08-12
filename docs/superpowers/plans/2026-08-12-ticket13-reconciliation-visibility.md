# Ticket 13 Reconciliation Visibility Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full-run reconciliation atomically visible, independently bounded and resumable, exact per configuration, and memory-bounded during rendition cleanup.

**Architecture:** Schema version 14 introduces private physical embedding/work tables, canonical public views, and exact run-scoped reconciliation actions. Bounded staging writes invisible actions; one run-marker commit exposes the complete overlay; bounded compaction materializes the same logical state while checking expected generation/vector/work identity. Cleanup scans candidates incrementally and queries exact historical reference membership.

**Tech Stack:** Python 3.12, DuckDB, pytest generated fixtures, descriptor-relative POSIX file operations.

## Global Constraints

- Public `embeddings` and `embedding_work_items` views are the only canonical research and operational read interface.
- Private physical tables are mutation targets only.
- A staged action binds expected generation run ID, vector hash, source path, input hash, and work state; visibility and compaction reject mismatched base identity.
- Failed/interrupted staging exposes zero removals; marker visibility is atomic; compaction is bounded, resumable, and semantically invariant.
- Header reconciliation is exact by article, configuration, and modality for `NULL` to new, path A to path B, and new to `NULL`.
- Cleanup never builds a history-sized reference set or calls `fetchall` for rendition references.
- Bump exact schema to version 14 and refuse exact version 13 without mutation.
- Use generated fixtures only. Do not read the licensed archive, call hosted APIs, or run a real/full corpus.

---

### Task 1: Atomic reconciliation action protocol

**Files:**
- Modify: `tests/test_embeddings_pipeline.py`
- Modify: `src/wsj_embeddings/catalog.py`
- Modify: `src/wsj_embeddings/pipeline.py`
- Modify: `src/wsj_embeddings/validate.py`

**Interfaces:**
- Produces: private `embedding_rows` and `embedding_work_rows` tables; canonical `embeddings` and `embedding_work_items` views; `reconciliation_actions`; `EmbeddingCatalog.stage_reconciliation_page`, `commit_reconciliation_visibility`, and `compact_reconciliation_page`.
- Consumes: existing full inventory, generation history archival, run lifecycle, and pipeline failpoints.

- [ ] **Step 1: Write the page-two staging failure fixture**

Create three published generated articles, remove all three canonically, fail the second reconciliation staging page after the first page committed, and assert public `embeddings` and `embedding_work_items` rows exactly equal literals captured before the failed full run. Also assert physical base rows remain present and staged actions are invisible because the run marker is false.

- [ ] **Step 2: Run the staging fixture to verify RED**

Run only the named pytest node. Expected: public vector/work rows for page one disappeared under the version-13 mutating reconciler.

- [ ] **Step 3: Write visibility and compaction fixtures**

Add generated coordinator failpoints immediately after marker commit and during the second compaction page. Assert all public removals appear at the marker boundary while physical rows/actions remain, the public rows do not change across interrupted compaction, and a resumed full run drains committed actions in pages no larger than the configured page size.

- [ ] **Step 4: Run the new nodes to verify RED**

Expected: failpoints/interfaces are absent and version-13 reconciliation mutates page-by-page.

- [ ] **Step 5: Implement schema-14 tables, views, actions, and marker**

Rename physical tables to private names and recreate the public names as views. Store each action's exact article/modality/configuration key, target disposition, expected generation run ID, vector hash, source path, input hash, and work state. Public views overlay only actions whose owning run marker is visible. Stage in lexical keyset pages without active mutation; one transaction validates every staged action still matches its expected base identity and flips the marker.

- [ ] **Step 6: Implement bounded idempotent compaction**

Select one committed action page, verify expected identity, archive the exact superseded generation, remove the exact physical vector, materialize the target work disposition, and delete the action in one page transaction. If the physical state already equals the target disposition, delete the action idempotently. Otherwise reject divergence. Drain prior committed actions before processing a new full run.

- [ ] **Step 7: Teach validation the overlay protocol**

Validate pending visible actions against expected base identity and target work disposition. Reject invisible actions owned by completed successful runs, visible actions owned by non-successful runs, and overlay/base mismatches. Continue all existing research checks through public views.

- [ ] **Step 8: Run the three atomic visibility nodes to verify GREEN**

Expected: staging failure has zero public change; marker makes all changes visible; interrupted/resumed bounded compaction preserves public results.

### Task 2: Exact header association targeting

**Files:**
- Modify: `tests/test_embeddings_pipeline.py`
- Modify: `src/wsj_embeddings/catalog.py`

**Interfaces:**
- Consumes: `stage_reconciliation_page` and run inventory header association.
- Produces: exact per-configuration header and dependent multimodal actions.

- [ ] **Step 1: Write three coexistence fixtures**

For two retained configurations, exercise `NULL` to new, path A to path B, and new to `NULL`. Run the current configuration before reconciliation. Assert its matching current header/multimodal work remains active where eligible, while only the older configuration whose stored header path differs receives stale/not-applicable replacement work.

- [ ] **Step 2: Run the fixtures to verify RED**

Expected: current version-13 selection groups by article and invalidates both configurations.

- [ ] **Step 3: Implement exact decision selection**

Stage a header action only where that configuration's physical header source path is distinct from the discovered association. Stage its dependent multimodal action for the same configuration only. For new-to-null target header state is `not_applicable`; for null-to-new and path changes target is `stale_input`. Preserve matching current rows exactly.

- [ ] **Step 4: Run the coexistence fixtures to verify GREEN**

Expected: all literal public embedding/work assertions pass for each transition.

### Task 3: Incremental rendition cleanup

**Files:**
- Modify: `tests/test_embeddings_pipeline.py`
- Modify: `src/wsj_embeddings/image_rendition.py`
- Modify: `src/wsj_embeddings/catalog.py`
- Modify: `src/wsj_embeddings/pipeline.py`

**Interfaces:**
- Produces: `EmbeddingCatalog.rendition_is_referenced(relative_path: str) -> bool` and `cleanup_orphan_image_renditions(output_descriptor, is_referenced)`.
- Consumes: descriptor-safe rendition candidate scanner and all current/historical `image_input_provenance` rows.

- [ ] **Step 1: Write incremental cleanup fixture**

Create multiple generated rendition candidates including one referenced only by historical provenance. Inject a reference callable that records one candidate at a time and refuses bulk iteration. Assert the historical candidate remains, the orphan is removed, and peak Python candidate state is one.

- [ ] **Step 2: Run the fixture to verify RED**

Expected: current cleanup requires a prebuilt set and pipeline calls `fetchall`.

- [ ] **Step 3: Implement exact per-candidate reference checks**

Make cleanup yield and validate one candidate, call `SELECT 1 FROM image_input_provenance WHERE rendition_relative_path = ? LIMIT 1`, and unlink only when false. Remove the pipeline `fetchall` and reference set.

- [ ] **Step 4: Run the cleanup fixture and existing cleanup node GREEN**

Expected: historical reference is preserved and orphan cleanup remains post-commit/resumable.

### Task 4: Exact schema, refusal, docs, report, verification

**Files:**
- Modify: `tests/test_embeddings_pipeline.py`
- Modify: `README.md`
- Modify: `docs/architecture-crash-course.md`
- Modify: `.superpowers/sdd/all-tickets-plan/task-12-report.md`

**Interfaces:**
- Consumes: final schema-14 table/view/action contract.
- Produces: exact v14 schema test, exact prior-v13 byte-preserving refusal, operator/research docs, correction evidence.

- [ ] **Step 1: Update exact schema test first and observe RED**

Expect the private tables, public views, reconciliation action columns/keys, version 14 metadata, and no indexes. Construct exact version-13 shape and assert a refused run leaves bytes unchanged.

- [ ] **Step 2: Implement exact catalog declarations and refusal GREEN**

Update creation/validation constants and ensure schema before any writable use. Run only exact v14 and prior-v13 nodes.

- [ ] **Step 3: Update public documentation and implementation report**

Document canonical views versus private storage, atomic marker visibility, bounded compaction recovery, exact header targeting, incremental cleanup, schema-14 fresh-root refusal, and focused RED/GREEN evidence.

- [ ] **Step 4: Run focused verification**

Run only affected ticket-13 nodes, changed-file Ruff, CLI help, embedding smoke, `git diff --check`, tracked artifact hygiene, and `git status --short`. Do not run the full suite or any real full/API/archive operation.

- [ ] **Step 5: Commit implementation separately**

Commit with `Willis Erdman <willis.erdman@gmail.com>` and report exact commit, focused results, and remaining concerns.
