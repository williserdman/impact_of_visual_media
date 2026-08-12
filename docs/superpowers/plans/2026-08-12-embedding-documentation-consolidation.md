# Embedding Documentation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the completed Jina v4 embedding pipeline usable by a first-time operator and efficiently referenceable by a maintainer through the repository's two user-facing documents.

**Architecture:** Use progressive disclosure: `README.md` owns copyable operation, safety, recovery, and query guidance, while `docs/architecture-crash-course.md` owns implementation flow, invariants, state, provenance, and maintenance guidance. Consolidate existing material rather than adding a third tutorial or changing runtime behavior.

**Tech Stack:** GitHub-flavored Markdown, argparse CLI help, generated fixture-safe smoke commands, DuckDB SQL examples.

## Global Constraints

- Modify current user-facing guidance only in `README.md` and `docs/architecture-crash-course.md`.
- Preserve `docs/superpowers/` as design and execution history rather than presenting it as operator documentation.
- Use generated examples and content-free placeholders only; never include licensed article text, image bytes, credentials, or claimed live observations.
- Describe `article_text` and `header_image` as real hosted Jina v4 source embeddings and `multimodal_article` as a deterministic local derivation.
- State embedding catalog schema version 16 and fail-closed refusal of versions 1 through 15 without migration.
- Require a generated-only pilot and operator-reviewed `--observed-model` before a production run.
- Never imply that a pilot automatically authorizes processing or that an unattended real `--full` run is safe.
- Do not change code, schemas, dependencies, machine configuration, environment-variable names, or generated data.
- If documentation audit reveals a behavioral mismatch, stop this plan and handle it as a separate test-driven correction.

---

### Task 1: Add the first-time embedding workflow to the README

**Files:**
- Modify: `README.md:386-618`

**Interfaces:**
- Consumes: the existing `wsj-embeddings` commands `smoke`, `pilot`, `inventory`, `run`, and `validate`.
- Produces: a copyable operator path from generated smoke through one bounded hosted run and offline validation.

- [ ] **Step 1: Capture the public command surface before writing examples**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' pilot --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' inventory --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' run --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' validate --help
```

Expected: root help lists exactly `smoke`, `pilot`, `inventory`, `validate`, and `run`; run help requires the three roots and exactly one of `--limit` or `--full`, and exposes `--reprocess`, `--authorize-hosted-processing`, and `--observed-model`.

- [ ] **Step 2: Lead the embedding section with the three outputs and their origin**

In `README.md`, make the first embedding paragraphs say explicitly:

- `article_text` is a hosted Jina v4 embedding of canonical Markdown;
- `header_image` is a hosted Jina v4 embedding of the optional local canonical header image;
- `multimodal_article` is the locally computed normalized midpoint of matching text and image generations, not a joint hosted API response;
- all vectors are 2,048-dimensional normalized float32 research observations.

Expected: a reader understands what is sent to Jina and what is computed locally before seeing schema details.

- [ ] **Step 3: Add the prerequisite and generated-only safety gate**

Write a short ordered checklist covering:

1. confirm the right to submit the licensed input to hosted processing;
2. install the declared project dependencies;
3. set `JINA_API_KEY` without documenting a value;
4. run `wsj-embeddings smoke` without a key or network dependency;
5. run the generated-only `wsj-embeddings pilot` manually;
6. review the returned model and record it for `--observed-model`;
7. keep source, preprocessing output, and embedding output roots pairwise disjoint.

Expected: the text distinguishes fixture smoke, real hosted pilot, and real archive processing.

- [ ] **Step 4: Add one bounded first production run and validation sequence**

Use explicit non-secret example paths and include these command shapes:

```bash
wsj-embeddings inventory \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output

wsj-embeddings run \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output \
  --limit 1 \
  --authorize-hosted-processing \
  --observed-model '<pilot-returned-model>'

wsj-embeddings validate \
  --source-root /path/to/wsj_archive \
  --preprocessing-output-root /path/to/preprocessing-output \
  --embedding-output-root /path/to/embedding-output \
  --configuration-id '<configuration-id>'
```

Explain that validation is offline/read-only, multiple configurations require an explicit ID, and the first real mutation remains deliberately bounded.

- [ ] **Step 5: Review and commit the first-time workflow**

Run:

```bash
rg -n 'article_text|header_image|multimodal_article|observed-model|authorize-hosted-processing|--limit 1|generated-only' README.md
git diff --check
git diff -- README.md
```

Expected: every required concept and safety gate appears, examples match the captured help, and Markdown has no whitespace errors.

Commit:

```bash
git add README.md
git -c user.name='Willis Erdman' -c user.email='willis.erdman@gmail.com' commit -m 'docs: add embedding operator workflow'
```

---

### Task 2: Consolidate the README's maintainer reference

**Files:**
- Modify: `README.md:386-692`

**Interfaces:**
- Consumes: the first-time workflow from Task 1 and the schema-v16 public catalog contract.
- Produces: concise repeat-operation, schema, query, reprocessing, and recovery reference without duplicating architectural explanations.

- [ ] **Step 1: Reorder the existing reference behind the runnable workflow**

Keep the exact schema-v16 table inventory and work-state vocabulary, but place them after the operator path under clearly named reference headings. Retain these current truths:

- thirteen base tables and two canonical views;
- schema versions 1 through 15 are refused without migration;
- configuration identity includes the requested alias and exact observed hosted model;
- current work begins `eligible`, is admitted to `queued`, then becomes `in_progress` before an adapter attempt;
- valid configurations coexist and require explicit selection.

Expected: the long schema table no longer blocks a new operator from reaching the first runnable commands.

- [ ] **Step 2: Replace implementation-heavy repetition with crash-course links**

Shorten detailed paragraphs about batching internals, long-text checkpoints, image-rendition descriptor safety, atomic reconciliation, and streaming validator internals to operator consequences plus relative links to the corresponding crash-course headings.

Retain operationally relevant values in the README: 2,048 dimensions, 8,000-token direct ceiling, three-attempt durable part/image policy where applicable, schema version, required roots, and modality names.

Expected: each implementation topic has one detailed owner in the crash course while the README still answers what an operator must do.

- [ ] **Step 3: Consolidate repeat runs, full runs, reprocessing, and recovery**

Make the README state unambiguously:

- unchanged matching work is reused;
- a limited or interrupted run never infers deletion;
- only a successfully completed explicit `--full` run may reconcile disappearance;
- `--reprocess` is deliberate bounded regeneration, not schema migration;
- a real `--full` or `--full --reprocess` requires explicit authorization in the current task;
- an incompatible catalog must be moved aside or replaced with a fresh output root;
- a stale lock may be removed only after confirming no pipeline process is active;
- harmless rendition orphans are reported by validation and safely retried by a later completed full run.

Expected: recovery instructions do not suggest deleting source data, broad directory trees, catalogs, or active locks.

- [ ] **Step 4: Retain practical queries and add navigation for maintainers**

Keep one active-vector query that selects an explicit `configuration_id`, modality, publication time, vector hash, and vector. Keep or add compact queries for configuration IDs, run summaries, and coverage states only when their selected columns exist in schema v16.

Add links near the reference entry points to the crash-course schema, recovery, validation, downstream-boundary, and glossary sections.

Expected: repeat operators can find commands and queries without reading the implementation narrative.

- [ ] **Step 5: Audit and commit the README consolidation**

Run:

```bash
rg -n 'schema(-| )version(-| )?1[0-5]|schema v1[0-5]|versions 1 through 14|joint.*API|automatically approve|unattended' README.md
rg -n '^#{2,3} ' README.md
git diff --check
git diff -- README.md
```

Expected: no stale current schema claim remains; any mention of unattended processing is a prohibition; headings form a clear operator-to-reference progression.

Commit:

```bash
git add README.md
git -c user.name='Willis Erdman' -c user.email='willis.erdman@gmail.com' commit -m 'docs: consolidate embedding operator reference'
```

---

### Task 3: Reframe the crash course around one article's embedding journey

**Files:**
- Modify: `docs/architecture-crash-course.md:57-165`
- Modify: `docs/architecture-crash-course.md:268-471`
- Modify: `docs/architecture-crash-course.md:508-733`
- Modify: `docs/architecture-crash-course.md:791-903`

**Interfaces:**
- Consumes: the operator contract in the consolidated README and the existing module/schema/state/recovery/validation descriptions.
- Produces: one maintainer-oriented explanation of how the embedding pipeline works and where to change it safely.

- [ ] **Step 1: Extend “One article's journey” through all three modalities**

Describe the ordered path from preprocessing `articles` row and canonical Markdown through:

1. immutable configuration registration;
2. durable `eligible` and `queued` admission;
3. exact Markdown and optional local-header input hashing;
4. direct or part-based hosted embeddings;
5. deterministic image pass-through or rendition;
6. source-vector publication with response provenance;
7. local multimodal midpoint publication;
8. offline validation and downstream query.

Expected: the journey distinguishes provider work from local derivation and names the durable interruption boundaries.

- [ ] **Step 2: Tighten the embedding module map and schema narrative**

For each embedding module, state one responsibility and its principal boundary. Keep schema-v16 configuration identity, active canonical views, append-only provenance, long-text generation tables, and reconciliation actions in the schema section. Explain physical storage versus canonical public views once.

Expected: maintainers can identify the coordinator, adapter, catalog, rendition, tokenizer/chunking, batching, pilot, and validator modules without reading source first.

- [ ] **Step 3: Present one lifecycle and recovery model**

Add or update an embedding lifecycle diagram or compact transition block for:

```text
eligible -> queued -> in_progress -> succeeded
                              |-> retryable
                              |-> terminal
                              |-> interrupted
```

Explain `not_applicable`, `stale_input`, and reserved `stale_configuration` separately because they are reconciliation/configuration dispositions rather than adapter attempt outcomes. Connect each failure boundary to its durable result and replay behavior.

Expected: state terminology matches schema v16 and does not imply terminal work is automatically repurchased.

- [ ] **Step 4: Consolidate deep explanations under their architectural owners**

Ensure the crash course is the sole detailed owner for:

- long-text tokenization, part checkpoints, aggregation, and immutable provenance;
- image limits, exact-byte pass-through, deterministic rendition, and safe publication;
- batch packing, indexed outcomes, retries, rate observations, and fatal telemetry;
- limited versus full discovery, invisible staged reconciliation, atomic visibility, and bounded compaction;
- streaming cross-catalog/artifact validation and content-free issue reporting.

Replace repeated command sequences with a link to the README operator workflow.

Expected: architectural mechanics remain complete while operational commands have one authoritative home.

- [ ] **Step 5: Refresh extension recipes, downstream boundaries, and glossary**

Make extension recipes identify which exact surfaces move together for a modality, schema, tokenizer, image policy, batching policy, or reconciliation change. Ensure the glossary defines at least configuration, generation, modality, observed model, provenance, rendition, work item, and reconciliation action.

Expected: a maintainer knows when schema/version/refusal tests and documentation queries must change together.

- [ ] **Step 6: Audit and commit the crash-course consolidation**

Run:

```bash
rg -n 'schema(-| )version(-| )?1[0-5]|schema v1[0-5]|versions 1 through 14|joint.*API|observed model|eligible|stale_configuration' docs/architecture-crash-course.md
rg -n '^#{2,3} ' docs/architecture-crash-course.md
git diff --check
git diff -- docs/architecture-crash-course.md
```

Expected: current schema claims are version 16, the hosted/local boundary is explicit, and the lifecycle vocabulary is complete.

Commit:

```bash
git add docs/architecture-crash-course.md
git -c user.name='Willis Erdman' -c user.email='willis.erdman@gmail.com' commit -m 'docs: clarify embedding architecture'
```

---

### Task 4: Verify the complete documentation handoff

**Files:**
- Verify: `README.md`
- Verify: `docs/architecture-crash-course.md`
- Verify: tracked Markdown outside `docs/superpowers/` for stale current-facing claims

**Interfaces:**
- Consumes: the consolidated operator and maintainer documents from Tasks 1 through 3.
- Produces: evidence that commands, links, safety claims, smokes, and repository hygiene agree with the implementation.

- [ ] **Step 1: Compare documented commands with live help**

Run:

```bash
../../.venv/bin/wsj-pipeline --help
../../.venv/bin/wsj-pipeline run --help
../../.venv/bin/wsj-pipeline validate --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' pilot --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' inventory --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' run --help
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' validate --help
```

Expected: every documented option and required argument exists with the stated scope.

- [ ] **Step 2: Run generated fixture-safe smokes**

Run:

```bash
../../.venv/bin/wsj-pipeline smoke
PYTHONPATH=src ../../.venv/bin/python -c 'from wsj_embeddings.cli import main; main()' smoke
```

Expected: both commands exit zero and report `validation_ok` as true; neither uses the licensed archive or hosted API.

- [ ] **Step 3: Resolve relative Markdown links**

Run:

```bash
../../.venv/bin/python - <<'PY'
from pathlib import Path
import re

for document in (Path('README.md'), Path('docs/architecture-crash-course.md')):
    text = document.read_text(encoding='utf-8')
    for target in re.findall(r'\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)', text):
        if '://' in target or target.startswith('mailto:'):
            continue
        resolved = (document.parent / target).resolve()
        assert resolved.exists(), f'{document}: missing {target}'
print('relative Markdown links resolve')
PY
```

Expected: prints `relative Markdown links resolve`.

- [ ] **Step 4: Audit current-facing claims and tracked artifacts**

Run:

```bash
rg -n --glob '*.md' --glob '!docs/superpowers/**' 'schema(-| )version(-| )?1[0-5]|schema v1[0-5]|versions 1 through 14|joint.*API|automatic.*pilot.*approv'
git ls-files | rg '^(data|artifacts|outputs)/' || true
git status --short
git diff --check
```

Expected: the stale-claim search has no current-facing matches, no generated paths are tracked, and the worktree contains only intentional documentation state.

- [ ] **Step 5: Correct any documentation-only verification mismatch**

If Steps 1 through 4 expose a wording, command, or link mismatch, edit only `README.md` or `docs/architecture-crash-course.md`, rerun the failing check, and then rerun Steps 1 through 4.

Expected: no runtime source, dependency, schema, test fixture, generated output, or machine configuration changes.

- [ ] **Step 6: Commit verification corrections when present and record final evidence**

When Step 5 changed documentation, commit it:

```bash
git add README.md docs/architecture-crash-course.md
git -c user.name='Willis Erdman' -c user.email='willis.erdman@gmail.com' commit -m 'docs: finish embedding documentation handoff'
```

Then run:

```bash
git status --short
git log -4 --oneline
```

Expected: the worktree is clean and the recent commits show the operator workflow, reference consolidation, architecture clarification, and any final documentation-only correction.
