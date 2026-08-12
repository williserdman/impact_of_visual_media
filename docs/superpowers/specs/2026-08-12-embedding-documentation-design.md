# Embedding documentation consolidation design

**Date:** 2026-08-12

## Goal

Make the completed Jina v4 embedding pipeline understandable to both a
first-time operator and an experienced maintainer without creating another
user-facing tutorial. The current operator documentation remains `README.md`;
the implementation and maintenance explanation remains
`docs/architecture-crash-course.md`.

## Audience and layering

The README uses progressive disclosure. Its first embedding section is a
runnable, generated-data-first path for a new operator. Later sections retain a
compact reference for repeat operation, validation, recovery, queries, and
schema compatibility. A maintainer can skip the walkthrough and use those
reference sections directly.

The crash course explains why the workflow behaves as documented. It owns the
data flow, state machine, configuration identity, provenance, failure
boundaries, reconciliation, validation, and safe extension guidance. It does
not repeat installation commands except where a command identifies an
architectural boundary.

## README structure

The embedding material will be consolidated into this order:

1. Explain the three research observations: `article_text`, `header_image`,
   and locally derived `multimodal_article`.
2. State prerequisites and safety gates: licensed-input rights, Jina API key,
   a manual generated-only pilot, an operator-reviewed observed model, disjoint
   roots, and no unattended real full run.
3. Provide a copyable generated smoke and pilot workflow.
4. Provide the first bounded production workflow, including
   `--observed-model`, explicit authorization, `--limit`, and offline
   validation.
5. Explain when `--full` and `--reprocess` are appropriate and why they require
   explicit operator intent.
6. Provide concise recovery guidance for credentials, retries, stale locks,
   schema refusal, interrupted runs, and harmless rendition orphans.
7. Retain practical DuckDB queries and a compact contract/reference section
   for experienced operators.

Detailed column inventories remain available where they are genuinely useful
for queries, but lengthy implementation explanations move to the crash course
or become short links to its corresponding section.

## Crash-course structure

The crash course will describe one article's embedding journey in order:

1. Canonical Markdown and optional local header association enter the
   downstream coordinator.
2. Immutable configuration identity binds the requested alias, pilot-observed
   hosted model, tokenizer, batching, image, and multimodal policies.
3. Work moves through durable `eligible`, `queued`, `in_progress`, and terminal
   or reusable states.
4. Hosted source vectors preserve raw-response and stored-vector provenance.
5. Long text becomes independently checkpointed parts and one weighted
   aggregate; oversized images become verified deterministic renditions.
6. The multimodal vector is derived locally from matching text and image
   generations.
7. Limited runs never infer deletion; completed full runs expose staged
   reconciliation atomically and compact it in bounded pages.
8. Read-only validation streams both catalogs and artifacts with bounded
   memory and reports content-free issues.

The module map, schema description, failure matrix, test map, extension
recipes, and glossary remain maintainer-oriented reference material. Repeated
operator instructions will be shortened and linked back to the README.

## Safety and truthfulness

- Examples use generated fixtures and placeholders only. They contain no
  licensed excerpts, image bytes, secrets, or claimed live measurements.
- Documentation distinguishes the real hosted Jina API from injected or fake
  test adapters.
- The pilot is a manual observation gate, not automatic approval or a source
  of hard-coded provider policy.
- `multimodal_article` is described as a local deterministic derivation, not a
  joint vector returned by the hosted API.
- Current documentation states schema version 16 and refusal of versions 1
  through 15 without migration.
- No documentation change alters runtime behavior, generated state, machine
  configuration, or environment-variable names.

## Verification

After editing the two user-facing documents:

1. Compare every documented command and option with live `--help` output.
2. Run generated preprocessing and embedding smokes.
3. Resolve every relative Markdown link.
4. Search tracked current-facing Markdown for stale schema, modality, model,
   pilot, lifecycle, or full-run claims.
5. Run `git diff --check`, inspect `git status --short`, and confirm no generated
   data or artifacts are tracked.

Behavioral tests are unnecessary unless the audit uncovers a documentation
claim that requires a code correction. Any such correction is outside this
documentation-only design and must return to test-driven implementation.
