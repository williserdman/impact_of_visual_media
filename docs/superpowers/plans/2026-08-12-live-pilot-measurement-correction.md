# Live Pilot Measurement Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manual generated Jina pilot report only bounded live observations produced by the production transport, adapter, and scheduler.

**Architecture:** Add fixed bounded metadata GETs to the hosted adapter, retain only safe allowlisted response facts, and run generated embedding probes through the existing scheduler. A pilot-only observation wrapper measures actual concurrent overlap while preserving the production request and retry path.

**Tech Stack:** Python 3.11, urllib, dataclasses, concurrent futures, pytest, Ruff.

## Global Constraints

- Never call the live network or use a credential in automated verification.
- Never read the licensed archive or create embedding/preprocessing state.
- Keep the pilot selector-free and content-free.
- Bound all success/error response reads and all retained metadata shapes.
- Do not couple pilot results automatically to production `run`.

---

### Task 1: Bounded hosted metadata observations

**Files:**
- Modify: `tests/test_jina_pilot.py`
- Modify: `tests/test_jina_hosted_adapter.py`
- Modify: `src/wsj_embeddings/adapters.py`
- Modify: `src/wsj_embeddings/pilot.py`

**Interfaces:**
- Consumes: injected `JinaTransport` and `JinaEmbeddingAdapter`.
- Produces: fixed-endpoint bounded metadata observations in pilot JSON.

- [ ] Add CLI-level simulated GET success tests for the OpenAPI version and allowlisted v4 catalogue metadata, including returned price and absent currency.
- [ ] Run the focused tests and observe failure because the transport has no GET metadata path and the pilot still hard-codes observations.
- [ ] Add bounded GET support to `UrllibJinaTransport`, fixed endpoint access in `JinaEmbeddingAdapter`, and strict safe metadata extraction in `pilot.py`.
- [ ] Add and pass unavailable, malformed, and oversized metadata response tests without leaking response bodies.
- [ ] Run `tests/test_jina_pilot.py` and hosted-adapter bounded-reader tests.

### Task 2: Real scheduler, retry, concurrency, and readiness observations

**Files:**
- Modify: `tests/test_jina_pilot.py`
- Modify: `src/wsj_embeddings/adapters.py`
- Modify: `src/wsj_embeddings/pilot.py`

**Interfaces:**
- Consumes: `execute_rate_aware_batches(adapter, inputs, policy=...)`.
- Produces: per-probe request/retry/throttle observations, measured two-request overlap, safe billing status, and readiness.

- [ ] Add a retry-then-success CLI fixture asserting the actual request/retry counts and `retry_behavior=observed`.
- [ ] Run the test and observe failure because direct `embed()` calls bypass the scheduler.
- [ ] Route each probe through the production scheduler and retain safe per-response observations.
- [ ] Add a barrier-backed two-request fixture and assert attempted concurrency two and measured overlap two.
- [ ] Add billing-not-returned and readiness assertions, then implement strict safe billing extraction and truthful gate computation.
- [ ] Re-run focused pilot, adapter, and batching tests.

### Task 3: Operator contract and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture-crash-course.md`
- Modify: `.superpowers/sdd/all-tickets-plan/task-2-report.md`

**Interfaces:**
- Consumes: final deterministic pilot JSON contract.
- Produces: accurate operator and maintenance guidance plus implementation history.

- [ ] Replace hard-coded observed claims with returned/not-observed semantics and document manual readiness review.
- [ ] Run focused tests, Ruff, preprocessing smoke, embedding smoke, both CLI help commands, Markdown-link checks, and generated-artifact hygiene checks.
- [ ] Inspect `git diff --check` and `git status --short`.
- [ ] Commit with `Willis Erdman <willis.erdman@gmail.com>` and report exact verification evidence.
