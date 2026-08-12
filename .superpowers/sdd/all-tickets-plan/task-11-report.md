# Task 11 implementation report: rate-aware multimodal batching

## Delivered

The real hosted Jina adapter now exposes one mixed-input batch seam with exact
positional outcomes. Text, header-image base64, and long-text-part inputs can
share a synchronous v4 request. Valid response indexes are normalized and
returned independently; a missing, duplicate, malformed, or invalid sibling
becomes a content-free indexed outcome instead of discarding validated
siblings. Request-wide HTTP, transport, model, or body failures retain their
classified exception scope.

`batching.py` greedily packs stable input order under independent positive
item, estimated-token, and total encoded-input-byte ceilings. Canonical text
and long parts use pinned-tokenizer counts; images use the documented 28-by-28
tile estimate at 10 tokens per tile. It runs
bounded request waves through a thread pool, never above configured maximum
concurrency. A throttle halves later concurrency, with a floor of one. Eligible
timeouts, connections, rate limits, server failures, and explicitly retryable
indexed response gaps use bounded exponential backoff. Injectable jitter,
sleep, and monotonic clocks make all retry behavior deterministic in fixtures.
Production uses bounded nondeterministic jitter; tests inject deterministic
jitter. Numeric and RFC 9110 HTTP-date `Retry-After` plus zero-remaining
request/token reset headers can extend the wait only to the configured cap.
Authentication, authorization, missing credential, deterministic
request configuration, and other nonretryable request failures stop the run.

The coordinator stages all hosted work only after safe input preparation and
durable work registration. Every hosted attempt is checkpointed before the
call. The scheduler invokes a serialized coordinator callback as each indexed
outcome is processed, and every successful or failed source item and long-text
part then crosses its own catalog transaction before later sleeps or waves.
Durable prior attempt counts reduce the per-item remaining budget. Partial mixed
responses therefore preserve valid
text, image, and part work; replay sends only unproven items. Completed part
sets aggregate through the established complete long-text formula, and source
successes still produce the established multimodal derivative.

Run summaries and `runs` now include hosted requests, item retries, safe named
allowlisted finite nonnegative numeric usage totals, throttle observations, and
whole-run elapsed seconds. They contain no
Markdown, image bytes, vector values, credentials, response bodies, or
exception text.

## Schema and immutable configuration

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from 11 to 12.
- `embedding_configurations` adds the item, estimated-token, encoded-byte,
  concurrency, attempt, initial-backoff, and maximum-backoff policy fields.
  All participate in `configuration_id` through the complete public profile.
- `runs` adds hosted request/retry counts, compact key-sorted safe usage JSON,
  throttle count, and elapsed seconds.
- The catalog remains exactly eleven base tables with no indexes.
- An exact prior-v11 shape is refused without mutation or migration.
- The client configuration identity advances to
  `wsj-embeddings-config-v4`.

## TDD evidence

The first hosted-batch fixture observed RED at collection because public
`JinaBatchLimits` was absent (`1 error in 0.72s`). It then passed four mixed
index/payload-bound cases (`4 passed, 25 deselected in 0.41s`).

The scheduler fixtures observed RED because `wsj_embeddings.batching` did not
exist (`1 error in 0.54s`). The implemented scheduler plus adapter selections
passed `9 passed, 25 deselected in 0.41s` and later the expanded timeout suite.

The configuration identity fixture observed RED because seven batch-policy
fields were missing. The exact fresh schema and complete profile identity
selection passed `2 passed, 161 deselected in 1.11s`.

The coordinator batch branch was mutation-checked: disabling it produced the
expected three failures (no mixed calls and wrong partial behavior); restoring
it passed `3 passed, 163 deselected in 8.63s`. The long-part/image partial and
resume fixture passed `1 passed, 166 deselected in 15.60s`. Request-wide
rate/auth/config behavior passed `5 passed, 166 deselected in 28.46s`. Exact
schema-v12/prior-v11 refusal passed `2 passed, 170 deselected in 2.14s`.

The combined affected adapter, scheduler, schema, and coordinator selection
passed `15 passed, 186 deselected in 23.39s` before the final timeout and
request-scope fixtures were added.

A final rate-reset fixture observed RED because a successful response with
zero remaining requests immediately started deferred work (`1 failed,
6 deselected in 0.36s`), then GREEN after honoring its numeric reset
(`2 passed, 5 deselected in 0.35s` with the established Retry-After case).
An exact-token-estimate fixture observed RED because the input seam accepted
no supplied model-token count (`1 failed, 29 deselected in 0.61s`), then GREEN
after the coordinator attached tokenizer/tile estimates (`1 passed,
29 deselected in 0.34s`).

The correction round first observed three scheduler failures: the serialized
outcome callback and durable prior-attempt seam were absent, and uncapped
sleeps were `[53.0, 200.0]` instead of `[3.0, 3.0]`. The exact selection then
passed `3 passed, 7 deselected in 0.37s`. Default jitter passed `1 passed,
10 deselected`; valid/malformed HTTP-date parsing passed `2 passed, 30
deselected`. Coordinator crash/replay durability and entering-attempt-two
fixtures passed `2 passed, 172 deselected in 5.67s`. Hostile usage plus corrupt
run-metric validation first failed eight cases, then passed `8 passed, 174
deselected in 13.84s`. Adapter and scheduler files passed `43 passed in 5.25s`;
the final scheduler file, including preservation of parsed per-item Retry-After
metadata, passed `12 passed in 0.87s`.

The second correction round reproduced three additional boundaries. A fatal in
the first submission of a two-request concurrent wave left the completed second
success in progress; a retryable long part that succeeded on its next in-run
attempt still prevented parent aggregation; and a successful response carrying
HTTP-date `Retry-After` yielded a zero-second wait. After deferring deterministic
fatal selection until all completed exchanges are processed, tracking only final
unresolved long-part outcomes, and normalizing the parsed date seconds into safe
rate metadata, the coordinator pair passed `2 passed, 182 deselected in 9.92s`
and the date-delay fixture passed `1 passed, 32 deselected in 0.32s`.

## Self-review

- Production uses `JinaEmbeddingAdapter.embed_batch`; tests inject only local
  fake adapters and transports. No hosted async v5 route is used.
- Text token estimates use the pinned tokenizer counts; image estimates use
  documented tile accounting. Total encoded bytes count both UTF-8 text and
  base64 image values.
- Request indexes are the only mapping between an input and its outcome.
  Out-of-range entries cannot be assigned; duplicates invalidate only that
  ambiguous index; a missing index remains independently retryable.
- Attempt checkpoints happen before executor submission. Outcome callbacks,
  catalog publication, aggregation, and validation remain on the serialized
  coordinator; worker threads call only the hosted adapter.
- Existing fake adapters retain their established single-item seam. Only the
  real hosted adapter and explicit batching fakes opt into the new scheduler.
- Auth/authz errors intentionally leave in-progress checkpoints; replay can
  recover them after the operator corrects run-scope credentials or access.
- Concurrent wave exchanges are processed in stable submission order; the first
  fatal in that order is raised only after completed sibling outcomes commit.
- Schema validation reconstructs every batching profile field and refuses
  prior or malformed shapes before writable use.

## Safety and concerns

No network request, API credential, licensed archive read, real embedding
output, shared environment mutation, or full-corpus operation was used.

The default limits are conservative versioned policy, not a live throughput
promise. A credentialed generated-fixture pilot still owns measuring current
v4 throughput and rate headers. Malformed `Retry-After` values are safely
ignored. Jina's documented remaining/reset headers are not stable across
public sources, so only numeric named values affect scheduling.
