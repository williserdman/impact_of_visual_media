# Task 1 implementation report: hosted Jina v4 adapter

## Delivered

Added a production `JinaEmbeddingAdapter` behind an injectable `JinaTransport`
seam.  It uses the synchronous `POST https://api.jina.ai/v1/embeddings`
contract with the fixed `jina-embeddings-v4` / `retrieval.passage` / 2048-float
configuration, `truncate: false`, and `return_multivector: false`.

The adapter reads only `JINA_API_KEY` (through an injectable environment mapping
in tests), supports separate text and base64-image request items, validates
positional response indexes and vector integrity, normalizes accepted vectors,
and returns raw-response/stored-vector hashes plus safe usage and rate-limit
metadata.  It classifies authentication, authorization, deterministic-request,
rate-limit, transient-server, timeout, and connection failures without placing
request content, credentials, authorization headers, or response bodies in
exception messages.

`FakeEmbeddingAdapter` and the existing coordinator/smoke seam remain intact.
The default production transport is standard-library HTTPS; every automated
test injects a synthetic transport and opens no socket.

## Files changed

- `src/wsj_embeddings/adapters.py`
  - Hosted adapter, request/response records, injectable transport protocol,
    production urllib transport, safe failure classification, and vector
    validation/hashing.
- `tests/test_jina_hosted_adapter.py`
  - Synthetic public-adapter contract tests for request serialization, mixed
    text/image inputs, out-of-order indexed responses, validation failures,
    retry classification, Retry-After, metadata, hashes, and redaction.

## Red–green evidence

Red command:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_jina_hosted_adapter.py
```

Observed expected pre-implementation failure: test collection could not import
`JinaEmbeddingAdapter` from `wsj_embeddings.adapters`.

Green command:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_jina_hosted_adapter.py
```

Observed output: `14 passed in 0.28s`.

## Verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_jina_hosted_adapter.py tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check .
/home/willis/projects/finance_wsj/.venv/bin/wsj-pipeline smoke
/home/willis/projects/finance_wsj/.venv/bin/python -m wsj_embeddings smoke
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed summary:

- Focused hosted-adapter plus existing embedding regression suite: `50 passed
  in 28.06s`.
- Ruff: `All checks passed!`.
- Preprocessing smoke: content-free successful result with `validation_ok:
  true`.
- Embedding smoke: `{"articles": 1, "embeddings": 1,
  "validation_ok": true}`.
- Diff check was clean; no generated data/artifacts/outputs are tracked.

The complete `python -m pytest -q` gate was also run before this focused final
gate; it completed successfully and proceeded to subsequent lint/smoke
commands. The terminal integration did not retain its final count, so this
report intentionally does not invent one.

## Self-review

- The production adapter holds the bearer token only in its private transport
  header and never includes it, request input, or HTTP response body in its
  public error text or returned metadata.
- All response data are checked before a vector is returned: exact item count,
  unique complete indexes, exact dimension, finite values, and nonzero finite
  norm. Returned vectors are ordered by caller input index, not service order.
- Stored-vector hashing uses the same little-endian float32 representation as
  the existing embedding pipeline and validator. Raw-response vector hashes
  use canonical compact JSON for the received `embedding` value.
- The existing `embed_text` method is preserved for the current coordinator;
  its hosted implementation delegates to the validated one-item batch path.
- Tests contain only generated text/base64 bytes and a synthetic credential;
  no network or licensed archive input is used.

## Concern

The shared virtual environment does not currently contain the
`wsj-embeddings` console-script entry point for this worktree. The supported
module invocation (`python -m wsj_embeddings smoke`) passes; this is an
environment-installation detail, not an adapter behavior failure.
