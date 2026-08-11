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

## Fix round 1/5

### Review findings addressed

- Classified adapter errors now use explicit cause suppression for base64 input,
  transport, JSON decoding, scalar conversion, and vector serialization
  failures. This prevents exception chaining from exposing unsafe request,
  transport, or response text in formatted tracebacks.
- The returned model must exactly equal `jina-embeddings-v4`; arbitrary model
  labels are rejected before metadata is returned. Rate-limit metadata is now a
  bounded numeric allowlist of known header names, and invalid/arbitrary header
  values are omitted.
- Vector entries must be JSON numeric values (`int` or `float`, excluding
  `bool`) before conversion. Numeric strings and booleans are rejected.

### Red–green evidence

Red command:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_jina_hosted_adapter.py
```

Observed output: `8 failed, 13 passed in 0.47s`. Failures demonstrated the
original string rate-limit metadata, chained transport/JSON/base64 causes,
untrusted returned model, and accepted boolean/numeric-string vector values.

Green verification:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q tests/test_jina_hosted_adapter.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check src/wsj_embeddings/adapters.py tests/test_jina_hosted_adapter.py
git diff --check
```

Observed output: `21 passed in 0.36s`; Ruff reported `All checks passed!`; the
diff check was clean.

### Self-review

- The new regression test inspects both `__cause__` and formatted tracebacks
  for synthetic transport, response, and base64 secrets.
- Response model metadata has a single trusted value: the fixed configured
  alias. The rate-limit allowlist returns finite, nonnegative numeric values
  no greater than `1e15`; arbitrary header names and string values are never
  retained.
- Strict vector type checking occurs before `float()` and before any vector is
  returned. Existing mapping, dimension, finite-value, and zero-norm checks
  remain in place.
