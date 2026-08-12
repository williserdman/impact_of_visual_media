# Live pilot measurement correction

## Purpose

The generated-content pilot must report only facts returned or exercised by
the current hosted Jina service. It must not present a client contract revision,
research date, documented limit, or unexercised retry as a live observation.
The pilot remains manual, stateless, generated-only, and separate from corpus
authorization and execution.

## Boundaries

The public seams are the `wsj-embeddings pilot` JSON/exit contract, the injected
hosted HTTP transport, and the existing rate-aware batch scheduler. The pilot
accepts no source, output, article, text, image, or file selector. It does not
read or create project state and never calls the licensed-corpus coordinator.

The hosted adapter may issue bounded GET requests only to:

- `https://api.jina.ai/openapi.json`
- `https://api.jina.ai/v1/models`

The production transport applies the same bounded success/error body reader to
GET and POST. Metadata parsing retains only strict, bounded, content-free
allowlisted labels, numeric fields, and lists. It never emits an authorization
value, arbitrary response body, error detail, billing currency that was not
returned, or an invented provider fact.

## Measurements

The deterministic, versioned JSON result separates the requested model alias
and `client_api_contract_version` from live observations. OpenAPI and model
catalogue observations are either `observed` with safely returned fields or
`not_observed` with a classified reason and safe status code. Model-catalogue
price fields are retained only if returned; currency is `not_returned` when it
is absent.

Every embedding probe uses the production adapter and rate-aware scheduler with
`truncate: false`. Fixed generated probes cover normal text, PNG, a mixed
request, locally token-counted text boundaries, and exact-byte PNG boundaries.
Per-probe output reports actual returned model labels, dimensions, raw vector
norms, usage, safe billing fields or `not_returned`, safe rate headers, request,
retry, and throttle counts, and the classified outcome.

A separate generated concurrency probe creates two one-item requests with an
attempted fan-out of two. A thread-safe observation wrapper records real call
overlap; scheduler fan-out and measured overlap are reported separately and
make no throughput promise. Natural transient failures may exercise the
existing bounded retry path. Retry behavior is `observed` only when a retry
actually occurs and otherwise `not_observed`; the pilot never induces a
failure merely to measure retries.

## Readiness

`ready_for_operator_review` requires successful normal text, image, and mixed
probes, configured dimensions, the requested task, `truncate: false`, and safe
returned model labels. It is not corpus authorization and does not enable
`run`. Missing metadata is a warning. Context and image settings are confirmed
only by the corresponding actual boundary probe outcome, never by documentation
or metadata alone; unavailable observations remain explicit.

## Tests and documentation

Automated tests use simulated GET/POST transports and generated data only. They
cover metadata success, unavailable, malformed, and oversized bodies; a
retry-then-success path; two-request overlap; absent billing fields; selectors;
and secret/body redaction. README and the architecture crash course describe
the manual review gate and remove hard-coded live-observation claims.
