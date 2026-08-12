# Task 14 implementation report: operator handoff and final integration gate

## Delivered

The operator handoff now covers clean installation, explicit disjoint roots,
hosted-processing authorization, secret handling, generated smoke, the real
generated-content Jina pilot, credential-free inventory/validation, bounded and
explicit full runs, interruption recovery, and DuckDB research queries.

The README and architecture crash course now state the transmission license and
privacy prerequisite, real hosted production/pilot versus fake or simulated
automated tests, the mutable hosted-model alias limitation, unknown cost and
throughput before a current pilot, and the prohibition on unattended real full
runs. Stale claims that full embedding runs were unavailable and that the
production Pillow path remained unexecuted were removed.

A clean install exposed and fixed one packaging defect: DuckDB 1.5.5 requires
`pytz` while converting generated `TIMESTAMPTZ` results, but the branch did not
declare it. `pytz>=2026.3.post1` is now a runtime dependency, matching the exact
dependency already selected in the parent checkout and its untracked lock.

The clean environment also executed the real Pillow 11.3.0 production codec
with generated images. The regression binds linked codec-build identity into
the adapter configuration, proves exact PNG pass-through, and publishes and
reopens a deterministic derived JPEG for a generated source above the hosted
byte threshold.

## Final integration corrections

The clean production-codec run reproduced the three deferred hosted-failure CLI
cases. Timeout and rate-limit expectations predated rate-aware batching: both
now use the documented durable three-attempt budget and return terminal summary
counts. A request-wide deterministic 413 exposed a real lifecycle defect: the
scheduler checkpointed attempts but delivered no terminal item outcomes, and a
limited run remained unfinished. The focused fix commits terminal
`deterministic_request` outcomes for every unresolved item in the rejected
request, marks the run failed and finished, preserves completed siblings, and
does not repurchase terminal work on replay. Authentication and authorization
remain recoverable in-progress scope failures.

The first full clean suite then found six stale schema-14 or DuckDB-1.5 fixture
expectations. Corrections retained the production validation contract: prior
schema fixtures distinguish physical tables through `duckdb_tables()` from the
two canonical views; damaged-schema coverage mutates private storage and proves
byte-preserving refusal; corruption fixtures expect the additive public-vector
checkpoint issue; and removed-header multimodal work uses documented
`stale_input`. That last fixture exposed one production mismatch: the state
transition retained an old input hash. The transition now clears source/input
identity when it creates `stale_input`, and offline validation accepts the
result.

## Clean installation evidence

An initial standard-library `python3 -m venv` attempt failed because this host
lacks `ensurepip`; no system configuration was changed. The isolated install
used:

```bash
uv venv --clear /tmp/wsj-jina-handoff.co2CYY/venv
uv pip install --python /tmp/wsj-jina-handoff.co2CYY/venv/bin/python -e '.[dev]'
```

The install completed from the feature worktree. Both `wsj-pipeline` and
`wsj-embeddings` console scripts exist. Relevant installed versions were:

```text
wsj-preprocessing==0.1.0
Pillow==11.3.0
tokenizers==0.21.4
pytz==2026.3.post1
duckdb==1.5.5
```

The generated production-codec fixture observed this runtime identity:

```text
pillow-11.3.0-exif-transpose-alpha-white-rgb-jpeg-q85-420-lanczos-if-over20000000px-optimize0-progressive0-metadata-none-v1-build-jpeg-6.2-turbo-3.1.1-zlib-1.3-webp-1.5.0
```

It retained an exact 78-byte generated PNG and converted a generated
5,883,411-byte PNG to a verified 1,486,202-byte JPEG rendition.

## Red-green and correction evidence

- Clean CLI baseline before `pytz`: `15 failed, 10 passed`; DuckDB reported
  `ModuleNotFoundError: No module named 'pytz'`. After installing the declared
  dependency: only the three deferred hosted cases failed.
- Deterministic-request RED: exact public CLI fixture left both source items
  `in_progress` and the run `running`. GREEN: `1 passed`; both items were
  terminal attempt one/413, the run was failed/finished, replay attempted zero
  items and made no second transport call, and validation reported only the
  expected header terminal coverage issue.
- Timeout, 429, and 413 public cases: `3 passed, 22 deselected in 20.03s`.
- Authentication/sibling/retry regression selection:
  `6 passed, 231 deselected in 18.05s`.
- Clean Pillow plus complete CLI file: `26 passed in 52.24s`.
- First full gate: `6 failed, 555 passed in 952.42s`; this result is retained
  here and is not represented as passing.
- Exact correction selection, including four damaged-schema parameters:
  `9 passed in 22.89s`; changed-file Ruff passed.

## Final fixture-safe gate

The corrected tree was verified once with:

```bash
/tmp/wsj-jina-handoff.co2CYY/venv/bin/python -m pytest -q
/tmp/wsj-jina-handoff.co2CYY/venv/bin/ruff check .
/tmp/wsj-jina-handoff.co2CYY/venv/bin/wsj-pipeline smoke
/tmp/wsj-jina-handoff.co2CYY/venv/bin/wsj-embeddings smoke
```

Observed results:

- full generated-fixture suite: `561 passed in 858.18s (0:14:18)`;
- Ruff: `All checks passed!`;
- preprocessing smoke: two articles, three successful generated candidates,
  one duplicate, zero failures, and `validation_ok: true`;
- embedding smoke: `{"articles": 1, "embeddings": 1,
  "validation_ok": true}`.

A separate generated one-article fake-adapter run published all three
modalities and the installed `wsj-embeddings validate` command returned exit
zero, no issues, no stale/retryable/orphan state, and `validation_ok: true`.

Every parent and subcommand help surface for both installed console scripts was
captured and checked. It exposes exactly four preprocessing commands and five
embedding commands, the three explicit embedding roots, the optional validation
configuration selector, the exactly-one-of limit/full scope, reprocess, worker
override, and hosted-processing authorization described in the operator docs.

## Hygiene and self-review

- README/crash-course relative links resolve.
- Tracked generated-path audit found no `data/`, `artifacts/`, `outputs/`,
  DuckDB catalog, rendition, lock, or log artifact.
- Tracked files contain no private-key marker or credential assignment.
- Current-doc stale schema/full-run/Pillow wording audit found no match.
- `git diff --check` is clean.
- All inputs, catalogs, Markdown, images, and vectors used by verification were
  generated below temporary directories.
- No real archive, real embedding output, hosted request, credential, or real
  `--full` operation was used. Package installation was the only network use.
- No machine, daemon, environment-variable, or system configuration changed,
  so setup replication and remote-control restart rules were not triggered.

The request-wide fatal correction is deliberately narrow: only deterministic
request rejection receives terminal per-item outcomes before the fatal is
raised. Authentication and authorization behavior remains covered and
unchanged. The stale-input correction changes no schema shape or configuration
meaning; it enforces the already documented generation-free state invariant.

## Whole-branch review corrections

The final branch review identified provenance and hard-crash boundaries that
required schema version 15. The configuration now distinguishes the requested
model alias and implemented-against API contract from the safe response model
actually observed per hosted generation. Active source vectors, append-only
history, mutable long-text parts, and immutable part generations retain the
adapter-produced raw embedding-array SHA-256 and stored normalized float-byte
SHA-256 without coordinator recomputation. Local long-text aggregates and
multimodal midpoints are explicit derived records with no raw hosted
representation. Exact version-14 output is refused byte-for-byte.

On lock reacquisition, abandoned `running` runs become `interrupted` before
their invisible reconciliation actions are removed in bounded pages. Fatal
batched errors now carry completed-exchange metrics so failed runs retain safe
requests, retries, usage, throttles, and whole-run elapsed time. Indexed
deterministic image rejection retries only the image through its durable
three-attempt limit; request-wide HTTP 413 remains immediately terminal and
fatal. Production urllib response reads use the immutable response ceiling on
success and error paths and reject ceiling-plus-one without exposing the body.

Focused generated-fixture evidence covered exact schema/refusal, hosted direct
and long-part provenance, validator acceptance, abandoned-run recovery,
concurrent fatal sibling telemetry, authentication telemetry, indexed image
rejection/replay, and bounded success/error reads. No live Jina call, secret,
licensed archive, or real full run was used.
