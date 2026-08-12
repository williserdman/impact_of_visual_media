# Task 14 implementation report: operator handoff and final integration gate

## Corpus-scale validation correction

The read-only validator now streams every corpus-sized relation through fixed
cursor ceilings and performs canonical metadata joins once per bounded page.
Long-text aggregate validation uses one ordered join across active, historical,
and reconciliation-visible descriptors and their immutable part generations;
it no longer repeats the descriptor UNION or issues one query per aggregate.
Hidden reconciliation-visible multimodal vectors are recomputed from the exact
physical text and image generations named by their provenance, even when a
tampered vector and action agree on a valid unit-vector hash.

Rendition orphan validation no longer trusts the process default temporary
directory. It resolves and probes an explicit sequence of scratch parents,
rejects any candidate inside or above a configured source or output root, and
fails closed with `validation_scratch_unavailable` when none is safe and
writable. The content-free SQLite reference index is always removed on return.
The verified parent and private child descriptors remain open throughout the
SQLite lifetime; creation and cleanup are descriptor-relative, while SQLite
receives only the held child's `/proc/self/fd` path. Generated failpoints replace
the parent pathname after parent verification and after child creation, proving
that neither substitution redirects a write into configured roots.
The focused descriptor-race gate passed nine generated validator fixtures,
including both substitution windows, unavailable procfs, unsafe/default-temp
candidates, orphan detection, concurrent artifact change, and ordinary output.
Both generated smokes and changed-file Ruff also passed.
Generated tests cover a default-temp/source collision, an injected no-safe
candidate set with byte-identical configured trees, scratch cleanup, a corrupt
vector beyond two pages, one canonical query per page, one long-aggregate join
for several historical generations, and hidden midpoint tampering.

Focused correction evidence passed: all four stream-module tests and five
public validator regressions covering the scratch, multi-page, aggregate-query,
and hidden-multimodal boundaries. A broader validator selection reached 106
passes before its final orphan-vector fixture exposed an inner-join regression;
the left-join correction and its four directly affected public-vector fixtures
then passed. Changed-file Ruff and whitespace checks, both generated smokes,
and the live embedding parent/validate help surfaces also passed. The embedding
console script was exercised via its installed module entry point because this
existing development environment did not contain the `wsj-embeddings` wrapper.

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

### Core review correction round 1

The schema remained at version 15. Validation now treats derivation labels as
semantic claims: only source text/images may identify a hosted response,
multimodal rows must identify their local midpoint, long-text aggregate rows
must have matching aggregation provenance, and synthetic source vectors are
accepted only under the generated-fixture API/tokenizer profile marker.
Mutable successful long-text parts must match their immutable generation in
derivation kind, raw-response hash, and response model as well as vector facts.

Visible but uncompacted reconciliation actions no longer hide physical vector
corruption. The validator checks the hidden storage vector's dimension,
finiteness, norm, stored hash, and semantic provenance even when the expected
action fields were changed to match the corrupt storage. Malformed hosted batch
item counts or local indexes now enter the fatal wrapper after all completed
exchange usage and throttle observations are accumulated, allowing the
coordinator to persist those metrics before failing the run.

The correction used generated fixtures only. Focused selections covered nine
validator cases and three scheduler/coordinator cases; changed-file Ruff passed.
Per review instruction, the full test suite was not run. No live API call, key,
real archive, or real full run was used.

### Corpus-scale validation correction

The scale review found corpus-sized Python materializations of canonical
articles, active vectors, current and historical long-text parts, and
provenance. Validation now uses one bounded-row interface: metadata pages are
capped at 256 rows, single-vector pages at 32, and three-vector multimodal pages
at eight. Relational reference/equality checks run as SQL anti-joins or counts.

Long-text vectors are completely streamed. Aggregate descriptors advance by
keyset and each provenance group feeds one 2,048-value token-weighted
accumulator plus scalar span cursors; only its current Markdown bytes remain in
memory. Artifact tokens stream articles and references. Their complete
no-follow rendition-tree digest is order-independent and duplicate-sensitive.
Orphan checks stream references into a content-free temporary SQLite
primary-key index outside configured roots, probe it one verified leaf at a
time, and remove it on return.

Focused evidence covers the bounded cursor and digest contracts, corruption
beyond the first 32-vector page, fresh and long-text validation, visible
reconciliation, multimodal/history provenance, concurrent artifact changes,
and coverage. `validate.py` contains no `fetchall()` call. No real archive,
live API, credential, real full run, or full suite was used.

### Final integration review corrections

The fatal-telemetry production correction was already present: malformed item
counts and local indexes are accumulated into `BatchExecutionFatal`, and both
limited and full coordinators persist its completed request, retry, usage, and
throttle observations before failing the run. The public catalog fixture now
exercises both malformed shapes after one retry and a durable sibling success.
It observes three requests, one retry, eight prompt tokens, one total token,
one throttle, and four seconds of whole-run elapsed time in the failed `runs`
row. A separate generated packing failure proves that a plain hosted error
before any exchange stores zero requests, retries, usage, and throttles while
retaining the three-second whole-run elapsed time. The two scheduler shape
fixtures and three public coordinator fixtures passed.

The README quick command synopsis now states the same exactly-one-positive-
limit-or-explicit-full contract as the live embedding parser. Recovery now
distinguishes the preprocessing lock from
`<embedding-output-root>/pipeline.lock`: the operator must first verify that no
embedding pipeline process is active, remove only that exact stale lock, and
rerun the same idempotent embedding command.

An isolated `duckdb==1.5.5` environment fetched a generated `TIMESTAMPTZ`
successfully with `pytz==2024.1`. After changing the declared dependency to the
bounded verified range `pytz>=2024.1,<2027`, a clean editable development
install explicitly pinned to the floor passed the three public telemetry
fixtures and `wsj-embeddings smoke`. No shared environment, lock file, machine
configuration, service, or daemon was changed.

The final focused gate used that isolated floor environment. Scheduler tests
reported `2 passed`; public coordinator tests reported `3 passed`; changed-file
Ruff passed; preprocessing and embedding smokes returned successful generated
summaries. Installed preprocessing, embedding, and embedding-run help exposed
the documented command sets, exactly-one scope, and hosted authorization.
Current-document relative links resolved, whitespace and tracked-artifact
hygiene checks passed, and no full suite was rerun. The earlier first and final
full-suite results above remain the complete full-suite chronology.
