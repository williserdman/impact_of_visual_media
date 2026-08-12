# Whole-branch embedding review corrections

Schema version 15 makes hosted vector provenance explicit. The immutable
configuration records the requested model alias, the OpenAPI contract revision
the client was implemented against, and the response-body ceiling. It does not
claim runtime observations. Every hosted source vector and long-text part keeps
the adapter-produced SHA-256 of canonical compact JSON for the raw embedding
array, the SHA-256 of stored normalized little-endian float32 bytes, and the
safe model string actually returned with that generation. Local long-text
aggregates and multimodal midpoints have explicit derivation kinds and no raw
hosted representation. Exact version-14 catalogs are refused without mutation.

Recovery under the exclusive embedding lock now terminalizes abandoned
`running` records as `interrupted` before discarding only their invisible staged
reconciliation actions. Completed visible reconciliation remains intact.

The bounded scheduler carries partial request, retry, usage, throttle, and
elapsed observations with fatal errors so the coordinator can persist them
before failing the run. Request-wide deterministic failures, including HTTP
413, remain immediately fatal and terminalize unresolved batch items. An
indexed deterministic image rejection retries only that image through its
durable three-attempt limit, preserving successful siblings. Indexed text
configuration failures remain terminal.

Production HTTP response reads are bounded for both success and error paths.
The adapter reads at most the configured ceiling plus one detection byte and
classifies an oversized body as a content-free invalid response.

All automated evidence uses generated fixtures, simulated transports, and
temporary catalogs. No live credential, hosted request, licensed archive, or
real full-corpus operation is part of these corrections.
