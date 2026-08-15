"""Simulated public-CLI tests for the generated-content Jina pilot."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import threading
import zlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from wsj_embeddings.adapters import JinaEmbeddingAdapter, JinaHttpResponse
from wsj_embeddings.cli import build_parser, main
from wsj_embeddings.pilot import (
    _boundary_probes,
    _encoded_png,
    _normal_image_probe,
    run_jina_pilot,
)
from wsj_embeddings.tokenizer import PinnedJinaV4Tokenizer, PinnedTokenizerError


class PilotTransport:
    """Return complete synthetic hosted responses without opening a socket."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.metadata_requests: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse:
        del headers, timeout_seconds, max_response_bytes
        self.metadata_requests.append(url)
        if url == "https://api.jina.ai/openapi.json":
            payload = {"info": {"version": "2026.07.27.1603"}}
        elif url == "https://api.jina.ai/v1/models":
            payload = {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "context_length": 32_768,
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["embedding"],
                        "pricing": {
                            "prompt_tokens": 0.00000005,
                            "image": 0,
                            "request": 0,
                        },
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected metadata URL: {url}")
        return JinaHttpResponse(200, {}, json.dumps(payload).encode())

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse:
        del url, headers, timeout_seconds, max_response_bytes
        request = json.loads(body)
        self.requests.append(request)
        count = len(request["input"])
        return JinaHttpResponse(
            status_code=200,
            headers={"X-RateLimit-Remaining-Requests": "499"},
            body=json.dumps(
                {
                    "model": "jina-embeddings-v4",
                    "data": [
                        {"index": index, "embedding": [3.0, 4.0, *([0.0] * 2046)]}
                        for index in range(count)
                    ],
                    "usage": {"prompt_tokens": 11, "total_tokens": 11},
                }
            ).encode(),
        )


class PilotTokenizer:
    """Offline generated-text token counter for the public pilot seam."""

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple(
            (match.start(), match.end())
            for match in __import__("re").finditer(r"\S+", text)
        )


class FailingPilotTokenizer:
    def __init__(self, message: str) -> None:
        self.message = message

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        del text
        raise PinnedTokenizerError(self.message)


def _run_pilot(transport, *, capsys) -> dict[str, object]:
    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=PilotTokenizer(),
        )
        == 0
    )
    return json.loads(capsys.readouterr().out)


def test_all_live_pilot_probes_share_one_state_free_quota_window(tmp_path):
    transport = PilotTransport()
    adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=PilotTokenizer(),
    )
    adapter.profile = replace(
        adapter.profile,
        quota_max_requests=2,
        quota_max_estimated_tokens=90_000,
        quota_window_seconds=60,
    )
    now = [datetime(2026, 8, 13, 12, 0, tzinfo=UTC)]
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += timedelta(seconds=seconds)

    result = run_jina_pilot(
        adapter,
        quota_clock=lambda: now[0],
        quota_sleep=advance,
    )

    assert result["readiness"]["outcome"] == "ready_for_operator_review"
    assert len(transport.requests) > 2
    assert len(sleeps) == math.ceil(len(transport.requests) / 2) - 1
    assert set(sleeps) == {60.0}
    assert list(tmp_path.iterdir()) == []


def test_live_pilot_uses_pinned_counts_for_normal_text_quota():
    class WeightedPilotTokenizer(PilotTokenizer):
        def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
            if text == "generated pilot text":
                return ((0, 1),) * 44_360
            return super().token_offsets(text)

    class ZeroUsageTransport(PilotTransport):
        def post(self, *args, **kwargs):
            response = super().post(*args, **kwargs)
            payload = json.loads(response.body)
            payload["usage"] = {"prompt_tokens": 0, "total_tokens": 0}
            return JinaHttpResponse(
                response.status_code, response.headers, json.dumps(payload).encode()
            )

    transport = ZeroUsageTransport()
    adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=WeightedPilotTokenizer(),
    )
    adapter.profile = replace(
        adapter.profile,
        quota_max_requests=90,
        quota_max_estimated_tokens=90_000,
        batch_max_estimated_tokens=45_000,
    )
    now = [datetime(2026, 8, 13, 12, 0, tzinfo=UTC)]
    request_counts_at_sleep: list[int] = []

    def advance(seconds: float) -> None:
        request_counts_at_sleep.append(len(transport.requests))
        now[0] += timedelta(seconds=seconds)

    run_jina_pilot(
        adapter,
        quota_clock=lambda: now[0],
        quota_sleep=advance,
    )

    assert request_counts_at_sleep[0] == 3


def test_live_pilot_sleeps_once_until_enough_token_debt_expires():
    now = [datetime(2026, 8, 13, 12, 0, tzinfo=UTC)]
    observed_usage = iter((1_000, 90_000, 1_000))

    class StaggeredUsageTransport(PilotTransport):
        def post(self, *args, **kwargs):
            response = super().post(*args, **kwargs)
            payload = json.loads(response.body)
            tokens = next(observed_usage, 1)
            payload["usage"] = {
                "prompt_tokens": tokens,
                "total_tokens": tokens,
            }
            now[0] += timedelta(seconds=10)
            return JinaHttpResponse(
                response.status_code, response.headers, json.dumps(payload).encode()
            )

    adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=StaggeredUsageTransport(),
        tokenizer=PilotTokenizer(),
    )
    adapter.profile = replace(
        adapter.profile,
        quota_max_requests=90,
        quota_max_estimated_tokens=90_000,
        batch_max_estimated_tokens=45_000,
    )
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += timedelta(seconds=seconds)

    run_jina_pilot(
        adapter,
        quota_clock=lambda: now[0],
        quota_sleep=advance,
    )

    assert sleeps[0] == 50.0


@pytest.mark.parametrize(
    ("metadata_response", "expected"),
    [
        (
            JinaHttpResponse(503, {}, b'{"detail":"metadata-secret-503"}'),
            {
                "outcome": "not_observed",
                "reason": "transient_server",
                "status_code": 503,
            },
        ),
        (
            JinaHttpResponse(200, {}, b'{"info":{"version":"unterminated"}'),
            {"outcome": "not_observed", "reason": "invalid_response"},
        ),
        (
            JinaHttpResponse(200, {}, b"x" * 1_000_001),
            {"outcome": "not_observed", "reason": "invalid_response"},
        ),
        (
            JinaHttpResponse(
                200,
                {},
                b'{"info":{"version":"too-deep"},"nested":'
                + (b"[" * 33)
                + b"0"
                + (b"]" * 33)
                + b"}",
            ),
            {"outcome": "not_observed", "reason": "invalid_response"},
        ),
        (
            JinaHttpResponse(200, {}, b'{"info":{"version":"\\ud800"}}'),
            {"outcome": "not_observed", "reason": "invalid_response"},
        ),
        (
            JinaHttpResponse(
                200,
                {},
                (
                        '{"info":{"version":"2026.07.27.1603"},"data":[],"number":'
                    + ("9" * 500)
                    + "}"
                ).encode(),
            ),
            {
                "outcome": "observed",
                "status_code": 200,
                "version": "2026.07.27.1603",
            },
        ),
        (
            JinaHttpResponse(
                200,
                {},
                b'{"info":{"version":"2026.07.27.1603"},"number":'
                + (b"9" * 5_000)
                + b"}",
            ),
            {"outcome": "not_observed", "reason": "invalid_response"},
        ),
    ],
)
def test_pilot_classifies_unavailable_malformed_and_oversized_metadata(
    metadata_response: JinaHttpResponse,
    expected: dict[str, object],
    capsys,
) -> None:
    """Break caught: optional metadata aborts probes or leaks provider bodies."""

    class MetadataFailureTransport(PilotTransport):
        def get(self, url: str, **kwargs) -> JinaHttpResponse:
            del kwargs
            self.metadata_requests.append(url)
            if url.endswith("openapi.json"):
                return metadata_response
            return JinaHttpResponse(200, {}, b'{"data":[]}')

    transport = MetadataFailureTransport()

    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=PilotTokenizer(),
        ) == 0
    )
    output = capsys.readouterr().out
    result = json.loads(output)

    assert result["metadata"]["openapi"] == expected
    assert result["metadata"]["model_catalogue"] == {
        "outcome": "not_observed",
        "reason": "model_not_returned",
        "status_code": 200,
    }
    assert "metadata-secret" not in output


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "openapi.json",
            {"info": {"version": "Bearer metadata-secret-token"}},
        ),
        (
            "openapi.json",
            {"info": {"version": "metadata-secret-token"}},
        ),
        (
            "openapi.json",
            {"info": {"version": "secret-token"}},
        ),
        (
            "v1/models",
            {"data": [{"id": "jina_secret_token"}]},
        ),
        (
            "v1/models",
            {"data": [{"id": "jina-embeddings-v4-secret-token"}]},
        ),
        (
            "v1/models",
            {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "context_length": 10**500,
                        "input_modalities": ["text"],
                    }
                ]
            },
        ),
        (
            "v1/models",
            {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "input_modalities": ["text", "metadata-secret-token"],
                    }
                ]
            },
        ),
        (
            "v1/models",
            {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "output_modalities": ["secret-token"],
                    }
                ]
            },
        ),
        (
            "v1/models",
            {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "pricing": {
                            "currency": "Bearer metadata-secret-token",
                            "prompt_tokens": 0.00000005,
                        },
                    }
                ]
            },
        ),
        (
            "v1/models",
            {
                "data": [
                    {
                        "id": "jina-embeddings-v4",
                        "input_modalities": ["text", "Bearer metadata-secret-token"],
                    }
                ]
            },
        ),
    ],
)
def test_pilot_rejects_unsafe_allowlisted_metadata_values_content_free(
    endpoint: str,
    payload: dict[str, object],
    capsys,
) -> None:
    """Break caught: allowlisted metadata names make arbitrary values safe."""

    class UnsafeMetadataTransport(PilotTransport):
        def get(self, url: str, **kwargs) -> JinaHttpResponse:
            if url.endswith(endpoint):
                return JinaHttpResponse(200, {}, json.dumps(payload).encode())
            return super().get(url, **kwargs)

    result = _run_pilot(UnsafeMetadataTransport(), capsys=capsys)
    output = json.dumps(result)
    observation = (
        result["metadata"]["openapi"]
        if endpoint == "openapi.json"
        else result["metadata"]["model_catalogue"]
    )

    assert observation["outcome"] == "not_observed"
    assert "metadata-secret-token" not in output


@pytest.mark.parametrize(
    "unsafe_model",
    (
        "Bearer response-model-secret",
        "jina-embeddings-v4-secret-token",
        "jina-embeddings-v4-metadata-secret-token",
        "jina_secret_token",
    ),
)
def test_pilot_rejects_secret_like_returned_model_without_echo(
    unsafe_model: str, capsys
) -> None:
    """Break caught: printable response model labels can carry arbitrary secrets."""

    class UnsafeModelTransport(PilotTransport):
        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            response = super().post(url, **kwargs)
            payload = json.loads(response.body)
            payload["model"] = unsafe_model
            return JinaHttpResponse(
                response.status_code,
                response.headers,
                json.dumps(payload).encode(),
            )

    result = _run_pilot(UnsafeModelTransport(), capsys=capsys)
    output = json.dumps(result)

    assert result["probes"][0]["status"] == "rejected"
    assert result["probes"][0]["error"] == "invalid_response"
    assert unsafe_model not in output


def test_pilot_metadata_is_observed_from_fixed_live_endpoints(capsys) -> None:
    """Break caught: recorded research facts are presented as live observations."""

    transport = PilotTransport()

    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=PilotTokenizer(),
        ) == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert transport.metadata_requests == [
        "https://api.jina.ai/openapi.json",
        "https://api.jina.ai/v1/models",
    ]
    assert result["client_contract"] == {
        "client_api_contract_version": "openapi-2026.07.27.1603",
        "requested_model": "jina-embeddings-v4",
        "task": "retrieval.passage",
        "truncate": False,
    }


@pytest.mark.parametrize(
    "unsafe_message",
    [
        "/private/cache/tokenizer.json missing",
        "checksum expected-secret-sha",
        "loader response-body-secret",
    ],
)
def test_pilot_maps_tokenizer_failures_after_auth_without_leaking_details(
    unsafe_message: str,
    capsys,
) -> None:
    """Break caught: tokenizer failure escapes with cache/checksum/load details."""

    transport = PilotTransport()

    exit_code = main(
        ["pilot"],
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=FailingPilotTokenizer(unsafe_message),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert json.loads(captured.out) == {"error": "pilot_tokenizer_unavailable"}
    assert captured.err == ""
    assert unsafe_message not in captured.out
    assert len(transport.requests) == 1


def test_pilot_checks_hosted_auth_before_loading_boundary_tokenizer(
    capsys,
) -> None:
    """Break caught: an invalid credential is hidden by local tokenizer failure."""

    class AuthenticationFailureTransport(PilotTransport):
        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            del url, kwargs
            self.requests.append({})
            return JinaHttpResponse(401, {}, b"{}")

    transport = AuthenticationFailureTransport()

    exit_code = main(
        ["pilot"],
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=FailingPilotTokenizer("private-tokenizer-secret"),
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(output) == {"error": "authentication"}
    assert "private-tokenizer-secret" not in output
    assert len(transport.requests) == 1


def test_pilot_maps_real_resolve_checksum_and_load_failures_content_free(
    capsys, tmp_path
) -> None:
    """Break caught: only a synthetic exception bypasses tokenizer detail leaks."""

    artifact = tmp_path / "tokenizer.json"
    artifact.write_bytes(b"generated tokenizer fixture")

    def missing_resolver(_repo: str, _filename: str, _revision: str):
        raise OSError("/private/cache/tokenizer-secret.json")

    tokenizers = (
        PinnedJinaV4Tokenizer(resolver=missing_resolver),
        PinnedJinaV4Tokenizer(resolver=lambda *_args: artifact),
        PinnedJinaV4Tokenizer(
            resolver=lambda *_args: artifact,
            expected_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            loader=lambda _serialized: (_ for _ in ()).throw(
                RuntimeError("loader-response-secret")
            ),
        ),
    )

    for tokenizer in tokenizers:
        transport = PilotTransport()
        exit_code = main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=tokenizer,
        )
        captured = capsys.readouterr()

        assert exit_code == 1
        assert json.loads(captured.out) == {"error": "pilot_tokenizer_unavailable"}
        assert captured.err == ""
        assert "private" not in captured.out
        assert "checksum" not in captured.out
        assert "loader-response-secret" not in captured.out
        assert len(transport.requests) == 1


def test_pilot_reports_natural_retry_from_production_scheduler(capsys) -> None:
    """Break caught: pilot hard-codes zero retries instead of exercising policy."""

    class RetryThenSuccessTransport(PilotTransport):
        def __init__(self) -> None:
            super().__init__()
            self.failed_once = False

        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            if not self.failed_once:
                self.failed_once = True
                return JinaHttpResponse(
                    429,
                    {"Retry-After": "0"},
                    b'{"detail":"generated retry fixture"}',
                )
            return super().post(url, **kwargs)

    transport = RetryThenSuccessTransport()

    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=PilotTokenizer(),
        ) == 0
    )
    result = json.loads(capsys.readouterr().out)

    authentication_probe = next(
        probe for probe in result["probes"] if probe["name"] == "image_normal"
    )
    assert authentication_probe["requests"] == 2
    assert authentication_probe["retries"] == 1
    assert authentication_probe["rate_limit_headers"]["retry-after"] == [0.0]
    assert result["retry_behavior"] == {"outcome": "observed", "retries": 1}


def test_pilot_preserves_prior_observations_when_retry_becomes_fatal(
    capsys,
) -> None:
    """Break caught: a later fatal attempt discards safe earlier measurements."""

    class MissingThenFatalTransport(PilotTransport):
        def __init__(self) -> None:
            super().__init__()
            self.sequence = 0

        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            self.sequence += 1
            if self.sequence == 1:
                return JinaHttpResponse(
                    200,
                    {"X-RateLimit-Remaining-Requests": "9"},
                    json.dumps(
                        {
                            "billing": {"amount": 0.25, "currency": "USD"},
                            "data": [],
                            "model": "jina-embeddings-v4-observed-a",
                            "usage": {"prompt_tokens": 5, "total_tokens": 5},
                        }
                    ).encode(),
                )
            if self.sequence == 2:
                return JinaHttpResponse(
                    400,
                    {"X-RateLimit-Remaining-Requests": "7"},
                    b'{"detail":"fatal-response-secret"}',
                )
            return super().post(url, **kwargs)

    result = _run_pilot(MissingThenFatalTransport(), capsys=capsys)
    probe = result["probes"][1]

    assert probe == {
        "billing": {
            "outcome": "returned",
            "responses": [{"amount": 0.25, "currency": "USD"}],
        },
        "concurrency_attempted": 1,
        "concurrency_observed": 1,
        "dimensions": [],
        "error": "deterministic_request",
        "name": "image_normal",
        "rate_limit_headers": {
            "x-ratelimit-remaining-requests": [7.0, 9.0]
        },
        "requests": 2,
        "response_models": ["jina-embeddings-v4-observed-a"],
        "retries": 1,
        "status": "rejected",
        "status_codes": [200, 400],
        "throttles": 0,
        "usage": {"prompt_tokens": 5.0, "total_tokens": 5.0},
        "vector_norms": [],
    }
    assert "fatal-response-secret" not in json.dumps(result)


def test_pilot_attempts_and_measures_two_concurrent_generated_requests(
    capsys,
) -> None:
    """Break caught: reported concurrency is a configured number, not overlap."""

    class ConcurrentPilotTransport(PilotTransport):
        def __init__(self) -> None:
            super().__init__()
            self.concurrent_probe_gate = threading.Barrier(2)

        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            request = json.loads(kwargs["body"])
            values = [next(iter(item.values())) for item in request["input"]]
            if any("concurrency probe" in value for value in values):
                self.concurrent_probe_gate.wait(timeout=2)
            return super().post(url, **kwargs)

    transport = ConcurrentPilotTransport()

    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=transport,
            tokenizer=PilotTokenizer(),
        ) == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["concurrency"] == {
        "attempted": 2,
        "measured_overlap": 2,
        "outcome": "succeeded",
        "scheduler_wave": 2,
    }


def test_pilot_reports_absent_billing_and_operator_review_readiness(capsys) -> None:
    """Break caught: absent billing is zero or successful probes lack a gate."""

    assert (
        main(
            ["pilot"],
            environment={"JINA_API_KEY": "synthetic-secret"},
            transport=PilotTransport(),
            tokenizer=PilotTokenizer(),
        ) == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert all(
        probe["billing"] == {"outcome": "not_returned"}
        for probe in result["probes"]
        if probe["status"] == "succeeded"
    )
    assert result["readiness"] == {
        "outcome": "ready_for_operator_review",
        "required_probes": {
            "image_normal": "succeeded",
            "mixed_normal": "succeeded",
            "text_normal": "succeeded",
        },
    }
    assert result["retry_behavior"] == {
        "outcome": "not_observed",
        "retries": 0,
    }
    assert result["metadata"] == {
        "model_catalogue": {
            "model": {
                "context_length": 32_768,
                "id": "jina-embeddings-v4",
                "input_modalities": ["text", "image"],
                "output_modalities": ["embedding"],
                "pricing": {
                    "currency": "not_returned",
                    "image": 0,
                    "prompt_tokens": 0.00000005,
                    "request": 0,
                },
            },
            "outcome": "observed",
            "status_code": 200,
        },
        "openapi": {
            "outcome": "observed",
            "status_code": 200,
            "version": "2026.07.27.1603",
        },
    }


def test_pilot_reports_only_returned_safe_billing_fields(capsys) -> None:
    """Break caught: returned billing is dropped or arbitrary fields are emitted."""

    class BillingTransport(PilotTransport):
        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            response = super().post(url, **kwargs)
            payload = json.loads(response.body)
            payload["billing"] = {
                "amount": "0.0025",
                "currency": "USD",
                "provider_detail": "billing-secret-detail",
            }
            return JinaHttpResponse(
                response.status_code,
                response.headers,
                json.dumps(payload).encode(),
            )

    result = _run_pilot(BillingTransport(), capsys=capsys)

    assert all(
        probe["billing"]
        == {
            "outcome": "returned",
            "responses": [{"amount": 0.0025, "currency": "USD"}],
        }
        for probe in result["probes"]
    )
    assert "provider_detail" not in json.dumps(result)
    assert "billing-secret-detail" not in json.dumps(result)


def test_pilot_rejected_boundary_keeps_explicit_measurement_schema(capsys) -> None:
    """Break caught: rejected probes make absent observations ambiguous."""

    class BoundaryRejectionTransport(PilotTransport):
        def post(self, url: str, **kwargs) -> JinaHttpResponse:
            request = json.loads(kwargs["body"])
            item = request["input"][0]
            if "image" in item and len(base64.b64decode(item["image"])) == 5_000_000:
                return JinaHttpResponse(
                    413,
                    {"X-RateLimit-Remaining-Requests": "7"},
                    b'{"detail":"boundary-body-secret"}',
                )
            return super().post(url, **kwargs)

    result = _run_pilot(BoundaryRejectionTransport(), capsys=capsys)
    probe = next(
        item
        for item in result["probes"]
        if item["name"] == "image_nominal_5000000_bytes"
    )

    assert probe == {
        "billing": {"outcome": "not_returned"},
        "concurrency_attempted": 1,
        "concurrency_observed": 1,
        "dimensions": [],
        "error": "deterministic_request",
        "name": "image_nominal_5000000_bytes",
        "rate_limit_headers": {"x-ratelimit-remaining-requests": [7.0]},
        "requests": 1,
        "response_models": [],
        "retries": 0,
        "status": "rejected",
        "status_codes": [413],
        "throttles": 0,
        "usage": {},
        "vector_norms": [],
    }
    assert "boundary-body-secret" not in json.dumps(result)


def test_pilot_reports_generated_text_image_mixed_and_boundary_observations(
    capsys,
) -> None:
    """Break caught: pilot skips a modality/boundary or reports unsafe output."""

    transport = PilotTransport()

    exit_code = main(
        ["pilot"],
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=PilotTokenizer(),
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == f"{json.dumps(json.loads(output), sort_keys=True)}\n"
    result = json.loads(output)
    assert result["schema_version"] == "jina-live-pilot-v2"
    assert result["effective_constraints"] == {
        "image_bytes": [
            {"outcome": "succeeded", "value": 5_000_000},
            {"outcome": "succeeded", "value": 8_000_000},
        ],
        "text_tokens": [
            {
                "local_token_count": 8_192,
                "outcome": "succeeded",
                "target": 8_192,
            },
            {
                "local_token_count": 32_768,
                "outcome": "succeeded",
                "target": 32_768,
            },
        ],
    }
    assert [probe["name"] for probe in result["probes"]] == [
        "text_normal",
        "image_normal",
        "mixed_normal",
        "text_target_8192_tokens",
        "text_target_32768_tokens",
        "image_nominal_5000000_bytes",
        "image_nominal_8000000_bytes",
    ]
    assert [probe["dimensions"] for probe in result["probes"]] == [
        [2048],
        [2048],
        [2048, 2048],
        [2048],
        [2048],
        [2048],
        [2048],
    ]
    assert all(probe["vector_norms"] == [5.0] for probe in result["probes"][:2])
    assert [len(request["input"]) for request in transport.requests] == [
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert all(request["truncate"] is False for request in transport.requests)
    assert [
        sorted(item) for request in transport.requests for item in request["input"]
    ] == [
        ["image"],
        ["text"],
        ["text"],
        ["image"],
        ["text"],
        ["text"],
        ["image"],
        ["image"],
        ["text"],
        ["text"],
    ]
    encoded_images = [
        item["image"]
        for request in transport.requests
        for item in request["input"]
        if "image" in item
    ]
    decoded_images = [
        base64.b64decode(image, validate=True) for image in encoded_images
    ]
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in decoded_images)
    assert len(decoded_images[0]) == len(decoded_images[1])
    assert len(decoded_images[0]) < 1_000
    assert [len(image) for image in decoded_images[2:]] == [5_000_000, 8_000_000]


def test_pilot_rejects_selectors_and_missing_credential_without_state(
    capsys, monkeypatch, tmp_path
) -> None:
    """Break caught: pilot accepts corpus input or creates state before auth fails."""

    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["pilot", "--source", "licensed-input"])

    monkeypatch.chdir(tmp_path)
    exit_code = main(["pilot"], environment={})
    output = capsys.readouterr().out

    assert raised.value.code == 2
    assert exit_code == 1
    assert json.loads(output) == {"error": "missing_credential"}
    assert list(tmp_path.iterdir()) == []


def test_pilot_stops_on_invalid_credential_without_creating_state(
    capsys, monkeypatch, tmp_path
) -> None:
    """Break caught: a rejected credential is treated as a successful probe result."""

    class AuthenticationFailureTransport:
        calls = 0

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            body: bytes,
            timeout_seconds: float,
            max_response_bytes: int,
        ) -> JinaHttpResponse:
            del url, headers, body, timeout_seconds, max_response_bytes
            self.calls += 1
            return JinaHttpResponse(status_code=401, headers={}, body=b"{}")

    transport = AuthenticationFailureTransport()
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        ["pilot"],
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=PilotTokenizer(),
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(output) == {"error": "authentication"}
    assert transport.calls == 1
    assert list(tmp_path.iterdir()) == []


def test_normal_generated_png_has_realistic_static_dimensions() -> None:
    """Break caught: the normal hosted probe uses an unusable 1x1 image."""

    image = base64.b64decode(_encoded_png(), validate=True)

    _assert_valid_png(image, padded=False)


def test_generated_image_probes_use_their_patch_token_estimate() -> None:
    """Break caught: 224x224 probes retain the former 1x1 token estimate."""

    adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=PilotTransport(),
        tokenizer=PilotTokenizer(),
    )
    image_inputs = (
        _normal_image_probe().inputs[0],
        *(
            probe.inputs[0]
            for probe in _boundary_probes(adapter)
            if "image" in probe.name
        ),
    )

    assert {item.estimated_tokens for item in image_inputs} == {640}


def test_near_limit_generated_pngs_are_structurally_valid() -> None:
    """Break caught: padded image probes are base64 but not valid PNG streams."""

    for target_size in (5_000_000, 8_000_000):
        image = base64.b64decode(_encoded_png(target_size), validate=True)

        assert len(image) == target_size
        _assert_valid_png(image, padded=True)


@pytest.mark.parametrize(
    ("argv", "option", "rejected_value"),
    [
        (
            ["pilot", "--text", "generated-content-must-not-echo"],
            "--text",
            "generated-content-must-not-echo",
        ),
        (
            ["pilot", "--source=/private/archive/path"],
            "--source",
            "/private/archive/path",
        ),
    ],
)
def test_pilot_parse_errors_redact_rejected_values(
    argv: list[str], option: str, rejected_value: str, capsys
) -> None:
    """Break caught: argparse echoes rejected content or source paths to stderr."""

    with pytest.raises(SystemExit) as raised:
        main(argv)
    captured = capsys.readouterr()

    assert raised.value.code == 2
    assert option in captured.err
    assert rejected_value not in captured.out
    assert rejected_value not in captured.err


def _assert_valid_png(image: bytes, *, padded: bool) -> None:
    """Validate framing, CRCs, optional padding, and 224x224 RGB pixels."""

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    chunks: list[tuple[bytes, bytes]] = []
    while position < len(image):
        size = struct.unpack(">I", image[position : position + 4])[0]
        kind = image[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + size
        value = image[data_start:data_end]
        crc = struct.unpack(">I", image[data_end : data_end + 4])[0]
        assert zlib.crc32(kind + value) & 0xFFFFFFFF == crc
        chunks.append((kind, value))
        position = data_end + 4

    assert position == len(image)
    expected_kinds = [b"IHDR", b"IDAT", *([b"tEXt"] if padded else []), b"IEND"]
    assert [kind for kind, _ in chunks] == expected_kinds
    assert chunks[0][1] == struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0)
    if padded:
        keyword, text = chunks[2][1].split(b"\x00", 1)
        assert keyword == b"p"
        assert text
    pixels = zlib.decompress(chunks[1][1])
    row_size = 1 + 224 * 3
    assert len(pixels) == 224 * row_size
    assert all(pixels[row * row_size] == 0 for row in range(224))
    assert len(set(pixels)) > 2
