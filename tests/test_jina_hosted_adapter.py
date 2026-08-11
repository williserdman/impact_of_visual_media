"""Synthetic contract tests for the hosted Jina v4 adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from collections.abc import Mapping

import pytest

from wsj_embeddings.adapters import (
    JinaEmbeddingAdapter,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
    JinaHttpResponse,
)


class RecordingTransport:
    """Synthetic exchange recorder; it never makes an HTTP request."""

    def __init__(self, response: JinaHttpResponse | BaseException) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, str], bytes, float]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> JinaHttpResponse:
        self.calls.append((url, headers, body, timeout_seconds))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _embedding(first: float, second: float) -> list[float]:
    return [first, second, *([0.0] * 2046)]


def _response(
    data: list[dict[str, object]],
    *,
    status: int = 200,
) -> JinaHttpResponse:
    return JinaHttpResponse(
        status_code=status,
        headers={"x-ratelimit-remaining-requests": "499"},
        body=json.dumps(
            {
                "model": "jina-embeddings-v4",
                "data": data,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
            }
        ).encode(),
    )


def _adapter(transport: RecordingTransport) -> JinaEmbeddingAdapter:
    return JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        timeout_seconds=12.5,
    )


def test_hosted_adapter_serializes_fixed_contract_and_maps_indexed_vectors():
    """Break caught: a changed request or positional rather than index mapping."""

    transport = RecordingTransport(
        _response(
            [
                {"index": 1, "embedding": _embedding(0.0, 4.0)},
                {"index": 0, "embedding": _embedding(3.0, 0.0)},
            ]
        )
    )

    result = _adapter(transport).embed(
        (
            JinaEmbeddingInput.text("generated test text"),
            JinaEmbeddingInput.image_base64(base64.b64encode(b"image").decode()),
        )
    )

    assert len(transport.calls) == 1
    url, headers, body, timeout_seconds = transport.calls[0]
    assert url == "https://api.jina.ai/v1/embeddings"
    assert headers == {
        "Authorization": "Bearer synthetic-secret",
        "Content-Type": "application/json",
    }
    assert timeout_seconds == 12.5
    assert json.loads(body) == {
        "model": "jina-embeddings-v4",
        "task": "retrieval.passage",
        "dimensions": 2048,
        "embedding_type": "float",
        "truncate": False,
        "return_multivector": False,
        "input": [
            {"text": "generated test text"},
            {"image": base64.b64encode(b"image").decode()},
        ],
    }
    assert tuple(vector.index for vector in result.vectors) == (0, 1)
    assert result.vectors[0].stored_vector[:2] == (1.0, 0.0)
    assert result.vectors[1].stored_vector[:2] == (0.0, 1.0)
    assert result.usage == {"prompt_tokens": 11, "total_tokens": 11}
    assert result.response_metadata.status_code == 200
    assert result.response_metadata.rate_limit_headers == {
        "x-ratelimit-remaining-requests": "499"
    }
    expected_raw_hash = hashlib.sha256(
        json.dumps(_embedding(3.0, 0.0), separators=(",", ":")).encode()
    ).hexdigest()
    assert result.vectors[0].raw_response_sha256 == expected_raw_hash
    expected_stored_hash = hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in result.vectors[0].stored_vector)
    ).hexdigest()
    assert result.vectors[0].stored_vector_sha256 == expected_stored_hash
    assert math.isclose(
        math.sqrt(sum(value * value for value in result.vectors[0].stored_vector)),
        1.0,
        abs_tol=1e-6,
    )


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ([{"index": 0, "embedding": _embedding(1.0, 0.0)}], "invalid_response"),
        (
            [
                {"index": 0, "embedding": _embedding(1.0, 0.0)},
                {"index": 2, "embedding": _embedding(0.0, 1.0)},
            ],
            "invalid_response",
        ),
        (
            [
                {"index": 0, "embedding": _embedding(1.0, 0.0)[:-1]},
                {"index": 1, "embedding": _embedding(0.0, 1.0)},
            ],
            "invalid_response",
        ),
        (
            [
                {"index": 0, "embedding": [float("nan"), *([0.0] * 2047)]},
                {"index": 1, "embedding": _embedding(0.0, 1.0)},
            ],
            "invalid_response",
        ),
        (
            [
                {"index": 0, "embedding": [0.0] * 2048},
                {"index": 1, "embedding": _embedding(0.0, 1.0)},
            ],
            "invalid_response",
        ),
    ],
)
def test_hosted_adapter_rejects_malformed_vectors_before_returning_them(
    data: list[dict[str, object]], code: str
):
    """Break caught: malformed endpoint output reaches the coordinator."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        _adapter(RecordingTransport(_response(data))).embed(
            (
                JinaEmbeddingInput.text("generated test text"),
                JinaEmbeddingInput.text("two"),
            )
        )

    assert raised.value.code == code
    assert not raised.value.retryable


@pytest.mark.parametrize(
    ("response", "code", "retryable", "retry_after_seconds"),
    [
        (_response([], status=401), "authentication", False, None),
        (_response([], status=403), "authorization", False, None),
        (_response([], status=400), "deterministic_request", False, None),
        (
            JinaHttpResponse(429, {"Retry-After": "3"}, b'{"detail":"secret"}'),
            "rate_limit",
            True,
            3.0,
        ),
        (_response([], status=503), "transient_server", True, None),
        (TimeoutError(), "timeout", True, None),
        (OSError(), "connection", True, None),
    ],
)
def test_hosted_adapter_classifies_safe_transport_failures(
    response: JinaHttpResponse | BaseException,
    code: str,
    retryable: bool,
    retry_after_seconds: float | None,
):
    """Break caught: later run policy cannot distinguish retryable failures."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        _adapter(RecordingTransport(response)).embed((JinaEmbeddingInput.text("private"),))

    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert raised.value.retry_after_seconds == retry_after_seconds
    assert "synthetic-secret" not in str(raised.value)
    assert "private" not in str(raised.value)
    assert "detail" not in str(raised.value)


def test_hosted_adapter_requires_only_named_environment_credential():
    """Break caught: a missing credential fails late or leaks environment details."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        JinaEmbeddingAdapter(environment={"UNRELATED_SECRET": "nope"})

    assert raised.value.code == "missing_credential"
    assert raised.value.retryable is False
    assert "UNRELATED_SECRET" not in str(raised.value)
