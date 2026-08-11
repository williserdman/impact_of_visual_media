"""Synthetic contract tests for the hosted Jina v4 adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import traceback
from collections.abc import Mapping
from pathlib import Path

import pytest

from wsj_embeddings.adapters import (
    JinaEmbeddingAdapter,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
    JinaHttpResponse,
)
from wsj_embeddings.tokenizer import PinnedJinaV4Tokenizer, PinnedTokenizerError


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


class SyntheticEncoding:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self.offsets = offsets


class SyntheticLoadedTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> SyntheticEncoding:
        assert add_special_tokens is False
        return SyntheticEncoding([(index, index + 1) for index in range(len(text))])


def test_pinned_tokenizer_resolves_immutable_official_artifact_and_checks_bytes(
    tmp_path,
):
    """Break caught: runtime tokenization follows mutable main or unchecked bytes."""

    artifact = tmp_path / "tokenizer.json"
    artifact.write_bytes(b"generated wrong tokenizer bytes")
    resolution_calls: list[tuple[str, str, str]] = []
    loader_calls: list[Path] = []

    def resolve(repo_id: str, filename: str, revision: str) -> Path:
        resolution_calls.append((repo_id, filename, revision))
        return artifact

    tokenizer = PinnedJinaV4Tokenizer(
        resolver=resolve,
        loader=lambda path: loader_calls.append(path),
    )

    with pytest.raises(PinnedTokenizerError, match="checksum"):
        tokenizer.token_offsets("generated")

    assert resolution_calls == [
        (
            "jinaai/jina-embeddings-v4",
            "tokenizer.json",
            "d1e5d70b7b34d927a8cddac458583c4fbe50a914",
        )
    ]
    assert loader_calls == []


def test_verified_pinned_tokenizer_exposes_exact_offsets_without_network(tmp_path):
    """Break caught: verified tokenizer offsets are discarded or add special tokens."""

    artifact = tmp_path / "tokenizer.json"
    artifact_bytes = b"generated tokenizer fixture"
    artifact.write_bytes(artifact_bytes)
    loaded = SyntheticLoadedTokenizer()
    loaded_serialized: list[str] = []

    def load_verified(serialized: str) -> SyntheticLoadedTokenizer:
        loaded_serialized.append(serialized)
        artifact.write_bytes(b"generated replacement after verification")
        return loaded

    tokenizer = PinnedJinaV4Tokenizer(
        resolver=lambda _repo, _filename, _revision: artifact,
        loader=load_verified,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )

    assert tokenizer.token_offsets("abc") == ((0, 1), (1, 2), (2, 3))
    assert loaded_serialized == [artifact_bytes.decode("utf-8")]


def test_hosted_adapter_uses_injected_local_tokenizer_without_transport():
    """Break caught: context planning sends a hosted request or ignores its pin."""

    class RecordingTokenizer:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
            self.inputs.append(text)
            return ((0, len(text)),)

    tokenizer = RecordingTokenizer()
    transport = RecordingTransport(AssertionError("network transport used"))
    adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
        tokenizer=tokenizer,
    )

    assert adapter.token_offsets("generated") == ((0, 9),)
    assert tokenizer.inputs == ["generated"]
    assert transport.calls == []


def _embedding(first: float, second: float) -> list[float]:
    return [first, second, *([0.0] * 2046)]


def _response(
    data: list[dict[str, object]],
    *,
    status: int = 200,
    model: str = "jina-embeddings-v4",
    headers: Mapping[str, str] | None = None,
) -> JinaHttpResponse:
    return JinaHttpResponse(
        status_code=status,
        headers=(
            {"x-ratelimit-remaining-requests": "499"}
            if headers is None
            else headers
        ),
        body=json.dumps(
            {
                "model": model,
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
        "x-ratelimit-remaining-requests": 499.0
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


def test_hosted_adapter_embeds_one_already_encoded_image_without_remote_url():
    """Break caught: coordinator image calls use text or publish a remote URL."""

    image_base64 = base64.b64encode(b"generated image bytes").decode("ascii")
    transport = RecordingTransport(
        _response([{"index": 0, "embedding": _embedding(0.0, 5.0)}])
    )

    vector = _adapter(transport).embed_image(image_base64)

    assert vector[:2] == (0.0, 1.0)
    assert json.loads(transport.calls[0][2])["input"] == [{"image": image_base64}]


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


@pytest.mark.parametrize(
    ("transport", "input_item", "secret"),
    [
        (
            RecordingTransport(OSError("transport-secret-123")),
            JinaEmbeddingInput.text("generated test text"),
            "transport-secret-123",
        ),
        (
            RecordingTransport(
                JinaHttpResponse(
                    200,
                    {},
                    b'{"response-secret-456":',
                )
            ),
            JinaEmbeddingInput.text("generated test text"),
            "response-secret-456",
        ),
        (
            RecordingTransport(
                _response([{"index": 0, "embedding": _embedding(1, 0)}])
            ),
            JinaEmbeddingInput.image_base64("image-secret-789!"),
            "image-secret-789",
        ),
    ],
)
def test_hosted_adapter_suppresses_unsafe_exception_causes(
    transport: RecordingTransport,
    input_item: JinaEmbeddingInput,
    secret: str,
):
    """Break caught: exception chaining exposes request or response content."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        _adapter(transport).embed((input_item,))

    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_hosted_adapter_rejects_untrusted_model_metadata():
    """Break caught: an arbitrary server model label reaches durable metadata."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        _adapter(
            RecordingTransport(
                _response(
                    [{"index": 0, "embedding": _embedding(1, 0)}],
                    model="model-secret-123",
                )
            )
        ).embed((JinaEmbeddingInput.text("generated test text"),))

    assert raised.value.code == "invalid_response"
    assert "model-secret-123" not in "".join(traceback.format_exception(raised.value))


def test_hosted_adapter_retains_only_numeric_named_rate_limit_metadata():
    """Break caught: arbitrary rate-limit header strings become metadata."""

    result = _adapter(
        RecordingTransport(
            _response(
                [{"index": 0, "embedding": _embedding(1, 0)}],
                headers={
                    "X-RateLimit-Remaining-Requests": "499",
                    "Retry-After": "3.5",
                    "X-RateLimit-Remaining-Tokens": "header-secret-123",
                    "X-Unrelated": "header-secret-456",
                },
            )
        )
    ).embed((JinaEmbeddingInput.text("generated test text"),))

    assert result.response_metadata.rate_limit_headers == {
        "x-ratelimit-remaining-requests": 499.0,
        "retry-after": 3.5,
    }


@pytest.mark.parametrize("invalid_value", [True, "1.0"])
def test_hosted_adapter_rejects_non_json_numeric_vector_values(invalid_value: object):
    """Break caught: booleans or numeric strings are accepted as embeddings."""

    with pytest.raises(JinaHostedAdapterError) as raised:
        _adapter(
            RecordingTransport(
                _response(
                    [
                        {
                            "index": 0,
                            "embedding": [invalid_value, *([0.0] * 2047)],
                        }
                    ]
                )
            )
        ).embed((JinaEmbeddingInput.text("generated test text"),))

    assert raised.value.code == "invalid_response"
