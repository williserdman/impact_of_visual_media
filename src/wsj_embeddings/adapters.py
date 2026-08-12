"""Embedding adapters, including a safely injectable hosted Jina client."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from wsj_embeddings.image_rendition import (
    PRODUCTION_IMAGE_INPUT_RULES,
    PRODUCTION_IMAGE_TRANSFORM_ID,
    FixturePassthroughImageCodec,
)
from wsj_embeddings.long_text import TextOffsetTokenizer
from wsj_embeddings.models import EmbeddingProfile
from wsj_embeddings.tokenizer import (
    JINA_V4_TOKENIZER_REVISION,
    JINA_V4_TOKENIZER_SHA256,
    PinnedJinaV4Tokenizer,
)

_JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v4"
_JINA_TASK = "retrieval.passage"
_JINA_DIMENSIONS = 2048
_CONSERVATIVE_CONTEXT_TOKEN_LIMIT = 8_000
_TOKENIZER_IDENTITY = (
    "jinaai/jina-embeddings-v4@"
    f"{JINA_V4_TOKENIZER_REVISION}:tokenizer.json#sha256={JINA_V4_TOKENIZER_SHA256}"
)
_SAFE_USAGE_FIELDS = frozenset(
    {"input_tokens", "output_tokens", "prompt_tokens", "total_tokens"}
)
_SAFE_RATE_LIMIT_HEADERS = frozenset(
    {
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    }
)
_MAX_SAFE_RATE_LIMIT_VALUE = 1_000_000_000_000_000.0


class EmbeddingAdapter(Protocol):
    """Encodes one complete canonical article text without network policy."""

    @property
    def profile(self) -> EmbeddingProfile: ...

    def embed_text(self, text: str) -> tuple[float, ...]: ...

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]: ...

    def embed_image(self, image_base64: str) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class JinaEmbeddingInput:
    """One text or already-base64-encoded image request item."""

    kind: str
    value: str

    @classmethod
    def text(cls, value: str) -> JinaEmbeddingInput:
        return cls("text", value)

    @classmethod
    def image_base64(cls, value: str) -> JinaEmbeddingInput:
        return cls("image", value)

    def as_request_item(self) -> dict[str, str]:
        if self.kind == "text":
            if not isinstance(self.value, str) or not self.value:
                raise JinaHostedAdapterError("deterministic_request", retryable=False)
            return {"text": self.value}
        if self.kind == "image":
            if not isinstance(self.value, str) or not self.value:
                raise JinaHostedAdapterError("deterministic_request", retryable=False)
            try:
                base64.b64decode(self.value, validate=True)
            except (ValueError, TypeError):
                raise JinaHostedAdapterError(
                    "deterministic_request", retryable=False
                ) from None
            return {"image": self.value}
        raise JinaHostedAdapterError("deterministic_request", retryable=False)


@dataclass(frozen=True, slots=True)
class JinaHttpResponse:
    """The content-free transport result consumed by the hosted adapter."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


class JinaTransport(Protocol):
    """Minimal HTTP seam; tests inject a synthetic implementation here."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> JinaHttpResponse: ...


class UrllibJinaTransport:
    """Production HTTPS implementation, intentionally absent from tests."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> JinaHttpResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return JinaHttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return JinaHttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers else {},
                body=error.read(),
            )


@dataclass(frozen=True, slots=True)
class JinaResponseMetadata:
    """Safe response facts that later run policy may persist or summarize."""

    status_code: int
    model: str
    rate_limit_headers: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class JinaEmbeddedVector:
    """One validated response vector and its audit hashes."""

    index: int
    raw_vector: tuple[float, ...]
    stored_vector: tuple[float, ...]
    raw_response_sha256: str
    stored_vector_sha256: str


@dataclass(frozen=True, slots=True)
class JinaEmbeddingResponse:
    """Validated, content-free hosted response metadata and vectors."""

    vectors: tuple[JinaEmbeddedVector, ...]
    usage: dict[str, int | float]
    response_metadata: JinaResponseMetadata


class JinaHostedAdapterError(RuntimeError):
    """A content-free classified hosted-adapter failure."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        suffix = "" if status_code is None else f" status={status_code}"
        super().__init__(f"hosted Jina embedding request failed: {code}{suffix}")


class JinaEmbeddingAdapter:
    """Synchronously call the fixed Jina v4 endpoint through an injected transport."""

    profile = EmbeddingProfile(
        model=_JINA_MODEL,
        task=_JINA_TASK,
        dimensions=_JINA_DIMENSIONS,
        output_type="float",
        normalization="l2-client-float32-v1",
        observed_model=_JINA_MODEL,
        observed_api_version="2026.07.27.1603",
        tokenizer_revision=_TOKENIZER_IDENTITY,
        context_token_limit=_CONSERVATIVE_CONTEXT_TOKEN_LIMIT,
        context_rules="markdown-block-greedy-no-overlap-truncate-false-v1",
        long_text_aggregation="l2-token-count-weighted-mean-float32-v1",
        image_input_rules=PRODUCTION_IMAGE_INPUT_RULES,
        image_transform=PRODUCTION_IMAGE_TRANSFORM_ID,
        multimodal_formula="l2-normalize-0.5-text-0.5-image-v1",
        client_configuration_version="wsj-embeddings-config-v3",
    )

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        transport: JinaTransport | None = None,
        tokenizer: TextOffsetTokenizer | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        credential_environment = os.environ if environment is None else environment
        api_key = credential_environment.get("JINA_API_KEY")
        if not api_key:
            raise JinaHostedAdapterError("missing_credential", retryable=False)
        self._authorization = f"Bearer {api_key}"
        self._transport = UrllibJinaTransport() if transport is None else transport
        self._tokenizer = PinnedJinaV4Tokenizer() if tokenizer is None else tokenizer
        self._timeout_seconds = timeout_seconds

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Tokenize locally with the immutable checksum-verified v4 artifact."""

        return self._tokenizer.token_offsets(text)

    def embed_text(self, text: str) -> tuple[float, ...]:
        """Maintain the existing coordinator adapter seam for text-only callers."""

        return self.embed((JinaEmbeddingInput.text(text),)).vectors[0].stored_vector

    def embed_image(self, image_base64: str) -> tuple[float, ...]:
        """Embed source bytes already encoded by the trusted coordinator."""

        return self.embed(
            (JinaEmbeddingInput.image_base64(image_base64),)
        ).vectors[0].stored_vector

    def embed(self, inputs: Sequence[JinaEmbeddingInput]) -> JinaEmbeddingResponse:
        """Embed text/image items and return vectors in caller input order."""

        request_items = tuple(item.as_request_item() for item in inputs)
        if not request_items:
            raise JinaHostedAdapterError("deterministic_request", retryable=False)
        body = json.dumps(
            {
                "model": _JINA_MODEL,
                "task": _JINA_TASK,
                "dimensions": _JINA_DIMENSIONS,
                "embedding_type": "float",
                "truncate": False,
                "return_multivector": False,
                "input": request_items,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        response = self._send(body)
        if not 200 <= response.status_code < 300:
            self._raise_for_status(response)
        payload = self._decode_success(response)
        vectors = _validated_vectors(payload, expected_count=len(request_items))
        model = payload.get("model")
        if model != _JINA_MODEL:
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        return JinaEmbeddingResponse(
            vectors=vectors,
            usage=_safe_usage(payload.get("usage")),
            response_metadata=JinaResponseMetadata(
                status_code=response.status_code,
                model=model,
                rate_limit_headers=_safe_rate_limit_headers(response.headers),
            ),
        )

    def _send(self, body: bytes) -> JinaHttpResponse:
        try:
            return self._transport.post(
                _JINA_EMBEDDINGS_URL,
                headers={
                    "Authorization": self._authorization,
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_seconds=self._timeout_seconds,
            )
        except TimeoutError:
            raise JinaHostedAdapterError("timeout", retryable=True) from None
        except OSError:
            raise JinaHostedAdapterError("connection", retryable=True) from None

    @staticmethod
    def _decode_success(response: JinaHttpResponse) -> Mapping[str, object]:
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        if not isinstance(payload, dict):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        return payload

    @staticmethod
    def _raise_for_status(response: JinaHttpResponse) -> None:
        status = response.status_code
        if status == 401:
            code, retryable = "authentication", False
        elif status == 403:
            code, retryable = "authorization", False
        elif status in {400, 404, 409, 413, 422}:
            code, retryable = "deterministic_request", False
        elif status == 429:
            code, retryable = "rate_limit", True
        elif 500 <= status < 600:
            code, retryable = "transient_server", True
        else:
            code, retryable = "request_failure", False
        raise JinaHostedAdapterError(
            code,
            retryable=retryable,
            status_code=status,
            retry_after_seconds=_retry_after_seconds(response.headers),
        )


def _validated_vectors(
    payload: Mapping[str, object], *, expected_count: int
) -> tuple[JinaEmbeddedVector, ...]:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise JinaHostedAdapterError("invalid_response", retryable=False)
    by_index: dict[int, JinaEmbeddedVector] = {}
    for item in data:
        if not isinstance(item, dict):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        index = item.get("index")
        raw_embedding = item.get("embedding")
        if isinstance(index, bool) or not isinstance(index, int):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        if index < 0 or index >= expected_count or index in by_index:
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        if (
            not isinstance(raw_embedding, list)
            or len(raw_embedding) != _JINA_DIMENSIONS
        ):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_embedding
        ):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        try:
            raw_vector = tuple(float(value) for value in raw_embedding)
        except (OverflowError, TypeError, ValueError):
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        if not all(math.isfinite(value) for value in raw_vector):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        norm = math.sqrt(sum(value * value for value in raw_vector))
        if not math.isfinite(norm) or norm == 0:
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        try:
            stored_vector = tuple(
                struct.unpack("<f", struct.pack("<f", value / norm))[0]
                for value in raw_vector
            )
            packed_stored = b"".join(
                struct.pack("<f", value) for value in stored_vector
            )
            raw_representation = json.dumps(
                raw_embedding, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (OverflowError, TypeError, ValueError, struct.error):
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        by_index[index] = JinaEmbeddedVector(
            index=index,
            raw_vector=raw_vector,
            stored_vector=stored_vector,
            raw_response_sha256=hashlib.sha256(raw_representation).hexdigest(),
            stored_vector_sha256=hashlib.sha256(packed_stored).hexdigest(),
        )
    if set(by_index) != set(range(expected_count)):
        raise JinaHostedAdapterError("invalid_response", retryable=False)
    return tuple(by_index[index] for index in range(expected_count))


def _safe_usage(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    usage: dict[str, int | float] = {}
    for key in _SAFE_USAGE_FIELDS:
        count = value.get(key)
        number = _safe_numeric_value(count, maximum=None)
        if number is not None:
            usage[key] = count
    return usage


def _safe_rate_limit_headers(headers: Mapping[str, str]) -> dict[str, float]:
    safe_headers: dict[str, float] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        normalized_name = name.lower()
        if normalized_name not in _SAFE_RATE_LIMIT_HEADERS:
            continue
        number = _safe_numeric_value(value, maximum=_MAX_SAFE_RATE_LIMIT_VALUE)
        if number is not None:
            safe_headers[normalized_name] = number
    return safe_headers


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if not isinstance(name, str) or name.lower() != "retry-after":
            continue
        return _safe_numeric_value(value, maximum=_MAX_SAFE_RATE_LIMIT_VALUE)
    return None


def _safe_numeric_value(value: object, *, maximum: float | None) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
    elif isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
    else:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


class FakeEmbeddingAdapter:
    """Deterministic generated-fixture encoder for the first smoke slice."""

    profile = EmbeddingProfile(
        model="fake-jina-embeddings-v4",
        task="retrieval.passage",
        dimensions=2048,
        output_type="float",
        normalization="l2-client-float32-v1",
        observed_model="fake-jina-embeddings-v4",
        observed_api_version="synthetic-v1",
        tokenizer_revision="synthetic-jina-v4-tokenizer-v1",
        tokenizer_engine="synthetic-codepoint-tokenizer-v1",
        context_token_limit=8_000,
        context_rules="markdown-block-greedy-no-overlap-truncate-false-v1",
        long_text_aggregation="l2-token-count-weighted-mean-float32-v1",
        image_input_rules=PRODUCTION_IMAGE_INPUT_RULES,
        image_transform=PRODUCTION_IMAGE_TRANSFORM_ID,
        multimodal_formula="l2-normalize-0.5-text-0.5-image-v1",
        client_configuration_version="wsj-embeddings-config-v3",
    )
    image_codec = FixturePassthroughImageCodec()

    def embed_text(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("text input must not be empty")
        return (1.0, *(0.0 for _ in range(2047)))

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Tokenize generated fixtures as one exact Unicode codepoint each."""

        return tuple((index, index + 1) for index in range(len(text)))

    def embed_image(self, image_base64: str) -> tuple[float, ...]:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except (TypeError, ValueError):
            image_bytes = b""
        if not image_bytes:
            raise ValueError("image input must contain base64 source bytes")
        return (0.0, 1.0, *(0.0 for _ in range(2046)))
