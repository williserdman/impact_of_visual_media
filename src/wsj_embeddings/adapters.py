"""Embedding adapters, including a safely injectable hosted Jina client."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

from wsj_embeddings.image_rendition import (
    PRODUCTION_IMAGE_INPUT_RULES,
    PRODUCTION_IMAGE_TRANSFORM_ID,
    FixturePassthroughImageCodec,
    ImageCodec,
    ImageCodecError,
)
from wsj_embeddings.long_text import TextOffsetTokenizer
from wsj_embeddings.models import EmbeddingProfile
from wsj_embeddings.tokenizer import (
    JINA_V4_TOKENIZER_REVISION,
    JINA_V4_TOKENIZER_SHA256,
    PinnedJinaV4Tokenizer,
)

_JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODELS_URL = "https://api.jina.ai/v1/models"
_JINA_OPENAPI_URL = "https://api.jina.ai/openapi.json"
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
_SAFE_BILLING_FIELDS = frozenset(
    {"amount", "charged_tokens", "cost", "credits", "total"}
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
_SAFE_JINA_MODEL_LABEL = re.compile(
    r"^jina-embeddings-v4(?:-[a-z0-9][a-z0-9._-]{0,95})?$"
)
_UNSAFE_MODEL_COMPONENTS = frozenset(
    {"bearer", "credential", "key", "password", "secret", "token"}
)
_SAFE_CURRENCY = re.compile(r"^[A-Z]{3}$")


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
    token_estimate: int

    @classmethod
    def text(
        cls, value: str, *, estimated_tokens: int | None = None
    ) -> JinaEmbeddingInput:
        estimate = len(value) if estimated_tokens is None else estimated_tokens
        if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
            raise ValueError("estimated tokens must be a nonnegative integer")
        return cls("text", value, estimate)

    @classmethod
    def image_base64(
        cls, value: str, *, estimated_tokens: int = 0
    ) -> JinaEmbeddingInput:
        if (
            isinstance(estimated_tokens, bool)
            or not isinstance(estimated_tokens, int)
            or estimated_tokens < 0
        ):
            raise ValueError("estimated tokens must be a nonnegative integer")
        return cls("image", value, estimated_tokens)

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

    @property
    def estimated_tokens(self) -> int:
        """Return a conservative content-free text budget estimate."""

        return self.token_estimate

    @property
    def encoded_bytes(self) -> int:
        """Return UTF-8 bytes contributed by this encoded request value."""

        return len(self.value.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class JinaBatchLimits:
    """Independent hard ceilings for one synchronous hosted request."""

    max_items: int
    max_estimated_tokens: int
    max_encoded_bytes: int
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.max_items,
                self.max_estimated_tokens,
                self.max_encoded_bytes,
                self.max_response_bytes,
            )
        ):
            raise ValueError("batch limits must be positive integers")


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
        max_response_bytes: int,
    ) -> JinaHttpResponse: ...

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse: ...


class _JinaResponseTooLarge(RuntimeError):
    """Internal content-free signal from the bounded HTTP reader."""


def _read_bounded_body(stream: object, *, max_response_bytes: int) -> bytes:
    """Read at most the configured body ceiling plus one detection byte."""

    chunks: list[bytes] = []
    received = 0
    while received <= max_response_bytes:
        chunk = stream.read(min(65_536, max_response_bytes + 1 - received))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        received += len(chunk)
    raise _JinaResponseTooLarge


class UrllibJinaTransport:
    """Production HTTPS implementation, intentionally absent from tests."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse:
        return self._request(
            url,
            method="POST",
            headers=headers,
            body=body,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse:
        return self._request(
            url,
            method="GET",
            headers=headers,
            body=None,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )

    @staticmethod
    def _request(
        url: str,
        *,
        method: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> JinaHttpResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return JinaHttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=_read_bounded_body(
                        response, max_response_bytes=max_response_bytes
                    ),
                )
        except urllib.error.HTTPError as error:
            return JinaHttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers else {},
                body=_read_bounded_body(
                    error, max_response_bytes=max_response_bytes
                ),
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
    response_model: str | None = None


@dataclass(frozen=True, slots=True)
class JinaEmbeddingResponse:
    """Validated, content-free hosted response metadata and vectors."""

    vectors: tuple[JinaEmbeddedVector, ...]
    usage: dict[str, int | float]
    response_metadata: JinaResponseMetadata
    billing: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JinaBatchItemOutcome:
    """One input-indexed success or content-free response error."""

    index: int
    vector: JinaEmbeddedVector | None
    error_code: str | None
    retryable: bool = False
    status_code: int | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class JinaEmbeddingBatchResponse:
    """Independent outcomes and shared safe metadata for one hosted call."""

    items: tuple[JinaBatchItemOutcome, ...]
    usage: dict[str, int | float]
    response_metadata: JinaResponseMetadata
    billing: dict[str, int | float | str] = field(default_factory=dict)


class JinaHostedAdapterError(RuntimeError):
    """A content-free classified hosted-adapter failure."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        rate_limit_headers: Mapping[str, float] | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_headers = (
            {} if rate_limit_headers is None else dict(rate_limit_headers)
        )
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
        client_api_contract_version="openapi-2026.07.27.1603",
        tokenizer_revision=_TOKENIZER_IDENTITY,
        context_token_limit=_CONSERVATIVE_CONTEXT_TOKEN_LIMIT,
        context_rules="markdown-block-greedy-no-overlap-truncate-false-v1",
        long_text_aggregation="l2-token-count-weighted-mean-float32-v1",
        image_input_rules=PRODUCTION_IMAGE_INPUT_RULES,
        image_transform=PRODUCTION_IMAGE_TRANSFORM_ID,
        multimodal_formula="l2-normalize-0.5-text-0.5-image-v1",
        batch_max_response_bytes=2_000_000,
        client_configuration_version="wsj-embeddings-config-v6",
    )
    rate_aware_batching = True

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        transport: JinaTransport | None = None,
        tokenizer: TextOffsetTokenizer | None = None,
        observed_model: str | None = None,
        timeout_seconds: float = 30.0,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        credential_environment = os.environ if environment is None else environment
        api_key = credential_environment.get("JINA_API_KEY")
        if not api_key:
            raise JinaHostedAdapterError("missing_credential", retryable=False)
        if observed_model is not None and not is_safe_jina_model_label(observed_model):
            raise JinaHostedAdapterError("pilot_observation_required", retryable=False)
        self.profile = replace(self.__class__.profile, observed_model=observed_model)
        self._authorization = f"Bearer {api_key}"
        self._transport = UrllibJinaTransport() if transport is None else transport
        self._tokenizer = PinnedJinaV4Tokenizer() if tokenizer is None else tokenizer
        self._timeout_seconds = timeout_seconds
        self._wall_clock = wall_clock

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Tokenize locally with the immutable checksum-verified v4 artifact."""

        return self._tokenizer.token_offsets(text)

    def fetch_openapi_document(
        self, *, max_response_bytes: int = 4_000_000
    ) -> tuple[Mapping[str, object], int]:
        """Fetch the fixed public OpenAPI document through the bounded transport."""

        return self._fetch_metadata_document(
            _JINA_OPENAPI_URL,
            headers={"Accept": "application/json"},
            max_response_bytes=max_response_bytes,
        )

    def fetch_model_catalogue(
        self, *, max_response_bytes: int = 1_000_000
    ) -> tuple[Mapping[str, object], int]:
        """Fetch the fixed authenticated model catalogue through the transport."""

        return self._fetch_metadata_document(
            _JINA_MODELS_URL,
            headers={
                "Accept": "application/json",
                "Authorization": self._authorization,
            },
            max_response_bytes=max_response_bytes,
        )

    def bind_image_codec(self, codec: ImageCodec) -> None:
        """Bind runtime encoder-build meaning before configuration publication."""

        if (
            codec.input_rules != PRODUCTION_IMAGE_INPUT_RULES
            or not codec.transform_id.startswith(
                f"{PRODUCTION_IMAGE_TRANSFORM_ID}-build-"
            )
        ):
            raise ImageCodecError("ambiguous_image_configuration")
        existing = getattr(self, "image_codec", None)
        if existing is not None and (
            existing.input_rules != codec.input_rules
            or existing.transform_id != codec.transform_id
        ):
            raise ImageCodecError("ambiguous_image_configuration")
        self.image_codec = codec
        self.profile = replace(
            self.profile,
            image_transform=codec.transform_id,
        )

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

        inputs = tuple(inputs)
        batch = self.embed_batch(
            inputs,
            limits=JinaBatchLimits(
                max_items=max(1, len(inputs)),
                max_estimated_tokens=max(
                    1, sum(item.estimated_tokens for item in inputs)
                ),
                max_encoded_bytes=max(
                    1, sum(item.encoded_bytes for item in inputs)
                ),
                max_response_bytes=self.profile.batch_max_response_bytes,
            ),
        )
        if any(item.vector is None for item in batch.items):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        return JinaEmbeddingResponse(
            vectors=tuple(
                item.vector for item in batch.items if item.vector is not None
            ),
            usage=batch.usage,
            response_metadata=batch.response_metadata,
            billing=batch.billing,
        )

    def embed_batch(
        self,
        inputs: Sequence[JinaEmbeddingInput],
        *,
        limits: JinaBatchLimits,
    ) -> JinaEmbeddingBatchResponse:
        """Send one bounded mixed request and preserve unambiguous item outcomes."""

        request_items = tuple(item.as_request_item() for item in inputs)
        if not request_items:
            raise JinaHostedAdapterError("deterministic_request", retryable=False)
        if len(request_items) > limits.max_items:
            raise JinaHostedAdapterError("batch_item_limit", retryable=False)
        if sum(item.estimated_tokens for item in inputs) > limits.max_estimated_tokens:
            raise JinaHostedAdapterError("batch_token_limit", retryable=False)
        if (
            sum(item.encoded_bytes for item in inputs)
            > limits.max_encoded_bytes
        ):
            raise JinaHostedAdapterError(
                "batch_encoded_byte_limit", retryable=False
            )
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
        response = self._send(
            body,
            max_response_bytes=min(
                limits.max_response_bytes,
                self.profile.batch_max_response_bytes,
            ),
        )
        if not 200 <= response.status_code < 300:
            self._raise_for_status(response)
        payload = self._decode_success(response)
        model = _safe_response_model(payload.get("model"))
        if model is None:
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        rate_headers = _safe_rate_limit_headers(response.headers)
        retry_after = _retry_after_seconds(
            response.headers, now=self._wall_clock()
        )
        if retry_after is not None:
            rate_headers["retry-after"] = retry_after
        return JinaEmbeddingBatchResponse(
            items=_validated_batch_items(
                payload,
                expected_count=len(request_items),
                status_code=response.status_code,
                retry_after_seconds=retry_after,
                response_model=model,
            ),
            usage=_safe_usage(payload.get("usage")),
            response_metadata=JinaResponseMetadata(
                status_code=response.status_code,
                model=model,
                rate_limit_headers=rate_headers,
            ),
            billing=_safe_billing(payload.get("billing")),
        )

    def _send(self, body: bytes, *, max_response_bytes: int) -> JinaHttpResponse:
        try:
            return self._transport.post(
                _JINA_EMBEDDINGS_URL,
                headers={
                    "Authorization": self._authorization,
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        except _JinaResponseTooLarge:
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        except TimeoutError:
            raise JinaHostedAdapterError("timeout", retryable=True) from None
        except OSError:
            raise JinaHostedAdapterError("connection", retryable=True) from None

    def _fetch_metadata_document(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_response_bytes: int,
    ) -> tuple[Mapping[str, object], int]:
        if max_response_bytes < 1:
            raise ValueError("metadata response limit must be positive")
        try:
            response = self._transport.get(
                url,
                headers=headers,
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        except _JinaResponseTooLarge:
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        except TimeoutError:
            raise JinaHostedAdapterError("timeout", retryable=True) from None
        except OSError:
            raise JinaHostedAdapterError("connection", retryable=True) from None
        if len(response.body) > max_response_bytes:
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        if not 200 <= response.status_code < 300:
            self._raise_for_status(response)
        payload = self._decode_success(response)
        if not _metadata_shape_is_bounded(payload):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        return payload, response.status_code

    @staticmethod
    def _decode_success(response: JinaHttpResponse) -> Mapping[str, object]:
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise JinaHostedAdapterError("invalid_response", retryable=False) from None
        if not isinstance(payload, dict):
            raise JinaHostedAdapterError("invalid_response", retryable=False)
        return payload

    def _raise_for_status(self, response: JinaHttpResponse) -> None:
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
            retry_after_seconds=_retry_after_seconds(
                response.headers, now=self._wall_clock()
            ),
            rate_limit_headers=_safe_rate_limit_headers(response.headers),
        )


def _validated_vectors(
    payload: Mapping[str, object], *, expected_count: int, response_model: str
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
            response_model=response_model,
        )
    if set(by_index) != set(range(expected_count)):
        raise JinaHostedAdapterError("invalid_response", retryable=False)
    return tuple(by_index[index] for index in range(expected_count))


def _validated_batch_items(
    payload: Mapping[str, object],
    *,
    expected_count: int,
    status_code: int,
    retry_after_seconds: float | None,
    response_model: str,
) -> tuple[JinaBatchItemOutcome, ...]:
    """Validate each unambiguous response index without discarding siblings."""

    data = payload.get("data")
    if not isinstance(data, list):
        return tuple(
            JinaBatchItemOutcome(
                index,
                None,
                "invalid_response",
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )
            for index in range(expected_count)
        )
    candidates: dict[int, list[Mapping[str, object]]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= expected_count
        ):
            continue
        candidates.setdefault(index, []).append(item)
    outcomes: list[JinaBatchItemOutcome] = []
    for index in range(expected_count):
        indexed = candidates.get(index, [])
        if not indexed:
            outcomes.append(
                JinaBatchItemOutcome(
                    index,
                    None,
                    "missing_response_item",
                    retryable=True,
                    status_code=status_code,
                    retry_after_seconds=retry_after_seconds,
                )
            )
            continue
        if len(indexed) != 1:
            outcomes.append(
                JinaBatchItemOutcome(
                    index,
                    None,
                    "invalid_response",
                    status_code=status_code,
                    retry_after_seconds=retry_after_seconds,
                )
            )
            continue
        try:
            vector = _validated_vectors(
                {"data": [{**indexed[0], "index": 0}]},
                expected_count=1,
                response_model=response_model,
            )[0]
        except JinaHostedAdapterError:
            outcomes.append(
                JinaBatchItemOutcome(
                    index,
                    None,
                    "invalid_response",
                    status_code=status_code,
                    retry_after_seconds=retry_after_seconds,
                )
            )
            continue
        outcomes.append(
            JinaBatchItemOutcome(
                index=index,
                vector=replace(vector, index=index),
                error_code=None,
            )
        )
    return tuple(outcomes)


def _safe_usage(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    usage: dict[str, int | float] = {}
    for key in _SAFE_USAGE_FIELDS:
        count = value.get(key)
        number = _safe_numeric_value(count, maximum=None)
        if number is not None:
            usage[key] = number
    return usage


def _metadata_shape_is_bounded(value: object) -> bool:
    """Reject deeply nested or item-heavy metadata before pilot parsing."""

    remaining = 100_000
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > 32:
            return False
        if isinstance(item, dict):
            if len(item) > 20_000:
                return False
            stack.extend((key, depth + 1) for key in item)
            stack.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            if len(item) > 20_000:
                return False
            stack.extend((nested, depth + 1) for nested in item)
        elif isinstance(item, str):
            try:
                if len(item.encode("utf-8")) > 65_536:
                    return False
            except UnicodeError:
                return False
    return True


def _safe_billing(value: object) -> dict[str, int | float | str]:
    if not isinstance(value, dict) or len(value) > 32:
        return {}
    billing: dict[str, int | float | str] = {}
    currency = value.get("currency")
    if isinstance(currency, str) and _SAFE_CURRENCY.fullmatch(currency) is not None:
        billing["currency"] = currency
    for key in _SAFE_BILLING_FIELDS:
        number = _safe_numeric_value(
            value.get(key), maximum=_MAX_SAFE_RATE_LIMIT_VALUE
        )
        if number is not None:
            billing[key] = number
    return billing


def _safe_response_model(value: object) -> str | None:
    """Retain a bounded content-free model label exactly as returned."""

    if not isinstance(value, str) or _SAFE_JINA_MODEL_LABEL.fullmatch(value) is None:
        return None
    components = frozenset(re.split(r"[-._]", value.lower()))
    if components & _UNSAFE_MODEL_COMPONENTS:
        return None
    return value


def is_safe_jina_model_label(value: object) -> bool:
    """Return whether a value is a bounded, non-secret Jina v4 model label."""

    return _safe_response_model(value) is not None


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


def _retry_after_seconds(
    headers: Mapping[str, str], *, now: datetime
) -> float | None:
    for name, value in headers.items():
        if not isinstance(name, str) or name.lower() != "retry-after":
            continue
        numeric = _safe_numeric_value(value, maximum=_MAX_SAFE_RATE_LIMIT_VALUE)
        if numeric is not None:
            return numeric
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None or now.tzinfo is None:
            return None
        seconds = max(0.0, (parsed - now).total_seconds())
        return _safe_numeric_value(seconds, maximum=_MAX_SAFE_RATE_LIMIT_VALUE)
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
        observed_model="synthetic-fake-jina-v4",
        task="retrieval.passage",
        dimensions=2048,
        output_type="float",
        normalization="l2-client-float32-v1",
        client_api_contract_version="synthetic-v1",
        tokenizer_revision="synthetic-jina-v4-tokenizer-v1",
        tokenizer_engine="synthetic-codepoint-tokenizer-v1",
        context_token_limit=8_000,
        context_rules="markdown-block-greedy-no-overlap-truncate-false-v1",
        long_text_aggregation="l2-token-count-weighted-mean-float32-v1",
        image_input_rules=PRODUCTION_IMAGE_INPUT_RULES,
        image_transform=PRODUCTION_IMAGE_TRANSFORM_ID,
        multimodal_formula="l2-normalize-0.5-text-0.5-image-v1",
        client_configuration_version="wsj-embeddings-config-v6",
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
