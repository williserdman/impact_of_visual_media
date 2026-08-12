"""Stateless generated-content probes for the hosted Jina v4 contract."""

from __future__ import annotations

import base64
import json
import math
import re
import struct
import threading
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wsj_embeddings.adapters import (
    JinaEmbeddingAdapter,
    JinaEmbeddingBatchResponse,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
)
from wsj_embeddings.batching import (
    BatchExecutionFatal,
    BatchPolicy,
    execute_rate_aware_batches,
)

_SAFE_METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_API_VERSION = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{4}$")
_SAFE_JINA_MODEL_LABEL = re.compile(
    r"^jina-embeddings-v4(?:-[a-z0-9][a-z0-9._-]{0,95})?$"
)
_UNSAFE_MODEL_COMPONENTS = frozenset(
    {"bearer", "credential", "key", "password", "secret", "token"}
)
_SAFE_INPUT_MODALITIES = frozenset({"image", "pdf", "text"})
_SAFE_OUTPUT_MODALITIES = frozenset({"embedding"})
_SAFE_CURRENCY = re.compile(r"^[A-Z]{3}$")
_MAX_SAFE_METADATA_NUMBER = 1_000_000_000_000_000.0
_SAFE_MODEL_NUMERIC_FIELDS = (
    "context_length",
    "max_image_bytes",
    "max_image_pixels",
    "max_input_tokens",
    "max_tokens",
)
_SAFE_PRICING_FIELDS = frozenset(
    {
        "amount",
        "completion_tokens",
        "image",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "request",
        "total",
    }
)
_TEXT_PROBE_UNITS = (8_192, 32_768)
_IMAGE_PROBE_BYTES = (5_000_000, 8_000_000)


@dataclass(frozen=True, slots=True)
class _PilotProbe:
    """One fixed in-memory request and its content-free label."""

    name: str
    inputs: tuple[JinaEmbeddingInput, ...]


class _ObservingAdapter:
    """Measure actual adapter overlap and safe responses for one probe."""

    def __init__(self, adapter: JinaEmbeddingAdapter) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.responses: list[JinaEmbeddingBatchResponse] = []
        self.error_rate_limit_headers: list[Mapping[str, float]] = []

    def embed_batch(self, inputs, *, limits):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            try:
                response = self._adapter.embed_batch(inputs, limits=limits)
            except JinaHostedAdapterError as error:
                with self._lock:
                    self.error_rate_limit_headers.append(error.rate_limit_headers)
                raise
            with self._lock:
                self.responses.append(response)
            return response
        finally:
            with self._lock:
                self._active -= 1


def run_jina_pilot(adapter: JinaEmbeddingAdapter) -> dict[str, object]:
    """Measure fixed synthetic hosted requests without reading or writing state."""

    normal_probes = _normal_probes()
    first_result = _run_probe(adapter, normal_probes[0])
    boundary_probes = _boundary_probes(adapter)
    results = (
        first_result,
        *(_run_probe(adapter, probe) for probe in normal_probes[1:]),
        *(_run_probe(adapter, probe) for probe in boundary_probes),
    )
    concurrency_result = _run_concurrency_probe(adapter)
    openapi_observation = _observe_openapi(adapter)
    model_catalogue_observation = _observe_model_catalogue(adapter)
    total_retries = sum(
        int(result["retries"]) for result in (*results, concurrency_result)
    )
    return {
        "client_contract": {
            "client_api_contract_version": adapter.profile.client_api_contract_version,
            "requested_model": adapter.profile.model,
            "task": adapter.profile.task,
            "truncate": False,
        },
        "concurrency": {
            "attempted": concurrency_result["concurrency_attempted"],
            "measured_overlap": concurrency_result["concurrency_observed"],
            "outcome": concurrency_result["outcome"],
            "scheduler_wave": concurrency_result["scheduler_wave"],
        },
        "effective_constraints": {
            "image_bytes": [
                _constraint_result(results, f"image_nominal_{size}_bytes", size)
                for size in _IMAGE_PROBE_BYTES
            ],
            "text_tokens": [
                _text_constraint_result(
                    results, f"text_target_{tokens}_tokens", tokens
                )
                for tokens in _TEXT_PROBE_UNITS
            ],
        },
        "metadata": {
            "model_catalogue": model_catalogue_observation,
            "openapi": openapi_observation,
        },
        "probes": list(results),
        "readiness": _readiness(results, adapter),
        "retry_behavior": {
            "outcome": "observed" if total_retries else "not_observed",
            "retries": total_retries,
        },
        "schema_version": "jina-live-pilot-v2",
    }


def _run_probe(
    adapter: JinaEmbeddingAdapter, probe: _PilotProbe
) -> dict[str, object]:
    observer = _ObservingAdapter(adapter)
    text_token_count = sum(
        item.estimated_tokens for item in probe.inputs if item.kind == "text"
    )
    try:
        execution = execute_rate_aware_batches(
            observer,
            probe.inputs,
            policy=_pilot_policy(adapter, probe.inputs, max_concurrency=1),
            jitter=lambda _attempt: 0.0,
        )
    except BatchExecutionFatal as error:
        if error.code in {"authentication", "authorization"}:
            raise
        status_codes = {
            response.response_metadata.status_code for response in observer.responses
        }
        if error.status_code is not None:
            status_codes.add(error.status_code)
        result: dict[str, object] = {
            "billing": _billing_observation(observer.responses),
            "concurrency_attempted": 1,
            "concurrency_observed": observer.max_active,
            "dimensions": [],
            "error": error.code,
            "name": probe.name,
            "rate_limit_headers": _rate_header_observation(observer),
            "requests": error.requests,
            "response_models": sorted(
                {response.response_metadata.model for response in observer.responses}
            ),
            "retries": error.retries,
            "status": "rejected",
            "status_codes": sorted(status_codes),
            "throttles": error.throttles,
            "usage": dict(error.usage),
            "vector_norms": [],
        }
        if probe.name.startswith("text_target_"):
            result["local_token_count"] = text_token_count
        return result
    succeeded = all(item.vector is not None for item in execution.items)
    if not succeeded:
        failed = next(item for item in execution.items if item.vector is None)
        result = {
            "billing": _billing_observation(observer.responses),
            "concurrency_attempted": 1,
            "concurrency_observed": observer.max_active,
            "dimensions": [],
            "error": failed.error_code,
            "name": probe.name,
            "rate_limit_headers": _rate_header_observation(observer),
            "requests": execution.requests,
            "response_models": sorted(
                {response.response_metadata.model for response in observer.responses}
            ),
            "retries": execution.retries,
            "status": "rejected",
            "status_codes": (
                [] if failed.status_code is None else [failed.status_code]
            ),
            "throttles": execution.throttles,
            "usage": dict(execution.usage),
            "vector_norms": [],
        }
        if probe.name.startswith("text_target_"):
            result["local_token_count"] = text_token_count
        return result
    vectors = [item.vector for item in execution.items if item.vector is not None]
    result = {
        "billing": _billing_observation(observer.responses),
        "concurrency_attempted": 1,
        "concurrency_observed": observer.max_active,
        "dimensions": [len(vector.raw_vector) for vector in vectors],
        "name": probe.name,
        "rate_limit_headers": _rate_header_observation(observer),
        "requests": execution.requests,
        "response_models": sorted(
            {response.response_metadata.model for response in observer.responses}
        ),
        "retries": execution.retries,
        "status": "succeeded",
        "status_codes": sorted(
            {response.response_metadata.status_code for response in observer.responses}
        ),
        "throttles": execution.throttles,
        "usage": dict(execution.usage),
        "vector_norms": [
            round(math.sqrt(sum(value * value for value in vector.raw_vector)), 6)
            for vector in vectors
        ],
    }
    if probe.name.startswith("text_target_"):
        result["local_token_count"] = text_token_count
    return result


def _run_concurrency_probe(adapter: JinaEmbeddingAdapter) -> dict[str, object]:
    inputs = (
        JinaEmbeddingInput.text("generated concurrency probe one"),
        JinaEmbeddingInput.text("generated concurrency probe two"),
    )
    observer = _ObservingAdapter(adapter)
    try:
        execution = execute_rate_aware_batches(
            observer,
            inputs,
            policy=_pilot_policy(
                adapter,
                inputs,
                max_concurrency=2,
                max_items=1,
            ),
            jitter=lambda _attempt: 0.0,
        )
    except BatchExecutionFatal as error:
        if error.code in {"authentication", "authorization"}:
            raise
        return {
            "concurrency_attempted": 2,
            "concurrency_observed": observer.max_active,
            "outcome": "rejected",
            "retries": error.retries,
            "scheduler_wave": min(2, error.requests),
        }
    outcome = (
        "succeeded"
        if all(item.vector is not None for item in execution.items)
        else "rejected"
    )
    return {
        "concurrency_attempted": 2,
        "concurrency_observed": observer.max_active,
        "outcome": outcome,
        "retries": execution.retries,
        "scheduler_wave": execution.max_concurrency_observed,
    }


def _pilot_policy(
    adapter: JinaEmbeddingAdapter,
    inputs: Sequence[JinaEmbeddingInput],
    *,
    max_concurrency: int,
    max_items: int | None = None,
) -> BatchPolicy:
    profile = adapter.profile
    return BatchPolicy(
        max_items=len(inputs) if max_items is None else max_items,
        max_estimated_tokens=max(1, sum(item.estimated_tokens for item in inputs)),
        max_encoded_bytes=max(1, sum(item.encoded_bytes for item in inputs)),
        max_concurrency=max_concurrency,
        max_attempts=profile.batch_max_attempts,
        initial_backoff_seconds=profile.batch_initial_backoff_seconds,
        max_backoff_seconds=profile.batch_max_backoff_seconds,
        max_response_bytes=profile.batch_max_response_bytes,
    )


def _billing_observation(
    responses: Sequence[JinaEmbeddingBatchResponse],
) -> dict[str, object]:
    returned = [response.billing for response in responses if response.billing]
    if not returned:
        return {"outcome": "not_returned"}
    values = sorted(
        returned,
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
    )
    return {"outcome": "returned", "responses": values}


def _rate_header_observation(observer: _ObservingAdapter) -> dict[str, list[float]]:
    values: dict[str, set[float]] = {}
    for response in observer.responses:
        for name, value in response.response_metadata.rate_limit_headers.items():
            values.setdefault(name, set()).add(value)
    for headers in observer.error_rate_limit_headers:
        for name, value in headers.items():
            values.setdefault(name, set()).add(value)
    return {name: sorted(observed) for name, observed in sorted(values.items())}


def _readiness(
    results: Sequence[Mapping[str, object]], adapter: JinaEmbeddingAdapter
) -> dict[str, object]:
    required: dict[str, str] = {}
    ready = True
    for name, expected_count in (
        ("text_normal", 1),
        ("image_normal", 1),
        ("mixed_normal", 2),
    ):
        result = next(result for result in results if result["name"] == name)
        status = str(result["status"])
        required[name] = status
        ready = ready and status == "succeeded"
        ready = ready and result.get("dimensions") == [
            adapter.profile.dimensions
        ] * expected_count
        models = result.get("response_models")
        ready = ready and isinstance(models, list) and bool(models)
    return {
        "outcome": "ready_for_operator_review" if ready else "not_ready",
        "required_probes": dict(sorted(required.items())),
    }


def _observe_openapi(adapter: JinaEmbeddingAdapter) -> dict[str, object]:
    try:
        payload, status_code = adapter.fetch_openapi_document()
    except JinaHostedAdapterError as error:
        return _metadata_failure(error)
    info = payload.get("info")
    version = (
        _safe_domain_label(info.get("version"), _SAFE_API_VERSION)
        if isinstance(info, Mapping)
        else None
    )
    if version is None:
        return {
            "outcome": "not_observed",
            "reason": "malformed_metadata",
            "status_code": status_code,
        }
    return {"outcome": "observed", "status_code": status_code, "version": version}


def _observe_model_catalogue(adapter: JinaEmbeddingAdapter) -> dict[str, object]:
    try:
        payload, status_code = adapter.fetch_model_catalogue()
    except JinaHostedAdapterError as error:
        return _metadata_failure(error)
    data = payload.get("data")
    if not isinstance(data, list) or len(data) > 256:
        return {
            "outcome": "not_observed",
            "reason": "malformed_metadata",
            "status_code": status_code,
        }
    model = next(
        (
            item
            for item in data
            if isinstance(item, Mapping)
            and _safe_jina_model_label(item.get("id")) == adapter.profile.model
        ),
        None,
    )
    if model is None:
        return {
            "outcome": "not_observed",
            "reason": "model_not_returned",
            "status_code": status_code,
        }
    safe_model: dict[str, object] = {"id": adapter.profile.model}
    for name in _SAFE_MODEL_NUMERIC_FIELDS:
        if name not in model:
            continue
        number = _safe_nonnegative_number(model.get(name))
        if number is None:
            return _malformed_metadata(status_code)
        safe_model[name] = number
    for name, allowed in (
        ("input_modalities", _SAFE_INPUT_MODALITIES),
        ("output_modalities", _SAFE_OUTPUT_MODALITIES),
    ):
        if name not in model:
            continue
        labels = _safe_label_list(model.get(name), allowed=allowed)
        if labels is None:
            return _malformed_metadata(status_code)
        safe_model[name] = labels
    pricing = _safe_pricing(model.get("pricing"))
    if pricing is None:
        return _malformed_metadata(status_code)
    safe_model["pricing"] = pricing
    return {
        "model": safe_model,
        "outcome": "observed",
        "status_code": status_code,
    }


def _metadata_failure(error: JinaHostedAdapterError) -> dict[str, object]:
    result: dict[str, object] = {
        "outcome": "not_observed",
        "reason": error.code,
    }
    if error.status_code is not None:
        result["status_code"] = error.status_code
    return result


def _malformed_metadata(status_code: int) -> dict[str, object]:
    return {
        "outcome": "not_observed",
        "reason": "malformed_metadata",
        "status_code": status_code,
    }


def _safe_pricing(value: object) -> dict[str, object] | None:
    pricing: dict[str, object] = {"currency": "not_returned"}
    if value is None:
        return pricing
    if not isinstance(value, Mapping) or len(value) > 32:
        return None
    for key, raw_value in value.items():
        if not isinstance(key, str) or _SAFE_METADATA_KEY.fullmatch(key) is None:
            continue
        if key == "currency":
            label = _safe_domain_label(raw_value, _SAFE_CURRENCY)
            if label is None:
                return None
            pricing[key] = label
            continue
        if key not in _SAFE_PRICING_FIELDS:
            continue
        number = _safe_nonnegative_number(raw_value)
        if number is None:
            return None
        pricing[key] = number
    return pricing


def _safe_label_list(
    value: object, *, allowed: frozenset[str]
) -> list[str] | None:
    if not isinstance(value, list) or len(value) > 32:
        return None
    if any(not isinstance(item, str) or item not in allowed for item in value):
        return None
    return list(value)


def _safe_jina_model_label(value: object) -> str | None:
    if not isinstance(value, str) or _SAFE_JINA_MODEL_LABEL.fullmatch(value) is None:
        return None
    components = frozenset(re.split(r"[-._]", value.lower()))
    if components & _UNSAFE_MODEL_COMPONENTS:
        return None
    return value


def _safe_domain_label(value: object, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        return None
    return value


def _safe_nonnegative_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(number)
        or number < 0
        or number > _MAX_SAFE_METADATA_NUMBER
    ):
        return None
    return value


def _normal_probes() -> tuple[_PilotProbe, ...]:
    normal_text = JinaEmbeddingInput.text("generated pilot text")
    normal_image = JinaEmbeddingInput.image_base64(_encoded_png())
    return (
        _PilotProbe("text_normal", (normal_text,)),
        _PilotProbe("image_normal", (normal_image,)),
        _PilotProbe("mixed_normal", (normal_text, normal_image)),
    )


def _boundary_probes(adapter: JinaEmbeddingAdapter) -> tuple[_PilotProbe, ...]:
    return (
        *(
            _text_probe(adapter, tokens) for tokens in _TEXT_PROBE_UNITS
        ),
        *(
            _PilotProbe(
                f"image_nominal_{size}_bytes",
                (JinaEmbeddingInput.image_base64(_encoded_png(size)),),
            )
            for size in _IMAGE_PROBE_BYTES
        ),
    )


def _constraint_result(
    results: Sequence[Mapping[str, object]], name: str, value: int
) -> dict[str, object]:
    result = next(result for result in results if result["name"] == name)
    return {"outcome": result["status"], "value": value}


def _text_constraint_result(
    results: Sequence[Mapping[str, object]], name: str, target: int
) -> dict[str, object]:
    result = next(result for result in results if result["name"] == name)
    return {
        "local_token_count": result["local_token_count"],
        "outcome": result["status"],
        "target": target,
    }


def _text_probe(adapter: JinaEmbeddingAdapter, target: int) -> _PilotProbe:
    generated = _generated_text(target)
    local_token_count = len(adapter.token_offsets(generated))
    return _PilotProbe(
        f"text_target_{target}_tokens",
        (
            JinaEmbeddingInput.text(
                generated,
                estimated_tokens=local_token_count,
            ),
        ),
    )


def _generated_text(units: int) -> str:
    """Create an intentionally simple nominal token-like boundary input."""

    return "probe " * units


def _encoded_png(target_size: int | None = None) -> str:
    """Return an in-memory valid 1x1 PNG, optionally padded to an exact size."""

    png = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    png += _png_chunk(b"IEND", b"")
    png = b"\x89PNG\r\n\x1a\n" + png
    if target_size is not None:
        padding_size = target_size - len(png) - 14
        if padding_size < 0:
            raise ValueError("target_size is too small for a generated PNG")
        png = (
            png[:-12]
            + _png_chunk(b"tEXt", b"p\x00" + (b"0" * padding_size))
            + png[-12:]
        )
    return base64.b64encode(png).decode("ascii")


def _png_chunk(kind: bytes, value: bytes) -> bytes:
    return (
        struct.pack(">I", len(value))
        + kind
        + value
        + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
    )
