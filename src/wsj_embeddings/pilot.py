"""Stateless generated-content probes for the hosted Jina v4 contract."""

from __future__ import annotations

import base64
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wsj_embeddings.adapters import (
    JinaEmbeddingAdapter,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
)

_OPENAPI_METADATA = {
    "observed_on": "2026-08-11",
    "version": "2026.07.27.1603",
}
_TEXT_PROBE_UNITS = (8_192, 32_768)
_IMAGE_PROBE_BYTES = (5_000_000, 8_000_000)


@dataclass(frozen=True, slots=True)
class _PilotProbe:
    """One fixed in-memory request and its content-free label."""

    name: str
    inputs: tuple[JinaEmbeddingInput, ...]


def run_jina_pilot(adapter: JinaEmbeddingAdapter) -> dict[str, object]:
    """Measure fixed synthetic hosted requests without reading or writing state."""

    results = tuple(_run_probe(adapter, probe) for probe in _probes())
    return {
        "effective_constraints": {
            "image_bytes": [
                _constraint_result(results, f"image_nominal_{size}_bytes", size)
                for size in _IMAGE_PROBE_BYTES
            ],
            "text_nominal_units": [
                _constraint_result(results, f"text_nominal_{units}_units", units)
                for units in _TEXT_PROBE_UNITS
            ],
        },
        "openapi_metadata": dict(_OPENAPI_METADATA),
        "probes": list(results),
    }


def _probes() -> tuple[_PilotProbe, ...]:
    normal_text = JinaEmbeddingInput.text("generated pilot text")
    normal_image = JinaEmbeddingInput.image_base64(_encoded_png())
    return (
        _PilotProbe("text_normal", (normal_text,)),
        _PilotProbe("image_normal", (normal_image,)),
        _PilotProbe("mixed_normal", (normal_text, normal_image)),
        *(
            _PilotProbe(
                f"text_nominal_{units}_units",
                (JinaEmbeddingInput.text(_generated_text(units)),),
            )
            for units in _TEXT_PROBE_UNITS
        ),
        *(
            _PilotProbe(
                f"image_nominal_{size}_bytes",
                (JinaEmbeddingInput.image_base64(_encoded_png(size)),),
            )
            for size in _IMAGE_PROBE_BYTES
        ),
    )


def _run_probe(
    adapter: JinaEmbeddingAdapter, probe: _PilotProbe
) -> dict[str, object]:
    try:
        response = adapter.embed(probe.inputs)
    except JinaHostedAdapterError as error:
        if error.code in {"authentication", "authorization"}:
            raise
        return {
            "error": error.code,
            "name": probe.name,
            "retries": 0,
            "status": "rejected",
            "status_code": error.status_code,
        }
    return {
        "dimensions": [len(vector.raw_vector) for vector in response.vectors],
        "model": response.response_metadata.model,
        "name": probe.name,
        "rate_limit_headers": dict(response.response_metadata.rate_limit_headers),
        "retries": 0,
        "status": "succeeded",
        "status_code": response.response_metadata.status_code,
        "usage": dict(response.usage),
        "vector_norms": [
            round(math.sqrt(sum(value * value for value in vector.raw_vector)), 6)
            for vector in response.vectors
        ],
    }


def _constraint_result(
    results: Sequence[Mapping[str, object]], name: str, value: int
) -> dict[str, object]:
    result = next(result for result in results if result["name"] == name)
    return {"outcome": result["status"], "value": value}


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
        padding_size = target_size - len(png) - 13
        if padding_size < 0:
            raise ValueError("target_size is too small for a generated PNG")
        png = png[:-12] + _png_chunk(b"tEXt", b"p" + (b"0" * padding_size)) + png[-12:]
    return base64.b64encode(png).decode("ascii")


def _png_chunk(kind: bytes, value: bytes) -> bytes:
    return (
        struct.pack(">I", len(value))
        + kind
        + value
        + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
    )
