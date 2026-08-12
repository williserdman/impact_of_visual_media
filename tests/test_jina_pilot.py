"""Simulated public-CLI tests for the generated-content Jina pilot."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Mapping

import pytest

from wsj_embeddings.adapters import JinaHttpResponse
from wsj_embeddings.cli import build_parser, main
from wsj_embeddings.pilot import _encoded_png


class PilotTransport:
    """Return complete synthetic hosted responses without opening a socket."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

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


def test_pilot_reports_generated_text_image_mixed_and_boundary_observations(
    capsys,
) -> None:
    """Break caught: pilot skips a modality/boundary or reports unsafe output."""

    transport = PilotTransport()

    exit_code = main(
        ["pilot"],
        environment={"JINA_API_KEY": "synthetic-secret"},
        transport=transport,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == f"{json.dumps(json.loads(output), sort_keys=True)}\n"
    assert json.loads(output) == {
        "effective_constraints": {
            "image_bytes": [
                {"outcome": "succeeded", "value": 5_000_000},
                {"outcome": "succeeded", "value": 8_000_000},
            ],
            "text_nominal_units": [
                {"outcome": "succeeded", "value": 8_192},
                {"outcome": "succeeded", "value": 32_768},
            ],
        },
        "openapi_metadata": {
            "observed_on": "2026-08-11",
            "version": "2026.07.27.1603",
        },
        "probes": [
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "text_normal",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "image_normal",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
            {
                "dimensions": [2048, 2048],
                "model": "jina-embeddings-v4",
                "name": "mixed_normal",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0, 5.0],
            },
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "text_nominal_8192_units",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "text_nominal_32768_units",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "image_nominal_5000000_bytes",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
            {
                "dimensions": [2048],
                "model": "jina-embeddings-v4",
                "name": "image_nominal_8000000_bytes",
                "rate_limit_headers": {"x-ratelimit-remaining-requests": 499.0},
                "retries": 0,
                "status": "succeeded",
                "status_code": 200,
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
                "vector_norms": [5.0],
            },
        ],
    }
    assert [len(request["input"]) for request in transport.requests] == [
        1,
        1,
        2,
        1,
        1,
        1,
        1,
    ]
    assert all(request["truncate"] is False for request in transport.requests)
    assert [
        sorted(item) for request in transport.requests for item in request["input"]
    ] == [
        ["text"],
        ["image"],
        ["text"],
        ["image"],
        ["text"],
        ["text"],
        ["image"],
        ["image"],
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
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(output) == {"error": "authentication"}
    assert transport.calls == 1
    assert list(tmp_path.iterdir()) == []


def test_near_limit_generated_pngs_are_structurally_valid() -> None:
    """Break caught: padded image probes are base64 but not valid PNG streams."""

    for target_size in (5_000_000, 8_000_000):
        image = base64.b64decode(_encoded_png(target_size), validate=True)

        assert len(image) == target_size
        _assert_valid_png(image)


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


def _assert_valid_png(image: bytes) -> None:
    """Validate PNG framing, CRCs, text grammar, and decompressed 1x1 pixels."""

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
    assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"tEXt", b"IEND"]
    assert chunks[0][1] == struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    keyword, text = chunks[2][1].split(b"\x00", 1)
    assert keyword == b"p"
    assert text
    assert zlib.decompress(chunks[1][1]) == b"\x00\x00\x00\x00"
