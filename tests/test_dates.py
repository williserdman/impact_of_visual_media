from datetime import UTC, date, datetime

import pytest

from wsj_pipeline import extract as extract_module


def test_parses_z_and_offset_timestamps_as_utc() -> None:
    assert extract_module.parse_wsj_timestamp("2024-01-01T01:30:00Z") == datetime(
        2024, 1, 1, 1, 30, tzinfo=UTC
    )
    assert extract_module.parse_wsj_timestamp(
        "2023-12-31T20:30:00-05:00"
    ) == datetime(
        2024, 1, 1, 1, 30, tzinfo=UTC
    )


def test_missing_optional_timestamp_returns_none() -> None:
    assert extract_module.parse_wsj_timestamp(None) is None
    assert extract_module.parse_wsj_timestamp("") is None


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("not-a-date", "invalid_timestamp"),
        ("2024-01-01T10:00:00", "naive_timestamp"),
    ],
)
def test_invalid_timestamp_has_stable_content_free_error(
    value: str,
    code: str,
) -> None:
    with pytest.raises(extract_module.ExtractionError) as exc:
        extract_module.parse_wsj_timestamp(value)

    assert exc.value.code == code
    assert value not in str(exc.value)


def test_derives_distinct_new_york_date() -> None:
    assert extract_module.derive_publication_date_new_york(
        datetime(2024, 1, 1, 2, 30, tzinfo=UTC)
    ) == date(2023, 12, 31)


@pytest.mark.parametrize(
    ("utc_value", "new_york_date"),
    [
        ("2024-01-15T04:59:00+00:00", date(2024, 1, 14)),
        ("2024-01-15T05:01:00+00:00", date(2024, 1, 15)),
        ("2024-07-15T03:59:00+00:00", date(2024, 7, 14)),
        ("2024-07-15T04:01:00+00:00", date(2024, 7, 15)),
    ],
)
def test_new_york_date_conversion_observes_dst(
    utc_value: str,
    new_york_date: date,
) -> None:
    assert extract_module.derive_publication_date_new_york(
        datetime.fromisoformat(utc_value)
    ) == new_york_date


def test_new_york_date_rejects_naive_timestamp() -> None:
    with pytest.raises(extract_module.ExtractionError) as exc:
        extract_module.derive_publication_date_new_york(
            datetime(2024, 1, 1, 10, 0)
        )

    assert exc.value.code == "naive_timestamp"
