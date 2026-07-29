from datetime import UTC, date, datetime

import pytest

from wsj_pipeline.extract import (
    ExtractionError,
    derive_publication_dates,
    parse_wsj_timestamp,
)


def test_parses_z_and_offset_timestamps_as_utc() -> None:
    assert parse_wsj_timestamp("2024-01-01T01:30:00Z") == datetime(
        2024, 1, 1, 1, 30, tzinfo=UTC
    )
    assert parse_wsj_timestamp("2023-12-31T20:30:00-05:00") == datetime(
        2024, 1, 1, 1, 30, tzinfo=UTC
    )


def test_missing_optional_timestamp_returns_none() -> None:
    assert parse_wsj_timestamp(None) is None
    assert parse_wsj_timestamp("") is None


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("not-a-date", "invalid_timestamp"),
        ("2024-01-01T10:00:00", "naive_timestamp"),
    ],
)
def test_invalid_timestamp_has_stable_error_code(value: str, code: str) -> None:
    with pytest.raises(ExtractionError) as exc:
        parse_wsj_timestamp(value)

    assert exc.value.code == code
    assert value not in str(exc.value)


def test_derives_distinct_utc_and_new_york_dates() -> None:
    dates = derive_publication_dates(datetime(2024, 1, 1, 2, 30, tzinfo=UTC))

    assert dates.publication_date_utc == date(2024, 1, 1)
    assert dates.publication_date_new_york == date(2023, 12, 31)
    assert dates.published_at_new_york.isoformat() == "2023-12-31T21:30:00-05:00"
    assert dates.publication_year_ny == 2023
    assert dates.publication_month_ny == 12


@pytest.mark.parametrize(
    ("utc_value", "new_york_iso"),
    [
        ("2024-03-10T06:59:00+00:00", "2024-03-10T01:59:00-05:00"),
        ("2024-03-10T07:01:00+00:00", "2024-03-10T03:01:00-04:00"),
        ("2024-11-03T05:30:00+00:00", "2024-11-03T01:30:00-04:00"),
        ("2024-11-03T06:30:00+00:00", "2024-11-03T01:30:00-05:00"),
    ],
)
def test_new_york_conversion_observes_dst(utc_value: str, new_york_iso: str) -> None:
    published = datetime.fromisoformat(utc_value)

    assert (
        derive_publication_dates(published).published_at_new_york.isoformat()
        == new_york_iso
    )
