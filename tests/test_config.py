from pathlib import Path

import pytest

from wsj_pipeline.config import ConfigError, PipelineConfig


def write_config(
    path: Path,
    *,
    source: str,
    output: str,
    batch_size: int = 250,
    workers: int | None = None,
) -> None:
    pipeline_lines = [f"batch_size = {batch_size}"]
    if workers is not None:
        pipeline_lines.append(f"workers = {workers}")
    path.write_text(
        "\n".join(
            [
                "[paths]",
                f'source = "{source}"',
                f'output = "{output}"',
                "",
                "[pipeline]",
                *pipeline_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_config_resolves_relative_paths_from_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        source="data/wsj_archive",
        output="data/processed/wsj",
    )

    config = PipelineConfig.from_toml(config_path, project_root=tmp_path)

    assert config.source_root == (tmp_path / "data/wsj_archive").resolve()
    assert config.output_root == (tmp_path / "data/processed/wsj").resolve()
    assert config.catalog_db == config.output_root / "catalog.duckdb"
    assert config.text_root == config.output_root / "text"
    assert config.staging_root == config.output_root / "staging"
    assert config.lock_path == config.output_root / "pipeline.lock"
    assert config.batch_size == 250
    assert config.workers == 4


def test_config_reads_worker_count(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        source="data/wsj_archive",
        output="data/processed/wsj",
        workers=7,
    )

    config = PipelineConfig.from_toml(config_path, project_root=tmp_path)

    assert config.workers == 7


def test_config_exposes_single_catalog_and_text_paths(tmp_path: Path) -> None:
    config = PipelineConfig(tmp_path / "archive", tmp_path / "processed")

    assert config.catalog_db == tmp_path / "processed" / "catalog.duckdb"
    assert config.text_root == tmp_path / "processed" / "text"
    assert config.staging_root == tmp_path / "processed" / "staging"
    assert config.lock_path == tmp_path / "processed" / "pipeline.lock"


def test_direct_config_resolves_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = PipelineConfig(Path("archive"), Path("processed"))

    assert config.source_root == (tmp_path / "archive").resolve()
    assert config.output_root == (tmp_path / "processed").resolve()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_config_rejects_nonpositive_batch_size(tmp_path: Path, batch_size: int) -> None:
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        source="data/wsj_archive",
        output="data/processed/wsj",
        batch_size=batch_size,
    )

    with pytest.raises(ConfigError, match="batch_size must be positive"):
        PipelineConfig.from_toml(config_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("data/wsj_archive", "data/wsj_archive"),
        ("data/wsj_archive", "data/wsj_archive/processed"),
        ("data/wsj_archive/raw", "data/wsj_archive"),
    ],
)
def test_config_rejects_overlapping_source_and_output(
    tmp_path: Path, source: str, output: str
) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path, source=source, output=output)

    with pytest.raises(ConfigError, match="must not overlap"):
        PipelineConfig.from_toml(config_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("archive", "archive"),
        ("archive", "archive/processed"),
        ("archive/raw", "archive"),
    ],
)
def test_direct_config_rejects_resolved_source_output_overlap(
    tmp_path: Path,
    source: str,
    output: str,
) -> None:
    with pytest.raises(ConfigError, match="must not overlap"):
        PipelineConfig(tmp_path / source, tmp_path / output)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_direct_config_rejects_nonpositive_batch_size(
    tmp_path: Path,
    batch_size: int,
) -> None:
    with pytest.raises(ConfigError, match="batch_size must be positive"):
        PipelineConfig(tmp_path / "archive", tmp_path / "processed", batch_size)


@pytest.mark.parametrize("workers", [0, -1])
def test_config_rejects_nonpositive_worker_count(
    tmp_path: Path,
    workers: int,
) -> None:
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        source="data/wsj_archive",
        output="data/processed/wsj",
        workers=workers,
    )

    with pytest.raises(ConfigError, match="workers must be positive"):
        PipelineConfig.from_toml(config_path, project_root=tmp_path)


@pytest.mark.parametrize("workers", [0, -1])
def test_direct_config_rejects_nonpositive_worker_count(
    tmp_path: Path,
    workers: int,
) -> None:
    with pytest.raises(ConfigError, match="workers must be positive"):
        PipelineConfig(
            tmp_path / "archive",
            tmp_path / "processed",
            workers=workers,
        )
