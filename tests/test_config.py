from pathlib import Path

import pytest

from wsj_pipeline.config import ConfigError, PipelineConfig


def write_config(
    path: Path, *, source: str, output: str, batch_size: int = 250
) -> None:
    path.write_text(
        "\n".join(
            [
                "[paths]",
                f'source = "{source}"',
                f'output = "{output}"',
                "",
                "[pipeline]",
                f"batch_size = {batch_size}",
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
    assert config.state_db == config.output_root / "state" / "pipeline.duckdb"
    assert config.parquet_root == config.output_root / "parquet"
    assert config.index_db == config.output_root / "index" / "wsj.duckdb"
    assert config.staging_root == config.output_root / "staging"
    assert config.batch_size == 250


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
