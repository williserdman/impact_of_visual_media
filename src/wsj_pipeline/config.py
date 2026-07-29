"""Pipeline configuration and path-safety validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when pipeline configuration is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Resolved paths and bounded processing settings."""

    source_root: Path
    output_root: Path
    batch_size: int = 250

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        project_root: Path | None = None,
    ) -> PipelineConfig:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)

        root = (project_root or path.parent.parent).resolve()
        source_root = _resolve_path(root, raw["paths"]["source"])
        output_root = _resolve_path(root, raw["paths"]["output"])
        batch_size = int(raw.get("pipeline", {}).get("batch_size", 250))

        if batch_size < 1:
            raise ConfigError("batch_size must be positive")
        if (
            source_root == output_root
            or source_root.is_relative_to(output_root)
            or output_root.is_relative_to(source_root)
        ):
            raise ConfigError("source and output paths must not overlap")

        return cls(
            source_root=source_root,
            output_root=output_root,
            batch_size=batch_size,
        )

    @property
    def state_db(self) -> Path:
        return self.output_root / "state" / "pipeline.duckdb"

    @property
    def parquet_root(self) -> Path:
        return self.output_root / "parquet"

    @property
    def index_db(self) -> Path:
        return self.output_root / "index" / "wsj.duckdb"

    @property
    def staging_root(self) -> Path:
        return self.output_root / "staging"


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()
