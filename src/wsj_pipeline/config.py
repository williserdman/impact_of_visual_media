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
    workers: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_root",
            Path(self.source_root).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )
        self.validate()

    def validate(self) -> None:
        """Reject an invalid or tampered configuration without changing it."""

        if self.batch_size < 1:
            raise ConfigError("batch_size must be positive")
        if self.workers < 1:
            raise ConfigError("workers must be positive")
        source_root = self.source_root.expanduser().resolve()
        output_root = self.output_root.expanduser().resolve()
        if source_root != self.source_root or output_root != self.output_root:
            raise ConfigError("source and output paths must be resolved")
        if (
            source_root == output_root
            or source_root.is_relative_to(output_root)
            or output_root.is_relative_to(source_root)
        ):
            raise ConfigError("source and output paths must not overlap")

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
        workers = int(raw.get("pipeline", {}).get("workers", 4))

        return cls(
            source_root=source_root,
            output_root=output_root,
            batch_size=batch_size,
            workers=workers,
        )

    @property
    def catalog_db(self) -> Path:
        return self.output_root / "catalog.duckdb"

    @property
    def text_root(self) -> Path:
        return self.output_root / "text"

    @property
    def staging_root(self) -> Path:
        return self.output_root / "staging"

    @property
    def lock_path(self) -> Path:
        return self.output_root / "pipeline.lock"


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()
