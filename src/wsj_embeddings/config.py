"""Path-safe configuration for the downstream embedding output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class EmbeddingConfigError(ValueError):
    """Raised when embedding roots are not safely separated."""


@dataclass(frozen=True, slots=True)
class EmbeddingPipelineConfig:
    """Resolved, disjoint roots for one embedding coordinator run."""

    source_root: Path
    preprocessing_output_root: Path
    embedding_output_root: Path

    def __post_init__(self) -> None:
        for field_name in (
            "source_root",
            "preprocessing_output_root",
            "embedding_output_root",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)).expanduser().resolve(),
            )
        self.validate()

    @property
    def preprocessing_catalog(self) -> Path:
        return self.preprocessing_output_root / "catalog.duckdb"

    @property
    def embedding_catalog(self) -> Path:
        return self.embedding_output_root / "catalog.duckdb"

    @property
    def embedding_lock_path(self) -> Path:
        return self.embedding_output_root / "pipeline.lock"

    def validate(self) -> None:
        """Reject equal or nested roots before any coordinator access."""

        roots = (
            self.source_root,
            self.preprocessing_output_root,
            self.embedding_output_root,
        )
        if any(
            left == right
            or left.is_relative_to(right)
            or right.is_relative_to(left)
            for index, left in enumerate(roots)
            for right in roots[index + 1 :]
        ):
            raise EmbeddingConfigError("embedding roots must be disjoint")
