from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


EXPECTED_YEARS = list(range(2017, 2025))
EXPECTED_EMBEDDING_DIM = 64


def load_metadata(sample_dir: Path) -> dict:
    """Load and validate Bishan sample metadata."""
    metadata_path = Path(sample_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    years = list(metadata.get("years", []))
    if years != EXPECTED_YEARS:
        raise ValueError(f"Expected years {EXPECTED_YEARS}, got {years}")
    if metadata.get("embedding_dim") != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding_dim {EXPECTED_EMBEDDING_DIM}, "
            f"got {metadata.get('embedding_dim')}"
        )

    grid_shape = metadata.get("grid_shape")
    if not isinstance(grid_shape, list) or len(grid_shape) != 2:
        raise ValueError(f"Expected grid_shape [rows, cols], got {grid_shape}")

    return metadata


def load_embedding(sample_dir: Path, year: int, mmap: bool = True) -> np.ndarray:
    """Load a single annual Bishan embedding grid."""
    sample_dir = Path(sample_dir)
    path = sample_dir / f"bishan_emb_{year}.npy"
    mode = "r" if mmap else None
    embedding = np.load(path, mmap_mode=mode)

    if embedding.ndim != 3 or embedding.shape[-1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"{path.name} must have shape [rows, cols, {EXPECTED_EMBEDDING_DIM}], "
            f"got {embedding.shape}"
        )
    return embedding


def load_annual_embeddings(
    sample_dir: Path, years: Iterable[int]
) -> dict[int, np.ndarray]:
    """Load multiple annual Bishan embedding grids."""
    return {int(year): load_embedding(sample_dir, int(year)) for year in years}
