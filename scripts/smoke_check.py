from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "pytest.ini",
    ".gitignore",
    "reproducibility/REPRODUCTION_GUIDE.md",
    "reproducibility/DATA_MANIFEST.md",
    "reproducibility/FILE_MANIFEST.tsv",
    "docs/source_notes/paper11_design_thought.md",
    "docs/superpowers/specs/2026-06-08-phase1-bishan-geofm-baseline-design.md",
    "docs/superpowers/specs/2026-06-09-phase2-block-geofm-feature-assembly-design.md",
    "docs/superpowers/plans/2026-06-08-phase1-bishan-geofm-baseline.md",
    "paper/design/01_design_synthesis.md",
    "paper/design/02_system_design.md",
    "paper/design/03_experiment_plan.md",
    "paper/design/04_manuscript_outline.md",
    "paper/design/05_risks_and_boundaries.md",
    "paper/phase1_results/README.md",
    "paper/phase1_results/01_phase1_result_interpretation.md",
    "paper/phase1_results/02_next_experiment_matrix.md",
    "experiments/phase1_bishan_baseline/run_phase1.py",
    "experiments/geofm_runtime/embedding_space_env.py",
    "experiments/geofm_runtime/train_embedding_rl.py",
    "src/paper11_geofm/__init__.py",
    "src/paper11_geofm/sample_data.py",
    "src/paper11_geofm/regions.py",
    "src/paper11_geofm/features.py",
    "src/paper11_geofm/suitability.py",
    "src/paper11_geofm/artifacts.py",
    "src/legacy_runtime/county_env.py",
    "tests/test_phase1_geofm.py",
    "data/bishan_alphaearth_sample/metadata.json",
]


@dataclass(frozen=True)
class SmokeCheckResult:
    ok: bool
    sample_years: list[int]
    embedding_shape: tuple[int, ...]
    errors: list[str]


def _load_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_checks(root: Path | None = None) -> SmokeCheckResult:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    errors: list[str] = []

    for relative_path in REQUIRED_PATHS:
        if not (root / relative_path).exists():
            errors.append(f"missing required path: {relative_path}")

    sample_dir = root / "data" / "bishan_alphaearth_sample"
    sample_years: list[int] = []
    embedding_shape: tuple[int, ...] = ()

    try:
        metadata = _load_metadata(sample_dir)
        sample_years = list(metadata["years"])
        if sample_years != list(range(2017, 2025)):
            errors.append(f"unexpected sample years: {sample_years}")
        if metadata.get("embedding_dim") != 64:
            errors.append(f"unexpected embedding_dim: {metadata.get('embedding_dim')}")
    except Exception as exc:
        errors.append(f"failed to read metadata: {exc}")
        metadata = {}

    for year in range(2017, 2025):
        path = sample_dir / f"bishan_emb_{year}.npy"
        if not path.exists():
            errors.append(f"missing embedding sample: {path.name}")
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.shape[-1] != 64:
                errors.append(f"{path.name} has final dimension {arr.shape[-1]}, expected 64")
            if not embedding_shape:
                embedding_shape = tuple(arr.shape)
        except Exception as exc:
            errors.append(f"failed to load {path.name}: {exc}")

    context_path = sample_dir / "bishan_context.npy"
    if not context_path.exists():
        errors.append("missing context sample: bishan_context.npy")
    else:
        try:
            np.load(context_path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"failed to load bishan_context.npy: {exc}")

    grid_shape = tuple(metadata.get("grid_shape", ()))
    if embedding_shape and grid_shape and embedding_shape[:2] != grid_shape:
        errors.append(
            f"embedding shape {embedding_shape[:2]} does not match metadata grid_shape {grid_shape}"
        )

    return SmokeCheckResult(
        ok=not errors,
        sample_years=sample_years,
        embedding_shape=embedding_shape,
        errors=errors,
    )


def main() -> int:
    result = run_checks()
    if result.ok:
        print("Paper11 smoke check passed.")
        print(f"Sample years: {result.sample_years}")
        print(f"Embedding shape: {result.embedding_shape}")
        return 0

    print("Paper11 smoke check failed.")
    for error in result.errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
