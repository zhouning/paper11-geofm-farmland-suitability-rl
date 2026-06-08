import sys
import json
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_load_bishan_metadata_and_base_embedding():
    from paper11_geofm.sample_data import load_embedding, load_metadata

    sample_dir = ROOT / "data" / "bishan_alphaearth_sample"

    metadata = load_metadata(sample_dir)
    embedding = load_embedding(sample_dir, 2020)

    assert metadata["years"] == list(range(2017, 2025))
    assert metadata["embedding_dim"] == 64
    assert tuple(metadata["grid_shape"]) == (67, 70)
    assert embedding.shape == (67, 70, 64)


def test_grid_regions_cover_full_embedding():
    from paper11_geofm.regions import make_grid_region_labels

    labels = make_grid_region_labels((67, 70), n_row_bins=5, n_col_bins=5)
    unique_ids = np.unique(labels)

    assert labels.shape == (67, 70)
    assert unique_ids.tolist() == list(range(25))
    assert labels.size == 67 * 70


def test_region_features_have_expected_schema():
    from paper11_geofm.features import compute_region_features
    from paper11_geofm.regions import make_grid_region_labels
    from paper11_geofm.sample_data import (
        load_annual_embeddings,
        load_embedding,
        load_metadata,
    )

    sample_dir = ROOT / "data" / "bishan_alphaearth_sample"
    metadata = load_metadata(sample_dir)
    base_embedding = load_embedding(sample_dir, 2020)
    annual_embeddings = load_annual_embeddings(sample_dir, metadata["years"])
    labels = make_grid_region_labels(base_embedding.shape[:2], 5, 5)

    rows = compute_region_features(base_embedding, labels, annual_embeddings)

    assert len(rows) == 25
    assert sum(row["pixel_count"] for row in rows) == 67 * 70
    assert "embedding_mean_00" in rows[0]
    assert "embedding_mean_63" in rows[0]
    assert "embedding_std_mean" in rows[0]
    assert "temporal_stability" in rows[0]
    assert np.isfinite(rows[0]["embedding_mean_00"])


def _phase1_feature_rows():
    from paper11_geofm.features import compute_region_features
    from paper11_geofm.regions import make_grid_region_labels
    from paper11_geofm.sample_data import (
        load_annual_embeddings,
        load_embedding,
        load_metadata,
    )

    sample_dir = ROOT / "data" / "bishan_alphaearth_sample"
    metadata = load_metadata(sample_dir)
    base_embedding = load_embedding(sample_dir, 2020)
    annual_embeddings = load_annual_embeddings(sample_dir, metadata["years"])
    labels = make_grid_region_labels(base_embedding.shape[:2], 5, 5)
    return compute_region_features(base_embedding, labels, annual_embeddings), metadata


def test_suitability_proxy_is_bounded():
    from paper11_geofm.suitability import add_suitability_proxy

    rows, _ = _phase1_feature_rows()

    scored_rows = add_suitability_proxy(rows)
    scores = np.array([row["suitability_proxy"] for row in scored_rows])

    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert scores.max() > scores.min()


def test_artifacts_are_written_with_claim_boundary(tmp_path):
    from paper11_geofm.artifacts import write_phase1_artifacts
    from paper11_geofm.suitability import add_suitability_proxy

    rows, metadata = _phase1_feature_rows()
    scored_rows = add_suitability_proxy(rows)

    paths = write_phase1_artifacts(
        scored_rows,
        tmp_path,
        {
            "metadata_source": metadata["source"],
            "base_year": 2020,
            "years": metadata["years"],
            "grid_shape": metadata["grid_shape"],
            "embedding_dim": metadata["embedding_dim"],
        },
    )

    assert paths["region_table"].exists()
    assert paths["summary"].exists()

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    boundary = summary["claim_boundary"].lower()

    assert summary["n_regions"] == 25
    assert 0.0 <= summary["suitability_min"] <= summary["suitability_max"] <= 1.0
    assert "does not directly measure soil" in boundary
    assert "fertility" in boundary
    assert "irrigation" in boundary


def test_phase1_runner_writes_artifacts(tmp_path):
    runner_path = (
        ROOT / "experiments" / "phase1_bishan_baseline" / "run_phase1.py"
    )
    spec = importlib.util.spec_from_file_location("phase1_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "region_features.csv").exists()
    assert (tmp_path / "summary.json").exists()
