import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _region_config(path: Path, *, independent: bool = True) -> Path:
    payload = {
        "source": {
            "source_id": "esri_global_lulc_10m_ts",
            "collection": (
                "projects/sat-io/open-datasets/landcover/"
                "ESRI_Global-LULC_10m_TS"
            ),
            "label_role": "independent_annual_product_label",
            "independent_from_dltb_slope_reward_geofm": independent,
            "crop_class_code": 5,
            "scale_m": 500,
        },
        "regions": [
            {
                "region_id": "alpha",
                "bbox": [100.0, 20.0, 101.0, 21.0],
                "years": [2017, 2018, 2019, 2020],
                "grid_shape": [2, 3],
                "embedding_dim": 2,
                "embedding_pattern": "alpha_emb_{year}.npy",
                "label_pattern": "alpha_lulc_{year}.npy",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase72a_region_contract_loads_independent_source(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )

    contract = load_phase72a_region_contract(
        _region_config(tmp_path / "regions.json")
    )
    assert contract.source_id == "esri_global_lulc_10m_ts"
    assert contract.crop_class_code == 5
    assert [region.region_id for region in contract.regions] == ["alpha"]
    assert contract.regions[0].grid_shape == (2, 3)


def test_phase72a_region_contract_rejects_nonindependent_source(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        load_phase72a_region_contract,
    )

    try:
        load_phase72a_region_contract(
            _region_config(tmp_path / "regions.json", independent=False)
        )
    except ValueError as exc:
        assert "independent" in str(exc).lower()
    else:
        raise AssertionError("Expected non-independent labels to be rejected")


def _asset_dirs(tmp_path: Path, *, bad_label_shape: bool = False):
    embedding_dir = tmp_path / "embeddings"
    label_dir = tmp_path / "labels"
    embedding_dir.mkdir()
    label_dir.mkdir()
    for year in (2017, 2018, 2019, 2020):
        np.save(
            embedding_dir / f"alpha_emb_{year}.npy",
            np.full((2, 3, 2), float(year), dtype=np.float32),
        )
        shape = (2, 2) if bad_label_shape and year == 2020 else (2, 3)
        np.save(
            label_dir / f"alpha_lulc_{year}.npy",
            np.full(shape, 5 if year < 2020 else 7, dtype=np.int32),
        )
    return embedding_dir, label_dir


def test_phase72a_asset_audit_hashes_complete_aligned_years(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        audit_phase72a_region_assets,
        load_phase72a_region_contract,
    )

    contract = load_phase72a_region_contract(
        _region_config(tmp_path / "regions.json")
    )
    embedding_dir, label_dir = _asset_dirs(tmp_path)
    audit = audit_phase72a_region_assets(
        contract,
        contract.regions[0],
        embedding_dir=embedding_dir,
        label_dir=label_dir,
    )
    assert audit["status"] == "region_label_inputs_ready"
    assert audit["years_ready"] == [2017, 2018, 2019, 2020]
    assert len(audit["file_manifest_rows"]) == 8
    assert all(
        len(row["sha256"]) == 64 for row in audit["file_manifest_rows"]
    )


def test_phase72a_asset_audit_blocks_shape_mismatch(tmp_path):
    from paper11_geofm.phase72a_label_sources import (
        audit_phase72a_region_assets,
        load_phase72a_region_contract,
    )

    contract = load_phase72a_region_contract(
        _region_config(tmp_path / "regions.json")
    )
    embedding_dir, label_dir = _asset_dirs(tmp_path, bad_label_shape=True)
    audit = audit_phase72a_region_assets(
        contract,
        contract.regions[0],
        embedding_dir=embedding_dir,
        label_dir=label_dir,
    )
    assert audit["status"] == "label_inputs_not_ready"
    assert "shape" in " ".join(audit["errors"]).lower()


def test_phase72a_samples_are_temporally_truncated_and_build_endpoints():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec
    from paper11_geofm.phase72a_temporal_samples import (
        build_phase72a_temporal_samples,
    )

    region = Phase72ARegionSpec(
        "alpha",
        (100.0, 20.0, 101.0, 21.0),
        (2017, 2018, 2019, 2020),
        (1, 2),
        2,
        "alpha_emb_{year}.npy",
        "alpha_lulc_{year}.npy",
    )
    embeddings = {
        year: np.full((1, 2, 2), float(year), dtype=np.float32)
        for year in region.years
    }
    labels = {
        2017: np.array([[5, 7]], dtype=np.int32),
        2018: np.array([[5, 5]], dtype=np.int32),
        2019: np.array([[7, 5]], dtype=np.int32),
        2020: np.array([[5, 7]], dtype=np.int32),
    }
    samples = build_phase72a_temporal_samples(
        region,
        embeddings=embeddings,
        labels=labels,
        crop_class_code=5,
        source_id="esri_global_lulc_10m_ts",
        source_role="independent_annual_product_label",
        max_history_years=4,
        spatial_block_size=2,
    )
    first = next(
        row
        for row in samples["sample_rows"]
        if row["unit_id"] == "r0000_c0000"
        and row["origin_year"] == 2017
    )
    index = int(first["sample_index"])
    assert (
        first["y_1y"],
        first["y_2y"],
        first["y_continuous_2y"],
    ) == (1, 0, 0)
    assert samples["tensors"]["history_mask"][index].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert samples["tensors"]["embedding_history"][index, 0].tolist() == [
        2017.0,
        2017.0,
    ]
    assert (
        float(
            samples["tensors"]["embedding_history"][index, 1:].sum()
        )
        == 0.0
    )


def test_phase72a_samples_mark_unavailable_two_year_target():
    from paper11_geofm.phase72a_label_sources import Phase72ARegionSpec
    from paper11_geofm.phase72a_temporal_samples import (
        build_phase72a_temporal_samples,
    )

    region = Phase72ARegionSpec(
        "alpha",
        (100.0, 20.0, 101.0, 21.0),
        (2018, 2019, 2020),
        (1, 1),
        1,
        "alpha_emb_{year}.npy",
        "alpha_lulc_{year}.npy",
    )
    samples = build_phase72a_temporal_samples(
        region,
        embeddings={
            year: np.array([[[float(year)]]], dtype=np.float32)
            for year in region.years
        },
        labels={
            year: np.array([[5]], dtype=np.int32) for year in region.years
        },
        crop_class_code=5,
        source_id="esri_global_lulc_10m_ts",
        source_role="independent_annual_product_label",
        max_history_years=3,
        spatial_block_size=1,
    )
    latest = next(
        row
        for row in samples["sample_rows"]
        if row["origin_year"] == 2019
    )
    index = int(latest["sample_index"])
    assert latest["y_1y"] == 1
    assert latest["y_2y"] == ""
    assert samples["tensors"]["y_2y"][index] == -1


def test_phase72a_package_writes_outputs_and_blank_review_labels(tmp_path):
    from paper11_geofm.phase72a_temporal_label_package import (
        build_phase72a_temporal_label_package,
        write_phase72a_temporal_label_package_artifacts,
    )

    embedding_dir, label_dir = _asset_dirs(tmp_path)
    package = build_phase72a_temporal_label_package(
        region_config=_region_config(tmp_path / "regions.json"),
        embedding_dirs={"alpha": embedding_dir},
        label_dirs={"alpha": label_dir},
        manual_review_per_stratum=2,
        spatial_block_size=2,
    )
    paths = write_phase72a_temporal_label_package_artifacts(
        package, tmp_path / "outputs"
    )
    assert package["phase72a_status"] == "phase72a_label_inputs_ready"
    assert package["row_counts"]["sample_rows"] > 0
    assert set(paths) == {
        "manifest_csv",
        "audit_csv",
        "sample_index_csv",
        "sample_tensors_npz",
        "review_frame_csv",
        "summary_csv",
        "package_json",
        "package_md",
    }
    review = pd.read_csv(
        paths["review_frame_csv"], keep_default_na=False
    )
    assert {
        "review_label",
        "review_source",
        "review_date",
        "review_confidence",
    }.issubset(review.columns)
    assert review["review_label"].eq("").all()
    tensors = np.load(paths["sample_tensors_npz"])
    assert (
        tensors["embedding_history"].shape[0]
        == package["row_counts"]["sample_rows"]
    )


def test_phase72a_package_blocks_samples_when_an_asset_is_missing(tmp_path):
    from paper11_geofm.phase72a_temporal_label_package import (
        build_phase72a_temporal_label_package,
    )

    embedding_dir, label_dir = _asset_dirs(tmp_path)
    (label_dir / "alpha_lulc_2020.npy").unlink()
    package = build_phase72a_temporal_label_package(
        region_config=_region_config(tmp_path / "regions.json"),
        embedding_dirs={"alpha": embedding_dir},
        label_dirs={"alpha": label_dir},
    )
    assert package["phase72a_status"] == "label_inputs_not_ready"
    assert package["sample_rows"] == []


def test_phase72a_fetcher_uses_injected_extractor(tmp_path):
    from experiments.phase72a_temporal_label_package.fetch_phase72a_esri_lulc import (
        fetch_phase72a_labels,
    )

    def fake_extractor(*, bbox, year, scale, collection):
        assert bbox == (100.0, 20.0, 101.0, 21.0)
        assert scale == 500
        assert "ESRI_Global-LULC" in collection
        return np.full(
            (2, 3), 5 if year < 2020 else 7, dtype=np.int32
        )

    manifest = fetch_phase72a_labels(
        region_config=_region_config(tmp_path / "regions.json"),
        output_dir=tmp_path / "labels",
        regions=("alpha",),
        years=(2017, 2018, 2019, 2020),
        extractor=fake_extractor,
    )
    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 4
    assert manifest["n_failures"] == 0
    assert (tmp_path / "labels" / "alpha_lulc_2020.npy").exists()


def test_phase72a_local_runner_succeeds_on_fixture(tmp_path):
    embedding_dir, label_dir = _asset_dirs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase72a_temporal_label_package"
        / "run_phase72a_temporal_label_package.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--region-config",
            str(_region_config(tmp_path / "regions.json")),
            "--embedding-dir",
            f"alpha={embedding_dir}",
            "--label-dir",
            f"alpha={label_dir}",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--manual-review-per-stratum",
            "2",
            "--spatial-block-size",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "phase72a_label_inputs_ready" in result.stdout
    assert (
        tmp_path
        / "outputs"
        / "phase72a_temporal_label_package.json"
    ).exists()
