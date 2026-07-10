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
