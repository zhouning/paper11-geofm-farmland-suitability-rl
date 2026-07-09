import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _feature_rows() -> list[dict[str, object]]:
    return [
        {
            "block_id": "b1",
            "explicit_feature_00": 5.0,
            "explicit_feature_01": 0.0,
            "explicit_feature_02": 0.0,
            "explicit_feature_04": 0.4,
            "explicit_feature_07": 0.1,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.8,
            "explicit_feature_16": 0.9,
            "current_farmland_label": 1,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 1,
            "embedding_pca_00": 0.9,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b2",
            "explicit_feature_00": 4.0,
            "explicit_feature_01": 5.0,
            "explicit_feature_02": 7.0,
            "explicit_feature_04": 0.2,
            "explicit_feature_07": 0.6,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.7,
            "explicit_feature_16": 0.8,
            "current_farmland_label": 1,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.8,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b3",
            "explicit_feature_00": 3.0,
            "explicit_feature_01": 15.0,
            "explicit_feature_02": 20.0,
            "explicit_feature_04": 0.0,
            "explicit_feature_07": 0.1,
            "explicit_feature_09": 0.2,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 0.2,
            "explicit_feature_16": 0.3,
            "current_farmland_label": 0,
            "farmland_or_orchard_label": 1,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.3,
            "embedding_pca_01": 0.0,
        },
        {
            "block_id": "b4",
            "explicit_feature_00": 1.0,
            "explicit_feature_01": 25.0,
            "explicit_feature_02": 35.0,
            "explicit_feature_04": 0.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.5,
            "explicit_feature_10": 0.4,
            "explicit_feature_13": 0.1,
            "explicit_feature_16": 0.1,
            "current_farmland_label": 0,
            "farmland_or_orchard_label": 0,
            "low_slope_farmland_label": 0,
            "embedding_pca_00": 0.1,
            "embedding_pca_01": 0.0,
        },
    ]


def test_phase67_builds_base_weak_geofm_and_residual_candidate_targets():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_targets,
    )

    targets = build_phase67_candidate_targets(
        rows=_feature_rows(),
        label_columns=[
            "current_farmland_label",
            "farmland_or_orchard_label",
            "low_slope_farmland_label",
        ],
        representation_prefixes=["embedding_pca_"],
    )
    by_id = {target["target_id"]: target for target in targets}

    assert "base_planning_reward" in by_id
    assert "weak_label_current_farmland_label" in by_id
    assert "weak_label_farmland_or_orchard_label" in by_id
    assert "weak_label_low_slope_farmland_label" in by_id
    assert "geofm_norm_embedding_pca" in by_id
    assert "residual_base_after_explicit" in by_id
    assert "residual_weak_label_current_farmland_label_after_explicit" in by_id
    assert "residual_geofm_norm_embedding_pca_after_explicit" in by_id
    assert by_id["base_planning_reward"]["target_family"] == "base_reward"
    assert by_id["weak_label_current_farmland_label"]["target_kind"] == "binary"
    assert by_id["geofm_norm_embedding_pca"]["depends_on_geofm"] is True
    assert by_id["residual_base_after_explicit"]["target_family"] == "explicit_residual"


def test_phase67_inventory_marks_zero_variance_and_missing_targets_unusable():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_inventory,
    )

    targets = [
        {
            "target_id": "constant_score",
            "target_family": "fixture",
            "values_by_block": {"b1": 1.0, "b2": 1.0, "b3": 1.0},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "source_detail": "fixture",
            "target_kind": "continuous",
        },
        {
            "target_id": "partial_score",
            "target_family": "fixture",
            "values_by_block": {"b1": 0.1, "b2": None, "b3": 0.9},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "source_detail": "fixture",
            "target_kind": "continuous",
        },
    ]

    rows = build_phase67_candidate_target_inventory(targets, expected_block_ids=["b1", "b2", "b3"])
    by_id = {row["target_id"]: row for row in rows}

    assert by_id["constant_score"]["usable"] is False
    assert by_id["constant_score"]["unusable_reason"] == "zero_variance"
    assert by_id["partial_score"]["non_missing_count"] == 2
    assert by_id["partial_score"]["usable"] is True
