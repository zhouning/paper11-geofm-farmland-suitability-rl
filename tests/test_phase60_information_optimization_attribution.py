import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _phase48(status="compressed_geofm_route_supported"):
    return {
        "phase48_compressed_geofm_status": status,
        "pooled_compressed_control_delta": {
            "mean_reward_delta": 0.2921767818,
            "positive_fraction": 0.6166666667,
            "positive_tile_seed_count": 74,
            "total_tile_seed_count": 120,
        },
        "coverage_issues": {
            "missing_variant_rows": [],
            "duplicate_variant_rows": [],
            "unexpected_variant_rows": [],
        },
    }


def _phase53(status="cluster_mean_support"):
    return {
        "phase53_cluster_mean_status": status,
        "cluster_mean_summary": {
            "mean_cluster_delta": 0.2921767818,
            "exact_sign_flip_mean_p": 0.0196838379,
        },
    }


def _phase57(status="compressed_geometry_consistent"):
    return {
        "phase57_mechanism_status": status,
        "geometry_rows": [
            {
                "variant_id": "B1",
                "effective_rank": 9.49,
                "raw_variance_retention": 1.0,
            },
            {
                "variant_id": "D4P8",
                "effective_rank": 5.13,
                "raw_variance_retention": 0.858,
            },
            {
                "variant_id": "D4P16",
                "effective_rank": 7.30,
                "raw_variance_retention": 0.949,
            },
        ],
        "reward_gain_rows": [
            {"compressed_variant_id": "D4P8", "mean_delta": 0.2356980264},
            {"compressed_variant_id": "D4P16", "mean_delta": 0.3486555373},
        ],
    }


def _phase59(status="matched_dimension_geofm_not_supported"):
    matched_deltas = {
        "D4P8_minus_D5R8": {"mean_delta": -0.0107871307},
        "D4P8_minus_D5S8": {"mean_delta": 0.0003232239},
        "D4P16_minus_D5R16": {"mean_delta": -0.1193811247},
        "D4P16_minus_D5S16": {"mean_delta": 0.060921975},
    }
    if status == "matched_dimension_geofm_supported":
        matched_deltas = {key: {"mean_delta": 0.1} for key in matched_deltas}
    return {
        "phase59_matched_dimension_status": status,
        "learned_policy": {"matched_deltas": matched_deltas},
        "pooled_matched_control_delta": {
            "mean_delta": (
                0.1
                if status == "matched_dimension_geofm_supported"
                else -0.0172307641
            ),
            "positive_fraction": (
                0.75
                if status == "matched_dimension_geofm_supported"
                else 0.3666666667
            ),
            "positive_count": (
                45 if status == "matched_dimension_geofm_supported" else 22
            ),
            "total_count": 60,
        },
        "coverage_issues": {
            "missing_variant_rows": [],
            "duplicate_variant_rows": [],
            "unexpected_variant_rows": [],
        },
    }


def test_phase60_reports_mechanism_claim_narrowed():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48(),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59(),
    )

    assert analysis["phase"] == "phase60_information_optimization_attribution"
    assert analysis["phase60_attribution_status"] == "mechanism_claim_narrowed"
    axes = {row["axis_id"]: row for row in analysis["attribution_axes"]}
    assert axes["compressed_route_performance"]["axis_status"] == "supported"
    assert axes["geofm_specific_matched_dimension"]["axis_status"] == "not_supported"
    assert analysis["claim_boundary_recommendation"] == "narrow_to_low_dimensional_route"


def test_phase60_reports_geofm_specific_information_supported():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48(),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59("matched_dimension_geofm_supported"),
    )

    assert (
        analysis["phase60_attribution_status"]
        == "geofm_specific_information_supported"
    )
    assert (
        analysis["claim_boundary_recommendation"]
        == "allow_geofm_specific_matched_dimension_claim"
    )


def test_phase60_reports_low_dimensional_route_uncertain():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48("not_supported"),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59(),
    )

    assert analysis["phase60_attribution_status"] == "low_dimensional_route_uncertain"
