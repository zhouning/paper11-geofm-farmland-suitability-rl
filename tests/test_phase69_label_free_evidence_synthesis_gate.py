import csv
import json
import pytest
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    phase60 = _write_json(
        tmp_path / "phase60.json",
        {
            "phase": "phase60_information_optimization_attribution",
            "phase60_attribution_status": "mechanism_claim_narrowed",
            "attribution_axes": [
                {
                    "axis_id": "compressed_route_performance",
                    "axis_status": "supported",
                    "primary_metric": "pooled_mean_delta",
                    "primary_value": 0.2921767818,
                    "source_phase": "phase48_phase52_expanded",
                    "interpretation": "Compressed route performance is supported.",
                },
                {
                    "axis_id": "cluster_level_robustness",
                    "axis_status": "supported",
                    "primary_metric": "cluster_mean_delta",
                    "primary_value": 0.2921767818,
                    "source_phase": "phase53",
                    "interpretation": "Cluster-level support is present.",
                },
                {
                    "axis_id": "geofm_specific_matched_dimension",
                    "axis_status": "not_supported",
                    "primary_metric": "pooled_matched_control_mean_delta",
                    "primary_value": -0.0172307641,
                    "source_phase": "phase59",
                    "interpretation": "GeoFM-specific matched-dimension advantage is not supported.",
                },
            ],
        },
    )
    phase57 = _write_json(
        tmp_path / "phase57.json",
        {
            "phase": "phase57_compressed_representation_mechanism",
            "phase57_mechanism_status": "compressed_geometry_consistent",
            "geometry_rows": [
                {"variant_id": "B1", "effective_rank": 9.4947211626, "condition_number": 6658.9542931381},
                {"variant_id": "D4P8", "effective_rank": 5.1322783588, "variance_retention_vs_b1": 0.8587823898},
                {"variant_id": "D4P16", "effective_rank": 7.3009059917, "variance_retention_vs_b1": 0.9496006154},
            ],
        },
    )
    phase59 = _write_json(
        tmp_path / "phase59.json",
        {
            "phase": "phase59_matched_dimension_controls",
            "phase59_matched_dimension_status": "matched_dimension_geofm_not_supported",
            "pooled_matched_control_delta": {"mean_delta": -0.0172307641},
        },
    )
    phase62 = _write_json(
        tmp_path / "phase62.json",
        {
            "phase": "phase62_d4_d6_matched_ppo",
            "phase62_d4_d6_status": "d6_random_projection_advantage",
            "pooled_primary_delta": {"mean_delta": -0.0641900514},
        },
    )
    phase66 = _write_json(
        tmp_path / "phase66.json",
        {
            "phase": "phase66_reward_label_representation_audit",
            "phase66_status": "base_reward_target_masks_geofm_signal",
            "diagnostic_gate": {
                "phase66_status": "base_reward_target_masks_geofm_signal",
                "alignment_advantage": {
                    "b0_explicit_proxy_r2_mean": 0.9973990529,
                    "geofm_representation_proxy_r2_mean": 0.029462,
                },
            },
        },
    )
    phase67 = _write_json(
        tmp_path / "phase67.json",
        {
            "phase": "phase67_candidate_reward_label_target_audit",
            "phase67_status": "independent_label_required_before_reward_redesign",
            "candidate_target_gate": {"candidate_count": 0},
        },
    )
    phase68 = _write_json(
        tmp_path / "phase68.json",
        {
            "phase": "phase68_external_independent_label_package",
            "phase68_status": "external_label_package_ready",
            "row_counts": {
                "phase2_block_rows": 64984,
                "external_label_csvs": 0,
                "registry_rows": 0,
                "label_preflight_rows": 0,
            },
        },
    )
    return {
        "phase60_json": phase60,
        "phase57_json": phase57,
        "phase59_json": phase59,
        "phase62_json": phase62,
        "phase66_json": phase66,
        "phase67_json": phase67,
        "phase68_json": phase68,
    }


def test_phase69_current_style_evidence_yields_narrowed_route_status(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
        write_phase69_label_free_evidence_synthesis_gate_artifacts,
    )

    paths = _fixture_paths(tmp_path)
    analysis = build_phase69_label_free_evidence_synthesis_gate(**paths)

    assert analysis["phase"] == "phase69_label_free_evidence_synthesis_gate"
    assert analysis["phase69_status"] == "claim_must_be_narrowed_to_low_dimensional_route"
    assert analysis["row_counts"]["evidence_axis_rows"] == 5
    assert analysis["row_counts"]["claim_boundary_rows"] == 10
    assert "does not train" in analysis["claim_boundary"]

    axes = {row["axis_id"]: row for row in analysis["evidence_axis_rows"]}
    assert axes["route_support"]["axis_class"] == "support"
    assert axes["mechanism_support"]["axis_class"] == "support"
    assert axes["mechanism_limits"]["axis_class"] == "limit"
    assert axes["reward_target_limits"]["axis_class"] == "blocked"
    assert axes["external_label_state"]["axis_class"] == "blocked"

    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    assert claims["bounded_low_dimensional_route"]["claim_status"] == "allowed"
    assert claims["suitability_reward_readiness"]["claim_status"] == "blocked"
    assert claims["geofm_specific_matched_dimension_superiority"]["claim_status"] == "blocked"

    artifacts = write_phase69_label_free_evidence_synthesis_gate_artifacts(
        analysis,
        tmp_path / "outputs",
    )
    assert {path.name for path in artifacts.values()} == {
        "phase69_evidence_axes.csv",
        "phase69_claim_boundary_matrix.csv",
        "phase69_label_free_evidence_synthesis_gate.json",
        "phase69_label_free_evidence_synthesis_gate.md",
    }
    markdown = (tmp_path / "outputs" / "phase69_label_free_evidence_synthesis_gate.md").read_text(
        encoding="utf-8"
    )
    assert "claim_must_be_narrowed_to_low_dimensional_route" in markdown
    assert "bounded low-dimensional compressed state route" in markdown

def test_phase69_missing_route_support_is_insufficient(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    paths = _fixture_paths(tmp_path)
    phase60 = json.loads(paths["phase60_json"].read_text(encoding="utf-8"))
    phase60["attribution_axes"] = [
        row
        for row in phase60["attribution_axes"]
        if row["axis_id"] != "compressed_route_performance"
    ]
    paths["phase60_json"].write_text(json.dumps(phase60), encoding="utf-8")

    analysis = build_phase69_label_free_evidence_synthesis_gate(**paths)

    assert analysis["phase69_status"] == "label_free_evidence_insufficient"
    route_axis = {
        row["axis_id"]: row for row in analysis["evidence_axis_rows"]
    }["route_support"]
    assert route_axis["axis_class"] == "missing"
    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    assert claims["bounded_low_dimensional_route"]["claim_status"] == "blocked"


def test_phase69_claim_boundary_blocks_stronger_claims(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    analysis = build_phase69_label_free_evidence_synthesis_gate(**_fixture_paths(tmp_path))

    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    blocked_claims = {
        "raw_b1_superiority",
        "pca_optimality",
        "geofm_specific_matched_dimension_superiority",
        "suitability_reward_readiness",
        "b2_b3_reward_integration",
        "external_independent_label_passed",
        "independent_agronomic_suitability",
        "cross_region_transfer",
    }
    for claim_id in blocked_claims:
        assert claims[claim_id]["claim_status"] == "blocked"
    assert claims["formal_submission_readiness"]["claim_status"] == "out_of_scope"


def test_phase69_missing_required_status_field_raises(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    paths = _fixture_paths(tmp_path)
    phase66 = json.loads(paths["phase66_json"].read_text(encoding="utf-8"))
    phase66.pop("phase66_status")
    paths["phase66_json"].write_text(json.dumps(phase66), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Phase 66 JSON is missing required status field: phase66_status",
    ):
        build_phase69_label_free_evidence_synthesis_gate(**paths)