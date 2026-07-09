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


def test_phase67_gate_audit_blocks_base_weak_and_geofm_self_reference_targets():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_gate_audit,
    )

    inventory_rows = [
        {
            "target_id": "base_planning_reward",
            "target_family": "base_reward",
            "usable": True,
            "directly_uses_explicit": True,
            "depends_on_geofm": False,
            "self_referential": False,
        },
        {
            "target_id": "weak_label_current_farmland_label",
            "target_family": "weak_label",
            "usable": True,
            "directly_uses_explicit": True,
            "depends_on_geofm": False,
            "self_referential": False,
        },
        {
            "target_id": "geofm_norm_embedding_pca",
            "target_family": "geofm_self_reference",
            "usable": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": True,
            "self_referential": True,
        },
    ]
    gate_context = {
        "phase10_status": "not_ready_for_suitability_reward",
        "phase10_recommendation": "do_not_enable_suitability_reward",
        "phase18_suitability_reward_allowed": False,
        "phase39_status": "independent_label_inputs_missing",
        "phase40_status": "independent_label_inputs_missing",
    }

    rows = build_phase67_candidate_target_gate_audit(inventory_rows, gate_context)
    by_id = {row["target_id"]: row for row in rows}

    assert by_id["base_planning_reward"]["gate_risk"] == "explicit_reward_defined"
    assert by_id["base_planning_reward"]["reward_training_allowed"] is False
    assert by_id["weak_label_current_farmland_label"]["gate_risk"] == "explicit_label_leakage_risk"
    assert by_id["geofm_norm_embedding_pca"]["gate_risk"] == "geofm_self_reference"
    assert by_id["geofm_norm_embedding_pca"]["diagnostic_only_allowed"] is True


def test_phase67_gate_context_accepts_real_phase10_and_phase18_keys():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_gate_context,
    )

    context = build_phase67_gate_context(
        phase10={
            "status": "not_ready_for_suitability_reward",
            "recommendation": "do_not_enable_suitability_reward",
        },
        phase18={
            "suitability_reward_allowed": False,
            "phase10_status": "not_ready_for_suitability_reward",
        },
        phase39={},
        phase40={},
    )

    assert context["phase10_status"] == "not_ready_for_suitability_reward"
    assert context["phase10_recommendation"] == "do_not_enable_suitability_reward"
    assert context["phase18_suitability_reward_allowed"] is False
    assert context["phase39_status"] == "missing"
    assert context["phase40_status"] == "missing"


def test_phase67_information_gain_detects_geofm_residual_signal():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_information_gain,
    )

    rows = _feature_rows()
    for row in rows:
        for column in list(row):
            if str(column).startswith("explicit_feature_"):
                row[column] = 0.0
    targets = [
        {
            "target_id": "geofm_explained_residual",
            "target_family": "explicit_residual",
            "values_by_block": {"b1": 0.9, "b2": 0.8, "b3": 0.3, "b4": 0.1},
            "higher_is_better": True,
            "directly_uses_explicit": False,
            "depends_on_geofm": False,
            "self_referential": False,
            "target_kind": "continuous",
            "source_detail": "fixture",
        }
    ]

    info_rows = build_phase67_candidate_target_information_gain(
        feature_rows_by_variant={"D4P8": rows},
        targets=targets,
        top_k_values=[2],
    )
    row = info_rows[0]

    assert row["target_id"] == "geofm_explained_residual"
    assert row["variant_id"] == "D4P8"
    assert row["explicit_proxy_r2"] >= 0.0
    assert row["geofm_proxy_r2"] > 0.9
    assert row["geofm_spearman"] > 0.9
    assert row["geofm_minus_explicit_r2"] > 0.0
    assert row["geofm_topk_enrichment"] == 1.0


def test_phase67_candidate_gate_covers_all_statuses():
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        build_phase67_candidate_target_gate,
    )

    candidate_info = [
        {
            "target_id": "residual_a",
            "target_family": "explicit_residual",
            "variant_id": "D4P8",
            "geofm_minus_explicit_r2": 0.2,
            "geofm_minus_d6_r2": 0.1,
            "residual_after_explicit_r2": 0.2,
        }
    ]
    diagnostic_gate = [
        {
            "target_id": "residual_a",
            "usable": True,
            "gate_risk": "diagnostic_only_allowed",
            "diagnostic_only_allowed": True,
        },
    ]
    explicit_info = [
        {
            "target_id": "base_planning_reward",
            "target_family": "base_reward",
            "variant_id": "B0",
            "geofm_minus_explicit_r2": -0.9,
            "geofm_minus_d6_r2": 0.0,
            "residual_after_explicit_r2": 0.0,
        }
    ]
    explicit_gate = [
        {
            "target_id": "base_planning_reward",
            "usable": True,
            "gate_risk": "explicit_reward_defined",
            "diagnostic_only_allowed": True,
        },
    ]

    assert build_phase67_candidate_target_gate([], candidate_info, diagnostic_gate)["phase67_status"] == "candidate_target_found_for_diagnostic_training"
    assert build_phase67_candidate_target_gate([], explicit_info, explicit_gate)["phase67_status"] == "only_leakage_or_explicit_targets_found"
    assert build_phase67_candidate_target_gate([], [], [])["phase67_status"] == "independent_label_required_before_reward_redesign"
    assert build_phase67_candidate_target_gate(["missing artifact"], explicit_info, explicit_gate)["phase67_status"] == "insufficient"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_variant_fixture(output_dir: Path, variant_id: str, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_columns = sorted(column for column in rows[0] if str(column).startswith("explicit_feature_"))
    representation_columns = sorted(
        column
        for column in rows[0]
        if str(column).startswith(("embedding_pca_", "projection_"))
    )
    required_columns = explicit_columns if variant_id == "B0" else explicit_columns + representation_columns
    table = output_dir / f"variant_{variant_id}_features.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", *required_columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({"block_id": row["block_id"], **{column: row[column] for column in required_columns}})
    manifest = {
        "variants": {
            variant_id: {
                "ready": True,
                "feature_table": table.name,
                "required_columns": required_columns,
                "reward": "base_planning_reward",
                "state_groups": ["synthetic"],
            }
        }
    }
    (output_dir / "experiment_variants.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_phase67_writer_outputs_json_csv_and_markdown(tmp_path):
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        write_phase67_artifacts,
    )

    analysis = {
        "phase": "phase67_candidate_reward_label_target_audit",
        "candidate_target_inventory_rows": [
            {"target_id": "base_planning_reward", "claim_boundary": "phase67"}
        ],
        "candidate_target_gate_audit_rows": [
            {
                "target_id": "base_planning_reward",
                "gate_risk": "explicit_reward_defined",
                "claim_boundary": "phase67",
            }
        ],
        "candidate_target_information_gain_rows": [
            {"target_id": "base_planning_reward", "variant_id": "B0", "claim_boundary": "phase67"}
        ],
        "candidate_target_summary_rows": [
            {"target_id": "base_planning_reward", "summary": "control", "claim_boundary": "phase67"}
        ],
        "candidate_target_gate": {"phase67_status": "only_leakage_or_explicit_targets_found"},
        "claim_boundary": "phase67",
    }

    paths = write_phase67_artifacts(analysis, tmp_path / "outputs")

    assert paths["inventory_csv"].name == "phase67_candidate_target_inventory.csv"
    assert paths["gate_audit_csv"].name == "phase67_candidate_target_gate_audit.csv"
    assert paths["information_gain_csv"].name == "phase67_candidate_target_information_gain.csv"
    assert paths["summary_csv"].name == "phase67_candidate_target_summary.csv"
    assert paths["audit_json"].name == "phase67_candidate_reward_label_target_audit.json"
    assert paths["audit_md"].name == "phase67_candidate_reward_label_target_audit.md"
    saved = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    assert saved["phase67_status"] == "only_leakage_or_explicit_targets_found"
    assert "Phase 67 Candidate Reward/Label Target Audit" in paths["audit_md"].read_text(encoding="utf-8")


def test_phase67_cli_parser_accepts_required_and_optional_inputs():
    runner_path = ROOT / "experiments" / "phase67_candidate_reward_label_target_audit" / "run_phase67_candidate_reward_label_target_audit.py"
    spec = importlib.util.spec_from_file_location("phase67_runner_args", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--phase2-output-dir", "phase2",
            "--phase8-output-dir", "phase8",
            "--phase61-output-dir", "phase61",
            "--tile-index-csv", "tiles.csv",
            "--phase10-json", "phase10.json",
            "--phase18-json", "phase18.json",
            "--phase66-json", "phase66.json",
            "--phase39-json", "phase39.json",
            "--phase40-json", "phase40.json",
            "--variants", "B0,D4P8,D6R8",
            "--label-columns", "current_farmland_label,farmland_or_orchard_label",
            "--top-k-values", "8,16",
            "--output-dir", "outputs",
        ]
    )

    assert args.phase2_output_dir == Path("phase2")
    assert args.phase39_json == Path("phase39.json")
    assert args.output_dir == Path("outputs")


def test_phase67_run_wrapper_loads_fixture_and_returns_gate(tmp_path):
    from paper11_geofm.phase67_candidate_reward_label_target_audit import (
        run_phase67_candidate_reward_label_target_audit,
    )

    phase2 = tmp_path / "phase2"
    _write_variant_fixture(phase2, "B0", _feature_rows())
    _write_csv(phase2 / "block_geofm_features.csv", _feature_rows())
    phase10 = tmp_path / "phase10.json"
    phase10.write_text(
        json.dumps(
            {
                "status": "not_ready_for_suitability_reward",
                "recommendation": "do_not_enable_suitability_reward",
            }
        ),
        encoding="utf-8",
    )
    phase18 = tmp_path / "phase18.json"
    phase18.write_text(
        json.dumps(
            {
                "suitability_reward_allowed": False,
                "phase10_status": "not_ready_for_suitability_reward",
            }
        ),
        encoding="utf-8",
    )
    phase66 = tmp_path / "phase66.json"
    phase66.write_text(
        json.dumps({"phase66_status": "base_reward_target_masks_geofm_signal"}),
        encoding="utf-8",
    )

    analysis = run_phase67_candidate_reward_label_target_audit(
        phase2_output_dir=phase2,
        phase8_output_dir=None,
        phase61_output_dir=None,
        tile_index_csv=None,
        phase10_json=phase10,
        phase18_json=phase18,
        phase66_json=phase66,
        phase39_json=None,
        phase40_json=None,
        variants=["B0"],
        label_columns=[
            "current_farmland_label",
            "farmland_or_orchard_label",
            "low_slope_farmland_label",
        ],
        top_k_values=[2],
    )

    assert analysis["phase"] == "phase67_candidate_reward_label_target_audit"
    target_ids = {row["target_id"] for row in analysis["candidate_target_inventory_rows"]}
    assert "weak_label_current_farmland_label" in target_ids
    assert "geofm_norm_embedding_pca" in target_ids
    assert len(analysis["candidate_target_inventory_rows"]) >= 4
    assert analysis["candidate_target_gate"]["phase67_status"] in {
        "candidate_target_found_for_diagnostic_training",
        "only_leakage_or_explicit_targets_found",
        "independent_label_required_before_reward_redesign",
        "insufficient",
    }
