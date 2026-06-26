import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def test_phase37_builds_decision_alignment_supported_for_proxy_rebuild(tmp_path):
    from paper11_geofm.phase37_decision_alignment import (
        PHASE37_CASE_FIELDNAMES,
        PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
        build_phase37_decision_alignment,
    )

    positive_case_id = "tile_positive|0|N1ZR|D4P8"
    failure_case_id = "tile_failure|1|N1ZR|D4P8"
    phase34_cases = [
        {
            "case_id": positive_case_id,
            "case_role": "phase33_positive_case",
            "eval_tile_id": "tile_positive",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "stability_class": "flip_to_positive",
            "variant_mean_base_planning_reward": 0.8,
            "comparator_mean_base_planning_reward": 0.6,
            "variant_mean_suitability_proxy": 0.7,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.8,
            "comparator_mean_low_slope_farmland_label": 0.3,
        },
        {
            "case_id": failure_case_id,
            "case_role": "phase33_failure_case",
            "eval_tile_id": "tile_failure",
            "seed": 1,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "stability_class": "stable_negative",
            "variant_mean_base_planning_reward": 0.2,
            "comparator_mean_base_planning_reward": 0.3,
            "variant_mean_suitability_proxy": 0.4,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.2,
            "comparator_mean_low_slope_farmland_label": 0.2,
        },
    ]
    phase34_blocks = [
        {
            "case_id": positive_case_id,
            "block_id": "good",
            "variant_step": 1,
            "current_farmland_label": 1.0,
            "slope_mean": 4.0,
            "slope_max": 8.0,
        },
        {
            "case_id": positive_case_id,
            "block_id": "bad",
            "comparator_step": 1,
            "current_farmland_label": 0.0,
            "slope_mean": 12.0,
            "slope_max": 20.0,
        },
        {
            "case_id": failure_case_id,
            "block_id": "bad",
            "variant_step": 1,
            "current_farmland_label": 0.0,
            "slope_mean": 12.0,
            "slope_max": 20.0,
        },
        {
            "case_id": failure_case_id,
            "block_id": "good",
            "comparator_step": 1,
            "current_farmland_label": 1.0,
            "slope_mean": 8.0,
            "slope_max": 12.0,
        },
    ]
    phase35_cases = [
        {
            "case_id": positive_case_id,
            "summary_reward_gap": 0.7,
            "action_overlap_pattern": "disjoint_positive_gap",
        },
        {
            "case_id": failure_case_id,
            "summary_reward_gap": -0.6,
            "action_overlap_pattern": "disjoint_negative_gap",
        },
    ]

    phase34_cases_csv = _write_csv(
        tmp_path / "phase34_case_map_cases.csv",
        phase34_cases,
        list(phase34_cases[0].keys()),
    )
    phase34_blocks_csv = _write_csv(
        tmp_path / "phase34_case_map_blocks.csv",
        phase34_blocks,
        [
            "case_id",
            "block_id",
            "variant_step",
            "comparator_step",
            "current_farmland_label",
            "slope_mean",
            "slope_max",
        ],
    )
    phase35_cases_csv = _write_csv(
        tmp_path / "phase35_action_overlap_cases.csv",
        phase35_cases,
        list(phase35_cases[0].keys()),
    )
    phase36_json = tmp_path / "phase36_suitability_proxy_validation.json"
    phase36_json.write_text(
        json.dumps({"phase36_proxy_validation_status": "proxy_signal_not_supported"}),
        encoding="utf-8",
    )

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv,
        phase34_blocks_csv,
        phase35_cases_csv,
        phase36_diagnosis_json=phase36_json,
    )

    assert analysis["phase"] == "phase37_decision_alignment"
    assert (
        analysis["phase37_decision_alignment_status"]
        == "decision_alignment_supported_for_proxy_rebuild"
    )
    assert analysis["claim_boundary"] == PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY
    assert analysis["phase36_proxy_validation_status"] == "proxy_signal_not_supported"
    assert analysis["row_counts"]["case_rows"] == 2
    assert analysis["row_counts"]["summary_rows"] > 0

    cases = {row["case_id"]: row for row in analysis["case_rows"]}
    assert set(PHASE37_CASE_FIELDNAMES).issuperset(cases[positive_case_id])
    assert cases[positive_case_id]["summary_reward_gap"] == 0.7
    assert cases[positive_case_id]["suitability_proxy_gap"] == 0.2
    assert cases[positive_case_id]["low_slope_farmland_label_gap"] == 0.5
    assert cases[positive_case_id]["current_farmland_label_gap"] == 1.0
    assert cases[positive_case_id]["slope_mean_gap"] < 0.0
    assert (
        cases[positive_case_id]["proxy_alignment_pattern"]
        == "proxy_or_label_alignment"
    )

    assert cases[failure_case_id]["summary_reward_gap"] == -0.6
    assert cases[failure_case_id]["case_role"] == "phase33_failure_case"
    assert cases[failure_case_id]["suitability_proxy_gap"] == -0.1
    assert cases[failure_case_id]["current_farmland_label_gap"] == -1.0
    assert cases[failure_case_id]["proxy_alignment_pattern"] == "no_proxy_alignment"


def test_phase37_status_uses_grouped_positive_cases_not_positive_aggregate(tmp_path):
    from paper11_geofm.phase37_decision_alignment import build_phase37_decision_alignment

    aligned_positive_id = "tile_aligned|0|N1ZR|D4P8"
    unaligned_positive_id = "tile_unaligned|0|N1ZR|D4P8"
    failure_id = "tile_failure|0|N1ZR|D4P8"
    phase34_cases = [
        {
            "case_id": aligned_positive_id,
            "case_role": "phase33_positive_case",
            "eval_tile_id": "tile_aligned",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.8,
            "comparator_mean_base_planning_reward": 0.6,
            "variant_mean_suitability_proxy": 0.8,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.7,
            "comparator_mean_low_slope_farmland_label": 0.4,
        },
        {
            "case_id": unaligned_positive_id,
            "case_role": "phase33_positive_case",
            "eval_tile_id": "tile_unaligned",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.2,
            "comparator_mean_base_planning_reward": 0.6,
            "variant_mean_suitability_proxy": 0.0,
            "comparator_mean_suitability_proxy": 1.0,
            "variant_mean_low_slope_farmland_label": 0.0,
            "comparator_mean_low_slope_farmland_label": 1.0,
        },
        {
            "case_id": failure_id,
            "case_role": "phase33_failure_case",
            "eval_tile_id": "tile_failure",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.1,
            "comparator_mean_base_planning_reward": 0.3,
            "variant_mean_suitability_proxy": 0.2,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.2,
            "comparator_mean_low_slope_farmland_label": 0.4,
        },
    ]
    phase34_blocks = [
        {
            "case_id": case_id,
            "block_id": f"{case_id}_variant",
            "variant_step": 1,
            "current_farmland_label": 0.0,
            "slope_mean": 10.0,
            "slope_max": 15.0,
        }
        for case_id in (aligned_positive_id, unaligned_positive_id, failure_id)
    ] + [
        {
            "case_id": case_id,
            "block_id": f"{case_id}_comparator",
            "comparator_step": 1,
            "current_farmland_label": 1.0,
            "slope_mean": 8.0,
            "slope_max": 12.0,
        }
        for case_id in (aligned_positive_id, unaligned_positive_id, failure_id)
    ]
    phase35_cases = [
        {
            "case_id": aligned_positive_id,
            "summary_reward_gap": 0.4,
            "action_overlap_pattern": "disjoint_positive_gap",
        },
        {
            "case_id": unaligned_positive_id,
            "summary_reward_gap": 0.2,
            "action_overlap_pattern": "disjoint_positive_gap",
        },
        {
            "case_id": failure_id,
            "summary_reward_gap": -0.3,
            "action_overlap_pattern": "disjoint_negative_gap",
        },
    ]

    phase34_cases_csv = _write_csv(
        tmp_path / "phase34_case_map_cases.csv",
        phase34_cases,
        list(phase34_cases[0].keys()),
    )
    phase34_blocks_csv = _write_csv(
        tmp_path / "phase34_case_map_blocks.csv",
        phase34_blocks,
        [
            "case_id",
            "block_id",
            "variant_step",
            "comparator_step",
            "current_farmland_label",
            "slope_mean",
            "slope_max",
        ],
    )
    phase35_cases_csv = _write_csv(
        tmp_path / "phase35_action_overlap_cases.csv",
        phase35_cases,
        list(phase35_cases[0].keys()),
    )

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv,
        phase34_blocks_csv,
        phase35_cases_csv,
    )

    assert (
        analysis["phase37_decision_alignment_status"]
        == "decision_alignment_supported_for_proxy_rebuild"
    )


def test_phase37_positive_only_joined_cases_are_inputs_insufficient(tmp_path):
    from paper11_geofm.phase37_decision_alignment import build_phase37_decision_alignment

    positive_id = "tile_positive_only|0|N1ZR|D4P8"
    unjoined_failure_id = "tile_unjoined_failure|0|N1ZR|D4P8"
    phase34_cases = [
        {
            "case_id": positive_id,
            "case_role": "phase33_positive_case",
            "eval_tile_id": "tile_positive_only",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.8,
            "comparator_mean_base_planning_reward": 0.6,
            "variant_mean_suitability_proxy": 0.9,
            "comparator_mean_suitability_proxy": 0.4,
            "variant_mean_low_slope_farmland_label": 0.8,
            "comparator_mean_low_slope_farmland_label": 0.3,
        },
        {
            "case_id": unjoined_failure_id,
            "case_role": "phase33_failure_case",
            "eval_tile_id": "tile_unjoined_failure",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.2,
            "comparator_mean_base_planning_reward": 0.4,
            "variant_mean_suitability_proxy": 0.2,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.1,
            "comparator_mean_low_slope_farmland_label": 0.4,
        },
    ]
    phase34_blocks = [
        {
            "case_id": positive_id,
            "block_id": "positive_variant",
            "variant_step": 1,
            "current_farmland_label": 1.0,
            "slope_mean": 5.0,
            "slope_max": 10.0,
        },
        {
            "case_id": positive_id,
            "block_id": "positive_comparator",
            "comparator_step": 1,
            "current_farmland_label": 0.0,
            "slope_mean": 12.0,
            "slope_max": 18.0,
        },
        {
            "case_id": unjoined_failure_id,
            "block_id": "failure_variant",
            "variant_step": 1,
            "current_farmland_label": 0.0,
            "slope_mean": 12.0,
            "slope_max": 18.0,
        },
        {
            "case_id": unjoined_failure_id,
            "block_id": "failure_comparator",
            "comparator_step": 1,
            "current_farmland_label": 1.0,
            "slope_mean": 6.0,
            "slope_max": 11.0,
        },
    ]
    phase35_cases = [
        {
            "case_id": positive_id,
            "summary_reward_gap": 0.5,
            "action_overlap_pattern": "disjoint_positive_gap",
        },
    ]

    phase34_cases_csv = _write_csv(
        tmp_path / "phase34_case_map_cases.csv",
        phase34_cases,
        list(phase34_cases[0].keys()),
    )
    phase34_blocks_csv = _write_csv(
        tmp_path / "phase34_case_map_blocks.csv",
        phase34_blocks,
        [
            "case_id",
            "block_id",
            "variant_step",
            "comparator_step",
            "current_farmland_label",
            "slope_mean",
            "slope_max",
        ],
    )
    phase35_cases_csv = _write_csv(
        tmp_path / "phase35_action_overlap_cases.csv",
        phase35_cases,
        list(phase35_cases[0].keys()),
    )

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv,
        phase34_blocks_csv,
        phase35_cases_csv,
    )

    assert analysis["row_counts"]["case_rows"] == 1
    assert (
        analysis["phase37_decision_alignment_status"]
        == "decision_alignment_inputs_insufficient"
    )


def test_phase37_failure_subgroup_alignment_blocks_support(tmp_path):
    from paper11_geofm.phase37_decision_alignment import build_phase37_decision_alignment

    positive_id = "tile_positive_group|0|N1ZR|D4P8"
    aligned_failure_id = "tile_failure_aligned|0|N1ZR|D4P8"
    negative_failure_id = "tile_failure_negative|0|N1ZR|D4P8"
    phase34_cases = [
        {
            "case_id": positive_id,
            "case_role": "phase33_positive_case",
            "eval_tile_id": "tile_positive_group",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.8,
            "comparator_mean_base_planning_reward": 0.6,
            "variant_mean_suitability_proxy": 0.8,
            "comparator_mean_suitability_proxy": 0.5,
            "variant_mean_low_slope_farmland_label": 0.7,
            "comparator_mean_low_slope_farmland_label": 0.4,
        },
        {
            "case_id": aligned_failure_id,
            "case_role": "phase33_failure_case",
            "eval_tile_id": "tile_failure_aligned",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.4,
            "comparator_mean_base_planning_reward": 0.5,
            "variant_mean_suitability_proxy": 0.7,
            "comparator_mean_suitability_proxy": 0.4,
            "variant_mean_low_slope_farmland_label": 0.3,
            "comparator_mean_low_slope_farmland_label": 0.3,
        },
        {
            "case_id": negative_failure_id,
            "case_role": "phase33_failure_case",
            "eval_tile_id": "tile_failure_negative",
            "seed": 0,
            "variant_id": "N1ZR",
            "comparator_variant_id": "D4P8",
            "variant_mean_base_planning_reward": 0.1,
            "comparator_mean_base_planning_reward": 0.5,
            "variant_mean_suitability_proxy": 0.0,
            "comparator_mean_suitability_proxy": 1.0,
            "variant_mean_low_slope_farmland_label": 0.0,
            "comparator_mean_low_slope_farmland_label": 1.0,
        },
    ]
    phase34_blocks = []
    for case_id in (positive_id, aligned_failure_id, negative_failure_id):
        phase34_blocks.extend(
            [
                {
                    "case_id": case_id,
                    "block_id": f"{case_id}_variant",
                    "variant_step": 1,
                    "current_farmland_label": 0.0,
                    "slope_mean": 10.0,
                    "slope_max": 15.0,
                },
                {
                    "case_id": case_id,
                    "block_id": f"{case_id}_comparator",
                    "comparator_step": 1,
                    "current_farmland_label": 1.0,
                    "slope_mean": 8.0,
                    "slope_max": 12.0,
                },
            ]
        )
    phase35_cases = [
        {
            "case_id": positive_id,
            "summary_reward_gap": 0.4,
            "action_overlap_pattern": "disjoint_positive_gap",
        },
        {
            "case_id": aligned_failure_id,
            "summary_reward_gap": -0.2,
            "action_overlap_pattern": "disjoint_negative_gap",
        },
        {
            "case_id": negative_failure_id,
            "summary_reward_gap": -0.5,
            "action_overlap_pattern": "disjoint_negative_gap",
        },
    ]

    phase34_cases_csv = _write_csv(
        tmp_path / "phase34_case_map_cases.csv",
        phase34_cases,
        list(phase34_cases[0].keys()),
    )
    phase34_blocks_csv = _write_csv(
        tmp_path / "phase34_case_map_blocks.csv",
        phase34_blocks,
        [
            "case_id",
            "block_id",
            "variant_step",
            "comparator_step",
            "current_farmland_label",
            "slope_mean",
            "slope_max",
        ],
    )
    phase35_cases_csv = _write_csv(
        tmp_path / "phase35_action_overlap_cases.csv",
        phase35_cases,
        list(phase35_cases[0].keys()),
    )

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv,
        phase34_blocks_csv,
        phase35_cases_csv,
    )

    assert (
        analysis["phase37_decision_alignment_status"]
        == "decision_alignment_not_supported"
    )
