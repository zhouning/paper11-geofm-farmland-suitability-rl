import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(block_id, explicit_00, values, prefix="embedding_pca"):
    row = {
        "block_id": block_id,
        "explicit_feature_00": explicit_00,
    }
    for index, value in enumerate(values):
        row[f"{prefix}_{index:02d}"] = value
    return row


def _b0_rows():
    return [
        {"block_id": "b1", "explicit_feature_00": 1.0},
        {"block_id": "b2", "explicit_feature_00": 2.0},
        {"block_id": "b3", "explicit_feature_00": 3.0},
        {"block_id": "b4", "explicit_feature_00": 4.0},
    ]


def _d4p8_rows():
    return [
        _row("b1", 1.0, [0.0, 1.0]),
        _row("b2", 2.0, [2.0, 3.0]),
        _row("b3", 3.0, [4.0, 5.0]),
        _row("b4", 4.0, [6.0, 7.0]),
    ]


def _d4p16_rows():
    return [
        _row("b1", 1.0, [0.0, 10.0, 20.0]),
        _row("b2", 2.0, [1.0, 11.0, 21.0]),
        _row("b3", 3.0, [2.0, 12.0, 22.0]),
        _row("b4", 4.0, [3.0, 13.0, 23.0]),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase59_builds_deterministic_matched_control_tables():
    from paper11_geofm.phase59_matched_dimension_controls import (
        PHASE59_CLAIM_BOUNDARY,
        build_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )

    assert protocol["phase"] == "phase59_matched_dimension_control_features"
    assert protocol["claim_boundary"] == PHASE59_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["D5R8", "D5S8", "D5R16", "D5S16"]
    assert set(protocol["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    assert protocol["summary"]["D5R8"]["control_dimension"] == 2
    assert protocol["summary"]["D5R16"]["control_dimension"] == 3
    assert protocol["summary"]["D5S8"]["source_variant_id"] == "D4P8"
    assert protocol["summary"]["D5S16"]["source_variant_id"] == "D4P16"

    d5s8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5S8"]
    ]
    d4p8_values = [row["embedding_pca_00"] for row in _d4p8_rows()]
    assert sorted(d5s8_values) == sorted(d4p8_values)
    assert d5s8_values != d4p8_values

    d5r8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5R8"]
    ]
    assert len(d5r8_values) == 4
    assert not all(math.isclose(value, d4p8_values[0]) for value in d5r8_values)


def test_phase59_writes_control_tables_and_manifest(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_tables,
        write_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )
    paths = write_phase59_matched_dimension_control_tables(
        protocol,
        tmp_path / "controls",
    )

    assert paths["manifest"].name == "experiment_variants.json"
    assert paths["summary"].name == "phase59_matched_dimension_control_feature_summary.json"
    assert set(paths["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert saved["variants"]["D5R8"]["feature_table"] == "variant_D5R8_features.csv"
    with paths["variant_tables"]["D5S16"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["block_id"] == "b1"
    assert "matched_control_02" in rows[0]


def _summary_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase25_seed_rank": seed + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 2,
        "n_features": 25,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b2",
        "claim_boundary": "fixture",
    }


def _phase59_summary_rows(case="supported"):
    rewards = {
        "supported": {
            "D4P8": [1.2, 1.3, 1.1, 1.4],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 1.0, 0.8, 1.0],
            "D4P16": [1.5, 1.4, 1.3, 1.6],
            "D5R16": [1.1, 1.1, 1.2, 1.2],
            "D5S16": [1.0, 1.1, 1.0, 1.1],
        },
        "partial": {
            "D4P8": [1.2, 1.2, 1.2, 1.2],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 0.9, 0.9, 0.9],
            "D4P16": [1.0, 1.0, 1.0, 1.0],
            "D5R16": [1.1, 1.1, 1.1, 1.1],
            "D5S16": [1.2, 1.2, 1.2, 1.2],
        },
        "not_supported": {
            "D4P8": [0.8, 0.8, 0.8, 0.8],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 0.9, 0.9, 0.9],
            "D4P16": [0.9, 0.9, 0.9, 0.9],
            "D5R16": [1.0, 1.0, 1.0, 1.0],
            "D5S16": [1.1, 1.1, 1.1, 1.1],
        },
    }[case]
    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rows = []
    for index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards.items():
            rows.append(_summary_row(variant_id, values[index], tile_id, seed))
    return rows


def test_phase59_analysis_supports_matched_dimension_geofm_route():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    analysis = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("supported"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=200,
        random_seed=59,
    )

    assert analysis["phase"] == "phase59_matched_dimension_control_analysis"
    assert analysis["phase59_matched_dimension_status"] == "matched_dimension_geofm_supported"
    assert analysis["learned_policy"]["matched_deltas"]["D4P8_minus_D5R8"]["mean_delta"] == 0.25
    assert analysis["pooled_matched_control_delta"]["positive_count"] == 16
    assert analysis["cluster_summary"]["cluster_count"] == 4
    assert analysis["signed_rank_summary"]["positive_rank_sum"] == 10


def test_phase59_status_rules_distinguish_partial_and_not_supported():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    partial = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("partial"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    not_supported = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("not_supported"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )

    assert partial["phase59_matched_dimension_status"] == "matched_dimension_geofm_partial"
    assert not_supported["phase59_matched_dimension_status"] == "matched_dimension_geofm_not_supported"


def test_phase59_reports_insufficient_for_missing_variant_rows():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    rows = [
        row for row in _phase59_summary_rows("supported")
        if row["variant_id"] != "D5S16"
    ]
    analysis = build_phase59_matched_dimension_control_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
    )

    assert analysis["phase59_matched_dimension_status"] == "insufficient"
    missing = {
        row["variant_id"] for row in analysis["coverage_issues"]["missing_variant_rows"]
    }
    assert missing == {"D5S16"}


def test_phase59_writer_outputs_json_summary_delta_and_markdown(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
        write_phase59_matched_dimension_control_artifacts,
    )

    rows = _phase59_summary_rows("supported")
    analysis = build_phase59_matched_dimension_control_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    paths = write_phase59_matched_dimension_control_artifacts(
        {**analysis, "summaries": rows},
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase59_matched_dimension_control_summary.csv"
    assert paths["delta_csv"].name == "phase59_matched_dimension_delta_table.csv"
    assert paths["comparison_json"].name == "phase59_matched_dimension_controls.json"
    assert paths["readiness_md"].name == "phase59_matched_dimension_controls.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase59_matched_dimension_status"] == "matched_dimension_geofm_supported"
    with paths["delta_csv"].open("r", encoding="utf-8", newline="") as handle:
        delta_rows = list(csv.DictReader(handle))
    assert any(
        row["compressed_variant_id"] == "D4P16"
        and row["matched_control_variant_id"] == "D5S16"
        for row in delta_rows
    )
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Matched-dimension control audit" in markdown
    assert "does not enable suitability reward" in markdown
