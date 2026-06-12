import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "variant_id",
        "train_tile_id",
        "eval_tile_id",
        "seed",
        "total_contract_reward",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_phase23_comparison(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase": "phase23_multi_seed_training_comparison",
                "learned_policy": {
                    "mean_reward_by_variant": {"B0": 0.5, "B1": 0.92},
                    "B1_minus_B0_mean_reward": 0.42,
                },
                "remaining_evidence_gaps": [
                    "held_out_region_transfer_evaluation",
                    "suitability_reward_validation_before_B2_B3",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_phase24_builds_ijaeog_evidence_package(tmp_path):
    from paper11_geofm.ijaeog_evidence_package import (
        PHASE24_CLAIM_BOUNDARY,
        build_phase24_ijaeog_evidence_package,
    )

    phase22_summary = _write_summary_csv(
        tmp_path / "phase22.csv",
        [
            {
                "row_type": "learned_block_scorer",
                "variant_id": "B0",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_eval_a",
                "seed": 0,
                "total_contract_reward": 1.0,
            },
            {
                "row_type": "learned_block_scorer",
                "variant_id": "B1",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_eval_a",
                "seed": 0,
                "total_contract_reward": 1.2,
            },
            {
                "row_type": "first_valid",
                "variant_id": "B0",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_eval_b",
                "seed": 1,
                "total_contract_reward": 0.1,
            },
            {
                "row_type": "seeded_random",
                "variant_id": "B1",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_eval_b",
                "seed": 1,
                "total_contract_reward": 0.2,
            },
        ],
    )
    phase23_summary = _write_summary_csv(
        tmp_path / "phase23.csv",
        [
            {
                "row_type": "trained_policy",
                "variant_id": "B0",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_train",
                "seed": 0,
                "total_contract_reward": 0.5,
            },
            {
                "row_type": "trained_policy",
                "variant_id": "B1",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_train",
                "seed": 0,
                "total_contract_reward": 0.9,
            },
            {
                "row_type": "first_valid",
                "variant_id": "B0",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_train",
                "seed": 1,
                "total_contract_reward": 0.1,
            },
            {
                "row_type": "seeded_random",
                "variant_id": "B1",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_train",
                "seed": 1,
                "total_contract_reward": 0.2,
            },
        ],
    )
    phase23_comparison = _write_phase23_comparison(tmp_path / "phase23_comparison.json")

    package = build_phase24_ijaeog_evidence_package(
        phase22_summary,
        phase23_summary,
        phase23_comparison,
    )

    assert package["phase"] == "phase24_ijaeog_evidence_package"
    assert package["phase22"]["summary_rows"] == 4
    assert package["phase22"]["eval_tile_count"] == 2
    assert package["phase23"]["summary_rows"] == 4
    assert package["phase23"]["B1_minus_B0_mean_reward"] == 0.42
    assert package["claim_readiness"]["same_tile_b0_b1_training_pilot"]["status"] == "pilot_supported"
    assert package["claim_readiness"]["multi_tile_scorer_interface"]["status"] == "pilot_supported"
    assert package["claim_readiness"]["submission_ready"]["status"] == "not_ready"
    assert package["claim_boundary"] == PHASE24_CLAIM_BOUNDARY


def test_phase24_writer_outputs_table_json_and_markdown(tmp_path):
    from paper11_geofm.ijaeog_evidence_package import (
        PHASE24_CLAIM_BOUNDARY,
        write_phase24_ijaeog_evidence_artifacts,
    )

    package = {
        "phase": "phase24_ijaeog_evidence_package",
        "phase22": {"summary_rows": 4},
        "phase23": {"summary_rows": 4, "B1_minus_B0_mean_reward": 0.42},
        "evidence_table": [
            {
                "claim_area": "same_tile_b0_b1_training_pilot",
                "status": "pilot_supported",
                "evidence": "Phase 23 has multi-seed same-tile B0/B1 results.",
                "remaining_gap": "longer training and held-out evaluation",
            }
        ],
        "claim_readiness": {
            "submission_ready": {
                "status": "not_ready",
                "evidence": "Pilot evidence only.",
                "remaining_gap": "full manuscript evidence",
            }
        },
        "claim_boundary": PHASE24_CLAIM_BOUNDARY,
    }

    paths = write_phase24_ijaeog_evidence_artifacts(package, tmp_path / "outputs")

    assert paths["evidence_csv"].name == "phase24_ijaeog_evidence_table.csv"
    assert paths["summary_json"].name == "phase24_ijaeog_evidence_summary.json"
    assert paths["claim_readiness_md"].name == "phase24_ijaeog_claim_readiness.md"
    rows = list(csv.DictReader(paths["evidence_csv"].open("r", encoding="utf-8")))
    assert rows[0]["status"] == "pilot_supported"
    saved = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == PHASE24_CLAIM_BOUNDARY
    markdown = paths["claim_readiness_md"].read_text(encoding="utf-8")
    assert "submission_ready" in markdown
    assert "not_ready" in markdown


def test_phase24_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase24_ijaeog_evidence_package"
        / "run_phase24_ijaeog_evidence_package.py"
    )
    spec = importlib.util.spec_from_file_location("phase24_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase22_summary = _write_summary_csv(
        tmp_path / "phase22.csv",
        [
            {
                "row_type": "learned_block_scorer",
                "variant_id": "B0",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_eval_a",
                "seed": 0,
                "total_contract_reward": 1.0,
            }
        ],
    )
    phase23_summary = _write_summary_csv(
        tmp_path / "phase23.csv",
        [
            {
                "row_type": "trained_policy",
                "variant_id": "B1",
                "train_tile_id": "tile_train",
                "eval_tile_id": "tile_train",
                "seed": 0,
                "total_contract_reward": 0.9,
            }
        ],
    )
    phase23_comparison = _write_phase23_comparison(tmp_path / "phase23_comparison.json")

    exit_code = module.main(
        [
            "--phase22-summary-csv",
            str(phase22_summary),
            "--phase23-summary-csv",
            str(phase23_summary),
            "--phase23-comparison-json",
            str(phase23_comparison),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 22 summary rows: 1" in stdout
    assert "Phase 23 summary rows: 1" in stdout
    assert "B1-B0 learned-policy mean reward delta: 0.42" in stdout
    assert "Submission readiness: not_ready" in stdout
    assert "phase24_ijaeog_evidence_summary.json" in stdout
    assert "Claim boundary: Phase 24 is a synthesis and claim-readiness package" in stdout
