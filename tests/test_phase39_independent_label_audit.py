import csv
import json
import subprocess
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


def _phase2_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    phase2_dir = tmp_path / "phase2"
    if rows is None:
        rows = [
            {
                "block_id": f"b{index:03d}",
                "current_farmland_label": 1 if index % 2 == 0 else 0,
                "farmland_or_orchard_label": 1 if index % 3 == 0 else 0,
                "low_slope_farmland_label": 1 if index % 4 == 0 else 0,
                "source_bsm": f"s{index:03d}",
                "source_category": "farmland" if index % 2 == 0 else "other",
                "source_dlbm": "0101" if index % 2 == 0 else "0301",
                "source_dlmc": "paddy" if index % 2 == 0 else "forest",
                "split": "train" if index < 8 else "test",
            }
            for index in range(12)
        ]
    return _write_csv(
        phase2_dir / "block_geofm_features.csv",
        rows,
        [
            "block_id",
            "current_farmland_label",
            "farmland_or_orchard_label",
            "low_slope_farmland_label",
            "source_bsm",
            "source_category",
            "source_dlbm",
            "source_dlmc",
            "split",
        ],
    ).parent


def _external_labels(
    path: Path,
    values: list[int],
    block_ids: list[str] | None = None,
) -> Path:
    if block_ids is None:
        block_ids = [f"b{index:03d}" for index in range(len(values))]
    if len(block_ids) != len(values):
        raise ValueError("block_ids and values must have the same length")
    rows = [
        {"block_id": block_id, "irrigation_proxy_label": value}
        for block_id, value in zip(block_ids, values)
    ]
    return _write_csv(path, rows, ["block_id", "irrigation_proxy_label"])


def _registry(
    path: Path,
    provenance_class: str,
    allowed_for_phase38_rerun: str = "true",
) -> Path:
    rows = [
        {
            "label_column": "irrigation_proxy_label",
            "source_path": "external_irrigation.csv",
            "provenance_class": provenance_class,
            "description": "Synthetic non-DLTB irrigation proxy label",
            "external_source_name": "synthetic_irrigation_fixture",
            "independence_rationale": "not derived from DLTB, slope, or explicit planning features",
            "allowed_for_phase38_rerun": allowed_for_phase38_rerun,
        }
    ]
    return _write_csv(
        path,
        rows,
        [
            "label_column",
            "source_path",
            "provenance_class",
            "description",
            "external_source_name",
            "independence_rationale",
            "allowed_for_phase38_rerun",
        ],
    )


def test_phase39_current_labels_remain_missing_independent_inputs(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    requested_labels = [
        "current_farmland_label",
        "farmland_or_orchard_label",
        "low_slope_farmland_label",
    ]
    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_columns=",".join(requested_labels),
    )

    assert analysis["phase"] == "phase39_independent_label_audit"
    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_missing"
    readiness = analysis["label_readiness"]
    assert set(requested_labels).issubset(readiness)
    for label in requested_labels:
        assert readiness[label]["provenance_class"] == "explicit_label_leakage_risk"
        assert readiness[label]["allowed_for_phase38_rerun"] is False
    assert "does not train PPO" in analysis["claim_boundary"]


def test_phase39_source_fields_are_leakage_risks(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    requested_labels = ["source_category", "source_dlbm", "source_dlmc"]
    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_columns=requested_labels,
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_missing"
    readiness = analysis["label_readiness"]
    assert set(requested_labels).issubset(readiness)
    for label in requested_labels:
        assert readiness[label]["provenance_class"] == "source_field_leakage_risk"
        assert readiness[label]["allowed_for_phase38_rerun"] is False


def test_phase39_external_candidate_label_clears_phase38_rerun_gate(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    phase2_dir = _phase2_dir(tmp_path)
    shuffled_block_ids = [
        "b009",
        "b003",
        "b001",
        "b007",
        "b010",
        "b000",
        "b008",
        "b004",
        "b002",
        "b006",
        "b011",
        "b005",
    ]
    external = _external_labels(
        tmp_path / "external_irrigation.csv",
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        block_ids=shuffled_block_ids,
    )
    registry = _registry(tmp_path / "registry.csv", "candidate_independent_proxy")

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=phase2_dir,
        external_label_csvs=[external],
        label_registry=registry,
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_labels_ready_for_phase38_rerun"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["provenance_class"] == "candidate_independent_proxy"
    assert row["registry_entry_present"] is True
    assert row["allowed_for_phase38_rerun"] is True
    assert row["train_positive_count"] == 3
    assert row["eval_positive_count"] == 1


def test_phase39_unknown_or_blank_splits_do_not_create_eval_coverage(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    rows = []
    for index in range(8):
        if index < 4:
            split = "train"
        elif index % 2 == 0:
            split = ""
        else:
            split = "holdout"
        rows.append(
            {
                "block_id": f"b{index:03d}",
                "current_farmland_label": 1 if index % 2 == 0 else 0,
                "farmland_or_orchard_label": 1 if index % 3 == 0 else 0,
                "low_slope_farmland_label": 1 if index % 4 == 0 else 0,
                "source_bsm": f"s{index:03d}",
                "source_category": "farmland" if index % 2 == 0 else "other",
                "source_dlbm": "0101" if index % 2 == 0 else "0301",
                "source_dlmc": "paddy" if index % 2 == 0 else "forest",
                "split": split,
            }
        )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path, rows=rows),
        external_label_csvs=[
            _external_labels(
                tmp_path / "external_irrigation.csv",
                [1, 0, 1, 0, 1, 0, 1, 0],
                block_ids=[f"b{index:03d}" for index in range(8)],
            )
        ],
        label_registry=_registry(tmp_path / "registry.csv", "candidate_independent_proxy"),
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_insufficient"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["train_positive_count"] == 2
    assert row["train_negative_count"] == 2
    assert row["eval_count"] == 0
    assert row["usable"] is False
    assert row["allowed_for_phase38_rerun"] is False


def test_phase39_partial_external_candidate_coverage_is_insufficient(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[
            _external_labels(
                tmp_path / "external_irrigation.csv",
                [1, 0, 1, 0, 1, 0],
                block_ids=["b000", "b001", "b002", "b008", "b009", "b010"],
            )
        ],
        label_registry=_registry(tmp_path / "registry.csv", "candidate_independent_proxy"),
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_insufficient"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["usable"] is True
    assert row["join_missing_count"] == 6
    assert row["allowed_for_phase38_rerun"] is False
    assert "missing joined labels" in row["decision_reason"]


def test_phase39_registry_denied_usable_candidate_needs_review(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[
            _external_labels(
                tmp_path / "external_irrigation.csv",
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            )
        ],
        label_registry=_registry(
            tmp_path / "registry.csv",
            "candidate_independent_proxy",
            allowed_for_phase38_rerun="false",
        ),
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "candidate_proxy_labels_need_review"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["usable"] is True
    assert row["registry_allowed_for_phase38_rerun"] is False
    assert row["allowed_for_phase38_rerun"] is False


def test_phase39_unclassified_external_label_needs_review(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[
            _external_labels(
                tmp_path / "external_irrigation.csv",
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            )
        ],
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "candidate_proxy_labels_need_review"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["provenance_class"] == "unclassified"
    assert row["allowed_for_phase38_rerun"] is False


def test_phase39_rejects_blank_or_missing_registry_label_column(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    fieldnames = [
        "label_column",
        "source_path",
        "provenance_class",
        "description",
        "external_source_name",
        "independence_rationale",
        "allowed_for_phase38_rerun",
    ]
    base_row = {
        "label_column": "",
        "source_path": "external_irrigation.csv",
        "provenance_class": "candidate_independent_proxy",
        "description": "Synthetic non-DLTB irrigation proxy label",
        "external_source_name": "synthetic_irrigation_fixture",
        "independence_rationale": "not derived from DLTB, slope, or explicit planning features",
        "allowed_for_phase38_rerun": "true",
    }
    registries = [
        _write_csv(tmp_path / "blank_label_registry.csv", [base_row], fieldnames),
        _write_csv(
            tmp_path / "missing_label_registry.csv",
            [base_row],
            [field for field in fieldnames if field != "label_column"],
        ),
    ]

    for registry in registries:
        try:
            build_phase39_independent_label_audit(
                phase2_output_dir=_phase2_dir(tmp_path),
                label_registry=registry,
                label_columns=["current_farmland_label"],
            )
        except ValueError as exc:
            message = str(exc)
            assert "label_column" in message
            assert "blank or missing" in message
        else:
            raise AssertionError("blank or missing registry label_column should raise")


def test_phase39_single_class_candidate_is_insufficient(tmp_path):
    from paper11_geofm.phase39_independent_label_audit import (
        build_phase39_independent_label_audit,
    )

    analysis = build_phase39_independent_label_audit(
        phase2_output_dir=_phase2_dir(tmp_path),
        external_label_csvs=[_external_labels(tmp_path / "external_irrigation.csv", [1] * 12)],
        label_registry=_registry(tmp_path / "registry.csv", "candidate_independent_proxy"),
        label_columns=["irrigation_proxy_label"],
    )

    assert analysis["phase39_independent_label_audit_status"] == "independent_label_inputs_insufficient"
    row = analysis["label_readiness"]["irrigation_proxy_label"]
    assert row["usable"] is False
    assert row["allowed_for_phase38_rerun"] is False
    assert "both positive and negative labels" in row["decision_reason"]
