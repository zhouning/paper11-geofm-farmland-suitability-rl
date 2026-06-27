import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


EXPLICIT_COLUMNS = [f"explicit_feature_{index:02d}" for index in range(17)]
EMBEDDING_COLUMNS = [f"embedding_mean_{index:02d}" for index in range(64)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_variant_manifest(
    output_dir: Path,
    variants: dict[str, tuple[str, list[str]]],
) -> None:
    payload = {
        "claim_boundary": "phase38 test manifest",
        "variants": {
            variant_id: {
                "description": f"{variant_id} test variant",
                "state_groups": [],
                "reward": "base_planning_reward",
                "required_columns": columns,
                "ready": True,
                "missing": [],
                "feature_table": table_name,
                "row_count": 24,
            }
            for variant_id, (table_name, columns) in variants.items()
        },
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _base_row(index: int, split: str) -> dict[str, object]:
    independent_label = 1 if index % 4 in {0, 1} else 0
    leakage_label = 1 if index % 2 == 0 else 0
    row: dict[str, object] = {
        "block_id": f"b{index:03d}",
        "suitability_proxy": 0.7 if leakage_label else 0.3,
        "current_farmland_label": leakage_label,
        "farmland_or_orchard_label": leakage_label,
        "low_slope_farmland_label": leakage_label,
        "independent_proxy_label": independent_label,
        "split": split,
    }
    for column_index, column in enumerate(EXPLICIT_COLUMNS):
        row[column] = float((index + column_index) % 5) / 10.0
    row["explicit_feature_04"] = float(leakage_label)
    row["explicit_feature_13"] = float(leakage_label)
    for column_index, column in enumerate(EMBEDDING_COLUMNS):
        if column_index == 0:
            value = 3.0 if independent_label else -3.0
        elif column_index == 1:
            value = 1.5 if independent_label else -1.5
        else:
            value = float(((index + column_index) % 7) - 3) / 20.0
        row[column] = value
    return row


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    normalized_dir = tmp_path / "phase30_controls"
    rows = [
        _base_row(index, split="train" if index < 16 else "test")
        for index in range(24)
    ]
    block_columns = [
        "block_id",
        *EXPLICIT_COLUMNS,
        *EMBEDDING_COLUMNS,
        "suitability_proxy",
        "current_farmland_label",
        "farmland_or_orchard_label",
        "low_slope_farmland_label",
        "independent_proxy_label",
        "split",
    ]
    _write_csv(phase2_dir / "block_geofm_features.csv", rows, block_columns)
    _write_csv(
        phase2_dir / "variant_B0_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS],
    )
    _write_csv(
        phase2_dir / "variant_B1_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS],
    )
    _write_csv(
        phase2_dir / "variant_B2_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, "suitability_proxy"],
    )
    _write_csv(
        phase2_dir / "variant_B3_features.csv",
        rows,
        ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS, "suitability_proxy"],
    )
    _write_variant_manifest(
        phase2_dir,
        {
            "B0": ("variant_B0_features.csv", EXPLICIT_COLUMNS),
            "B1": ("variant_B1_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "B2": ("variant_B2_features.csv", [*EXPLICIT_COLUMNS, "suitability_proxy"]),
            "B3": (
                "variant_B3_features.csv",
                [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS, "suitability_proxy"],
            ),
        },
    )

    d2_rows: list[dict[str, object]] = []
    d3_rows: list[dict[str, object]] = []
    d4p8_rows: list[dict[str, object]] = []
    d4p16_rows: list[dict[str, object]] = []
    n1z_rows: list[dict[str, object]] = []
    n1zr_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        d2 = {"block_id": row["block_id"]}
        d3 = {"block_id": row["block_id"]}
        d4p8 = {"block_id": row["block_id"]}
        d4p16 = {"block_id": row["block_id"]}
        n1z = {"block_id": row["block_id"]}
        n1zr = {"block_id": row["block_id"]}
        for column in EXPLICIT_COLUMNS:
            for target in (d2, d3, d4p8, d4p16, n1z, n1zr):
                target[column] = row[column]
        for column_index, column in enumerate(EMBEDDING_COLUMNS):
            d2[column] = float(((index + column_index) % 11) - 5) / 10.0
            d3[column] = rows[(index + 3) % len(rows)][column]
            n1z[column] = row[column]
            n1zr[column] = row[column]
        for component in range(8):
            d4p8[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
            d4p16[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
        for component in range(8, 16):
            d4p16[f"embedding_pca_{component:02d}"] = row[EMBEDDING_COLUMNS[component]]
        d2_rows.append(d2)
        d3_rows.append(d3)
        d4p8_rows.append(d4p8)
        d4p16_rows.append(d4p16)
        n1z_rows.append(n1z)
        n1zr_rows.append(n1zr)

    _write_csv(phase8_dir / "variant_D2_features.csv", d2_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(phase8_dir / "variant_D3_features.csv", d3_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(
        phase8_dir / "variant_D4P8_features.csv",
        d4p8_rows,
        ["block_id", *EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(8)]],
    )
    _write_csv(
        phase8_dir / "variant_D4P16_features.csv",
        d4p16_rows,
        ["block_id", *EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(16)]],
    )
    _write_variant_manifest(
        phase8_dir,
        {
            "D2": ("variant_D2_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "D3": ("variant_D3_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "D4P8": (
                "variant_D4P8_features.csv",
                [*EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(8)]],
            ),
            "D4P16": (
                "variant_D4P16_features.csv",
                [*EXPLICIT_COLUMNS, *[f"embedding_pca_{index:02d}" for index in range(16)]],
            ),
        },
    )

    _write_csv(normalized_dir / "variant_N1Z_features.csv", n1z_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_csv(normalized_dir / "variant_N1ZR_features.csv", n1zr_rows, ["block_id", *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS])
    _write_variant_manifest(
        normalized_dir,
        {
            "N1Z": ("variant_N1Z_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
            "N1ZR": ("variant_N1ZR_features.csv", [*EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS]),
        },
    )
    return {
        "phase2_dir": phase2_dir,
        "phase8_dir": phase8_dir,
        "normalized_dir": normalized_dir,
    }


def test_phase38_classifies_label_boundaries_and_blocks_reward_unlock(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=["current_farmland_label", "independent_proxy_label"],
        label_classifications="independent_proxy_label:candidate_independent_proxy",
        model_families=["logistic_elastic_net"],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    assert analysis["phase"] == "phase38_proxy_rebuild"
    assert analysis["phase38_proxy_rebuild_status"] == "proxy_rebuild_supported_for_bounded_b2_b3_smoke"
    assert analysis["label_summaries"]["current_farmland_label"]["label_classification"] == "explicit_label_leakage_risk"
    assert analysis["label_summaries"]["independent_proxy_label"]["label_classification"] == "candidate_independent_proxy"
    assert analysis["row_counts"]["rebuilt_proxy_score_rows"] > 0
    assert "does not run PPO" in analysis["claim_boundary"]


def test_phase38_leakage_only_result_stays_diagnostic(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=["current_farmland_label"],
        model_families=["logistic_elastic_net"],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    assert analysis["phase38_proxy_rebuild_status"] == "proxy_rebuild_diagnostic_only"
    assert analysis["label_summaries"]["current_farmland_label"]["label_classification"] == "explicit_label_leakage_risk"


def test_phase38_missing_label_raises(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    try:
        build_phase38_proxy_rebuild(
            phase2_output_dir=paths["phase2_dir"],
            label_columns=["missing_label"],
        )
    except ValueError as exc:
        assert "no requested label columns are available" in str(exc)
    else:
        raise AssertionError("missing label should raise")


def test_phase38_rejects_unknown_model_family(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    try:
        build_phase38_proxy_rebuild(
            phase2_output_dir=paths["phase2_dir"],
            label_columns=["current_farmland_label"],
            model_families="logistic_elastic_net,unknown_model",
        )
    except ValueError as exc:
        assert "unknown model families" in str(exc)
    else:
        raise AssertionError("unknown model family should raise")


def test_phase38_raises_when_no_feature_families_load(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    phase2_dir = tmp_path / "phase2"
    rows = [
        _base_row(index, split="train" if index < 16 else "test")
        for index in range(24)
    ]
    _write_csv(
        phase2_dir / "block_geofm_features.csv",
        rows,
        ["block_id", "current_farmland_label", "split"],
    )

    try:
        build_phase38_proxy_rebuild(
            phase2_output_dir=phase2_dir,
            label_columns=["current_farmland_label"],
        )
    except ValueError as exc:
        assert "found no usable feature families" in str(exc)
    else:
        raise AssertionError("missing feature families should raise")


def test_phase38_model_rows_include_calibration_and_diagnostics(tmp_path):
    from paper11_geofm.phase38_proxy_rebuild import build_phase38_proxy_rebuild

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase38_proxy_rebuild(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=["independent_proxy_label"],
        label_classifications={"independent_proxy_label": "candidate_independent_proxy"},
        model_families=[
            "logistic_elastic_net",
            "random_forest",
            "hist_gradient_boosting",
        ],
        min_auc_delta=0.01,
        min_ap_delta=0.01,
    )

    evaluated = [
        row for row in analysis["model_rows"] if row["validation_status"] == "evaluated"
    ]
    assert evaluated
    for row in evaluated:
        assert isinstance(row["calibration_bins"], list)
        assert row["calibration_bins"]
        first_bin = row["calibration_bins"][0]
        assert {
            "bin",
            "count",
            "mean_probability",
            "positive_rate",
        }.issubset(first_bin)
        assert isinstance(row["top_diagnostics"], list)

    logistic = next(
        row for row in evaluated if row["model_family"] == "logistic_elastic_net"
    )
    random_forest = next(
        row for row in evaluated if row["model_family"] == "random_forest"
    )
    hist_gradient = next(
        row for row in evaluated if row["model_family"] == "hist_gradient_boosting"
    )
    assert logistic["top_diagnostics"]
    assert {"feature", "coefficient"}.issubset(logistic["top_diagnostics"][0])
    assert random_forest["top_diagnostics"]
    assert {"feature", "importance"}.issubset(random_forest["top_diagnostics"][0])
    assert hist_gradient["top_diagnostics"] == []
