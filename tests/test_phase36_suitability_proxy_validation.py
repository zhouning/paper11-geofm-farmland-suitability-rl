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


def _base_row(index: int, label: int, split: str) -> dict[str, object]:
    row = {
        "block_id": f"b{index:02d}",
        "suitability_proxy": 0.8 if label else 0.2,
        "current_farmland_label": label,
        "farmland_or_orchard_label": label,
        "low_slope_farmland_label": label,
        "independent_proxy_label": label,
        "split": split,
    }
    for column_index in range(17):
        row[f"explicit_feature_{column_index:02d}"] = 0.0
    row["explicit_feature_00"] = float(index % 3) / 10.0
    row["explicit_feature_04"] = float(label)
    row["explicit_feature_07"] = 0.0
    row["explicit_feature_13"] = float(label)
    row["explicit_feature_16"] = float(label)
    for column_index in range(64):
        if column_index == 0:
            value = 2.0 if label else -2.0
        elif column_index == 1:
            value = 1.0 if label else -1.0
        else:
            value = float(((index + column_index) % 5) - 2) / 10.0
        row[f"embedding_mean_{column_index:02d}"] = value
    return row


def _write_variant_manifest(
    output_dir: Path,
    variants: dict[str, tuple[str, list[str]]],
) -> None:
    payload = {
        "claim_boundary": "test manifest",
        "variants": {
            variant_id: {
                "description": f"{variant_id} test variant",
                "state_groups": [],
                "reward": "base_planning_reward",
                "required_columns": columns,
                "ready": True,
                "missing": [],
                "feature_table": table_name,
                "row_count": 12,
            }
            for variant_id, (table_name, columns) in variants.items()
        },
    }
    (output_dir / "experiment_variants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    normalized_dir = tmp_path / "phase30_controls"
    rows = [
        _base_row(index, label=1 if index % 2 == 0 else 0, split="train" if index < 8 else "test")
        for index in range(12)
    ]
    explicit_columns = [f"explicit_feature_{index:02d}" for index in range(17)]
    embedding_columns = [f"embedding_mean_{index:02d}" for index in range(64)]
    block_columns = [
        "block_id",
        *explicit_columns,
        *embedding_columns,
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
        ["block_id", *explicit_columns],
    )
    _write_csv(
        phase2_dir / "variant_B1_features.csv",
        rows,
        ["block_id", *explicit_columns, *embedding_columns],
    )
    _write_csv(
        phase2_dir / "variant_B2_features.csv",
        rows,
        ["block_id", *explicit_columns, "suitability_proxy"],
    )
    _write_csv(
        phase2_dir / "variant_B3_features.csv",
        rows,
        ["block_id", *explicit_columns, *embedding_columns, "suitability_proxy"],
    )
    _write_variant_manifest(
        phase2_dir,
        {
            "B0": ("variant_B0_features.csv", explicit_columns),
            "B1": ("variant_B1_features.csv", [*explicit_columns, *embedding_columns]),
            "B2": ("variant_B2_features.csv", [*explicit_columns, "suitability_proxy"]),
            "B3": (
                "variant_B3_features.csv",
                [*explicit_columns, *embedding_columns, "suitability_proxy"],
            ),
        },
    )

    d2_rows = []
    d3_rows = []
    d4p8_rows = []
    d4p16_rows = []
    n1z_rows = []
    n1zr_rows = []
    for index, row in enumerate(rows):
        d2 = {"block_id": row["block_id"]}
        d3 = {"block_id": row["block_id"]}
        d4p8 = {"block_id": row["block_id"]}
        d4p16 = {"block_id": row["block_id"]}
        n1z = {"block_id": row["block_id"]}
        n1zr = {"block_id": row["block_id"]}
        for column in explicit_columns:
            for target in (d2, d3, d4p8, d4p16, n1z, n1zr):
                target[column] = row[column]
        for column_index, column in enumerate(embedding_columns):
            d2[column] = float(((index + column_index) % 7) - 3) / 5.0
            d3[column] = rows[(index + 1) % len(rows)][column]
            n1z[column] = row[column]
            n1zr[column] = row[column]
        for component in range(8):
            d4p8[f"embedding_pca_{component:02d}"] = row[embedding_columns[component]]
            d4p16[f"embedding_pca_{component:02d}"] = row[embedding_columns[component]]
        for component in range(8, 16):
            d4p16[f"embedding_pca_{component:02d}"] = row[embedding_columns[component]]
        d2_rows.append(d2)
        d3_rows.append(d3)
        d4p8_rows.append(d4p8)
        d4p16_rows.append(d4p16)
        n1z_rows.append(n1z)
        n1zr_rows.append(n1zr)

    _write_csv(phase8_dir / "variant_D2_features.csv", d2_rows, ["block_id", *explicit_columns, *embedding_columns])
    _write_csv(phase8_dir / "variant_D3_features.csv", d3_rows, ["block_id", *explicit_columns, *embedding_columns])
    _write_csv(
        phase8_dir / "variant_D4P8_features.csv",
        d4p8_rows,
        ["block_id", *explicit_columns, *[f"embedding_pca_{index:02d}" for index in range(8)]],
    )
    _write_csv(
        phase8_dir / "variant_D4P16_features.csv",
        d4p16_rows,
        ["block_id", *explicit_columns, *[f"embedding_pca_{index:02d}" for index in range(16)]],
    )
    _write_variant_manifest(
        phase8_dir,
        {
            "D2": ("variant_D2_features.csv", [*explicit_columns, *embedding_columns]),
            "D3": ("variant_D3_features.csv", [*explicit_columns, *embedding_columns]),
            "D4P8": (
                "variant_D4P8_features.csv",
                [*explicit_columns, *[f"embedding_pca_{index:02d}" for index in range(8)]],
            ),
            "D4P16": (
                "variant_D4P16_features.csv",
                [*explicit_columns, *[f"embedding_pca_{index:02d}" for index in range(16)]],
            ),
        },
    )

    _write_csv(normalized_dir / "variant_N1Z_features.csv", n1z_rows, ["block_id", *explicit_columns, *embedding_columns])
    _write_csv(normalized_dir / "variant_N1ZR_features.csv", n1zr_rows, ["block_id", *explicit_columns, *embedding_columns])
    _write_variant_manifest(
        normalized_dir,
        {
            "N1Z": ("variant_N1Z_features.csv", [*explicit_columns, *embedding_columns]),
            "N1ZR": ("variant_N1ZR_features.csv", [*explicit_columns, *embedding_columns]),
        },
    )

    return {
        "phase2_dir": phase2_dir,
        "phase8_dir": phase8_dir,
        "normalized_dir": normalized_dir,
    }


def test_phase36_builds_spatial_holdout_proxy_validation(tmp_path):
    from paper11_geofm.phase36_suitability_proxy_validation import (
        PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY,
        build_phase36_suitability_proxy_validation,
    )

    paths = _fixture_inputs(tmp_path)

    analysis = build_phase36_suitability_proxy_validation(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        normalized_controls_dir=paths["normalized_dir"],
        label_columns=[
            "current_farmland_label",
            "independent_proxy_label",
        ],
        min_delta=0.02,
    )

    assert analysis["phase"] == "phase36_suitability_proxy_validation"
    assert analysis["claim_boundary"] == PHASE36_SUITABILITY_PROXY_CLAIM_BOUNDARY
    assert analysis["phase36_proxy_validation_status"] in {
        "proxy_signal_supported_for_bounded_reward_smoke",
        "proxy_signal_not_supported",
    }
    assert analysis["row_counts"]["block_rows"] == 12
    assert analysis["row_counts"]["model_rows"] > 0
    assert analysis["label_summaries"]["current_farmland_label"][
        "label_leakage_risk"
    ] == "explicit_label_leakage_risk"
    assert analysis["label_summaries"]["independent_proxy_label"][
        "label_leakage_risk"
    ] == "not_flagged"

    rows = {
        (row["label_column"], row["feature_family"]): row
        for row in analysis["model_rows"]
    }
    raw = rows[("independent_proxy_label", "raw_geofm_only")]
    random = rows[("independent_proxy_label", "explicit_plus_random_geofm")]
    assert raw["validation_status"] == "evaluated"
    assert raw["roc_auc"] >= random["roc_auc"]
    assert raw["feature_count"] == 64
    assert isinstance(raw["top_coefficients"], list)


def test_phase36_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase36_suitability_proxy_validation import (
        build_phase36_suitability_proxy_validation,
        write_phase36_suitability_proxy_validation_artifacts,
    )

    paths = _fixture_inputs(tmp_path)
    analysis = build_phase36_suitability_proxy_validation(
        phase2_output_dir=paths["phase2_dir"],
        phase8_output_dir=paths["phase8_dir"],
        label_columns=["current_farmland_label", "independent_proxy_label"],
    )

    artifacts = write_phase36_suitability_proxy_validation_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifacts["label_summary_csv"].name == "phase36_label_summary.csv"
    assert artifacts["model_summary_csv"].name == "phase36_model_summary.csv"
    assert artifacts["diagnosis_json"].name == "phase36_suitability_proxy_validation.json"
    assert artifacts["diagnosis_md"].name == "phase36_suitability_proxy_validation.md"
    assert all(path.exists() for path in artifacts.values())
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase36_suitability_proxy_validation"
    markdown = artifacts["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 36 Suitability-Proxy Validation" in markdown
    assert "explicit_label_leakage_risk" in markdown


def test_phase36_missing_label_raises(tmp_path):
    from paper11_geofm.phase36_suitability_proxy_validation import (
        build_phase36_suitability_proxy_validation,
    )

    paths = _fixture_inputs(tmp_path)

    try:
        build_phase36_suitability_proxy_validation(
            phase2_output_dir=paths["phase2_dir"],
            label_columns=["missing_label"],
        )
    except ValueError as exc:
        assert "no requested label columns are available" in str(exc)
    else:
        raise AssertionError("missing label should raise")


def test_phase36_cli_writes_outputs(tmp_path):
    paths = _fixture_inputs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase36_suitability_proxy_validation"
        / "run_phase36_suitability_proxy_validation.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase2-output-dir",
            str(paths["phase2_dir"]),
            "--phase8-output-dir",
            str(paths["phase8_dir"]),
            "--normalized-controls-dir",
            str(paths["normalized_dir"]),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
            "--label-columns",
            "current_farmland_label,independent_proxy_label",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 36 proxy-validation status:" in result.stdout
    assert "Claim boundary:" in result.stdout
    assert (
        tmp_path
        / "cli_outputs"
        / "phase36_suitability_proxy_validation.json"
    ).exists()
