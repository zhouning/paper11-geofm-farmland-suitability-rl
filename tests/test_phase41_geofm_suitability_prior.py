import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPLICIT_COLUMNS = [f"explicit_feature_{index:02d}" for index in range(4)]
EMBEDDING_COLUMNS = [f"embedding_mean_{index:02d}" for index in range(64)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _registry_csv(path: Path, label_column: str = "independent_suitability_label") -> Path:
    return _write_csv(
        path,
        [
            {
                "label_column": label_column,
                "label_source": "synthetic independent suitability fixture",
                "source_type": "external_soil",
                "independence_level": "independent",
                "allowed_eval_roles": "test,validation,eval",
                "provenance_note": "test fixture not derived from DLTB, slope, source metadata, or GeoFM",
                "license_or_access": "test fixture",
                "expected_positive_definition": "1",
            }
        ],
        [
            "label_column",
            "label_source",
            "source_type",
            "independence_level",
            "allowed_eval_roles",
            "provenance_note",
            "license_or_access",
            "expected_positive_definition",
        ],
    )


def _phase2_dir(
    tmp_path: Path,
    *,
    geofm_signal: bool = True,
    explicit_signal: bool = False,
    label_column: str = "independent_suitability_label",
    row_count: int = 48,
) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        split = "train" if index < row_count // 2 else "test"
        label = 1 if index % 4 in {0, 1} else 0
        row: dict[str, object] = {
            "block_id": f"b{index:03d}",
            "split": split,
            "tile_id": f"tile_{index % 6}",
            label_column: label,
        }
        for column_index, column in enumerate(EXPLICIT_COLUMNS):
            if explicit_signal and column_index == 0:
                value = 2.0 if label else -2.0
            else:
                value = ((index + column_index) % 5) / 10.0
            row[column] = value
        for column_index, column in enumerate(EMBEDDING_COLUMNS):
            if geofm_signal and column_index < 3:
                value = (3.0 - column_index * 0.25) if label else (-3.0 + column_index * 0.25)
            else:
                value = (((index + column_index) % 7) - 3) / 50.0
            row[column] = value
        rows.append(row)
    return _write_csv(
        tmp_path / "phase2" / "block_geofm_features.csv",
        rows,
        ["block_id", "split", "tile_id", label_column, *EXPLICIT_COLUMNS, *EMBEDDING_COLUMNS],
    ).parent


def test_phase41_no_registry_returns_missing_inputs(tmp_path):
    from paper11_geofm.phase41_geofm_suitability_prior import (
        run_phase41_geofm_suitability_prior,
    )

    result = run_phase41_geofm_suitability_prior(
        phase2_output_dir=_phase2_dir(tmp_path),
        label_registry=None,
        min_valid_count=20,
        min_split_valid_count=8,
    )

    assert result["phase"] == "phase41_geofm_suitability_prior"
    assert result["phase41_geofm_prior_status"] == "phase41_independent_label_inputs_missing"
    assert result["row_counts"]["phase40_passed_labels"] == 0
    assert not result["prior_rows"]


def test_phase41_supported_when_geofm_pca_beats_explicit_and_controls(tmp_path):
    from paper11_geofm.phase41_geofm_suitability_prior import (
        run_phase41_geofm_suitability_prior,
    )

    phase2_dir = _phase2_dir(tmp_path, geofm_signal=True, explicit_signal=False)
    registry = _registry_csv(tmp_path / "registry.csv")

    result = run_phase41_geofm_suitability_prior(
        phase2_output_dir=phase2_dir,
        label_registry=registry,
        min_valid_count=20,
        min_split_valid_count=8,
        min_auc_delta=0.05,
        min_ap_delta=0.05,
        min_positive_fold_fraction=1.0,
        n_pca_components=3,
    )

    assert result["phase41_geofm_prior_status"] == "geofm_suitability_prior_supported"
    assert result["supported_prior"]["label_column"] == "independent_suitability_label"
    families = {row["feature_family"] for row in result["metric_rows"]}
    assert {"explicit_only", "geofm_pca_only", "explicit_plus_geofm_pca", "geofm_shuffled_control", "geofm_random_control"} <= families
    assert result["prior_rows"]


def test_phase41_not_supported_when_geofm_adds_no_increment(tmp_path):
    from paper11_geofm.phase41_geofm_suitability_prior import (
        run_phase41_geofm_suitability_prior,
    )

    phase2_dir = _phase2_dir(tmp_path, geofm_signal=False, explicit_signal=True)
    registry = _registry_csv(tmp_path / "registry.csv")

    result = run_phase41_geofm_suitability_prior(
        phase2_output_dir=phase2_dir,
        label_registry=registry,
        min_valid_count=20,
        min_split_valid_count=8,
        min_auc_delta=0.05,
        min_ap_delta=0.05,
        n_pca_components=3,
    )

    assert result["phase41_geofm_prior_status"] == "geofm_suitability_prior_not_supported"
    assert not result["prior_rows"]


def test_phase41_control_failed_status_from_metric_rows():
    from paper11_geofm.phase41_geofm_suitability_prior import summarize_phase41_gate

    rows = [
        {"label_column": "external_label", "feature_family": "explicit_only", "roc_auc": 0.60, "average_precision": 0.60, "brier_score": 0.20, "positive_fold_fraction": 1.0},
        {"label_column": "external_label", "feature_family": "explicit_plus_geofm_pca", "roc_auc": 0.80, "average_precision": 0.80, "brier_score": 0.18, "positive_fold_fraction": 1.0},
        {"label_column": "external_label", "feature_family": "geofm_shuffled_control", "roc_auc": 0.82, "average_precision": 0.82, "brier_score": 0.18, "positive_fold_fraction": 1.0},
    ]

    summary = summarize_phase41_gate(
        metric_rows=rows,
        thresholds={"min_auc_delta": 0.05, "min_ap_delta": 0.05, "min_positive_fold_fraction": 0.67, "max_brier_regression": 0.02},
    )

    assert summary["phase41_geofm_prior_status"] == "geofm_suitability_prior_control_failed"
    assert summary["supported_prior"] is None


def test_phase41_artifacts_write_prior_only_when_supported(tmp_path):
    from paper11_geofm.phase41_geofm_suitability_prior import (
        run_phase41_geofm_suitability_prior,
        write_phase41_geofm_suitability_prior_artifacts,
    )

    result = run_phase41_geofm_suitability_prior(
        phase2_output_dir=_phase2_dir(tmp_path, geofm_signal=True, explicit_signal=False),
        label_registry=_registry_csv(tmp_path / "registry.csv"),
        min_valid_count=20,
        min_split_valid_count=8,
        min_auc_delta=0.05,
        min_ap_delta=0.05,
        n_pca_components=3,
    )
    artifacts = write_phase41_geofm_suitability_prior_artifacts(result, tmp_path / "outputs")

    assert artifacts["summary_csv"].name == "phase41_geofm_prior_summary.csv"
    assert artifacts["metrics_csv"].name == "phase41_geofm_prior_metrics.csv"
    assert artifacts["diagnosis_json"].name == "phase41_geofm_prior.json"
    assert artifacts["diagnosis_md"].name == "phase41_geofm_prior.md"
    assert artifacts["prior_csv"].name == "block_geofm_suitability_prior.csv"
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase41_geofm_prior_status"] == "geofm_suitability_prior_supported"


def test_phase41_cli_writes_outputs(tmp_path):
    phase2_dir = _phase2_dir(tmp_path, geofm_signal=True, explicit_signal=False)
    registry = _registry_csv(tmp_path / "registry.csv")
    output_dir = tmp_path / "outputs"
    runner = ROOT / "experiments" / "phase41_geofm_suitability_prior" / "run_phase41_geofm_suitability_prior.py"

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--phase2-output-dir",
            str(phase2_dir),
            "--label-registry",
            str(registry),
            "--output-dir",
            str(output_dir),
            "--min-valid-count",
            "20",
            "--min-split-valid-count",
            "8",
            "--min-auc-delta",
            "0.05",
            "--min-ap-delta",
            "0.05",
            "--n-pca-components",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 41 GeoFM prior status:" in result.stdout
    assert "geofm_suitability_prior_supported" in result.stdout
    assert (output_dir / "phase41_geofm_prior.json").exists()
    assert (output_dir / "block_geofm_suitability_prior.csv").exists()
