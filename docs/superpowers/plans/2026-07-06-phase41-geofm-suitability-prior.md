# Phase 41 GeoFM Suitability Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 41 as an independent-label-calibrated GeoFM suitability-prior gate before any later B2/B3 reward or action-prior experiment.

**Architecture:** Add one focused Phase 41 module that reuses the Phase 40 independent-label registry gate, evaluates GeoFM readouts against explicit and representation-control families, and writes reviewer-facing artifacts. Add one thin CLI runner, focused pytest coverage, one reviewer-facing result document, and documentation updates. Phase 41 does not run PPO, alter reward, or claim planning-policy improvement.

**Tech Stack:** Python standard library (`csv`, `json`, `dataclasses`, `pathlib`, `hashlib`), NumPy, scikit-learn (`LogisticRegression`, `StandardScaler`, `PCA`, metrics), pytest, existing Paper11 experiment runner layout, ignored `experiments/**/outputs/` artifact convention.

---

## Decision Backbone

Phase 41 is an admission gate, not a rescue experiment:

- If Phase 40 has no passed independent label, Phase 41 returns `phase41_independent_label_inputs_missing`.
- If an independent label exists but GeoFM does not improve over explicit-only under controls, Phase 41 returns `geofm_suitability_prior_not_supported`.
- If GeoFM appears to improve but shuffled or random controls also pass, Phase 41 returns `geofm_suitability_prior_control_failed`.
- Only if an independent label passes Phase 40 and GeoFM clears baseline, control, fold-stability, and calibration checks does Phase 41 return `geofm_suitability_prior_supported` and write `block_geofm_suitability_prior.csv`.

Do not let Phase 41 re-open B2/B3 automatically. A supported Phase 41 prior only authorizes a later bounded low-dimensional prior interface phase.

## File Structure

- Create: `tests/test_phase41_geofm_suitability_prior.py`
  - Synthetic fixtures for Phase 2 feature tables, Phase 40-compatible registries, supported/not-supported/control-failed status rules, artifact writing, and CLI behavior.
- Create: `src/paper11_geofm/phase41_geofm_suitability_prior.py`
  - Owns constants, thresholds, feature-family construction, deterministic spatial split fallback, readout fitting/evaluation, gate reduction, prior creation, and artifact writing.
- Create: `experiments/phase41_geofm_suitability_prior/run_phase41_geofm_suitability_prior.py`
  - Thin argparse runner that calls the module and prints the decision.
- Create after the current real no-registry run: `paper/phase28_results/15_phase41_geofm_suitability_prior.md`
  - Reviewer-facing interpretation of the current Phase 41 status.
- Modify after the real run: `README.md`
  - Add Phase 41 runner and status boundary.
- Modify after the real run: `paper/phase28_results/README.md`
  - Add Phase 41 entry and reproduction command.
- Modify after the real run: `paper/submission/01_ijaeog_submission_readiness.md`
  - Record that Phase 41 remains blocked until Phase 40 has a passed independent label.
- Modify after the real run: `paper/submission/03_conclusion_manuscript_draft.md`
  - Add Phase 41 as a future gate design, not as new positive evidence.
- Modify after the real run: `reproducibility/FILE_MANIFEST.tsv`
  - Add Phase 41 plan, source, runner, test, and result-doc entries.
- Modify after the real run: `docs/superpowers/phase33_current_progress_handoff.md`
  - Record Phase 41 design/implementation status and the decision boundary.

## Task 1: Failing Tests For Phase 41 Gate Behavior

**Files:**
- Create: `tests/test_phase41_geofm_suitability_prior.py`
- Target later: `src/paper11_geofm/phase41_geofm_suitability_prior.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phase41_geofm_suitability_prior.py` with fixtures and tests covering no-registry, supported, not-supported, control-failed, artifact writing, and CLI behavior:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_phase41_geofm_suitability_prior.py -q --basetemp=.pytest_tmp_phase41_t1 -p no:cacheprovider
```

Expected: FAIL because `paper11_geofm.phase41_geofm_suitability_prior` and the Phase 41 runner do not exist.

## Task 2: Core Phase 41 Module

**Files:**
- Create: `src/paper11_geofm/phase41_geofm_suitability_prior.py`
- Test: `tests/test_phase41_geofm_suitability_prior.py`

- [ ] **Step 1: Implement the module skeleton and public API**

Create `src/paper11_geofm/phase41_geofm_suitability_prior.py` with these public functions and constants:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .block_schema import EMBEDDING_COLUMNS
from .phase40_independent_label_gate import (
    evaluate_label_candidate,
    load_label_registry,
    Phase40Thresholds,
)

PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY = (
    "Phase 41 evaluates whether an independent-label-calibrated GeoFM "
    "suitability prior clears baseline, representation-control, fold-stability, "
    "and calibration checks. It does not train PPO, does not alter rewards, "
    "does not enable B2/B3, and does not prove planning-policy improvement."
)

PHASE41_SUMMARY_FIELDNAMES = (
    "phase41_geofm_prior_status",
    "label_column",
    "supported_feature_family",
    "decision_reason",
    "claim_boundary",
)

PHASE41_METRIC_FIELDNAMES = (
    "label_column",
    "feature_family",
    "feature_count",
    "fold_count",
    "positive_fold_count",
    "positive_fold_fraction",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "brier_score",
    "auc_delta_vs_explicit",
    "ap_delta_vs_explicit",
    "brier_delta_vs_explicit",
    "gate_role",
    "claim_boundary",
)

PHASE41_PRIOR_FIELDNAMES = (
    "block_id",
    "label_column",
    "calibrated_suitability_prior",
    "prior_uncertainty",
    "feature_family",
    "model_family",
    "claim_boundary",
)


@dataclass(frozen=True)
class Phase41Thresholds:
    min_valid_count: int = 100
    max_missing_rate: float = 0.20
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98
    min_split_valid_count: int = 20
    min_auc_delta: float = 0.03
    min_ap_delta: float = 0.03
    min_positive_fold_fraction: float = 0.67
    max_brier_regression: float = 0.02
    n_pca_components: int = 8
```

- [ ] **Step 2: Implement CSV reading, feature detection, and Phase 40 label selection**

Add helper functions:

```python
def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _embedding_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = set(rows[0])
    return [column for column in EMBEDDING_COLUMNS if column in columns]


def _explicit_columns(rows: Sequence[Mapping[str, str]], label_columns: set[str]) -> list[str]:
    if not rows:
        return []
    excluded = {"block_id", "split", "tile_id", "suitability_proxy", *label_columns}
    excluded.update(column for column in rows[0] if column.startswith("embedding_mean_"))
    excluded.update(column for column in rows[0] if column.endswith("_label"))
    candidates: list[str] = []
    for column in rows[0]:
        if column in excluded:
            continue
        if all(_numeric(row.get(column)) is not None for row in rows[: min(len(rows), 10)]):
            candidates.append(column)
    return candidates


def select_phase40_passed_labels(
    feature_rows: Sequence[Mapping[str, str]],
    label_registry: Path | str | None,
    thresholds: Phase41Thresholds,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry_rows = load_label_registry(label_registry)
    phase40_thresholds = Phase40Thresholds(
        min_valid_count=thresholds.min_valid_count,
        max_missing_rate=thresholds.max_missing_rate,
        min_positive_rate=thresholds.min_positive_rate,
        max_positive_rate=thresholds.max_positive_rate,
        min_split_valid_count=thresholds.min_split_valid_count,
    )
    label_gate_rows = [
        evaluate_label_candidate(feature_rows, row, phase40_thresholds)
        for row in registry_rows
    ]
    passed = [row for row in label_gate_rows if row.get("label_gate_status") == "label_gate_passed"]
    return passed, label_gate_rows
```

- [ ] **Step 3: Implement deterministic split, feature matrices, and readout evaluation**

Add functions:

```python
def _split_role(row: Mapping[str, str]) -> str:
    text = str(row.get("split", "")).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"test", "eval", "evaluation", "validation", "val"}:
        return "eval"
    digest = hashlib.sha1(str(row.get("block_id", "")).encode("utf-8")).hexdigest()
    return "eval" if int(digest[:2], 16) % 5 == 0 else "train"


def _labels(rows: Sequence[Mapping[str, str]], label_column: str) -> np.ndarray:
    return np.asarray([int(float(row[label_column])) for row in rows], dtype=int)


def _matrix(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def _safe_metric(value: float) -> float:
    return round(float(value), 10) if math.isfinite(float(value)) else 0.0
```

Implement this function:

```python
def evaluate_feature_family(
    rows: Sequence[Mapping[str, str]],
    label_column: str,
    feature_family: str,
    feature_matrix: np.ndarray,
) -> dict[str, object]:
    train_mask = np.asarray([_split_role(row) == "train" for row in rows])
    eval_mask = np.asarray([_split_role(row) == "eval" for row in rows])
    y = _labels(rows, label_column)
    train_y = y[train_mask]
    eval_y = y[eval_mask]
    if len(set(train_y.tolist())) < 2 or len(set(eval_y.tolist())) < 2:
        return _blocked_metric_row(label_column, feature_family, feature_matrix.shape[1])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(feature_matrix[train_mask], train_y)
    scores = model.predict_proba(feature_matrix[eval_mask])[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return {
        "label_column": label_column,
        "feature_family": feature_family,
        "feature_count": int(feature_matrix.shape[1]),
        "fold_count": 1,
        "positive_fold_count": 1,
        "positive_fold_fraction": 1.0,
        "roc_auc": _safe_metric(roc_auc_score(eval_y, scores)),
        "average_precision": _safe_metric(average_precision_score(eval_y, scores)),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score(eval_y, predictions)),
        "brier_score": _safe_metric(brier_score_loss(eval_y, scores)),
        "gate_role": "candidate" if "geofm" in feature_family else "baseline",
        "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
    }
```

This function uses train rows where `_split_role(row) == "train"`, eval rows where `_split_role(row) == "eval"`, and returns blocked metrics with zero scores when train/eval has a single class.

- [ ] **Step 4: Implement feature families and gate reduction**

Build these families for each label:

```python
feature_families = {
    "explicit_only": explicit_columns,
    "geofm_raw_only": embedding_columns,
    "geofm_pca_only": pca_columns,
    "explicit_plus_geofm_pca": explicit_columns + pca_columns,
    "geofm_shuffled_control": shuffled_pca_columns,
    "geofm_random_control": random_pca_columns,
}
```

Use train-fitted PCA for `geofm_pca_only` and deterministic row-wise shuffling/random controls with `np.random.default_rng(0)`.

Implement `summarize_phase41_gate(metric_rows, thresholds)` with this rule:

```python
candidate_passes = (
    auc_delta_vs_explicit >= min_auc_delta
    or ap_delta_vs_explicit >= min_ap_delta
) and positive_fold_fraction >= min_positive_fold_fraction
and brier_delta_vs_explicit <= max_brier_regression
```

If `geofm_shuffled_control` or `geofm_random_control` also passes, return `geofm_suitability_prior_control_failed`. If `explicit_plus_geofm_pca` or `geofm_pca_only` passes and controls do not, return `geofm_suitability_prior_supported`. Otherwise return `geofm_suitability_prior_not_supported`.

- [ ] **Step 5: Implement top-level run and prior rows**

Implement:

```python
def run_phase41_geofm_suitability_prior(
    phase2_output_dir: Path | str,
    label_registry: Path | str | None = None,
    min_valid_count: int = 100,
    max_missing_rate: float = 0.20,
    min_positive_rate: float = 0.02,
    max_positive_rate: float = 0.98,
    min_split_valid_count: int = 20,
    min_auc_delta: float = 0.03,
    min_ap_delta: float = 0.03,
    min_positive_fold_fraction: float = 0.67,
    max_brier_regression: float = 0.02,
    n_pca_components: int = 8,
) -> dict[str, object]:
    phase2_dir = Path(phase2_output_dir)
    feature_rows = _read_csv_rows(
        phase2_dir / "block_geofm_features.csv",
        "Phase 2 block feature CSV",
    )
    thresholds = Phase41Thresholds(
        min_valid_count=min_valid_count,
        max_missing_rate=max_missing_rate,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        min_split_valid_count=min_split_valid_count,
        min_auc_delta=min_auc_delta,
        min_ap_delta=min_ap_delta,
        min_positive_fold_fraction=min_positive_fold_fraction,
        max_brier_regression=max_brier_regression,
        n_pca_components=n_pca_components,
    )
    passed_labels, label_gate_rows = select_phase40_passed_labels(
        feature_rows,
        label_registry,
        thresholds,
    )
    if not passed_labels:
        return _phase41_missing_result(phase2_dir, thresholds, label_gate_rows, feature_rows)
    metric_rows = _evaluate_passed_labels(feature_rows, passed_labels, thresholds)
    gate_summary = summarize_phase41_gate(metric_rows, thresholds.__dict__)
    prior_rows = _build_prior_rows(feature_rows, gate_summary, thresholds)
    return _phase41_result_payload(
        phase2_dir=phase2_dir,
        thresholds=thresholds,
        label_gate_rows=label_gate_rows,
        metric_rows=metric_rows,
        gate_summary=gate_summary,
        prior_rows=prior_rows,
    )
```

The returned dict must include:

```python
{
    "phase": "phase41_geofm_suitability_prior",
    "phase41_geofm_prior_status": status,
    "thresholds": thresholds.__dict__,
    "row_counts": {
        "feature_rows": len(feature_rows),
        "phase40_label_gate_rows": len(label_gate_rows),
        "phase40_passed_labels": len(passed_labels),
        "metric_rows": len(metric_rows),
        "prior_rows": len(prior_rows),
    },
    "phase40_label_gate_rows": label_gate_rows,
    "metric_rows": metric_rows,
    "summary_rows": summary_rows,
    "supported_prior": supported_prior,
    "prior_rows": prior_rows,
    "claim_boundary": PHASE41_GEOFM_PRIOR_CLAIM_BOUNDARY,
}
```

Create `prior_rows` only for `geofm_suitability_prior_supported`. Use the supported family probabilities fitted on all valid rows, and set `prior_uncertainty = min(score, 1.0 - score)`.

- [ ] **Step 6: Implement artifact writing**

Implement `write_phase41_geofm_suitability_prior_artifacts(analysis, output_dir)` so it always writes:

```text
phase41_geofm_prior_summary.csv
phase41_geofm_prior_metrics.csv
phase41_geofm_prior.json
phase41_geofm_prior.md
```

It writes `block_geofm_suitability_prior.csv` only when `analysis["prior_rows"]` is non-empty. Return an artifact dict with `"prior_csv": None` when no prior is written.

- [ ] **Step 7: Run focused module tests**

Run:

```powershell
python -m pytest tests\test_phase41_geofm_suitability_prior.py -q --basetemp=.pytest_tmp_phase41_t2 -p no:cacheprovider
```

Expected: all tests except the CLI test pass.

- [ ] **Step 8: Commit core module and tests**

Run:

```powershell
git add tests\test_phase41_geofm_suitability_prior.py src\paper11_geofm\phase41_geofm_suitability_prior.py
git commit -m "feat: add Phase 41 GeoFM suitability prior gate"
```

Expected: commit succeeds.

## Task 3: Thin CLI Runner

**Files:**
- Create: `experiments/phase41_geofm_suitability_prior/run_phase41_geofm_suitability_prior.py`
- Test: `tests/test_phase41_geofm_suitability_prior.py`

- [ ] **Step 1: Add the runner**

Create `experiments/phase41_geofm_suitability_prior/run_phase41_geofm_suitability_prior.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase41_geofm_suitability_prior import (
    run_phase41_geofm_suitability_prior,
    write_phase41_geofm_suitability_prior_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Paper11 Phase 41 GeoFM suitability prior gate.")
    parser.add_argument("--phase2-output-dir", type=Path, required=True)
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-count", type=int, default=100)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--min-positive-rate", type=float, default=0.02)
    parser.add_argument("--max-positive-rate", type=float, default=0.98)
    parser.add_argument("--min-split-valid-count", type=int, default=20)
    parser.add_argument("--min-auc-delta", type=float, default=0.03)
    parser.add_argument("--min-ap-delta", type=float, default=0.03)
    parser.add_argument("--min-positive-fold-fraction", type=float, default=0.67)
    parser.add_argument("--max-brier-regression", type=float, default=0.02)
    parser.add_argument("--n-pca-components", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        analysis = run_phase41_geofm_suitability_prior(
            phase2_output_dir=args.phase2_output_dir,
            label_registry=args.label_registry,
            min_valid_count=args.min_valid_count,
            max_missing_rate=args.max_missing_rate,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_split_valid_count=args.min_split_valid_count,
            min_auc_delta=args.min_auc_delta,
            min_ap_delta=args.min_ap_delta,
            min_positive_fold_fraction=args.min_positive_fold_fraction,
            max_brier_regression=args.max_brier_regression,
            n_pca_components=args.n_pca_components,
        )
        artifacts = write_phase41_geofm_suitability_prior_artifacts(analysis, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 41 GeoFM prior status: {analysis['phase41_geofm_prior_status']}")
    print(f"Summary CSV: {artifacts['summary_csv']}")
    print(f"Metrics CSV: {artifacts['metrics_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    if artifacts.get("prior_csv") is not None:
        print(f"Suitability prior CSV: {artifacts['prior_csv']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run CLI test**

Run:

```powershell
python -m pytest tests\test_phase41_geofm_suitability_prior.py::test_phase41_cli_writes_outputs -q --basetemp=.pytest_tmp_phase41_cli -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 3: Run all Phase 41 tests**

Run:

```powershell
python -m pytest tests\test_phase41_geofm_suitability_prior.py -q --basetemp=.pytest_tmp_phase41_all -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 4: Commit the runner**

Run:

```powershell
git add experiments\phase41_geofm_suitability_prior\run_phase41_geofm_suitability_prior.py tests\test_phase41_geofm_suitability_prior.py
git commit -m "feat: add Phase 41 prior runner"
```

Expected: commit succeeds.

## Task 4: Real No-Registry Run And Reviewer-Facing Result

**Files:**
- Create generated local outputs under ignored `experiments/phase41_geofm_suitability_prior/outputs/real_bishan_no_registry`
- Create: `paper/phase28_results/15_phase41_geofm_suitability_prior.md`
- Modify: `paper/phase28_results/README.md`

- [ ] **Step 1: Run Phase 41 on real Bishan without a registry**

Run:

```powershell
python experiments\phase41_geofm_suitability_prior\run_phase41_geofm_suitability_prior.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase41_geofm_suitability_prior\outputs\real_bishan_no_registry
```

Expected stdout includes:

```text
Phase 41 GeoFM prior status: phase41_independent_label_inputs_missing
```

- [ ] **Step 2: Inspect generated JSON**

Run:

```powershell
Get-Content -Raw experiments\phase41_geofm_suitability_prior\outputs\real_bishan_no_registry\phase41_geofm_prior.json
```

Expected JSON fields:

```json
{
  "phase": "phase41_geofm_suitability_prior",
  "phase41_geofm_prior_status": "phase41_independent_label_inputs_missing"
}
```

- [ ] **Step 3: Create reviewer-facing Phase 41 result document**

Create `paper/phase28_results/15_phase41_geofm_suitability_prior.md`:

```markdown
# Phase 41 GeoFM Suitability Prior Gate

Phase 41 tests whether GeoFM can be used more safely as a low-dimensional
independent-label-calibrated suitability prior rather than as raw 64-dimensional
policy state.

## Current Real Bishan Run

The current real run used:

```text
experiments/phase11_bishan_dltb_real/outputs/phase2_real
```

No independent label registry was supplied. The current status is:

```text
phase41_independent_label_inputs_missing
```

## Interpretation

This status does not mean GeoFM is permanently useless. It means Paper11 still
does not have the independent label evidence required to calibrate GeoFM into a
defensible suitability prior. Therefore Phase 41 cannot generate
`block_geofm_suitability_prior.csv` for the current real run.

## Claim Boundary

Phase 41 does not run PPO, alter rewards, enable B2/B3, prove suitability, or
support planning-policy improvement. It only tests whether a Phase 40-passed
independent label allows GeoFM to clear baseline, control, fold-stability, and
calibration checks.

## Next Step

If the authors supply an external independent label registry, rerun Phase 40
and then Phase 41. Only a real `geofm_suitability_prior_supported` Phase 41
result should authorize a later bounded low-dimensional prior experiment.
```

- [ ] **Step 4: Update Phase 28 results README**

Add a bullet for `15_phase41_geofm_suitability_prior.md` and add the reproduction command from Step 1. Add this boundary text:

```text
The current Phase 41 status is phase41_independent_label_inputs_missing. Phase
41 therefore does not produce a calibrated GeoFM suitability prior for the real
Bishan run, and B2/B3 remains blocked.
```

- [ ] **Step 5: Commit result document**

Run:

```powershell
git add paper\phase28_results\README.md paper\phase28_results\15_phase41_geofm_suitability_prior.md
git commit -m "docs: record Phase 41 GeoFM prior gate result"
```

Expected: commit succeeds.

## Task 5: Submission And Repository Documentation

**Files:**
- Modify: `README.md`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`
- Modify: `paper/submission/03_conclusion_manuscript_draft.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Update README**

Add Phase 41 to the repository layout list and entry points. Add this status paragraph near the Phase 40 paragraph:

```text
Phase 41 implements the revised GeoFM route: GeoFM may be used only as an
independent-label-calibrated low-dimensional suitability prior that clears
baseline, shuffled-control, random-control, fold-stability, and calibration
checks. The current real no-registry run reports
`phase41_independent_label_inputs_missing`, so no calibrated prior is produced
and B2/B3 remains blocked.
```

Add runner command:

```powershell
python experiments\phase41_geofm_suitability_prior\run_phase41_geofm_suitability_prior.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase41_geofm_suitability_prior\outputs\real_bishan_no_registry
```

- [ ] **Step 2: Update submission readiness audit**

In `paper/submission/01_ijaeog_submission_readiness.md`, add Phase 41 to the readiness table and add:

```text
Phase 41 changes the proposed GeoFM route from raw embedding concatenation to a
strict prior gate. The current real status remains
`phase41_independent_label_inputs_missing`, so no calibrated GeoFM suitability
prior exists for the manuscript and B2/B3 remains blocked.
```

- [ ] **Step 3: Update conclusion manuscript**

In `paper/submission/03_conclusion_manuscript_draft.md`, add one sentence to the Discussion boundary paragraph:

```text
Phase 41 defines a future route for GeoFM as a calibrated suitability prior,
but the current real run cannot produce that prior because Phase 40 has no
accepted independent label registry.
```

Do not change the manuscript conclusion from negative to positive.

- [ ] **Step 4: Update file manifest**

Append entries to `reproducibility/FILE_MANIFEST.tsv`:

```text
docs/superpowers/plans/2026-07-06-phase41-geofm-suitability-prior.md	implementation_plan	Phase 41 implementation plan.
src/paper11_geofm/phase41_geofm_suitability_prior.py	source	Phase 41 independent-label-calibrated GeoFM suitability prior gate module.
experiments/phase41_geofm_suitability_prior/run_phase41_geofm_suitability_prior.py	experiment_runner	Phase 41 GeoFM suitability prior gate CLI runner.
tests/test_phase41_geofm_suitability_prior.py	test	Tests for Phase 41 prior support, control failure, artifacts, and CLI behavior.
paper/phase28_results/15_phase41_geofm_suitability_prior.md	results	Reviewer-facing interpretation of the Phase 41 real no-registry prior gate result.
```

- [ ] **Step 5: Update handoff**

Append a Phase 41 section to `docs/superpowers/phase33_current_progress_handoff.md`:

```markdown
## Phase 41 GeoFM Suitability Prior Gate

Phase 41 implements the revised route for making GeoFM useful: it blocks raw
64-dimensional GeoFM state injection and requires an independent-label-
calibrated low-dimensional prior that clears explicit baseline, shuffled
control, random control, fold-stability, and calibration checks.

Current real no-registry status:

```text
phase41_independent_label_inputs_missing
```

Decision: no calibrated GeoFM suitability prior exists for the current real
run. B2/B3 remains blocked until a real independent label registry passes Phase
40 and Phase 41 reports `geofm_suitability_prior_supported`.
```

- [ ] **Step 6: Commit documentation**

Run:

```powershell
git add README.md paper\submission\01_ijaeog_submission_readiness.md paper\submission\03_conclusion_manuscript_draft.md reproducibility\FILE_MANIFEST.tsv docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: update Paper11 gate after Phase 41"
```

Expected: commit succeeds.

## Task 6: Final Verification And Handoff

**Files:**
- Verify all edited files.

- [ ] **Step 1: Run focused Phase 41 tests**

Run:

```powershell
python -m pytest tests\test_phase41_geofm_suitability_prior.py -q --basetemp=.pytest_tmp_phase41_final -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent gate tests**

Run:

```powershell
python -m pytest tests\test_phase40_independent_label_gate.py tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase41_adjacent -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected stdout includes:

```text
Paper11 smoke check passed.
```

- [ ] **Step 4: Check docs for unresolved placeholders**

Run:

```powershell
$pattern = 'T' + 'BD|TO' + 'DO|REPLACE_' + 'WITH|PLACE' + 'HOLDER'
rg -n $pattern README.md paper\phase28_results\README.md paper\phase28_results\15_phase41_geofm_suitability_prior.md paper\submission\01_ijaeog_submission_readiness.md paper\submission\03_conclusion_manuscript_draft.md docs\superpowers\phase33_current_progress_handoff.md docs\superpowers\plans\2026-07-06-phase41-geofm-suitability-prior.md
```

Expected: no matches.

- [ ] **Step 5: Run diff and status checks**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected: diff check passes and working tree is clean except ignored generated Phase 41 outputs.

- [ ] **Step 6: Final response**

Report:

- Phase 41 status from the real no-registry run.
- Whether Phase 41 changed the decision on B2/B3.
- Test commands and results.
- Commit hashes created during implementation.
- The remaining blocker: a real external independent label registry that passes Phase 40 and Phase 41.
