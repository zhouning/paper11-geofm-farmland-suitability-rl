# Phase 37 Decision-Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 37 diagnostic that joins Phase 34/35 case outputs and audits whether Phase 33 normalized-B1 decisions align with available proxy, slope, and weak farmland indicators.

**Architecture:** Add one pure analysis module, one CLI runner, one focused pytest file, and result documentation. The module reads existing CSV/JSON artifacts, reconstructs variant and comparator selected sets, computes per-case and grouped alignment metrics, writes CSV/JSON/Markdown outputs, and never emits a reward-ready status.

**Tech Stack:** Python standard library CSV/JSON/argparse/pathlib, pytest, existing Paper11 artifact patterns.

---

## File Structure

- Create `src/paper11_geofm/phase37_decision_alignment.py`
  - Owns constants, fieldnames, CSV/JSON readers, selected-set reconstruction, case join, summary reduction, status rule, Markdown rendering, and artifact writing.
- Create `experiments/phase37_decision_alignment/run_phase37_decision_alignment.py`
  - Thin CLI wrapper matching Phase 34/35/36 runner style.
- Create `tests/test_phase37_decision_alignment.py`
  - Synthetic fixtures for supported, unsupported, insufficient, artifact writer, and CLI behavior.
- Modify `README.md`
  - Add Phase 37 folder/module bullets, run command, current real-run status after execution.
- Modify `paper/phase28_results/README.md`
  - Add `11_phase37_decision_alignment.md` to the file list and reproduction section.
- Create `paper/phase28_results/11_phase37_decision_alignment.md`
  - Summarize real-run result, row counts, status, and claim boundary.
- Modify `reproducibility/FILE_MANIFEST.tsv`
  - Add Phase 37 runner/module/test/docs entries.
- Modify `docs/superpowers/phase33_current_progress_handoff.md`
  - Record Phase 37 result and next step.

## Task 1: Core RED Test For Case Join And Supported Status

**Files:**
- Create: `tests/test_phase37_decision_alignment.py`
- Create later: `src/paper11_geofm/phase37_decision_alignment.py`

- [ ] **Step 1: Write the failing test and fixture helpers**

Add this initial test file:

```python
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


def _phase34_case(case_id: str, *, role: str, tile: str, seed: int, variant: str, comparator: str, base_gap: float, suitability_gap: float, low_slope_gap: float, spatial: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "case_role": role,
        "eval_tile_id": tile,
        "seed": seed,
        "variant_id": variant,
        "comparator_variant_id": comparator,
        "stability_class": "synthetic",
        "lower_delta": -0.1,
        "higher_delta": 0.4 if role == "phase33_positive_case" else -0.4,
        "delta_change": 0.5,
        "variant_reward": 1.0,
        "comparator_reward": 0.5,
        "variant_minus_comparator_reward": 0.5,
        "selected_block_jaccard": 0.0,
        "shared_selected_block_count": 0,
        "variant_selected_block_count": 2,
        "comparator_selected_block_count": 2,
        "variant_mean_base_planning_reward": 1.0 + base_gap,
        "comparator_mean_base_planning_reward": 1.0,
        "variant_mean_suitability_proxy": 0.5 + suitability_gap,
        "comparator_mean_suitability_proxy": 0.5,
        "variant_mean_low_slope_farmland_label": 0.5 + low_slope_gap,
        "comparator_mean_low_slope_farmland_label": 0.5,
        "variant_row_min": 1,
        "variant_row_max": 2,
        "variant_col_min": 1,
        "variant_col_max": 2,
        "comparator_row_min": 3,
        "comparator_row_max": 4,
        "comparator_col_min": 3,
        "comparator_col_max": 4,
        "spatial_pattern": spatial,
        "source_phase33_output_dir": "synthetic",
        "claim_boundary": "phase34 boundary",
    }


def _phase34_block(case_id: str, block_id: str, *, role: str, variant_step: int | str, comparator_step: int | str, farmland: float, slope_mean: float, slope_max: float, suitability: float, low_slope: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "eval_tile_id": case_id.split("|")[0],
        "seed": case_id.split("|")[1],
        "variant_id": case_id.split("|")[2],
        "comparator_variant_id": case_id.split("|")[3],
        "selection_role": role,
        "block_id": block_id,
        "variant_step": variant_step,
        "comparator_step": comparator_step,
        "row_min": 1,
        "row_max": 1,
        "col_min": 1,
        "col_max": 1,
        "row_center": 1.0,
        "col_center": 1.0,
        "base_planning_reward": 1.0,
        "suitability_proxy": suitability,
        "current_farmland_label": farmland,
        "low_slope_farmland_label": low_slope,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
        "area_m2": 10000.0,
        "claim_boundary": "phase34 boundary",
    }


def _phase35_case(case_id: str, *, role: str, summary_gap: float, pattern: str, jaccard: float) -> dict[str, object]:
    tile, seed, variant, comparator = case_id.split("|")
    return {
        "case_id": case_id,
        "case_role": role,
        "eval_tile_id": tile,
        "seed": seed,
        "variant_id": variant,
        "comparator_variant_id": comparator,
        "stability_class": "synthetic",
        "lower_delta": -0.1,
        "higher_delta": summary_gap,
        "delta_change": summary_gap + 0.1,
        "variant_summary_reward": 1.0 + summary_gap,
        "comparator_summary_reward": 1.0,
        "summary_reward_gap": summary_gap,
        "variant_step_source": "trace",
        "comparator_step_source": "trace",
        "variant_step_count": 2,
        "comparator_step_count": 2,
        "shared_block_count": 0,
        "union_block_count": 4,
        "selected_block_jaccard": jaccard,
        "same_step_match_count": 0,
        "mean_abs_shared_step_displacement": "",
        "max_abs_shared_step_displacement": "",
        "variant_trace_cumulative_reward": "",
        "comparator_trace_cumulative_reward": "",
        "trace_cumulative_reward_gap": "",
        "first_step_reward_gap": "",
        "action_overlap_pattern": pattern,
        "source_phase33_output_dir": "synthetic",
        "claim_boundary": "phase35 boundary",
    }


def _write_fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    positive_case = "tile_positive|0|N1ZR|D4P8"
    failure_case = "tile_failure|1|N1ZR|D4P8"
    case_fieldnames = [
        "case_id", "case_role", "eval_tile_id", "seed", "variant_id",
        "comparator_variant_id", "stability_class", "lower_delta", "higher_delta",
        "delta_change", "variant_reward", "comparator_reward",
        "variant_minus_comparator_reward", "selected_block_jaccard",
        "shared_selected_block_count", "variant_selected_block_count",
        "comparator_selected_block_count", "variant_mean_base_planning_reward",
        "comparator_mean_base_planning_reward", "variant_mean_suitability_proxy",
        "comparator_mean_suitability_proxy", "variant_mean_low_slope_farmland_label",
        "comparator_mean_low_slope_farmland_label", "variant_row_min", "variant_row_max",
        "variant_col_min", "variant_col_max", "comparator_row_min", "comparator_row_max",
        "comparator_col_min", "comparator_col_max", "spatial_pattern",
        "source_phase33_output_dir", "claim_boundary",
    ]
    block_fieldnames = [
        "case_id", "eval_tile_id", "seed", "variant_id", "comparator_variant_id",
        "selection_role", "block_id", "variant_step", "comparator_step", "row_min",
        "row_max", "col_min", "col_max", "row_center", "col_center",
        "base_planning_reward", "suitability_proxy", "current_farmland_label",
        "low_slope_farmland_label", "slope_mean", "slope_max", "area_m2",
        "claim_boundary",
    ]
    phase35_fieldnames = [
        "case_id", "case_role", "eval_tile_id", "seed", "variant_id",
        "comparator_variant_id", "stability_class", "lower_delta", "higher_delta",
        "delta_change", "variant_summary_reward", "comparator_summary_reward",
        "summary_reward_gap", "variant_step_source", "comparator_step_source",
        "variant_step_count", "comparator_step_count", "shared_block_count",
        "union_block_count", "selected_block_jaccard", "same_step_match_count",
        "mean_abs_shared_step_displacement", "max_abs_shared_step_displacement",
        "variant_trace_cumulative_reward", "comparator_trace_cumulative_reward",
        "trace_cumulative_reward_gap", "first_step_reward_gap",
        "action_overlap_pattern", "source_phase33_output_dir", "claim_boundary",
    ]
    phase34_cases = _write_csv(
        tmp_path / "phase34_case_map_cases.csv",
        [
            _phase34_case(positive_case, role="phase33_positive_case", tile="tile_positive", seed=0, variant="N1ZR", comparator="D4P8", base_gap=0.3, suitability_gap=0.2, low_slope_gap=0.5, spatial="variant_selects_higher_base_reward_blocks"),
            _phase34_case(failure_case, role="phase33_failure_case", tile="tile_failure", seed=1, variant="N1ZR", comparator="D4P8", base_gap=-0.2, suitability_gap=-0.1, low_slope_gap=-0.5, spatial="variant_selects_lower_base_reward_blocks"),
        ],
        case_fieldnames,
    )
    phase34_blocks = _write_csv(
        tmp_path / "phase34_case_map_blocks.csv",
        [
            _phase34_block(positive_case, "p_var_a", role="variant_only", variant_step=1, comparator_step="", farmland=1.0, slope_mean=4.0, slope_max=6.0, suitability=0.9, low_slope=1.0),
            _phase34_block(positive_case, "p_var_b", role="variant_only", variant_step=2, comparator_step="", farmland=1.0, slope_mean=6.0, slope_max=8.0, suitability=0.8, low_slope=1.0),
            _phase34_block(positive_case, "p_cmp_a", role="comparator_only", variant_step="", comparator_step=1, farmland=0.0, slope_mean=18.0, slope_max=24.0, suitability=0.2, low_slope=0.0),
            _phase34_block(positive_case, "p_cmp_b", role="comparator_only", variant_step="", comparator_step=2, farmland=0.0, slope_mean=20.0, slope_max=28.0, suitability=0.1, low_slope=0.0),
            _phase34_block(failure_case, "f_var_a", role="variant_only", variant_step=1, comparator_step="", farmland=0.0, slope_mean=22.0, slope_max=30.0, suitability=0.2, low_slope=0.0),
            _phase34_block(failure_case, "f_var_b", role="variant_only", variant_step=2, comparator_step="", farmland=0.0, slope_mean=24.0, slope_max=32.0, suitability=0.1, low_slope=0.0),
            _phase34_block(failure_case, "f_cmp_a", role="comparator_only", variant_step="", comparator_step=1, farmland=1.0, slope_mean=6.0, slope_max=8.0, suitability=0.9, low_slope=1.0),
            _phase34_block(failure_case, "f_cmp_b", role="comparator_only", variant_step="", comparator_step=2, farmland=1.0, slope_mean=8.0, slope_max=10.0, suitability=0.8, low_slope=1.0),
        ],
        block_fieldnames,
    )
    phase35_cases = _write_csv(
        tmp_path / "phase35_action_overlap_cases.csv",
        [
            _phase35_case(positive_case, role="phase33_positive_case", summary_gap=0.7, pattern="disjoint_positive_gap", jaccard=0.0),
            _phase35_case(failure_case, role="phase33_failure_case", summary_gap=-0.6, pattern="disjoint_negative_gap", jaccard=0.0),
        ],
        phase35_fieldnames,
    )
    phase36_json = tmp_path / "phase36_suitability_proxy_validation.json"
    phase36_json.write_text(
        json.dumps({"phase36_proxy_validation_status": "proxy_signal_not_supported"}),
        encoding="utf-8",
    )
    return {
        "phase34_cases": phase34_cases,
        "phase34_blocks": phase34_blocks,
        "phase35_cases": phase35_cases,
        "phase36_json": phase36_json,
    }


def test_phase37_builds_decision_alignment_supported_for_proxy_rebuild(tmp_path):
    from paper11_geofm.phase37_decision_alignment import (
        PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
        build_phase37_decision_alignment,
    )

    paths = _write_fixture_inputs(tmp_path)

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv=paths["phase34_cases"],
        phase34_blocks_csv=paths["phase34_blocks"],
        phase35_cases_csv=paths["phase35_cases"],
        phase36_diagnosis_json=paths["phase36_json"],
    )

    assert analysis["phase"] == "phase37_decision_alignment"
    assert analysis["phase37_decision_alignment_status"] == "decision_alignment_supported_for_proxy_rebuild"
    assert analysis["claim_boundary"] == PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY
    assert analysis["phase36_proxy_validation_status"] == "proxy_signal_not_supported"
    assert analysis["row_counts"]["case_rows"] == 2
    assert analysis["row_counts"]["summary_rows"] > 0

    cases = {row["case_id"]: row for row in analysis["case_rows"]}
    positive = cases["tile_positive|0|N1ZR|D4P8"]
    assert positive["summary_reward_gap"] == 0.7
    assert positive["suitability_proxy_gap"] == 0.2
    assert positive["low_slope_farmland_label_gap"] == 0.5
    assert positive["current_farmland_label_gap"] == 1.0
    assert positive["slope_mean_gap"] < 0.0
    assert positive["proxy_alignment_pattern"] == "proxy_or_label_alignment"

    failure = cases["tile_failure|1|N1ZR|D4P8"]
    assert failure["case_role"] == "phase33_failure_case"
    assert failure["suitability_proxy_gap"] == -0.1
    assert failure["current_farmland_label_gap"] == -1.0
    assert failure["proxy_alignment_pattern"] == "no_proxy_alignment"
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_builds_decision_alignment_supported_for_proxy_rebuild -q --basetemp=.pytest_tmp_phase37_red -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paper11_geofm.phase37_decision_alignment'`.

- [ ] **Step 3: Create minimal module with constants and build function**

Create `src/paper11_geofm/phase37_decision_alignment.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY = (
    "Phase 37 is a read-only decision-alignment diagnostic over existing "
    "Phase 34/35/36 artifacts; it does not run policy training, does not alter "
    "rewards, does not enable suitability reward, does not test B2/B3, and does "
    "not support final submission-level planning-performance claims."
)

PHASE37_CASE_FIELDNAMES = [
    "case_id", "case_role", "eval_tile_id", "seed", "variant_id",
    "comparator_variant_id", "stability_class", "summary_reward_gap",
    "spatial_pattern", "action_overlap_pattern", "selected_block_jaccard",
    "base_planning_reward_gap", "suitability_proxy_gap",
    "low_slope_farmland_label_gap", "current_farmland_label_gap",
    "slope_mean_gap", "slope_max_gap", "proxy_alignment_pattern",
    "phase36_proxy_validation_status", "claim_boundary",
]

PHASE37_SUMMARY_FIELDNAMES = [
    "group_field", "group_value", "case_count", "positive_case_count",
    "failure_case_count", "mean_summary_reward_gap",
    "mean_base_planning_reward_gap", "mean_suitability_proxy_gap",
    "mean_low_slope_farmland_label_gap", "mean_current_farmland_label_gap",
    "mean_slope_mean_gap", "mean_slope_max_gap",
    "positive_suitability_proxy_gap_count",
    "positive_low_slope_farmland_label_gap_count",
    "positive_current_farmland_label_gap_count",
    "lower_slope_mean_gap_count", "claim_boundary",
]


def build_phase37_decision_alignment(
    phase34_cases_csv: Path | str,
    phase34_blocks_csv: Path | str,
    phase35_cases_csv: Path | str,
    phase36_diagnosis_json: Path | str | None = None,
) -> dict[str, object]:
    phase34_cases = _read_csv_rows(Path(phase34_cases_csv), "Phase 34 case CSV")
    phase34_blocks = _read_csv_rows(Path(phase34_blocks_csv), "Phase 34 block CSV")
    phase35_cases = _read_csv_rows(Path(phase35_cases_csv), "Phase 35 case CSV")
    phase36_status = _phase36_status(phase36_diagnosis_json)
    phase35_by_case = {str(row.get("case_id", "")).strip(): row for row in phase35_cases}
    blocks_by_case = _blocks_by_case(phase34_blocks)
    case_rows = []
    for phase34_case in phase34_cases:
        case_id = str(phase34_case.get("case_id", "")).strip()
        if not case_id or case_id not in phase35_by_case:
            continue
        block_metrics = _case_block_metrics(blocks_by_case.get(case_id, []))
        phase35_case = phase35_by_case[case_id]
        case_rows.append(_case_row(phase34_case, phase35_case, block_metrics, phase36_status))
    summary_rows = _summary_rows(case_rows)
    status = _status(case_rows)
    return {
        "phase": "phase37_decision_alignment",
        "phase37_decision_alignment_status": status,
        "phase36_proxy_validation_status": phase36_status,
        "source_paths": {
            "phase34_cases_csv": str(Path(phase34_cases_csv)),
            "phase34_blocks_csv": str(Path(phase34_blocks_csv)),
            "phase35_cases_csv": str(Path(phase35_cases_csv)),
            "phase36_diagnosis_json": str(Path(phase36_diagnosis_json)) if phase36_diagnosis_json is not None else None,
        },
        "row_counts": {
            "phase34_cases": len(phase34_cases),
            "phase34_blocks": len(phase34_blocks),
            "phase35_cases": len(phase35_cases),
            "case_rows": len(case_rows),
            "summary_rows": len(summary_rows),
        },
        "case_rows": case_rows,
        "summary_rows": summary_rows,
        "interpretation": _interpretation(status),
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }
```

Add helper functions in the same file:

```python
def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _phase36_status(path: Path | str | None) -> str:
    if path is None:
        return "phase36_not_supplied"
    payload_path = Path(path)
    if not payload_path.exists():
        return "phase36_not_supplied"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return str(payload.get("phase36_proxy_validation_status", "phase36_status_missing"))


def _blocks_by_case(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            grouped.setdefault(case_id, []).append(row)
    return grouped


def _case_block_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | str]:
    variant_rows = [row for row in rows if _has_step(row, "variant_step")]
    comparator_rows = [row for row in rows if _has_step(row, "comparator_step")]
    return {
        "current_farmland_label_gap": _mean_gap(variant_rows, comparator_rows, "current_farmland_label"),
        "slope_mean_gap": _mean_gap(variant_rows, comparator_rows, "slope_mean"),
        "slope_max_gap": _mean_gap(variant_rows, comparator_rows, "slope_max"),
    }


def _case_row(
    phase34_case: Mapping[str, object],
    phase35_case: Mapping[str, object],
    block_metrics: Mapping[str, object],
    phase36_status: str,
) -> dict[str, object]:
    suitability_gap = _float_value(phase34_case, "variant_mean_suitability_proxy") - _float_value(phase34_case, "comparator_mean_suitability_proxy")
    low_slope_gap = _float_value(phase34_case, "variant_mean_low_slope_farmland_label") - _float_value(phase34_case, "comparator_mean_low_slope_farmland_label")
    current_gap = _csv_float(block_metrics.get("current_farmland_label_gap"))
    slope_mean_gap = _csv_float(block_metrics.get("slope_mean_gap"))
    slope_max_gap = _csv_float(block_metrics.get("slope_max_gap"))
    return {
        "case_id": str(phase34_case.get("case_id", "")),
        "case_role": str(phase34_case.get("case_role", "")),
        "eval_tile_id": str(phase34_case.get("eval_tile_id", "")),
        "seed": _int_value(phase34_case, "seed"),
        "variant_id": str(phase34_case.get("variant_id", "")),
        "comparator_variant_id": str(phase34_case.get("comparator_variant_id", "")),
        "stability_class": str(phase34_case.get("stability_class", "")),
        "summary_reward_gap": _round_float(_float_value(phase35_case, "summary_reward_gap")),
        "spatial_pattern": str(phase34_case.get("spatial_pattern", "")),
        "action_overlap_pattern": str(phase35_case.get("action_overlap_pattern", "")),
        "selected_block_jaccard": _round_float(_float_value(phase35_case, "selected_block_jaccard")),
        "base_planning_reward_gap": _round_float(_float_value(phase34_case, "variant_mean_base_planning_reward") - _float_value(phase34_case, "comparator_mean_base_planning_reward")),
        "suitability_proxy_gap": _round_float(suitability_gap),
        "low_slope_farmland_label_gap": _round_float(low_slope_gap),
        "current_farmland_label_gap": current_gap,
        "slope_mean_gap": slope_mean_gap,
        "slope_max_gap": slope_max_gap,
        "proxy_alignment_pattern": _proxy_alignment_pattern(suitability_gap, low_slope_gap, current_gap, slope_mean_gap),
        "phase36_proxy_validation_status": phase36_status,
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }


def _summary_rows(case_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in ("case_role", "eval_tile_id", "variant_id", "comparator_variant_id", "spatial_pattern", "action_overlap_pattern"):
        values = sorted({str(row.get(field, "")) for row in case_rows if str(row.get(field, "")).strip()})
        for value in values:
            grouped = [row for row in case_rows if str(row.get(field, "")) == value]
            rows.append(_summary_row(field, value, grouped))
    if case_rows:
        rows.append(_summary_row("all_cases", "all_cases", list(case_rows)))
    return rows
```

Add the remaining helpers:

```python
def _summary_row(group_field: str, group_value: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "group_field": group_field,
        "group_value": group_value,
        "case_count": len(rows),
        "positive_case_count": sum(1 for row in rows if row.get("case_role") == "phase33_positive_case"),
        "failure_case_count": sum(1 for row in rows if row.get("case_role") == "phase33_failure_case"),
        "mean_summary_reward_gap": _mean_metric(rows, "summary_reward_gap"),
        "mean_base_planning_reward_gap": _mean_metric(rows, "base_planning_reward_gap"),
        "mean_suitability_proxy_gap": _mean_metric(rows, "suitability_proxy_gap"),
        "mean_low_slope_farmland_label_gap": _mean_metric(rows, "low_slope_farmland_label_gap"),
        "mean_current_farmland_label_gap": _mean_metric(rows, "current_farmland_label_gap"),
        "mean_slope_mean_gap": _mean_metric(rows, "slope_mean_gap"),
        "mean_slope_max_gap": _mean_metric(rows, "slope_max_gap"),
        "positive_suitability_proxy_gap_count": _positive_count(rows, "suitability_proxy_gap"),
        "positive_low_slope_farmland_label_gap_count": _positive_count(rows, "low_slope_farmland_label_gap"),
        "positive_current_farmland_label_gap_count": _positive_count(rows, "current_farmland_label_gap"),
        "lower_slope_mean_gap_count": sum(1 for row in rows if _optional_float(row.get("slope_mean_gap")) is not None and _optional_float(row.get("slope_mean_gap")) < 0.0),
        "claim_boundary": PHASE37_DECISION_ALIGNMENT_CLAIM_BOUNDARY,
    }


def _status(case_rows: Sequence[Mapping[str, object]]) -> str:
    if not case_rows:
        return "decision_alignment_inputs_insufficient"
    positives = [row for row in case_rows if row.get("case_role") == "phase33_positive_case"]
    failures = [row for row in case_rows if row.get("case_role") == "phase33_failure_case"]
    positive_alignment = _group_has_proxy_alignment(positives)
    failure_alignment = _group_has_proxy_alignment(failures)
    if positive_alignment and not failure_alignment:
        return "decision_alignment_supported_for_proxy_rebuild"
    return "decision_alignment_not_supported"


def _group_has_proxy_alignment(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    suitability = _mean_metric(rows, "suitability_proxy_gap")
    low_slope = _mean_metric(rows, "low_slope_farmland_label_gap")
    return (
        _optional_float(suitability) is not None
        and float(suitability) > 0.0
    ) or (
        _optional_float(low_slope) is not None
        and float(low_slope) > 0.0
    )


def _proxy_alignment_pattern(suitability_gap: object, low_slope_gap: object, current_gap: object, slope_mean_gap: object) -> str:
    if _optional_float(suitability_gap) is not None and float(suitability_gap) > 0.0:
        return "proxy_or_label_alignment"
    if _optional_float(low_slope_gap) is not None and float(low_slope_gap) > 0.0:
        return "proxy_or_label_alignment"
    if _optional_float(current_gap) is not None and float(current_gap) > 0.0:
        return "proxy_or_label_alignment"
    if _optional_float(slope_mean_gap) is not None and float(slope_mean_gap) < 0.0:
        return "proxy_or_label_alignment"
    return "no_proxy_alignment"


def _interpretation(status: str) -> str:
    if status == "decision_alignment_supported_for_proxy_rebuild":
        return "Positive Phase 33 cases separate from failure cases in available proxy or weak environmental diagnostics. This supports proxy-rebuild follow-up only."
    if status == "decision_alignment_not_supported":
        return "Phase 33 case outcomes do not separate cleanly in available proxy or weak environmental diagnostics."
    return "Phase 37 could not join enough Phase 34/35 cases for decision-alignment diagnostics."
```

Add numeric helpers:

```python
def _has_step(row: Mapping[str, object], field: str) -> bool:
    return str(row.get(field, "")).strip() != ""


def _mean_gap(variant_rows: Sequence[Mapping[str, object]], comparator_rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    variant_mean = _mean([_float_value(row, field) for row in variant_rows if _has_value(row, field)])
    comparator_mean = _mean([_float_value(row, field) for row in comparator_rows if _has_value(row, field)])
    if variant_mean == "" or comparator_mean == "":
        return ""
    return _round_float(float(variant_mean) - float(comparator_mean))


def _mean_metric(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    return _mean([float(row[field]) for row in rows if _optional_float(row.get(field)) is not None])


def _mean(values: Sequence[float]) -> float | str:
    clean = [float(value) for value in values]
    if not clean:
        return ""
    return _round_float(sum(clean) / len(clean))


def _positive_count(rows: Sequence[Mapping[str, object]], field: str) -> int:
    return sum(1 for row in rows if _optional_float(row.get(field)) is not None and float(row[field]) > 0.0)


def _has_value(row: Mapping[str, object], field: str) -> bool:
    return field in row and str(row.get(field, "")).strip() != ""


def _float_value(row: Mapping[str, object], field: str) -> float:
    try:
        return float(row[field])
    except KeyError as exc:
        raise ValueError(f"Missing numeric field {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {field}: {row.get(field)!r}") from exc


def _int_value(row: Mapping[str, object], field: str) -> int:
    try:
        return int(float(row[field]))
    except KeyError as exc:
        raise ValueError(f"Missing integer field {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field {field}: {row.get(field)!r}") from exc


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_float(value: object) -> float | str:
    parsed = _optional_float(value)
    if parsed is None:
        return ""
    return _round_float(parsed)


def _round_float(value: float | int) -> float:
    return round(float(value), 10)
```

- [ ] **Step 4: Run the test to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_builds_decision_alignment_supported_for_proxy_rebuild -q --basetemp=.pytest_tmp_phase37_green1 -p no:cacheprovider
```

Expected: `1 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add tests\test_phase37_decision_alignment.py src\paper11_geofm\phase37_decision_alignment.py
git commit -m "feat: add Phase 37 decision-alignment core"
```

## Task 2: RED/GREEN Tests For Unsupported And Insufficient Status

**Files:**
- Modify: `tests/test_phase37_decision_alignment.py`
- Modify: `src/paper11_geofm/phase37_decision_alignment.py`

- [ ] **Step 1: Add failing status tests**

Append:

```python
def test_phase37_marks_alignment_not_supported_when_failures_share_alignment(tmp_path):
    from paper11_geofm.phase37_decision_alignment import build_phase37_decision_alignment

    paths = _write_fixture_inputs(tmp_path)
    rows = list(csv.DictReader(paths["phase34_cases"].open("r", encoding="utf-8")))
    for row in rows:
        if row["case_role"] == "phase33_failure_case":
            row["variant_mean_suitability_proxy"] = "0.8"
            row["comparator_mean_suitability_proxy"] = "0.5"
            row["variant_mean_low_slope_farmland_label"] = "0.8"
            row["comparator_mean_low_slope_farmland_label"] = "0.5"
    _write_csv(paths["phase34_cases"], rows, list(rows[0].keys()))

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv=paths["phase34_cases"],
        phase34_blocks_csv=paths["phase34_blocks"],
        phase35_cases_csv=paths["phase35_cases"],
    )

    assert analysis["phase37_decision_alignment_status"] == "decision_alignment_not_supported"


def test_phase37_marks_inputs_insufficient_when_cases_do_not_join(tmp_path):
    from paper11_geofm.phase37_decision_alignment import build_phase37_decision_alignment

    paths = _write_fixture_inputs(tmp_path)
    _write_csv(
        paths["phase35_cases"],
        [],
        [
            "case_id", "case_role", "eval_tile_id", "seed", "variant_id",
            "comparator_variant_id", "summary_reward_gap", "selected_block_jaccard",
            "action_overlap_pattern",
        ],
    )

    analysis = build_phase37_decision_alignment(
        phase34_cases_csv=paths["phase34_cases"],
        phase34_blocks_csv=paths["phase34_blocks"],
        phase35_cases_csv=paths["phase35_cases"],
    )

    assert analysis["phase37_decision_alignment_status"] == "decision_alignment_inputs_insufficient"
    assert analysis["row_counts"]["case_rows"] == 0
```

- [ ] **Step 2: Run tests to verify RED or existing GREEN**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py -q --basetemp=.pytest_tmp_phase37_status_red -p no:cacheprovider
```

Expected: if Task 1 implementation already satisfies these cases, all tests pass. If a failure appears, it must be about the Phase 37 status rule, not import or fixture errors.

- [ ] **Step 3: Adjust implementation only if the new tests fail**

If `decision_alignment_not_supported` fails, update `_status()` to use the full failure-case aggregate:

```python
def _status(case_rows: Sequence[Mapping[str, object]]) -> str:
    if not case_rows:
        return "decision_alignment_inputs_insufficient"
    positives = [row for row in case_rows if row.get("case_role") == "phase33_positive_case"]
    failures = [row for row in case_rows if row.get("case_role") == "phase33_failure_case"]
    positive_alignment = _group_has_proxy_alignment(positives)
    failure_alignment = _group_has_proxy_alignment(failures)
    if positive_alignment and not failure_alignment:
        return "decision_alignment_supported_for_proxy_rebuild"
    return "decision_alignment_not_supported"
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py -q --basetemp=.pytest_tmp_phase37_status_green -p no:cacheprovider
```

Expected: all Phase 37 tests pass.

- [ ] **Step 5: Commit Task 2 if implementation or tests changed**

Run:

```powershell
git add tests\test_phase37_decision_alignment.py src\paper11_geofm\phase37_decision_alignment.py
git commit -m "test: cover Phase 37 alignment status rules"
```

## Task 3: Artifact Writer And Markdown

**Files:**
- Modify: `tests/test_phase37_decision_alignment.py`
- Modify: `src/paper11_geofm/phase37_decision_alignment.py`

- [ ] **Step 1: Add failing writer test**

Append:

```python
def test_phase37_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase37_decision_alignment import (
        build_phase37_decision_alignment,
        write_phase37_decision_alignment_artifacts,
    )

    paths = _write_fixture_inputs(tmp_path)
    analysis = build_phase37_decision_alignment(
        phase34_cases_csv=paths["phase34_cases"],
        phase34_blocks_csv=paths["phase34_blocks"],
        phase35_cases_csv=paths["phase35_cases"],
        phase36_diagnosis_json=paths["phase36_json"],
    )

    artifacts = write_phase37_decision_alignment_artifacts(
        analysis,
        tmp_path / "outputs",
    )

    assert artifacts["case_alignment_csv"].name == "phase37_decision_alignment_cases.csv"
    assert artifacts["summary_csv"].name == "phase37_decision_alignment_summary.csv"
    assert artifacts["diagnosis_json"].name == "phase37_decision_alignment.json"
    assert artifacts["diagnosis_md"].name == "phase37_decision_alignment.md"
    assert all(path.exists() for path in artifacts.values())
    saved = json.loads(artifacts["diagnosis_json"].read_text(encoding="utf-8"))
    assert saved["phase"] == "phase37_decision_alignment"
    markdown = artifacts["diagnosis_md"].read_text(encoding="utf-8")
    assert "Phase 37 Decision-Alignment" in markdown
    assert "decision_alignment_supported_for_proxy_rebuild" in markdown
    assert "does not enable suitability reward" in markdown
```

- [ ] **Step 2: Run writer test to verify RED**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_writer_outputs_csv_json_and_markdown -q --basetemp=.pytest_tmp_phase37_writer_red -p no:cacheprovider
```

Expected: FAIL with `ImportError` or `AttributeError` for missing `write_phase37_decision_alignment_artifacts`.

- [ ] **Step 3: Implement writer helpers**

Append to `src/paper11_geofm/phase37_decision_alignment.py`:

```python
def write_phase37_decision_alignment_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    case_path = output_path / "phase37_decision_alignment_cases.csv"
    summary_path = output_path / "phase37_decision_alignment_summary.csv"
    json_path = output_path / "phase37_decision_alignment.json"
    markdown_path = output_path / "phase37_decision_alignment.md"
    _write_csv_mapping_rows(case_path, PHASE37_CASE_FIELDNAMES, analysis.get("case_rows"), "case_rows")
    _write_csv_mapping_rows(summary_path, PHASE37_SUMMARY_FIELDNAMES, analysis.get("summary_rows"), "summary_rows")
    json_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_phase37_markdown(analysis), encoding="utf-8")
    return {
        "case_alignment_csv": case_path,
        "summary_csv": summary_path,
        "diagnosis_json": json_path,
        "diagnosis_md": markdown_path,
    }
```

Add helper functions:

```python
def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 37 analysis is missing {label}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 37 {label} contains a non-mapping row")
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _phase37_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Phase 37 Decision-Alignment",
        "",
        f"Status: {analysis.get('phase37_decision_alignment_status', '')}",
        "",
        f"Phase 36 status: {analysis.get('phase36_proxy_validation_status', '')}",
        "",
        "## Case Summary",
        "",
        "| Case | Role | Reward gap | Suitability gap | Low-slope gap | Current farmland gap | Slope mean gap | Pattern |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis.get("case_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {case} | {role} | {reward} | {proxy} | {low} | {farmland} | {slope} | {pattern} |".format(
                case=row.get("case_id", ""),
                role=row.get("case_role", ""),
                reward=row.get("summary_reward_gap", ""),
                proxy=row.get("suitability_proxy_gap", ""),
                low=row.get("low_slope_farmland_label_gap", ""),
                farmland=row.get("current_farmland_label_gap", ""),
                slope=row.get("slope_mean_gap", ""),
                pattern=row.get("proxy_alignment_pattern", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(analysis.get("interpretation", "")),
            "",
            "## Claim Boundary",
            "",
            str(analysis.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
```

- [ ] **Step 4: Run writer test to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_writer_outputs_csv_json_and_markdown -q --basetemp=.pytest_tmp_phase37_writer_green -p no:cacheprovider
```

Expected: `1 passed`.

- [ ] **Step 5: Run all Phase 37 tests**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py -q --basetemp=.pytest_tmp_phase37_writer_all -p no:cacheprovider
```

Expected: all Phase 37 tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add tests\test_phase37_decision_alignment.py src\paper11_geofm\phase37_decision_alignment.py
git commit -m "feat: write Phase 37 decision-alignment artifacts"
```

## Task 4: CLI Runner

**Files:**
- Modify: `tests/test_phase37_decision_alignment.py`
- Create: `experiments/phase37_decision_alignment/run_phase37_decision_alignment.py`

- [ ] **Step 1: Add failing CLI test**

Append:

```python
def test_phase37_cli_writes_outputs(tmp_path):
    paths = _write_fixture_inputs(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase37_decision_alignment"
        / "run_phase37_decision_alignment.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase34-cases-csv",
            str(paths["phase34_cases"]),
            "--phase34-blocks-csv",
            str(paths["phase34_blocks"]),
            "--phase35-cases-csv",
            str(paths["phase35_cases"]),
            "--phase36-diagnosis-json",
            str(paths["phase36_json"]),
            "--output-dir",
            str(tmp_path / "cli_outputs"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 37 decision-alignment status:" in result.stdout
    assert "Claim boundary:" in result.stdout
    assert (
        tmp_path
        / "cli_outputs"
        / "phase37_decision_alignment.json"
    ).exists()
```

- [ ] **Step 2: Run CLI test to verify RED**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_cli_writes_outputs -q --basetemp=.pytest_tmp_phase37_cli_red -p no:cacheprovider
```

Expected: FAIL because `experiments\phase37_decision_alignment\run_phase37_decision_alignment.py` does not exist.

- [ ] **Step 3: Implement runner**

Create `experiments/phase37_decision_alignment/run_phase37_decision_alignment.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase37_decision_alignment import (
    build_phase37_decision_alignment,
    write_phase37_decision_alignment_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Paper11 Phase 37 decision-alignment diagnostic "
            "over existing Phase 34/35/36 artifacts."
        )
    )
    parser.add_argument("--phase34-cases-csv", type=Path, required=True)
    parser.add_argument("--phase34-blocks-csv", type=Path, required=True)
    parser.add_argument("--phase35-cases-csv", type=Path, required=True)
    parser.add_argument("--phase36-diagnosis-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase37_decision_alignment(
            phase34_cases_csv=args.phase34_cases_csv,
            phase34_blocks_csv=args.phase34_blocks_csv,
            phase35_cases_csv=args.phase35_cases_csv,
            phase36_diagnosis_json=args.phase36_diagnosis_json,
        )
        paths = write_phase37_decision_alignment_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Phase 37 decision-alignment status: "
        f"{analysis['phase37_decision_alignment_status']}"
    )
    print(
        "Phase 36 proxy-validation status: "
        f"{analysis['phase36_proxy_validation_status']}"
    )
    print(f"Case alignment CSV: {paths['case_alignment_csv']}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Diagnosis JSON: {paths['diagnosis_json']}")
    print(f"Diagnosis Markdown: {paths['diagnosis_md']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py::test_phase37_cli_writes_outputs -q --basetemp=.pytest_tmp_phase37_cli_green -p no:cacheprovider
```

Expected: `1 passed`.

- [ ] **Step 5: Run all Phase 37 tests**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py -q --basetemp=.pytest_tmp_phase37_cli_all -p no:cacheprovider
```

Expected: all Phase 37 tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add tests\test_phase37_decision_alignment.py experiments\phase37_decision_alignment\run_phase37_decision_alignment.py
git commit -m "feat: add Phase 37 decision-alignment runner"
```

## Task 5: Real Run And Result Documentation

**Files:**
- Modify: `README.md`
- Modify: `paper/phase28_results/README.md`
- Create: `paper/phase28_results/11_phase37_decision_alignment.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run focused verification before real artifacts**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase37_pre_real -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real Phase 37 diagnostic**

Run:

```powershell
python experiments\phase37_decision_alignment\run_phase37_decision_alignment.py --phase34-cases-csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_cases.csv --phase34-blocks-csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_blocks.csv --phase35-cases-csv experiments\phase35_phase33_action_overlap_diagnostics\outputs\real_bishan_5120_phase33_9run\phase35_action_overlap_cases.csv --phase36-diagnosis-json experiments\phase36_suitability_proxy_validation\outputs\real_bishan\phase36_suitability_proxy_validation.json --output-dir experiments\phase37_decision_alignment\outputs\real_bishan_5120_phase33_9run
```

Expected stdout includes:

```text
Phase 37 decision-alignment status:
Phase 36 proxy-validation status: proxy_signal_not_supported
Claim boundary: Phase 37 is a read-only decision-alignment diagnostic
```

- [ ] **Step 3: Inspect real output JSON and Markdown**

Run:

```powershell
Get-Content -Raw experiments\phase37_decision_alignment\outputs\real_bishan_5120_phase33_9run\phase37_decision_alignment.json
Get-Content -Raw experiments\phase37_decision_alignment\outputs\real_bishan_5120_phase33_9run\phase37_decision_alignment.md
```

Expected: JSON has `phase37_decision_alignment_status`, `row_counts.case_rows`, `summary_rows`, and `phase36_proxy_validation_status`.

- [ ] **Step 4: Create result documentation**

Create `paper/phase28_results/11_phase37_decision_alignment.md` using the real JSON/Markdown values:

```markdown
# Phase 37 Decision-Alignment Audit

## One-Sentence Argument

Phase 37 audits whether completed Phase 33 normalized-B1 decisions separate
from comparator decisions in available proxy, slope, and weak farmland
diagnostics, while Phase 36 continues to block suitability reward use.

## Current Experiment Snapshot

Inputs:

- Phase 34 case map cases:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_cases.csv`
- Phase 34 selected blocks:
  `experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run/phase34_case_map_blocks.csv`
- Phase 35 action-overlap cases:
  `experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run/phase35_action_overlap_cases.csv`
- Phase 36 diagnosis:
  `experiments/phase36_suitability_proxy_validation/outputs/real_bishan/phase36_suitability_proxy_validation.json`

Local ignored outputs:

```text
experiments/phase37_decision_alignment/outputs/real_bishan_5120_phase33_9run
```

Generated artifacts:

```text
phase37_decision_alignment_cases.csv
phase37_decision_alignment_summary.csv
phase37_decision_alignment.json
phase37_decision_alignment.md
```

## Main Result

The current Phase 37 status is:

```text
decision_alignment_supported_for_proxy_rebuild
```

Phase 36 remains:

```text
proxy_signal_not_supported
```

## Interpretation

Positive Phase 33 cases separate from failure cases in available proxy or weak
environmental diagnostics. This supports proxy-rebuild follow-up only. It does
not make B2/B3 or suitability reward ready.

## Claim Boundary

Phase 37 is diagnostic only. It does not run policy training, alter rewards,
enable suitability reward, test B2/B3, prove GeoFM agronomic validity, or
support final planning-performance claims.
```

If the real JSON reports a different `phase37_decision_alignment_status`, edit the status block and interpretation to match the real JSON before committing.

- [ ] **Step 5: Update README and phase28 result index**

In `README.md`:

- Add `experiments/phase37_decision_alignment/` to the repository layout list.
- Extend the `src/paper11_geofm/` description with `Phase 37 decision-alignment diagnostics`.
- Add a run command after the Phase 36 section using the exact command from Step 2.
- State that Phase 37 is diagnostic-only and Phase 36 still blocks B2/B3 suitability reward.

In `paper/phase28_results/README.md`:

- Add `11_phase37_decision_alignment.md` to the file list.
- Add a Phase 37 reproduction section with the exact command from Step 2.
- Add expected artifacts:
  - `phase37_decision_alignment_cases.csv`
  - `phase37_decision_alignment_summary.csv`
  - `phase37_decision_alignment.json`
  - `phase37_decision_alignment.md`

- [ ] **Step 6: Update manifest and handoff**

Add these lines to `reproducibility/FILE_MANIFEST.tsv`:

```text
experiments/phase37_decision_alignment/run_phase37_decision_alignment.py	experiment	Executable read-only Phase 37 decision-alignment diagnostic runner for existing Phase 34/35/36 artifacts.
src/paper11_geofm/phase37_decision_alignment.py	source	Phase 37 decision-alignment diagnostic builder and artifact writer.
tests/test_phase37_decision_alignment.py	verification	Pytest checks for Phase 37 decision-alignment joins, status rules, artifact writing, and CLI behavior.
paper/phase28_results/11_phase37_decision_alignment.md	documentation	Reviewer-facing interpretation of the Phase 37 decision-alignment audit.
docs/superpowers/specs/2026-06-26-phase37-decision-alignment-design.md	documentation	Phase 37 decision-alignment design specification.
docs/superpowers/plans/2026-06-26-phase37-decision-alignment.md	documentation	Phase 37 decision-alignment implementation plan.
```

In `docs/superpowers/phase33_current_progress_handoff.md`, add a Phase 37 section with:

- real output directory;
- generated artifact names;
- real status;
- row counts from JSON;
- interpretation;
- explicit statement that Phase 36 still blocks B2/B3 reward.

- [ ] **Step 7: Run final verification**

Run:

```powershell
python -m pytest tests\test_phase37_decision_alignment.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase37_final -p no:cacheprovider
python scripts\smoke_check.py
```

Expected:

```text
All selected Phase 37/36 tests pass.
Paper11 smoke check passed.
```

- [ ] **Step 8: Commit Task 5**

Run:

```powershell
git add README.md paper\phase28_results\README.md paper\phase28_results\11_phase37_decision_alignment.md reproducibility\FILE_MANIFEST.tsv docs\superpowers\phase33_current_progress_handoff.md
git commit -m "docs: record Phase 37 decision-alignment result"
```

## Task 6: Final Status Check

**Files:**
- No file edits expected.

- [ ] **Step 1: Confirm git status and recent commits**

Run:

```powershell
git status --short --branch
git log --oneline -6
```

Expected: branch may be ahead of `origin/main`; no unstaged tracked changes remain. Ignored outputs under `experiments/**/outputs/` may exist locally and must not be committed.

- [ ] **Step 2: Final response content**

Report:

- Phase 37 implementation commit hash;
- Phase 37 docs/result commit hash;
- real `phase37_decision_alignment_status`;
- verification commands and pass/fail results;
- remaining claim boundary: Phase 36 still blocks B2/B3 suitability reward.



