# Phase 60 Information-vs-Optimization Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a read-only Phase 60 attribution audit that reconciles Phase 48/52, Phase 53, Phase 57, and Phase 59 evidence into a clear information-vs-optimization claim boundary.

**Architecture:** Add one focused Phase 60 module that loads existing JSON artifacts, evaluates four attribution axes, assigns a bounded status, and writes JSON/CSV/Markdown outputs. Add a thin experiment runner and a result note; do not retrain policies or modify the formal manuscript.

**Tech Stack:** Python standard library, CSV/JSON artifact writers, pytest, existing Paper11 artifact conventions.

---

## File Structure

- Create `src/paper11_geofm/phase60_information_optimization_attribution.py`.
  This module owns Phase 60 constants, JSON loading, axis evaluation, status rules, claim-boundary recommendations, and artifact writers.
- Create `experiments/phase60_information_optimization_attribution/run_phase60_information_optimization_attribution.py`.
  This CLI exposes one read-only analysis command over existing Phase 48/53/57/59 JSON artifacts.
- Create `tests/test_phase60_information_optimization_attribution.py`.
  Tests cover status rules, missing-field handling, artifact writing, and CLI parsing.
- Create `paper/phase28_results/26_phase60_information_optimization_attribution.md` after the real run.
  This records Phase 60 evidence without changing the formal manuscript.
- Modify `paper/phase28_results/README.md` and `docs/superpowers/phase33_current_progress_handoff.md` after the real run.
  These records point to Phase 60 outputs and preserve the claim boundary.

---

### Task 1: Add Phase 60 Status Logic

**Files:**
- Create: `tests/test_phase60_information_optimization_attribution.py`
- Create: `src/paper11_geofm/phase60_information_optimization_attribution.py`

- [ ] **Step 1: Write the failing status tests**

Create `tests/test_phase60_information_optimization_attribution.py`:

```python
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _phase48(status="compressed_geofm_route_supported"):
    return {
        "phase48_compressed_geofm_status": status,
        "pooled_compressed_control_delta": {
            "mean_reward_delta": 0.2921767818,
            "positive_fraction": 0.6166666667,
            "positive_tile_seed_count": 74,
            "total_tile_seed_count": 120,
        },
        "coverage_issues": {
            "missing_variant_rows": [],
            "duplicate_variant_rows": [],
            "unexpected_variant_rows": [],
        },
    }


def _phase53(status="cluster_mean_support"):
    return {
        "phase53_cluster_mean_status": status,
        "cluster_mean_summary": {
            "mean_cluster_delta": 0.2921767818,
            "exact_sign_flip_mean_p": 0.0196838379,
        },
    }


def _phase57(status="compressed_geometry_consistent"):
    return {
        "phase57_mechanism_status": status,
        "geometry_rows": [
            {"variant_id": "B1", "effective_rank": 9.49, "raw_variance_retention": 1.0},
            {"variant_id": "D4P8", "effective_rank": 5.13, "raw_variance_retention": 0.858},
            {"variant_id": "D4P16", "effective_rank": 7.30, "raw_variance_retention": 0.949},
        ],
        "reward_gain_rows": [
            {"compressed_variant_id": "D4P8", "mean_delta": 0.2356980264},
            {"compressed_variant_id": "D4P16", "mean_delta": 0.3486555373},
        ],
    }


def _phase59(status="matched_dimension_geofm_not_supported"):
    matched_deltas = {
        "D4P8_minus_D5R8": {"mean_delta": -0.0107871307},
        "D4P8_minus_D5S8": {"mean_delta": 0.0003232239},
        "D4P16_minus_D5R16": {"mean_delta": -0.1193811247},
        "D4P16_minus_D5S16": {"mean_delta": 0.060921975},
    }
    if status == "matched_dimension_geofm_supported":
        matched_deltas = {key: {"mean_delta": 0.1} for key in matched_deltas}
    return {
        "phase59_matched_dimension_status": status,
        "learned_policy": {"matched_deltas": matched_deltas},
        "pooled_matched_control_delta": {
            "mean_delta": 0.1 if status == "matched_dimension_geofm_supported" else -0.0172307641,
            "positive_fraction": 0.75 if status == "matched_dimension_geofm_supported" else 0.3666666667,
            "positive_count": 45 if status == "matched_dimension_geofm_supported" else 22,
            "total_count": 60,
        },
        "coverage_issues": {
            "missing_variant_rows": [],
            "duplicate_variant_rows": [],
            "unexpected_variant_rows": [],
        },
    }


def test_phase60_reports_mechanism_claim_narrowed():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48(),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59(),
    )

    assert analysis["phase"] == "phase60_information_optimization_attribution"
    assert analysis["phase60_attribution_status"] == "mechanism_claim_narrowed"
    axes = {row["axis_id"]: row for row in analysis["attribution_axes"]}
    assert axes["compressed_route_performance"]["axis_status"] == "supported"
    assert axes["geofm_specific_matched_dimension"]["axis_status"] == "not_supported"
    assert analysis["claim_boundary_recommendation"] == "narrow_to_low_dimensional_route"


def test_phase60_reports_geofm_specific_information_supported():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48(),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59("matched_dimension_geofm_supported"),
    )

    assert analysis["phase60_attribution_status"] == "geofm_specific_information_supported"
    assert analysis["claim_boundary_recommendation"] == "allow_geofm_specific_matched_dimension_claim"


def test_phase60_reports_low_dimensional_route_uncertain():
    from paper11_geofm.phase60_information_optimization_attribution import (
        build_phase60_information_optimization_attribution,
    )

    analysis = build_phase60_information_optimization_attribution(
        phase48=_phase48("not_supported"),
        phase53=_phase53(),
        phase57=_phase57(),
        phase59=_phase59(),
    )

    assert analysis["phase60_attribution_status"] == "low_dimensional_route_uncertain"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase60_information_optimization_attribution.py -q --basetemp=.pytest_tmp_phase60_red -p no:cacheprovider
```

Expected result: FAIL with `ModuleNotFoundError` for `paper11_geofm.phase60_information_optimization_attribution`.

- [ ] **Step 3: Implement minimal status logic**

Create `src/paper11_geofm/phase60_information_optimization_attribution.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path


PHASE60_CLAIM_BOUNDARY = (
    "Phase 60 is a read-only information-vs-optimization attribution audit over "
    "existing Paper11 Phase 48/53/57/59 artifacts. It reconciles compressed-route "
    "performance, cluster robustness, representation geometry, and matched-dimension "
    "controls; it does not retrain RL policies, does not enable B2/B3, does not "
    "add suitability reward, does not test transfer, and does not validate "
    "independent agronomic suitability."
)

PHASE60_AXIS_FIELDNAMES = [
    "axis_id",
    "axis_label",
    "axis_status",
    "source_phase",
    "primary_metric",
    "primary_value",
    "support_threshold",
    "interpretation",
    "claim_boundary",
]


def build_phase60_information_optimization_attribution(
    phase48: Mapping[str, object],
    phase53: Mapping[str, object],
    phase57: Mapping[str, object],
    phase59: Mapping[str, object],
    source_paths: Mapping[str, object] | None = None,
) -> dict[str, object]:
    axes = [
        _compressed_route_axis(phase48),
        _cluster_robustness_axis(phase53),
        _compressed_geometry_axis(phase57),
        _matched_dimension_axis(phase59),
    ]
    status = _phase60_status(axes)
    return {
        "phase": "phase60_information_optimization_attribution",
        "phase60_attribution_status": status,
        "attribution_axes": axes,
        "source_paths": {} if source_paths is None else dict(source_paths),
        "claim_boundary_recommendation": _claim_boundary_recommendation(status),
        "next_experiment_recommendation": _next_experiment_recommendation(status),
        "conclusion": _phase60_conclusion(status),
        "claim_boundary": PHASE60_CLAIM_BOUNDARY,
    }
```

Then add axis helpers that compute:

```python
def _phase60_status(axes: Sequence[Mapping[str, object]]) -> str:
    by_axis = {str(row["axis_id"]): str(row["axis_status"]) for row in axes}
    early_axes = (
        "compressed_route_performance",
        "cluster_level_robustness",
        "compressed_geometry_consistency",
    )
    if any(by_axis.get(axis) != "supported" for axis in early_axes):
        return "low_dimensional_route_uncertain"
    if by_axis.get("geofm_specific_matched_dimension") == "supported":
        return "geofm_specific_information_supported"
    return "mechanism_claim_narrowed"
```

The axis helpers must enforce these thresholds:

- Phase 48 axis: `phase48_compressed_geofm_status == "compressed_geofm_route_supported"`, mean delta `> 0`, positive fraction `>= 0.5`, and empty coverage issues.
- Phase 53 axis: `phase53_cluster_mean_status == "cluster_mean_support"`, cluster mean `> 0`, and `exact_sign_flip_mean_p < 0.05` when present.
- Phase 57 axis: `phase57_mechanism_status == "compressed_geometry_consistent"`, D4P8/D4P16 effective ranks below B1, D4P8/D4P16 variance retention `> 0`, and D4P8/D4P16 reward mean deltas `> 0`.
- Phase 59 axis: `phase59_matched_dimension_status == "matched_dimension_geofm_supported"`, all four matched deltas positive, pooled mean `> 0`, positive fraction `>= 0.5`, and empty coverage issues.

Use these shared helpers:

```python
def _axis_row(axis_id, axis_label, axis_status, source_phase, primary_metric, primary_value, support_threshold, interpretation):
    return {
        "axis_id": axis_id,
        "axis_label": axis_label,
        "axis_status": axis_status,
        "source_phase": source_phase,
        "primary_metric": primary_metric,
        "primary_value": round(float(primary_value), 10),
        "support_threshold": support_threshold,
        "interpretation": interpretation,
        "claim_boundary": PHASE60_CLAIM_BOUNDARY,
    }


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 60 missing object field: {key}")
    return value


def _required_list(mapping: Mapping[str, object], key: str) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Phase 60 missing list field: {key}")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Phase 60 missing numeric field: {key}")
    return float(value)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase60_information_optimization_attribution.py -q --basetemp=.pytest_tmp_phase60_green -p no:cacheprovider
```

Expected result: `3 passed`.

- [ ] **Step 5: Commit status logic**

Run:

```powershell
git add src/paper11_geofm/phase60_information_optimization_attribution.py tests/test_phase60_information_optimization_attribution.py
git commit -m "feat: add Phase 60 attribution status logic"
```

---

### Task 2: Add Artifact Writers and CLI

**Files:**
- Modify: `src/paper11_geofm/phase60_information_optimization_attribution.py`
- Create: `experiments/phase60_information_optimization_attribution/run_phase60_information_optimization_attribution.py`
- Modify: `tests/test_phase60_information_optimization_attribution.py`

- [ ] **Step 1: Add writer and CLI tests**

Append tests that:

- build Phase 60 analysis from the synthetic fixtures;
- write `phase60_information_optimization_attribution.json`;
- write `phase60_attribution_axes.csv`;
- write `phase60_information_optimization_attribution.md`;
- import the runner and execute it against four temporary JSON files;
- assert stdout includes `Phase 60 attribution status: mechanism_claim_narrowed`.

Use this assertion set:

```python
assert paths["comparison_json"].name == "phase60_information_optimization_attribution.json"
assert paths["axes_csv"].name == "phase60_attribution_axes.csv"
assert paths["readiness_md"].name == "phase60_information_optimization_attribution.md"
assert saved["phase60_attribution_status"] == "mechanism_claim_narrowed"
assert "GeoFM-specific matched-dimension advantage is not supported" in readiness_text
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\test_phase60_information_optimization_attribution.py -q --basetemp=.pytest_tmp_phase60_writer_red -p no:cacheprovider
```

Expected result: FAIL because the writer and runner do not exist.

- [ ] **Step 3: Add load/build/write public functions**

Append:

```python
def build_phase60_information_optimization_attribution_from_paths(
    phase48_json: Path | str,
    phase53_json: Path | str,
    phase57_json: Path | str,
    phase59_json: Path | str,
) -> dict[str, object]:
    paths = {
        "phase48": str(Path(phase48_json)),
        "phase53": str(Path(phase53_json)),
        "phase57": str(Path(phase57_json)),
        "phase59": str(Path(phase59_json)),
    }
    return build_phase60_information_optimization_attribution(
        phase48=_load_json(phase48_json, "Phase 48 JSON"),
        phase53=_load_json(phase53_json, "Phase 53 JSON"),
        phase57=_load_json(phase57_json, "Phase 57 JSON"),
        phase59=_load_json(phase59_json, "Phase 59 JSON"),
        source_paths=paths,
    )
```

Add `write_phase60_information_optimization_attribution_artifacts()` that writes JSON, axis CSV, and Markdown. The Markdown must list status, conclusion, all axis rows, claim-boundary recommendation, next-experiment recommendation, and claim boundary.

- [ ] **Step 4: Create CLI runner**

Create `experiments/phase60_information_optimization_attribution/run_phase60_information_optimization_attribution.py` with argparse flags:

```text
--phase48-json
--phase53-json
--phase57-json
--phase59-json
--output-dir
```

The runner must call `build_phase60_information_optimization_attribution_from_paths()`, write artifacts, print the status and output paths, and return `1` for `OSError` or `ValueError`.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase60_information_optimization_attribution.py -q --basetemp=.pytest_tmp_phase60_writer_green -p no:cacheprovider
```

Expected result: all Phase 60 tests pass.

- [ ] **Step 6: Commit writer and CLI**

Run:

```powershell
git add src/paper11_geofm/phase60_information_optimization_attribution.py experiments/phase60_information_optimization_attribution/run_phase60_information_optimization_attribution.py tests/test_phase60_information_optimization_attribution.py
git commit -m "feat: add Phase 60 attribution runner"
```

---

### Task 3: Run Real Phase 60 and Record Evidence

**Files:**
- Create ignored outputs under: `experiments/phase60_information_optimization_attribution/outputs/phase52_full5_seed3/`
- Create: `paper/phase28_results/26_phase60_information_optimization_attribution.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Run real Phase 60**

Run:

```powershell
python experiments\phase60_information_optimization_attribution\run_phase60_information_optimization_attribution.py --phase48-json experiments\phase52_expanded_cluster_replication\outputs\phase52_full5_seed3_phase48_compressed_rescue\phase48_compressed_geofm_rescue_comparison.json --phase53-json experiments\phase53_cluster_mean_support\outputs\phase52_full5_seed3\phase53_cluster_mean_support.json --phase57-json experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3\phase57_compressed_representation_mechanism.json --phase59-json experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json --output-dir experiments\phase60_information_optimization_attribution\outputs\phase52_full5_seed3
```

Expected result: `Phase 60 attribution status: mechanism_claim_narrowed`.

- [ ] **Step 2: Create Phase 60 result note**

Create `paper/phase28_results/26_phase60_information_optimization_attribution.md` with:

- real Phase 60 status;
- a table of the four axis statuses and primary values;
- the recommendation to narrow the mechanism claim;
- the optional D6-style GeoFM projection-control next experiment;
- the explicit note that no formal manuscript files were changed.

- [ ] **Step 3: Update README and handoff**

Add one README bullet for `26_phase60_information_optimization_attribution.md`.

Append a handoff section with the real command, status, axis outcomes, claim-boundary recommendation, and next-experiment recommendation.

- [ ] **Step 4: Commit real evidence docs**

Run:

```powershell
git add paper/phase28_results/26_phase60_information_optimization_attribution.md paper/phase28_results/README.md docs/superpowers/phase33_current_progress_handoff.md
git commit -m "docs: record Phase 60 attribution evidence"
```

---

### Task 4: Verification and Save

**Files:**
- All Phase 60 implementation, test, runner, and documentation files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_phase60_information_optimization_attribution.py tests\test_phase59_matched_dimension_controls.py tests\test_phase57_compressed_representation_mechanism.py tests\test_phase48_compressed_geofm_rescue.py -q --basetemp=.pytest_tmp_phase60_verify -p no:cacheprovider
```

Expected result: all selected tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected result: `Paper11 smoke check passed.`

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected result: no output and exit code `0`.

- [ ] **Step 4: Review final git state**

Run:

```powershell
git status --short --branch
git log --oneline -6
```

Expected result: branch is `main`, local branch is ahead by Phase 60 commits unless pushed, and no unstaged source/documentation edits remain.

- [ ] **Step 5: Push after final verification**

Run:

```powershell
git push origin main
```

Expected result: `main` synchronizes with `origin/main`.
