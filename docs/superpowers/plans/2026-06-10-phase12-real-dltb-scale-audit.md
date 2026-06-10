# Phase 12 Real DLTB Scale Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable Phase 12 audit gate that reads real Bishan Phase 11/2/9/10 artifacts and reports defensible next actions.

**Architecture:** Add a pure `paper11_geofm.real_scale_audit` module that loads JSON artifacts, computes per-variant real-scale dimensions, and emits decision flags. Add a CLI runner under `experiments/phase12_real_scale_audit/`, plus tests and documentation updates.

**Tech Stack:** Python standard library, JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/real_scale_audit.py`: Phase 12 constants, JSON loading, artifact consistency checks, per-variant scale audit, gate decisions, and JSON writer.
- Create `experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py`: CLI wrapper around the audit builder/writer.
- Create `tests/test_phase12_real_scale_audit.py`: synthetic artifact tests for decisions, threshold behavior, writer output, and CLI output.
- Modify `README.md`: add Phase 12 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 12 after Phase 11 and renumber later sections.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 12 design, plan, module, CLI, and tests.

## Task 1: Builder Contract Tests

- [ ] **Step 1: Write failing builder tests**

Create `tests/test_phase12_real_scale_audit.py` with helpers:

```python
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _artifact_fixture(tmp_path: Path) -> dict[str, Path]:
    phase11 = _write_json(
        tmp_path / "phase11" / "phase11_bishan_dltb_adapter_summary.json",
        {
            "phase": "phase11_bishan_dltb_real_adapter",
            "rows_exported": 10,
            "rows_read_in_bbox": 12,
            "category_counts": {"Farmland": 4, "Other": 6},
            "label_positive_counts": {
                "current_farmland_label": 4,
                "low_slope_farmland_label": 2,
                "farmland_or_orchard_label": 5,
            },
        },
    )
    phase2_dir = tmp_path / "phase2"
    _write_json(
        phase2_dir / "summary.json",
        {
            "n_blocks": 10,
            "feature_groups_present": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
            "feature_readiness": {
                "B0": {"ready": True, "missing": []},
                "B1": {"ready": True, "missing": []},
                "B2": {"ready": True, "missing": []},
                "B3": {"ready": True, "missing": []},
            },
        },
    )
    variants = {
        "B0": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": ["explicit_feature_00", "explicit_feature_01"],
            "reward": "base_planning_reward",
            "feature_table": "variant_B0_features.csv",
            "state_groups": ["explicit_planning_features"],
        },
        "B1": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
            ],
            "reward": "base_planning_reward",
            "feature_table": "variant_B1_features.csv",
            "state_groups": ["explicit_planning_features", "geofm_embedding"],
        },
        "B2": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B2_features.csv",
            "state_groups": ["explicit_planning_features", "suitability_proxy"],
        },
        "B3": {
            "ready": True,
            "missing": [],
            "row_count": 10,
            "required_columns": [
                "explicit_feature_00",
                "explicit_feature_01",
                "embedding_mean_00",
                "suitability_proxy",
            ],
            "reward": "base_plus_suitability_reward",
            "feature_table": "variant_B3_features.csv",
            "state_groups": [
                "explicit_planning_features",
                "geofm_embedding",
                "suitability_proxy",
            ],
        },
    }
    _write_json(phase2_dir / "experiment_variants.json", {"variants": variants})
    phase9 = _write_json(
        tmp_path / "phase9" / "phase9_proxy_validation_report.json",
        {
            "n_blocks": 10,
            "labels": {
                "current_farmland_label": {
                    "interpretation": "positive_alignment",
                    "rank_auc": 0.51,
                    "mean_difference": 0.01,
                },
                "low_slope_farmland_label": {
                    "interpretation": "negative_or_no_alignment",
                    "rank_auc": 0.49,
                    "mean_difference": -0.01,
                },
            },
        },
    )
    phase10 = _write_json(
        tmp_path / "phase10" / "phase10_reward_readiness_gate.json",
        {
            "status": "not_ready_for_suitability_reward",
            "recommendation": "do_not_enable_suitability_reward",
            "passing_label_count": 1,
            "failing_label_count": 1,
            "insufficient_label_count": 0,
            "labels": {
                "current_farmland_label": {"passes_gate": True},
                "low_slope_farmland_label": {"passes_gate": False},
            },
        },
    )
    return {
        "phase11": phase11,
        "phase2_dir": phase2_dir,
        "phase9": phase9,
        "phase10": phase10,
    }
```

Add the first test:

```python
def test_phase12_audit_blocks_reward_and_flat_training_when_gate_fails(tmp_path):
    from paper11_geofm.real_scale_audit import (
        PHASE12_CLAIM_BOUNDARY,
        build_phase12_real_scale_audit,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=20,
    )

    assert report["n_blocks"] == 10
    assert report["real_feature_tables_ready"] is True
    assert report["representation_only_smoke_allowed"] is True
    assert report["suitability_reward_allowed"] is False
    assert report["flat_full_scale_training_ready"] is False
    assert report["requires_tiled_or_hierarchical_env"] is True
    assert report["variants"]["B3"]["observation_dimension"] == 43
    assert report["max_observation_dimension"] == 43
    assert report["phase10"]["status"] == "not_ready_for_suitability_reward"
    assert "keep_suitability_reward_disabled" in report["recommendation"]
    assert report["claim_boundary"] == PHASE12_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py::test_phase12_audit_blocks_reward_and_flat_training_when_gate_fails -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.real_scale_audit'`.

## Task 2: Builder Implementation

- [ ] **Step 1: Create `src/paper11_geofm/real_scale_audit.py`**

Implement:

```python
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PHASE12_CLAIM_BOUNDARY = (
    "Phase 12 audits real DLTB-derived artifact readiness and scale; it does "
    "not train, tune, evaluate, or compare a DRL policy and does not override "
    "the Phase 10 reward-readiness gate."
)
DEFAULT_FLAT_OBSERVATION_THRESHOLD = 1_000_000
REQUIRED_VARIANTS = ("B0", "B1", "B2", "B3")
FLOAT32_BYTES = 4
MIB = 1024 * 1024
```

Required public functions:

- `build_phase12_real_scale_audit(phase11_summary_path, phase2_output_dir, phase9_report_path, phase10_gate_path, flat_observation_threshold=DEFAULT_FLAT_OBSERVATION_THRESHOLD)`;
- `write_phase12_real_scale_audit(report, output_dir)`.

Required behavior:

- load all JSON files and raise `FileNotFoundError` for missing paths;
- validate `flat_observation_threshold > 0`;
- derive `n_blocks` from Phase 2 summary;
- derive Phase 11 exported row count and compare it to Phase 2 `n_blocks`;
- derive per-variant:
  - `ready`;
  - `row_count`;
  - `n_features`;
  - `reward_mode`;
  - `feature_table`;
  - `state_groups`;
  - `observation_dimension = row_count * n_features + 3`;
  - `estimated_observation_mib = round(observation_dimension * 4 / 1048576, 6)`;
  - `within_flat_observation_threshold`;
- derive label summaries from Phase 9 `labels`;
- derive Phase 10 status/recommendation/counts;
- compute decision flags exactly as specified in the design;
- emit a deterministic recommendation string:
  - if real feature tables are not ready: `repair_real_feature_tables_before_downstream_experiments`;
  - else if suitability reward is not allowed and a tiled env is required: `continue_real_dltb_representation_only_analysis; keep_suitability_reward_disabled; design_tiled_or_hierarchical_env_before_full_scale_training`;
  - else if suitability reward is not allowed: `continue_real_dltb_representation_only_analysis; keep_suitability_reward_disabled`;
  - else if a tiled env is required: `design_tiled_or_hierarchical_env_before_full_scale_training`;
  - otherwise: `flat_full_scale_training_gate_passed_for_smoke_only`;

- [ ] **Step 2: Run builder test**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py::test_phase12_audit_blocks_reward_and_flat_training_when_gate_fails -q
```

Expected: pass.

## Task 3: Writer, Threshold, and Error Tests

- [ ] **Step 1: Add tests**

Append:

```python
def test_phase12_can_pass_flat_training_gate_when_reward_ready_and_threshold_high(tmp_path):
    from paper11_geofm.real_scale_audit import build_phase12_real_scale_audit

    paths = _artifact_fixture(tmp_path)
    phase10_payload = json.loads(paths["phase10"].read_text(encoding="utf-8"))
    phase10_payload["status"] = "ready_for_suitability_reward"
    phase10_payload["recommendation"] = "enable_bounded_suitability_reward_smoke"
    paths["phase10"].write_text(json.dumps(phase10_payload), encoding="utf-8")

    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=100,
    )

    assert report["suitability_reward_allowed"] is True
    assert report["requires_tiled_or_hierarchical_env"] is False
    assert report["flat_full_scale_training_ready"] is True
```

```python
def test_phase12_writer_outputs_json_report(tmp_path):
    from paper11_geofm.real_scale_audit import (
        build_phase12_real_scale_audit,
        write_phase12_real_scale_audit,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase12_real_scale_audit(
        paths["phase11"],
        paths["phase2_dir"],
        paths["phase9"],
        paths["phase10"],
        flat_observation_threshold=20,
    )
    output_path = write_phase12_real_scale_audit(report, tmp_path / "outputs")

    assert output_path.name == "phase12_real_dltb_scale_audit.json"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["phase"] == "phase12_real_dltb_scale_audit"
    assert written["variants"]["B0"]["n_features"] == 2
```

```python
def test_phase12_rejects_non_positive_threshold(tmp_path):
    from paper11_geofm.real_scale_audit import build_phase12_real_scale_audit

    paths = _artifact_fixture(tmp_path)
    try:
        build_phase12_real_scale_audit(
            paths["phase11"],
            paths["phase2_dir"],
            paths["phase9"],
            paths["phase10"],
            flat_observation_threshold=0,
        )
    except ValueError as exc:
        assert "flat_observation_threshold must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py -q
```

Expected: all Phase 12 tests pass.

## Task 4: CLI Runner

- [ ] **Step 1: Add CLI test**

Append:

```python
def test_phase12_cli_writes_audit_report(tmp_path, capsys):
    paths = _artifact_fixture(tmp_path)
    runner_path = (
        ROOT
        / "experiments"
        / "phase12_real_scale_audit"
        / "run_phase12_real_scale_audit.py"
    )
    spec = importlib.util.spec_from_file_location("phase12_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase11-summary",
            str(paths["phase11"]),
            "--phase2-output-dir",
            str(paths["phase2_dir"]),
            "--phase9-report",
            str(paths["phase9"]),
            "--phase10-gate",
            str(paths["phase10"]),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--flat-observation-threshold",
            "20",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Real feature tables ready: True" in stdout
    assert "Suitability reward allowed: False" in stdout
    assert "Flat full-scale training ready: False" in stdout
    assert "phase12_real_dltb_scale_audit.json" in stdout
```

- [ ] **Step 2: Run and confirm failure before CLI exists**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py::test_phase12_cli_writes_audit_report -q
```

Expected: fail because the CLI file does not exist.

- [ ] **Step 3: Create CLI**

Create `experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py` with flags:

- `--phase11-summary`;
- `--phase2-output-dir`;
- `--phase9-report`;
- `--phase10-gate`;
- `--output-dir`;
- `--flat-observation-threshold`.

The CLI prints:

- blocks;
- max observation dimension;
- real feature table readiness;
- representation-only allowance;
- suitability reward allowance;
- flat full-scale training readiness;
- tiled/hierarchical requirement;
- output path;
- claim boundary.

Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 4: Run CLI test**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py::test_phase12_cli_writes_audit_report -q
```

Expected: pass.

## Task 5: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase12_real_scale_audit/` to the repository layout, add the Phase 12 real audit command after the Phase 11 command block, and add the runner to key entry points.

- [ ] **Step 2: Update reproduction guide**

Add a Phase 12 section after Phase 11 with the real Bishan command and expected flags. Renumber later sections.

- [ ] **Step 3: Update file manifest**

Add rows for:

- `docs/superpowers/specs/2026-06-10-phase12-real-dltb-scale-audit-design.md`;
- `docs/superpowers/plans/2026-06-10-phase12-real-dltb-scale-audit.md`;
- `src/paper11_geofm/real_scale_audit.py`;
- `experiments/phase12_real_scale_audit/run_phase12_real_scale_audit.py`;
- `tests/test_phase12_real_scale_audit.py`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase12_real_scale_audit.py -q
```

Expected: all Phase 12 tests pass.

## Task 6: Real Run, Verification, Commit, Merge

- [ ] **Step 1: Run real Phase 12 audit**

Run:

```powershell
python experiments\phase12_real_scale_audit\run_phase12_real_scale_audit.py --phase11-summary experiments\phase11_bishan_dltb_real\outputs\adapter\phase11_bishan_dltb_adapter_summary.json --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase9-report experiments\phase11_bishan_dltb_real\outputs\phase9_real\phase9_proxy_validation_report.json --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --output-dir experiments\phase12_real_scale_audit\outputs\real_bishan
```

Expected real Bishan outcome:

- blocks: `64984`;
- max observation dimension: `5328691`;
- real feature tables ready: `True`;
- representation-only smoke allowed: `True`;
- suitability reward allowed: `False`;
- flat full-scale training ready: `False`;
- tiled/hierarchical environment required: `True`.

- [ ] **Step 2: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

Expected: smoke check passes, all tests pass, and diff check reports no whitespace errors.

- [ ] **Step 3: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\real_scale_audit.py experiments\phase12_real_scale_audit\run_phase12_real_scale_audit.py tests\test_phase12_real_scale_audit.py docs\superpowers\plans\2026-06-10-phase12-real-dltb-scale-audit.md
git commit -m "Add Phase 12 real DLTB scale audit"
```

- [ ] **Step 4: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun the real Phase 12 audit plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: tasks cover the builder, writer, CLI, docs, manifest, real run, and integration.
- Placeholder scan: no placeholder steps remain.
- Type consistency: function names, artifact filenames, CLI flags, and decision keys match the design spec.
