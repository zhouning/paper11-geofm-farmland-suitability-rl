# Phase 18 Planning Reward Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable Phase 18 gate that reports whether Paper11 can start true planning-performance DRL experiments on real Bishan tiled artifacts.

**Architecture:** Add a pure `paper11_geofm.planning_reward_readiness` module that reads Phase 2, Phase 10, Phase 12, and optional Phase 17 artifacts, audits the current base-planning-reward implementation boundary, and writes a deterministic JSON readiness report. Add a CLI under `experiments/phase18_planning_reward_readiness/`, tests, and documentation updates.

**Tech Stack:** Python standard library, JSON artifacts, pytest.

---

## File Structure

- Create `src/paper11_geofm/planning_reward_readiness.py`: Phase 18 constants, JSON loading, Phase 2/10/12/17 artifact reduction, current reward-implementation evidence, readiness decision, and JSON writer.
- Create `experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py`: CLI wrapper around the builder/writer.
- Create `tests/test_phase18_planning_reward_readiness.py`: focused tests for blocked readiness, optional Phase 17 handling, writer output, and CLI output.
- Modify `README.md`: add Phase 18 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 18 reproduction section and runtime file list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 18 design, plan, module, CLI, and tests.

## Task 1: Builder Contract Tests

- [ ] **Step 1: Write failing builder tests**

Create `tests/test_phase18_planning_reward_readiness.py` with helpers that write:

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
```

Add `_artifact_fixture(tmp_path)` that creates:

- `phase2/experiment_variants.json` with ready B0/B1 using `base_planning_reward` and ready B2/B3 using `base_plus_suitability_reward`;
- `phase10/phase10_reward_readiness_gate.json` with status `not_ready_for_suitability_reward` and recommendation `do_not_enable_suitability_reward`;
- `phase12/phase12_real_dltb_scale_audit.json` with `real_feature_tables_ready: true`, `representation_only_smoke_allowed: true`, `suitability_reward_allowed: false`, `flat_full_scale_training_ready: false`, `requires_tiled_or_hierarchical_env: true`, `n_blocks: 64984`, and `max_observation_dimension: 5328691`;
- `phase17/phase17_tiled_maskableppo_readiness.json` with `readiness_status: passed_tiled_maskableppo_smoke`, `masking_supported: true`, `predicted_action_valid: true`, and tile metadata.

Add:

```python
def test_phase18_blocks_performance_experiment_when_base_reward_missing(tmp_path):
    from paper11_geofm.planning_reward_readiness import (
        PHASE18_CLAIM_BOUNDARY,
        build_phase18_planning_reward_readiness,
    )

    paths = _artifact_fixture(tmp_path)
    report = build_phase18_planning_reward_readiness(
        paths["phase2_dir"],
        paths["phase10"],
        paths["phase12"],
        phase17_readiness_path=paths["phase17"],
    )

    assert report["phase"] == "phase18_planning_reward_readiness"
    assert report["base_planning_reward_implemented"] is False
    assert report["base_reward_modes"] == {
        "B0": "base_planning_reward",
        "B1": "base_planning_reward",
    }
    assert "returns 0.0" in report["base_planning_reward_evidence"]
    assert report["suitability_reward_allowed"] is False
    assert report["flat_full_scale_training_ready"] is False
    assert report["tiled_maskableppo_api_ready"] is True
    assert report["performance_experiment_ready"] is False
    assert "base_planning_reward_not_implemented" in report["blocked_reasons"]
    assert "suitability_reward_not_allowed" in report["blocked_reasons"]
    assert "flat_full_scale_training_not_ready" in report["blocked_reasons"]
    assert (
        report["recommended_next_step"]
        == "implement_real_tiled_planning_reward_before_policy_evaluation"
    )
    assert report["claim_boundary"] == PHASE18_CLAIM_BOUNDARY
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase18_planning_reward_readiness.py::test_phase18_blocks_performance_experiment_when_base_reward_missing -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.planning_reward_readiness'`.

## Task 2: Builder Implementation

- [ ] **Step 1: Create `src/paper11_geofm/planning_reward_readiness.py`**

Implement:

```python
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .drl_smoke_env import Phase4InputContractEnv


PHASE18_CLAIM_BOUNDARY = (
    "Phase 18 is a planning-reward readiness audit; it does not implement a "
    "planning reward, train, tune, evaluate, or compare a DRL policy, enable "
    "suitability reward, or report planning performance."
)
```

Required public functions:

- `build_phase18_planning_reward_readiness(phase2_output_dir, phase10_gate_path, phase12_audit_path, phase17_readiness_path=None)`;
- `write_phase18_planning_reward_readiness(report, output_dir)`.

Required behavior:

- read Phase 2 `experiment_variants.json`;
- summarize B0/B1 readiness and reward modes;
- determine current base reward implementation using source inspection of `Phase4InputContractEnv._contract_reward`;
- set `base_planning_reward_implemented` to `False` when non-suitability modes still return `0.0`;
- read Phase 10 status/recommendation and derive `phase10_allows_suitability_reward`;
- read Phase 12 flags and preserve `real_feature_tables_ready`, `representation_only_smoke_allowed`, `suitability_reward_allowed`, `flat_full_scale_training_ready`, `requires_tiled_or_hierarchical_env`, `n_blocks`, and `max_observation_dimension`;
- read optional Phase 17 and derive `tiled_maskableppo_api_ready` only when status is `passed_tiled_maskableppo_smoke`, masking is supported, and predicted action is valid;
- compute blocked reasons;
- set `performance_experiment_ready` to `False` unless B0/B1 are ready, base planning reward is implemented, real feature tables are ready, and either flat training is ready or tiled API readiness is available;
- choose recommendation `implement_real_tiled_planning_reward_before_policy_evaluation` when the base reward is missing.

- [ ] **Step 2: Run builder test**

Run:

```powershell
python -m pytest tests\test_phase18_planning_reward_readiness.py::test_phase18_blocks_performance_experiment_when_base_reward_missing -q
```

Expected: pass.

## Task 3: Optional Phase 17, Writer, and CLI Tests

- [ ] **Step 1: Add tests**

Append tests for:

- missing optional Phase 17 path, expecting `tiled_maskableppo_api_ready: false` and `tiled_maskableppo_status: not_supplied`;
- writer output file named `phase18_planning_reward_readiness.json`;
- CLI output containing the base reward status, suitability reward status, tiled API status, performance readiness, recommendation, artifact path, and claim boundary.

- [ ] **Step 2: Run and confirm failure where implementation is missing**

Run:

```powershell
python -m pytest tests\test_phase18_planning_reward_readiness.py -q
```

Expected: writer or CLI test fails until the missing implementation is added.

- [ ] **Step 3: Implement writer and CLI**

Create `experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py` with flags:

- `--phase2-output-dir`;
- `--phase10-gate`;
- `--phase12-audit`;
- optional `--phase17-readiness`;
- `--output-dir`.

The CLI prints:

- base planning reward implemented;
- suitability reward allowed;
- flat full-scale training ready;
- tiled MaskablePPO API ready;
- performance experiment ready;
- blocked reasons;
- recommendation;
- artifact path;
- claim boundary.

Return `1` for `FileNotFoundError` or `ValueError`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase18_planning_reward_readiness.py -q
```

Expected: all Phase 18 tests pass.

## Task 4: Documentation and Manifest

- [ ] **Step 1: Update README**

Add `experiments/phase18_planning_reward_readiness/` to the repository layout, add the real Bishan Phase 18 command after Phase 17, and add the runner to key entry points.

- [ ] **Step 2: Update reproduction guide**

Add a Phase 18 section after Phase 17 with the real Bishan command and expected current outcome.

- [ ] **Step 3: Update file manifest**

Add rows for:

- `docs/superpowers/specs/2026-06-11-phase18-planning-reward-readiness-design.md`;
- `docs/superpowers/plans/2026-06-11-phase18-planning-reward-readiness.md`;
- `src/paper11_geofm/planning_reward_readiness.py`;
- `experiments/phase18_planning_reward_readiness/run_phase18_planning_reward_readiness.py`;
- `tests/test_phase18_planning_reward_readiness.py`.

## Task 5: Real Run, Verification, Commit, Merge

- [ ] **Step 1: Run real Phase 18 readiness gate**

Run:

```powershell
python experiments\phase18_planning_reward_readiness\run_phase18_planning_reward_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase12-audit experiments\phase12_real_scale_audit\outputs\real_bishan\phase12_real_dltb_scale_audit.json --phase17-readiness experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile\phase17_tiled_maskableppo_readiness.json --output-dir experiments\phase18_planning_reward_readiness\outputs\real_bishan
```

Expected current Bishan outcome:

- base planning reward implemented: `False`;
- suitability reward allowed: `False`;
- flat full-scale training ready: `False`;
- tiled MaskablePPO API ready: `True`;
- performance experiment ready: `False`;
- recommendation: `implement_real_tiled_planning_reward_before_policy_evaluation`.

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
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\planning_reward_readiness.py experiments\phase18_planning_reward_readiness\run_phase18_planning_reward_readiness.py tests\test_phase18_planning_reward_readiness.py docs\superpowers\plans\2026-06-11-phase18-planning-reward-readiness.md
git commit -m "Add Phase 18 planning reward readiness gate"
```

- [ ] **Step 4: Integrate**

Push the feature branch, fast-forward merge it to `main`, rerun the real Phase 18 gate plus full verification on `main`, push `main`, and delete the local feature branch after `main` is synchronized with `origin/main`.

---

## Self-Review

- Spec coverage: tasks cover the builder, writer, CLI, docs, manifest, real run, verification, and integration.
- Placeholder scan: no placeholder sections remain.
- Type consistency: function names, artifact filenames, CLI flags, JSON field names, and recommendation strings match the design spec.
