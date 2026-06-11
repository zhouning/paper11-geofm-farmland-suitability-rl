# Phase 19 Base Planning Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first deterministic `base_planning_reward` for Paper11 B0/B1 contracts and update readiness evidence so Phase 18 no longer treats base reward modes as constant-zero.

**Architecture:** Add a focused `paper11_geofm.planning_reward` module with a pure reward formula, metadata, and feature validation. Wire `Phase4InputContractEnv` to call it for `base_planning_reward` and to add it to the existing suitability proxy for `base_plus_suitability_reward`; update Phase 18 to use reward metadata instead of brittle source inspection.

**Tech Stack:** Python standard library, NumPy arrays already loaded by Phase 3/4, pytest, JSON artifacts.

---

## File Structure

- Create `src/paper11_geofm/planning_reward.py`: reward constants, required feature list, clipped score helper, row-level reward function, matrix-row wrapper, and metadata/evidence text.
- Modify `src/paper11_geofm/drl_smoke_env.py`: import and call the base reward for base reward modes.
- Modify `src/paper11_geofm/planning_reward_readiness.py`: replace `inspect.getsource` evidence with `planning_reward` metadata.
- Create `tests/test_phase19_base_planning_reward.py`: formula, missing-column, environment, and Phase 18 evidence tests.
- Modify `tests/test_phase4_drl_smoke_env.py`, `tests/test_phase14_tiled_smoke.py`, and `tests/test_phase18_planning_reward_readiness.py`: update assertions that currently expect zero base reward or missing implementation.
- Modify `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and `reproducibility/FILE_MANIFEST.tsv`: document Phase 19 command/evidence and new files.

## Task 1: Reward Formula Tests

**Files:**
- Create: `tests/test_phase19_base_planning_reward.py`
- Create later: `src/paper11_geofm/planning_reward.py`

- [ ] **Step 1: Write the failing pure-formula tests**

Add tests that call the desired API before it exists:

```python
def test_base_planning_reward_matches_weighted_formula():
    from paper11_geofm.planning_reward import (
        BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
        compute_base_planning_reward,
    )

    row = {column: 0.0 for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS}
    row.update(
        {
            "explicit_feature_00": 2.5,
            "explicit_feature_01": 10.0,
            "explicit_feature_02": 28.0,
            "explicit_feature_04": 1.0,
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0,
            "explicit_feature_16": 1.0,
        }
    )

    reward = compute_base_planning_reward(row)

    expected = (
        0.35
        + 0.20
        + 0.10
        + 0.10 * 0.5
        - 0.15 * 0.4
        - 0.05 * 0.8
    )
    assert reward == round(expected, 10)
```

- [ ] **Step 2: Add validation tests**

Add tests for missing explicit columns and clipped area/slope behavior:

```python
def test_base_planning_reward_rejects_missing_explicit_feature():
    from paper11_geofm.planning_reward import compute_base_planning_reward

    with pytest.raises(ValueError, match="explicit_feature_16"):
        compute_base_planning_reward({"explicit_feature_00": 1.0})


def test_base_planning_reward_clips_area_and_slope_terms():
    from paper11_geofm.planning_reward import (
        BASE_PLANNING_REWARD_REQUIRED_COLUMNS,
        compute_base_planning_reward,
    )

    row = {column: 0.0 for column in BASE_PLANNING_REWARD_REQUIRED_COLUMNS}
    row.update(
        {
            "explicit_feature_00": 50.0,
            "explicit_feature_01": 250.0,
            "explicit_feature_02": 350.0,
        }
    )

    assert compute_base_planning_reward(row) == -0.10
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
python -m pytest tests\test_phase19_base_planning_reward.py -q
```

Expected: fail with `ModuleNotFoundError` or import error for `paper11_geofm.planning_reward`.

## Task 2: Reward Module Implementation

**Files:**
- Create: `src/paper11_geofm/planning_reward.py`
- Test: `tests/test_phase19_base_planning_reward.py`

- [ ] **Step 1: Implement the minimal reward module**

Create:

```python
BASE_PLANNING_REWARD_REQUIRED_COLUMNS = (
    "explicit_feature_00",
    "explicit_feature_01",
    "explicit_feature_02",
    "explicit_feature_04",
    "explicit_feature_07",
    "explicit_feature_09",
    "explicit_feature_10",
    "explicit_feature_13",
    "explicit_feature_16",
)

BASE_PLANNING_REWARD_IMPLEMENTED = True
BASE_PLANNING_REWARD_EVIDENCE = (
    "base_planning_reward is implemented as a bounded weighted score over "
    "explicit planning features exported by Phase 11."
)
```

Implement `compute_base_planning_reward(row)` and `compute_base_planning_reward_from_matrix_row(feature_columns, values)`.

- [ ] **Step 2: Run reward tests and verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase19_base_planning_reward.py -q
```

Expected: formula and validation tests pass.

## Task 3: Environment Reward Behavior

**Files:**
- Modify: `src/paper11_geofm/drl_smoke_env.py`
- Modify: `tests/test_phase19_base_planning_reward.py`
- Modify: `tests/test_phase4_drl_smoke_env.py`
- Modify: `tests/test_phase14_tiled_smoke.py`

- [ ] **Step 1: Write failing environment tests**

Add tests that create Phase 2 fixture outputs and assert B1 now returns the formula result, and B3 returns base reward plus suitability proxy:

```python
def test_phase4_env_uses_base_planning_reward_for_b1(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B1")
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == 0.6
    assert info["reward_mode"] == "base_planning_reward"
```

```python
def test_phase4_env_adds_base_reward_to_suitability_proxy_for_b3(tmp_path):
    from paper11_geofm.drl_smoke_env import make_phase4_smoke_env

    _write_ready_phase2_outputs(tmp_path)
    env = make_phase4_smoke_env(tmp_path, "B3")
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert reward == 1.35
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest tests\test_phase19_base_planning_reward.py::test_phase4_env_uses_base_planning_reward_for_b1 tests\test_phase19_base_planning_reward.py::test_phase4_env_adds_base_reward_to_suitability_proxy_for_b3 -q
```

Expected: B1 still returns `0.0` and B3 still returns only `0.75`.

- [ ] **Step 3: Wire the environment**

Update `_contract_reward()` so:

- `base_planning_reward` returns `compute_base_planning_reward_from_matrix_row(...)`;
- `base_plus_suitability_reward` returns base reward plus `suitability_proxy`;
- missing explicit features in base reward modes raise `ValueError`.

- [ ] **Step 4: Update existing zero-reward assertions**

Change existing Phase 4 and Phase 14 tests that expect `0.0` for B1 to expect the deterministic base reward from the fixture.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase19_base_planning_reward.py tests\test_phase4_drl_smoke_env.py tests\test_phase14_tiled_smoke.py -q
```

Expected: all focused reward and smoke environment tests pass.

## Task 4: Phase 18 Evidence Update

**Files:**
- Modify: `src/paper11_geofm/planning_reward_readiness.py`
- Modify: `tests/test_phase18_planning_reward_readiness.py`
- Modify: `tests/test_phase19_base_planning_reward.py`

- [ ] **Step 1: Write failing Phase 18 evidence test**

Add a test that expects:

```python
assert report["base_planning_reward_implemented"] is True
assert "bounded weighted score" in report["base_planning_reward_evidence"]
assert "base_planning_reward_not_implemented" not in report["blocked_reasons"]
assert report["recommended_next_step"] == "resolve_suitability_reward_gate_before_suitability_reward_experiments"
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest tests\test_phase19_base_planning_reward.py::test_phase18_reads_base_reward_metadata_after_phase19 -q
```

Expected: existing Phase 18 source inspection still reports base reward as not implemented.

- [ ] **Step 3: Replace source inspection with metadata**

Remove the `inspect` dependency and have `_base_reward_evidence()` return `BASE_PLANNING_REWARD_IMPLEMENTED` and `BASE_PLANNING_REWARD_EVIDENCE` from `planning_reward.py`.

- [ ] **Step 4: Update existing Phase 18 expected values**

Update prior Phase 18 tests so they no longer expect `base_planning_reward_not_implemented` or the old recommendation once Phase 19 is active.

- [ ] **Step 5: Run Phase 18/19 focused tests**

Run:

```powershell
python -m pytest tests\test_phase18_planning_reward_readiness.py tests\test_phase19_base_planning_reward.py -q
```

Expected: all tests pass.

## Task 5: Documentation, Real Runs, and Verification

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`

- [ ] **Step 1: Update documentation**

Document Phase 19 as the first executable base reward. State explicitly that it does not train, tune, evaluate, or compare a DRL policy and does not enable suitability reward.

- [ ] **Step 2: Run real Phase 14 largest-tile smoke check**

Run:

```powershell
python experiments\phase14_tiled_smoke_env\run_phase14_tiled_smoke.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --tile-id tile_r003_c003 --variant B1 --output-dir experiments\phase14_tiled_smoke_env\outputs\real_bishan_largest_tile
```

Expected: `Step reward` is non-zero.

- [ ] **Step 3: Run real Phase 18 readiness gate**

Run:

```powershell
python experiments\phase18_planning_reward_readiness\run_phase18_planning_reward_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase10-gate experiments\phase11_bishan_dltb_real\outputs\phase10_real\phase10_reward_readiness_gate.json --phase12-audit experiments\phase12_real_scale_audit\outputs\real_bishan\phase12_real_dltb_scale_audit.json --phase17-readiness experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile\phase17_tiled_maskableppo_readiness.json --output-dir experiments\phase18_planning_reward_readiness\outputs\real_bishan
```

Expected: `Base planning reward implemented: True`; suitability reward and flat full-scale training remain false.

- [ ] **Step 4: Run full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

Expected: smoke check passes, all tests pass, and diff check reports no whitespace errors.

## Self-Review

- Spec coverage: formula, feature mapping, missing-feature behavior, Phase 4 behavior, Phase 18 metadata evidence, docs, real run, and verification are covered.
- Placeholder scan: no `TBD`, `TODO`, or incomplete implementation steps remain.
- Type consistency: function names and artifact fields match the Phase 19 design spec and existing codebase conventions.
