# Phase 17 Tiled MaskablePPO Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-tile real-data MaskablePPO readiness smoke check that verifies the tiled B1 contract can run a tiny CPU-only `learn()` and masked `predict()` call.

**Architecture:** Add `paper11_geofm.tiled_maskableppo_readiness` as a narrow module for tile selection, tiled environment creation, MaskablePPO compatibility smoke execution, and JSON writing. Reuse Phase 14 tiled input loading, Phase 4 action masks, and Phase 7 dependency metadata patterns while preserving strict claim boundaries.

**Tech Stack:** Python, Gymnasium, NumPy, stable-baselines3, sb3-contrib, pytest, CSV/JSON artifacts.

---

## File Structure

- Create `src/paper11_geofm/tiled_maskableppo_readiness.py`: tile-index parsing, largest-tile selection, B1 tiled environment creation, MaskablePPO smoke, JSON writer.
- Create `experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py`: CLI runner.
- Create `tests/test_phase17_tiled_maskableppo_readiness.py`: focused tests for selection, reward rejection, smoke summary, writer, and CLI.
- Modify `README.md`: add Phase 17 command and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 17 reproduction section and runtime file list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 17 design, plan, module, CLI, and tests.

## Task 1: Tile Selection and Summary Contract

**Files:**

- Create: `tests/test_phase17_tiled_maskableppo_readiness.py`
- Create: `src/paper11_geofm/tiled_maskableppo_readiness.py`

- [ ] **Step 1: Write failing tests for tile selection and reward rejection**

Create `tests/test_phase17_tiled_maskableppo_readiness.py` with:

```python
import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", 0.25),
            _complete_phase2_feature_row("b2", 0.50),
            _complete_phase2_feature_row("b3", 0.75),
            _complete_phase2_feature_row("b4", 1.00),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 2],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )


def _write_tile_index(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "tile_row", "tile_col", "n_blocks", "block_ids"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": "tile_r000_c000",
                "tile_row": 0,
                "tile_col": 0,
                "n_blocks": 1,
                "block_ids": "b2",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b4",
            }
        )
    return path
```

Add:

```python
def test_phase17_selects_largest_tile_and_reports_contract(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = build_phase17_tiled_contract_summary(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        tile_selection="largest",
        seed=0,
        total_timesteps=8,
    )

    assert summary["phase"] == "phase17_tiled_maskableppo_readiness"
    assert summary["tile_id"] == "tile_r000_c001"
    assert summary["tile_selection"] == "largest"
    assert summary["variant_id"] == "B1"
    assert summary["seed"] == 0
    assert summary["learn_timesteps"] == 8
    assert summary["n_blocks"] == 3
    assert summary["n_features"] == 81
    assert summary["observation_shape"] == 246
    assert summary["action_space_n"] == 3
    assert summary["reward_mode"] == "base_planning_reward"
    assert summary["initial_valid_actions"] == 3
    assert summary["claim_boundary"] == PHASE17_CLAIM_BOUNDARY


def test_phase17_tile_id_override_selects_requested_tile(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    summary = build_phase17_tiled_contract_summary(
        tmp_path / "phase2",
        _write_tile_index(tmp_path / "phase13_tile_index.csv"),
        variant_id="B1",
        tile_id="tile_r000_c000",
    )

    assert summary["tile_id"] == "tile_r000_c000"
    assert summary["tile_selection"] == "explicit"
    assert summary["n_blocks"] == 1
    assert summary["observation_shape"] == 84


def test_phase17_rejects_suitability_reward_variant_by_default(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        build_phase17_tiled_contract_summary,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with pytest.raises(ValueError, match="suitability reward variants are disabled"):
        build_phase17_tiled_contract_summary(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B3",
        )
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_selects_largest_tile_and_reports_contract tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_tile_id_override_selects_requested_tile tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_rejects_suitability_reward_variant_by_default -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.tiled_maskableppo_readiness'`.

- [ ] **Step 3: Implement minimal tile selection and contract summary**

Create `src/paper11_geofm/tiled_maskableppo_readiness.py` with:

```python
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .drl_smoke_env import Phase4InputContractEnv
from .tiled_inputs import load_tiled_variant_input


PHASE17_CLAIM_BOUNDARY = (
    "Phase 17 is a tiled MaskablePPO readiness smoke check; it does not "
    "train, tune, evaluate, or compare a useful DRL policy, does not enable "
    "suitability reward, and does not report planning performance."
)


def build_phase17_tiled_contract_summary(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    tile_id: str | None = None,
    tile_selection: str = "largest",
    seed: int = 0,
    total_timesteps: int = 8,
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")

    selected = _select_tile(Path(tile_index_csv), tile_id, tile_selection)
    try:
        tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            selected["tile_id"],
            variant_id=variant_id,
        )
    except ValueError as exc:
        if "suitability reward variants are disabled" in str(exc):
            raise ValueError(
                "Phase 17 suitability reward variants are disabled by default; "
                "use a representation-only variant such as B0 or B1"
            ) from exc
        raise

    env = Phase4InputContractEnv(tiled, max_steps=int(total_timesteps))
    obs, info = env.reset(seed=int(seed))
    initial_mask = env.action_masks()
    return {
        "phase": "phase17_tiled_maskableppo_readiness",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "tile_id": tiled.tile_id,
        "tile_selection": selected["selection"],
        "variant_id": str(info["variant_id"]),
        "seed": int(seed),
        "learn_timesteps": int(total_timesteps),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "reward_mode": str(info["reward_mode"]),
        "initial_valid_actions": int(initial_mask.sum()),
        "claim_boundary": PHASE17_CLAIM_BOUNDARY,
    }
```

Also implement `_select_tile`, `_read_tile_rows`, `_dependency_metadata`, `_package_metadata`, and a placeholder-free `write_phase17_tiled_maskableppo_readiness_artifact` in later tasks only when tests require it.

- [ ] **Step 4: Run tests and confirm pass**

Run:

```powershell
python -m pytest tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_selects_largest_tile_and_reports_contract tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_tile_id_override_selects_requested_tile tests\test_phase17_tiled_maskableppo_readiness.py::test_phase17_rejects_suitability_reward_variant_by_default -q
```

Expected: `3 passed`.

## Task 2: MaskablePPO Smoke, Writer, and CLI

**Files:**

- Modify: `tests/test_phase17_tiled_maskableppo_readiness.py`
- Modify: `src/paper11_geofm/tiled_maskableppo_readiness.py`
- Create: `experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py`

- [ ] **Step 1: Add failing tests for smoke, writer, and CLI**

Append tests:

```python
import importlib.util
import faulthandler
from contextlib import contextmanager


pytestmark = pytest.mark.filterwarnings("ignore:XPU device count is zero!:UserWarning")


@contextmanager
def _torch_windows_faulthandler_guard():
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    try:
        yield
    finally:
        if was_enabled:
            faulthandler.enable(file=sys.__stderr__)


def _require_maskableppo_dependencies():
    with _torch_windows_faulthandler_guard():
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("sb3_contrib")
```

Add:

```python
def test_phase17_runs_tiny_tiled_maskableppo_smoke(tmp_path):
    _require_maskableppo_dependencies()
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        run_phase17_tiled_maskableppo_readiness,
    )

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        summary = run_phase17_tiled_maskableppo_readiness(
            tmp_path / "phase2",
            _write_tile_index(tmp_path / "phase13_tile_index.csv"),
            variant_id="B1",
            total_timesteps=8,
            seed=0,
        )

    assert summary["masking_supported"] is True
    assert summary["learn_timesteps"] == 8
    assert summary["device"] == "cpu"
    assert summary["predicted_action_valid"] is True
    assert 0 <= summary["predicted_action"] < summary["action_space_n"]
    assert str(summary["selected_block_id"]) in {"b1", "b3", "b4"}
    assert summary["readiness_status"] == "passed_tiled_maskableppo_smoke"
    assert summary["recommendation"] == "tiled_maskableppo_contract_ready_for_larger_controlled_smokes"
    assert summary["dependencies"]["stable_baselines3"]["available"] is True
    assert summary["dependencies"]["sb3_contrib"]["available"] is True
    assert summary["claim_boundary"] == PHASE17_CLAIM_BOUNDARY


def test_phase17_writer_outputs_json(tmp_path):
    from paper11_geofm.tiled_maskableppo_readiness import (
        PHASE17_CLAIM_BOUNDARY,
        write_phase17_tiled_maskableppo_readiness_artifact,
    )

    summary = {
        "phase": "phase17_tiled_maskableppo_readiness",
        "tile_id": "tile_r000_c001",
        "readiness_status": "passed_tiled_maskableppo_smoke",
        "claim_boundary": PHASE17_CLAIM_BOUNDARY,
    }
    path = write_phase17_tiled_maskableppo_readiness_artifact(
        summary,
        tmp_path / "outputs",
    )

    assert path.name == "phase17_tiled_maskableppo_readiness.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == summary


def test_phase17_cli_writes_outputs_and_prints_summary(tmp_path, capsys):
    _require_maskableppo_dependencies()
    runner_path = (
        ROOT
        / "experiments"
        / "phase17_tiled_maskableppo_readiness"
        / "run_phase17_tiled_maskableppo_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("phase17_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path / "phase2")
    with _torch_windows_faulthandler_guard():
        exit_code = module.main(
            [
                "--phase2-output-dir",
                str(tmp_path / "phase2"),
                "--tile-index-csv",
                str(_write_tile_index(tmp_path / "phase13_tile_index.csv")),
                "--variant",
                "B1",
                "--total-timesteps",
                "8",
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Tile: tile_r000_c001" in stdout
    assert "Variant: B1" in stdout
    assert "Masking supported: True" in stdout
    assert "Predicted action valid: True" in stdout
    assert "Readiness status: passed_tiled_maskableppo_smoke" in stdout
    assert "phase17_tiled_maskableppo_readiness.json" in stdout
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\test_phase17_tiled_maskableppo_readiness.py -q
```

Expected: existing Task 1 tests pass, new tests fail because `run_phase17_tiled_maskableppo_readiness`, writer, and CLI are missing.

- [ ] **Step 3: Implement MaskablePPO smoke and writer**

Add to `src/paper11_geofm/tiled_maskableppo_readiness.py`:

```python
def run_phase17_tiled_maskableppo_readiness(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    tile_id: str | None = None,
    tile_selection: str = "largest",
    total_timesteps: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 17 tiled MaskablePPO readiness requires stable-baselines3 and sb3-contrib"
        ) from exc

    selected = _select_tile(Path(tile_index_csv), tile_id, tile_selection)
    try:
        tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            selected["tile_id"],
            variant_id=variant_id,
        )
    except ValueError as exc:
        if "suitability reward variants are disabled" in str(exc):
            raise ValueError(
                "Phase 17 suitability reward variants are disabled by default; "
                "use a representation-only variant such as B0 or B1"
            ) from exc
        raise

    env = Phase4InputContractEnv(tiled, max_steps=int(total_timesteps))
    obs, info = env.reset(seed=int(seed))
    initial_mask = env.action_masks()
    masking_supported = bool(is_masking_supported(env))
    if not masking_supported:
        raise ValueError("Phase 17 tiled env does not expose action_masks for sb3-contrib")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="XPU device count is zero!.*",
            category=UserWarning,
        )
        model.learn(total_timesteps=int(total_timesteps))

    obs, _ = env.reset(seed=int(seed))
    action_masks = env.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    predicted_action = int(action)
    predicted_action_valid = bool(action_masks[predicted_action])
    if not predicted_action_valid:
        raise ValueError("Phase 17 predicted action is not valid under the action mask")

    summary = build_phase17_tiled_contract_summary(
        phase2_output_dir,
        tile_index_csv,
        variant_id=variant_id,
        tile_id=tiled.tile_id,
        seed=seed,
        total_timesteps=total_timesteps,
    )
    summary.update(
        {
            "masking_supported": masking_supported,
            "device": "cpu",
            "predicted_action": predicted_action,
            "predicted_action_valid": predicted_action_valid,
            "selected_block_id": str(env.block_ids[predicted_action]),
            "dependencies": _dependency_metadata(),
            "readiness_status": "passed_tiled_maskableppo_smoke",
            "recommendation": "tiled_maskableppo_contract_ready_for_larger_controlled_smokes",
        }
    )
    return summary


def write_phase17_tiled_maskableppo_readiness_artifact(
    summary: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "phase17_tiled_maskableppo_readiness.json"
    artifact_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path
```

- [ ] **Step 4: Implement CLI**

Create `experiments/phase17_tiled_maskableppo_readiness/run_phase17_tiled_maskableppo_readiness.py` following the Phase 7/16 CLI style with flags:

- `--phase2-output-dir`;
- `--tile-index-csv`;
- `--variant` default `B1`;
- `--tile-id`;
- `--tile-selection` default `largest`;
- `--total-timesteps` default `8`;
- `--seed` default `0`;
- `--output-dir`.

Return `1` for `FileNotFoundError`, `RuntimeError`, or `ValueError`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase17_tiled_maskableppo_readiness.py -q
```

Expected: all Phase 17 tests pass.

## Task 3: Docs, Real Run, Verification, Integration

**Files:**

- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`

- [ ] **Step 1: Update docs and manifest**

Add Phase 17 to:

- repository layout and key entry points in `README.md`;
- reproduction guide after Phase 16;
- runtime code list in the reproduction guide;
- file manifest rows for the design, plan, module, CLI, and tests.

- [ ] **Step 2: Run real Phase 17**

Run:

```powershell
python experiments\phase17_tiled_maskableppo_readiness\run_phase17_tiled_maskableppo_readiness.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variant B1 --tile-selection largest --total-timesteps 8 --seed 0 --output-dir experiments\phase17_tiled_maskableppo_readiness\outputs\real_bishan_largest_tile
```

Expected:

- tile is `tile_r003_c003`;
- variant is `B1`;
- blocks are `2234`;
- features are `81`;
- observation shape is `180957`;
- action space is `2234`;
- masking supported is `True`;
- predicted action valid is `True`;
- readiness status is `passed_tiled_maskableppo_smoke`.

- [ ] **Step 3: Full verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests -q
git diff --check
```

- [ ] **Step 4: Commit, merge, push**

Commit with message `Add Phase 17 tiled MaskablePPO readiness smoke`, push the feature branch, fast-forward merge to `main`, rerun real Phase 17 and full verification on `main`, push `main`, and delete the local feature branch.

---

## Self-Review

- Spec coverage: covers tile selection, B1 reward rejection, MaskablePPO readiness smoke, writer, CLI, docs, real run, and integration.
- Placeholder scan: no placeholder steps remain; every command and expected behavior is explicit.
- Type consistency: function names, file paths, CLI flags, JSON field names, and readiness status match the design.
