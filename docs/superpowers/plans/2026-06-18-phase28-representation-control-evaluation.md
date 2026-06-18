# Phase 28 Representation-Control Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase 28 diagnostic package that evaluates B1 against B0, D2, D3, D4P8, and D4P16 under the same padded held-out base-reward protocol.

**Architecture:** Add a focused Phase 28 module that routes B variants to Phase 2 outputs and D variants to Phase 8 outputs, reuses the Phase 25 padded environment, computes B1-vs-control diagnostics from summary rows, and writes CSV/JSON/Markdown artifacts. Keep Phase 25 public B0/B1 behavior unchanged and expose Phase 28 through a separate runner and claim boundary.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `pathlib`, `statistics`, `warnings`), NumPy, pytest, existing `paper11_geofm` loaders and `Phase25PaddedTileEnv`, optional runtime dependency on `stable-baselines3` and `sb3-contrib` for real training.

---

## File Structure

- Create `src/paper11_geofm/phase28_representation_controls.py`: Phase 28 contract builder, source routing, optional MaskablePPO training runner, analysis builder, artifact writer, and Markdown readiness note.
- Create `experiments/phase28_representation_controls/run_phase28_representation_controls.py`: CLI for `run-and-analyze` and `analyze-only`.
- Create `tests/test_phase28_representation_controls.py`: fixture builders, contract tests, source-routing tests, analysis/status tests, writer tests, fake-training runner test, and CLI validation tests.
- Modify `README.md`: add a Phase 28 diagnostic section after Phase 27.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 28 commands, input requirements, outputs, and claim boundary.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add the Phase 28 spec, plan, module, runner, tests, and documentation rows.
- Modify `paper/phase26_results/02_next_experiment_matrix.md`: mark Phase 28 representation-control evaluation as the next execution target after Phase 27.

Do not modify Phase 25 public function names, Phase 26 claim rules, suitability reward code, B2/B3 contracts, or legacy county/parcel environments.

## Task 1: Write Phase 28 Failing Tests

**Files:**
- Create: `tests/test_phase28_representation_controls.py`

- [ ] **Step 1: Add fixtures, contract tests, source-routing tests, analysis tests, writer tests, fake runner test, and CLI tests**

Create `tests/test_phase28_representation_controls.py` with this structure:

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, slope_mean, farmland, suitability=0.75):
    row = {"block_id": block_id, "suitability_proxy": suitability}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim) / 100.0
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = 0.0
    row.update(
        {
            "explicit_feature_00": 2.0,
            "explicit_feature_01": float(slope_mean),
            "explicit_feature_02": float(slope_mean) + 5.0,
            "explicit_feature_04": float(farmland),
            "explicit_feature_07": 0.0,
            "explicit_feature_09": 0.0,
            "explicit_feature_10": 0.0,
            "explicit_feature_13": 1.0 if slope_mean <= 15.0 else 0.0,
            "explicit_feature_16": float(farmland),
        }
    )
    return row


def _write_ready_phase2_outputs(output_dir: Path):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("b1", slope_mean=8.0, farmland=1.0),
            _complete_phase2_feature_row("b2", slope_mean=30.0, farmland=0.0),
            _complete_phase2_feature_row("b3", slope_mean=12.0, farmland=1.0),
            _complete_phase2_feature_row("b4", slope_mean=25.0, farmland=0.0),
            _complete_phase2_feature_row("b5", slope_mean=6.0, farmland=1.0),
            _complete_phase2_feature_row("b6", slope_mean=22.0, farmland=0.0),
        ],
        output_dir,
        {
            "metadata_source": "test",
            "base_year_requested": 2020,
            "base_year_used": 2020,
            "years": [2020],
            "grid_shape": [2, 3],
            "embedding_dim": 64,
            "mapping_mode": "test",
        },
    )


def _write_phase8_outputs(phase2_dir: Path, output_dir: Path):
    from paper11_geofm.ablation_controls import (
        build_phase8_ablation_controls,
        write_phase8_ablation_artifacts,
    )

    protocol = build_phase8_ablation_controls(
        phase2_dir,
        seed=0,
        pca_dimensions=(8, 16),
    )
    return write_phase8_ablation_artifacts(protocol, output_dir)


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
                "block_ids": "b6",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c001",
                "tile_row": 0,
                "tile_col": 1,
                "n_blocks": 3,
                "block_ids": "b1;b3;b5",
            }
        )
        writer.writerow(
            {
                "tile_id": "tile_r000_c002",
                "tile_row": 0,
                "tile_col": 2,
                "n_blocks": 2,
                "block_ids": "b2;b4",
            }
        )
    return path


def _phase28_summary_rows(status_case="supported"):
    rewards_by_case = {
        "supported": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.6, 1.4, 1.5, 1.3],
            "D2": [1.0, 1.0, 1.0, 1.0],
            "D3": [0.9, 1.0, 0.8, 1.0],
            "D4P8": [1.2, 1.1, 1.0, 1.0],
            "D4P16": [1.2, 1.1, 1.0, 1.0],
        },
        "control_limited": {
            "B0": [1.6, 1.5, 1.4, 1.4],
            "B1": [1.2, 1.3, 1.2, 1.3],
            "D2": [1.0, 1.0, 1.1, 1.1],
            "D3": [1.1, 1.1, 1.2, 1.2],
            "D4P8": [1.0, 1.0, 1.1, 1.1],
            "D4P16": [1.0, 1.0, 1.1, 1.1],
        },
        "not_distinguishable": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.0, 1.0, 1.0, 1.0],
            "D2": [1.1, 1.1, 1.1, 1.1],
            "D3": [1.2, 1.2, 1.2, 1.2],
            "D4P8": [0.8, 0.8, 0.8, 0.8],
            "D4P16": [0.8, 0.8, 0.8, 0.8],
        },
        "compression_match": {
            "B0": [1.0, 1.0, 1.0, 1.0],
            "B1": [1.2, 1.2, 1.2, 1.2],
            "D2": [1.0, 1.0, 1.0, 1.0],
            "D3": [1.0, 1.0, 1.0, 1.0],
            "D4P8": [1.2, 1.2, 1.2, 1.2],
            "D4P16": [1.1, 1.1, 1.1, 1.1],
        },
    }
    rewards = rewards_by_case[status_case]
    rows = []
    pairs = [
        ("tile_eval_a", 0),
        ("tile_eval_a", 1),
        ("tile_eval_b", 0),
        ("tile_eval_b", 1),
    ]
    for pair_index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards.items():
            rows.append(
                {
                    "row_type": "trained_policy",
                    "variant_id": variant_id,
                    "train_tile_id": "tile_train",
                    "eval_tile_id": tile_id,
                    "eval_tile_rank": 1 if tile_id == "tile_eval_a" else 2,
                    "seed": seed,
                    "phase25_seed_rank": seed + 1,
                    "train_timesteps": 128,
                    "eval_max_steps": 4,
                    "max_blocks": 3,
                    "train_n_blocks": 3,
                    "eval_n_blocks": 2,
                    "n_features": 17 if variant_id == "B0" else 81,
                    "observation_shape": 260,
                    "action_space_n": 3,
                    "episode_steps": 2,
                    "terminated": True,
                    "truncated": False,
                    "all_actions_valid": True,
                    "invalid_action_count": 0,
                    "total_contract_reward": values[pair_index],
                    "selected_block_ids": "b1;b2",
                    "claim_boundary": "phase28 fixture",
                }
            )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    from paper11_geofm.padded_heldout_policy import SUMMARY_FIELDNAMES

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase28_contract_routes_b_and_d_variant_sources(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        PHASE28_CLAIM_BOUNDARY,
        build_phase28_representation_control_contract,
    )

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    contract = build_phase28_representation_control_contract(
        phase2_output_dir=phase2_dir,
        phase8_output_dir=phase8_dir,
        tile_index_csv=tile_index,
        variants=("B0", "B1", "D2", "D3", "D4P8", "D4P16"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds="0,1",
        max_eval_tiles=2,
    )

    assert contract["phase"] == "phase28_representation_control_evaluation"
    assert contract["variants"] == ["B0", "B1", "D2", "D3", "D4P8", "D4P16"]
    assert contract["variant_source_dirs"]["B0"] == str(phase2_dir)
    assert contract["variant_source_dirs"]["B1"] == str(phase2_dir)
    assert contract["variant_source_dirs"]["D2"] == str(phase8_dir)
    assert contract["variant_source_dirs"]["D4P16"] == str(phase8_dir)
    assert contract["train_tile_id"] == "tile_r000_c001"
    assert contract["eval_tile_ids"] == ["tile_r000_c002", "tile_r000_c000"]
    assert contract["max_blocks"] == 3
    assert contract["claim_boundary"] == PHASE28_CLAIM_BOUNDARY


def test_phase28_contract_rejects_unsupported_and_missing_b1(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_contract,
    )

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")

    with pytest.raises(ValueError, match="unsupported Phase 28 variants"):
        build_phase28_representation_control_contract(
            phase2_dir,
            phase8_dir,
            tile_index,
            variants=("B3",),
        )

    with pytest.raises(ValueError, match="requires B1"):
        build_phase28_representation_control_contract(
            phase2_dir,
            phase8_dir,
            tile_index,
            variants=("B0", "D2"),
        )


def test_phase28_analysis_computes_b1_control_deltas_and_supported_status(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    summary_csv = _write_summary_csv(
        tmp_path / "phase28_representation_control_summary.csv",
        _phase28_summary_rows("supported"),
    )

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase"] == "phase28_representation_control_analysis"
    assert analysis["phase28_diagnostic_status"] == "representation_signal_supported"
    assert analysis["learned_policy"]["mean_reward_by_variant"]["B1"] == 1.45
    assert analysis["learned_policy"]["comparator_deltas"]["B1_minus_D2"]["mean_reward_delta"] == 0.45
    assert analysis["learned_policy"]["comparator_deltas"]["B1_minus_D3"]["positive_tile_seed_count"] == 4
    assert len(analysis["tile_seed_delta_rows"]) == 20


@pytest.mark.parametrize(
    ("status_case", "expected_status"),
    [
        ("control_limited", "representation_signal_control_limited"),
        ("not_distinguishable", "representation_signal_not_distinguishable"),
        ("compression_match", "compression_matches_raw"),
    ],
)
def test_phase28_diagnostic_status_rules(tmp_path, status_case, expected_status):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    summary_csv = _write_summary_csv(
        tmp_path / f"{status_case}.csv",
        _phase28_summary_rows(status_case),
    )

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase28_diagnostic_status"] == expected_status


def test_phase28_reports_insufficient_for_missing_comparator_rows(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
    )

    rows = [
        row
        for row in _phase28_summary_rows("supported")
        if not (
            row["row_type"] == "trained_policy"
            and row["variant_id"] == "D3"
            and row["eval_tile_id"] == "tile_eval_b"
            and row["seed"] == 1
        )
    ]
    summary_csv = _write_summary_csv(tmp_path / "missing.csv", rows)

    analysis = build_phase28_representation_control_analysis(summary_csv)

    assert analysis["phase28_diagnostic_status"] == "insufficient"
    assert analysis["coverage_issues"]["missing_variant_rows"] == [
        {"eval_tile_id": "tile_eval_b", "seed": 1, "variant_id": "D3"}
    ]


def test_phase28_writer_outputs_summary_trace_comparison_delta_and_markdown(tmp_path):
    from paper11_geofm.phase28_representation_controls import (
        build_phase28_representation_control_analysis,
        write_phase28_representation_control_artifacts,
    )

    summary_rows = _phase28_summary_rows("supported")
    analysis = build_phase28_representation_control_analysis(summary_rows)
    protocol = {
        **analysis,
        "summaries": summary_rows,
        "traces": {"trained_policy": {"B1": {"tile_eval_a": {"0": []}}}},
    }

    paths = write_phase28_representation_control_artifacts(
        protocol,
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase28_representation_control_summary.csv"
    assert paths["traces_json"].name == "phase28_representation_control_traces.json"
    assert paths["comparison_json"].name == "phase28_representation_control_comparison.json"
    assert paths["tile_seed_delta_csv"].name == "phase28_tile_seed_delta_table.csv"
    assert paths["control_readiness_md"].name == "phase28_control_readiness.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase28_diagnostic_status"] == "representation_signal_supported"
    markdown = paths["control_readiness_md"].read_text(encoding="utf-8")
    assert "representation_signal_supported" in markdown
    assert "GeoFM improves planning decisions" not in markdown


def test_phase28_run_uses_fake_training_model_for_all_variants(tmp_path, monkeypatch):
    from paper11_geofm import phase28_representation_controls as phase28

    class FakeModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            valid_actions = [
                index
                for index, valid in enumerate(action_masks.tolist())
                if bool(valid)
            ]
            return valid_actions[0], None

    phase2_dir = tmp_path / "phase2"
    phase8_dir = tmp_path / "phase8"
    _write_ready_phase2_outputs(phase2_dir)
    _write_phase8_outputs(phase2_dir, phase8_dir)
    tile_index = _write_tile_index(tmp_path / "phase13_tile_index.csv")
    monkeypatch.setattr(
        phase28,
        "_train_maskable_ppo_model",
        lambda train_env, seed, total_timesteps: FakeModel(),
    )

    protocol = phase28.run_phase28_representation_control_evaluation(
        phase2_output_dir=phase2_dir,
        phase8_output_dir=phase8_dir,
        tile_index_csv=tile_index,
        variants=("B1", "D2"),
        total_timesteps=8,
        eval_max_steps=2,
        seeds=(0,),
        max_eval_tiles=1,
    )

    assert protocol["training_completed"] is True
    assert protocol["summary_count"] == 6
    assert {row["variant_id"] for row in protocol["summaries"]} == {"B1", "D2"}
    assert protocol["phase28_diagnostic_status"] in {
        "representation_signal_control_limited",
        "representation_signal_not_distinguishable",
        "compression_matches_raw",
        "insufficient",
    }


def test_phase28_cli_analyze_only_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase28_representation_controls"
        / "run_phase28_representation_controls.py"
    )
    spec = importlib.util.spec_from_file_location("phase28_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    summary_csv = _write_summary_csv(
        tmp_path / "summary.csv",
        _phase28_summary_rows("supported"),
    )

    exit_code = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 28 diagnostic status: representation_signal_supported" in stdout
    assert "phase28_representation_control_comparison.json" in stdout


def test_phase28_cli_run_and_analyze_requires_explicit_training_settings(
    tmp_path,
    capsys,
):
    runner_path = (
        ROOT
        / "experiments"
        / "phase28_representation_controls"
        / "run_phase28_representation_controls.py"
    )
    spec = importlib.util.spec_from_file_location("phase28_runner_validation", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--mode",
            "run-and-analyze",
            "--phase2-output-dir",
            str(tmp_path / "phase2"),
            "--phase8-output-dir",
            str(tmp_path / "phase8"),
            "--tile-index-csv",
            str(tmp_path / "phase13_tile_index.csv"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "run-and-analyze requires" in stderr
    assert "--total-timesteps" in stderr
    assert "--eval-max-steps" in stderr
    assert "--seeds" in stderr
```

- [ ] **Step 2: Run the new Phase 28 tests and confirm RED**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py -q --basetemp=.pytest_tmp_phase28_red -p no:cacheprovider
```

Expected: fail because `paper11_geofm.phase28_representation_controls` and the runner do not exist yet.

- [ ] **Step 3: Commit the RED tests**

Run:

```powershell
git add tests/test_phase28_representation_controls.py
git commit -m "test: add Phase 28 representation-control expectations"
```

Expected: one commit containing only the new failing tests.

## Task 2: Implement Phase 28 Analysis and Artifact Writer

**Files:**
- Create: `src/paper11_geofm/phase28_representation_controls.py`
- Test: `tests/test_phase28_representation_controls.py`

- [ ] **Step 1: Add constants, fieldnames, CSV readers, float helpers, and variant normalization**

Create `src/paper11_geofm/phase28_representation_controls.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import statistics
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .drl_inputs import load_variant_input
from .padded_heldout_policy import (
    SUMMARY_FIELDNAMES,
    Phase25PaddedTileEnv,
    _dependency_metadata,
    _evaluate_baseline_policy,
    _evaluate_trained_policy,
    _round_float,
    _select_train_eval_tiles,
    _store_trace,
)
from .tiled_inputs import load_tiled_variant_input


PHASE28_CLAIM_BOUNDARY = (
    "Phase 28 is a representation-control diagnostic over B0/B1/D2/D3/D4 "
    "base-reward padded held-out Bishan policy runs; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not support submission-level planning-performance "
    "claims."
)

PHASE28_REMAINING_EVIDENCE_GAPS = [
    "stable_B1_vs_B0_evidence_before_positive_claim",
    "suitability_reward_validation_before_B2_B3",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
]

PHASE28_DEFAULT_VARIANTS = ("B0", "B1", "D2", "D3", "D4P8", "D4P16")
PHASE28_ALLOWED_VARIANTS = frozenset(PHASE28_DEFAULT_VARIANTS)
PHASE28_PRIMARY_COMPARATORS = ("B0", "D2", "D3", "D4P8", "D4P16")
TILE_SEED_DELTA_FIELDNAMES = [
    "comparator_variant_id",
    "eval_tile_id",
    "seed",
    "b1_reward",
    "comparator_reward",
    "b1_minus_comparator_reward",
    "b1_improves_comparator",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]
```

Add these helpers:

```python
def _normalize_phase28_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip() for part in variants.split(",")]
    else:
        values = [str(item).strip() for item in variants]
    normalized = [item.upper() for item in values if item]
    if not normalized:
        raise ValueError("At least one Phase 28 variant must be requested")
    unsupported = [
        variant for variant in normalized if variant not in PHASE28_ALLOWED_VARIANTS
    ]
    if unsupported:
        raise ValueError(f"unsupported Phase 28 variants: {unsupported}")
    if "B1" not in normalized:
        raise ValueError("Phase 28 analysis requires B1")
    comparators = [variant for variant in normalized if variant != "B1"]
    if not comparators:
        raise ValueError("Phase 28 analysis requires at least one comparator")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 28 variants must be unique")
    return normalized


def _normalize_seeds(seeds: Sequence[int | str] | str | int | None) -> list[int]:
    if seeds is None:
        values: list[int | str] = [0, 1, 2]
    elif isinstance(seeds, str):
        values = [part.strip() for part in seeds.split(",")]
    elif isinstance(seeds, int):
        values = [seeds]
    else:
        values = list(seeds)
    normalized = [int(value) for value in values if str(value).strip() != ""]
    if not normalized:
        raise ValueError("At least one Phase 28 seed must be requested")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 28 seeds must be unique")
    return normalized


def _variant_source_dir(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    variant_id: str,
) -> Path:
    normalized = str(variant_id).upper()
    if normalized in {"B0", "B1"}:
        return Path(phase2_output_dir)
    return Path(phase8_output_dir)


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _int_value(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing integer field: {field}")
    return int(value)


def _read_summary_rows(path: Path | str) -> list[dict[str, object]]:
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Phase 28 summary CSV: {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
```

- [ ] **Step 2: Implement `build_phase28_representation_control_analysis`**

Add:

```python
def build_phase28_representation_control_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    *,
    compression_match_tolerance: float = 1e-9,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(summary_rows_or_csv, (str, Path)):
        rows = _read_summary_rows(summary_rows_or_csv)
        source_summary_csv = str(Path(summary_rows_or_csv))
    else:
        rows = [dict(row) for row in summary_rows_or_csv]
        source_summary_csv = None
    if not rows:
        raise ValueError("Phase 28 analysis requires at least one summary row")

    learned_rows = [row for row in rows if str(row.get("row_type", "")) == "trained_policy"]
    variants = _unique_strings(learned_rows, "variant_id")
    if "B1" not in variants:
        status = "insufficient"
    else:
        status = ""
    eval_tile_ids = _unique_strings(learned_rows, "eval_tile_id")
    seeds = _unique_ints(learned_rows, "seed")
    train_timesteps = _first_int(learned_rows, "train_timesteps")
    eval_max_steps = _first_int(learned_rows, "eval_max_steps")
    coverage_issues = _coverage_issues(learned_rows, variants, eval_tile_ids, seeds)
    tile_seed_delta_rows = _tile_seed_delta_rows(learned_rows, variants, coverage_issues)
    learned_policy = _learned_policy_summary(
        learned_rows,
        tile_seed_delta_rows,
        variants,
    )
    if status != "insufficient":
        status = _diagnostic_status(
            learned_policy,
            coverage_issues,
            compression_match_tolerance,
        )
    payload: dict[str, object] = {
        "phase": "phase28_representation_control_analysis",
        "source_summary_csv": source_summary_csv,
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "main_summary_rows": _main_summary_rows(rows),
        "tile_seed_delta_rows": tile_seed_delta_rows,
        "learned_policy": learned_policy,
        "baselines": {
            policy_id: _policy_summary(rows, policy_id, variants)
            for policy_id in ("first_valid", "seeded_random")
        },
        "coverage_issues": coverage_issues,
        "phase28_diagnostic_status": status,
        "compression_match_tolerance": float(compression_match_tolerance),
        "remaining_evidence_gaps": list(PHASE28_REMAINING_EVIDENCE_GAPS),
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload
```

Add the supporting analysis helpers in the same file:

```python
def _unique_strings(rows: Sequence[Mapping[str, object]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _unique_ints(rows: Sequence[Mapping[str, object]], field: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        number = int(value)
        if number not in seen:
            values.append(number)
            seen.add(number)
    return values


def _first_int(rows: Sequence[Mapping[str, object]], field: str) -> int | None:
    for row in rows:
        value = row.get(field)
        if value is not None and str(value).strip() != "":
            return int(value)
    return None


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(values) / len(values))


def _coverage_issues(
    learned_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    expected = {
        (str(tile_id), int(seed), str(variant_id))
        for tile_id in eval_tile_ids
        for seed in seeds
        for variant_id in variants
    }
    observed: set[tuple[str, int, str]] = set()
    duplicates: set[tuple[str, int, str]] = set()
    for row in learned_rows:
        key = (
            str(row.get("eval_tile_id", "")),
            _int_value(row, "seed"),
            str(row.get("variant_id", "")),
        )
        if key in observed:
            duplicates.add(key)
        observed.add(key)
    return {
        "missing_variant_rows": _variant_key_dicts(expected - observed),
        "unexpected_variant_rows": _variant_key_dicts(observed - expected),
        "duplicate_variant_rows": _variant_key_dicts(duplicates),
    }


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": tile_id, "seed": seed, "variant_id": variant_id}
        for tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]
```

Add delta and status helpers:

```python
def _tile_seed_delta_rows(
    learned_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    coverage_issues: Mapping[str, object],
) -> list[dict[str, object]]:
    duplicate_keys = {
        (
            str(item.get("eval_tile_id", "")),
            int(item.get("seed", 0)),
            str(item.get("variant_id", "")),
        )
        for item in coverage_issues.get("duplicate_variant_rows", [])
        if isinstance(item, Mapping)
    }
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in learned_rows:
        key = (
            str(row.get("eval_tile_id", "")),
            _int_value(row, "seed"),
            str(row.get("variant_id", "")),
        )
        if key in duplicate_keys:
            continue
        pair = (key[0], key[1])
        by_pair.setdefault(pair, {})[key[2]] = row

    comparator_ids = [variant for variant in variants if variant != "B1"]
    delta_rows: list[dict[str, object]] = []
    for pair in sorted(by_pair, key=lambda item: (item[0], item[1])):
        variant_rows = by_pair[pair]
        b1_row = variant_rows.get("B1")
        if b1_row is None:
            continue
        b1_reward = _float_value(b1_row, "total_contract_reward")
        for comparator_id in comparator_ids:
            comparator_row = variant_rows.get(comparator_id)
            if comparator_row is None:
                continue
            comparator_reward = _float_value(comparator_row, "total_contract_reward")
            delta = _round_float(b1_reward - comparator_reward)
            delta_rows.append(
                {
                    "comparator_variant_id": comparator_id,
                    "eval_tile_id": pair[0],
                    "seed": pair[1],
                    "b1_reward": _round_float(b1_reward),
                    "comparator_reward": _round_float(comparator_reward),
                    "b1_minus_comparator_reward": delta,
                    "b1_improves_comparator": delta > 0.0,
                    "train_timesteps": _int_value(b1_row, "train_timesteps"),
                    "eval_max_steps": _int_value(b1_row, "eval_max_steps"),
                    "claim_boundary": PHASE28_CLAIM_BOUNDARY,
                }
            )
    return delta_rows


def _learned_policy_summary(
    learned_rows: Sequence[Mapping[str, object]],
    tile_seed_delta_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
) -> dict[str, object]:
    mean_reward_by_variant = _mean_reward_by_variant(learned_rows, variants)
    comparator_deltas: dict[str, dict[str, object]] = {}
    for comparator_id in [variant for variant in variants if variant != "B1"]:
        rows = [
            row
            for row in tile_seed_delta_rows
            if str(row.get("comparator_variant_id", "")) == comparator_id
        ]
        deltas = [_float_value(row, "b1_minus_comparator_reward") for row in rows]
        positive_count = sum(1 for row in rows if bool(row.get("b1_improves_comparator")))
        total_count = len(rows)
        comparator_deltas[f"B1_minus_{comparator_id}"] = {
            "mean_reward_delta": _mean_or_none(deltas),
            "std_reward_delta": _round_float(statistics.pstdev(deltas))
            if len(deltas) > 1
            else (0.0 if deltas else None),
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": total_count,
            "positive_fraction": _round_float(positive_count / total_count)
            if total_count
            else None,
        }
    best_variant = None
    if mean_reward_by_variant:
        best_variant = max(
            mean_reward_by_variant,
            key=lambda variant_id: mean_reward_by_variant[variant_id],
        )
    return {
        "mean_reward_by_variant": mean_reward_by_variant,
        "comparator_deltas": comparator_deltas,
        "best_variant_by_mean_reward": best_variant,
        "b1_beats_all_requested_comparators": all(
            value.get("mean_reward_delta") is not None
            and float(value["mean_reward_delta"]) > 0.0
            for value in comparator_deltas.values()
        ),
    }


def _mean_reward_by_variant(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for variant_id in variants:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == variant_id
        ]
        if values:
            result[variant_id] = _round_float(sum(values) / len(values))
    return result


def _diagnostic_status(
    learned_policy: Mapping[str, object],
    coverage_issues: Mapping[str, object],
    compression_match_tolerance: float,
) -> str:
    if any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_variant_rows",
            "unexpected_variant_rows",
            "duplicate_variant_rows",
        )
    ):
        return "insufficient"
    comparator_deltas = learned_policy.get("comparator_deltas")
    mean_reward_by_variant = learned_policy.get("mean_reward_by_variant")
    if not isinstance(comparator_deltas, Mapping) or not isinstance(
        mean_reward_by_variant,
        Mapping,
    ):
        return "insufficient"
    required = ["B1_minus_D2", "B1_minus_D3"]
    if not all(key in comparator_deltas for key in required):
        return "insufficient"
    b1_vs_b0 = comparator_deltas.get("B1_minus_B0")
    b1_vs_d2 = comparator_deltas.get("B1_minus_D2")
    b1_vs_d3 = comparator_deltas.get("B1_minus_D3")
    if not all(isinstance(item, Mapping) for item in (b1_vs_d2, b1_vs_d3)):
        return "insufficient"

    b1_mean = mean_reward_by_variant.get("B1")
    if b1_mean is None:
        return "insufficient"
    d4_matches = any(
        variant in mean_reward_by_variant
        and float(mean_reward_by_variant[variant])
        >= float(b1_mean) - float(compression_match_tolerance)
        for variant in ("D4P8", "D4P16")
    )

    def _positive_mean(delta: object) -> bool:
        return (
            isinstance(delta, Mapping)
            and delta.get("mean_reward_delta") is not None
            and float(delta["mean_reward_delta"]) > 0.0
        )

    def _stable_fraction(delta: object) -> bool:
        return (
            isinstance(delta, Mapping)
            and delta.get("positive_fraction") is not None
            and float(delta["positive_fraction"]) >= 0.6
        )

    b1_beats_b0 = _positive_mean(b1_vs_b0) if b1_vs_b0 is not None else False
    b1_beats_d2 = _positive_mean(b1_vs_d2)
    b1_beats_d3 = _positive_mean(b1_vs_d3)
    if (
        b1_beats_b0
        and b1_beats_d2
        and b1_beats_d3
        and _stable_fraction(b1_vs_d2)
        and _stable_fraction(b1_vs_d3)
    ):
        return "representation_signal_supported"
    if d4_matches:
        return "compression_matches_raw"
    if not b1_beats_d2 and not b1_beats_d3:
        return "representation_signal_not_distinguishable"
    return "representation_signal_control_limited"
```

Add summary and baseline helpers:

```python
def _main_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("row_type", "")),
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
        )
        groups.setdefault(key, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (row_type, variant_id, eval_tile_id), group_rows in sorted(groups.items()):
        values = [_float_value(row, "total_contract_reward") for row in group_rows]
        seeds = {_int_value(row, "seed") for row in group_rows}
        summary_rows.append(
            {
                "row_type": row_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed_count": len(seeds),
                "mean_total_contract_reward": _round_float(sum(values) / len(values)),
                "std_total_contract_reward": _round_float(statistics.pstdev(values))
                if len(values) > 1
                else 0.0,
                "min_total_contract_reward": _round_float(min(values)),
                "max_total_contract_reward": _round_float(max(values)),
                "train_timesteps": _int_value(group_rows[0], "train_timesteps"),
                "eval_max_steps": _int_value(group_rows[0], "eval_max_steps"),
                "claim_boundary": PHASE28_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _policy_summary(
    rows: Sequence[Mapping[str, object]],
    row_type: str,
    variants: Sequence[str],
) -> dict[str, object]:
    policy_rows = [row for row in rows if str(row.get("row_type", "")) == row_type]
    return {"mean_reward_by_variant": _mean_reward_by_variant(policy_rows, variants)}
```

- [ ] **Step 3: Implement writer and Markdown readiness note**

Add:

```python
def write_phase28_representation_control_artifacts(
    protocol_or_analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase28_representation_control_summary.csv"
    traces_path = output_path / "phase28_representation_control_traces.json"
    comparison_path = output_path / "phase28_representation_control_comparison.json"
    delta_path = output_path / "phase28_tile_seed_delta_table.csv"
    readiness_path = output_path / "phase28_control_readiness.md"

    summaries = protocol_or_analysis.get("summaries")
    if summaries is None:
        summaries = protocol_or_analysis.get("summary_rows")
    if summaries is None:
        summaries = []
    if not isinstance(summaries, list):
        raise ValueError("Phase 28 protocol summaries must be a list")
    _write_csv_mapping_rows(summary_path, SUMMARY_FIELDNAMES, summaries)

    traces_path.write_text(
        json.dumps(dict(protocol_or_analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(dict(protocol_or_analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    delta_rows = protocol_or_analysis.get("tile_seed_delta_rows")
    if not isinstance(delta_rows, list):
        raise ValueError("Phase 28 analysis is missing tile_seed_delta_rows")
    _write_csv_mapping_rows(delta_path, TILE_SEED_DELTA_FIELDNAMES, delta_rows)
    readiness_path.write_text(
        _phase28_control_readiness_markdown(protocol_or_analysis),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "comparison_json": comparison_path,
        "tile_seed_delta_csv": delta_path,
        "control_readiness_md": readiness_path,
    }


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 28 CSV rows must be a list for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 28 CSV row must be an object for {path.name}")
            selected = row.get("selected_block_ids")
            patched = dict(row)
            if isinstance(selected, list):
                patched["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow({field: patched.get(field, "") for field in fieldnames})


def _phase28_control_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    deltas = learned.get("comparator_deltas")
    if not isinstance(deltas, Mapping):
        deltas = {}
    lines = [
        "# Phase 28 Representation-Control Readiness",
        "",
        f"Status: {analysis.get('phase28_diagnostic_status', '')}",
        "",
        "Comparator deltas:",
    ]
    for key in sorted(deltas):
        value = deltas[key]
        if isinstance(value, Mapping):
            lines.append(
                "- "
                + key
                + ": mean="
                + str(value.get("mean_reward_delta"))
                + ", positive="
                + str(value.get("positive_tile_seed_count"))
                + " / "
                + str(value.get("total_tile_seed_count"))
            )
    lines.extend(
        [
            "",
            "Safe wording:",
            (
                "Phase 28 evaluates whether the current B1 representation is "
                "distinguishable from random, shuffled, and PCA-compressed "
                "controls under the Bishan base-reward held-out protocol."
            ),
            "",
            "Unsafe wording:",
            "GeoFM improves planning decisions.",
            "",
            str(analysis.get("claim_boundary", PHASE28_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run targeted tests and confirm analysis/writer tests pass**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py::test_phase28_analysis_computes_b1_control_deltas_and_supported_status tests\test_phase28_representation_controls.py::test_phase28_diagnostic_status_rules tests\test_phase28_representation_controls.py::test_phase28_reports_insufficient_for_missing_comparator_rows tests\test_phase28_representation_controls.py::test_phase28_writer_outputs_summary_trace_comparison_delta_and_markdown -q --basetemp=.pytest_tmp_phase28_analysis -p no:cacheprovider
```

Expected: these analysis and writer tests pass.

- [ ] **Step 5: Commit analysis and writer**

Run:

```powershell
git add src/paper11_geofm/phase28_representation_controls.py tests/test_phase28_representation_controls.py
git commit -m "feat: add Phase 28 control analysis"
```

Expected: one commit with the Phase 28 analysis core and writer.

## Task 3: Implement Contract, Source Routing, and Evaluation Runner

**Files:**
- Modify: `src/paper11_geofm/phase28_representation_controls.py`
- Test: `tests/test_phase28_representation_controls.py`

- [ ] **Step 1: Add contract builder and source validation**

Add:

```python
def build_phase28_representation_control_contract(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE28_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    normalized_variants = _normalize_phase28_variants(variants)
    normalized_seeds = _normalize_seeds(seeds)
    _validate_variant_sources(
        phase2_output_dir,
        phase8_output_dir,
        normalized_variants,
    )
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    eval_ids = list(selected["eval_tile_ids"])
    seed_ranks = {
        str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
    }
    eval_tile_ranks = {
        str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
    }
    variant_source_dirs = {
        variant_id: str(
            _variant_source_dir(phase2_output_dir, phase8_output_dir, variant_id)
        )
        for variant_id in normalized_variants
    }
    return {
        "phase": "phase28_representation_control_evaluation",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variant_source_dirs": variant_source_dirs,
        "train_tile_id": str(selected["train_tile_id"]),
        "train_tile_ids": [str(selected["train_tile_id"])],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": eval_tile_ranks,
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
        "remaining_evidence_gaps": list(PHASE28_REMAINING_EVIDENCE_GAPS),
    }


def _validate_variant_sources(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    variants: Sequence[str],
) -> None:
    for variant_id in variants:
        source_dir = _variant_source_dir(phase2_output_dir, phase8_output_dir, variant_id)
        loaded = load_variant_input(source_dir, variant_id)
        if loaded.reward_mode != "base_planning_reward":
            raise ValueError(
                "Phase 28 only supports base_planning_reward variants; "
                f"{variant_id} uses {loaded.reward_mode}"
            )
```

- [ ] **Step 2: Add Phase 28 tiled loader and training hook**

Add:

```python
def _load_phase28_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_sources = contract.get("variant_source_dirs")
    if not isinstance(variant_sources, Mapping):
        raise ValueError("Phase 28 contract is missing variant_source_dirs")
    source_dir = variant_sources.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 28 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _train_maskable_ppo_model(
    train_env: Phase25PaddedTileEnv,
    seed: int,
    total_timesteps: int,
):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 28 representation-control evaluation requires "
            "stable-baselines3 and sb3-contrib"
        ) from exc
    if not is_masking_supported(train_env):
        raise ValueError("Phase 28 train env does not expose action_masks")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
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
    return model
```

- [ ] **Step 3: Add run function**

Add:

```python
def run_phase28_representation_control_evaluation(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    tile_index_csv: Path | str,
    variants: Sequence[str] | str = PHASE28_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
    compression_match_tolerance: float = 1e-9,
) -> dict[str, object]:
    contract = build_phase28_representation_control_contract(
        phase2_output_dir=phase2_output_dir,
        phase8_output_dir=phase8_output_dir,
        tile_index_csv=tile_index_csv,
        variants=variants,
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
        total_timesteps=total_timesteps,
        eval_max_steps=eval_max_steps,
        seeds=seeds,
    )
    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }
    for variant_id in contract["variants"]:
        for seed in contract["seeds"]:
            train_tiled = _load_phase28_tiled_variant_input(
                contract,
                str(contract["train_tile_id"]),
                str(variant_id),
            )
            train_env = Phase25PaddedTileEnv(
                train_tiled,
                max_blocks=int(contract["max_blocks"]),
                max_steps=int(contract["total_timesteps"]),
            )
            train_env.reset(seed=int(seed))
            model = _train_maskable_ppo_model(
                train_env,
                seed=int(seed),
                total_timesteps=int(contract["total_timesteps"]),
            )
            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = _load_phase28_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                eval_tile_rank = int(contract["eval_tile_ranks"][str(eval_tile_id)])
                seed_rank = int(contract["seed_ranks"][str(int(seed))])
                train_n_blocks = int(
                    contract["selected_tile_block_counts"][str(contract["train_tile_id"])]
                )
                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=train_n_blocks,
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=eval_tile_rank,
                    phase25_seed_rank=seed_rank,
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                trained_summary["claim_boundary"] = PHASE28_CLAIM_BOUNDARY
                summaries.append(trained_summary)
                _store_trace(
                    traces,
                    "trained_policy",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    trained_steps,
                )
                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=train_n_blocks,
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=eval_tile_rank,
                        phase25_seed_rank=seed_rank,
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    baseline_summary["claim_boundary"] = PHASE28_CLAIM_BOUNDARY
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )
    analysis = build_phase28_representation_control_analysis(
        summaries,
        compression_match_tolerance=compression_match_tolerance,
        metadata={
            "phase2_output_dir": contract["phase2_output_dir"],
            "phase8_output_dir": contract["phase8_output_dir"],
            "tile_index_csv": contract["tile_index_csv"],
        },
    )
    return {
        **contract,
        **analysis,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in summaries
        ),
        "summary_count": len(summaries),
        "summaries": summaries,
        "traces": traces,
        "dependencies": _dependency_metadata(),
        "claim_boundary": PHASE28_CLAIM_BOUNDARY,
    }
```

- [ ] **Step 4: Run contract and fake-run tests**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py::test_phase28_contract_routes_b_and_d_variant_sources tests\test_phase28_representation_controls.py::test_phase28_contract_rejects_unsupported_and_missing_b1 tests\test_phase28_representation_controls.py::test_phase28_run_uses_fake_training_model_for_all_variants -q --basetemp=.pytest_tmp_phase28_runner -p no:cacheprovider
```

Expected: selected tests pass.

- [ ] **Step 5: Run all Phase 28 tests**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py -q --basetemp=.pytest_tmp_phase28_module -p no:cacheprovider
```

Expected: all Phase 28 tests except CLI tests pass if the runner file has not been created yet; CLI tests fail with missing runner.

- [ ] **Step 6: Commit contract and evaluation runner core**

Run:

```powershell
git add src/paper11_geofm/phase28_representation_controls.py tests/test_phase28_representation_controls.py
git commit -m "feat: add Phase 28 control evaluator"
```

Expected: one commit with contract, source routing, and fake-training-tested evaluation logic.

## Task 4: Implement Phase 28 CLI

**Files:**
- Create: `experiments/phase28_representation_controls/run_phase28_representation_controls.py`
- Test: `tests/test_phase28_representation_controls.py`

- [ ] **Step 1: Create the runner**

Create `experiments/phase28_representation_controls/run_phase28_representation_controls.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase28_representation_controls import (
    build_phase28_representation_control_analysis,
    run_phase28_representation_control_evaluation,
    write_phase28_representation_control_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    provided_args = set(argv or sys.argv[1:])
    try:
        if args.mode == "run-and-analyze":
            _validate_run_and_analyze_args(args, provided_args)
            protocol = run_phase28_representation_control_evaluation(
                phase2_output_dir=args.phase2_output_dir,
                phase8_output_dir=args.phase8_output_dir,
                tile_index_csv=args.tile_index_csv,
                variants=tuple(
                    part.strip() for part in args.variants.split(",") if part.strip()
                ),
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                total_timesteps=args.total_timesteps,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
                compression_match_tolerance=args.compression_match_tolerance,
            )
        else:
            if args.existing_summary_csv is None:
                raise ValueError("analyze-only requires --existing-summary-csv")
            protocol = build_phase28_representation_control_analysis(
                args.existing_summary_csv,
                compression_match_tolerance=args.compression_match_tolerance,
            )
        paths = write_phase28_representation_control_artifacts(
            protocol,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    learned = protocol.get("learned_policy", {})
    print(f"Mode: {args.mode}")
    print(f"Phase 28 diagnostic status: {protocol['phase28_diagnostic_status']}")
    if isinstance(learned, dict):
        comparator_deltas = learned.get("comparator_deltas", {})
        if isinstance(comparator_deltas, dict):
            for key in sorted(comparator_deltas):
                value = comparator_deltas[key]
                if isinstance(value, dict):
                    print(f"{key} mean reward delta: {value.get('mean_reward_delta')}")
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Trace JSON: {paths['traces_json']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Tile-seed delta CSV: {paths['tile_seed_delta_csv']}")
    print(f"Control readiness Markdown: {paths['control_readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze the Paper11 Phase 28 B1 representation-control "
            "evaluation package."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run-and-analyze", "analyze-only"),
        default="analyze-only",
    )
    parser.add_argument("--phase2-output-dir", type=Path, default=None)
    parser.add_argument("--phase8-output-dir", type=Path, default=None)
    parser.add_argument("--tile-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-summary-csv", type=Path, default=None)
    parser.add_argument("--variants", default="B0,B1,D2,D3,D4P8,D4P16")
    parser.add_argument("--train-tile-id", default=None)
    parser.add_argument("--eval-tile-ids", default=None)
    parser.add_argument("--max-eval-tiles", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--eval-max-steps", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--compression-match-tolerance", type=float, default=1e-9)
    return parser


def _validate_run_and_analyze_args(
    args: argparse.Namespace,
    provided_args: set[str],
) -> None:
    missing = []
    if args.phase2_output_dir is None:
        missing.append("--phase2-output-dir")
    if args.phase8_output_dir is None:
        missing.append("--phase8-output-dir")
    if args.tile_index_csv is None:
        missing.append("--tile-index-csv")
    for flag in (
        "--variants",
        "--total-timesteps",
        "--eval-max-steps",
        "--seeds",
    ):
        if not _was_provided(provided_args, flag):
            missing.append(flag)
    if not _was_provided(provided_args, "--max-eval-tiles") and not _was_provided(
        provided_args,
        "--eval-tile-ids",
    ):
        missing.append("--max-eval-tiles or --eval-tile-ids")
    if missing:
        raise ValueError("run-and-analyze requires " + ", ".join(missing))


def _was_provided(provided_args: set[str], flag: str) -> bool:
    return flag in provided_args or any(
        item.startswith(f"{flag}=") for item in provided_args
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run CLI tests**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py::test_phase28_cli_analyze_only_writes_outputs tests\test_phase28_representation_controls.py::test_phase28_cli_run_and_analyze_requires_explicit_training_settings -q --basetemp=.pytest_tmp_phase28_cli -p no:cacheprovider
```

Expected: CLI tests pass.

- [ ] **Step 3: Run all Phase 28 tests**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py -q --basetemp=.pytest_tmp_phase28_all -p no:cacheprovider
```

Expected: all Phase 28 tests pass.

- [ ] **Step 4: Commit CLI**

Run:

```powershell
git add experiments/phase28_representation_controls/run_phase28_representation_controls.py tests/test_phase28_representation_controls.py
git commit -m "feat: add Phase 28 representation-control CLI"
```

Expected: one commit with the Phase 28 runner and passing CLI tests.

## Task 5: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/phase26_results/02_next_experiment_matrix.md`

- [ ] **Step 1: Update README**

Add a Phase 28 section after the Phase 27 content:

```markdown
### Phase 28: Representation-Control Evaluation

Phase 28 evaluates whether B1 differs from random, shuffled, and
PCA-compressed representation controls under the same padded held-out
base-reward policy protocol. It uses B0/B1 from Phase 2 outputs and D2/D3/D4
from Phase 8 outputs.

Analyze-only example:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode analyze-only --existing-summary-csv <phase28_summary.csv> --output-dir experiments\phase28_representation_controls\outputs\analysis
```

Run-and-analyze example:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir <phase2_outputs> --phase8-output-dir <phase8_outputs> --tile-index-csv <phase13_tile_index.csv> --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 1024 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\main
```

Claim boundary: Phase 28 is diagnostic only; it does not enable suitability
reward, test B2/B3, test cross-region transfer, or support final
submission-level planning-performance claims.
```

- [ ] **Step 2: Update reproduction guide**

Add a section that states:

```markdown
## Phase 28 Representation-Control Evaluation

Inputs:
- Phase 2 B0/B1 feature outputs;
- Phase 8 D2/D3/D4P8/D4P16 control feature outputs;
- Phase 13/14 tile index CSV.

Outputs:
- `phase28_representation_control_summary.csv`;
- `phase28_representation_control_traces.json`;
- `phase28_representation_control_comparison.json`;
- `phase28_tile_seed_delta_table.csv`;
- `phase28_control_readiness.md`.

Use `analyze-only` for existing summaries and `run-and-analyze` for real
training. `run-and-analyze` requires explicit `--variants`,
`--total-timesteps`, `--eval-max-steps`, `--seeds`, and either
`--max-eval-tiles` or `--eval-tile-ids`.
```

- [ ] **Step 3: Update file manifest**

Add rows for:

```text
docs/superpowers/specs/2026-06-18-phase28-representation-control-evaluation-design.md
docs/superpowers/plans/2026-06-18-phase28-representation-control-evaluation.md
src/paper11_geofm/phase28_representation_controls.py
experiments/phase28_representation_controls/run_phase28_representation_controls.py
tests/test_phase28_representation_controls.py
```

- [ ] **Step 4: Update Phase 26 next experiment matrix**

In `paper/phase26_results/02_next_experiment_matrix.md`, update the recommended next implementation target so Phase 28 is listed as the active representation-control evaluation step after Phase 27. Keep the B1-over-B0 claim boundary negative until Phase 28 produces stable evidence.

- [ ] **Step 5: Commit documentation**

Run:

```powershell
git add README.md reproducibility/REPRODUCTION_GUIDE.md reproducibility/FILE_MANIFEST.tsv paper/phase26_results/02_next_experiment_matrix.md
git commit -m "docs: document Phase 28 control evaluation"
```

Expected: one documentation commit.

## Task 6: Verification and Final Commit Check

**Files:**
- All files touched by Tasks 1-5.

- [ ] **Step 1: Run Phase 28 tests**

Run:

```powershell
python -m pytest tests\test_phase28_representation_controls.py -q --basetemp=.pytest_tmp_phase28_final -p no:cacheprovider
```

Expected: all Phase 28 tests pass.

- [ ] **Step 2: Run Phase 8 regression tests**

Run:

```powershell
python -m pytest tests\test_phase8_ablation_controls.py -q --basetemp=.pytest_tmp_phase8_phase28_regression -p no:cacheprovider
```

Expected: Phase 8 control feature generation tests pass.

- [ ] **Step 3: Run Phase 25 regression tests**

Run:

```powershell
python -m pytest tests\test_phase25_padded_heldout_policy.py -q --basetemp=.pytest_tmp_phase25_phase28_regression -p no:cacheprovider
```

Expected: Phase 25 B0/B1 padded held-out behavior remains unchanged.

- [ ] **Step 4: Run Phase 26 regression tests**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q --basetemp=.pytest_tmp_phase26_phase28_regression -p no:cacheprovider
```

Expected: Phase 26 analysis tests pass.

- [ ] **Step 5: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected: smoke check passes.

- [ ] **Step 6: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 7: Inspect git status**

Run:

```powershell
git status --short --branch
git log -5 --oneline
```

Expected: branch contains the Phase 28 spec, plan, tests, implementation, runner, and docs commits. No unstaged files remain.

## Self-Review

- Spec coverage: the plan covers Phase 28 inputs, variant source routing, base-reward validation, D2/D3/D4 controls, analysis metrics, diagnostic status rules, output artifacts, CLI modes, tests, documentation, and claim boundaries.
- Placeholder scan: no TBD sections, TODO markers, incomplete tasks, or future-fill instructions remain.
- Type consistency: function names, artifact names, status strings, and CLI flags match the Phase 28 design spec.
- Scope check: the plan stays within one implementation package and does not enable suitability reward, B2/B3, transfer, or final paper claims.
