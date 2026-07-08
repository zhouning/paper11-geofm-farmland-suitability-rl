# Phase 59 Matched-Dimension Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a matched-dimension control audit that tests whether D4P8/D4P16 outperform 8- and 16-dimensional random or shuffled controls under the same Bishan base-reward held-out policy protocol.

**Architecture:** Add a focused Phase 59 module with three responsibilities: matched-control feature-table construction, matched held-out policy evaluation, and read-only matched-control analysis/artifact writing. Keep Phase 28 constants unchanged; reuse Phase 28/Padded-policy helpers only through Phase 59 routing code so the new variants do not leak into older phases.

**Tech Stack:** Python standard library, NumPy, sb3-contrib MaskablePPO through existing Phase 28 helpers, pytest, CSV/JSON/Markdown artifact writers.

---

## File Structure

- Create `src/paper11_geofm/phase59_matched_dimension_controls.py`.
  This file owns Phase 59 constants, CSV loading, matched-control table construction, run contract creation, policy evaluation orchestration, matched-delta analysis, status rules, and artifact writers.
- Create `experiments/phase59_matched_dimension_controls/run_phase59_matched_dimension_controls.py`.
  This runner exposes `build-controls`, `run-and-analyze`, and `analyze-only` modes.
- Create `tests/test_phase59_matched_dimension_controls.py`.
  This test file covers deterministic feature construction, manifest compatibility, coverage checks, status rules, writers, and CLI behavior on synthetic data.
- Create `paper/phase28_results/25_phase59_matched_dimension_controls.md` after the real run.
  This document records the real Phase 59 evidence without changing the formal manuscript.
- Modify `paper/phase28_results/README.md` and `docs/superpowers/phase33_current_progress_handoff.md` after the real run.
  These records should point to Phase 59 outputs and preserve the claim boundary.

---

### Task 1: Define Matched-Control Feature Construction Tests

**Files:**
- Create: `tests/test_phase59_matched_dimension_controls.py`

- [ ] **Step 1: Add fixture helpers and the first failing feature-construction test**

Add this content at the top of `tests/test_phase59_matched_dimension_controls.py`:

```python
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(block_id, explicit_00, values, prefix="embedding_pca"):
    row = {
        "block_id": block_id,
        "explicit_feature_00": explicit_00,
    }
    for index, value in enumerate(values):
        row[f"{prefix}_{index:02d}"] = value
    return row


def _b0_rows():
    return [
        {"block_id": "b1", "explicit_feature_00": 1.0},
        {"block_id": "b2", "explicit_feature_00": 2.0},
        {"block_id": "b3", "explicit_feature_00": 3.0},
        {"block_id": "b4", "explicit_feature_00": 4.0},
    ]


def _d4p8_rows():
    return [
        _row("b1", 1.0, [0.0, 1.0]),
        _row("b2", 2.0, [2.0, 3.0]),
        _row("b3", 3.0, [4.0, 5.0]),
        _row("b4", 4.0, [6.0, 7.0]),
    ]


def _d4p16_rows():
    return [
        _row("b1", 1.0, [0.0, 10.0, 20.0]),
        _row("b2", 2.0, [1.0, 11.0, 21.0]),
        _row("b3", 3.0, [2.0, 12.0, 22.0]),
        _row("b4", 4.0, [3.0, 13.0, 23.0]),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
```

Then add the test:

```python
def test_phase59_builds_deterministic_matched_control_tables():
    from paper11_geofm.phase59_matched_dimension_controls import (
        PHASE59_CLAIM_BOUNDARY,
        build_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )

    assert protocol["phase"] == "phase59_matched_dimension_control_features"
    assert protocol["claim_boundary"] == PHASE59_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["D5R8", "D5S8", "D5R16", "D5S16"]
    assert set(protocol["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    assert protocol["summary"]["D5R8"]["control_dimension"] == 2
    assert protocol["summary"]["D5R16"]["control_dimension"] == 3
    assert protocol["summary"]["D5S8"]["source_variant_id"] == "D4P8"
    assert protocol["summary"]["D5S16"]["source_variant_id"] == "D4P16"

    d5s8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5S8"]
    ]
    d4p8_values = [row["embedding_pca_00"] for row in _d4p8_rows()]
    assert sorted(d5s8_values) == sorted(d4p8_values)
    assert d5s8_values != d4p8_values

    d5r8_values = [
        row["matched_control_00"] for row in protocol["variant_tables"]["D5R8"]
    ]
    assert len(d5r8_values) == 4
    assert not all(math.isclose(value, d4p8_values[0]) for value in d5r8_values)
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m pytest tests\test_phase59_matched_dimension_controls.py::test_phase59_builds_deterministic_matched_control_tables -q
```

Expected result: FAIL with `ModuleNotFoundError: No module named 'paper11_geofm.phase59_matched_dimension_controls'`.

### Task 2: Implement Matched-Control Feature Tables and Writers

**Files:**
- Create: `src/paper11_geofm/phase59_matched_dimension_controls.py`
- Modify: `tests/test_phase59_matched_dimension_controls.py`

- [ ] **Step 1: Add Phase 59 constants and feature-table builders**

Create `src/paper11_geofm/phase59_matched_dimension_controls.py` with these public names and behavior:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from os import PathLike
from pathlib import Path

import numpy as np

from .block_schema import EXPLICIT_FEATURE_COLUMNS


PHASE59_CLAIM_BOUNDARY = (
    "Phase 59 is a matched-dimension control audit over compressed GeoFM "
    "base-reward held-out Bishan policy runs. It tests D4P8/D4P16 against "
    "8- and 16-dimensional random or shuffled controls; it does not enable "
    "suitability reward, does not test B2/B3, does not test cross-region "
    "transfer, and does not validate independent agronomic suitability."
)

PHASE59_COMPRESSED_VARIANTS = ("D4P8", "D4P16")
PHASE59_MATCHED_CONTROL_VARIANTS = ("D5R8", "D5S8", "D5R16", "D5S16")
PHASE59_REQUIRED_VARIANTS = (
    "D4P8",
    "D4P16",
    "D5R8",
    "D5S8",
    "D5R16",
    "D5S16",
)
PHASE59_MATCHED_COMPARISONS = (
    ("D4P8", "D5R8"),
    ("D4P8", "D5S8"),
    ("D4P16", "D5R16"),
    ("D4P16", "D5S16"),
)


def build_phase59_matched_dimension_control_tables(
    b0_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p8_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    d4p16_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    seed: int = 59,
) -> dict[str, object]:
    b0_rows = _load_rows(b0_rows_or_csv, "B0")
    d4p8_rows = _load_rows(d4p8_rows_or_csv, "D4P8")
    d4p16_rows = _load_rows(d4p16_rows_or_csv, "D4P16")
    _require_aligned_block_ids(b0_rows, d4p8_rows, "D4P8")
    _require_aligned_block_ids(b0_rows, d4p16_rows, "D4P16")

    explicit_matrix = _matrix_for_columns(b0_rows, _available_explicit_columns(b0_rows))
    d4p8_matrix = _matrix_for_prefix(d4p8_rows, "embedding_pca_")
    d4p16_matrix = _matrix_for_prefix(d4p16_rows, "embedding_pca_")

    rng = np.random.default_rng(int(seed))
    tables = {
        "D5R8": _build_rows(
            b0_rows,
            explicit_matrix,
            _matched_control_matrix(d4p8_matrix, rng),
        ),
        "D5S8": _build_rows(
            b0_rows,
            explicit_matrix,
            _shuffled_matrix(d4p8_matrix, rng),
        ),
        "D5R16": _build_rows(
            b0_rows,
            explicit_matrix,
            _matched_control_matrix(d4p16_matrix, rng),
        ),
        "D5S16": _build_rows(
            b0_rows,
            explicit_matrix,
            _shuffled_matrix(d4p16_matrix, rng),
        ),
    }
    manifest = _build_manifest(tables, _available_explicit_columns(b0_rows))
    summary = _build_feature_summary(tables, d4p8_matrix, d4p16_matrix)
    return {
        "phase": "phase59_matched_dimension_control_features",
        "seed": int(seed),
        "variant_ids": list(PHASE59_MATCHED_CONTROL_VARIANTS),
        "summary": summary,
        "manifest": manifest,
        "variant_tables": tables,
        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
    }
```

Use these helper rules in the same file:

- `_load_rows` accepts a path or a list of mappings and returns `list[dict[str, object]]`.
- `_available_explicit_columns` returns the ordered intersection of `EXPLICIT_FEATURE_COLUMNS` and columns present in the rows; if none exist, it raises `ValueError("Phase 59 requires explicit planning feature columns")`.
- `_matrix_for_prefix` selects sorted `embedding_pca_` columns and raises `ValueError("Phase 59 requires embedding_pca columns for <variant>")` when no columns exist.
- `_matched_control_matrix` draws standard normal values and rescales each generated column to the source PCA column mean and population standard deviation, using `1.0` for zero source standard deviation.
- `_shuffled_matrix` permutes row order and rolls by one row if the random permutation equals the identity.
- `_build_rows` writes `block_id`, explicit columns, and `matched_control_00` through `matched_control_NN`.
- `_build_manifest` writes `ready: True`, `missing: []`, `reward: base_planning_reward`, and feature table names `variant_D5R8_features.csv`, `variant_D5S8_features.csv`, `variant_D5R16_features.csv`, `variant_D5S16_features.csv`.

- [ ] **Step 2: Add writer tests**

Append this test:

```python
def test_phase59_writes_control_tables_and_manifest(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_tables,
        write_phase59_matched_dimension_control_tables,
    )

    protocol = build_phase59_matched_dimension_control_tables(
        _b0_rows(),
        _d4p8_rows(),
        _d4p16_rows(),
        seed=59,
    )
    paths = write_phase59_matched_dimension_control_tables(
        protocol,
        tmp_path / "controls",
    )

    assert paths["manifest"].name == "experiment_variants.json"
    assert paths["summary"].name == "phase59_matched_dimension_control_feature_summary.json"
    assert set(paths["variant_tables"]) == {"D5R8", "D5S8", "D5R16", "D5S16"}
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert saved["variants"]["D5R8"]["feature_table"] == "variant_D5R8_features.csv"
    with paths["variant_tables"]["D5S16"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["block_id"] == "b1"
    assert "matched_control_02" in rows[0]
```

- [ ] **Step 3: Implement the writer**

Add `write_phase59_matched_dimension_control_tables(protocol, output_dir)` to the module. It should:

```python
def write_phase59_matched_dimension_control_tables(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = protocol["manifest"]
    summary = protocol["summary"]
    variant_tables = protocol["variant_tables"]

    table_paths: dict[str, Path] = {}
    for variant_id, rows in variant_tables.items():
        feature_table = manifest["variants"][variant_id]["feature_table"]
        required_columns = manifest["variants"][variant_id]["required_columns"]
        path = output_path / feature_table
        _write_variant_csv(path, rows, required_columns)
        table_paths[str(variant_id)] = path

    manifest_path = output_path / "experiment_variants.json"
    summary_path = output_path / "phase59_matched_dimension_control_feature_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest_path, "summary": summary_path, "variant_tables": table_paths}
```

- [ ] **Step 4: Run feature construction tests**

Run:

```powershell
python -m pytest tests\test_phase59_matched_dimension_controls.py::test_phase59_builds_deterministic_matched_control_tables tests\test_phase59_matched_dimension_controls.py::test_phase59_writes_control_tables_and_manifest -q
```

Expected result: `2 passed`.

- [ ] **Step 5: Commit feature-table construction**

Run:

```powershell
git add src/paper11_geofm/phase59_matched_dimension_controls.py tests/test_phase59_matched_dimension_controls.py
git commit -m "feat: add Phase 59 matched control tables"
```

### Task 3: Add Matched Analysis and Artifact Writers

**Files:**
- Modify: `src/paper11_geofm/phase59_matched_dimension_controls.py`
- Modify: `tests/test_phase59_matched_dimension_controls.py`

- [ ] **Step 1: Add synthetic summary-row fixtures and status tests**

Append these helpers to the test file:

```python
def _summary_row(variant_id, reward, tile_id="tile_a", seed=0):
    return {
        "row_type": "trained_policy",
        "variant_id": variant_id,
        "train_tile_id": "tile_train",
        "eval_tile_id": tile_id,
        "eval_tile_rank": 1 if tile_id == "tile_a" else 2,
        "seed": seed,
        "phase25_seed_rank": seed + 1,
        "train_timesteps": 4096,
        "eval_max_steps": 8,
        "max_blocks": 4,
        "train_n_blocks": 4,
        "eval_n_blocks": 2,
        "n_features": 25,
        "observation_shape": 100,
        "action_space_n": 4,
        "episode_steps": 2,
        "terminated": True,
        "truncated": False,
        "all_actions_valid": True,
        "invalid_action_count": 0,
        "total_contract_reward": reward,
        "selected_block_ids": "b1;b2",
        "claim_boundary": "fixture",
    }


def _phase59_summary_rows(case="supported"):
    rewards = {
        "supported": {
            "D4P8": [1.2, 1.3, 1.1, 1.4],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 1.0, 0.8, 1.0],
            "D4P16": [1.5, 1.4, 1.3, 1.6],
            "D5R16": [1.1, 1.1, 1.2, 1.2],
            "D5S16": [1.0, 1.1, 1.0, 1.1],
        },
        "partial": {
            "D4P8": [1.2, 1.2, 1.2, 1.2],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 0.9, 0.9, 0.9],
            "D4P16": [1.0, 1.0, 1.0, 1.0],
            "D5R16": [1.1, 1.1, 1.1, 1.1],
            "D5S16": [1.2, 1.2, 1.2, 1.2],
        },
        "not_supported": {
            "D4P8": [0.8, 0.8, 0.8, 0.8],
            "D5R8": [1.0, 1.0, 1.0, 1.0],
            "D5S8": [0.9, 0.9, 0.9, 0.9],
            "D4P16": [0.9, 0.9, 0.9, 0.9],
            "D5R16": [1.0, 1.0, 1.0, 1.0],
            "D5S16": [1.1, 1.1, 1.1, 1.1],
        },
    }[case]
    pairs = [("tile_a", 0), ("tile_a", 1), ("tile_b", 0), ("tile_b", 1)]
    rows = []
    for index, (tile_id, seed) in enumerate(pairs):
        for variant_id, values in rewards.items():
            rows.append(_summary_row(variant_id, values[index], tile_id, seed))
    return rows
```

Append these tests:

```python
def test_phase59_analysis_supports_matched_dimension_geofm_route():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    analysis = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("supported"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=200,
        random_seed=59,
    )

    assert analysis["phase"] == "phase59_matched_dimension_control_analysis"
    assert analysis["phase59_matched_dimension_status"] == "matched_dimension_geofm_supported"
    assert analysis["learned_policy"]["matched_deltas"]["D4P8_minus_D5R8"]["mean_delta"] == 0.25
    assert analysis["pooled_matched_control_delta"]["positive_count"] == 16
    assert analysis["cluster_summary"]["cluster_count"] == 4
    assert analysis["signed_rank_summary"]["positive_rank_sum"] == 10


def test_phase59_status_rules_distinguish_partial_and_not_supported():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    partial = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("partial"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    not_supported = build_phase59_matched_dimension_control_analysis(
        _phase59_summary_rows("not_supported"),
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )

    assert partial["phase59_matched_dimension_status"] == "matched_dimension_geofm_partial"
    assert not_supported["phase59_matched_dimension_status"] == "matched_dimension_geofm_not_supported"


def test_phase59_reports_insufficient_for_missing_variant_rows():
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
    )

    rows = [
        row for row in _phase59_summary_rows("supported")
        if row["variant_id"] != "D5S16"
    ]
    analysis = build_phase59_matched_dimension_control_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
    )

    assert analysis["phase59_matched_dimension_status"] == "insufficient"
    missing = {
        row["variant_id"] for row in analysis["coverage_issues"]["missing_variant_rows"]
    }
    assert missing == {"D5S16"}
```

- [ ] **Step 2: Implement the analysis functions**

Add these public functions:

```python
def build_phase59_matched_dimension_control_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    bootstrap_iterations: int = 5000,
    random_seed: int = 59,
    pooled_positive_threshold: float = 0.5,
) -> dict[str, object]:
    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [
        row for row in rows if str(row.get("row_type", "")) == "trained_policy"
    ]
    metadata_map = {} if metadata is None else dict(metadata)
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(trained_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(trained_rows, "seed"),
    )
    coverage_issues = _coverage_issues(
        trained_rows,
        variants=PHASE59_REQUIRED_VARIANTS,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    delta_rows = _matched_delta_rows(trained_rows, eval_tile_ids, seeds)
    learned_policy = _phase59_policy_summary(trained_rows, delta_rows)
    pooled = _delta_summary(
        [
            float(row["compressed_minus_matched_control_reward"])
            for row in delta_rows
        ],
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(random_seed),
    )
    cluster_rows = _cluster_rows(delta_rows)
    cluster_summary = _cluster_summary(cluster_rows)
    signed_rank_summary = _signed_rank_summary(cluster_rows)
    status = _phase59_status(
        learned_policy["matched_deltas"],
        pooled,
        coverage_issues,
        pooled_positive_threshold=float(pooled_positive_threshold),
    )
    return {
        "phase": "phase59_matched_dimension_control_analysis",
        "variants": list(PHASE59_REQUIRED_VARIANTS),
        "matched_comparisons": [
            {"compressed_variant_id": left, "matched_control_variant_id": right}
            for left, right in PHASE59_MATCHED_COMPARISONS
        ],
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(rows),
        "delta_rows": delta_rows,
        "learned_policy": learned_policy,
        "pooled_matched_control_delta": pooled,
        "cluster_rows": cluster_rows,
        "cluster_summary": cluster_summary,
        "signed_rank_summary": signed_rank_summary,
        "coverage_issues": coverage_issues,
        "phase59_matched_dimension_status": status,
        "conclusion": _phase59_conclusion(status),
        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
    }
```

The implementation should mirror Phase 48, Phase 49, Phase 50, and Phase 51 naming:

- `_matched_delta_rows` writes fields `compressed_variant_id`, `matched_control_variant_id`, `eval_tile_id`, `seed`, `compressed_reward`, `matched_control_reward`, `compressed_minus_matched_control_reward`, `compressed_improves_matched_control`, `train_timesteps`, `eval_max_steps`, and `claim_boundary`.
- `_delta_summary` returns `mean_delta`, `std_delta`, `positive_count`, `total_count`, `positive_fraction`, `one_sided_sign_test_p`, `bootstrap_ci95_low`, and `bootstrap_ci95_high`.
- `_cluster_rows` groups deltas by `(eval_tile_id, seed)` and writes `mean_cluster_delta`.
- `_signed_rank_summary` ranks nonzero absolute cluster means and computes an exact one-sided signed-rank p-value using the same enumeration approach as Phase 51.
- `_phase59_status` follows the spec status rule exactly.

- [ ] **Step 3: Add writer tests**

Append this test:

```python
def test_phase59_writer_outputs_json_summary_delta_and_markdown(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_analysis,
        write_phase59_matched_dimension_control_artifacts,
    )

    rows = _phase59_summary_rows("supported")
    analysis = build_phase59_matched_dimension_control_analysis(
        rows,
        metadata={"eval_tile_ids": ["tile_a", "tile_b"], "seeds": [0, 1]},
        bootstrap_iterations=100,
    )
    paths = write_phase59_matched_dimension_control_artifacts(
        {**analysis, "summaries": rows},
        tmp_path / "outputs",
    )

    assert paths["summary_csv"].name == "phase59_matched_dimension_control_summary.csv"
    assert paths["delta_csv"].name == "phase59_matched_dimension_delta_table.csv"
    assert paths["comparison_json"].name == "phase59_matched_dimension_controls.json"
    assert paths["readiness_md"].name == "phase59_matched_dimension_controls.md"
    saved = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert saved["phase59_matched_dimension_status"] == "matched_dimension_geofm_supported"
    with paths["delta_csv"].open("r", encoding="utf-8", newline="") as handle:
        delta_rows = list(csv.DictReader(handle))
    assert any(
        row["compressed_variant_id"] == "D4P16"
        and row["matched_control_variant_id"] == "D5S16"
        for row in delta_rows
    )
    markdown = paths["readiness_md"].read_text(encoding="utf-8")
    assert "Matched-dimension control audit" in markdown
    assert "does not enable suitability reward" in markdown
```

- [ ] **Step 4: Implement artifact writers**

Add `write_phase59_matched_dimension_control_artifacts(analysis, output_dir)` that writes:

- `phase59_matched_dimension_control_summary.csv` using `SUMMARY_FIELDNAMES` from `paper11_geofm.padded_heldout_policy`;
- `phase59_matched_dimension_delta_table.csv` using Phase 59 delta fieldnames;
- `phase59_matched_dimension_controls.json` with `source_rows` and `summaries` removed;
- `phase59_matched_dimension_controls.md` with status, conclusion, per-comparison deltas, pooled delta, cluster summary, signed-rank summary, and claim boundary.

- [ ] **Step 5: Run analysis and writer tests**

Run:

```powershell
python -m pytest tests\test_phase59_matched_dimension_controls.py -q
```

Expected result: all Phase 59 tests written so far pass.

- [ ] **Step 6: Commit analysis and writers**

Run:

```powershell
git add src/paper11_geofm/phase59_matched_dimension_controls.py tests/test_phase59_matched_dimension_controls.py
git commit -m "feat: add Phase 59 matched control analysis"
```

### Task 4: Add Phase 59 Policy Evaluation and CLI

**Files:**
- Modify: `src/paper11_geofm/phase59_matched_dimension_controls.py`
- Create: `experiments/phase59_matched_dimension_controls/run_phase59_matched_dimension_controls.py`
- Modify: `tests/test_phase59_matched_dimension_controls.py`

- [ ] **Step 1: Add run-contract and loader tests that avoid PPO training**

Append this test:

```python
def test_phase59_contract_routes_d4p_and_d5_variants(tmp_path):
    from paper11_geofm.phase59_matched_dimension_controls import (
        build_phase59_matched_dimension_control_contract,
    )

    tile_index = _write_csv(
        tmp_path / "tiles.csv",
        [
            {"tile_id": "tile_train", "block_ids": "b1;b2;b3;b4"},
            {"tile_id": "tile_a", "block_ids": "b1;b2"},
            {"tile_id": "tile_b", "block_ids": "b3;b4"},
        ],
    )
    contract = build_phase59_matched_dimension_control_contract(
        phase8_output_dir=tmp_path / "phase8",
        phase59_control_dir=tmp_path / "phase59_controls",
        tile_index_csv=tile_index,
        train_tile_id="tile_train",
        eval_tile_ids="tile_a,tile_b",
        total_timesteps=4096,
        eval_max_steps=8,
        seeds="0,1",
    )

    assert contract["variants"] == ["D4P8", "D4P16", "D5R8", "D5S8", "D5R16", "D5S16"]
    assert contract["variant_source_dirs"]["D4P8"].endswith("phase8")
    assert contract["variant_source_dirs"]["D5S16"].endswith("phase59_controls")
    assert contract["eval_tile_ids"] == ["tile_a", "tile_b"]
    assert contract["seeds"] == [0, 1]
```

- [ ] **Step 2: Implement contract and evaluation orchestration**

Add these public functions. The contract body should follow the shape below; the helper calls named here are defined in Task 2 and Task 3 or imported from existing Phase 25/28 modules:

```python
def build_phase59_matched_dimension_control_contract(
    phase8_output_dir: Path | str,
    phase59_control_dir: Path | str,
    tile_index_csv: Path | str,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    total_timesteps: int = 4096,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if int(eval_max_steps) <= 0:
        raise ValueError("eval_max_steps must be positive")
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    train_id = str(selected["train_tile_id"])
    return {
        "phase": "phase59_matched_dimension_control_evaluation",
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "phase59_control_dir": str(Path(phase59_control_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": list(PHASE59_REQUIRED_VARIANTS),
        "variant_source_dirs": {
            "D4P8": str(Path(phase8_output_dir)),
            "D4P16": str(Path(phase8_output_dir)),
            "D5R8": str(Path(phase59_control_dir)),
            "D5S8": str(Path(phase59_control_dir)),
            "D5R16": str(Path(phase59_control_dir)),
            "D5S16": str(Path(phase59_control_dir)),
        },
        "train_tile_id": train_id,
        "eval_tile_ids": eval_ids,
        "eval_tile_ranks": {
            str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
        },
        "selected_tile_block_counts": selected_counts,
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_ranks": {
            str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
        },
        "claim_boundary": PHASE59_CLAIM_BOUNDARY,
    }


def run_phase59_matched_dimension_control_evaluation(
    phase8_output_dir: Path | str,
    phase59_control_dir: Path | str,
    tile_index_csv: Path | str,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 5,
    total_timesteps: int = 4096,
    eval_max_steps: int = 8,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    contract = build_phase59_matched_dimension_control_contract(
        phase8_output_dir=phase8_output_dir,
        phase59_control_dir=phase59_control_dir,
        tile_index_csv=tile_index_csv,
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
            train_tiled = _load_phase59_tiled_variant_input(
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
                eval_tiled = _load_phase59_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=int(
                        contract["selected_tile_block_counts"][
                            str(contract["train_tile_id"])
                        ]
                    ),
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=int(contract["eval_tile_ranks"][str(eval_tile_id)]),
                    phase25_seed_rank=int(contract["seed_ranks"][str(int(seed))]),
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                trained_summary["claim_boundary"] = PHASE59_CLAIM_BOUNDARY
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
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=int(contract["eval_tile_ranks"][str(eval_tile_id)]),
                        phase25_seed_rank=int(contract["seed_ranks"][str(int(seed))]),
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    baseline_summary["claim_boundary"] = PHASE59_CLAIM_BOUNDARY
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )
    analysis = build_phase59_matched_dimension_control_analysis(
        summaries,
        metadata={
            "eval_tile_ids": contract["eval_tile_ids"],
            "seeds": contract["seeds"],
        },
    )
    analysis["contract"] = contract
    analysis["summaries"] = summaries
    analysis["traces"] = traces
    return analysis
```

Use these imports:

```python
from .padded_heldout_policy import (
    Phase25PaddedTileEnv,
    _evaluate_baseline_policy,
    _evaluate_trained_policy,
    _normalize_seeds,
    _select_train_eval_tiles,
    _store_trace,
)
from .phase28_representation_controls import _train_maskable_ppo_model
from .tiled_inputs import load_tiled_variant_input
```

The evaluation loop should:

- train one MaskablePPO model per variant and seed on the selected train tile;
- evaluate on each selected held-out tile;
- append trained-policy rows and first-valid/seeded-random baseline rows using existing helper outputs;
- set every summary row `claim_boundary` to `PHASE59_CLAIM_BOUNDARY`;
- write trace data under `trained_policy`, `first_valid`, and `seeded_random`;
- call `build_phase59_matched_dimension_control_analysis` on the collected summary rows before returning.

- [ ] **Step 3: Add CLI tests for build-controls and analyze-only modes**

Append this test:

```python
def test_phase59_cli_build_controls_and_analyze_only(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase59_matched_dimension_controls"
        / "run_phase59_matched_dimension_controls.py"
    )
    spec = importlib.util.spec_from_file_location("phase59_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    build_exit = module.main(
        [
            "--mode",
            "build-controls",
            "--b0-features-csv",
            str(_write_csv(tmp_path / "b0.csv", _b0_rows())),
            "--d4p8-features-csv",
            str(_write_csv(tmp_path / "d4p8.csv", _d4p8_rows())),
            "--d4p16-features-csv",
            str(_write_csv(tmp_path / "d4p16.csv", _d4p16_rows())),
            "--output-dir",
            str(tmp_path / "controls"),
            "--seed",
            "59",
        ]
    )
    assert build_exit == 0
    assert (tmp_path / "controls" / "experiment_variants.json").exists()

    summary_csv = _write_csv(tmp_path / "summary.csv", _phase59_summary_rows("supported"))
    analyze_exit = module.main(
        [
            "--mode",
            "analyze-only",
            "--existing-summary-csv",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--eval-tile-ids",
            "tile_a,tile_b",
            "--seeds",
            "0,1",
            "--bootstrap-iterations",
            "100",
        ]
    )

    stdout = capsys.readouterr().out
    assert analyze_exit == 0
    assert "Phase 59 matched-dimension status: matched_dimension_geofm_supported" in stdout
    assert "phase59_matched_dimension_controls.json" in stdout
```

- [ ] **Step 4: Create CLI runner**

Create `experiments/phase59_matched_dimension_controls/run_phase59_matched_dimension_controls.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase59_matched_dimension_controls import (
    build_phase59_matched_dimension_control_analysis,
    build_phase59_matched_dimension_control_tables,
    run_phase59_matched_dimension_control_evaluation,
    write_phase59_matched_dimension_control_artifacts,
    write_phase59_matched_dimension_control_tables,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "build-controls":
            protocol = build_phase59_matched_dimension_control_tables(
                args.b0_features_csv,
                args.d4p8_features_csv,
                args.d4p16_features_csv,
                seed=args.seed,
            )
            paths = write_phase59_matched_dimension_control_tables(protocol, args.output_dir)
            print(f"Phase 59 control tables: {paths['manifest']}")
            return 0
        if args.mode == "run-and-analyze":
            protocol = run_phase59_matched_dimension_control_evaluation(
                phase8_output_dir=args.phase8_output_dir,
                phase59_control_dir=args.phase59_control_dir,
                tile_index_csv=args.tile_index_csv,
                train_tile_id=args.train_tile_id,
                eval_tile_ids=args.eval_tile_ids,
                max_eval_tiles=args.max_eval_tiles,
                total_timesteps=args.total_timesteps,
                eval_max_steps=args.eval_max_steps,
                seeds=args.seeds,
            )
        else:
            protocol = build_phase59_matched_dimension_control_analysis(
                args.existing_summary_csv,
                metadata={
                    "eval_tile_ids": args.eval_tile_ids,
                    "seeds": args.seeds,
                },
                bootstrap_iterations=args.bootstrap_iterations,
                random_seed=args.seed,
            )
        paths = write_phase59_matched_dimension_control_artifacts(protocol, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 59 matched-dimension status: {protocol['phase59_matched_dimension_status']}")
    print(f"Comparison JSON: {paths['comparison_json']}")
    print(f"Delta CSV: {paths['delta_csv']}")
    print(f"Readiness Markdown: {paths['readiness_md']}")
    print(f"Claim boundary: {protocol['claim_boundary']}")
    return 0
```

The parser must require:

- `--b0-features-csv`, `--d4p8-features-csv`, `--d4p16-features-csv`, and `--output-dir` for `build-controls`;
- `--phase8-output-dir`, `--phase59-control-dir`, `--tile-index-csv`, and `--output-dir` for `run-and-analyze`;
- `--existing-summary-csv` and `--output-dir` for `analyze-only`.

- [ ] **Step 5: Run full Phase 59 unit tests**

Run:

```powershell
python -m pytest tests\test_phase59_matched_dimension_controls.py -q
```

Expected result: all Phase 59 tests pass.

- [ ] **Step 6: Commit policy evaluation and CLI**

Run:

```powershell
git add src/paper11_geofm/phase59_matched_dimension_controls.py experiments/phase59_matched_dimension_controls/run_phase59_matched_dimension_controls.py tests/test_phase59_matched_dimension_controls.py
git commit -m "feat: add Phase 59 matched control runner"
```

### Task 5: Run Real Phase 59 and Record Evidence

**Files:**
- Create ignored outputs under: `experiments/phase59_matched_dimension_controls/outputs/phase52_full5_seed3/`
- Create: `paper/phase28_results/25_phase59_matched_dimension_controls.md`
- Modify: `paper/phase28_results/README.md`
- Modify: `docs/superpowers/phase33_current_progress_handoff.md`

- [ ] **Step 1: Build real matched-control feature tables**

Run:

```powershell
python experiments\phase59_matched_dimension_controls\run_phase59_matched_dimension_controls.py --mode build-controls --b0-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B0_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\matched_control_features --seed 59
```

Expected result: output prints an `experiment_variants.json` path and creates all four `variant_D5*.csv` files.

- [ ] **Step 2: Run real matched policy evaluation and analysis**

Run:

```powershell
python experiments\phase59_matched_dimension_controls\run_phase59_matched_dimension_controls.py --mode run-and-analyze --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --phase59-control-dir experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\matched_control_features --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --output-dir experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3 --max-eval-tiles 5 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --bootstrap-iterations 5000 --seed 59
```

Expected result: output prints one of the Phase 59 statuses and writes:

- `phase59_matched_dimension_control_summary.csv`
- `phase59_matched_dimension_delta_table.csv`
- `phase59_matched_dimension_controls.json`
- `phase59_matched_dimension_controls.md`

- [ ] **Step 3: Inspect and summarize the real result**

Run:

```powershell
Get-Content -Raw -LiteralPath experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json
```

Record these values for the evidence note:

- `phase59_matched_dimension_status`
- all four matched comparison mean deltas;
- pooled matched-control mean delta and positive count;
- pooled bootstrap CI;
- cluster mean, sign-test p, and signed-rank p.

- [ ] **Step 4: Create the Phase 59 result note**

Generate `paper/phase28_results/25_phase59_matched_dimension_controls.md` from the real JSON values with this PowerShell script:

```powershell
$j = Get-Content -Raw -LiteralPath experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json | ConvertFrom-Json
$d = $j.learned_policy.matched_deltas
$p = $j.pooled_matched_control_delta
$c = $j.cluster_summary
$r = $j.signed_rank_summary
switch ($j.phase59_matched_dimension_status) {
  'matched_dimension_geofm_supported' { $interpretation = 'Phase 59 strengthens the compressed GeoFM interpretation: D4P8/D4P16 beat same-dimension random and shuffled controls under the current Bishan base-reward protocol.' }
  'matched_dimension_geofm_partial' { $interpretation = 'Phase 59 provides partial matched-dimension support. The mechanism wording should identify which compressed route beat both same-dimension controls and avoid a blanket D4P8/D4P16 claim.' }
  'matched_dimension_geofm_not_supported' { $interpretation = 'Phase 59 does not support a GeoFM-specific matched-dimension advantage. The mechanism wording should be narrowed toward low-dimensional representation effects rather than GeoFM-derived signal.' }
  default { $interpretation = 'Phase 59 was insufficient for a matched-dimension decision because required comparable rows were missing, duplicated, or otherwise not comparable.' }
}
@"
# Phase 59 Matched-Dimension Controls

Phase 59 tests whether the D4P8/D4P16 compressed GeoFM gains exceed
dimension-matched random and shuffled controls under the same Bishan
base-reward held-out protocol.

## Status

``$($j.phase59_matched_dimension_status)``

## Matched Comparisons

| Comparison | Mean delta | Positive rows |
|---|---:|---:|
| D4P8 - D5R8 | $($d.D4P8_minus_D5R8.mean_delta) | $($d.D4P8_minus_D5R8.positive_count) / $($d.D4P8_minus_D5R8.total_count) |
| D4P8 - D5S8 | $($d.D4P8_minus_D5S8.mean_delta) | $($d.D4P8_minus_D5S8.positive_count) / $($d.D4P8_minus_D5S8.total_count) |
| D4P16 - D5R16 | $($d.D4P16_minus_D5R16.mean_delta) | $($d.D4P16_minus_D5R16.positive_count) / $($d.D4P16_minus_D5R16.total_count) |
| D4P16 - D5S16 | $($d.D4P16_minus_D5S16.mean_delta) | $($d.D4P16_minus_D5S16.positive_count) / $($d.D4P16_minus_D5S16.total_count) |

## Pooled and Cluster Evidence

- Pooled mean delta: $($p.mean_delta)
- Pooled positive rows: $($p.positive_count) / $($p.total_count)
- Pooled bootstrap CI95: [$($p.bootstrap_ci95_low), $($p.bootstrap_ci95_high)]
- Cluster mean delta: $($c.mean_cluster_delta)
- Cluster sign-test p: $($c.one_sided_sign_test_p)
- Cluster signed-rank p: $($r.one_sided_signed_rank_p)

## Interpretation

$interpretation

## Boundary

Phase 59 does not enable suitability reward, B2/B3, cross-region transfer, PCA
optimality, or independent agronomic suitability claims.
"@ | Set-Content -LiteralPath paper\phase28_results\25_phase59_matched_dimension_controls.md -Encoding utf8
```

After running the script, open the generated Markdown and confirm it contains numeric values rather than blank fields.

- [ ] **Step 5: Update index and handoff records**

Add one bullet to `paper/phase28_results/README.md` for `25_phase59_matched_dimension_controls.md`.

Append a new `## Phase 59 Matched-Dimension Control Audit` section to `docs/superpowers/phase33_current_progress_handoff.md` with:

- new implementation files;
- real command;
- real status;
- core matched comparison values;
- conclusion boundary.

- [ ] **Step 6: Commit real evidence docs**

Run:

```powershell
git add paper/phase28_results/25_phase59_matched_dimension_controls.md paper/phase28_results/README.md docs/superpowers/phase33_current_progress_handoff.md
git commit -m "docs: record Phase 59 matched control evidence"
```

### Task 6: Verification and Final Save

**Files:**
- All Phase 59 implementation, test, runner, and documentation files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_phase59_matched_dimension_controls.py tests\test_phase48_compressed_geofm_rescue.py tests\test_phase57_compressed_representation_mechanism.py -q --basetemp=.pytest_tmp_phase59_verify -p no:cacheprovider
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
git log --oneline -5
```

Expected result: branch is `main`, local branch is ahead by the new Phase 59 commits unless pushed, and no unstaged source/documentation edits remain.

- [ ] **Step 5: Push when the user asks or after confirming the real run is complete**

Run:

```powershell
git push origin main
```

Expected result: `main` synchronizes with `origin/main`.


