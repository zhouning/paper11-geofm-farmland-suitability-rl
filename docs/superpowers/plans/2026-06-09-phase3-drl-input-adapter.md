# Phase 3 DRL Input Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Phase 3 adapter that loads ready Phase 2 variant CSV exports into numeric DRL input matrices without training or evaluating policies.

**Architecture:** Add a focused `paper11_geofm.drl_inputs` module for loading and validating variant inputs from `experiment_variants.json` plus `variant_B*_features.csv`. Add a small CLI under `experiments/phase3_drl_input_adapter/` for reviewer-facing inspection, then document the new smoke path.

**Tech Stack:** Python standard library (`csv`, `json`, `argparse`, `dataclasses`, `pathlib`), NumPy, pytest.

---

## File Structure

- Create `src/paper11_geofm/drl_inputs.py`: public `VariantInput` dataclass and `load_variant_input()` function.
- Create `experiments/phase3_drl_input_adapter/inspect_variant_inputs.py`: CLI wrapper around the loader.
- Create `tests/test_phase3_drl_inputs.py`: TDD coverage for loader and CLI behavior.
- Modify `README.md`: add Phase 3 quick-start command and clarify no DRL training is performed.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 3 reproduction section.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add the new module, CLI, test, spec, and plan rows.

## Task 1: Variant Loader Happy Path

**Files:**
- Create: `tests/test_phase3_drl_inputs.py`
- Create: `src/paper11_geofm/drl_inputs.py`

- [ ] **Step 1: Write the failing loader test**

Add this test file:

```python
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id):
    row = {"block_id": block_id, "suitability_proxy": 0.75}
    for dim in range(64):
        row[f"embedding_mean_{dim:02d}"] = float(dim)
    for idx in range(17):
        row[f"explicit_feature_{idx:02d}"] = float(idx)
    return row


def _phase2_test_summary():
    return {
        "metadata_source": "test",
        "base_year_requested": 2020,
        "base_year_used": 2020,
        "years": [2020],
        "grid_shape": [2, 2],
        "embedding_dim": 64,
        "mapping_mode": "test",
    }


def _write_ready_phase2_outputs(output_dir):
    from paper11_geofm.artifacts import write_phase2_artifacts

    return write_phase2_artifacts(
        [
            _complete_phase2_feature_row("sample_block_00"),
            _complete_phase2_feature_row("sample_block_01"),
        ],
        output_dir,
        _phase2_test_summary(),
    )


def test_load_variant_input_reads_ready_b3_matrix(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    loaded = load_variant_input(tmp_path, "b3")

    assert loaded.variant_id == "B3"
    assert loaded.block_ids == ("sample_block_00", "sample_block_01")
    assert loaded.feature_columns[0] == "explicit_feature_00"
    assert loaded.feature_columns[-1] == "suitability_proxy"
    assert loaded.reward_mode == "base_plus_suitability_reward"
    assert loaded.state_groups == (
        "explicit_planning_features",
        "geofm_embedding",
        "suitability_proxy",
    )
    assert loaded.source_table.name == "variant_B3_features.csv"
    assert loaded.state_matrix.dtype == np.float32
    assert loaded.state_matrix.shape == (2, 82)
    assert loaded.state_matrix[0, 0] == np.float32(0.0)
    assert loaded.state_matrix[0, 80] == np.float32(63.0)
    assert loaded.state_matrix[0, 81] == np.float32(0.75)
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py::test_load_variant_input_reads_ready_b3_matrix -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paper11_geofm.drl_inputs'`.

- [ ] **Step 3: Implement minimal loader**

Create `src/paper11_geofm/drl_inputs.py`:

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VariantInput:
    variant_id: str
    block_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    state_matrix: np.ndarray
    reward_mode: str
    state_groups: tuple[str, ...]
    source_table: Path


def load_variant_input(phase2_output_dir: Path | str, variant_id: str) -> VariantInput:
    output_dir = Path(phase2_output_dir)
    manifest_path = output_dir / "experiment_variants.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 variant manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Phase 2 variant manifest is missing a variants object")

    normalized_variant_id = variant_id.upper()
    variant = variants.get(normalized_variant_id)
    if not isinstance(variant, dict):
        raise ValueError(f"Unknown Phase 2 variant: {normalized_variant_id}")
    if not variant.get("ready"):
        missing = variant.get("missing", [])
        raise ValueError(
            f"Phase 2 variant {normalized_variant_id} is not ready: {missing}"
        )

    feature_table = variant.get("feature_table")
    if not feature_table:
        raise ValueError(
            f"Phase 2 variant {normalized_variant_id} has no feature_table"
        )
    source_table = output_dir / str(feature_table)
    if not source_table.exists():
        raise FileNotFoundError(
            f"Missing feature table for variant {normalized_variant_id}: "
            f"{source_table}"
        )

    required_columns = tuple(str(column) for column in variant["required_columns"])
    block_ids, matrix = _read_variant_csv(
        source_table,
        normalized_variant_id,
        required_columns,
    )

    return VariantInput(
        variant_id=normalized_variant_id,
        block_ids=block_ids,
        feature_columns=required_columns,
        state_matrix=matrix,
        reward_mode=str(variant.get("reward", "")),
        state_groups=tuple(str(group) for group in variant.get("state_groups", [])),
        source_table=source_table,
    )


def _read_variant_csv(
    path: Path,
    variant_id: str,
    required_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [
            column for column in ("block_id", *required_columns) if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Feature table for variant {variant_id} is missing columns: "
                f"{missing_columns}"
            )

        block_ids: list[str] = []
        rows: list[list[float]] = []
        seen_block_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            block_id = str(row.get("block_id", "")).strip()
            if not block_id:
                raise ValueError(f"Missing block_id at {path}:{row_number}")
            if block_id in seen_block_ids:
                raise ValueError(f"Duplicate block_id in {path}: {block_id}")
            seen_block_ids.add(block_id)
            block_ids.append(block_id)
            rows.append(
                [
                    _parse_required_float(
                        row.get(column),
                        path,
                        row_number,
                        column,
                    )
                    for column in required_columns
                ]
            )

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(0, len(required_columns))
    return tuple(block_ids), matrix


def _parse_required_float(
    value: Any,
    path: Path,
    row_number: int,
    column: str,
) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric value at {path}:{row_number} column {column}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value at {path}:{row_number} column {column}: {value!r}"
        ) from exc
```

- [ ] **Step 4: Run the test to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py::test_load_variant_input_reads_ready_b3_matrix -q
```

Expected: PASS.

## Task 2: Loader Validation and Variant Dimensions

**Files:**
- Modify: `tests/test_phase3_drl_inputs.py`
- Modify: `src/paper11_geofm/drl_inputs.py`

- [ ] **Step 1: Add failing validation tests**

Append these tests:

```python
import pytest


def test_load_variant_input_reports_expected_dimensions(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)

    expected = {
        "B0": 17,
        "B1": 81,
        "B2": 18,
        "B3": 82,
    }
    for variant_id, feature_count in expected.items():
        loaded = load_variant_input(tmp_path, variant_id)
        assert loaded.feature_columns == tuple(
            json.loads((tmp_path / "experiment_variants.json").read_text())
            ["variants"][variant_id]["required_columns"]
        )
        assert loaded.state_matrix.shape == (2, feature_count)


def test_load_variant_input_rejects_incomplete_variant(tmp_path):
    from paper11_geofm.artifacts import write_phase2_artifacts
    from paper11_geofm.drl_inputs import load_variant_input

    write_phase2_artifacts(
        [{"block_id": "b0", "suitability_proxy": 0.5}],
        tmp_path,
        _phase2_test_summary(),
    )

    with pytest.raises(ValueError, match="B3 is not ready"):
        load_variant_input(tmp_path, "B3")


def test_load_variant_input_rejects_duplicate_block_ids(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)
    table = tmp_path / "variant_B0_features.csv"
    rows = list(csv.DictReader(table.open("r", encoding="utf-8", newline="")))
    rows[1]["block_id"] = rows[0]["block_id"]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate block_id"):
        load_variant_input(tmp_path, "B0")


def test_load_variant_input_rejects_non_numeric_required_values(tmp_path):
    from paper11_geofm.drl_inputs import load_variant_input

    _write_ready_phase2_outputs(tmp_path)
    table = tmp_path / "variant_B0_features.csv"
    rows = list(csv.DictReader(table.open("r", encoding="utf-8", newline="")))
    rows[0]["explicit_feature_00"] = "not-a-number"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Non-numeric value"):
        load_variant_input(tmp_path, "B0")
```

- [ ] **Step 2: Run tests to verify RED or current partial behavior**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py -q
```

Expected: Any failures should identify missing import or validation gap. If all pass because Task 1 implementation already covered them, record that the broader validation behavior was covered by the first implementation and continue.

- [ ] **Step 3: Adjust implementation only if needed**

If tests fail due to missing validation details, update `src/paper11_geofm/drl_inputs.py` to match the test expectations. Do not add unrelated behavior.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py -q
```

Expected: PASS.

## Task 3: CLI Inspection Command

**Files:**
- Create: `experiments/phase3_drl_input_adapter/inspect_variant_inputs.py`
- Modify: `tests/test_phase3_drl_inputs.py`

- [ ] **Step 1: Add failing CLI test**

Append this test:

```python
def test_inspect_variant_inputs_cli_prints_contract_summary(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "inspect_variant_inputs",
        ROOT / "experiments" / "phase3_drl_input_adapter" / "inspect_variant_inputs.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_ready_phase2_outputs(tmp_path)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(tmp_path),
            "--variant",
            "B3",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant: B3" in stdout
    assert "Rows: 2" in stdout
    assert "Features: 82" in stdout
    assert "Matrix shape: 2 x 82" in stdout
    assert "Reward mode: base_plus_suitability_reward" in stdout
    assert "Claim boundary: input contract only; no DRL policy is trained or evaluated." in stdout
```

Also add `import importlib.util` near the top of `tests/test_phase3_drl_inputs.py`.

- [ ] **Step 2: Run CLI test to verify RED**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py::test_inspect_variant_inputs_cli_prints_contract_summary -q
```

Expected: FAIL because `inspect_variant_inputs.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `experiments/phase3_drl_input_adapter/inspect_variant_inputs.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.drl_inputs import load_variant_input


CLAIM_BOUNDARY = "input contract only; no DRL policy is trained or evaluated."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a ready Paper11 Phase 2 variant feature table as a DRL "
            "input matrix without training a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing experiment_variants.json and variant CSV exports.",
    )
    parser.add_argument(
        "--variant",
        default="B3",
        help="Variant ID to inspect: B0, B1, B2, or B3. Default: B3.",
    )
    args = parser.parse_args(argv)

    try:
        loaded = load_variant_input(args.phase2_output_dir, args.variant)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Variant: {loaded.variant_id}")
    print(f"Source table: {loaded.source_table}")
    print(f"Rows: {len(loaded.block_ids)}")
    print(f"Features: {len(loaded.feature_columns)}")
    print(
        f"Matrix shape: {loaded.state_matrix.shape[0]} x "
        f"{loaded.state_matrix.shape[1]}"
    )
    print(f"Reward mode: {loaded.reward_mode}")
    print(f"State groups: {', '.join(loaded.state_groups)}")
    print(f"Claim boundary: {CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test to verify GREEN**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py::test_inspect_variant_inputs_cli_prints_contract_summary -q
```

Expected: PASS.

## Task 4: Docs and Manifest

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`

- [ ] **Step 1: Update README**

Add a Phase 3 quick-start paragraph after the Phase 2 CSV fixture command explaining:

```text
After a Phase 2 run has produced ready variant CSVs, inspect the DRL input matrix without training a policy:

python experiments\phase3_drl_input_adapter\inspect_variant_inputs.py --phase2-output-dir .pytest_tmp\phase2_variant_csv_exports --variant B3

This command validates the `experiment_variants.json` contract, loads the requested variant CSV into a numeric matrix, and reports shape/reward metadata only.
```

- [ ] **Step 2: Update reproduction guide**

Add a new section after Phase 2:

```text
## 5. Inspect Phase 3 DRL Input Contracts

Run Phase 2 with the included fixture, then inspect B3:

...

Expected outcome:
- the command reports variant B3;
- row count is 4 for the fixture;
- feature count is 82;
- matrix shape is 4 x 82;
- the claim boundary states that no DRL policy is trained or evaluated.
```

Renumber later sections.

- [ ] **Step 3: Update file manifest**

Add rows for:

```text
docs/superpowers/specs/2026-06-09-phase3-drl-input-adapter-design.md
docs/superpowers/plans/2026-06-09-phase3-drl-input-adapter.md
src/paper11_geofm/drl_inputs.py
experiments/phase3_drl_input_adapter/inspect_variant_inputs.py
tests/test_phase3_drl_inputs.py
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\test_phase3_drl_inputs.py -q
```

Expected: PASS.

## Task 5: Full Verification and Commit

**Files:**
- All files touched above.

- [ ] **Step 1: Run full Phase 3 fixture flow**

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir .pytest_tmp\phase3_drl_adapter_fixture
python experiments\phase3_drl_input_adapter\inspect_variant_inputs.py --phase2-output-dir .pytest_tmp\phase3_drl_adapter_fixture --variant B3
```

Expected: Phase 3 output reports `Variant: B3`, `Rows: 4`, `Features: 82`, and `Matrix shape: 4 x 82`.

- [ ] **Step 2: Run repository verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected:

- smoke check passes;
- pytest reports all tests passing;
- `git diff --check` has no whitespace errors.

- [ ] **Step 3: Inspect git status**

Run:

```powershell
git status --short --branch
git diff --cached --name-only
```

Expected: only Phase 3 source, CLI, tests, docs, manifest, spec, and plan files are changed or staged. Generated outputs remain ignored.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\drl_inputs.py experiments\phase3_drl_input_adapter\inspect_variant_inputs.py tests\test_phase3_drl_inputs.py docs\superpowers\plans\2026-06-09-phase3-drl-input-adapter.md
git commit -m "Add Phase 3 DRL input adapter"
```

Expected: commit succeeds on `paper11-phase3-drl-input-adapter`.

## Task 6: Merge and Push

**Files:**
- Git branch state only.

- [ ] **Step 1: Push feature branch**

Run:

```powershell
git push -u origin paper11-phase3-drl-input-adapter
```

- [ ] **Step 2: Fast-forward main**

Run:

```powershell
git checkout main
git merge --ff-only paper11-phase3-drl-input-adapter
```

- [ ] **Step 3: Verify on main**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected: all pass on `main`.

- [ ] **Step 4: Push main and clean local branch**

Run:

```powershell
git push origin main
git branch -d paper11-phase3-drl-input-adapter
```

Expected: `main` is synchronized with `origin/main`; the local feature branch is deleted after merge.

---

## Self-Review

- Spec coverage: The plan covers loader API, validation rules, CLI, docs, tests, and verification.
- Placeholder scan: No `TBD`/`TODO` placeholders are present.
- Type consistency: `VariantInput`, `load_variant_input`, `feature_columns`, `state_matrix`, `reward_mode`, `state_groups`, and `source_table` are consistent across tasks.
