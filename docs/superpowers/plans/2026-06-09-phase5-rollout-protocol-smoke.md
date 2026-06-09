# Phase 5 Rollout Protocol Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic masked-rollout smoke protocol that runs ready B0/B1/B2/B3 Phase 2 variants through the Phase 4 Gymnasium environment and writes comparable CSV/JSON artifacts.

**Architecture:** Add a focused `paper11_geofm.rollout_smoke` module that owns the rollout loop and artifact writing. Add a CLI under `experiments/phase5_rollout_protocol/` that calls the module and prints one concise summary line per variant. Keep Phase 5 separate from real training and real planning evaluation.

**Tech Stack:** Python, NumPy, Gymnasium, csv/json standard library, pytest, existing Phase 4 `make_phase4_smoke_env()`.

---

## File Structure

- Create `src/paper11_geofm/rollout_smoke.py`: Phase 5 claim boundary, deterministic rollout protocol, CSV/JSON artifact writer.
- Create `experiments/phase5_rollout_protocol/run_phase5_rollout.py`: CLI runner.
- Create `tests/test_phase5_rollout_smoke.py`: protocol, artifact, max-step, validation, and CLI tests.
- Modify `README.md`: add Phase 5 command after Phase 4.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 5 expected output and artifacts after Phase 4.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 5 design, plan, module, CLI, and test rows.

## Task 1: Protocol Summary Contract

**Files:**
- Create: `tests/test_phase5_rollout_smoke.py`
- Create: `src/paper11_geofm/rollout_smoke.py`

- [ ] **Step 1: Write the failing protocol test**

Create `tests/test_phase5_rollout_smoke.py` with this initial content:

```python
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_phase2_feature_row(block_id, suitability):
    row = {"block_id": block_id, "suitability_proxy": suitability}
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

    rows = [
        _complete_phase2_feature_row("sample_block_00", 0.25),
        _complete_phase2_feature_row("sample_block_01", 0.50),
        _complete_phase2_feature_row("sample_block_02", 0.75),
        _complete_phase2_feature_row("sample_block_03", 1.00),
    ]
    return write_phase2_artifacts(rows, output_dir, _phase2_test_summary())


def _summaries_by_variant(protocol):
    return {row["variant_id"]: row for row in protocol["summaries"]}


def test_phase5_rollout_protocol_runs_all_ready_variants(tmp_path):
    from paper11_geofm.rollout_smoke import (
        PHASE5_CLAIM_BOUNDARY,
        run_phase5_rollout_protocol,
    )

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase5_rollout_protocol(tmp_path)

    assert protocol["claim_boundary"] == PHASE5_CLAIM_BOUNDARY
    assert protocol["variant_ids"] == ["B0", "B1", "B2", "B3"]
    summaries = _summaries_by_variant(protocol)
    assert set(summaries) == {"B0", "B1", "B2", "B3"}
    assert summaries["B0"]["n_features"] == 17
    assert summaries["B1"]["n_features"] == 81
    assert summaries["B2"]["n_features"] == 18
    assert summaries["B3"]["n_features"] == 82
    assert summaries["B3"]["observation_shape"] == 331
    assert summaries["B3"]["action_space_n"] == 4

    for summary in summaries.values():
        assert summary["n_blocks"] == 4
        assert summary["episode_steps"] == 4
        assert summary["max_steps"] == 4
        assert summary["terminated"] is True
        assert summary["truncated"] is False
        assert summary["valid_action_rate"] == 1.0
        assert summary["selected_block_ids"] == [
            "sample_block_00",
            "sample_block_01",
            "sample_block_02",
            "sample_block_03",
        ]
        assert summary["claim_boundary"] == PHASE5_CLAIM_BOUNDARY

    assert summaries["B0"]["total_contract_reward"] == 0.0
    assert summaries["B1"]["total_contract_reward"] == 0.0
    assert summaries["B2"]["total_contract_reward"] == 2.5
    assert summaries["B3"]["total_contract_reward"] == 2.5
    assert protocol["steps"]["B3"][0]["selected_block_id"] == "sample_block_00"
    assert protocol["steps"]["B3"][0]["valid_actions_before"] == 4
    assert protocol["steps"]["B3"][0]["valid_actions_after"] == 3
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_protocol_runs_all_ready_variants -q
```

Expected: fail with `ModuleNotFoundError: No module named 'paper11_geofm.rollout_smoke'`.

- [ ] **Step 3: Implement the rollout module**

Create `src/paper11_geofm/rollout_smoke.py` with:

```python
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .drl_smoke_env import make_phase4_smoke_env


PHASE5_CLAIM_BOUNDARY = (
    "Phase 5 is a deterministic rollout-protocol smoke check; it does not "
    "train or evaluate a policy and does not report planning performance."
)

SUMMARY_FIELDNAMES = [
    "variant_id",
    "n_blocks",
    "n_features",
    "observation_shape",
    "action_space_n",
    "reward_mode",
    "max_steps",
    "episode_steps",
    "terminated",
    "truncated",
    "valid_action_rate",
    "total_contract_reward",
    "selected_block_ids",
    "claim_boundary",
]


def run_phase5_rollout_protocol(
    phase2_output_dir: Path | str,
    variant_ids: Sequence[str] = ("B0", "B1", "B2", "B3"),
    max_steps: int | None = None,
) -> dict[str, object]:
    normalized_variant_ids = _normalize_variant_ids(variant_ids)
    summaries: list[dict[str, object]] = []
    steps_by_variant: dict[str, list[dict[str, object]]] = {}

    for variant_id in normalized_variant_ids:
        env = make_phase4_smoke_env(
            phase2_output_dir,
            variant_id,
            max_steps=max_steps,
        )
        obs, info = env.reset()
        steps: list[dict[str, object]] = []
        selected_block_ids: list[str] = []
        total_contract_reward = 0.0
        valid_attempts = 0
        terminated = False
        truncated = False

        while True:
            mask = env.action_masks()
            valid_actions = [
                int(index)
                for index, valid in enumerate(mask.tolist())
                if bool(valid)
            ]
            if not valid_actions:
                break

            action = valid_actions[0]
            valid_actions_before = len(valid_actions)
            _, reward, terminated, truncated, step_info = env.step(action)
            reward_value = float(reward)
            selected_block_id = str(step_info["selected_block_id"])
            selected_block_ids.append(selected_block_id)
            total_contract_reward += reward_value
            valid_attempts += 1
            valid_actions_after = int(env.action_masks().sum())
            steps.append(
                {
                    "step": int(step_info["step"]),
                    "action": action,
                    "selected_block_id": selected_block_id,
                    "reward": _round_float(reward_value),
                    "valid_actions_before": valid_actions_before,
                    "valid_actions_after": valid_actions_after,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if terminated or truncated:
                break

        episode_steps = len(steps)
        valid_action_rate = (
            float(valid_attempts / episode_steps) if episode_steps else 0.0
        )
        summary = {
            "variant_id": variant_id,
            "n_blocks": int(info["n_blocks"]),
            "n_features": int(info["n_features"]),
            "observation_shape": int(obs.shape[0]),
            "action_space_n": int(env.action_space.n),
            "reward_mode": str(info["reward_mode"]),
            "max_steps": int(env.max_steps),
            "episode_steps": episode_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "valid_action_rate": _round_float(valid_action_rate),
            "total_contract_reward": _round_float(total_contract_reward),
            "selected_block_ids": selected_block_ids,
            "claim_boundary": PHASE5_CLAIM_BOUNDARY,
        }
        summaries.append(summary)
        steps_by_variant[variant_id] = steps

    return {
        "phase": "phase5_rollout_protocol_smoke",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "variant_ids": normalized_variant_ids,
        "max_steps_requested": max_steps,
        "claim_boundary": PHASE5_CLAIM_BOUNDARY,
        "summaries": summaries,
        "steps": steps_by_variant,
    }


def write_phase5_rollout_artifacts(
    protocol: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "phase5_rollout_summary.csv"
    steps_path = output_path / "phase5_rollout_steps.json"

    summaries = protocol.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Phase 5 protocol is missing a summaries list")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                raise ValueError("Phase 5 summary rows must be objects")
            row = {field: summary.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = row.get("selected_block_ids")
            if isinstance(selected, list):
                row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(row)

    steps_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary_csv": summary_path, "steps_json": steps_path}


def _normalize_variant_ids(variant_ids: Sequence[str]) -> list[str]:
    normalized = [str(variant_id).strip().upper() for variant_id in variant_ids]
    normalized = [variant_id for variant_id in normalized if variant_id]
    if not normalized:
        raise ValueError("At least one Phase 5 variant must be requested")
    return normalized


def _round_float(value: float) -> float:
    return round(float(value), 10)
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_protocol_runs_all_ready_variants -q
```

Expected: `1 passed`.

## Task 2: Max-Step and Validation Behavior

**Files:**
- Modify: `tests/test_phase5_rollout_smoke.py`
- Modify: `src/paper11_geofm/rollout_smoke.py`

- [ ] **Step 1: Add max-step and empty-variant tests**

Append these tests to `tests/test_phase5_rollout_smoke.py`:

```python
def test_phase5_rollout_respects_max_steps(tmp_path):
    from paper11_geofm.rollout_smoke import run_phase5_rollout_protocol

    _write_ready_phase2_outputs(tmp_path)

    protocol = run_phase5_rollout_protocol(tmp_path, variant_ids=("B3",), max_steps=2)

    summary = protocol["summaries"][0]
    assert summary["variant_id"] == "B3"
    assert summary["max_steps"] == 2
    assert summary["episode_steps"] == 2
    assert summary["terminated"] is True
    assert summary["selected_block_ids"] == ["sample_block_00", "sample_block_01"]
    assert summary["total_contract_reward"] == 0.75
    assert len(protocol["steps"]["B3"]) == 2


def test_phase5_rollout_rejects_empty_variant_list(tmp_path):
    import pytest

    from paper11_geofm.rollout_smoke import run_phase5_rollout_protocol

    _write_ready_phase2_outputs(tmp_path)

    with pytest.raises(ValueError, match="At least one"):
        run_phase5_rollout_protocol(tmp_path, variant_ids=())
```

- [ ] **Step 2: Run the new tests**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_respects_max_steps tests\test_phase5_rollout_smoke.py::test_phase5_rollout_rejects_empty_variant_list -q
```

Expected: both tests pass with the Task 1 implementation.

## Task 3: Artifact Writer

**Files:**
- Modify: `tests/test_phase5_rollout_smoke.py`
- Modify: `src/paper11_geofm/rollout_smoke.py`

- [ ] **Step 1: Add artifact writer test**

Append this test:

```python
def test_phase5_rollout_artifacts_are_written(tmp_path):
    import csv

    from paper11_geofm.rollout_smoke import (
        PHASE5_CLAIM_BOUNDARY,
        run_phase5_rollout_protocol,
        write_phase5_rollout_artifacts,
    )

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase5"
    _write_ready_phase2_outputs(phase2_dir)
    protocol = run_phase5_rollout_protocol(phase2_dir, variant_ids=("B2", "B3"))

    paths = write_phase5_rollout_artifacts(protocol, output_dir)

    assert paths["summary_csv"].name == "phase5_rollout_summary.csv"
    assert paths["steps_json"].name == "phase5_rollout_steps.json"
    assert paths["summary_csv"].exists()
    assert paths["steps_json"].exists()

    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant_id"] for row in rows] == ["B2", "B3"]
    assert rows[0]["selected_block_ids"] == (
        "sample_block_00;sample_block_01;sample_block_02;sample_block_03"
    )
    assert rows[0]["claim_boundary"] == PHASE5_CLAIM_BOUNDARY

    saved = json.loads(paths["steps_json"].read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == PHASE5_CLAIM_BOUNDARY
    assert saved["variant_ids"] == ["B2", "B3"]
    assert saved["steps"]["B3"][3]["selected_block_id"] == "sample_block_03"
```

- [ ] **Step 2: Run artifact test**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_artifacts_are_written -q
```

Expected: pass with the Task 1 writer.

## Task 4: CLI Runner

**Files:**
- Modify: `tests/test_phase5_rollout_smoke.py`
- Create: `experiments/phase5_rollout_protocol/run_phase5_rollout.py`

- [ ] **Step 1: Add failing CLI test**

Add imports at the top of `tests/test_phase5_rollout_smoke.py`:

```python
import importlib.util
```

Append this test:

```python
def test_phase5_rollout_cli_prints_summary_and_artifacts(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_phase5_rollout",
        ROOT / "experiments" / "phase5_rollout_protocol" / "run_phase5_rollout.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase5"
    _write_ready_phase2_outputs(phase2_dir)

    exit_code = module.main(
        [
            "--phase2-output-dir",
            str(phase2_dir),
            "--output-dir",
            str(output_dir),
            "--variants",
            "B0,B3",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Variant B0: steps=4 features=17 total_contract_reward=0.000000" in stdout
    assert "Variant B3: steps=4 features=82 total_contract_reward=2.500000" in stdout
    assert "Summary CSV:" in stdout
    assert "Steps JSON:" in stdout
    assert "Claim boundary: Phase 5 is a deterministic rollout-protocol smoke check" in stdout
    assert (output_dir / "phase5_rollout_summary.csv").exists()
    assert (output_dir / "phase5_rollout_steps.json").exists()
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_cli_prints_summary_and_artifacts -q
```

Expected: fail with missing `experiments/phase5_rollout_protocol/run_phase5_rollout.py`.

- [ ] **Step 3: Implement CLI**

Create `experiments/phase5_rollout_protocol/run_phase5_rollout.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.rollout_smoke import (
    PHASE5_CLAIM_BOUNDARY,
    run_phase5_rollout_protocol,
    write_phase5_rollout_artifacts,
)


def _parse_variants(text: str) -> list[str]:
    return [part.strip().upper() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic Paper11 Phase 5 masked-rollout protocol "
            "smoke check without training a policy."
        )
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        required=True,
        help="Directory containing experiment_variants.json and variant CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Phase 5 summary artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        default="B0,B1,B2,B3",
        help="Comma-separated variant IDs. Default: B0,B1,B2,B3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum steps per variant rollout.",
    )
    args = parser.parse_args(argv)

    try:
        protocol = run_phase5_rollout_protocol(
            args.phase2_output_dir,
            variant_ids=_parse_variants(args.variants),
            max_steps=args.max_steps,
        )
        paths = write_phase5_rollout_artifacts(protocol, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for summary in protocol["summaries"]:
        print(
            "Variant "
            f"{summary['variant_id']}: "
            f"steps={summary['episode_steps']} "
            f"features={summary['n_features']} "
            f"total_contract_reward={float(summary['total_contract_reward']):.6f}"
        )
    print(f"Summary CSV: {paths['summary_csv']}")
    print(f"Steps JSON: {paths['steps_json']}")
    print(f"Claim boundary: {PHASE5_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py::test_phase5_rollout_cli_prints_summary_and_artifacts -q
```

Expected: `1 passed`.

## Task 5: Documentation and Manifest

**Files:**
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Test: `tests/test_phase5_rollout_smoke.py`

- [ ] **Step 1: Update README**

After the Phase 4 command block in `README.md`, add:

````markdown
Run the Phase 5 deterministic masked-rollout protocol smoke check:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture
python experiments\phase5_rollout_protocol\run_phase5_rollout.py --phase2-output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture --output-dir experiments\phase5_rollout_protocol\outputs\phase5_protocol --variants B0,B1,B2,B3
```

This command runs the same deterministic masked rollout protocol across ready variants and writes `phase5_rollout_summary.csv` and `phase5_rollout_steps.json`. It does not train or evaluate a policy and does not report planning performance.
````

Add this key entry point:

```markdown
- Phase 5 rollout protocol smoke runner: `experiments/phase5_rollout_protocol/run_phase5_rollout.py`
```

- [ ] **Step 2: Update reproduction guide**

After the Phase 4 section in `reproducibility/REPRODUCTION_GUIDE.md`, add a new section titled:

```markdown
## 7. Run the Phase 5 Rollout Protocol Smoke Check
```

Include these commands:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture
python experiments\phase5_rollout_protocol\run_phase5_rollout.py --phase2-output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture --output-dir experiments\phase5_rollout_protocol\outputs\phase5_protocol --variants B0,B1,B2,B3
```

State that expected artifacts are `phase5_rollout_summary.csv` and
`phase5_rollout_steps.json`; feature counts for the fixture are B0 = 17,
B1 = 81, B2 = 18, B3 = 82; B0/B1 have zero contract reward; B2/B3 have positive
contract reward from `suitability_proxy`; no training or planning performance is
reported. Renumber later sections by adding 1 to their current numbers.

- [ ] **Step 3: Update file manifest**

Add these rows to `reproducibility/FILE_MANIFEST.tsv`:

```text
docs/superpowers/specs/2026-06-09-phase5-rollout-protocol-smoke-design.md	design	Phase 5 design for deterministic masked-rollout protocol smoke artifacts without policy training.
docs/superpowers/plans/2026-06-09-phase5-rollout-protocol-smoke.md	plan	Implementation plan for the Phase 5 rollout protocol smoke check.
src/paper11_geofm/rollout_smoke.py	runtime	Phase 5 deterministic masked-rollout protocol and CSV/JSON artifact writer.
experiments/phase5_rollout_protocol/run_phase5_rollout.py	experiment	Executable Phase 5 rollout protocol smoke runner across ready variant tables.
tests/test_phase5_rollout_smoke.py	verification	Pytest checks for Phase 5 rollout summaries, max-step behavior, artifacts, validation, and CLI output.
```

- [ ] **Step 4: Run Phase 5 tests**

Run:

```powershell
python -m pytest tests\test_phase5_rollout_smoke.py -q
```

Expected: all Phase 5 tests pass.

## Task 6: Verification and Commit

**Files:**
- All files touched in Tasks 1-5

- [ ] **Step 1: Regenerate Phase 2 fixture output**

Run:

```powershell
python experiments\phase2_block_geofm_features\run_phase2.py --mapping-csv data\bishan_phase2_csv_sample\block_pixel_mapping.csv --attributes-csv data\bishan_phase2_csv_sample\block_attributes.csv --output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture
```

Expected: B0/B1/B2/B3 variant feature CSVs and weak-label validation are written.

- [ ] **Step 2: Run Phase 5 CLI**

Run:

```powershell
python experiments\phase5_rollout_protocol\run_phase5_rollout.py --phase2-output-dir experiments\phase5_rollout_protocol\outputs\phase2_fixture --output-dir experiments\phase5_rollout_protocol\outputs\phase5_protocol --variants B0,B1,B2,B3
```

Expected output contains:

```text
Variant B0: steps=4 features=17 total_contract_reward=0.000000
Variant B1: steps=4 features=81 total_contract_reward=0.000000
Variant B2: steps=4 features=18 total_contract_reward=
Variant B3: steps=4 features=82 total_contract_reward=
Summary CSV:
Steps JSON:
Claim boundary: Phase 5 is a deterministic rollout-protocol smoke check
```

- [ ] **Step 3: Run repository verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected: smoke check passes, all tests pass, and diff check reports no whitespace errors.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add README.md reproducibility\REPRODUCTION_GUIDE.md reproducibility\FILE_MANIFEST.tsv src\paper11_geofm\rollout_smoke.py experiments\phase5_rollout_protocol\run_phase5_rollout.py tests\test_phase5_rollout_smoke.py
git commit -m "Add Phase 5 rollout protocol smoke check"
```

Expected: one implementation commit after the Phase 5 design and plan commits.

## Task 7: Merge and Push

**Files:**
- Git branch state only

- [ ] **Step 1: Push feature branch**

Run:

```powershell
git push -u origin paper11-phase5-rollout-protocol-smoke
```

- [ ] **Step 2: Fast-forward merge to main**

Run:

```powershell
git checkout main
git merge --ff-only paper11-phase5-rollout-protocol-smoke
```

- [ ] **Step 3: Re-run main verification**

Run:

```powershell
python scripts\smoke_check.py
python -m pytest tests
git diff --check
```

Expected: smoke check passes, all tests pass, and diff check reports no whitespace errors on `main`.

- [ ] **Step 4: Push main and delete local feature branch**

Run:

```powershell
git push origin main
git branch -d paper11-phase5-rollout-protocol-smoke
```

Expected: `origin/main` contains the Phase 5 design, plan, and implementation commits; local branch cleanup succeeds.

---

## Self-Review

- Spec coverage: the plan covers the Phase 5 protocol loop, artifact outputs, CLI, documentation, verification, and merge.
- Red-flag scan: no task uses unspecified implementation gaps.
- Type consistency: public function names, artifact filenames, summary fields, CLI arguments, and test expectations match the design spec.
