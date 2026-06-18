# Phase 27 B0/B1 Stability Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 27 diagnostic package that compares the existing 1024-step and 4096-step Phase 26 B0/B1 learned-policy artifacts.

**Architecture:** Add a focused `phase27_stability_diagnosis` module that reads Phase 26 comparison JSON files, pairs tile-seed delta rows across budgets, classifies stability, and writes CSV/JSON/Markdown artifacts. Add a thin experiment runner plus synthetic pytest coverage and reviewer-facing docs.

**Tech Stack:** Python standard library (`csv`, `json`, `argparse`, `pathlib`, `statistics`), pytest, existing Paper11 repository layout.

---

## File Structure

- Create `src/paper11_geofm/phase27_stability_diagnosis.py`: pure analysis and artifact-writing functions.
- Create `experiments/phase27_stability_diagnosis/run_phase27_stability_diagnosis.py`: CLI wrapper around the analysis module.
- Create `tests/test_phase27_stability_diagnosis.py`: synthetic fixture tests and CLI tests.
- Create `paper/phase27_results/README.md`: index for Phase 27 interpretation package.
- Create `paper/phase27_results/01_phase27_stability_diagnosis.md`: reviewer-facing interpretation of Phase 27 outputs.
- Modify `README.md`: add Phase 27 runner, scope, and key entry point.
- Modify `reproducibility/REPRODUCTION_GUIDE.md`: add Phase 27 reproduction section and executable files list.
- Modify `reproducibility/FILE_MANIFEST.tsv`: add Phase 27 spec, plan, module, runner, tests, and results docs.
- Modify `paper/phase26_results/02_next_experiment_matrix.md`: record that Phase 27 is the immediate next diagnostic package.
- Modify `paper/submission/01_ijaeog_submission_readiness.md`: update readiness summary and recommendation with Phase 27 diagnostics.

### Task 1: Write Phase 27 Failing Tests

**Files:**
- Create: `tests/test_phase27_stability_diagnosis.py`

- [ ] **Step 1: Add synthetic fixture helpers and analyzer tests**

```python
import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_phase26_comparison(
    path: Path,
    *,
    timesteps: int,
    deltas: dict[tuple[str, int], float],
    claim_status: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_count = sum(1 for value in deltas.values() if value > 0)
    payload = {
        "phase": "phase26_main_empirical_experiment",
        "train_timesteps": timesteps,
        "eval_max_steps": 8,
        "eval_tile_ids": sorted({tile_id for tile_id, _seed in deltas}),
        "seeds": sorted({seed for _tile_id, seed in deltas}),
        "phase26_claim_status": claim_status,
        "learned_policy": {
            "B1_minus_B0_mean_reward": round(sum(deltas.values()) / len(deltas), 10),
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": len(deltas),
            "positive_fraction": round(positive_count / len(deltas), 10),
        },
        "tile_seed_delta_rows": [
            {
                "eval_tile_id": tile_id,
                "seed": seed,
                "b0_reward": 0.0,
                "b1_reward": value,
                "b1_minus_b0_reward": value,
                "b1_improves_b0": value > 0,
                "train_timesteps": timesteps,
                "eval_max_steps": 8,
            }
            for (tile_id, seed), value in sorted(deltas.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_phase27_builds_budget_transition_and_stability_classes(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={
            ("tile_a", 0): 0.2,
            ("tile_a", 1): -0.2,
            ("tile_b", 0): -0.3,
            ("tile_b", 1): 0.4,
        },
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={
            ("tile_a", 0): 0.1,
            ("tile_a", 1): -0.1,
            ("tile_b", 0): 0.2,
            ("tile_b", 1): -0.5,
        },
    )

    analysis = build_phase27_stability_diagnosis([lower, higher])

    assert analysis["phase"] == "phase27_b0_b1_stability_diagnosis"
    assert analysis["phase27_diagnostic_status"] == "budget_not_explanatory"
    assert analysis["budget_transition_rows"][1]["mean_delta_change_from_previous"] == 0.1
    assert analysis["budget_transition_rows"][1]["positive_count_change_from_previous"] == 0
    assert analysis["stability_counts"] == {
        "stable_positive": 1,
        "stable_negative": 1,
        "flip_to_positive": 1,
        "flip_to_negative": 1,
        "incomplete": 0,
    }
    classes = {
        (row["eval_tile_id"], row["seed"]): row["stability_class"]
        for row in analysis["tile_seed_stability_rows"]
    }
    assert classes == {
        ("tile_a", 0): "stable_positive",
        ("tile_a", 1): "stable_negative",
        ("tile_b", 0): "flip_to_positive",
        ("tile_b", 1): "flip_to_negative",
    }


def test_phase27_reports_insufficient_for_unpaired_tile_seed_rows(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1},
    )

    analysis = build_phase27_stability_diagnosis([lower, higher])

    assert analysis["phase27_diagnostic_status"] == "insufficient"
    assert analysis["stability_counts"]["incomplete"] == 1
    assert analysis["tile_seed_stability_rows"][1]["stability_class"] == "incomplete"


def test_phase27_writer_outputs_csv_json_and_markdown(tmp_path):
    from paper11_geofm.phase27_stability_diagnosis import (
        build_phase27_stability_diagnosis,
        write_phase27_stability_diagnosis_artifacts,
    )

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1, ("tile_b", 0): -0.3},
    )
    analysis = build_phase27_stability_diagnosis([lower, higher])
    paths = write_phase27_stability_diagnosis_artifacts(analysis, tmp_path / "outputs")

    assert paths["budget_transition_csv"].name == "phase27_budget_transition_table.csv"
    assert paths["tile_seed_stability_csv"].name == "phase27_tile_seed_stability.csv"
    assert paths["diagnostic_summary_json"].name == "phase27_diagnostic_summary.json"
    assert paths["diagnostic_readiness_md"].name == "phase27_diagnostic_readiness.md"
    transition_rows = list(csv.DictReader(paths["budget_transition_csv"].open("r", encoding="utf-8")))
    assert transition_rows[0]["train_timesteps"] == "1024"
    summary = json.loads(paths["diagnostic_summary_json"].read_text(encoding="utf-8"))
    assert summary["phase27_diagnostic_status"] == "budget_not_explanatory"
    markdown = paths["diagnostic_readiness_md"].read_text(encoding="utf-8")
    assert "budget_not_explanatory" in markdown
    assert "GeoFM improves planning decisions" not in markdown


def test_phase27_cli_writes_outputs(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase27_stability_diagnosis"
        / "run_phase27_stability_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase27_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    lower = _write_phase26_comparison(
        tmp_path / "lower" / "phase26_main_comparison.json",
        timesteps=1024,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.2, ("tile_b", 0): 0.1},
    )
    higher = _write_phase26_comparison(
        tmp_path / "higher" / "phase26_main_comparison.json",
        timesteps=4096,
        claim_status="not_supported",
        deltas={("tile_a", 0): -0.1, ("tile_b", 0): -0.3},
    )

    exit_code = module.main(
        [
            "--phase26-comparison-json",
            str(lower),
            "--phase26-comparison-json",
            str(higher),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Phase 27 diagnostic status: budget_not_explanatory" in stdout
    assert "phase27_diagnostic_summary.json" in stdout


def test_phase27_cli_reports_missing_input(tmp_path, capsys):
    runner_path = (
        ROOT
        / "experiments"
        / "phase27_stability_diagnosis"
        / "run_phase27_stability_diagnosis.py"
    )
    spec = importlib.util.spec_from_file_location("phase27_runner_error", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--phase26-comparison-json",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "Error:" in stderr
```

- [ ] **Step 2: Run the Phase 27 tests and confirm RED**

Run:

```powershell
python -m pytest tests\test_phase27_stability_diagnosis.py -q
```

Expected: fail because `paper11_geofm.phase27_stability_diagnosis` and the runner do not exist yet.

### Task 2: Implement Analyzer and Runner

**Files:**
- Create: `src/paper11_geofm/phase27_stability_diagnosis.py`
- Create: `experiments/phase27_stability_diagnosis/run_phase27_stability_diagnosis.py`

- [ ] **Step 1: Create `phase27_stability_diagnosis.py`**

Implement:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import statistics
from pathlib import Path
```

Public constants:

```python
PHASE27_CLAIM_BOUNDARY = (
    "Phase 27 is a read-only diagnosis of existing Phase 26 B0/B1 padded "
    "held-out Bishan learned-policy artifacts; it does not run new training, "
    "does not enable suitability reward, does not test B2/B3, and does not "
    "support cross-region transfer or final submission-level claims."
)

PHASE27_REMAINING_EVIDENCE_GAPS = [
    "representation_controls_against_random_shuffled_pca_features",
    "repeated_or_intermediate_budget_stability_sweep",
    "suitability_proxy_validation_before_reward_use",
    "held_out_region_transfer_evaluation",
    "spatial_case_maps_and_uncertainty",
]
```

Public functions:

```python
def build_phase27_stability_diagnosis(
    phase26_comparison_json_paths: Sequence[Path | str],
) -> dict[str, object]:
    ...

def write_phase27_stability_diagnosis_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    ...
```

Implementation details:

- read each JSON object and require `tile_seed_delta_rows`;
- sort budgets by `train_timesteps`;
- build transition rows with previous-budget change fields;
- pair only the first and last budgets for stability classification;
- mark any missing pair as `incomplete`;
- set `phase27_diagnostic_status` using spec rules;
- write the four artifacts with exact filenames from the spec.

- [ ] **Step 2: Create runner**

Implement a CLI with:

```text
--phase26-comparison-json
--output-dir
```

`--phase26-comparison-json` should be repeatable and require at least two paths.
Print status and artifact paths. Return `1` on `OSError`, `ValueError`, or
`RuntimeError`.

- [ ] **Step 3: Run the Phase 27 tests and confirm GREEN**

Run:

```powershell
python -m pytest tests\test_phase27_stability_diagnosis.py -q
```

Expected: all tests pass.

### Task 3: Run Phase 27 on Current macOS Artifacts

**Files:**
- Generated, ignored by Git: `experiments/phase27_stability_diagnosis/outputs/macos_1024_vs_4096/*`

- [ ] **Step 1: Run diagnostic CLI**

Run:

```powershell
python experiments\phase27_stability_diagnosis\run_phase27_stability_diagnosis.py --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main\phase26_analysis\phase26_main_comparison.json --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main_4096\phase26_analysis\phase26_main_comparison.json --output-dir experiments\phase27_stability_diagnosis\outputs\macos_1024_vs_4096
```

Expected:

- status is `budget_not_explanatory`;
- output paths are printed;
- no RL training runs.

- [ ] **Step 2: Inspect generated summary**

Run:

```powershell
Get-Content -Raw experiments\phase27_stability_diagnosis\outputs\macos_1024_vs_4096\phase27_diagnostic_summary.json
```

Expected key facts:

- lower mean delta `-0.4329022862`;
- higher mean delta `-0.1318712688`;
- mean change `0.3010310174`;
- positive count change `-1`;
- status `budget_not_explanatory`.

### Task 4: Update Reviewer-Facing Documentation

**Files:**
- Create: `paper/phase27_results/README.md`
- Create: `paper/phase27_results/01_phase27_stability_diagnosis.md`
- Modify: `README.md`
- Modify: `reproducibility/REPRODUCTION_GUIDE.md`
- Modify: `reproducibility/FILE_MANIFEST.tsv`
- Modify: `paper/phase26_results/02_next_experiment_matrix.md`
- Modify: `paper/submission/01_ijaeog_submission_readiness.md`

- [ ] **Step 1: Add Phase 27 results package**

Create `paper/phase27_results/README.md` and
`paper/phase27_results/01_phase27_stability_diagnosis.md` with the generated
diagnostic facts and conservative interpretation.

- [ ] **Step 2: Update README and reproduction guide**

Add Phase 27 as a read-only diagnostic runner after Phase 26, including the
exact command and outputs.

- [ ] **Step 3: Update manifest and submission readiness**

Add new file rows to `FILE_MANIFEST.tsv`. Update submission readiness so Phase
27 is diagnostic evidence and manuscript readiness remains `not_ready`.

### Task 5: Verification and Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_phase27_stability_diagnosis.py -q
```

Expected: all Phase 27 tests pass.

- [ ] **Step 2: Run existing Phase 26 tests**

Run:

```powershell
python -m pytest tests\test_phase26_main_experiment.py -q
```

Expected: all Phase 26 tests pass.

- [ ] **Step 3: Run smoke check**

Run:

```powershell
python scripts\smoke_check.py
```

Expected: smoke check passes.

- [ ] **Step 4: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Inspect git diff**

Run:

```powershell
git status --short
git diff --stat
```

Expected: only Phase 27-related files changed.

- [ ] **Step 6: Commit and push**

Run:

```powershell
git add docs/superpowers/specs/2026-06-18-phase27-b0-b1-stability-diagnosis-design.md docs/superpowers/plans/2026-06-18-phase27-b0-b1-stability-diagnosis.md src/paper11_geofm/phase27_stability_diagnosis.py experiments/phase27_stability_diagnosis/run_phase27_stability_diagnosis.py tests/test_phase27_stability_diagnosis.py paper/phase27_results/README.md paper/phase27_results/01_phase27_stability_diagnosis.md README.md reproducibility/REPRODUCTION_GUIDE.md reproducibility/FILE_MANIFEST.tsv paper/phase26_results/02_next_experiment_matrix.md paper/submission/01_ijaeog_submission_readiness.md
git commit -m "feat: add Phase 27 stability diagnosis"
git push
```

Expected: commit and push succeed.

## Self-Review

- Spec coverage: every Phase 27 design output, status rule, test requirement,
  and documentation update is mapped to a task.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: public function names, file paths, and artifact names are
  consistent across tasks.
