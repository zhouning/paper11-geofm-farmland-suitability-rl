# Phase 69 Label-Free Evidence Synthesis Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Phase 69 synthesis gate that reports the strongest defensible label-free Paper11 algorithm claim from existing cross-phase evidence.

**Architecture:** Add one focused module that loads existing JSON result artifacts, normalizes them into evidence-axis rows and claim-boundary rows, computes a conservative top-level status, and writes CSV/JSON/Markdown artifacts. Phase 60 is the canonical route-evidence input because it already records the Phase 48/52/53 compressed-route and cluster-support axes used by this synthesis gate. A thin CLI runner will call the module; a paper-facing result note will record the real-run status without modifying formal manuscript files.

**Tech Stack:** Python 3 standard library (`csv`, `json`, `math`, `pathlib`, `argparse`), pytest, existing Paper11 phase artifact conventions, PowerShell commands using `D:\adk\.venv\Scripts\python.exe`.

---

## File Structure

- Create `src/paper11_geofm/phase69_label_free_evidence_synthesis_gate.py`
  - Owns the Phase 69 claim boundary, JSON loading, evidence-axis normalization, claim-boundary matrix, synthesis status, Markdown rendering, and artifact writing.
- Create `experiments/phase69_label_free_evidence_synthesis_gate/run_phase69_label_free_evidence_synthesis_gate.py`
  - Thin CLI wrapper. It accepts Phase 60, 57, 59, 62, 66, 67, and 68 JSON paths plus output directory. Phase 60 is used instead of raw Phase 48/52/53 paths because its `attribution_axes` rows carry the compressed-route, cluster-robustness, and matched-control summaries needed here.
- Create `tests/test_phase69_label_free_evidence_synthesis_gate.py`
  - Covers current-style fixture evidence, missing route support, blocked stronger claims, writer output, and CLI behavior.
- Create `paper/phase28_results/35_phase69_label_free_evidence_synthesis_gate.md`
  - Filled after the real run. It reports status, key evidence axes, allowed bounded claim, blocked stronger claims, reproduction command, and claim boundary.
- Modify `paper/phase28_results/README.md`
  - Add a one-line entry for the Phase 69 result note.
- Do not modify `paper/submission/final/*`.

All pytest commands in this plan use `--basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider` because this repository currently has an old `.pytest_tmp` directory that can be undeletable on Windows.

---

### Task 1: Core Synthesis Gate And Artifact Writer

**Files:**
- Create: `tests/test_phase69_label_free_evidence_synthesis_gate.py`
- Create: `src/paper11_geofm/phase69_label_free_evidence_synthesis_gate.py`

- [ ] **Step 1: Write failing tests for current-style evidence synthesis**

Create `tests/test_phase69_label_free_evidence_synthesis_gate.py`:

```python
import csv
import json
import pytest
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    phase60 = _write_json(
        tmp_path / "phase60.json",
        {
            "phase": "phase60_information_optimization_attribution",
            "phase60_attribution_status": "mechanism_claim_narrowed",
            "attribution_axes": [
                {
                    "axis_id": "compressed_route_performance",
                    "axis_status": "supported",
                    "primary_metric": "pooled_mean_delta",
                    "primary_value": 0.2921767818,
                    "source_phase": "phase48_phase52_expanded",
                    "interpretation": "Compressed route performance is supported.",
                },
                {
                    "axis_id": "cluster_level_robustness",
                    "axis_status": "supported",
                    "primary_metric": "cluster_mean_delta",
                    "primary_value": 0.2921767818,
                    "source_phase": "phase53",
                    "interpretation": "Cluster-level support is present.",
                },
                {
                    "axis_id": "geofm_specific_matched_dimension",
                    "axis_status": "not_supported",
                    "primary_metric": "pooled_matched_control_mean_delta",
                    "primary_value": -0.0172307641,
                    "source_phase": "phase59",
                    "interpretation": "GeoFM-specific matched-dimension advantage is not supported.",
                },
            ],
        },
    )
    phase57 = _write_json(
        tmp_path / "phase57.json",
        {
            "phase": "phase57_compressed_representation_mechanism",
            "phase57_mechanism_status": "compressed_geometry_consistent",
            "geometry_rows": [
                {"variant_id": "B1", "effective_rank": 9.4947211626, "condition_number": 6658.9542931381},
                {"variant_id": "D4P8", "effective_rank": 5.1322783588, "variance_retention_vs_b1": 0.8587823898},
                {"variant_id": "D4P16", "effective_rank": 7.3009059917, "variance_retention_vs_b1": 0.9496006154},
            ],
        },
    )
    phase59 = _write_json(
        tmp_path / "phase59.json",
        {
            "phase": "phase59_matched_dimension_controls",
            "phase59_matched_dimension_status": "matched_dimension_geofm_not_supported",
            "pooled_matched_control_delta": {"mean_delta": -0.0172307641},
        },
    )
    phase62 = _write_json(
        tmp_path / "phase62.json",
        {
            "phase": "phase62_d4_d6_matched_ppo",
            "phase62_d4_d6_status": "d6_random_projection_advantage",
            "pooled_primary_delta": {"mean_delta": -0.0641900514},
        },
    )
    phase66 = _write_json(
        tmp_path / "phase66.json",
        {
            "phase": "phase66_reward_label_representation_audit",
            "phase66_status": "base_reward_target_masks_geofm_signal",
            "diagnostic_gate": {
                "phase66_status": "base_reward_target_masks_geofm_signal",
                "alignment_advantage": {
                    "b0_explicit_proxy_r2_mean": 0.9973990529,
                    "geofm_representation_proxy_r2_mean": 0.029462,
                },
            },
        },
    )
    phase67 = _write_json(
        tmp_path / "phase67.json",
        {
            "phase": "phase67_candidate_reward_label_target_audit",
            "phase67_status": "independent_label_required_before_reward_redesign",
            "candidate_target_gate": {"candidate_count": 0},
        },
    )
    phase68 = _write_json(
        tmp_path / "phase68.json",
        {
            "phase": "phase68_external_independent_label_package",
            "phase68_status": "external_label_package_ready",
            "row_counts": {
                "phase2_block_rows": 64984,
                "external_label_csvs": 0,
                "registry_rows": 0,
                "label_preflight_rows": 0,
            },
        },
    )
    return {
        "phase60_json": phase60,
        "phase57_json": phase57,
        "phase59_json": phase59,
        "phase62_json": phase62,
        "phase66_json": phase66,
        "phase67_json": phase67,
        "phase68_json": phase68,
    }


def test_phase69_current_style_evidence_yields_narrowed_route_status(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
        write_phase69_label_free_evidence_synthesis_gate_artifacts,
    )

    paths = _fixture_paths(tmp_path)
    analysis = build_phase69_label_free_evidence_synthesis_gate(**paths)

    assert analysis["phase"] == "phase69_label_free_evidence_synthesis_gate"
    assert analysis["phase69_status"] == "claim_must_be_narrowed_to_low_dimensional_route"
    assert analysis["row_counts"]["evidence_axis_rows"] == 5
    assert analysis["row_counts"]["claim_boundary_rows"] == 10
    assert "does not train" in analysis["claim_boundary"]

    axes = {row["axis_id"]: row for row in analysis["evidence_axis_rows"]}
    assert axes["route_support"]["axis_class"] == "support"
    assert axes["mechanism_support"]["axis_class"] == "support"
    assert axes["mechanism_limits"]["axis_class"] == "limit"
    assert axes["reward_target_limits"]["axis_class"] == "blocked"
    assert axes["external_label_state"]["axis_class"] == "blocked"

    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    assert claims["bounded_low_dimensional_route"]["claim_status"] == "allowed"
    assert claims["suitability_reward_readiness"]["claim_status"] == "blocked"
    assert claims["geofm_specific_matched_dimension_superiority"]["claim_status"] == "blocked"

    artifacts = write_phase69_label_free_evidence_synthesis_gate_artifacts(
        analysis,
        tmp_path / "outputs",
    )
    assert {path.name for path in artifacts.values()} == {
        "phase69_evidence_axes.csv",
        "phase69_claim_boundary_matrix.csv",
        "phase69_label_free_evidence_synthesis_gate.json",
        "phase69_label_free_evidence_synthesis_gate.md",
    }
    markdown = (tmp_path / "outputs" / "phase69_label_free_evidence_synthesis_gate.md").read_text(
        encoding="utf-8"
    )
    assert "claim_must_be_narrowed_to_low_dimensional_route" in markdown
    assert "bounded low-dimensional compressed state route" in markdown
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py::test_phase69_current_style_evidence_yields_narrowed_route_status -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `paper11_geofm.phase69_label_free_evidence_synthesis_gate`.

- [ ] **Step 3: Add the minimal Phase 69 module**

Create `src/paper11_geofm/phase69_label_free_evidence_synthesis_gate.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path


PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY = (
    "Phase 69 is a read-only label-free evidence synthesis gate over existing "
    "Paper11 artifacts. It does not train PPO, does not alter rewards, does not "
    "enable B2/B3, does not validate suitability, and does not justify formal "
    "submission-level claims."
)

PHASE69_EVIDENCE_AXIS_FIELDNAMES = (
    "axis_id",
    "axis_class",
    "axis_status",
    "source_phases",
    "primary_metric",
    "primary_value",
    "decision_reason",
    "claim_boundary",
)

PHASE69_CLAIM_BOUNDARY_FIELDNAMES = (
    "claim_id",
    "claim_status",
    "supporting_axis_ids",
    "blocking_axis_ids",
    "decision_reason",
    "claim_boundary",
)


def build_phase69_label_free_evidence_synthesis_gate(
    phase60_json: Path | str,
    phase57_json: Path | str,
    phase59_json: Path | str,
    phase62_json: Path | str,
    phase66_json: Path | str,
    phase67_json: Path | str,
    phase68_json: Path | str,
) -> dict[str, object]:
    phase60 = _read_json_object(phase60_json, "Phase 60 JSON")
    phase57 = _read_json_object(phase57_json, "Phase 57 JSON")
    phase59 = _read_json_object(phase59_json, "Phase 59 JSON")
    phase62 = _read_json_object(phase62_json, "Phase 62 JSON")
    phase66 = _read_json_object(phase66_json, "Phase 66 JSON")
    phase67 = _read_json_object(phase67_json, "Phase 67 JSON")
    phase68 = _read_json_object(phase68_json, "Phase 68 JSON")

    evidence_axis_rows = [
        _route_support_axis(phase60),
        _mechanism_support_axis(phase57),
        _mechanism_limits_axis(phase59, phase62),
        _reward_target_limits_axis(phase66, phase67),
        _external_label_state_axis(phase68),
    ]
    status = _phase69_status(evidence_axis_rows)
    claim_boundary_rows = _claim_boundary_rows(status, evidence_axis_rows)
    return {
        "phase": "phase69_label_free_evidence_synthesis_gate",
        "phase69_status": status,
        "source_paths": {
            "phase60_json": str(Path(phase60_json)),
            "phase57_json": str(Path(phase57_json)),
            "phase59_json": str(Path(phase59_json)),
            "phase62_json": str(Path(phase62_json)),
            "phase66_json": str(Path(phase66_json)),
            "phase67_json": str(Path(phase67_json)),
            "phase68_json": str(Path(phase68_json)),
        },
        "row_counts": {
            "evidence_axis_rows": len(evidence_axis_rows),
            "claim_boundary_rows": len(claim_boundary_rows),
        },
        "evidence_axis_rows": evidence_axis_rows,
        "claim_boundary_rows": claim_boundary_rows,
        "allowed_claim": _allowed_claim(status),
        "blocked_claims": [
            row["claim_id"]
            for row in claim_boundary_rows
            if row["claim_status"] == "blocked"
        ],
        "recommended_next_step": _phase69_next_step(status),
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def write_phase69_label_free_evidence_synthesis_gate_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "evidence_axes_csv": output_path / "phase69_evidence_axes.csv",
        "claim_boundary_matrix_csv": output_path / "phase69_claim_boundary_matrix.csv",
        "diagnosis_json": output_path / "phase69_label_free_evidence_synthesis_gate.json",
        "diagnosis_md": output_path / "phase69_label_free_evidence_synthesis_gate.md",
    }
    _write_csv_mapping_rows(
        artifacts["evidence_axes_csv"],
        PHASE69_EVIDENCE_AXIS_FIELDNAMES,
        analysis.get("evidence_axis_rows", []),
        "Phase 69 evidence axis rows",
    )
    _write_csv_mapping_rows(
        artifacts["claim_boundary_matrix_csv"],
        PHASE69_CLAIM_BOUNDARY_FIELDNAMES,
        analysis.get("claim_boundary_rows", []),
        "Phase 69 claim boundary rows",
    )
    artifacts["diagnosis_json"].write_text(
        json.dumps(_json_ready(analysis), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["diagnosis_md"].write_text(_phase69_markdown(analysis), encoding="utf-8")
    return artifacts


def _read_json_object(path: Path | str, label: str) -> dict[str, object]:
    json_path = Path(path)
    if not json_path.exists():
        raise ValueError(f"Missing {label}: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {json_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {json_path}")
    return dict(payload)


def _route_support_axis(phase60: Mapping[str, object]) -> dict[str, object]:
    axes = phase60.get("attribution_axes", [])
    route = _find_axis(axes, "compressed_route_performance")
    cluster = _find_axis(axes, "cluster_level_robustness")
    supported = (
        route.get("axis_status") == "supported"
        and cluster.get("axis_status") == "supported"
    )
    mean_delta = _as_float(route.get("primary_value"))
    return _axis_row(
        axis_id="route_support",
        axis_class="support" if supported else "missing",
        axis_status="supported" if supported else "missing_or_not_supported",
        source_phases="phase48_phase52_expanded,phase53",
        primary_metric="pooled_or_cluster_mean_delta",
        primary_value=mean_delta,
        decision_reason=(
            "Compressed route and cluster robustness are supported."
            if supported
            else "Compressed route support or cluster robustness is missing."
        ),
    )


def _mechanism_support_axis(phase57: Mapping[str, object]) -> dict[str, object]:
    status = str(phase57.get("phase57_mechanism_status", ""))
    supported = status == "compressed_geometry_consistent"
    return _axis_row(
        axis_id="mechanism_support",
        axis_class="support" if supported else "missing",
        axis_status=status or "missing",
        source_phases="phase57",
        primary_metric="phase57_mechanism_status",
        primary_value=status,
        decision_reason=(
            "Compressed geometry is consistent with the low-dimensional route."
            if supported
            else "Compressed geometry support is missing or not supported."
        ),
    )


def _mechanism_limits_axis(
    phase59: Mapping[str, object],
    phase62: Mapping[str, object],
) -> dict[str, object]:
    phase59_status = str(phase59.get("phase59_matched_dimension_status", ""))
    phase62_status = str(phase62.get("phase62_d4_d6_status", ""))
    limiting = (
        phase59_status == "matched_dimension_geofm_not_supported"
        and phase62_status == "d6_random_projection_advantage"
    )
    return _axis_row(
        axis_id="mechanism_limits",
        axis_class="limit" if limiting else "missing",
        axis_status=f"{phase59_status};{phase62_status}",
        source_phases="phase59,phase62",
        primary_metric="matched_control_limit_status",
        primary_value=f"{phase59_status};{phase62_status}",
        decision_reason=(
            "Matched-dimension and D6 controls block GeoFM-specific/PCA-optimality claims."
            if limiting
            else "Matched-control limiting evidence is missing or inconsistent."
        ),
    )


def _reward_target_limits_axis(
    phase66: Mapping[str, object],
    phase67: Mapping[str, object],
) -> dict[str, object]:
    phase66_status = str(phase66.get("phase66_status", ""))
    phase67_status = str(phase67.get("phase67_status", ""))
    blocked = (
        phase66_status == "base_reward_target_masks_geofm_signal"
        and phase67_status == "independent_label_required_before_reward_redesign"
    )
    return _axis_row(
        axis_id="reward_target_limits",
        axis_class="blocked" if blocked else "missing",
        axis_status=f"{phase66_status};{phase67_status}",
        source_phases="phase66,phase67",
        primary_metric="reward_target_block_status",
        primary_value=f"{phase66_status};{phase67_status}",
        decision_reason=(
            "Base reward masks GeoFM signal and candidate-target route requires independent labels."
            if blocked
            else "Reward/target limiting evidence is missing or inconsistent."
        ),
    )


def _external_label_state_axis(phase68: Mapping[str, object]) -> dict[str, object]:
    status = str(phase68.get("phase68_status", ""))
    blocked = status == "external_label_package_ready"
    return _axis_row(
        axis_id="external_label_state",
        axis_class="blocked" if blocked else "support",
        axis_status=status or "missing",
        source_phases="phase68",
        primary_metric="phase68_status",
        primary_value=status,
        decision_reason=(
            "External-label package is ready, but no external label CSV or registry has been supplied."
            if blocked
            else "External label state is not template-only."
        ),
    )


def _axis_row(
    axis_id: str,
    axis_class: str,
    axis_status: str,
    source_phases: str,
    primary_metric: str,
    primary_value: object,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "axis_id": axis_id,
        "axis_class": axis_class,
        "axis_status": axis_status,
        "source_phases": source_phases,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "decision_reason": decision_reason,
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def _phase69_status(evidence_axis_rows: Sequence[Mapping[str, object]]) -> str:
    rows = {str(row.get("axis_id")): row for row in evidence_axis_rows}
    route_supported = rows.get("route_support", {}).get("axis_class") == "support"
    mechanism_supported = rows.get("mechanism_support", {}).get("axis_class") == "support"
    if not route_supported or not mechanism_supported:
        return "label_free_evidence_insufficient"
    has_limits = rows.get("mechanism_limits", {}).get("axis_class") == "limit"
    has_reward_block = rows.get("reward_target_limits", {}).get("axis_class") == "blocked"
    has_label_block = rows.get("external_label_state", {}).get("axis_class") == "blocked"
    if has_limits or has_reward_block or has_label_block:
        return "claim_must_be_narrowed_to_low_dimensional_route"
    return "bounded_label_free_algorithm_claim_supported"


def _claim_boundary_rows(
    status: str,
    evidence_axis_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    allowed = status in {
        "bounded_label_free_algorithm_claim_supported",
        "claim_must_be_narrowed_to_low_dimensional_route",
    }
    return [
        _claim_row(
            "bounded_low_dimensional_route",
            "allowed" if allowed else "blocked",
            "route_support,mechanism_support",
            "",
            "Bounded low-dimensional compressed state route is allowed under label-free evidence."
            if allowed
            else "Required label-free support axes are missing.",
        ),
        _claim_row("raw_b1_superiority", "blocked", "", "mechanism_limits", "Raw B1 superiority is not supported."),
        _claim_row("pca_optimality", "blocked", "", "mechanism_limits", "Matched controls block PCA optimality."),
        _claim_row("geofm_specific_matched_dimension_superiority", "blocked", "", "mechanism_limits", "Matched-dimension evidence blocks this claim."),
        _claim_row("suitability_reward_readiness", "blocked", "", "reward_target_limits,external_label_state", "Suitability reward requires independent labels."),
        _claim_row("b2_b3_reward_integration", "blocked", "", "reward_target_limits,external_label_state", "B2/B3 reward integration remains blocked."),
        _claim_row("external_independent_label_passed", "blocked", "", "external_label_state", "Phase 68 is template-ready only."),
        _claim_row("independent_agronomic_suitability", "blocked", "", "external_label_state", "No independent agronomic label has passed."),
        _claim_row("cross_region_transfer", "blocked", "", "not_tested", "Cross-region transfer is not tested."),
        _claim_row("formal_submission_readiness", "out_of_scope", "", "paper_not_revised", "Formal submission readiness is outside Phase 69."),
    ]


def _claim_row(
    claim_id: str,
    claim_status: str,
    supporting_axis_ids: str,
    blocking_axis_ids: str,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_status": claim_status,
        "supporting_axis_ids": supporting_axis_ids,
        "blocking_axis_ids": blocking_axis_ids,
        "decision_reason": decision_reason,
        "claim_boundary": PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY,
    }


def _find_axis(axes: object, axis_id: str) -> dict[str, object]:
    if not isinstance(axes, list):
        return {}
    for row in axes:
        if isinstance(row, Mapping) and row.get("axis_id") == axis_id:
            return dict(row)
    return {}


def _allowed_claim(status: str) -> str:
    if status == "label_free_evidence_insufficient":
        return "No label-free algorithm claim is supported by the required axes."
    return "Bounded low-dimensional compressed state route under the Bishan base-reward protocol."


def _phase69_next_step(status: str) -> str:
    if status == "label_free_evidence_insufficient":
        return "Do not revise manuscript claims; repair missing label-free evidence first."
    return "Keep claims narrowed; obtain external independent labels before suitability reward or reward redesign."


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    label: str,
) -> None:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must be a list of mappings")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)
    return _json_ready(value)


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _as_float(value: object) -> float | str:
    try:
        return round(float(value), 10)
    except (TypeError, ValueError):
        return ""


def _phase69_markdown(analysis: Mapping[str, object]) -> str:
    axes = analysis.get("evidence_axis_rows", [])
    claims = analysis.get("claim_boundary_rows", [])
    axis_rows = axes if isinstance(axes, list) else []
    claim_rows = claims if isinstance(claims, list) else []
    lines = [
        "# Phase 69 Label-Free Evidence Synthesis Gate",
        "",
        f"Status: {analysis.get('phase69_status', '')}",
        "",
        "## Allowed Claim",
        "",
        str(analysis.get("allowed_claim", "")),
        "",
        "## Evidence Axes",
        "",
        *_markdown_table(("axis_id", "axis_class", "axis_status", "decision_reason"), axis_rows),
        "",
        "## Claim Boundary Matrix",
        "",
        *_markdown_table(("claim_id", "claim_status", "decision_reason"), claim_rows),
        "",
        "## Recommended Next Step",
        "",
        str(analysis.get("recommended_next_step", "")),
        "",
        "## Boundary",
        "",
        str(analysis.get("claim_boundary", PHASE69_LABEL_FREE_SYNTHESIS_CLAIM_BOUNDARY)),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(fieldnames: Sequence[str], rows: Sequence[object]) -> list[str]:
    header = [str(field) for field in fieldnames]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append("| " + " | ".join(_markdown_cell(row.get(field, "")) for field in fieldnames) + " |")
    return lines


def _markdown_cell(value: object) -> str:
    return str(_csv_value(value)).replace("|", "\\|").replace("\n", " ")
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py::test_phase69_current_style_evidence_yields_narrowed_route_status -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src\paper11_geofm\phase69_label_free_evidence_synthesis_gate.py tests\test_phase69_label_free_evidence_synthesis_gate.py
git commit -m "feat: add Phase 69 label-free synthesis gate"
```

Expected: commit succeeds.

---

### Task 2: Insufficient Evidence, Required Status Validation, And Boundary Matrix Tests

**Files:**
- Modify: `tests/test_phase69_label_free_evidence_synthesis_gate.py`
- Modify: `src/paper11_geofm/phase69_label_free_evidence_synthesis_gate.py`

- [ ] **Step 1: Add failing tests for insufficient evidence, required status validation, and blocked claims**

Append to `tests/test_phase69_label_free_evidence_synthesis_gate.py`:

```python
def test_phase69_missing_route_support_is_insufficient(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    paths = _fixture_paths(tmp_path)
    phase60 = json.loads(paths["phase60_json"].read_text(encoding="utf-8"))
    phase60["attribution_axes"] = [
        row
        for row in phase60["attribution_axes"]
        if row["axis_id"] != "compressed_route_performance"
    ]
    paths["phase60_json"].write_text(json.dumps(phase60), encoding="utf-8")

    analysis = build_phase69_label_free_evidence_synthesis_gate(**paths)

    assert analysis["phase69_status"] == "label_free_evidence_insufficient"
    route_axis = {
        row["axis_id"]: row for row in analysis["evidence_axis_rows"]
    }["route_support"]
    assert route_axis["axis_class"] == "missing"
    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    assert claims["bounded_low_dimensional_route"]["claim_status"] == "blocked"


def test_phase69_claim_boundary_blocks_stronger_claims(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    analysis = build_phase69_label_free_evidence_synthesis_gate(**_fixture_paths(tmp_path))

    claims = {row["claim_id"]: row for row in analysis["claim_boundary_rows"]}
    blocked_claims = {
        "raw_b1_superiority",
        "pca_optimality",
        "geofm_specific_matched_dimension_superiority",
        "suitability_reward_readiness",
        "b2_b3_reward_integration",
        "external_independent_label_passed",
        "independent_agronomic_suitability",
        "cross_region_transfer",
    }
    for claim_id in blocked_claims:
        assert claims[claim_id]["claim_status"] == "blocked"
    assert claims["formal_submission_readiness"]["claim_status"] == "out_of_scope"


def test_phase69_missing_required_status_field_raises(tmp_path):
    from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
        build_phase69_label_free_evidence_synthesis_gate,
    )

    paths = _fixture_paths(tmp_path)
    phase66 = json.loads(paths["phase66_json"].read_text(encoding="utf-8"))
    phase66.pop("phase66_status")
    paths["phase66_json"].write_text(json.dumps(phase66), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Phase 66 JSON is missing required status field: phase66_status",
    ):
        build_phase69_label_free_evidence_synthesis_gate(**paths)
```

- [ ] **Step 2: Run the required-status test and verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py::test_phase69_missing_required_status_field_raises -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: FAIL with `Failed: DID NOT RAISE <class 'ValueError'>`.

- [ ] **Step 3: Add required status-field validation**

In `build_phase69_label_free_evidence_synthesis_gate`, after the seven `_read_json_object(...)` calls, add:

```python
    _require_status(phase60, "phase60_attribution_status", "Phase 60 JSON")
    _require_status(phase57, "phase57_mechanism_status", "Phase 57 JSON")
    _require_status(phase59, "phase59_matched_dimension_status", "Phase 59 JSON")
    _require_status(phase62, "phase62_d4_d6_status", "Phase 62 JSON")
    _require_status(phase66, "phase66_status", "Phase 66 JSON")
    _require_status(phase67, "phase67_status", "Phase 67 JSON")
    _require_status(phase68, "phase68_status", "Phase 68 JSON")
```

After `_read_json_object`, add:

```python
def _require_status(payload: Mapping[str, object], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing required status field: {field}")
    return value
```

Expected: required phase status fields are hard errors, while missing axis-level support remains represented as `missing` evidence rows.

- [ ] **Step 4: Rerun Phase 69 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: all Phase 69 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\paper11_geofm\phase69_label_free_evidence_synthesis_gate.py tests\test_phase69_label_free_evidence_synthesis_gate.py
git commit -m "test: cover Phase 69 evidence validation boundaries"
```

Expected: commit succeeds.

---

### Task 3: CLI Runner

**Files:**
- Create: `experiments/phase69_label_free_evidence_synthesis_gate/run_phase69_label_free_evidence_synthesis_gate.py`
- Modify: `tests/test_phase69_label_free_evidence_synthesis_gate.py`

- [ ] **Step 1: Add failing CLI test**

Append:

```python
def test_phase69_runner_cli_writes_artifacts(tmp_path):
    paths = _fixture_paths(tmp_path)
    script = (
        ROOT
        / "experiments"
        / "phase69_label_free_evidence_synthesis_gate"
        / "run_phase69_label_free_evidence_synthesis_gate.py"
    )
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase60-json",
            str(paths["phase60_json"]),
            "--phase57-json",
            str(paths["phase57_json"]),
            "--phase59-json",
            str(paths["phase59_json"]),
            "--phase62-json",
            str(paths["phase62_json"]),
            "--phase66-json",
            str(paths["phase66_json"]),
            "--phase67-json",
            str(paths["phase67_json"]),
            "--phase68-json",
            str(paths["phase68_json"]),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 69 label-free synthesis status: claim_must_be_narrowed_to_low_dimensional_route" in result.stdout
    assert (output_dir / "phase69_label_free_evidence_synthesis_gate.json").exists()
```

- [ ] **Step 2: Run CLI test and verify it fails**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py::test_phase69_runner_cli_writes_artifacts -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: FAIL because the runner file does not exist.

- [ ] **Step 3: Add the runner**

Create `experiments/phase69_label_free_evidence_synthesis_gate/run_phase69_label_free_evidence_synthesis_gate.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper11_geofm.phase69_label_free_evidence_synthesis_gate import (
    build_phase69_label_free_evidence_synthesis_gate,
    write_phase69_label_free_evidence_synthesis_gate_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Paper11 Phase 69 label-free evidence synthesis gate."
    )
    parser.add_argument("--phase60-json", type=Path, required=True)
    parser.add_argument("--phase57-json", type=Path, required=True)
    parser.add_argument("--phase59-json", type=Path, required=True)
    parser.add_argument("--phase62-json", type=Path, required=True)
    parser.add_argument("--phase66-json", type=Path, required=True)
    parser.add_argument("--phase67-json", type=Path, required=True)
    parser.add_argument("--phase68-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        analysis = build_phase69_label_free_evidence_synthesis_gate(
            phase60_json=args.phase60_json,
            phase57_json=args.phase57_json,
            phase59_json=args.phase59_json,
            phase62_json=args.phase62_json,
            phase66_json=args.phase66_json,
            phase67_json=args.phase67_json,
            phase68_json=args.phase68_json,
        )
        artifacts = write_phase69_label_free_evidence_synthesis_gate_artifacts(
            analysis,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Phase 69 label-free synthesis status: {analysis['phase69_status']}")
    print(f"Evidence axes CSV: {artifacts['evidence_axes_csv']}")
    print(f"Claim boundary matrix CSV: {artifacts['claim_boundary_matrix_csv']}")
    print(f"Diagnosis JSON: {artifacts['diagnosis_json']}")
    print(f"Diagnosis Markdown: {artifacts['diagnosis_md']}")
    print(f"Allowed claim: {analysis['allowed_claim']}")
    print(f"Recommended next step: {analysis['recommended_next_step']}")
    print(f"Claim boundary: {analysis['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run Phase 69 tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: all Phase 69 tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add experiments\phase69_label_free_evidence_synthesis_gate\run_phase69_label_free_evidence_synthesis_gate.py tests\test_phase69_label_free_evidence_synthesis_gate.py
git commit -m "feat: add Phase 69 label-free synthesis runner"
```

Expected: commit succeeds.

---

### Task 4: Real Run And Result Note

**Files:**
- Create: `paper/phase28_results/35_phase69_label_free_evidence_synthesis_gate.md`
- Modify: `paper/phase28_results/README.md`

- [ ] **Step 1: Run the real Phase 69 synthesis gate**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase69_label_free_evidence_synthesis_gate\run_phase69_label_free_evidence_synthesis_gate.py --phase60-json experiments\phase60_information_optimization_attribution\outputs\phase52_full5_seed3\phase60_information_optimization_attribution.json --phase57-json experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3\phase57_compressed_representation_mechanism.json --phase59-json experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json --phase62-json experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo.json --phase66-json experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json --phase67-json experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3\phase67_candidate_reward_label_target_audit.json --phase68-json experiments\phase68_external_independent_label_package\outputs\real_bishan_template_only\phase68_external_independent_label_package.json --output-dir experiments\phase69_label_free_evidence_synthesis_gate\outputs\phase52_full5_seed3
```

Expected: exit code `0`, console prints `Phase 69 label-free synthesis status: claim_must_be_narrowed_to_low_dimensional_route`.

- [ ] **Step 2: Inspect real JSON status**

Run:

```powershell
$j = Get-Content -Raw experiments\phase69_label_free_evidence_synthesis_gate\outputs\phase52_full5_seed3\phase69_label_free_evidence_synthesis_gate.json | ConvertFrom-Json
$j | Select-Object phase, phase69_status, allowed_claim
$j.row_counts | ConvertTo-Json -Depth 3
```

Expected:

```text
phase                                      phase69_status
-----                                      --------------
phase69_label_free_evidence_synthesis_gate claim_must_be_narrowed_to_low_dimensional_route
```

Row counts should include `evidence_axis_rows: 5` and `claim_boundary_rows: 10`.

- [ ] **Step 3: Add the Phase 69 result note**

Create `paper/phase28_results/35_phase69_label_free_evidence_synthesis_gate.md`:

```markdown
# Phase 69 Label-Free Evidence Synthesis Gate

Status: claim_must_be_narrowed_to_low_dimensional_route

## Key Evidence

- Phase 69 synthesized existing label-free Paper11 evidence across route support, mechanism support, mechanism limits, reward/target limits, and external-label state.
- Allowed bounded claim: low-dimensional compressed state routes remain defensible under the Bishan base-reward protocol.
- Blocked stronger claims: raw B1 superiority, PCA optimality, GeoFM-specific matched-dimension superiority, suitability reward readiness, B2/B3 reward integration, external independent-label pass, independent agronomic suitability, and cross-region transfer.
- Phase 68 remains template-ready only; no external independent label CSV or registry has been supplied.
- Phase 69 does not train policies, alter rewards, enable B2/B3, or revise formal manuscript files.

## Reproduction

Run from the repository root:

```powershell
D:\adk\.venv\Scripts\python.exe experiments\phase69_label_free_evidence_synthesis_gate\run_phase69_label_free_evidence_synthesis_gate.py --phase60-json experiments\phase60_information_optimization_attribution\outputs\phase52_full5_seed3\phase60_information_optimization_attribution.json --phase57-json experiments\phase57_compressed_representation_mechanism\outputs\phase52_full5_seed3\phase57_compressed_representation_mechanism.json --phase59-json experiments\phase59_matched_dimension_controls\outputs\phase52_full5_seed3\phase59_matched_dimension_controls.json --phase62-json experiments\phase62_d4_d6_matched_ppo\outputs\phase52_full5_seed3\phase62_d4_d6_matched_ppo.json --phase66-json experiments\phase66_reward_label_representation_audit\outputs\phase52_full5_seed3\phase66_reward_label_representation_audit.json --phase67-json experiments\phase67_candidate_reward_label_target_audit\outputs\phase52_full5_seed3\phase67_candidate_reward_label_target_audit.json --phase68-json experiments\phase68_external_independent_label_package\outputs\real_bishan_template_only\phase68_external_independent_label_package.json --output-dir experiments\phase69_label_free_evidence_synthesis_gate\outputs\phase52_full5_seed3
```

## Boundary

Phase 69 is a read-only label-free evidence synthesis gate over existing Paper11 artifacts. It does not train PPO, alter rewards, enable B2/B3, validate suitability, or justify formal submission-level claims.
```

- [ ] **Step 4: Add README entry**

In `paper/phase28_results/README.md`, add this bullet near the current result-note list:

```markdown
- `35_phase69_label_free_evidence_synthesis_gate.md`: read-only label-free synthesis gate showing that the strongest current algorithm claim must remain narrowed to a bounded low-dimensional compressed state route.
```

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add paper\phase28_results\35_phase69_label_free_evidence_synthesis_gate.md paper\phase28_results\README.md
git commit -m "docs: record Phase 69 label-free synthesis result"
```

Expected: commit succeeds.

---

### Task 5: Final Verification And Push

**Files:**
- No new files expected beyond prior tasks.

- [ ] **Step 1: Run targeted regression tests**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe -m pytest tests\test_phase69_label_free_evidence_synthesis_gate.py tests\test_phase68_external_independent_label_package.py tests\test_phase67_candidate_reward_label_target_audit.py tests\test_phase66_reward_label_representation_audit.py -q --basetemp=D:\tmp\paper11_phase69_pytest_tmp -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run smoke check**

Run:

```powershell
D:\adk\.venv\Scripts\python.exe scripts\smoke_check.py
```

Expected: prints `Paper11 smoke check passed.`

- [ ] **Step 3: Check whitespace and formal manuscript untouched**

Run:

```powershell
git diff --check
git diff --name-only HEAD -- paper\submission\final
```

Expected: `git diff --check` prints nothing. Formal manuscript diff command prints nothing.

- [ ] **Step 4: Check status**

Run:

```powershell
git status --short --branch
git log -1 --oneline
```

Expected: worktree is clean except intentional unpushed commits on `main`.

- [ ] **Step 5: Push completed Phase 69 work**

Run:

```powershell
git push
```

Expected: push succeeds and `main` is synchronized with `origin/main`.

---

## Self-Review Checklist

- Spec coverage:
  - Evidence axes are implemented in Task 1, with Phase 60 explicitly serving as the Phase 48/52/53 compressed-route evidence aggregator.
  - Status model is implemented in Task 1; insufficient-evidence behavior and hard errors for missing required status fields are verified in Task 2.
  - Claim boundary matrix is implemented in Task 1 and verified in Task 2.
  - Artifact writer and CLI are covered in Tasks 1 and 3.
  - Real result note and README update are covered in Task 4.
  - Formal manuscript untouched check is covered in Task 5.
- Deferred-marker scan:
  - The plan contains no deferred implementation markers and no unspecified test commands.
- Type consistency:
  - Public functions are `build_phase69_label_free_evidence_synthesis_gate` and `write_phase69_label_free_evidence_synthesis_gate_artifacts`.
  - Public status key is `phase69_status`.
  - Evidence rows use `axis_id`, `axis_class`, and `axis_status`.
  - Claim rows use `claim_id` and `claim_status`.
