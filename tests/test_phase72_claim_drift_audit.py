from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "\n".join(
            [
                "These results support a bounded conclusion: GeoFM information improved farmland layout optimization under the protocol.",
                "PCA-compressed GeoFM representations, however, improved held-out learned-policy reward.",
                "The main finding is that GeoFM information was useful only after representation control.",
                "GeoFM improved the learned planning policy when represented through controlled compressed state features.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manuscript_md": manuscript,
        "phase60_json": _write_json(
            tmp_path / "phase60.json",
            {"phase60_attribution_status": "mechanism_claim_narrowed"},
        ),
        "phase62_json": _write_json(
            tmp_path / "phase62.json",
            {"phase62_d4_d6_status": "d6_random_projection_advantage"},
        ),
        "phase69_json": _write_json(
            tmp_path / "phase69.json",
            {"phase69_status": "claim_must_be_narrowed_to_low_dimensional_route"},
        ),
        "phase71_json": _write_json(
            tmp_path / "phase71.json",
            {"phase71_status": "ranker_improves_but_target_masks_geofm"},
        ),
        "phase72b_json": _write_json(
            tmp_path / "phase72b.json",
            {"phase72b_status": "geofm_information_not_supported"},
        ),
        "phase72_exhaustion_json": _write_json(
            tmp_path / "exhaustion.json",
            {"phase72_exhaustion_status": "phase72_exhaustion_criteria_not_fully_evaluated"},
        ),
    }


def test_phase72_claim_drift_requires_narrowing(tmp_path):
    from paper11_geofm.phase72_claim_drift_audit import (
        build_phase72_claim_drift_audit,
        write_phase72_claim_drift_audit_artifacts,
    )

    analysis = build_phase72_claim_drift_audit(**_fixture_paths(tmp_path))

    assert analysis["phase72_claim_drift_status"] == "claim_drift_requires_narrowing"
    assert analysis["counts"] == {"claims": 8, "drift_claims": 6, "missing_anchors": 0}
    claims = {row["claim_id"]: row for row in analysis["claim_rows"]}
    assert claims["real_bishan_planning_workflow"]["claim_status"] == "supported"
    assert claims["bounded_low_dimensional_compressed_route"]["claim_status"] == "bounded_supported"
    assert claims["geofm_specific_compressed_information"]["claim_status"] == "blocked"
    assert claims["pca_optimality"]["claim_status"] == "blocked"
    assert claims["formal_manuscript_current_wording"]["claim_status"] == "needs_narrowing"

    artifacts = write_phase72_claim_drift_audit_artifacts(analysis, tmp_path / "outputs")
    assert {path.name for path in artifacts.values()} == {
        "phase72_claim_drift_claims.csv",
        "phase72_claim_drift_audit.json",
        "phase72_claim_drift_audit.md",
    }
    assert "claim_drift_requires_narrowing" in (
        tmp_path / "outputs" / "phase72_claim_drift_audit.md"
    ).read_text(encoding="utf-8")


def test_phase72_claim_drift_detects_missing_manuscript_anchor(tmp_path):
    from paper11_geofm.phase72_claim_drift_audit import build_phase72_claim_drift_audit

    paths = _fixture_paths(tmp_path)
    paths["manuscript_md"].write_text("unrelated text", encoding="utf-8")
    analysis = build_phase72_claim_drift_audit(**paths)

    assert analysis["phase72_claim_drift_status"] == "claim_drift_inputs_incomplete"
    assert len(analysis["missing_anchors"]) == 4


def test_phase72_claim_drift_requires_status_fields(tmp_path):
    from paper11_geofm.phase72_claim_drift_audit import build_phase72_claim_drift_audit

    paths = _fixture_paths(tmp_path)
    phase62 = json.loads(paths["phase62_json"].read_text(encoding="utf-8"))
    phase62.pop("phase62_d4_d6_status")
    paths["phase62_json"].write_text(json.dumps(phase62), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required status field: phase62_d4_d6_status"):
        build_phase72_claim_drift_audit(**paths)


def test_phase72_claim_drift_runner_cli_writes_artifacts(tmp_path):
    paths = _fixture_paths(tmp_path)
    output_dir = tmp_path / "cli-output"
    script = ROOT / "experiments" / "phase72_claim_drift_audit" / "run_phase72_claim_drift_audit.py"
    args = []
    for key, value in paths.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    result = subprocess.run(
        [sys.executable, str(script), *args, "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "claim_drift_requires_narrowing" in result.stdout
    assert (output_dir / "phase72_claim_drift_audit.json").is_file()
