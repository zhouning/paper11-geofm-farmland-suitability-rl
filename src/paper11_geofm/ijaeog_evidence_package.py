from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path


PHASE24_CLAIM_BOUNDARY = (
    "Phase 24 is a synthesis and claim-readiness package; it summarizes "
    "current pilot evidence and remaining gaps, but does not create new "
    "policy-performance, transfer, or suitability-reward evidence."
)

EVIDENCE_FIELDNAMES = ["claim_area", "status", "evidence", "remaining_gap"]


def build_phase24_ijaeog_evidence_package(
    phase22_summary_csv: Path | str,
    phase23_summary_csv: Path | str,
    phase23_comparison_json: Path | str,
) -> dict[str, object]:
    phase22_rows = _read_summary_rows(Path(phase22_summary_csv))
    phase23_rows = _read_summary_rows(Path(phase23_summary_csv))
    phase23_comparison = _read_json_object(Path(phase23_comparison_json))

    phase22 = _summarize_rows(phase22_rows)
    phase23 = _summarize_rows(phase23_rows)
    learned = phase23_comparison.get("learned_policy", {})
    if not isinstance(learned, Mapping):
        learned = {}
    phase23["B1_minus_B0_mean_reward"] = learned.get("B1_minus_B0_mean_reward")
    phase23["comparison_source"] = str(Path(phase23_comparison_json))
    remaining_evidence_gaps = phase23_comparison.get("remaining_evidence_gaps", [])
    if not isinstance(remaining_evidence_gaps, list):
        remaining_evidence_gaps = []

    claim_readiness = _claim_readiness(phase22, phase23, remaining_evidence_gaps)
    evidence_table = [
        {"claim_area": key, **value} for key, value in claim_readiness.items()
    ]
    return {
        "phase": "phase24_ijaeog_evidence_package",
        "phase22": phase22,
        "phase23": phase23,
        "claim_readiness": claim_readiness,
        "evidence_table": evidence_table,
        "claim_boundary": PHASE24_CLAIM_BOUNDARY,
    }


def write_phase24_ijaeog_evidence_artifacts(
    package: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evidence_path = output_path / "phase24_ijaeog_evidence_table.csv"
    summary_path = output_path / "phase24_ijaeog_evidence_summary.json"
    markdown_path = output_path / "phase24_ijaeog_claim_readiness.md"

    evidence_table = package.get("evidence_table")
    if not isinstance(evidence_table, list):
        raise ValueError("Phase 24 package is missing an evidence_table list")
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_FIELDNAMES)
        writer.writeheader()
        for row in evidence_table:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 24 evidence table rows must be objects")
            writer.writerow({field: row.get(field, "") for field in EVIDENCE_FIELDNAMES})

    summary_path.write_text(
        json.dumps(dict(package), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_claim_readiness(package), encoding="utf-8")
    return {
        "evidence_csv": evidence_path,
        "summary_json": summary_path,
        "claim_readiness_md": markdown_path,
    }


def _read_summary_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 24 input summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 24 input JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 24 JSON input must be an object")
    return value


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    variants = sorted({str(row.get("variant_id", "")) for row in rows if row.get("variant_id")})
    policies = sorted({str(row.get("row_type", "")) for row in rows if row.get("row_type")})
    eval_tiles = sorted({str(row.get("eval_tile_id", "")) for row in rows if row.get("eval_tile_id")})
    train_tiles = sorted({str(row.get("train_tile_id", "")) for row in rows if row.get("train_tile_id")})
    seeds = sorted(
        {
            int(str(row.get("seed", "0")))
            for row in rows
            if str(row.get("seed", "")).strip() != ""
        }
    )
    mean_reward_by_policy_variant: dict[str, dict[str, float]] = {}
    for policy in policies:
        mean_reward_by_policy_variant[policy] = {}
        for variant in variants:
            values = [
                float(row["total_contract_reward"])
                for row in rows
                if row.get("row_type") == policy
                and row.get("variant_id") == variant
                and str(row.get("total_contract_reward", "")).strip() != ""
            ]
            if values:
                mean_reward_by_policy_variant[policy][variant] = _round_float(
                    sum(values) / len(values)
                )
    return {
        "summary_rows": len(rows),
        "variants": variants,
        "policies": policies,
        "train_tiles": train_tiles,
        "eval_tiles": eval_tiles,
        "eval_tile_count": len(eval_tiles),
        "seeds": seeds,
        "seed_count": len(seeds),
        "mean_reward_by_policy_variant": mean_reward_by_policy_variant,
    }


def _claim_readiness(
    phase22: Mapping[str, object],
    phase23: Mapping[str, object],
    remaining_evidence_gaps: list[object],
) -> dict[str, dict[str, str]]:
    phase22_rows = int(phase22.get("summary_rows", 0))
    phase23_rows = int(phase23.get("summary_rows", 0))
    b1_minus_b0 = phase23.get("B1_minus_B0_mean_reward")
    gap_text = "; ".join(str(item) for item in remaining_evidence_gaps)
    if not gap_text:
        gap_text = "longer training, ablations, suitability validation, and transfer"

    return {
        "same_tile_b0_b1_training_pilot": {
            "status": "pilot_supported" if phase23_rows > 0 else "not_ready",
            "evidence": (
                f"Phase 23 has {phase23_rows} same-tile multi-seed training rows; "
                f"B1-B0 learned-policy mean reward delta is {b1_minus_b0}."
            ),
            "remaining_gap": "longer-budget training and held-out evaluation are still required",
        },
        "multi_tile_scorer_interface": {
            "status": "pilot_supported" if phase22_rows > 0 else "not_ready",
            "evidence": f"Phase 22 has {phase22_rows} multi-tile scorer-evaluation rows.",
            "remaining_gap": "this is a scorer interface pilot, not PPO transfer evidence",
        },
        "suitability_reward": {
            "status": "not_ready",
            "evidence": "Phase 10/12 keep suitability reward disabled.",
            "remaining_gap": "weak-label validation must support B2/B3 suitability reward use",
        },
        "transfer": {
            "status": "not_ready",
            "evidence": "Current learned-policy training is same-tile only.",
            "remaining_gap": "variable-size policy or held-out-region transfer evaluation is required",
        },
        "submission_ready": {
            "status": "not_ready",
            "evidence": "The current package contains pilot evidence, not full IJAEOG results.",
            "remaining_gap": gap_text,
        },
    }


def _markdown_claim_readiness(package: Mapping[str, object]) -> str:
    claim_readiness = package.get("claim_readiness")
    if not isinstance(claim_readiness, Mapping):
        raise ValueError("Phase 24 package is missing a claim_readiness object")
    lines = [
        "# Phase 24 IJAEOG Claim Readiness",
        "",
        str(package.get("claim_boundary", PHASE24_CLAIM_BOUNDARY)),
        "",
        "| Claim area | Status | Evidence | Remaining gap |",
        "|---|---|---|---|",
    ]
    for key, value in claim_readiness.items():
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    str(value.get("status", "")),
                    str(value.get("evidence", "")),
                    str(value.get("remaining_gap", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _round_float(value: float) -> float:
    return round(float(value), 10)
