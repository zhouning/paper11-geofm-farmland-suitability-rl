from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from os import PathLike
from pathlib import Path

from .phase50_cluster_level_robustness import build_phase50_cluster_level_robustness
from .phase51_cluster_magnitude_support import build_phase51_cluster_magnitude_support
from .phase53_cluster_mean_support import build_phase53_cluster_mean_support


PHASE54_CLAIM_BOUNDARY = (
    "Phase 54 is a read-only artifact-lineage consistency audit over the "
    "authoritative Phase 52/53 compressed GeoFM evidence chain. It recomputes "
    "Phase 50 cluster means from the Phase 48 delta table, recomputes Phase 51 "
    "and Phase 53 statistics from the authoritative cluster CSV, and checks "
    "that the formal manuscript values come from one consistent artifact "
    "lineage; it does not enable suitability reward, does not test B2/B3, does "
    "not test cross-region transfer, and does not validate independent "
    "agronomic suitability."
)

CHECK_FIELDNAMES = [
    "check_name",
    "expected_value",
    "actual_value",
    "tolerance",
    "passed",
    "detail",
    "claim_boundary",
]


def build_phase54_artifact_lineage_consistency(
    delta_csv: Path | str | Sequence[Mapping[str, object]],
    cluster_csv: Path | str | Sequence[Mapping[str, object]],
    phase51_json: Path | str | Mapping[str, object],
    phase53_json: Path | str | Mapping[str, object],
    tolerance: float = 1e-9,
) -> dict[str, object]:
    authoritative_clusters = _load_cluster_rows(cluster_csv)
    authoritative_phase51 = _load_json_mapping(phase51_json)
    authoritative_phase53 = _load_json_mapping(phase53_json)

    phase50_recomputed = build_phase50_cluster_level_robustness(delta_csv)
    recomputed_clusters = phase50_recomputed["cluster_rows"]
    phase51_recomputed = build_phase51_cluster_magnitude_support(authoritative_clusters)
    phase53_recomputed = build_phase53_cluster_mean_support(
        authoritative_clusters,
        bootstrap_iterations=int(authoritative_phase53.get("bootstrap_iterations", 5000)),
        random_seed=int(authoritative_phase53.get("random_seed", 53)),
        alpha=float(authoritative_phase53.get("alpha", 0.05)),
    )

    check_rows = [
        _cluster_rows_check(
            authoritative_clusters,
            recomputed_clusters,
            tolerance=float(tolerance),
        ),
        *_phase51_checks(authoritative_phase51, phase51_recomputed, float(tolerance)),
        *_phase53_checks(authoritative_phase53, phase53_recomputed, float(tolerance)),
    ]
    all_checks_passed = all(bool(row["passed"]) for row in check_rows)
    status = (
        "artifact_lineage_consistent"
        if all_checks_passed
        else "artifact_lineage_inconsistent"
    )
    return {
        "phase": "phase54_artifact_lineage_consistency",
        "phase54_lineage_status": status,
        "all_checks_passed": all_checks_passed,
        "tolerance": float(tolerance),
        "check_rows": check_rows,
        "recomputed_cluster_count": phase53_recomputed.get("cluster_count"),
        "recomputed_mean_cluster_delta": phase53_recomputed.get("mean_cluster_delta"),
        "recomputed_phase51_signed_rank_p": phase51_recomputed.get(
            "one_sided_signed_rank_p"
        ),
        "recomputed_phase53_sign_flip_p": phase53_recomputed.get(
            "exact_sign_flip_mean_p"
        ),
        "recomputed_phase53_bootstrap_ci95_low": phase53_recomputed.get(
            "bootstrap_ci95_low"
        ),
        "recomputed_phase53_bootstrap_ci95_high": phase53_recomputed.get(
            "bootstrap_ci95_high"
        ),
        "phase50_recomputed_summary": phase50_recomputed.get("cluster_summary"),
        "phase51_recomputed_status": phase51_recomputed.get("phase51_magnitude_status"),
        "phase53_recomputed_status": phase53_recomputed.get(
            "phase53_cluster_mean_status"
        ),
        "conclusion": _phase54_conclusion(status),
        "claim_boundary": PHASE54_CLAIM_BOUNDARY,
    }


def write_phase54_artifact_lineage_consistency_artifacts(
    analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / "phase54_artifact_lineage_consistency.json"
    checks_path = output_path / "phase54_artifact_lineage_checks.csv"
    readiness_path = output_path / "phase54_artifact_lineage_consistency.md"

    comparison_path.write_text(
        json.dumps(_json_ready(dict(analysis)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        checks_path,
        CHECK_FIELDNAMES,
        analysis.get("check_rows"),
        "check_rows",
    )
    readiness_path.write_text(_readiness_markdown(analysis), encoding="utf-8")
    return {
        "comparison_json": comparison_path,
        "checks_csv": checks_path,
        "readiness_md": readiness_path,
    }


def _load_cluster_rows(
    rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, PathLike)):
        path = Path(rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 54 cluster CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in rows_or_csv]


def _load_json_mapping(path_or_mapping: Path | str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(path_or_mapping, (str, PathLike)):
        path = Path(path_or_mapping)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 54 JSON: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(path_or_mapping)


def _cluster_rows_check(
    authoritative_rows: Sequence[Mapping[str, object]],
    recomputed_rows: object,
    tolerance: float,
) -> dict[str, object]:
    authoritative = _cluster_signature(authoritative_rows)
    recomputed = _cluster_signature(recomputed_rows if isinstance(recomputed_rows, list) else [])
    passed = len(authoritative) == len(recomputed)
    mismatch_count = 0
    if passed:
        for expected, actual in zip(authoritative, recomputed):
            if expected[:3] != actual[:3] or abs(expected[3] - actual[3]) > tolerance:
                mismatch_count += 1
        passed = mismatch_count == 0
    else:
        mismatch_count = abs(len(authoritative) - len(recomputed))
    return _check_row(
        "phase50_cluster_rows_match_delta_recompute",
        expected_value=f"{len(authoritative)} cluster rows",
        actual_value=f"{len(recomputed)} cluster rows",
        tolerance=tolerance,
        passed=passed,
        detail=f"mismatch_count={mismatch_count}",
    )


def _phase51_checks(
    authoritative: Mapping[str, object],
    recomputed: Mapping[str, object],
    tolerance: float,
) -> list[dict[str, object]]:
    return [
        _numeric_check(
            "phase51_cluster_count_matches",
            authoritative.get("cluster_count"),
            recomputed.get("cluster_count"),
            tolerance,
        ),
        _numeric_check(
            "phase51_mean_cluster_delta_matches",
            authoritative.get("mean_cluster_delta"),
            recomputed.get("mean_cluster_delta"),
            tolerance,
        ),
        _numeric_check(
            "phase51_positive_rank_sum_matches",
            authoritative.get("positive_rank_sum"),
            recomputed.get("positive_rank_sum"),
            tolerance,
        ),
        _numeric_check(
            "phase51_total_rank_sum_matches",
            authoritative.get("total_rank_sum"),
            recomputed.get("total_rank_sum"),
            tolerance,
        ),
        _numeric_check(
            "phase51_signed_rank_p_matches",
            authoritative.get("one_sided_signed_rank_p"),
            recomputed.get("one_sided_signed_rank_p"),
            tolerance,
        ),
        _string_check(
            "phase51_status_matches",
            authoritative.get("phase51_magnitude_status"),
            recomputed.get("phase51_magnitude_status"),
        ),
    ]


def _phase53_checks(
    authoritative: Mapping[str, object],
    recomputed: Mapping[str, object],
    tolerance: float,
) -> list[dict[str, object]]:
    influence_authoritative = authoritative.get("influence_summary")
    if not isinstance(influence_authoritative, Mapping):
        influence_authoritative = {}
    influence_recomputed = recomputed.get("influence_summary")
    if not isinstance(influence_recomputed, Mapping):
        influence_recomputed = {}
    return [
        _numeric_check(
            "phase53_cluster_count_matches",
            authoritative.get("cluster_count"),
            recomputed.get("cluster_count"),
            tolerance,
        ),
        _numeric_check(
            "phase53_mean_cluster_delta_matches",
            authoritative.get("mean_cluster_delta"),
            recomputed.get("mean_cluster_delta"),
            tolerance,
        ),
        _numeric_check(
            "phase53_sign_flip_p_matches",
            authoritative.get("exact_sign_flip_mean_p"),
            recomputed.get("exact_sign_flip_mean_p"),
            tolerance,
        ),
        _numeric_check(
            "phase53_bootstrap_low_matches",
            authoritative.get("bootstrap_ci95_low"),
            recomputed.get("bootstrap_ci95_low"),
            tolerance,
        ),
        _numeric_check(
            "phase53_bootstrap_high_matches",
            authoritative.get("bootstrap_ci95_high"),
            recomputed.get("bootstrap_ci95_high"),
            tolerance,
        ),
        _numeric_check(
            "phase53_min_leave_one_cluster_matches",
            influence_authoritative.get("min_leave_one_cluster_mean"),
            influence_recomputed.get("min_leave_one_cluster_mean"),
            tolerance,
        ),
        _numeric_check(
            "phase53_min_leave_one_tile_matches",
            influence_authoritative.get("min_leave_one_tile_mean"),
            influence_recomputed.get("min_leave_one_tile_mean"),
            tolerance,
        ),
        _numeric_check(
            "phase53_min_leave_one_seed_matches",
            influence_authoritative.get("min_leave_one_seed_mean"),
            influence_recomputed.get("min_leave_one_seed_mean"),
            tolerance,
        ),
        _string_check(
            "phase53_status_matches",
            authoritative.get("phase53_cluster_mean_status"),
            recomputed.get("phase53_cluster_mean_status"),
        ),
    ]


def _numeric_check(
    check_name: str,
    expected: object,
    actual: object,
    tolerance: float,
) -> dict[str, object]:
    passed = (
        expected is not None
        and actual is not None
        and abs(float(expected) - float(actual)) <= tolerance
    )
    return _check_row(
        check_name,
        expected_value=expected,
        actual_value=actual,
        tolerance=tolerance,
        passed=passed,
        detail="numeric equality within tolerance",
    )


def _string_check(check_name: str, expected: object, actual: object) -> dict[str, object]:
    passed = str(expected) == str(actual)
    return _check_row(
        check_name,
        expected_value=expected,
        actual_value=actual,
        tolerance=0.0,
        passed=passed,
        detail="string equality",
    )


def _check_row(
    check_name: str,
    expected_value: object,
    actual_value: object,
    tolerance: float,
    passed: bool,
    detail: str,
) -> dict[str, object]:
    return {
        "check_name": check_name,
        "expected_value": expected_value,
        "actual_value": actual_value,
        "tolerance": float(tolerance),
        "passed": bool(passed),
        "detail": detail,
        "claim_boundary": PHASE54_CLAIM_BOUNDARY,
    }


def _cluster_signature(rows: Sequence[Mapping[str, object]]) -> list[tuple[str, int, int, float]]:
    signature = []
    for row in rows:
        signature.append(
            (
                str(row.get("eval_tile_id", "")),
                int(row.get("seed", 0)),
                int(row.get("cluster_delta_count", 0)),
                round(float(row.get("mean_cluster_delta", 0.0)), 10),
            )
        )
    return sorted(signature, key=lambda item: (item[0], item[1]))


def _phase54_conclusion(status: str) -> str:
    if status == "artifact_lineage_consistent":
        return (
            "Phase 54 conclusion: the formal Phase 52/53 compressed-route "
            "values are internally reproducible from the authoritative artifact "
            "chain."
        )
    return (
        "Phase 54 conclusion: the formal Phase 52/53 compressed-route artifact "
        "chain is inconsistent and must not be used without correction."
    )


def _readiness_markdown(analysis: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Phase 54 Artifact Lineage Consistency",
            "",
            f"Status: {analysis.get('phase54_lineage_status', '')}",
            "",
            "Authoritative artifact chain:",
            "- Phase 48 compressed-route delta table",
            "- Phase 50 cluster delta summary",
            "- Phase 51 cluster magnitude JSON",
            "- Phase 53 cluster mean support JSON",
            "",
            "Recomputed values:",
            "- "
            f"clusters={analysis.get('recomputed_cluster_count')}, "
            f"mean={analysis.get('recomputed_mean_cluster_delta')}, "
            f"Phase 51 p={analysis.get('recomputed_phase51_signed_rank_p')}, "
            f"Phase 53 sign-flip p={analysis.get('recomputed_phase53_sign_flip_p')}",
            "- "
            f"Phase 53 bootstrap CI95=["
            f"{analysis.get('recomputed_phase53_bootstrap_ci95_low')}, "
            f"{analysis.get('recomputed_phase53_bootstrap_ci95_high')}]",
            "",
            "Conclusion:",
            str(analysis.get("conclusion", "")),
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE54_CLAIM_BOUNDARY)),
            "",
        ]
    )


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 54 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 54 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
