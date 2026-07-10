from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path

import numpy as np

from .phase72a_label_sources import (
    PHASE72A_CLAIM_BOUNDARY,
    audit_phase72a_region_assets,
    load_phase72a_region_contract,
)
from .phase72a_review_frame import (
    REVIEW_FIELDS,
    build_phase72a_review_frame,
)
from .phase72a_temporal_samples import build_phase72a_temporal_samples


MANIFEST_FIELDS = (
    "region_id",
    "year",
    "asset_type",
    "source_id",
    "path",
    "shape",
    "dtype",
    "sha256",
    "independent_label",
)

AUDIT_FIELDS = (
    "region_id",
    "status",
    "years_ready",
    "errors",
    "manifest_rows",
    "claim_boundary",
)

SAMPLE_FIELDS = (
    "sample_index",
    "region_id",
    "unit_id",
    "row",
    "col",
    "spatial_block_id",
    "origin_year",
    "history_start_year",
    "history_end_year",
    "history_length",
    "current_lulc_class",
    "target_year_1y",
    "y_1y",
    "target_year_2y",
    "y_2y",
    "y_continuous_2y",
    "label_source_id",
    "label_source_role",
    "label_confidence",
    "claim_boundary",
)

REVIEW_FRAME_FIELDS = (
    "sample_index",
    "region_id",
    "unit_id",
    "row",
    "col",
    "spatial_block_id",
    "origin_year",
    "target_year_1y",
    "transition_type",
    "label_source_id",
    *REVIEW_FIELDS,
)

SUMMARY_FIELDS = (
    "region_id",
    "horizon",
    "eligible_rows",
    "positive_rows",
    "positive_rate",
    "phase72a_status",
    "claim_boundary",
)


def build_phase72a_temporal_label_package(
    *,
    region_config: Path | str,
    embedding_dirs: Mapping[str, Path | str],
    label_dirs: Mapping[str, Path | str],
    manual_review_per_stratum: int = 20,
    spatial_block_size: int = 8,
) -> dict[str, object]:
    contract = load_phase72a_region_contract(region_config)
    audits = []
    manifest_rows = []
    for region in contract.regions:
        if (
            region.region_id not in embedding_dirs
            or region.region_id not in label_dirs
        ):
            audits.append(
                {
                    "region_id": region.region_id,
                    "status": "label_inputs_not_ready",
                    "years_ready": [],
                    "errors": [
                        f"missing directory mapping for {region.region_id}"
                    ],
                    "file_manifest_rows": [],
                    "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
                }
            )
            continue
        audit = audit_phase72a_region_assets(
            contract,
            region,
            embedding_dir=embedding_dirs[region.region_id],
            label_dir=label_dirs[region.region_id],
        )
        audits.append(audit)
        manifest_rows.extend(audit["file_manifest_rows"])

    if any(
        audit["status"] != "region_label_inputs_ready"
        for audit in audits
    ):
        return {
            "phase": "phase72a_temporal_label_package",
            "phase72a_status": "label_inputs_not_ready",
            "region_audits": audits,
            "manifest_rows": manifest_rows,
            "sample_rows": [],
            "review_rows": [],
            "tensors": {},
            "row_counts": {
                "regions": len(audits),
                "manifest_rows": len(manifest_rows),
                "sample_rows": 0,
                "review_rows": 0,
            },
            "recommended_next_step": (
                "Resolve Phase 72A label blockers before model work."
            ),
            "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
        }

    sample_rows = []
    tensor_parts: dict[str, list[np.ndarray]] = {}
    region_index_parts = []
    max_history = max(len(region.years) for region in contract.regions)
    for region_index, region in enumerate(contract.regions):
        embeddings = {
            year: np.load(
                Path(embedding_dirs[region.region_id])
                / region.embedding_pattern.format(year=year)
            )
            for year in region.years
        }
        labels = {
            year: np.load(
                Path(label_dirs[region.region_id])
                / region.label_pattern.format(year=year)
            )
            for year in region.years
        }
        built = build_phase72a_temporal_samples(
            region,
            embeddings=embeddings,
            labels=labels,
            crop_class_code=contract.crop_class_code,
            source_id=contract.source_id,
            source_role=contract.label_role,
            max_history_years=max_history,
            spatial_block_size=spatial_block_size,
        )
        offset = len(sample_rows)
        for row in built["sample_rows"]:
            adjusted = dict(row)
            adjusted["sample_index"] = int(row["sample_index"]) + offset
            sample_rows.append(adjusted)
        for key, value in built["tensors"].items():
            tensor_parts.setdefault(key, []).append(value)
        region_index_parts.append(
            np.full(
                len(built["sample_rows"]), region_index, dtype=np.int8
            )
        )

    tensors = {
        key: np.concatenate(parts, axis=0)
        for key, parts in tensor_parts.items()
    }
    tensors["region_index"] = np.concatenate(
        region_index_parts, axis=0
    )
    review_rows = build_phase72a_review_frame(
        sample_rows, per_stratum=manual_review_per_stratum
    )
    return {
        "phase": "phase72a_temporal_label_package",
        "phase72a_status": "phase72a_label_inputs_ready",
        "region_audits": audits,
        "manifest_rows": manifest_rows,
        "sample_rows": sample_rows,
        "review_rows": review_rows,
        "tensors": tensors,
        "row_counts": {
            "regions": len(audits),
            "manifest_rows": len(manifest_rows),
            "sample_rows": len(sample_rows),
            "review_rows": len(review_rows),
        },
        "recommended_next_step": (
            "Design Phase 72B after checking class-support summaries."
        ),
        "claim_boundary": PHASE72A_CLAIM_BOUNDARY,
    }


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _audit_rows(
    audits: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "region_id": audit["region_id"],
            "status": audit["status"],
            "years_ready": "|".join(
                str(year) for year in audit.get("years_ready", [])
            ),
            "errors": "|".join(
                str(error) for error in audit.get("errors", [])
            ),
            "manifest_rows": len(audit.get("file_manifest_rows", [])),
            "claim_boundary": audit.get(
                "claim_boundary", PHASE72A_CLAIM_BOUNDARY
            ),
        }
        for audit in audits
    ]


def _summary_rows(package: Mapping[str, object]) -> list[dict[str, object]]:
    rows = package["sample_rows"]
    regions = sorted(
        {str(row["region_id"]) for row in rows}
        | {
            str(audit["region_id"])
            for audit in package["region_audits"]
        }
    )
    output = []
    for region_id in regions:
        region_rows = [
            row for row in rows if str(row["region_id"]) == region_id
        ]
        endpoints = (
            ("1y", "y_1y"),
            ("2y", "y_2y"),
            ("continuous_2y", "y_continuous_2y"),
        )
        for horizon, field in endpoints:
            values = [
                int(row[field])
                for row in region_rows
                if row.get(field, "") != ""
            ]
            positives = int(sum(values))
            output.append(
                {
                    "region_id": region_id,
                    "horizon": horizon,
                    "eligible_rows": len(values),
                    "positive_rows": positives,
                    "positive_rate": (
                        f"{positives / len(values):.8f}" if values else ""
                    ),
                    "phase72a_status": package["phase72a_status"],
                    "claim_boundary": package["claim_boundary"],
                }
            )
    return output


def _render_markdown(package: Mapping[str, object]) -> str:
    lines = [
        "# Phase 72A Temporal Label Package",
        "",
        f"- Status: `{package['phase72a_status']}`",
        f"- Row counts: `{package['row_counts']}`",
        f"- Next step: {package['recommended_next_step']}",
        "",
        "## Region Audits",
        "",
    ]
    for audit in package["region_audits"]:
        lines.append(
            f"- `{audit['region_id']}`: `{audit['status']}`; "
            f"years={audit.get('years_ready', [])}"
        )
        for error in audit.get("errors", []):
            lines.append(f"  - Blocker: {error}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(package["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_phase72a_temporal_label_package_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_csv": output / "phase72a_label_manifest.csv",
        "audit_csv": output / "phase72a_region_label_audit.csv",
        "sample_index_csv": output
        / "phase72a_temporal_sample_index.csv",
        "sample_tensors_npz": output / "phase72a_temporal_samples.npz",
        "review_frame_csv": output / "phase72a_manual_review_frame.csv",
        "summary_csv": output / "phase72a_package_summary.csv",
        "package_json": output / "phase72a_temporal_label_package.json",
        "package_md": output / "phase72a_temporal_label_package.md",
    }
    _write_rows(
        paths["manifest_csv"], package["manifest_rows"], MANIFEST_FIELDS
    )
    _write_rows(
        paths["audit_csv"],
        _audit_rows(package["region_audits"]),
        AUDIT_FIELDS,
    )
    _write_rows(
        paths["sample_index_csv"], package["sample_rows"], SAMPLE_FIELDS
    )
    _write_rows(
        paths["review_frame_csv"],
        package["review_rows"],
        REVIEW_FRAME_FIELDS,
    )
    _write_rows(
        paths["summary_csv"], _summary_rows(package), SUMMARY_FIELDS
    )
    np.savez_compressed(paths["sample_tensors_npz"], **package["tensors"])
    preview = {
        key: value
        for key, value in package.items()
        if key not in {"sample_rows", "review_rows", "tensors"}
    }
    paths["package_json"].write_text(
        json.dumps(preview, indent=2), encoding="utf-8"
    )
    paths["package_md"].write_text(
        _render_markdown(package), encoding="utf-8"
    )
    return paths
