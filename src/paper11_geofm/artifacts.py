from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .block_schema import summarize_phase2_readiness


CLAIM_BOUNDARY = (
    "The suitability_proxy is derived from latent remote-sensing embeddings "
    "and does not directly measure soil quality, fertility, or irrigation access."
)

BASE_COLUMNS = [
    "region_id",
    "pixel_count",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
]
EMBEDDING_COLUMNS = [f"embedding_mean_{idx:02d}" for idx in range(64)]
METRIC_COLUMNS = [
    "embedding_std_mean",
    "temporal_stability",
    "suitability_proxy",
]
BLOCK_BASE_COLUMNS = [
    "block_id",
    "pixel_count",
    "pixel_weight_sum",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
]


def write_phase1_artifacts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    summary: Mapping[str, object],
) -> dict[str, Path]:
    """Write Phase 1 CSV and JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_table = output_dir / "region_features.csv"
    summary_path = output_dir / "summary.json"

    fieldnames = _fieldnames(rows)
    with region_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    suitability = np.array(
        [float(row["suitability_proxy"]) for row in rows], dtype=np.float64
    )
    output_summary = dict(summary)
    output_summary.update(
        {
            "n_regions": len(rows),
            "region_table": region_table.name,
            "claim_boundary": CLAIM_BOUNDARY,
            "suitability_min": float(suitability.min()) if suitability.size else None,
            "suitability_max": float(suitability.max()) if suitability.size else None,
            "suitability_mean": float(suitability.mean()) if suitability.size else None,
        }
    )
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {"region_table": region_table, "summary": summary_path}


def write_phase2_artifacts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    summary: Mapping[str, object],
) -> dict[str, Path]:
    """Write Phase 2 block-level CSV and JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_table = output_dir / "block_geofm_features.csv"
    summary_path = output_dir / "summary.json"
    fieldnames = _phase2_fieldnames(rows)

    with block_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    suitability = np.array(
        [float(row["suitability_proxy"]) for row in rows if "suitability_proxy" in row],
        dtype=np.float64,
    )
    output_summary = dict(summary)
    output_summary.update(
        {
            "n_blocks": len(rows),
            "block_table": block_table.name,
            "feature_readiness": summarize_phase2_readiness(rows),
            "claim_boundary": CLAIM_BOUNDARY,
            "suitability_min": float(suitability.min()) if suitability.size else None,
            "suitability_max": float(suitability.max()) if suitability.size else None,
            "suitability_mean": float(suitability.mean()) if suitability.size else None,
        }
    )
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {"block_table": block_table, "summary": summary_path}


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    known = BASE_COLUMNS + EMBEDDING_COLUMNS + METRIC_COLUMNS
    extras = sorted({key for row in rows for key in row if key not in known})
    return [field for field in known if any(field in row for row in rows)] + extras


def _phase2_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    known = BLOCK_BASE_COLUMNS + EMBEDDING_COLUMNS + METRIC_COLUMNS
    extras = sorted({key for row in rows for key in row if key not in known})
    return [field for field in known if any(field in row for row in rows)] + extras
