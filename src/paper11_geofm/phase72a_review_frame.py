from __future__ import annotations

import hashlib


REVIEW_FIELDS = (
    "review_label",
    "review_source",
    "review_source_id",
    "review_date",
    "review_confidence",
    "reviewer_note",
)


def build_phase72a_review_frame(
    sample_rows: list[dict[str, object]],
    *,
    per_stratum: int = 20,
    seed: int = 72,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, int, str], list[dict[str, object]]
    ] = {}
    for row in sample_rows:
        transition = (
            "persistent_crop"
            if int(row["y_1y"]) == 1
            else "crop_conversion"
        )
        key = (
            str(row["region_id"]),
            int(row["origin_year"]),
            transition,
        )
        candidate = dict(row)
        candidate["transition_type"] = transition
        grouped.setdefault(key, []).append(candidate)

    output = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:{row['region_id']}:{row['unit_id']}:"
                    f"{row['origin_year']}"
                ).encode("utf-8")
            ).hexdigest(),
        )[: int(per_stratum)]
        for row in ordered:
            review = {
                field: row[field]
                for field in (
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
                )
            }
            review.update({field: "" for field in REVIEW_FIELDS})
            output.append(review)
    return output
