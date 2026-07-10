from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import re


_BLOCK_PATTERN = re.compile(
    r"^(?P<region>.+)_br(?P<br>\d+)_bc(?P<bc>\d+)$"
)


def _fold_for_block(block_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(block_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(folds)


def _parse_block(block_id: str) -> tuple[str, int, int]:
    match = _BLOCK_PATTERN.match(str(block_id))
    if match is None:
        raise ValueError(f"Invalid Phase 72B spatial block: {block_id}")
    return (
        str(match.group("region")),
        int(match.group("br")),
        int(match.group("bc")),
    )


def _indexes(
    rows: Sequence[Mapping[str, object]],
    *,
    regions: set[str] | None = None,
    years: set[int] | None = None,
    blocks: set[str] | None = None,
) -> list[int]:
    output = []
    for index, row in enumerate(rows):
        if regions is not None and str(row["region_id"]) not in regions:
            continue
        if years is not None and int(row["origin_year"]) not in years:
            continue
        if blocks is not None and str(row["spatial_block_id"]) not in blocks:
            continue
        output.append(index)
    return output


def _class_counts(
    rows: Sequence[Mapping[str, object]], indexes: Sequence[int]
) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for index in indexes:
        if "conversion_1y" not in rows[index]:
            continue
        value = int(rows[index]["conversion_1y"])
        if value in (0, 1):
            counts[str(value)] += 1
    return counts


def _axis(
    rows: Sequence[Mapping[str, object]],
    train: list[int],
    validation: list[int],
    test: list[int],
    *,
    train_blocks: Sequence[str] = (),
    test_blocks: Sequence[str] = (),
    buffer_blocks: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_block_ids": sorted(set(train_blocks)),
        "test_block_ids": sorted(set(test_blocks)),
        "buffer_block_ids": sorted(set(buffer_blocks)),
        "class_counts": {
            "train": _class_counts(rows, train),
            "validation": _class_counts(rows, validation),
            "test": _class_counts(rows, test),
        },
    }


def build_phase72b_split_registry(
    sample_rows: Sequence[Mapping[str, object]],
    *,
    train_years: Sequence[int],
    validation_year: int,
    test_year: int,
    folds: int,
    buffer_rings: int,
) -> dict[str, dict[str, object]]:
    if int(folds) <= 1:
        raise ValueError("Phase 72B requires at least two spatial folds")
    train_year_set = {int(value) for value in train_years}
    validation_year_set = {int(validation_year)}
    test_year_set = {int(test_year)}
    regions = {str(row["region_id"]) for row in sample_rows}
    required_regions = {"bishan", "dongxing"}
    if not required_regions.issubset(regions):
        raise ValueError("Phase 72B requires Bishan and Dongxing rows")

    registry: dict[str, dict[str, object]] = {}
    registry["pooled_temporal"] = _axis(
        sample_rows,
        _indexes(sample_rows, years=train_year_set),
        _indexes(sample_rows, years=validation_year_set),
        _indexes(sample_rows, years=test_year_set),
    )
    for source_region, target_region in (
        ("bishan", "dongxing"),
        ("dongxing", "bishan"),
    ):
        axis_id = f"{source_region}_to_{target_region}"
        registry[axis_id] = _axis(
            sample_rows,
            _indexes(
                sample_rows,
                regions={source_region},
                years=train_year_set,
            ),
            _indexes(
                sample_rows,
                regions={source_region},
                years=validation_year_set,
            ),
            _indexes(
                sample_rows,
                regions={target_region},
                years=test_year_set,
            ),
        )

    for region_id in sorted(required_regions):
        region_blocks = sorted(
            {
                str(row["spatial_block_id"])
                for row in sample_rows
                if str(row["region_id"]) == region_id
            }
        )
        coordinates = {
            block_id: _parse_block(block_id)[1:] for block_id in region_blocks
        }
        for fold in range(int(folds)):
            test_blocks = {
                block_id
                for block_id in region_blocks
                if _fold_for_block(block_id, int(folds)) == fold
            }
            buffer_blocks = {
                block_id
                for block_id, (block_row, block_col) in coordinates.items()
                if block_id not in test_blocks
                and any(
                    max(
                        abs(block_row - coordinates[test_id][0]),
                        abs(block_col - coordinates[test_id][1]),
                    )
                    <= int(buffer_rings)
                    for test_id in test_blocks
                )
            }
            train_blocks = set(region_blocks) - test_blocks - buffer_blocks
            registry[f"spatial_{region_id}_fold{fold}"] = _axis(
                sample_rows,
                _indexes(
                    sample_rows,
                    regions={region_id},
                    years=train_year_set,
                    blocks=train_blocks,
                ),
                _indexes(
                    sample_rows,
                    regions={region_id},
                    years=validation_year_set,
                    blocks=train_blocks,
                ),
                _indexes(
                    sample_rows,
                    regions={region_id},
                    years=test_year_set,
                    blocks=test_blocks,
                ),
                train_blocks=sorted(train_blocks),
                test_blocks=sorted(test_blocks),
                buffer_blocks=sorted(buffer_blocks),
            )
    return registry


def audit_phase72b_splits(
    sample_rows: Sequence[Mapping[str, object]],
    registry: Mapping[str, Mapping[str, object]],
    *,
    train_years: Sequence[int],
    validation_year: int,
    test_year: int,
) -> dict[str, object]:
    errors = []
    invalid_spatial_axes = []
    required = {
        "pooled_temporal",
        "bishan_to_dongxing",
        "dongxing_to_bishan",
    }
    missing = sorted(required - set(registry))
    if missing:
        errors.append(f"missing required split axes: {missing}")
    train_year_set = {int(value) for value in train_years}
    for axis_id, axis in registry.items():
        train = [int(value) for value in axis["train"]]
        validation = [int(value) for value in axis["validation"]]
        test = [int(value) for value in axis["test"]]
        if set(train) & set(validation) or set(train) & set(test) or set(
            validation
        ) & set(test):
            errors.append(f"split index overlap: {axis_id}")
        if any(
            int(sample_rows[index]["origin_year"]) not in train_year_set
            for index in train
        ):
            errors.append(f"wrong training year: {axis_id}")
        if any(
            int(sample_rows[index]["origin_year"]) != int(validation_year)
            for index in validation
        ):
            errors.append(f"wrong validation year: {axis_id}")
        if any(
            int(sample_rows[index]["origin_year"]) != int(test_year)
            for index in test
        ):
            errors.append(f"wrong test year: {axis_id}")
        train_blocks = set(axis.get("train_block_ids", []))
        test_blocks = set(axis.get("test_block_ids", []))
        buffer_blocks = set(axis.get("buffer_block_ids", []))
        if train_blocks & (test_blocks | buffer_blocks):
            errors.append(f"spatial buffer leakage: {axis_id}")
        counts = dict(axis.get("class_counts", {}))
        train_counts = dict(counts.get("train", {}))
        validation_counts = dict(counts.get("validation", {}))
        has_development_classes = all(
            int(train_counts.get(str(value), 0)) > 0
            and int(validation_counts.get(str(value), 0)) > 0
            for value in (0, 1)
        )
        if axis_id in required and not has_development_classes:
            errors.append(f"missing development class support: {axis_id}")
        if axis_id.startswith("spatial_") and not has_development_classes:
            invalid_spatial_axes.append(axis_id)
    return {
        "status": "leakage_audit_passed" if not errors else "phase72b_inputs_not_ready",
        "errors": errors,
        "invalid_spatial_axes": sorted(invalid_spatial_axes),
    }
