from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import re


_BLOCK_PATTERN = re.compile(
    r"^(?P<region>.+)_br(?P<br>\d+)_bc(?P<bc>\d+)$"
)
_SPATIAL_AXIS_PATTERN = re.compile(
    r"^spatial_(?P<region>bishan|dongxing)_fold(?P<fold>\d+)$"
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
    split_indexes = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "region_summary": {
            name: sorted(
                {str(rows[index]["region_id"]) for index in indexes}
            )
            for name, indexes in split_indexes.items()
        },
        "year_summary": {
            name: sorted(
                {int(rows[index]["origin_year"]) for index in indexes}
            )
            for name, indexes in split_indexes.items()
        },
        "train_block_ids": sorted(set(train_blocks)),
        "test_block_ids": sorted(set(test_blocks)),
        "buffer_block_ids": sorted(set(buffer_blocks)),
        "class_counts": {
            "train": _class_counts(rows, train),
            "validation": _class_counts(rows, validation),
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
    if not sample_rows:
        raise ValueError("Phase 72B split registry requires sample rows")
    if int(folds) <= 1:
        raise ValueError("Phase 72B requires at least two spatial folds")
    if int(buffer_rings) < 0:
        raise ValueError("Phase 72B spatial buffer rings cannot be negative")
    if [int(row.get("sample_index", -1)) for row in sample_rows] != list(
        range(len(sample_rows))
    ):
        raise ValueError("Phase 72B split sample indexes must be contiguous")
    train_year_set = {int(value) for value in train_years}
    validation_year_set = {int(validation_year)}
    test_year_set = {int(test_year)}
    if (
        not train_year_set
        or train_year_set & validation_year_set
        or train_year_set & test_year_set
        or validation_year_set & test_year_set
    ):
        raise ValueError("Phase 72B split years must be disjoint")
    regions = {str(row["region_id"]) for row in sample_rows}
    required_regions = {"bishan", "dongxing"}
    if regions != required_regions:
        raise ValueError("Phase 72B requires exact Bishan-Dongxing rows")
    allowed_years = train_year_set | validation_year_set | test_year_set
    for row in sample_rows:
        region_id = str(row["region_id"])
        block_region, _, _ = _parse_block(str(row["spatial_block_id"]))
        if block_region != region_id:
            raise ValueError(
                "Phase 72B spatial block region mismatch: "
                f"{region_id}/{row['spatial_block_id']}"
            )
        if int(row["origin_year"]) not in allowed_years:
            raise ValueError(
                f"Phase 72B split row has undeclared origin year: "
                f"{row['origin_year']}"
            )

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
    spatial_folds: int | None = None,
    control_partition_local: bool = True,
    reuse_phase8_d4_tables: bool = False,
) -> dict[str, object]:
    errors = []
    invalid_spatial_axes = []
    if control_partition_local is not True:
        errors.append("partition-local controls are not required")
    if reuse_phase8_d4_tables is not False:
        errors.append("Phase 8 D4 table reuse is enabled")
    if [int(row.get("sample_index", -1)) for row in sample_rows] != list(
        range(len(sample_rows))
    ):
        errors.append("sample indexes are missing or non-contiguous")
    mandatory_model_axes = {
        "pooled_temporal",
        "bishan_to_dongxing",
        "dongxing_to_bishan",
    }
    required = set(mandatory_model_axes)
    if spatial_folds is not None:
        expected_spatial_axes = {
            f"spatial_{region}_fold{fold}"
            for region in ("bishan", "dongxing")
            for fold in range(int(spatial_folds))
        }
        required.update(expected_spatial_axes)
        unexpected_spatial = sorted(
            {
                str(axis_id)
                for axis_id in registry
                if str(axis_id).startswith("spatial_")
            }
            - expected_spatial_axes
        )
        if unexpected_spatial:
            errors.append(
                f"unexpected spatial split axes: {unexpected_spatial}"
            )
    missing = sorted(required - set(registry))
    if missing:
        errors.append(f"missing required split axes: {missing}")
    train_year_set = {int(value) for value in train_years}
    allowed_years = train_year_set | {
        int(validation_year),
        int(test_year),
    }
    if any(
        int(row["origin_year"]) not in allowed_years for row in sample_rows
    ):
        errors.append("sample rows contain undeclared origin years")

    for axis_id, axis in registry.items():
        split_indexes: dict[str, list[int]] = {}
        for split_name in ("train", "validation", "test"):
            raw_indexes = axis.get(split_name)
            if not isinstance(raw_indexes, list):
                errors.append(f"missing split indexes: {axis_id}/{split_name}")
                split_indexes[split_name] = []
                continue
            try:
                indexes = [int(value) for value in raw_indexes]
            except (TypeError, ValueError):
                errors.append(f"invalid split index: {axis_id}/{split_name}")
                split_indexes[split_name] = []
                continue
            if len(indexes) != len(set(indexes)):
                errors.append(f"duplicate split index: {axis_id}/{split_name}")
            invalid = [
                index
                for index in indexes
                if index < 0 or index >= len(sample_rows)
            ]
            if invalid:
                errors.append(
                    f"split index out of range: {axis_id}/{split_name}/"
                    f"{invalid[:5]}"
                )
            split_indexes[split_name] = [
                index for index in indexes if index not in invalid
            ]
        train = split_indexes["train"]
        validation = split_indexes["validation"]
        test = split_indexes["test"]
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
        actual_region_summary = {
            name: sorted(
                {
                    str(sample_rows[index]["region_id"])
                    for index in indexes
                }
            )
            for name, indexes in split_indexes.items()
        }
        actual_year_summary = {
            name: sorted(
                {
                    int(sample_rows[index]["origin_year"])
                    for index in indexes
                }
            )
            for name, indexes in split_indexes.items()
        }
        if axis.get("region_summary") != actual_region_summary:
            errors.append(f"region summary mismatch: {axis_id}")
        if axis.get("year_summary") != actual_year_summary:
            errors.append(f"year summary mismatch: {axis_id}")

        transfer_regions = {
            "bishan_to_dongxing": ("bishan", "dongxing"),
            "dongxing_to_bishan": ("dongxing", "bishan"),
        }
        if axis_id in transfer_regions:
            source_region, target_region = transfer_regions[axis_id]
            if any(
                str(sample_rows[index]["region_id"]) != source_region
                for index in [*train, *validation]
            ) or any(
                str(sample_rows[index]["region_id"]) != target_region
                for index in test
            ):
                errors.append(f"transfer region leakage: {axis_id}")

        train_blocks = set(axis.get("train_block_ids", []))
        test_blocks = set(axis.get("test_block_ids", []))
        buffer_blocks = set(axis.get("buffer_block_ids", []))
        if train_blocks & (test_blocks | buffer_blocks):
            errors.append(f"spatial buffer leakage: {axis_id}")
        spatial_match = _SPATIAL_AXIS_PATTERN.match(str(axis_id))
        if str(axis_id).startswith("spatial_") and spatial_match is None:
            errors.append(f"invalid spatial axis identity: {axis_id}")
        if spatial_match is not None:
            spatial_region = str(spatial_match.group("region"))
            if any(
                str(sample_rows[index]["region_id"]) != spatial_region
                for index in [*train, *validation, *test]
            ):
                errors.append(f"spatial region leakage: {axis_id}")
            if any(
                str(sample_rows[index]["spatial_block_id"])
                not in train_blocks
                for index in [*train, *validation]
            ):
                errors.append(f"spatial train block mismatch: {axis_id}")
            if any(
                str(sample_rows[index]["spatial_block_id"])
                not in test_blocks
                for index in test
            ):
                errors.append(f"spatial test block mismatch: {axis_id}")
        counts = dict(axis.get("class_counts", {}))
        if any("conversion_1y" in row for row in sample_rows):
            recomputed_counts = {
                name: _class_counts(sample_rows, indexes)
                for name, indexes in split_indexes.items()
                if name in {"train", "validation"}
            }
            if counts != recomputed_counts:
                errors.append(f"class count mismatch: {axis_id}")
        train_counts = dict(counts.get("train", {}))
        validation_counts = dict(counts.get("validation", {}))
        has_development_classes = all(
            int(train_counts.get(str(value), 0)) > 0
            and int(validation_counts.get(str(value), 0)) > 0
            for value in (0, 1)
        )
        if axis_id in mandatory_model_axes and not has_development_classes:
            errors.append(f"missing development class support: {axis_id}")
        if axis_id.startswith("spatial_") and not has_development_classes:
            invalid_spatial_axes.append(axis_id)

    pooled = registry.get("pooled_temporal")
    if isinstance(pooled, Mapping):
        try:
            pooled_indexes = {
                int(value)
                for split_name in ("train", "validation", "test")
                for value in pooled.get(split_name, [])
            }
        except (TypeError, ValueError):
            pooled_indexes = set()
        if pooled_indexes != set(range(len(sample_rows))):
            errors.append("pooled split indexes do not cover all sample rows")
    return {
        "status": "leakage_audit_passed" if not errors else "phase72b_inputs_not_ready",
        "errors": errors,
        "invalid_spatial_axes": sorted(invalid_spatial_axes),
    }
