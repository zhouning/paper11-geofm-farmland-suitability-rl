from __future__ import annotations

from collections.abc import Iterable, Mapping


BlockPixel = dict[str, str | int | float]


def validate_block_pixel_mapping(
    rows: Iterable[Mapping[str, object]],
    grid_shape: tuple[int, int],
) -> list[BlockPixel]:
    """Validate a table that maps embedding-grid pixels to planning blocks."""
    rows_count, cols_count = grid_shape
    if rows_count <= 0 or cols_count <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")

    validated: list[BlockPixel] = []
    for index, row in enumerate(rows):
        try:
            block_id = str(row["block_id"])
            pixel_row = int(row["row"])
            pixel_col = int(row["col"])
        except KeyError as exc:
            raise ValueError(
                f"mapping row {index} missing required column {exc.args[0]}"
            ) from exc

        weight = float(row.get("weight", 1.0))
        if not block_id:
            raise ValueError(f"mapping row {index} has empty block_id")
        if (
            pixel_row < 0
            or pixel_row >= rows_count
            or pixel_col < 0
            or pixel_col >= cols_count
        ):
            raise ValueError(
                f"mapping row {index} points outside grid_shape {grid_shape}: "
                f"row={pixel_row}, col={pixel_col}"
            )
        if weight <= 0:
            raise ValueError(f"mapping row {index} has non-positive weight {weight}")

        validated.append(
            {
                "block_id": block_id,
                "row": pixel_row,
                "col": pixel_col,
                "weight": weight,
            }
        )

    if not validated:
        raise ValueError("block pixel mapping must contain at least one row")
    return validated
