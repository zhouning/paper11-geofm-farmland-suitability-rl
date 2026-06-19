from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import json
import statistics
import warnings
from os import PathLike
from pathlib import Path

import numpy as np

from .block_schema import EMBEDDING_COLUMNS, EXPLICIT_FEATURE_COLUMNS
from .drl_inputs import VariantInput, load_variant_input
from .padded_heldout_policy import (
    SUMMARY_FIELDNAMES,
    Phase25PaddedTileEnv,
    _dependency_metadata,
    _evaluate_baseline_policy,
    _evaluate_trained_policy,
    _normalize_seeds,
    _select_train_eval_tiles,
    _store_trace,
)
from .phase28_representation_controls import PHASE28_DEFAULT_VARIANTS
from .tiled_inputs import load_tiled_variant_input


PHASE30_CLAIM_BOUNDARY = (
    "Phase 30 is a bounded representation-only ablation under the existing "
    "Bishan base-reward held-out protocol. It does not validate suitability "
    "reward, does not test B2/B3, does not test cross-region transfer, does "
    "not prove that normalization is generally beneficial, and does not "
    "support submission-level planning-performance claims."
)

PHASE30_NORMALIZED_VARIANTS = ("N1Z", "N1ZR")
PHASE30_DEFAULT_VARIANTS = ("B0", "B1", "N1Z", "N1ZR", *PHASE28_DEFAULT_VARIANTS[2:])
PHASE30_FOCAL_VARIANTS = ("N1Z", "N1ZR")
PHASE30_COMPARATORS = ("B1", "B0", "D4P8", "D4P16")

PHASE30_DELTA_FIELDNAMES = [
    "variant_id",
    "comparator_variant_id",
    "eval_tile_id",
    "seed",
    "variant_reward",
    "comparator_reward",
    "variant_minus_comparator_reward",
    "variant_improves_comparator",
    "train_timesteps",
    "eval_max_steps",
    "claim_boundary",
]


def build_phase30_normalized_b1_controls(
    phase2_output_dir: Path | str,
) -> dict[str, object]:
    source_b0 = load_variant_input(phase2_output_dir, "B0")
    source_b1 = load_variant_input(phase2_output_dir, "B1")
    if source_b0.block_ids != source_b1.block_ids:
        raise ValueError("Phase 30 requires aligned B0 and B1 block IDs")

    explicit_matrix = _matrix_for_columns(source_b1, EXPLICIT_FEATURE_COLUMNS)
    embedding_matrix = _matrix_for_columns(source_b1, EMBEDDING_COLUMNS)
    zscore_matrix = _column_zscore(embedding_matrix)
    zscore_row_l2_matrix = _row_l2_normalize(zscore_matrix)

    variant_tables = {
        "N1Z": _build_variant_rows(
            source_b1.block_ids,
            explicit_matrix,
            zscore_matrix,
        ),
        "N1ZR": _build_variant_rows(
            source_b1.block_ids,
            explicit_matrix,
            zscore_row_l2_matrix,
        ),
    }
    manifest = _build_normalized_manifest(variant_tables)
    return {
        "phase": "phase30_normalized_b1_controls",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "source_variants": {
            "B0": _source_summary(source_b0),
            "B1": _source_summary(source_b1),
        },
        "variant_ids": list(PHASE30_NORMALIZED_VARIANTS),
        "manifest": manifest,
        "variant_tables": variant_tables,
        "normalization_summary": {
            "N1Z": _embedding_summary(zscore_matrix),
            "N1ZR": _embedding_summary(zscore_row_l2_matrix),
        },
        "claim_boundary": PHASE30_CLAIM_BOUNDARY,
    }


def write_phase30_normalized_b1_controls(
    controls: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = controls.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Phase 30 normalized controls are missing a manifest")
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Phase 30 normalized manifest is missing variants")
    variant_tables = controls.get("variant_tables")
    if not isinstance(variant_tables, Mapping):
        raise ValueError("Phase 30 normalized controls are missing variant tables")

    table_paths: dict[str, Path] = {}
    for variant_id in PHASE30_NORMALIZED_VARIANTS:
        variant = variants.get(variant_id)
        if not isinstance(variant, Mapping):
            raise ValueError(f"Phase 30 manifest is missing variant {variant_id}")
        rows = variant_tables.get(variant_id)
        if not isinstance(rows, list):
            raise ValueError(f"Phase 30 controls are missing rows for {variant_id}")
        required_columns = variant.get("required_columns")
        if not isinstance(required_columns, list):
            raise ValueError(f"Phase 30 variant is missing required columns: {variant_id}")
        table_path = output_path / str(variant["feature_table"])
        _write_variant_csv(table_path, rows, [str(item) for item in required_columns])
        table_paths[variant_id] = table_path

    manifest_path = output_path / "experiment_variants.json"
    summary_path = output_path / "phase30_normalized_b1_controls_summary.json"
    manifest_path.write_text(
        json.dumps(_json_ready(dict(manifest)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_payload = {
        key: value for key, value in dict(controls).items() if key != "variant_tables"
    }
    summary_payload["artifacts"] = {
        "manifest": manifest_path.name,
        "variant_tables": {variant_id: path.name for variant_id, path in table_paths.items()},
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "variant_tables": table_paths,
    }


def run_phase30_normalized_b1_ablation(
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    tile_index_csv: Path | str,
    output_dir: Path | str,
    normalized_controls_dir: Path | str | None = None,
    existing_control_summary_csv: Path | str | None = None,
    variants: Sequence[str] | str = PHASE30_DEFAULT_VARIANTS,
    train_tile_id: str | None = None,
    eval_tile_ids: Sequence[str] | str | None = None,
    max_eval_tiles: int = 3,
    total_timesteps: int = 32,
    eval_max_steps: int = 4,
    seeds: Sequence[int | str] | str | int | None = (0, 1, 2),
) -> dict[str, object]:
    output_path = Path(output_dir)
    controls_dir = (
        Path(normalized_controls_dir)
        if normalized_controls_dir is not None
        else output_path / "derived_normalized_controls"
    )
    controls = build_phase30_normalized_b1_controls(phase2_output_dir)
    control_paths = write_phase30_normalized_b1_controls(controls, controls_dir)

    normalized_variants = _normalize_phase30_variants(variants)
    variants_to_train = (
        [variant for variant in normalized_variants if variant in PHASE30_NORMALIZED_VARIANTS]
        if existing_control_summary_csv is not None
        else list(normalized_variants)
    )
    normalized_seeds = _normalize_seeds(seeds)
    selected = _select_train_eval_tiles(
        Path(tile_index_csv),
        train_tile_id=train_tile_id,
        eval_tile_ids=eval_tile_ids,
        max_eval_tiles=max_eval_tiles,
    )
    eval_ids = list(selected["eval_tile_ids"])
    selected_counts = dict(selected["selected_tile_block_counts"])
    max_blocks = max(int(selected_counts[tile_id]) for tile_id in selected_counts)
    eval_tile_ranks = {
        str(tile_id): rank for rank, tile_id in enumerate(eval_ids, start=1)
    }
    seed_ranks = {
        str(seed): rank for rank, seed in enumerate(normalized_seeds, start=1)
    }
    train_id = str(selected["train_tile_id"])

    contract = {
        "phase": "phase30_normalized_b1_ablation",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "phase8_output_dir": str(Path(phase8_output_dir)),
        "normalized_controls_dir": str(controls_dir),
        "existing_control_summary_csv": str(Path(existing_control_summary_csv))
        if existing_control_summary_csv is not None
        else None,
        "tile_index_csv": str(Path(tile_index_csv)),
        "variants": normalized_variants,
        "variants_to_train": variants_to_train,
        "variant_source_dirs": _phase30_variant_source_dirs(
            normalized_variants,
            phase2_output_dir=phase2_output_dir,
            phase8_output_dir=phase8_output_dir,
            normalized_controls_dir=controls_dir,
        ),
        "train_tile_id": train_id,
        "train_tile_ids": [train_id],
        "eval_tile_ids": eval_ids,
        "eval_tile_count": len(eval_ids),
        "eval_tile_ranks": eval_tile_ranks,
        "selected_tile_block_counts": selected_counts,
        "train_tile_selection": selected["train_tile_selection"],
        "eval_tile_selection": selected["eval_tile_selection"],
        "max_blocks": int(max_blocks),
        "total_timesteps": int(total_timesteps),
        "eval_max_steps": int(eval_max_steps),
        "seeds": normalized_seeds,
        "seed_count": len(normalized_seeds),
        "seed_ranks": seed_ranks,
        "normalized_control_artifacts": _json_ready(control_paths),
        "claim_boundary": PHASE30_CLAIM_BOUNDARY,
    }

    summaries: list[dict[str, object]] = []
    traces: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {
        "trained_policy": {},
        "first_valid": {},
        "seeded_random": {},
    }

    for variant_id in contract["variants_to_train"]:
        for seed in contract["seeds"]:
            train_tiled = _load_phase30_tiled_variant_input(
                contract,
                str(contract["train_tile_id"]),
                str(variant_id),
            )
            train_env = Phase25PaddedTileEnv(
                train_tiled,
                max_blocks=int(contract["max_blocks"]),
                max_steps=int(contract["total_timesteps"]),
            )
            train_env.reset(seed=int(seed))
            model = _train_maskable_ppo_model(
                train_env,
                seed=int(seed),
                total_timesteps=int(contract["total_timesteps"]),
            )

            for eval_tile_id in contract["eval_tile_ids"]:
                eval_tiled = _load_phase30_tiled_variant_input(
                    contract,
                    str(eval_tile_id),
                    str(variant_id),
                )
                eval_tile_rank = int(contract["eval_tile_ranks"][str(eval_tile_id)])
                seed_rank = int(contract["seed_ranks"][str(int(seed))])
                train_n_blocks = int(
                    contract["selected_tile_block_counts"][
                        str(contract["train_tile_id"])
                    ]
                )
                trained_summary, trained_steps = _evaluate_trained_policy(
                    model,
                    eval_tiled,
                    train_tile_id=str(contract["train_tile_id"]),
                    train_n_blocks=train_n_blocks,
                    max_blocks=int(contract["max_blocks"]),
                    eval_tile_rank=eval_tile_rank,
                    phase25_seed_rank=seed_rank,
                    eval_max_steps=int(contract["eval_max_steps"]),
                    train_timesteps=int(contract["total_timesteps"]),
                    seed=int(seed),
                )
                trained_summary["claim_boundary"] = PHASE30_CLAIM_BOUNDARY
                summaries.append(trained_summary)
                _store_trace(
                    traces,
                    "trained_policy",
                    str(variant_id),
                    str(eval_tile_id),
                    int(seed),
                    trained_steps,
                )

                for policy_id in ("first_valid", "seeded_random"):
                    baseline_summary, baseline_steps = _evaluate_baseline_policy(
                        eval_tiled,
                        policy_id=policy_id,
                        train_tile_id=str(contract["train_tile_id"]),
                        train_n_blocks=train_n_blocks,
                        max_blocks=int(contract["max_blocks"]),
                        eval_tile_rank=eval_tile_rank,
                        phase25_seed_rank=seed_rank,
                        eval_max_steps=int(contract["eval_max_steps"]),
                        train_timesteps=int(contract["total_timesteps"]),
                        seed=int(seed),
                    )
                    baseline_summary["claim_boundary"] = PHASE30_CLAIM_BOUNDARY
                    summaries.append(baseline_summary)
                    _store_trace(
                        traces,
                        policy_id,
                        str(variant_id),
                        str(eval_tile_id),
                        int(seed),
                        baseline_steps,
                    )

    analysis_rows = _merged_existing_and_new_summary_rows(
        existing_control_summary_csv,
        summaries,
        variants=normalized_variants,
        eval_tile_ids=eval_ids,
        seeds=normalized_seeds,
    )
    analysis = build_phase30_normalized_b1_analysis(
        analysis_rows,
        metadata={
            "variants": normalized_variants,
            "eval_tile_ids": eval_ids,
            "seeds": normalized_seeds,
            "train_timesteps": int(total_timesteps),
            "eval_max_steps": int(eval_max_steps),
        },
    )
    return {
        **contract,
        **analysis,
        "training_completed": True,
        "all_evaluations_completed": all(
            bool(row["terminated"]) or bool(row["truncated"]) for row in analysis_rows
        ),
        "new_summary_count": len(summaries),
        "summary_count": len(analysis_rows),
        "summaries": analysis_rows,
        "new_summaries": summaries,
        "traces": traces,
        "dependencies": _dependency_metadata(),
        "claim_boundary": PHASE30_CLAIM_BOUNDARY,
    }


def build_phase30_normalized_b1_analysis(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    rows = _load_summary_rows(summary_rows_or_csv)
    trained_rows = [row for row in rows if str(row.get("row_type", "")) == "trained_policy"]
    metadata_map = {} if metadata is None else dict(metadata)
    variants = _metadata_string_list(
        metadata_map,
        "variants",
        fallback=_unique_strings(trained_rows, "variant_id"),
    )
    eval_tile_ids = _metadata_string_list(
        metadata_map,
        "eval_tile_ids",
        fallback=_unique_strings(trained_rows, "eval_tile_id"),
    )
    seeds = _metadata_int_list(
        metadata_map,
        "seeds",
        fallback=_unique_ints(trained_rows, "seed"),
    )
    train_timesteps = _metadata_int(
        metadata_map,
        ("train_timesteps", "total_timesteps"),
        trained_rows,
        "train_timesteps",
    )
    eval_max_steps = _metadata_int(
        metadata_map,
        ("eval_max_steps",),
        trained_rows,
        "eval_max_steps",
    )
    coverage_issues = _coverage_issues(
        trained_rows,
        variants=variants,
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    delta_rows = _phase30_delta_rows(
        trained_rows,
        focal_variants=[variant for variant in PHASE30_FOCAL_VARIANTS if variant in variants],
        comparators=[variant for variant in PHASE30_COMPARATORS if variant in variants],
        eval_tile_ids=eval_tile_ids,
        seeds=seeds,
    )
    learned_policy = _policy_summary(
        trained_rows,
        variants=variants,
        delta_rows=delta_rows,
    )
    status = _phase30_status(learned_policy, coverage_issues)
    analysis: dict[str, object] = {
        "phase": "phase30_normalized_b1_analysis",
        "variants": variants,
        "eval_tile_ids": eval_tile_ids,
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "eval_max_steps": eval_max_steps,
        "source_rows": rows,
        "main_summary_rows": _main_summary_rows(rows),
        "delta_rows": delta_rows,
        "learned_policy": learned_policy,
        "coverage_issues": coverage_issues,
        "phase30_normalized_b1_status": status,
        "interpretation": _phase30_interpretation(status),
        "claim_boundary": PHASE30_CLAIM_BOUNDARY,
    }
    if metadata is not None:
        analysis["metadata"] = metadata_map
    return analysis


def _merged_existing_and_new_summary_rows(
    existing_control_summary_csv: Path | str | None,
    new_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    merged = []
    if existing_control_summary_csv is not None:
        existing_rows = _load_summary_rows(existing_control_summary_csv)
        needed_existing_variants = {
            str(variant)
            for variant in variants
            if str(variant) not in PHASE30_NORMALIZED_VARIANTS
        }
        needed_tiles = {str(tile_id) for tile_id in eval_tile_ids}
        needed_seeds = {int(seed) for seed in seeds}
        for row in existing_rows:
            if str(row.get("variant_id", "")) not in needed_existing_variants:
                continue
            if str(row.get("eval_tile_id", "")) not in needed_tiles:
                continue
            try:
                seed = _int_value(row, "seed")
            except ValueError:
                continue
            if seed not in needed_seeds:
                continue
            merged.append(dict(row))
    merged.extend(dict(row) for row in new_rows)
    return merged


def write_phase30_normalized_b1_artifacts(
    protocol_or_analysis: Mapping[str, object],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "phase30_normalized_b1_summary.csv"
    traces_path = output_path / "phase30_normalized_b1_traces.json"
    comparison_path = output_path / "phase30_normalized_b1_comparison.json"
    delta_path = output_path / "phase30_normalized_b1_delta_table.csv"
    readiness_path = output_path / "phase30_normalized_b1_readiness.md"

    _write_summary_csv(
        summary_path,
        protocol_or_analysis.get("summaries", protocol_or_analysis.get("source_rows", [])),
    )
    traces_path.write_text(
        json.dumps(_json_ready(protocol_or_analysis.get("traces", {})), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison = {
        key: value
        for key, value in dict(protocol_or_analysis).items()
        if key not in {"summaries", "traces", "source_rows"}
    }
    comparison_path.write_text(
        json.dumps(_json_ready(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv_mapping_rows(
        delta_path,
        PHASE30_DELTA_FIELDNAMES,
        protocol_or_analysis.get("delta_rows"),
        "delta_rows",
    )
    readiness_path.write_text(
        _phase30_readiness_markdown(protocol_or_analysis),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "traces_json": traces_path,
        "comparison_json": comparison_path,
        "delta_csv": delta_path,
        "readiness_md": readiness_path,
    }


def _matrix_for_columns(
    variant_input: VariantInput,
    columns: Sequence[str],
) -> np.ndarray:
    indexes = [variant_input.feature_columns.index(column) for column in columns]
    return variant_input.state_matrix[:, indexes].astype(float, copy=True)


def _column_zscore(matrix: np.ndarray) -> np.ndarray:
    if not matrix.size:
        return np.zeros_like(matrix, dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    return (matrix - means) / safe_stds


def _row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if not matrix.size:
        return np.zeros_like(matrix, dtype=float)
    row_norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(row_norms > 0.0, row_norms, 1.0)
    return matrix / safe_norms[:, None]


def _build_variant_rows(
    block_ids: Sequence[str],
    explicit_matrix: np.ndarray,
    embedding_matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for row_index, block_id in enumerate(block_ids):
        row: dict[str, object] = {"block_id": str(block_id)}
        for column_index, column in enumerate(EXPLICIT_FEATURE_COLUMNS):
            row[column] = float(explicit_matrix[row_index, column_index])
        for column_index, column in enumerate(EMBEDDING_COLUMNS):
            row[column] = float(embedding_matrix[row_index, column_index])
        rows.append(row)
    return rows


def _build_normalized_manifest(
    variant_tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    required_columns = list(EXPLICIT_FEATURE_COLUMNS) + list(EMBEDDING_COLUMNS)
    return {
        "claim_boundary": PHASE30_CLAIM_BOUNDARY,
        "variants": {
            "N1Z": {
                "description": (
                    "Explicit planning features plus column-centered, "
                    "standard-deviation-scaled B1 GeoFM embeddings."
                ),
                "state_groups": [
                    "explicit_planning_features",
                    "column_zscore_geofm_embedding",
                ],
                "reward": "base_planning_reward",
                "required_columns": required_columns,
                "ready": True,
                "missing": [],
                "feature_table": "variant_N1Z_features.csv",
                "row_count": len(variant_tables.get("N1Z", [])),
            },
            "N1ZR": {
                "description": (
                    "Explicit planning features plus column-centered, "
                    "standard-deviation-scaled, row-L2-normalized B1 GeoFM embeddings."
                ),
                "state_groups": [
                    "explicit_planning_features",
                    "column_zscore_row_l2_geofm_embedding",
                ],
                "reward": "base_planning_reward",
                "required_columns": required_columns,
                "ready": True,
                "missing": [],
                "feature_table": "variant_N1ZR_features.csv",
                "row_count": len(variant_tables.get("N1ZR", [])),
            },
        },
    }


def _embedding_summary(matrix: np.ndarray) -> dict[str, object]:
    row_norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.asarray([], dtype=float)
    return {
        "n_blocks": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "mean_column_mean": _round_float(float(np.mean(matrix.mean(axis=0)))) if matrix.size else 0.0,
        "mean_column_std": _round_float(float(np.mean(matrix.std(axis=0)))) if matrix.size else 0.0,
        "mean_row_l2_norm": _round_float(float(np.mean(row_norms))) if row_norms.size else 0.0,
        "std_row_l2_norm": _round_float(float(np.std(row_norms))) if row_norms.size else 0.0,
    }


def _source_summary(variant_input: VariantInput) -> dict[str, object]:
    return {
        "variant_id": variant_input.variant_id,
        "n_blocks": len(variant_input.block_ids),
        "n_features": len(variant_input.feature_columns),
        "source_table": str(variant_input.source_table),
    }


def _write_variant_csv(
    path: Path,
    rows: list[dict[str, object]],
    required_columns: Sequence[str],
) -> None:
    fieldnames = ["block_id", *required_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _normalize_phase30_variants(variants: Sequence[str] | str) -> list[str]:
    if isinstance(variants, str):
        values = [part.strip() for part in variants.split(",")]
    else:
        values = [str(item).strip() for item in variants]
    normalized = [item.upper() for item in values if item]
    if not normalized:
        raise ValueError("At least one Phase 30 variant must be requested")
    supported = {"B0", "B1", "N1Z", "N1ZR", "D2", "D3", "D4P8", "D4P16"}
    unsupported = [variant for variant in normalized if variant not in supported]
    if unsupported:
        raise ValueError(f"unsupported Phase 30 variants: {unsupported}")
    if "B1" not in normalized:
        raise ValueError("Phase 30 normalized-B1 ablation requires B1")
    if not any(variant in normalized for variant in PHASE30_FOCAL_VARIANTS):
        raise ValueError("Phase 30 requires at least one normalized B1 variant")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Phase 30 variants must be unique")
    return normalized


def _phase30_variant_source_dirs(
    variants: Sequence[str],
    phase2_output_dir: Path | str,
    phase8_output_dir: Path | str,
    normalized_controls_dir: Path | str,
) -> dict[str, str]:
    source_dirs: dict[str, str] = {}
    for variant_id in variants:
        if variant_id in {"B0", "B1"}:
            source_dirs[str(variant_id)] = str(Path(phase2_output_dir))
        elif variant_id in PHASE30_NORMALIZED_VARIANTS:
            source_dirs[str(variant_id)] = str(Path(normalized_controls_dir))
        else:
            source_dirs[str(variant_id)] = str(Path(phase8_output_dir))
    return source_dirs


def _load_phase30_tiled_variant_input(
    contract: Mapping[str, object],
    tile_id: str,
    variant_id: str,
):
    variant_source_dirs = contract.get("variant_source_dirs")
    if not isinstance(variant_source_dirs, Mapping):
        raise ValueError("Phase 30 contract is missing variant source routing")
    source_dir = variant_source_dirs.get(variant_id)
    if source_dir is None:
        raise ValueError(f"Phase 30 contract has no source for variant {variant_id}")
    return load_tiled_variant_input(
        source_dir,
        str(contract["tile_index_csv"]),
        tile_id,
        variant_id=variant_id,
    )


def _train_maskable_ppo_model(
    train_env: Phase25PaddedTileEnv,
    seed: int,
    total_timesteps: int,
):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 30 normalized-B1 ablation requires stable-baselines3 and sb3-contrib"
        ) from exc
    if not is_masking_supported(train_env):
        raise ValueError("Phase 30 train env does not expose action_masks")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="XPU device count is zero!.*",
            category=UserWarning,
        )
        model.learn(total_timesteps=int(total_timesteps))
    return model


def _load_summary_rows(
    summary_rows_or_csv: Path | str | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(summary_rows_or_csv, (str, PathLike)):
        path = Path(summary_rows_or_csv)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 30 summary CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, object]] = []
    for row in summary_rows_or_csv:
        if not isinstance(row, Mapping):
            raise ValueError("Phase 30 summary rows must be objects")
        rows.append(dict(row))
    return rows


def _phase30_delta_rows(
    rows: list[dict[str, object]],
    focal_variants: Sequence[str],
    comparators: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("eval_tile_id", "")), _int_value(row, "seed"))
        indexed.setdefault(key, {})
        indexed[key][str(row.get("variant_id", ""))] = row

    delta_rows: list[dict[str, object]] = []
    for eval_tile_id in eval_tile_ids:
        for seed in seeds:
            by_variant = indexed.get((str(eval_tile_id), int(seed)), {})
            for variant_id in focal_variants:
                variant_row = by_variant.get(str(variant_id))
                if variant_row is None:
                    continue
                variant_reward = _float_value(variant_row, "total_contract_reward")
                for comparator in comparators:
                    if comparator == variant_id:
                        continue
                    comparator_row = by_variant.get(str(comparator))
                    if comparator_row is None:
                        continue
                    comparator_reward = _float_value(
                        comparator_row,
                        "total_contract_reward",
                    )
                    delta = _round_float(variant_reward - comparator_reward)
                    delta_rows.append(
                        {
                            "variant_id": variant_id,
                            "comparator_variant_id": comparator,
                            "eval_tile_id": str(eval_tile_id),
                            "seed": int(seed),
                            "variant_reward": _round_float(variant_reward),
                            "comparator_reward": _round_float(comparator_reward),
                            "variant_minus_comparator_reward": delta,
                            "variant_improves_comparator": delta > 0.0,
                            "train_timesteps": _optional_int(
                                variant_row,
                                "train_timesteps",
                                fallback=_optional_int(comparator_row, "train_timesteps"),
                            ),
                            "eval_max_steps": _optional_int(
                                variant_row,
                                "eval_max_steps",
                                fallback=_optional_int(comparator_row, "eval_max_steps"),
                            ),
                            "claim_boundary": PHASE30_CLAIM_BOUNDARY,
                        }
                    )
    return delta_rows


def _policy_summary(
    rows: list[dict[str, object]],
    variants: Sequence[str],
    delta_rows: list[dict[str, object]],
) -> dict[str, object]:
    mean_reward_by_variant: dict[str, float] = {}
    for variant_id in variants:
        values = [
            _float_value(row, "total_contract_reward")
            for row in rows
            if str(row.get("variant_id", "")) == str(variant_id)
        ]
        if values:
            mean_reward_by_variant[str(variant_id)] = _round_float(sum(values) / len(values))
    return {
        "mean_reward_by_variant": mean_reward_by_variant,
        "focal_deltas": _delta_summaries(delta_rows),
    }


def _delta_summaries(
    delta_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in delta_rows:
        key = (str(row["variant_id"]), str(row["comparator_variant_id"]))
        grouped.setdefault(key, []).append(row)

    summaries: dict[str, dict[str, object]] = {}
    for (variant_id, comparator), rows in sorted(grouped.items()):
        deltas = [_float_value(row, "variant_minus_comparator_reward") for row in rows]
        positive_count = sum(1 for row in rows if bool(row.get("variant_improves_comparator")))
        total_count = len(rows)
        summaries[f"{variant_id}_minus_{comparator}"] = {
            "mean_reward_delta": _mean_or_none(deltas),
            "std_reward_delta": _std_or_none(deltas),
            "positive_tile_seed_count": positive_count,
            "total_tile_seed_count": total_count,
            "positive_fraction": _round_float(positive_count / total_count)
            if total_count
            else None,
        }
    return summaries


def _phase30_status(
    learned_policy: Mapping[str, object],
    coverage_issues: Mapping[str, object],
) -> str:
    if _has_coverage_issues(coverage_issues):
        return "insufficient"
    focal_deltas = learned_policy.get("focal_deltas")
    if not isinstance(focal_deltas, Mapping):
        return "insufficient"
    normalized_over_raw = [
        str(key)
        for key, value in focal_deltas.items()
        if str(key).endswith("_minus_B1")
        and isinstance(value, Mapping)
        and _mean_delta(value) > 1e-9
    ]
    if not normalized_over_raw:
        return "normalization_not_supported"
    recovers_b0 = any(
        isinstance(focal_deltas.get(f"{variant}_minus_B0"), Mapping)
        and _mean_delta(focal_deltas[f"{variant}_minus_B0"]) > 1e-9
        for variant in PHASE30_FOCAL_VARIANTS
    )
    matches_or_exceeds_compressed = any(
        all(
            isinstance(focal_deltas.get(f"{variant}_minus_{comparator}"), Mapping)
            and _mean_delta(focal_deltas[f"{variant}_minus_{comparator}"]) >= -1e-9
            for comparator in ("D4P8", "D4P16")
            if f"{variant}_minus_{comparator}" in focal_deltas
        )
        and any(f"{variant}_minus_{comparator}" in focal_deltas for comparator in ("D4P8", "D4P16"))
        for variant in PHASE30_FOCAL_VARIANTS
    )
    if recovers_b0 and matches_or_exceeds_compressed:
        return "normalized_b1_matches_or_exceeds_compressed_controls"
    if recovers_b0:
        return "normalized_b1_recovers_b0_gap"
    return "normalized_b1_improves_raw_only"


def _phase30_interpretation(status: str) -> str:
    if status == "normalized_b1_matches_or_exceeds_compressed_controls":
        return (
            "At least one normalized B1 variant improves raw B1 and matches or "
            "exceeds the compressed controls under the current bounded Bishan "
            "protocol. This supports a narrow representation-scaling hypothesis, "
            "not a general claim that normalization or GeoFM improves planning."
        )
    if status == "normalized_b1_recovers_b0_gap":
        return (
            "At least one normalized B1 variant improves raw B1 and exceeds B0, "
            "but compressed-control comparisons remain unresolved."
        )
    if status == "normalized_b1_improves_raw_only":
        return (
            "At least one normalized B1 variant improves raw B1, but the result "
            "does not recover the B0 gap under this bounded protocol."
        )
    if status == "normalization_not_supported":
        return "The normalized B1 variants do not improve raw B1 in this result set."
    return "Required Phase 30 variant coverage is incomplete."


def _mean_delta(summary: Mapping[str, object]) -> float:
    value = summary.get("mean_reward_delta")
    return float(value) if value is not None else 0.0


def _coverage_issues(
    rows: list[dict[str, object]],
    variants: Sequence[str],
    eval_tile_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    expected_keys = {
        (str(eval_tile_id), int(seed), str(variant_id))
        for eval_tile_id in eval_tile_ids
        for seed in seeds
        for variant_id in variants
    }
    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    unexpected_keys: set[tuple[str, int, str]] = set()
    expected_variants = {str(item) for item in variants}
    expected_tiles = {str(item) for item in eval_tile_ids}
    expected_seeds = {int(item) for item in seeds}
    for row in rows:
        key = (
            str(row.get("eval_tile_id", "")),
            _int_value(row, "seed"),
            str(row.get("variant_id", "")),
        )
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        if (
            key[2] not in expected_variants
            or key[0] not in expected_tiles
            or key[1] not in expected_seeds
        ):
            unexpected_keys.add(key)
    return {
        "missing_variant_rows": _variant_key_dicts(expected_keys - observed_keys),
        "unexpected_variant_rows": _variant_key_dicts(unexpected_keys),
        "duplicate_variant_rows": _variant_key_dicts(duplicate_keys),
    }


def _has_coverage_issues(coverage_issues: Mapping[str, object]) -> bool:
    return any(
        bool(coverage_issues.get(key))
        for key in (
            "missing_variant_rows",
            "unexpected_variant_rows",
            "duplicate_variant_rows",
        )
    )


def _main_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("row_type", "")),
            str(row.get("variant_id", "")),
            str(row.get("eval_tile_id", "")),
        )
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for row_type, variant_id, eval_tile_id in order:
        group_rows = groups[(row_type, variant_id, eval_tile_id)]
        rewards = [_float_value(row, "total_contract_reward") for row in group_rows]
        seeds = {_int_value(row, "seed") for row in group_rows}
        summary_rows.append(
            {
                "row_type": row_type,
                "variant_id": variant_id,
                "eval_tile_id": eval_tile_id,
                "seed_count": len(seeds),
                "mean_total_contract_reward": _mean_or_none(rewards),
                "std_total_contract_reward": _std_or_none(rewards),
                "claim_boundary": PHASE30_CLAIM_BOUNDARY,
            }
        )
    return summary_rows


def _write_summary_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("Phase 30 protocol is missing summary rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Phase 30 summary rows must be objects")
            output_row = {field: row.get(field, "") for field in SUMMARY_FIELDNAMES}
            selected = output_row.get("selected_block_ids")
            if isinstance(selected, list):
                output_row["selected_block_ids"] = ";".join(str(item) for item in selected)
            writer.writerow(output_row)


def _write_csv_mapping_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: object,
    row_key: str,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"Phase 30 analysis is missing {row_key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Phase 30 {row_key} rows must be objects")
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _phase30_readiness_markdown(analysis: Mapping[str, object]) -> str:
    learned = analysis.get("learned_policy")
    if not isinstance(learned, Mapping):
        learned = {}
    mean_rewards = learned.get("mean_reward_by_variant")
    if not isinstance(mean_rewards, Mapping):
        mean_rewards = {}
    focal_deltas = learned.get("focal_deltas")
    if not isinstance(focal_deltas, Mapping):
        focal_deltas = {}
    lines = [
        "# Phase 30 Normalized-B1 Ablation",
        "",
        f"Status: {analysis.get('phase30_normalized_b1_status', '')}",
        "",
        "Scope: bounded representation-only ablation under the existing Bishan base-reward held-out protocol.",
        "",
        "Mean learned-policy reward by variant:",
    ]
    for variant_id in sorted(mean_rewards):
        lines.append(f"- {variant_id}: {mean_rewards[variant_id]}")
    lines.extend(["", "Focal normalized-B1 deltas:"])
    for key in sorted(focal_deltas):
        value = focal_deltas[key]
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "- "
            f"{key}: {value.get('mean_reward_delta')} "
            f"({value.get('positive_tile_seed_count')} / "
            f"{value.get('total_tile_seed_count')} positive)"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            str(analysis.get("interpretation", "")),
            "",
            "Claim boundary:",
            str(analysis.get("claim_boundary", PHASE30_CLAIM_BOUNDARY)),
            "",
        ]
    )
    return "\n".join(lines)


def _metadata_string_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[str],
) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return fallback


def _metadata_int_list(
    metadata: Mapping[str, object],
    key: str,
    fallback: list[int],
) -> list[int]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [int(item) for item in value if str(item).strip()]
    return fallback


def _metadata_int(
    metadata: Mapping[str, object],
    keys: tuple[str, ...],
    rows: list[dict[str, object]],
    row_field: str,
) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    for row in rows:
        value = row.get(row_field)
        if value is not None and str(value).strip() != "":
            return int(value)
    return None


def _unique_strings(rows: list[dict[str, object]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _unique_ints(rows: list[dict[str, object]], field: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        number = int(value)
        if number not in seen:
            values.append(number)
            seen.add(number)
    return values


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _int_value(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing integer field: {field}")
    return int(value)


def _optional_int(
    row: Mapping[str, object],
    field: str,
    fallback: int | None = None,
) -> int | None:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return fallback
    return int(value)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(sum(float(value) for value in values) / len(values))


def _std_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _round_float(statistics.pstdev(float(value) for value in values)) if len(values) > 1 else 0.0


def _variant_key_dicts(keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    return [
        {"eval_tile_id": eval_tile_id, "seed": seed, "variant_id": variant_id}
        for eval_tile_id, seed, variant_id in sorted(
            keys,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _round_float(value: float) -> float:
    return round(float(value), 10)
