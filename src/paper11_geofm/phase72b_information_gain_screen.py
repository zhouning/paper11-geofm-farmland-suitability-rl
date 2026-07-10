from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .phase72a_label_sources import (
    audit_phase72a_region_assets,
    load_phase72a_region_contract,
)
from .phase72b_explicit_features import build_phase72b_explicit_features
from .phase72b_geofm_features import build_phase72b_geofm_features
from .phase72b_metrics import (
    build_phase72b_gate,
    paired_block_bootstrap,
    phase72b_metrics,
)
from .phase72b_models import (
    _variant_matrix,
    load_phase72b_model_bundle,
    predict_phase72b_bundle,
)
from .phase72b_protocol import (
    PHASE72B_CLAIM_BOUNDARY,
    canonical_json_sha256,
    load_hashed_json,
    load_phase72b_protocol,
    write_hashed_json,
)
from .phase72b_splits import (
    audit_phase72b_splits,
    build_phase72b_split_registry,
)
from .phase72b_terrain import audit_phase72b_terrain_assets, _file_sha256


_PRIMARY_VARIANT = "explicit_plus_geofm_temporal_full"
_EXPLICIT_VARIANT = "explicit_history"
_CONTROL_VARIANTS = {
    "explicit_plus_temporal_order_shuffle": "temporal_order_shuffle",
    "explicit_plus_spatial_shuffle": "spatial_shuffle",
    "explicit_plus_random_projection": "random_projection",
}
_MANDATORY_AXES = (
    "pooled_temporal",
    "bishan_to_dongxing",
    "dongxing_to_bishan",
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _target_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_json_sha256(
        {
            str(name): _array_sha256(np.asarray(value))
            for name, value in sorted(arrays.items())
        }
    )


def _load_phase72a_inputs(
    package_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    package_path = package_dir / "phase72a_temporal_label_package.json"
    sample_path = package_dir / "phase72a_temporal_sample_index.csv"
    tensor_path = package_dir / "phase72a_temporal_samples.npz"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("phase72a_status") != "phase72a_label_inputs_ready":
        raise ValueError("Phase 72A package is not ready")
    rows = pd.read_csv(sample_path, keep_default_na=False).to_dict(
        orient="records"
    )
    tensors = {}
    with np.load(tensor_path) as loaded:
        for name in loaded.files:
            tensors[name] = loaded[name]
    return package, rows, tensors


def _geofm_registry(embedding_dim: int) -> dict[str, list[str]]:
    current = [f"geofm_current_{index:03d}" for index in range(embedding_dim)]
    mean = [f"geofm_temporal_mean_{index:03d}" for index in range(embedding_dim)]
    full = []
    for prefix in ("current", "mean", "std", "delta", "trend"):
        full.extend(
            [f"geofm_{prefix}_{index:03d}" for index in range(embedding_dim)]
        )
    return {
        "geofm_current": current,
        "geofm_temporal_mean": mean,
        "geofm_temporal_full": full,
    }


def prepare_phase72b_information_gain_screen(
    *,
    protocol_path: Path | str,
    phase72a_region_config: Path | str,
    phase72a_package_dir: Path | str,
    embedding_dirs: Mapping[str, Path | str],
    label_dirs: Mapping[str, Path | str],
    terrain_dir: Path | str,
) -> dict[str, object]:
    protocol = load_phase72b_protocol(protocol_path)
    contract = load_phase72a_region_contract(phase72a_region_config)
    phase72a_package, sample_rows, tensors = _load_phase72a_inputs(
        Path(phase72a_package_dir)
    )
    if [int(row["sample_index"]) for row in sample_rows] != list(
        range(len(sample_rows))
    ):
        raise ValueError("Phase 72B sample indexes must be contiguous")

    manifest_rows = []
    asset_errors = []
    region_specs = {region.region_id: region for region in contract.regions}
    for region in contract.regions:
        if region.region_id not in embedding_dirs or region.region_id not in label_dirs:
            asset_errors.append(f"missing Phase 72B path mapping: {region.region_id}")
            continue
        audit = audit_phase72a_region_assets(
            contract,
            region,
            embedding_dir=embedding_dirs[region.region_id],
            label_dir=label_dirs[region.region_id],
        )
        manifest_rows.extend(audit["file_manifest_rows"])
        asset_errors.extend(audit["errors"])
    terrain_audit = audit_phase72b_terrain_assets(
        protocol, contract, terrain_dir
    )
    asset_errors.extend(terrain_audit["errors"])
    if asset_errors:
        raise ValueError("Phase 72B inputs not ready: " + " | ".join(asset_errors))

    labels = {}
    terrain = {}
    for region in contract.regions:
        labels[region.region_id] = {
            year: np.load(
                Path(label_dirs[region.region_id])
                / region.label_pattern.format(year=year)
            )
            for year in region.years
        }
        terrain_path = Path(terrain_dir) / f"{region.region_id}_terrain.npz"
        with np.load(terrain_path) as loaded:
            terrain[region.region_id] = {
                name: loaded[name] for name in protocol.terrain_features
            }

    explicit = build_phase72b_explicit_features(
        sample_rows,
        regions=region_specs,
        labels=labels,
        terrain=terrain,
        crop_class_code=contract.crop_class_code,
    )
    geofm = build_phase72b_geofm_features(
        tensors["embedding_history"], tensors["history_mask"]
    )
    matrices = {
        "explicit_static": explicit["explicit_static"],
        "explicit_history": explicit["explicit_history"],
        **geofm,
        "embedding_history": tensors["embedding_history"].astype(np.float32),
        "history_mask": tensors["history_mask"].astype(bool),
    }
    registry = {
        **explicit["registry"],
        **_geofm_registry(contract.regions[0].embedding_dim),
    }
    split_rows = []
    for row in sample_rows:
        adjusted = dict(row)
        adjusted["conversion_1y"] = 1 - int(row["y_1y"])
        split_rows.append(adjusted)
    split_registry = build_phase72b_split_registry(
        split_rows,
        train_years=protocol.train_years,
        validation_year=protocol.validation_years[0],
        test_year=protocol.test_years[0],
        folds=protocol.spatial_folds,
        buffer_rings=protocol.buffer_rings,
    )
    leakage_audit = audit_phase72b_splits(
        split_rows,
        split_registry,
        train_years=protocol.train_years,
        validation_year=protocol.validation_years[0],
        test_year=protocol.test_years[0],
    )
    if leakage_audit["status"] != "leakage_audit_passed":
        raise ValueError(
            "Phase 72B leakage audit failed: "
            + " | ".join(leakage_audit["errors"])
        )

    origins = np.asarray(
        [int(row["origin_year"]) for row in split_rows], dtype=np.int16
    )
    sample_indexes = np.asarray(
        [int(row["sample_index"]) for row in split_rows], dtype=np.int32
    )
    conversion = np.asarray(
        [int(row["conversion_1y"]) for row in split_rows], dtype=np.int8
    )
    development_mask = origins <= protocol.validation_years[0]
    confirmation_mask = origins == protocol.test_years[0]
    development_targets = {
        "sample_index": sample_indexes[development_mask],
        "origin_year": origins[development_mask],
        "conversion_1y": conversion[development_mask],
    }
    confirmation_targets = {
        "sample_index": sample_indexes[confirmation_mask],
        "origin_year": origins[confirmation_mask],
        "conversion_1y": conversion[confirmation_mask],
    }

    feature_manifest_rows = [
        {
            "matrix_id": name,
            "shape": "x".join(map(str, value.shape)),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
        }
        for name, value in matrices.items()
    ]
    feature_rows = [
        {
            key: row[key]
            for key in (
                "sample_index",
                "region_id",
                "unit_id",
                "row",
                "col",
                "spatial_block_id",
                "origin_year",
            )
        }
        for row in split_rows
    ]
    frozen_protocol = {
        "status": "phase72b_protocol_frozen",
        "tracked_protocol": protocol.raw,
        "tracked_protocol_sha256": canonical_json_sha256(protocol.raw),
        "phase72a_package_sha256": _file_sha256(
            Path(phase72a_package_dir)
            / "phase72a_temporal_label_package.json"
        ),
        "terrain_manifest": terrain_audit["rows"],
        "annual_asset_manifest_rows": manifest_rows,
        "feature_manifest_rows": feature_manifest_rows,
        "feature_registry_sha256": canonical_json_sha256(registry),
        "split_registry_sha256": canonical_json_sha256(split_registry),
        "leakage_status": leakage_audit["status"],
        "development_target_rows": int(development_mask.sum()),
        "confirmation_target_rows": int(confirmation_mask.sum()),
        "development_targets_sha256": _target_arrays_sha256(
            development_targets
        ),
        "confirmation_targets_sha256": _target_arrays_sha256(
            confirmation_targets
        ),
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    return {
        "protocol": protocol,
        "phase72a_package": phase72a_package,
        "terrain_manifest_rows": terrain_audit["rows"],
        "feature_manifest_rows": feature_manifest_rows,
        "feature_registry": registry,
        "feature_rows": feature_rows,
        "matrices": matrices,
        "development_targets": development_targets,
        "confirmation_targets": confirmation_targets,
        "split_registry": split_registry,
        "leakage_audit": leakage_audit,
        "frozen_protocol": frozen_protocol,
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_phase72b_prepared_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "terrain_manifest_csv": output / "phase72b_terrain_manifest.csv",
        "feature_manifest_csv": output / "phase72b_feature_manifest.csv",
        "feature_registry_json": output / "phase72b_feature_registry.json",
        "feature_rows_csv": output / "phase72b_feature_rows.csv",
        "feature_matrices_npz": output / "phase72b_feature_matrices.npz",
        "development_targets_npz": output
        / "phase72b_development_targets.npz",
        "confirmation_targets_npz": output
        / "phase72b_confirmation_targets.npz",
        "split_registry_json": output / "phase72b_split_registry.json",
        "row_alignment_csv": output / "phase72b_row_alignment_audit.csv",
        "leakage_audit_json": output / "phase72b_leakage_audit.json",
        "protocol_json": output / "phase72b_frozen_protocol.json",
        "protocol_hash": output / "phase72b_frozen_protocol.sha256",
    }
    _write_csv(
        paths["terrain_manifest_csv"],
        list(package["terrain_manifest_rows"]),
        ["region_id", "path", "shape", "sha256"],
    )
    _write_csv(
        paths["feature_manifest_csv"],
        list(package["feature_manifest_rows"]),
        ["matrix_id", "shape", "dtype", "sha256"],
    )
    paths["feature_registry_json"].write_text(
        json.dumps(package["feature_registry"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feature_rows = list(package["feature_rows"])
    _write_csv(
        paths["feature_rows_csv"],
        feature_rows,
        [
            "sample_index",
            "region_id",
            "unit_id",
            "row",
            "col",
            "spatial_block_id",
            "origin_year",
        ],
    )
    np.savez_compressed(paths["feature_matrices_npz"], **package["matrices"])
    np.savez_compressed(
        paths["development_targets_npz"], **package["development_targets"]
    )
    np.savez_compressed(
        paths["confirmation_targets_npz"], **package["confirmation_targets"]
    )
    paths["split_registry_json"].write_text(
        json.dumps(package["split_registry"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        paths["row_alignment_csv"],
        [
            {
                "sample_rows": len(feature_rows),
                "first_sample_index": 0 if feature_rows else "",
                "last_sample_index": len(feature_rows) - 1 if feature_rows else "",
                "status": "row_alignment_passed",
            }
        ],
        [
            "sample_rows",
            "first_sample_index",
            "last_sample_index",
            "status",
        ],
    )
    paths["leakage_audit_json"].write_text(
        json.dumps(package["leakage_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    protocol_json, protocol_hash = write_hashed_json(
        paths["protocol_json"], package["frozen_protocol"]
    )
    paths["protocol_json"] = protocol_json
    paths["protocol_hash"] = protocol_hash
    return paths


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _control_seed(value: object) -> int | None:
    return None if value in (None, "") else int(value)


def _bundle_key(record: Mapping[str, object]) -> tuple[str, str, int | None]:
    return (
        str(record["axis_id"]),
        str(record["variant_id"]),
        _control_seed(record.get("control_seed")),
    )


def _metric_delta(
    explicit: Mapping[str, object], geofm: Mapping[str, object]
) -> dict[str, float]:
    return {
        "ap_delta": round(
            float(geofm["average_precision"])
            - float(explicit["average_precision"]),
            12,
        ),
        "brier_delta": round(
            float(explicit["brier"]) - float(geofm["brier"]), 12
        ),
        "ece_delta": round(
            float(explicit["ece"]) - float(geofm["ece"]), 12
        ),
    }


def _calibration_rows(
    *,
    axis_id: str,
    variant_id: str,
    control_seed: int | None,
    outcome: np.ndarray,
    probability: np.ndarray,
    bins: int,
) -> list[dict[str, object]]:
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, min(int(bins), len(order)))
    rows = []
    for bin_index, indexes in enumerate(groups, start=1):
        if not len(indexes):
            continue
        observed = float(outcome[indexes].mean())
        predicted = float(probability[indexes].mean())
        rows.append(
            {
                "axis_id": axis_id,
                "variant_id": variant_id,
                "control_seed": "" if control_seed is None else control_seed,
                "bin_id": bin_index,
                "rows": len(indexes),
                "probability_min": round(float(probability[indexes].min()), 12),
                "probability_max": round(float(probability[indexes].max()), 12),
                "probability_mean": round(predicted, 12),
                "observed_rate": round(observed, 12),
                "absolute_gap": round(abs(observed - predicted), 12),
            }
        )
    return rows


def _blocked_confirmation_result(
    *,
    protocol_hash: str,
    selected_hash: str,
    bundle_hashes: list[dict[str, object]],
    blockers: list[str],
    invalid_spatial_axes: list[str],
) -> dict[str, object]:
    return {
        "phase72b_status": "phase72b_inputs_not_ready",
        "reasons": ["mandatory Phase 72B confirmation audit failed"],
        "checks": {},
        "blockers": sorted(set(blockers)),
        "invalid_spatial_axes": sorted(set(invalid_spatial_axes)),
        "frozen_protocol_sha256": protocol_hash,
        "selected_models_sha256": selected_hash,
        "bundle_hashes": bundle_hashes,
        "counts": {
            "confirmation_rows": 0,
            "prediction_rows": 0,
            "metric_rows": 0,
        },
        "metrics_rows": [],
        "prediction_rows": [],
        "calibration_rows": [],
        "bootstrap_rows": [],
        "control_rows": [],
        "transfer_rows": [],
        "spatial_rows": [],
        "pooled_delta": {},
        "pooled_bootstrap": {},
        "next_action": "Remain in Phase 72B and resolve the recorded input or audit blocker.",
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }


def confirm_phase72b_information_gain_screen(
    *, prepared_dir: Path | str, frozen_dir: Path | str
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    frozen = Path(frozen_dir)

    frozen_protocol = load_hashed_json(
        prepared / "phase72b_frozen_protocol.json",
        prepared / "phase72b_frozen_protocol.sha256",
    )
    protocol_hash = (
        prepared / "phase72b_frozen_protocol.sha256"
    ).read_text(encoding="ascii").strip().lower()
    selected = load_hashed_json(
        frozen / "phase72b_selected_models.json",
        frozen / "phase72b_selected_models.sha256",
    )
    selected_hash = (
        frozen / "phase72b_selected_models.sha256"
    ).read_text(encoding="ascii").strip().lower()
    if str(selected.get("frozen_protocol_sha256", "")).lower() != protocol_hash:
        raise ValueError(
            "Phase 72B frozen protocol hash mismatch between prepared and selected models"
        )

    protocol = dict(frozen_protocol["tracked_protocol"])
    split_registry = json.loads(
        (prepared / "phase72b_split_registry.json").read_text(encoding="utf-8")
    )
    expected_split_hash = str(frozen_protocol["split_registry_sha256"])
    if canonical_json_sha256(split_registry) != expected_split_hash:
        raise ValueError("Phase 72B split registry hash mismatch")
    feature_registry = json.loads(
        (prepared / "phase72b_feature_registry.json").read_text(encoding="utf-8")
    )
    if canonical_json_sha256(feature_registry) != str(
        frozen_protocol["feature_registry_sha256"]
    ):
        raise ValueError("Phase 72B feature registry hash mismatch")
    feature_rows = pd.read_csv(
        prepared / "phase72b_feature_rows.csv", keep_default_na=False
    ).to_dict(orient="records")
    if [int(row["sample_index"]) for row in feature_rows] != list(
        range(len(feature_rows))
    ):
        raise ValueError("Phase 72B feature row alignment failed")
    matrices = _load_npz(prepared / "phase72b_feature_matrices.npz")
    expected_matrices = {
        str(row["matrix_id"]): str(row["sha256"])
        for row in frozen_protocol["feature_manifest_rows"]
    }
    if set(matrices) != set(expected_matrices):
        raise ValueError("Phase 72B feature matrix manifest mismatch")
    for matrix_id, expected_hash in expected_matrices.items():
        if _array_sha256(matrices[matrix_id]) != expected_hash:
            raise ValueError(
                f"Phase 72B feature matrix hash mismatch: {matrix_id}"
            )

    leakage_audit = json.loads(
        (prepared / "phase72b_leakage_audit.json").read_text(encoding="utf-8")
    )
    invalid_spatial_axes = list(leakage_audit.get("invalid_spatial_axes", []))
    blockers = []
    if leakage_audit.get("status") != "leakage_audit_passed":
        blockers.append("leakage audit failed")

    records = [dict(record) for record in selected.get("bundle_records", [])]
    record_by_key: dict[tuple[str, str, int | None], dict[str, object]] = {}
    bundles: dict[tuple[str, str, int | None], dict[str, object]] = {}
    bundle_hashes = []
    for record in records:
        key = _bundle_key(record)
        if key in record_by_key:
            raise ValueError(f"Duplicate Phase 72B model bundle record: {key}")
        relative = Path(str(record["bundle_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid Phase 72B model bundle path")
        bundle_path = frozen / relative
        bundle = load_phase72b_model_bundle(
            bundle_path, str(record["bundle_sha256"])
        )
        if (
            str(bundle.get("axis_id")) != key[0]
            or str(bundle.get("variant_id")) != key[1]
        ):
            raise ValueError(f"Phase 72B model bundle identity mismatch: {key}")
        record_by_key[key] = record
        bundles[key] = bundle
        bundle_hashes.append(
            {
                "axis_id": key[0],
                "variant_id": key[1],
                "control_seed": "" if key[2] is None else key[2],
                "bundle_path": str(relative),
                "bundle_sha256": str(record["bundle_sha256"]),
            }
        )

    selected_axes = dict(selected.get("axes", {}))
    selected_control_seeds = dict(selected.get("selected_control_seeds", {}))
    control_seeds = [int(value) for value in protocol["controls"]["seeds"]]
    for axis_id in _MANDATORY_AXES:
        if axis_id not in split_registry or axis_id not in selected_axes:
            blockers.append(f"missing mandatory axis: {axis_id}")
            continue
        for variant_id in (_EXPLICIT_VARIANT, _PRIMARY_VARIANT):
            if (axis_id, variant_id, None) not in bundles:
                blockers.append(f"missing mandatory bundle: {axis_id}/{variant_id}")
        axis_seeds = dict(selected_control_seeds.get(axis_id, {}))
        for variant_id in _CONTROL_VARIANTS:
            if variant_id not in axis_seeds:
                blockers.append(
                    f"missing selected control seed: {axis_id}/{variant_id}"
                )
            for seed in control_seeds:
                if (axis_id, variant_id, seed) not in bundles:
                    blockers.append(
                        f"missing control bundle: {axis_id}/{variant_id}/seed{seed}"
                    )

    expected_spatial_axes = sorted(
        axis_id
        for axis_id in split_registry
        if axis_id.startswith("spatial_") and axis_id not in invalid_spatial_axes
    )
    pooled_control_seeds = dict(
        selected_control_seeds.get("pooled_temporal", {})
    )
    for axis_id in expected_spatial_axes:
        if axis_id not in selected_axes:
            blockers.append(f"missing valid spatial axis: {axis_id}")
            continue
        for variant_id in (_EXPLICIT_VARIANT, _PRIMARY_VARIANT):
            if (axis_id, variant_id, None) not in bundles:
                blockers.append(f"missing spatial bundle: {axis_id}/{variant_id}")
        for variant_id in _CONTROL_VARIANTS:
            if variant_id not in pooled_control_seeds:
                blockers.append(f"missing pooled control seed: {variant_id}")
                continue
            seed = int(pooled_control_seeds[variant_id])
            if (axis_id, variant_id, seed) not in bundles:
                blockers.append(
                    f"missing spatial control bundle: {axis_id}/{variant_id}/seed{seed}"
                )

    if blockers:
        return _blocked_confirmation_result(
            protocol_hash=protocol_hash,
            selected_hash=selected_hash,
            bundle_hashes=bundle_hashes,
            blockers=blockers,
            invalid_spatial_axes=invalid_spatial_axes,
        )

    # The confirmation labels are intentionally opened only after every frozen
    # contract and model bundle has passed its integrity checks.
    confirmation = _load_npz(
        prepared / "phase72b_confirmation_targets.npz"
    )
    if _target_arrays_sha256(confirmation) != str(
        frozen_protocol.get("confirmation_targets_sha256", "")
    ):
        raise ValueError("Phase 72B confirmation target hash mismatch")
    confirmation_outcomes = {
        int(index): int(outcome)
        for index, outcome in zip(
            confirmation["sample_index"], confirmation["conversion_1y"]
        )
    }
    confirmation_years = set(
        int(value) for value in confirmation["origin_year"].tolist()
    )
    expected_test_years = set(int(value) for value in protocol["years"]["test"])
    if confirmation_years != expected_test_years:
        blockers.append(
            f"confirmation years mismatch: {sorted(confirmation_years)}"
        )

    prediction_rows = []
    metrics_rows = []
    calibration_rows = []
    groups: dict[tuple[str, str, int | None], dict[str, object]] = {}
    ece_bins = int(protocol["calibration"]["ece_bins"])
    budgets = tuple(float(value) for value in protocol["budgets"])
    for key, bundle in bundles.items():
        axis_id, variant_id, seed = key
        if axis_id not in split_registry:
            blockers.append(f"bundle references unknown axis: {axis_id}")
            continue
        indexes = np.asarray(
            [int(value) for value in split_registry[axis_id]["test"]],
            dtype=np.int64,
        )
        if not len(indexes):
            if axis_id.startswith("spatial_"):
                invalid_spatial_axes.append(axis_id)
                continue
            blockers.append(f"mandatory axis has no confirmation rows: {axis_id}")
            continue
        missing = [int(value) for value in indexes if int(value) not in confirmation_outcomes]
        if missing:
            blockers.append(
                f"confirmation outcomes missing indexes for {axis_id}: {missing[:5]}"
            )
            continue
        outcome = np.asarray(
            [confirmation_outcomes[int(value)] for value in indexes],
            dtype=np.int8,
        )
        matrix = _variant_matrix(
            variant_id, matrices, feature_rows, seed=seed
        )
        if int(bundle["feature_count"]) != int(matrix.shape[1]):
            raise ValueError(f"Phase 72B bundle feature count mismatch: {key}")
        probability = predict_phase72b_bundle(bundle, matrix[indexes])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            metric = phase72b_metrics(
                outcome,
                probability,
                threshold=float(bundle["f1_threshold"]),
                budgets=budgets,
                ece_bins=ece_bins,
                budget_thresholds=dict(bundle["budget_thresholds"]),
            )
        metric_row = {
            "axis_id": axis_id,
            "variant_id": variant_id,
            "control_seed": "" if seed is None else seed,
            "rows": len(indexes),
            "positives": int(outcome.sum()),
            "prevalence": round(float(outcome.mean()), 12),
            "model_family": str(bundle["model_family"]),
            "candidate_id": str(bundle["candidate_id"]),
            "calibration_method": str(bundle["calibration_method"]),
            "f1_threshold": round(float(bundle["f1_threshold"]), 12),
            **metric,
        }
        metrics_rows.append(metric_row)
        groups[key] = {
            "indexes": indexes,
            "outcome": outcome,
            "probability": probability,
            "metric": metric_row,
        }
        calibration_rows.extend(
            _calibration_rows(
                axis_id=axis_id,
                variant_id=variant_id,
                control_seed=seed,
                outcome=outcome,
                probability=probability,
                bins=ece_bins,
            )
        )
        for position, sample_index in enumerate(indexes):
            source_row = feature_rows[int(sample_index)]
            prediction_rows.append(
                {
                    "sample_index": int(sample_index),
                    "axis_id": axis_id,
                    "variant_id": variant_id,
                    "control_seed": "" if seed is None else seed,
                    "outcome": int(outcome[position]),
                    "probability": round(float(probability[position]), 12),
                    "threshold": round(float(bundle["f1_threshold"]), 12),
                    "predicted_class": int(
                        probability[position] >= float(bundle["f1_threshold"])
                    ),
                    "spatial_block_id": str(source_row["spatial_block_id"]),
                    "region_id": str(source_row["region_id"]),
                    "origin_year": int(source_row["origin_year"]),
                    "row": int(source_row["row"]),
                    "col": int(source_row["col"]),
                }
            )

    for axis_id in _MANDATORY_AXES:
        primary_group = groups.get((axis_id, _PRIMARY_VARIANT, None))
        explicit_group = groups.get((axis_id, _EXPLICIT_VARIANT, None))
        if primary_group is None or explicit_group is None:
            blockers.append(f"missing mandatory confirmation comparison: {axis_id}")
        elif len(np.unique(primary_group["outcome"])) != 2:
            blockers.append(f"missing confirmation class support: {axis_id}")

    if blockers:
        blocked = _blocked_confirmation_result(
            protocol_hash=protocol_hash,
            selected_hash=selected_hash,
            bundle_hashes=bundle_hashes,
            blockers=blockers,
            invalid_spatial_axes=invalid_spatial_axes,
        )
        blocked.update(
            {
                "metrics_rows": metrics_rows,
                "prediction_rows": prediction_rows,
                "calibration_rows": calibration_rows,
                "counts": {
                    "confirmation_rows": len(confirmation_outcomes),
                    "prediction_rows": len(prediction_rows),
                    "metric_rows": len(metrics_rows),
                },
            }
        )
        return blocked

    def group(axis_id: str, variant_id: str, seed: int | None = None):
        return groups[(axis_id, variant_id, seed)]

    pooled_explicit = group("pooled_temporal", _EXPLICIT_VARIANT)
    pooled_primary = group("pooled_temporal", _PRIMARY_VARIANT)
    pooled_delta = _metric_delta(
        pooled_explicit["metric"], pooled_primary["metric"]
    )

    bootstrap_rows = []

    def add_bootstrap(
        comparison_id: str,
        axis_id: str,
        baseline_group: Mapping[str, object],
        candidate_group: Mapping[str, object],
    ) -> dict[str, object] | None:
        indexes = np.asarray(candidate_group["indexes"], dtype=np.int64)
        rows = [feature_rows[int(index)] for index in indexes]
        try:
            result = paired_block_bootstrap(
                candidate_group["outcome"],
                baseline_group["probability"],
                candidate_group["probability"],
                rows,
                iterations=int(protocol["bootstrap"]["iterations"]),
                seed=int(protocol["bootstrap"]["seed"]),
            )
        except ValueError as exc:
            bootstrap_rows.append(
                {
                    "comparison_id": comparison_id,
                    "axis_id": axis_id,
                    "status": "invalid",
                    "error": str(exc),
                }
            )
            return None
        row = {
            "comparison_id": comparison_id,
            "axis_id": axis_id,
            "status": "valid",
            **result,
        }
        bootstrap_rows.append(row)
        return row

    pooled_bootstrap = add_bootstrap(
        "primary_vs_explicit",
        "pooled_temporal",
        pooled_explicit,
        pooled_primary,
    )
    if pooled_bootstrap is None:
        blockers.append("pooled paired block bootstrap was invalid")

    control_rows = []
    for variant_id, control_id in _CONTROL_VARIANTS.items():
        seed_deltas = []
        for seed in control_seeds:
            control_group = group("pooled_temporal", variant_id, seed)
            seed_deltas.append(
                {"seed": seed, **_metric_delta(control_group["metric"], pooled_primary["metric"])}
            )
        selected_seed = int(
            selected_control_seeds["pooled_temporal"][variant_id]
        )
        selected_delta = next(
            row for row in seed_deltas if int(row["seed"]) == selected_seed
        )
        control_row = {
            "control_id": control_id,
            "variant_id": variant_id,
            "selected_seed": selected_seed,
            "seed_count": len(seed_deltas),
            "ap_delta": selected_delta["ap_delta"],
            "brier_delta": selected_delta["brier_delta"],
            "ece_delta": selected_delta["ece_delta"],
            "ap_delta_seed_min": min(row["ap_delta"] for row in seed_deltas),
            "ap_delta_seed_max": max(row["ap_delta"] for row in seed_deltas),
            "brier_delta_seed_min": min(
                row["brier_delta"] for row in seed_deltas
            ),
            "brier_delta_seed_max": max(
                row["brier_delta"] for row in seed_deltas
            ),
            "ece_delta_seed_min": min(row["ece_delta"] for row in seed_deltas),
            "ece_delta_seed_max": max(row["ece_delta"] for row in seed_deltas),
        }
        control_rows.append(control_row)
        add_bootstrap(
            f"primary_vs_{control_id}",
            "pooled_temporal",
            group("pooled_temporal", variant_id, selected_seed),
            pooled_primary,
        )

    transfer_rows = []
    for axis_id in ("bishan_to_dongxing", "dongxing_to_bishan"):
        explicit_group = group(axis_id, _EXPLICIT_VARIANT)
        primary_group = group(axis_id, _PRIMARY_VARIANT)
        transfer_rows.append(
            {
                "axis_id": axis_id,
                "source_region": axis_id.split("_to_")[0],
                "target_region": axis_id.split("_to_")[1],
                "rows": len(primary_group["indexes"]),
                **_metric_delta(explicit_group["metric"], primary_group["metric"]),
            }
        )
        add_bootstrap(
            "primary_vs_explicit",
            axis_id,
            explicit_group,
            primary_group,
        )

    spatial_rows = []
    for axis_id in expected_spatial_axes:
        explicit_group = groups.get((axis_id, _EXPLICIT_VARIANT, None))
        primary_group = groups.get((axis_id, _PRIMARY_VARIANT, None))
        if explicit_group is None or primary_group is None:
            invalid_spatial_axes.append(axis_id)
            continue
        if len(np.unique(primary_group["outcome"])) != 2:
            invalid_spatial_axes.append(axis_id)
            continue
        region_id = axis_id.removeprefix("spatial_").split("_fold", 1)[0]
        spatial_rows.append(
            {
                "axis_id": axis_id,
                "region_id": region_id,
                "rows": len(primary_group["indexes"]),
                **_metric_delta(explicit_group["metric"], primary_group["metric"]),
            }
        )
        add_bootstrap(
            "primary_vs_explicit",
            axis_id,
            explicit_group,
            primary_group,
        )

    if blockers:
        gate = build_phase72b_gate(
            pooled_delta=pooled_delta,
            pooled_bootstrap=pooled_bootstrap or {},
            control_rows=control_rows,
            transfer_rows=transfer_rows,
            spatial_rows=spatial_rows,
            leakage_ok=False,
            gates=protocol["gates"],
        )
    else:
        gate = build_phase72b_gate(
            pooled_delta=pooled_delta,
            pooled_bootstrap=pooled_bootstrap,
            control_rows=control_rows,
            transfer_rows=transfer_rows,
            spatial_rows=spatial_rows,
            leakage_ok=True,
            gates=protocol["gates"],
        )
    status = str(gate["phase72b_status"])
    next_actions = {
        "geofm_information_supported": "Phase 72C design may begin.",
        "geofm_information_mixed": "Run only the frozen Phase 72B heterogeneity audit.",
        "geofm_information_not_supported": "Stop the GeoFM-STaR route and execute the approved exhaustion analysis.",
        "phase72b_inputs_not_ready": "Remain in Phase 72B and resolve the recorded input or audit blocker.",
    }
    return {
        **gate,
        "blockers": sorted(set(blockers)),
        "invalid_spatial_axes": sorted(set(invalid_spatial_axes)),
        "frozen_protocol_sha256": protocol_hash,
        "selected_models_sha256": selected_hash,
        "bundle_hashes": bundle_hashes,
        "counts": {
            "confirmation_rows": len(confirmation_outcomes),
            "prediction_rows": len(prediction_rows),
            "metric_rows": len(metrics_rows),
            "calibration_rows": len(calibration_rows),
            "bootstrap_rows": len(bootstrap_rows),
            "valid_spatial_axes": len(spatial_rows),
            "invalid_spatial_axes": len(set(invalid_spatial_axes)),
        },
        "metrics_rows": metrics_rows,
        "prediction_rows": prediction_rows,
        "calibration_rows": calibration_rows,
        "bootstrap_rows": bootstrap_rows,
        "control_rows": control_rows,
        "transfer_rows": transfer_rows,
        "spatial_rows": spatial_rows,
        "pooled_delta": pooled_delta,
        "pooled_bootstrap": pooled_bootstrap or {},
        "next_action": next_actions[status],
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_frame(
    path: Path, rows: list[dict[str, object]], columns: list[str]
) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        leading = [column for column in columns if column in frame.columns]
        trailing = [column for column in frame.columns if column not in leading]
        frame = frame[leading + sorted(trailing)]
    frame.to_csv(path, index=False)


def write_phase72b_confirmation_artifacts(
    result: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_csv": output / "phase72b_metrics.csv",
        "predictions_csv": output / "phase72b_predictions.csv",
        "calibration_csv": output / "phase72b_calibration.csv",
        "bootstrap_csv": output / "phase72b_bootstrap_deltas.csv",
        "control_csv": output / "phase72b_control_comparison.csv",
        "transfer_csv": output / "phase72b_transfer_summary.csv",
        "screen_json": output / "phase72b_information_gain_screen.json",
        "screen_md": output / "phase72b_information_gain_screen.md",
    }
    _write_frame(
        paths["metrics_csv"],
        list(result.get("metrics_rows", [])),
        ["axis_id", "variant_id", "control_seed", "rows", "positives"],
    )
    _write_frame(
        paths["predictions_csv"],
        list(result.get("prediction_rows", [])),
        [
            "sample_index",
            "axis_id",
            "variant_id",
            "control_seed",
            "outcome",
            "probability",
            "threshold",
            "spatial_block_id",
            "region_id",
            "origin_year",
        ],
    )
    _write_frame(
        paths["calibration_csv"],
        list(result.get("calibration_rows", [])),
        ["axis_id", "variant_id", "control_seed", "bin_id", "rows"],
    )
    _write_frame(
        paths["bootstrap_csv"],
        list(result.get("bootstrap_rows", [])),
        ["comparison_id", "axis_id", "status"],
    )
    _write_frame(
        paths["control_csv"],
        list(result.get("control_rows", [])),
        ["control_id", "variant_id", "selected_seed", "seed_count"],
    )
    _write_frame(
        paths["transfer_csv"],
        list(result.get("transfer_rows", [])),
        ["axis_id", "source_region", "target_region", "rows"],
    )
    screen_payload = dict(result)
    screen_payload.pop("prediction_rows", None)
    screen_payload.pop("calibration_rows", None)
    screen_payload["row_level_artifacts"] = {
        "predictions_csv": paths["predictions_csv"].name,
        "calibration_csv": paths["calibration_csv"].name,
    }
    safe_result = _json_safe(screen_payload)
    paths["screen_json"].write_text(
        json.dumps(safe_result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    pooled = dict(result.get("pooled_delta", {}))
    lines = [
        "# Phase 72B GeoFM Information-Gain Screen",
        "",
        f"- Status: `{result['phase72b_status']}`",
        f"- Frozen protocol SHA256: `{result.get('frozen_protocol_sha256', '')}`",
        f"- Selected models SHA256: `{result.get('selected_models_sha256', '')}`",
        f"- Confirmation rows: `{dict(result.get('counts', {})).get('confirmation_rows', 0)}`",
        f"- Pooled AP delta: `{pooled.get('ap_delta', '')}`",
        f"- Pooled Brier delta: `{pooled.get('brier_delta', '')}`",
        f"- Pooled ECE delta: `{pooled.get('ece_delta', '')}`",
        f"- Invalid spatial axes: `{', '.join(result.get('invalid_spatial_axes', []))}`",
        f"- Blockers: `{'; '.join(result.get('blockers', []))}`",
        f"- Next action: {result.get('next_action', '')}",
        "",
        str(result.get("claim_boundary", PHASE72B_CLAIM_BOUNDARY)),
        "",
    ]
    paths["screen_md"].write_text("\n".join(lines), encoding="utf-8")
    return paths
