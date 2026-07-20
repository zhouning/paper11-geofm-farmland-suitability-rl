from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .phase72b_geofm_features import build_phase72b_control_features
from .phase72b_metrics import phase72b_metrics
from .phase72b_protocol import (
    PHASE72B_CLAIM_BOUNDARY,
    canonical_json_sha256,
    load_hashed_json,
    write_hashed_json,
)
from .phase72b_prepared import load_verified_phase72b_prepared
from .phase72b_terrain import _file_sha256


_CORE_VARIANTS = (
    "explicit_static",
    "explicit_history",
    "geofm_current_only",
    "geofm_temporal_mean_only",
    "explicit_plus_geofm_current",
    "explicit_plus_geofm_temporal_full",
)

_CONTROL_VARIANTS = {
    "explicit_plus_temporal_order_shuffle": "temporal_order_shuffle",
    "explicit_plus_spatial_shuffle": "spatial_shuffle",
    "explicit_plus_random_projection": "random_projection",
}


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _target_npz_sha256(path: Path) -> str:
    with np.load(path) as loaded:
        hashes = {
            str(name): _array_sha256(loaded[name])
            for name in sorted(loaded.files)
        }
    return canonical_json_sha256(hashes)


def _candidate_configs(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    models = dict(protocol["models"])
    candidates = []
    for c_value, class_weight in itertools.product(
        models["logistic_c"], models["logistic_class_weight"]
    ):
        candidates.append(
            {
                "model_family": "logistic",
                "C": float(c_value),
                "class_weight": (
                    None if str(class_weight) == "none" else str(class_weight)
                ),
            }
        )
    for learning_rate, leaves, minimum_leaf, l2_value in itertools.product(
        models["hgb_learning_rate"],
        models["hgb_max_leaf_nodes"],
        models["hgb_min_samples_leaf"],
        models["hgb_l2_regularization"],
    ):
        candidates.append(
            {
                "model_family": "hgb",
                "learning_rate": float(learning_rate),
                "max_leaf_nodes": int(leaves),
                "min_samples_leaf": int(minimum_leaf),
                "max_iter": int(models["hgb_max_iter"]),
                "l2_regularization": float(l2_value),
            }
        )
    if not candidates:
        raise ValueError("Phase 72B protocol contains no model candidates")
    return candidates


def _candidate_id(config: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(dict(config), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _fit_estimator(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: Mapping[str, object],
    *,
    seed: int,
) -> tuple[StandardScaler | None, object]:
    with threadpool_limits(limits=1):
        return _fit_estimator_single_thread(
            train_x, train_y, config, seed=seed
        )


def _fit_estimator_single_thread(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: Mapping[str, object],
    *,
    seed: int,
) -> tuple[StandardScaler | None, object]:
    family = str(config["model_family"])
    if family == "logistic":
        scaler = StandardScaler().fit(train_x)
        estimator = LogisticRegression(
            C=float(config["C"]),
            class_weight=config.get("class_weight"),
            max_iter=2000,
            solver="lbfgs",
            random_state=int(seed),
        ).fit(scaler.transform(train_x), train_y)
        return scaler, estimator
    if family == "hgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_iter=int(config["max_iter"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=int(seed),
        ).fit(train_x, train_y)
        return None, estimator
    raise ValueError(f"Unknown Phase 72B model family: {family}")


def _raw_probability(
    scaler: StandardScaler | None, estimator: object, features: np.ndarray
) -> np.ndarray:
    matrix = scaler.transform(features) if scaler is not None else features
    probability = estimator.predict_proba(matrix)[:, 1]
    return np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)


def _fit_calibrator(
    method: str, probability: np.ndarray, outcome: np.ndarray
) -> object | None:
    if method == "none":
        return None
    if method == "sigmoid":
        logits = np.log(probability / (1 - probability))[:, None]
        return LogisticRegression(
            C=1e6, max_iter=2000, solver="lbfgs", random_state=72
        ).fit(logits, outcome)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(
            probability, outcome
        )
    raise ValueError(f"Unknown Phase 72B calibration method: {method}")


def _apply_calibrator(
    method: str, calibrator: object | None, probability: np.ndarray
) -> np.ndarray:
    raw = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    if method == "none":
        return raw
    if method == "sigmoid":
        logits = np.log(raw / (1 - raw))[:, None]
        return np.clip(calibrator.predict_proba(logits)[:, 1], 0, 1)
    if method == "isotonic":
        return np.clip(calibrator.predict(raw), 0, 1)
    raise ValueError(f"Unknown Phase 72B calibration method: {method}")


def _best_f1_threshold(outcome: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate([probability, np.asarray([0.5], dtype=np.float64)])
    )
    scored = [
        (
            float(
                f1_score(
                    outcome,
                    (probability >= threshold).astype(np.int8),
                    zero_division=0,
                )
            ),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored)[2]


def _budget_thresholds(
    probability: np.ndarray, budgets: Sequence[float]
) -> dict[str, float]:
    return {
        f"{int(round(100 * float(budget)))}pct": float(
            np.quantile(probability, 1 - float(budget), method="higher")
        )
        for budget in budgets
    }


def _fit_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    config: Mapping[str, object],
    calibration_methods: Sequence[str],
    budgets: Sequence[float],
    ece_bins: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scaler, estimator = _fit_estimator(
        train_x, train_y, config, seed=seed
    )
    raw = _raw_probability(scaler, estimator, validation_x)
    rows = []
    choices = []
    for method in calibration_methods:
        calibrator = _fit_calibrator(str(method), raw, validation_y)
        calibrated = _apply_calibrator(str(method), calibrator, raw)
        threshold = _best_f1_threshold(validation_y, calibrated)
        budget_thresholds = _budget_thresholds(calibrated, budgets)
        metrics = phase72b_metrics(
            validation_y,
            calibrated,
            threshold=threshold,
            budgets=budgets,
            ece_bins=ece_bins,
            budget_thresholds=budget_thresholds,
        )
        row = {
            "candidate_id": _candidate_id(config),
            "model_family": config["model_family"],
            "calibration_method": str(method),
            **metrics,
        }
        rows.append(row)
        choices.append(
            {
                "calibration_method": str(method),
                "calibrator": calibrator,
                "probability": calibrated,
                "threshold": threshold,
                "budget_thresholds": budget_thresholds,
                "metrics": metrics,
            }
        )
    selected = min(
        choices,
        key=lambda item: (
            float(item["metrics"]["brier"]),
            float(item["metrics"]["ece"]),
            str(item["calibration_method"]),
        ),
    )
    return (
        {
            "config": dict(config),
            "candidate_id": _candidate_id(config),
            "scaler": scaler,
            "estimator": estimator,
            **selected,
        },
        rows,
    )


def fit_select_phase72b_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    variant_id: str,
    axis_id: str,
    protocol: Mapping[str, object],
    train_indexes: Sequence[int] | None = None,
    validation_indexes: Sequence[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int8)
    validation_y = np.asarray(validation_y, dtype=np.int8)
    if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
        raise ValueError("Phase 72B model selection requires both classes")
    calibration = dict(protocol["calibration"])
    budgets = tuple(float(value) for value in protocol["budgets"])
    candidate_results = []
    validation_rows = []
    configs = _candidate_configs(protocol)
    fitted = Parallel(n_jobs=4, prefer="threads")(
        delayed(_fit_candidate)(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config=config,
            calibration_methods=tuple(calibration["methods"]),
            budgets=budgets,
            ece_bins=int(calibration["ece_bins"]),
            seed=int(protocol["seed"]),
        )
        for config in configs
    )
    for config, (result, rows) in zip(configs, fitted):
        candidate_results.append(result)
        for row in rows:
            validation_rows.append(
                {"variant_id": variant_id, "axis_id": axis_id, **row}
            )
    selected = min(
        candidate_results,
        key=lambda item: (
            -float(item["metrics"]["average_precision"]),
            float(item["metrics"]["brier"]),
            float(item["metrics"]["ece"]),
            str(item["candidate_id"]),
        ),
    )
    bundle = {
        "variant_id": str(variant_id),
        "axis_id": str(axis_id),
        "model_family": selected["config"]["model_family"],
        "candidate_id": selected["candidate_id"],
        "estimator_params": selected["config"],
        "feature_count": int(train_x.shape[1]),
        "scaler": selected["scaler"],
        "estimator": selected["estimator"],
        "calibration_method": selected["calibration_method"],
        "calibrator": selected["calibrator"],
        "f1_threshold": float(selected["threshold"]),
        "budget_thresholds": dict(selected["budget_thresholds"]),
        "validation_metrics": dict(selected["metrics"]),
        "train_index_sha256": _array_sha256(
            np.asarray(
                list(train_indexes) if train_indexes is not None else np.arange(len(train_x)),
                dtype=np.int64,
            )
        ),
        "validation_index_sha256": _array_sha256(
            np.asarray(
                list(validation_indexes)
                if validation_indexes is not None
                else np.arange(len(validation_x)),
                dtype=np.int64,
            )
        ),
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    return bundle, validation_rows


def fit_fixed_phase72b_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    variant_id: str,
    axis_id: str,
    protocol: Mapping[str, object],
    candidate_config: Mapping[str, object],
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    calibration = dict(protocol["calibration"])
    selected, rows = _fit_candidate(
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.int8),
        np.asarray(validation_x, dtype=np.float32),
        np.asarray(validation_y, dtype=np.int8),
        config=candidate_config,
        calibration_methods=tuple(calibration["methods"]),
        budgets=tuple(float(value) for value in protocol["budgets"]),
        ece_bins=int(calibration["ece_bins"]),
        seed=int(protocol["seed"]),
    )
    bundle = {
        "variant_id": str(variant_id),
        "axis_id": str(axis_id),
        "model_family": selected["config"]["model_family"],
        "candidate_id": selected["candidate_id"],
        "estimator_params": selected["config"],
        "feature_count": int(train_x.shape[1]),
        "scaler": selected["scaler"],
        "estimator": selected["estimator"],
        "calibration_method": selected["calibration_method"],
        "calibrator": selected["calibrator"],
        "f1_threshold": float(selected["threshold"]),
        "budget_thresholds": dict(selected["budget_thresholds"]),
        "validation_metrics": dict(selected["metrics"]),
        "train_index_sha256": _array_sha256(np.asarray(train_indexes, np.int64)),
        "validation_index_sha256": _array_sha256(
            np.asarray(validation_indexes, np.int64)
        ),
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    validation_rows = [
        {"variant_id": variant_id, "axis_id": axis_id, **row}
        for row in rows
    ]
    return bundle, validation_rows


def predict_phase72b_bundle(
    bundle: Mapping[str, object], features: np.ndarray
) -> np.ndarray:
    raw = _raw_probability(
        bundle.get("scaler"), bundle["estimator"], np.asarray(features, np.float32)
    )
    return _apply_calibrator(
        str(bundle["calibration_method"]), bundle.get("calibrator"), raw
    )


def load_phase72b_model_bundle(
    path: Path | str, expected_sha256: str
) -> dict[str, object]:
    source = Path(path)
    actual = _file_sha256(source)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError(
            f"Phase 72B model bundle hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return joblib.load(source)


def _variant_matrix(
    variant_id: str,
    matrices: Mapping[str, np.ndarray],
    feature_rows: Sequence[Mapping[str, object]],
    *,
    seed: int | None = None,
    partition_ids: Sequence[str] | None = None,
) -> np.ndarray:
    if variant_id == "explicit_static":
        return matrices["explicit_static"]
    if variant_id == "explicit_history":
        return matrices["explicit_history"]
    if variant_id == "geofm_current_only":
        return matrices["geofm_current"]
    if variant_id == "geofm_temporal_mean_only":
        return matrices["geofm_temporal_mean"]
    if variant_id == "explicit_plus_geofm_current":
        return np.concatenate(
            [matrices["explicit_history"], matrices["geofm_current"]], axis=1
        )
    if variant_id == "explicit_plus_geofm_temporal_full":
        return np.concatenate(
            [matrices["explicit_history"], matrices["geofm_temporal_full"]], axis=1
        )
    if variant_id in _CONTROL_VARIANTS:
        if seed is None:
            raise ValueError(f"Phase 72B control requires a seed: {variant_id}")
        control = build_phase72b_control_features(
            _CONTROL_VARIANTS[variant_id],
            matrices["embedding_history"],
            matrices["history_mask"],
            feature_rows,
            partition_ids=partition_ids,
            seed=int(seed),
            output_dim=matrices["geofm_temporal_full"].shape[1],
        )
        return np.concatenate(
            [matrices["explicit_history"], control["matrix"]], axis=1
        )
    raise ValueError(f"Unknown Phase 72B variant: {variant_id}")


def _axis_partition_ids(
    axis_id: str,
    axis: Mapping[str, object],
    row_count: int,
) -> list[str]:
    partitions = [
        f"{axis_id}:unused:{index}" for index in range(int(row_count))
    ]
    assigned: dict[int, str] = {}
    for split_name in ("train", "validation", "test"):
        partition_id = f"{axis_id}:{split_name}"
        for raw_index in axis.get(split_name, []):
            index = int(raw_index)
            if index < 0 or index >= int(row_count):
                raise ValueError(
                    "Phase 72B axis partition index out of range: "
                    f"{axis_id}/{split_name}/{index}"
                )
            previous = assigned.get(index)
            if previous is not None and previous != partition_id:
                raise ValueError(
                    "Phase 72B axis partition overlap: "
                    f"{axis_id}/{index}/{previous}/{partition_id}"
                )
            assigned[index] = partition_id
            partitions[index] = partition_id
    return partitions


def _fit_control_variant_matrix(
    variant_id: str,
    matrices: Mapping[str, np.ndarray],
    feature_rows: Sequence[Mapping[str, object]],
    *,
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    axis_id: str,
    seed: int,
) -> np.ndarray:
    if variant_id not in _CONTROL_VARIANTS:
        raise ValueError(
            f"Phase 72B fit control variant is unknown: {variant_id}"
        )
    row_count = len(feature_rows)
    result = np.full(
        (
            row_count,
            matrices["explicit_history"].shape[1]
            + matrices["geofm_temporal_full"].shape[1],
        ),
        np.nan,
        dtype=np.float32,
    )
    for split_name, raw_indexes in (
        ("train", train_indexes),
        ("validation", validation_indexes),
    ):
        indexes = np.asarray([int(value) for value in raw_indexes], np.int64)
        if not len(indexes):
            raise ValueError(
                f"Phase 72B control partition is empty: {axis_id}:{split_name}"
            )
        subset_rows = [feature_rows[int(index)] for index in indexes]
        control = build_phase72b_control_features(
            _CONTROL_VARIANTS[variant_id],
            matrices["embedding_history"][indexes],
            matrices["history_mask"][indexes],
            subset_rows,
            partition_ids=[f"{axis_id}:{split_name}"] * len(indexes),
            seed=int(seed),
            output_dim=matrices["geofm_temporal_full"].shape[1],
        )
        result[indexes] = np.concatenate(
            [matrices["explicit_history"][indexes], control["matrix"]],
            axis=1,
        )
    return result


def _development_outcome(
    target_path: Path,
) -> dict[int, int]:
    with np.load(target_path) as loaded:
        return {
            int(index): int(outcome)
            for index, outcome in zip(
                loaded["sample_index"], loaded["conversion_1y"]
            )
        }


def _outcomes_for_indexes(
    outcomes: Mapping[int, int], indexes: Sequence[int]
) -> np.ndarray:
    missing = [int(index) for index in indexes if int(index) not in outcomes]
    if missing:
        raise ValueError(
            f"Phase 72B development outcomes missing indexes: {missing[:5]}"
        )
    return np.asarray([outcomes[int(index)] for index in indexes], np.int8)


def _save_bundle(
    bundle: Mapping[str, object],
    *,
    bundles_dir: Path,
    axis_id: str,
    variant_id: str,
    seed: int | None,
) -> dict[str, object]:
    filename = _bundle_filename(axis_id, variant_id, seed)
    path = bundles_dir / filename
    joblib.dump(dict(bundle), path)
    return {
        "axis_id": axis_id,
        "variant_id": variant_id,
        "control_seed": "" if seed is None else int(seed),
        "bundle_path": str(Path("bundles") / filename),
        "bundle_sha256": _file_sha256(path),
        "candidate_id": bundle["candidate_id"],
        "model_family": bundle["model_family"],
        "calibration_method": bundle["calibration_method"],
        "validation_average_precision": bundle["validation_metrics"][
            "average_precision"
        ],
        "validation_brier": bundle["validation_metrics"]["brier"],
        "validation_ece": bundle["validation_metrics"]["ece"],
    }


def _bundle_filename(
    axis_id: str, variant_id: str, seed: int | None
) -> str:
    suffix = "" if seed is None else f"_seed{int(seed)}"
    return f"{axis_id}__{variant_id}{suffix}.joblib"


def _load_fit_progress(
    path: Path, *, protocol_hash: str, prepared_artifacts_hash: str
) -> dict[str, object]:
    hash_path = path.with_suffix(".sha256")
    if not path.exists() and not hash_path.exists():
        progress = {
            "status": "phase72b_fit_in_progress",
            "frozen_protocol_sha256": protocol_hash,
            "prepared_artifacts_sha256": prepared_artifacts_hash,
            "entries": {},
        }
        write_hashed_json(path, progress)
        return progress
    if not path.exists() or not hash_path.exists():
        raise ValueError("Phase 72B fit progress hash pair is incomplete")
    progress = load_hashed_json(path, hash_path)
    if str(progress.get("frozen_protocol_sha256", "")) != protocol_hash:
        raise ValueError("Phase 72B fit progress protocol hash mismatch")
    if (
        str(progress.get("prepared_artifacts_sha256", ""))
        != prepared_artifacts_hash
    ):
        raise ValueError(
            "Phase 72B prepared artifact manifest hash mismatch in fit progress"
        )
    if not isinstance(progress.get("entries"), dict):
        raise ValueError("Phase 72B fit progress entries are invalid")
    return progress


def _write_fit_progress(path: Path, progress: Mapping[str, object]) -> None:
    write_hashed_json(path, progress)


def _resume_bundle(
    progress: Mapping[str, object],
    *,
    bundles_dir: Path,
    axis_id: str,
    variant_id: str,
    seed: int | None,
    feature_count: int,
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]] | None:
    filename = _bundle_filename(axis_id, variant_id, seed)
    entry = dict(progress.get("entries", {})).get(filename)
    if entry is None:
        return None
    entry = dict(entry)
    record = dict(entry["record"])
    expected_seed = "" if seed is None else int(seed)
    if (
        str(record.get("axis_id")) != axis_id
        or str(record.get("variant_id")) != variant_id
        or record.get("control_seed", "") != expected_seed
        or Path(str(record.get("bundle_path", ""))).name != filename
    ):
        raise ValueError(f"Phase 72B fit progress identity mismatch: {filename}")
    bundle = load_phase72b_model_bundle(
        bundles_dir / filename, str(record["bundle_sha256"])
    )
    if (
        str(bundle.get("axis_id")) != axis_id
        or str(bundle.get("variant_id")) != variant_id
        or int(bundle.get("feature_count", -1)) != int(feature_count)
        or str(bundle.get("train_index_sha256"))
        != _array_sha256(np.asarray(train_indexes, dtype=np.int64))
        or str(bundle.get("validation_index_sha256"))
        != _array_sha256(np.asarray(validation_indexes, dtype=np.int64))
    ):
        raise ValueError(f"Phase 72B resumed bundle contract mismatch: {filename}")
    return bundle, record, [dict(row) for row in entry["validation_rows"]]


def _checkpoint_bundle(
    progress: dict[str, object],
    progress_path: Path,
    *,
    record: Mapping[str, object],
    validation_rows: Sequence[Mapping[str, object]],
) -> None:
    entries = dict(progress.get("entries", {}))
    filename = Path(str(record["bundle_path"])).name
    entries[filename] = {
        "record": dict(record),
        "validation_rows": [dict(row) for row in validation_rows],
    }
    progress["entries"] = entries
    _write_fit_progress(progress_path, progress)


def fit_freeze_phase72b_models(
    *, prepared_dir: Path | str, output_dir: Path | str
) -> tuple[dict[str, object], dict[str, Path]]:
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    bundles_dir = output / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    verified_prepared = load_verified_phase72b_prepared(
        prepared,
        deferred_names={
            "phase72b_development_targets.npz",
            "phase72b_confirmation_targets.npz",
        },
    )
    frozen_protocol = dict(verified_prepared["frozen_protocol"])
    protocol = dict(frozen_protocol["tracked_protocol"])
    protocol_hash = str(verified_prepared["protocol_hash"])
    development_target_path = prepared / "phase72b_development_targets.npz"
    if _target_npz_sha256(development_target_path) != str(
        frozen_protocol.get("development_targets_sha256", "")
    ):
        raise ValueError("Phase 72B development target hash mismatch")
    progress_path = output / "phase72b_fit_progress.json"
    progress = _load_fit_progress(
        progress_path,
        protocol_hash=protocol_hash,
        prepared_artifacts_hash=str(
            verified_prepared["manifest_sha256"]
        ),
    )
    feature_rows = list(verified_prepared["feature_rows"])
    matrices = dict(verified_prepared["matrices"])
    split_registry = dict(verified_prepared["split_registry"])
    outcomes = _development_outcome(
        development_target_path
    )
    bundle_records = []
    validation_rows = []
    axes: dict[str, list[str]] = {}
    selected_control_seeds: dict[str, dict[str, int]] = {}
    selected_configs: dict[str, dict[str, object]] = {}

    search_axes = (
        "pooled_temporal",
        "bishan_to_dongxing",
        "dongxing_to_bishan",
    )
    for axis_id in search_axes:
        axis = split_registry[axis_id]
        train_indexes = [int(value) for value in axis["train"]]
        validation_indexes = [int(value) for value in axis["validation"]]
        train_y = _outcomes_for_indexes(outcomes, train_indexes)
        validation_y = _outcomes_for_indexes(outcomes, validation_indexes)
        axes[axis_id] = []
        selected_configs[axis_id] = {}
        for variant_id in _CORE_VARIANTS:
            matrix = _variant_matrix(variant_id, matrices, feature_rows)
            resumed = _resume_bundle(
                progress,
                bundles_dir=bundles_dir,
                axis_id=axis_id,
                variant_id=variant_id,
                seed=None,
                feature_count=matrix.shape[1],
                train_indexes=train_indexes,
                validation_indexes=validation_indexes,
            )
            if resumed is None:
                bundle, rows = fit_select_phase72b_model(
                    matrix[train_indexes],
                    train_y,
                    matrix[validation_indexes],
                    validation_y,
                    variant_id=variant_id,
                    axis_id=axis_id,
                    protocol=protocol,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                )
                record = _save_bundle(
                    bundle,
                    bundles_dir=bundles_dir,
                    axis_id=axis_id,
                    variant_id=variant_id,
                    seed=None,
                )
                _checkpoint_bundle(
                    progress,
                    progress_path,
                    record=record,
                    validation_rows=rows,
                )
            else:
                bundle, record, rows = resumed
            bundle_records.append(record)
            validation_rows.extend(rows)
            axes[axis_id].append(record["bundle_path"])
            selected_configs[axis_id][variant_id] = dict(
                bundle["estimator_params"]
            )

        selected_control_seeds[axis_id] = {}
        for variant_id in _CONTROL_VARIANTS:
            seed_records = []
            for control_seed in protocol["controls"]["seeds"]:
                matrix = _fit_control_variant_matrix(
                    variant_id,
                    matrices,
                    feature_rows,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                    axis_id=axis_id,
                    seed=int(control_seed),
                )
                resumed = _resume_bundle(
                    progress,
                    bundles_dir=bundles_dir,
                    axis_id=axis_id,
                    variant_id=variant_id,
                    seed=int(control_seed),
                    feature_count=matrix.shape[1],
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                )
                if resumed is None:
                    bundle, rows = fit_select_phase72b_model(
                        matrix[train_indexes],
                        train_y,
                        matrix[validation_indexes],
                        validation_y,
                        variant_id=variant_id,
                        axis_id=axis_id,
                        protocol=protocol,
                        train_indexes=train_indexes,
                        validation_indexes=validation_indexes,
                    )
                    record = _save_bundle(
                        bundle,
                        bundles_dir=bundles_dir,
                        axis_id=axis_id,
                        variant_id=variant_id,
                        seed=int(control_seed),
                    )
                    checkpoint_rows = [
                        {**row, "control_seed": int(control_seed)}
                        for row in rows
                    ]
                    _checkpoint_bundle(
                        progress,
                        progress_path,
                        record=record,
                        validation_rows=checkpoint_rows,
                    )
                else:
                    bundle, record, checkpoint_rows = resumed
                bundle_records.append(record)
                validation_rows.extend(checkpoint_rows)
                axes[axis_id].append(record["bundle_path"])
                seed_records.append((record, bundle))
            best_record, best_bundle = min(
                seed_records,
                key=lambda item: (
                    -float(item[0]["validation_average_precision"]),
                    float(item[0]["validation_brier"]),
                    float(item[0]["validation_ece"]),
                    int(item[0]["control_seed"]),
                ),
            )
            selected_control_seeds[axis_id][variant_id] = int(
                best_record["control_seed"]
            )
            selected_configs[axis_id][variant_id] = dict(
                best_bundle["estimator_params"]
            )

    for axis_id, axis in split_registry.items():
        if not axis_id.startswith("spatial_"):
            continue
        train_indexes = [int(value) for value in axis["train"]]
        validation_indexes = [int(value) for value in axis["validation"]]
        if not train_indexes or not validation_indexes:
            continue
        train_y = _outcomes_for_indexes(outcomes, train_indexes)
        validation_y = _outcomes_for_indexes(outcomes, validation_indexes)
        if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
            continue
        axes[axis_id] = []
        spatial_variants = list(_CORE_VARIANTS)
        for control_variant in _CONTROL_VARIANTS:
            spatial_variants.append(control_variant)
        for variant_id in spatial_variants:
            control_seed = None
            if variant_id in _CONTROL_VARIANTS:
                control_seed = selected_control_seeds["pooled_temporal"][
                    variant_id
                ]
            matrix = (
                _fit_control_variant_matrix(
                    variant_id,
                    matrices,
                    feature_rows,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                    axis_id=axis_id,
                    seed=int(control_seed),
                )
                if variant_id in _CONTROL_VARIANTS
                else _variant_matrix(
                    variant_id, matrices, feature_rows
                )
            )
            config = selected_configs["pooled_temporal"][variant_id]
            resumed = _resume_bundle(
                progress,
                bundles_dir=bundles_dir,
                axis_id=axis_id,
                variant_id=variant_id,
                seed=control_seed,
                feature_count=matrix.shape[1],
                train_indexes=train_indexes,
                validation_indexes=validation_indexes,
            )
            if resumed is None:
                bundle, rows = fit_fixed_phase72b_model(
                    matrix[train_indexes],
                    train_y,
                    matrix[validation_indexes],
                    validation_y,
                    variant_id=variant_id,
                    axis_id=axis_id,
                    protocol=protocol,
                    candidate_config=config,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                )
                record = _save_bundle(
                    bundle,
                    bundles_dir=bundles_dir,
                    axis_id=axis_id,
                    variant_id=variant_id,
                    seed=control_seed,
                )
                checkpoint_rows = [
                    {
                        **row,
                        "control_seed": ""
                        if control_seed is None
                        else int(control_seed),
                    }
                    for row in rows
                ]
                _checkpoint_bundle(
                    progress,
                    progress_path,
                    record=record,
                    validation_rows=checkpoint_rows,
                )
            else:
                bundle, record, checkpoint_rows = resumed
            bundle_records.append(record)
            validation_rows.extend(checkpoint_rows)
            axes[axis_id].append(record["bundle_path"])

    selected = {
        "status": "phase72b_models_frozen",
        "frozen_protocol_sha256": protocol_hash,
        "prepared_artifacts_sha256": str(
            verified_prepared["manifest_sha256"]
        ),
        "axes": axes,
        "selected_control_seeds": selected_control_seeds,
        "bundle_records": bundle_records,
        "validation_metric_rows": len(validation_rows),
        "claim_boundary": PHASE72B_CLAIM_BOUNDARY,
    }
    validation_path = output / "phase72b_validation_metrics.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    selected_json, selected_hash = write_hashed_json(
        output / "phase72b_selected_models.json", selected
    )
    progress["status"] = "phase72b_fit_complete"
    progress["selected_models_sha256"] = selected_hash.read_text(
        encoding="ascii"
    ).strip()
    _write_fit_progress(progress_path, progress)
    return selected, {
        "validation_metrics_csv": validation_path,
        "selected_models_json": selected_json,
        "selected_models_hash": selected_hash,
    }
