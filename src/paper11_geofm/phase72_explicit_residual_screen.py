from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold

from .phase72_two_year_endpoint_screen import (
    load_verified_phase72_two_year_prepared,
)
from .phase72b_geofm_features import build_phase72b_control_features
from .phase72b_metrics import (
    build_phase72b_gate,
    paired_block_bootstrap,
    phase72b_metrics,
)
from .phase72b_models import (
    _apply_calibrator,
    _best_f1_threshold,
    _budget_thresholds,
    _fit_calibrator,
    _fit_estimator,
    _raw_probability,
    fit_fixed_phase72b_model,
    predict_phase72b_bundle,
)
from .phase72b_prepared import (
    load_verified_phase72b_prepared,
    verify_phase72b_prepared_artifact,
)
from .phase72b_protocol import (
    canonical_json_sha256,
    load_hashed_json,
    write_hashed_json,
)
from .phase72b_terrain import _file_sha256


PHASE72_RESIDUAL_ENDPOINTS = (
    "conversion_1y",
    "conversion_2y",
    "noncontinuous_persistence_2y",
)
PHASE72_RESIDUAL_PROTOCOL_SHA256 = (
    "774408f176e2e342de7b8a6027ff46542e2b602954faaaf77be97dff603df353"
)
PHASE72_RESIDUAL_CLAIM_BOUNDARY = (
    "This Phase 72 exhaustion experiment tests whether temporal GeoFM adds "
    "future-risk information above a cross-fitted explicit baseline. It does "
    "not enter Phase 72C, train a temporal neural model, run planning, alter "
    "rewards, establish agronomic suitability, or revise the formal manuscript."
)
PHASE72_RESIDUAL_PRIMARY_VARIANT = "explicit_plus_geofm_residual"
PHASE72_RESIDUAL_EXPLICIT_VARIANT = "explicit_history"
PHASE72_RESIDUAL_CONTROL_VARIANTS = {
    "residual_temporal_order_shuffle": "temporal_order_shuffle",
    "residual_spatial_shuffle": "spatial_shuffle",
    "residual_random_projection": "random_projection",
}
PHASE72_RESIDUAL_AXES = (
    "pooled_temporal",
    "bishan_to_dongxing",
    "dongxing_to_bishan",
    "spatial_bishan_fold0",
    "spatial_bishan_fold1",
    "spatial_bishan_fold2",
    "spatial_bishan_fold3",
    "spatial_bishan_fold4",
    "spatial_dongxing_fold0",
    "spatial_dongxing_fold1",
    "spatial_dongxing_fold2",
    "spatial_dongxing_fold3",
    "spatial_dongxing_fold4",
)


def validate_phase72_explicit_residual_protocol(
    payload: Mapping[str, object],
) -> dict[str, object]:
    protocol = json.loads(json.dumps(dict(payload)))
    if protocol.get("phase") != "phase72_explicit_residual_exhaustion_screen":
        raise ValueError("Phase 72 explicit residual protocol phase mismatch")
    if canonical_json_sha256(protocol) != PHASE72_RESIDUAL_PROTOCOL_SHA256:
        raise ValueError("Phase 72 explicit residual protocol is not frozen")
    return protocol


def load_phase72_explicit_residual_protocol(
    path: Path | str,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 72 explicit residual protocol must be an object")
    return validate_phase72_explicit_residual_protocol(payload)


def spatial_group_cross_fit_assignments(
    groups: Sequence[object], *, folds: int
) -> np.ndarray:
    labels = np.asarray([str(value) for value in groups], dtype=object)
    fold_count = int(folds)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Phase 72 residual cross-fit groups are empty")
    if any(not value.strip() for value in labels.tolist()):
        raise ValueError("Phase 72 residual cross-fit groups contain blanks")
    if len(set(labels.tolist())) < fold_count or fold_count < 2:
        raise ValueError("Phase 72 residual cross-fit has too few groups")
    assignments = np.full(len(labels), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=fold_count)
    positions = np.arange(len(labels), dtype=np.int64)
    for fold_id, (_, held_out) in enumerate(
        splitter.split(positions, groups=labels)
    ):
        assignments[held_out] = int(fold_id)
    if set(assignments.tolist()) != set(range(fold_count)):
        raise ValueError("Phase 72 residual cross-fit coverage is incomplete")
    for group in set(labels.tolist()):
        if len(set(assignments[labels == group].tolist())) != 1:
            raise ValueError("Phase 72 residual cross-fit split a spatial group")
    return assignments


def _validate_residual_inputs(
    explicit_probability: np.ndarray,
    residual_features: np.ndarray,
    outcome: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    probability = np.asarray(explicit_probability, dtype=np.float64)
    features = np.asarray(residual_features, dtype=np.float64)
    target = None if outcome is None else np.asarray(outcome, dtype=np.float64)
    if (
        probability.ndim != 1
        or features.ndim != 2
        or len(probability) != len(features)
        or len(probability) == 0
        or not np.isfinite(probability).all()
        or not np.isfinite(features).all()
        or np.any(probability <= 0)
        or np.any(probability >= 1)
    ):
        raise ValueError("Phase 72 residual model inputs are invalid")
    if target is not None and (
        target.ndim != 1
        or len(target) != len(probability)
        or set(np.unique(target).tolist()) != {0.0, 1.0}
    ):
        raise ValueError("Phase 72 residual model requires both binary classes")
    return probability, features, target


def fit_offset_logistic_residual(
    explicit_probability: np.ndarray,
    residual_features: np.ndarray,
    outcome: np.ndarray,
    *,
    l2_strength: float,
    class_weight: str,
    max_iter: int,
    tolerance: float,
    initial_coefficient: np.ndarray | None = None,
) -> dict[str, object]:
    probability, features, target = _validate_residual_inputs(
        explicit_probability, residual_features, outcome
    )
    assert target is not None
    penalty = float(l2_strength)
    if penalty <= 0 or not np.isfinite(penalty):
        raise ValueError("Phase 72 residual L2 strength must be positive")
    weight_mode = str(class_weight)
    if weight_mode == "none":
        weights = np.ones(len(target), dtype=np.float64)
    elif weight_mode == "balanced":
        positive = float(target.sum())
        negative = float(len(target) - positive)
        weights = np.where(
            target == 1,
            len(target) / (2.0 * positive),
            len(target) / (2.0 * negative),
        )
    else:
        raise ValueError("Phase 72 residual class weight is invalid")

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    standardized = (features - feature_mean) / feature_scale
    offset = np.log(probability / (1.0 - probability))
    weight_sum = float(weights.sum())

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + standardized @ coefficient
        loss = np.logaddexp(0.0, eta) - target * eta
        residual = expit(eta) - target
        value = float(np.dot(weights, loss) / weight_sum)
        value += 0.5 * penalty * float(np.dot(coefficient, coefficient))
        gradient = standardized.T @ (weights * residual) / weight_sum
        gradient += penalty * coefficient
        return value, np.asarray(gradient, dtype=np.float64)

    initial = (
        np.zeros(features.shape[1], dtype=np.float64)
        if initial_coefficient is None
        else np.asarray(initial_coefficient, dtype=np.float64)
    )
    if initial.shape != (features.shape[1],) or not np.isfinite(initial).all():
        raise ValueError("Phase 72 residual initial coefficient is invalid")
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    if not result.success:
        raise RuntimeError(
            "Phase 72 residual optimizer failed: " + str(result.message)
        )
    return {
        "model_family": "offset_logistic_residual",
        "residual_intercept": False,
        "feature_count": int(features.shape[1]),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_scale": np.asarray(feature_scale, dtype=np.float64),
        "coefficient": np.asarray(result.x, dtype=np.float64),
        "l2_strength": penalty,
        "class_weight": weight_mode,
        "optimizer_iterations": int(result.nit),
        "optimizer_objective": float(result.fun),
    }


def predict_offset_logistic_residual(
    bundle: Mapping[str, object],
    explicit_probability: np.ndarray,
    residual_features: np.ndarray,
) -> np.ndarray:
    probability, features, _ = _validate_residual_inputs(
        explicit_probability, residual_features
    )
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(bundle["coefficient"], dtype=np.float64)
    if (
        feature_mean.shape != (features.shape[1],)
        or feature_scale.shape != feature_mean.shape
        or coefficient.shape != feature_mean.shape
        or np.any(feature_scale <= 0)
        or not np.isfinite(coefficient).all()
    ):
        raise ValueError("Phase 72 residual bundle dimensions are invalid")
    standardized = (features - feature_mean) / feature_scale
    offset = np.log(probability / (1.0 - probability))
    return np.clip(expit(offset + standardized @ coefficient), 1e-6, 1 - 1e-6)


def phase72_explicit_residual_overall_status(
    endpoint_results: Mapping[str, Mapping[str, object]],
) -> str:
    if set(endpoint_results) != set(PHASE72_RESIDUAL_ENDPOINTS):
        return "phase72_explicit_residual_inputs_not_ready"
    statuses = {
        str(endpoint_results[endpoint].get("phase72b_status", ""))
        for endpoint in PHASE72_RESIDUAL_ENDPOINTS
    }
    if "phase72b_inputs_not_ready" in statuses or "" in statuses:
        return "phase72_explicit_residual_inputs_not_ready"
    if statuses == {"geofm_information_supported"}:
        return "explicit_residual_information_supported"
    if statuses == {"geofm_information_not_supported"}:
        return "explicit_residual_information_not_supported"
    return "explicit_residual_information_mixed"


def _read_sha256(path: Path) -> str:
    value = path.read_text(encoding="ascii").strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Invalid SHA256 sidecar: {path}")
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verify_source_hashes(
    protocol: Mapping[str, object],
    *,
    phase72b_prepared_dir: Path,
    phase72b_frozen_dir: Path,
    phase72b_confirmation_dir: Path,
    phase72_two_year_prepared_dir: Path,
    phase72_two_year_frozen_dir: Path,
    phase72_two_year_confirmation_dir: Path,
) -> None:
    expected = dict(protocol["source_bindings"])
    actual = {
        "phase72b_prepared_artifacts_sha256": _read_sha256(
            phase72b_prepared_dir / "phase72b_prepared_artifacts.sha256"
        ),
        "phase72b_selected_models_sha256": _read_sha256(
            phase72b_frozen_dir / "phase72b_selected_models.sha256"
        ),
        "phase72b_confirmation_receipt_sha256": _read_sha256(
            phase72b_confirmation_dir / "phase72b_confirmation_receipt.sha256"
        ),
        "phase72_two_year_prepared_sha256": _read_sha256(
            phase72_two_year_prepared_dir / "phase72_two_year_prepared.sha256"
        ),
        "phase72_two_year_selected_models_sha256": _read_sha256(
            phase72_two_year_frozen_dir
            / "phase72_two_year_selected_models.sha256"
        ),
        "phase72_two_year_confirmation_receipt_sha256": _read_sha256(
            phase72_two_year_confirmation_dir
            / "phase72_two_year_confirmation_receipt.sha256"
        ),
    }
    if actual != expected:
        raise ValueError("Phase 72 explicit residual source binding mismatch")
    load_hashed_json(
        phase72b_frozen_dir / "phase72b_selected_models.json"
    )
    load_hashed_json(
        phase72b_confirmation_dir / "phase72b_confirmation_receipt.json"
    )
    load_hashed_json(
        phase72_two_year_frozen_dir
        / "phase72_two_year_selected_models.json"
    )
    load_hashed_json(
        phase72_two_year_confirmation_dir
        / "phase72_two_year_confirmation_receipt.json"
    )


def prepare_phase72_explicit_residual_screen(
    *,
    protocol_path: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    phase72_two_year_prepared_dir: Path | str,
    phase72_two_year_frozen_dir: Path | str,
    phase72_two_year_confirmation_dir: Path | str,
) -> dict[str, object]:
    protocol = load_phase72_explicit_residual_protocol(protocol_path)
    paths = {
        "phase72b_prepared_dir": Path(phase72b_prepared_dir),
        "phase72b_frozen_dir": Path(phase72b_frozen_dir),
        "phase72b_confirmation_dir": Path(phase72b_confirmation_dir),
        "phase72_two_year_prepared_dir": Path(phase72_two_year_prepared_dir),
        "phase72_two_year_frozen_dir": Path(phase72_two_year_frozen_dir),
        "phase72_two_year_confirmation_dir": Path(
            phase72_two_year_confirmation_dir
        ),
    }
    _verify_source_hashes(protocol, **paths)
    one_year = load_verified_phase72b_prepared(
        paths["phase72b_prepared_dir"],
        deferred_names={"phase72b_confirmation_targets.npz"},
    )
    two_year = load_verified_phase72_two_year_prepared(
        paths["phase72_two_year_prepared_dir"],
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=paths["phase72b_prepared_dir"],
        include_confirmation=False,
    )
    endpoint_counts = {
        "conversion_1y": {
            "feature_rows": len(one_year["feature_rows"]),
            "development_rows": sum(
                len(one_year["split_registry"][axis]["train"])
                + len(one_year["split_registry"][axis]["validation"])
                for axis in ("pooled_temporal",)
            ),
            "confirmation_rows": len(
                one_year["split_registry"]["pooled_temporal"]["test"]
            ),
        },
        "conversion_2y": {
            "feature_rows": len(two_year["feature_rows"]),
            "development_rows": len(
                two_year["development_targets"]["sample_index"]
            ),
            "confirmation_rows": len(
                two_year["split_registries"]["conversion_2y"]
                ["pooled_temporal"]["test"]
            ),
        },
        "noncontinuous_persistence_2y": {
            "feature_rows": len(two_year["feature_rows"]),
            "development_rows": len(
                two_year["development_targets"]["sample_index"]
            ),
            "confirmation_rows": len(
                two_year["split_registries"]
                ["noncontinuous_persistence_2y"]["pooled_temporal"]["test"]
            ),
        },
    }
    return {
        "status": "phase72_explicit_residual_inputs_prepared",
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "source_bindings": dict(protocol["source_bindings"]),
        "endpoint_counts": endpoint_counts,
        "confirmation_targets_opened": False,
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
    }


def write_phase72_explicit_residual_prepared_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Phase 72 residual prepared output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest, sidecar = write_hashed_json(
        output / "phase72_explicit_residual_prepared.json", package
    )
    return {"manifest": manifest, "manifest_sha256": sidecar}


def load_verified_phase72_explicit_residual_prepared(
    prepared_dir: Path | str,
    *,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    phase72_two_year_prepared_dir: Path | str,
    phase72_two_year_frozen_dir: Path | str,
    phase72_two_year_confirmation_dir: Path | str,
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    manifest = load_hashed_json(
        prepared / "phase72_explicit_residual_prepared.json"
    )
    if manifest.get("status") != "phase72_explicit_residual_inputs_prepared":
        raise ValueError("Phase 72 explicit residual prepared status mismatch")
    protocol = validate_phase72_explicit_residual_protocol(manifest["protocol"])
    if manifest.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("Phase 72 explicit residual prepared protocol mismatch")
    if manifest.get("confirmation_targets_opened") is not False:
        raise ValueError("Phase 72 explicit residual prepare opened confirmation")
    _verify_source_hashes(
        protocol,
        phase72b_prepared_dir=Path(phase72b_prepared_dir),
        phase72b_frozen_dir=Path(phase72b_frozen_dir),
        phase72b_confirmation_dir=Path(phase72b_confirmation_dir),
        phase72_two_year_prepared_dir=Path(phase72_two_year_prepared_dir),
        phase72_two_year_frozen_dir=Path(phase72_two_year_frozen_dir),
        phase72_two_year_confirmation_dir=Path(
            phase72_two_year_confirmation_dir
        ),
    )
    return {
        "manifest": manifest,
        "manifest_sha256": _read_sha256(
            prepared / "phase72_explicit_residual_prepared.sha256"
        ),
        "protocol": protocol,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _outcome_lookup(
    targets: Mapping[str, np.ndarray], endpoint: str
) -> dict[int, int]:
    indexes = np.asarray(targets["sample_index"], dtype=np.int64)
    outcome = np.asarray(targets[endpoint], dtype=np.int8)
    if (
        indexes.ndim != 1
        or outcome.ndim != 1
        or len(indexes) != len(outcome)
        or len(set(indexes.tolist())) != len(indexes)
        or not set(np.unique(outcome).tolist()).issubset({0, 1})
    ):
        raise ValueError(f"Phase 72 residual target arrays are invalid: {endpoint}")
    return {int(index): int(value) for index, value in zip(indexes, outcome)}


def _y_for_indexes(
    lookup: Mapping[int, int], indexes: Sequence[int]
) -> np.ndarray:
    try:
        return np.asarray([lookup[int(index)] for index in indexes], dtype=np.int8)
    except KeyError as exc:
        raise ValueError("Phase 72 residual outcome alignment mismatch") from exc


def _load_endpoint_sources(
    *,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72_two_year_prepared_dir: Path | str,
    include_confirmation: bool,
) -> dict[str, dict[str, object]]:
    deferred = set()
    if not include_confirmation:
        deferred.add("phase72b_confirmation_targets.npz")
    one_year = load_verified_phase72b_prepared(
        phase72b_prepared_dir, deferred_names=deferred
    )
    one_target_name = (
        "phase72b_confirmation_targets.npz"
        if include_confirmation
        else "phase72b_development_targets.npz"
    )
    verify_phase72b_prepared_artifact(
        Path(phase72b_prepared_dir), one_year["manifest"], one_target_name
    )
    one_targets = _load_npz(Path(phase72b_prepared_dir) / one_target_name)
    two_year = load_verified_phase72_two_year_prepared(
        phase72_two_year_prepared_dir,
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        include_confirmation=include_confirmation,
    )
    two_targets = (
        two_year["confirmation_targets"]
        if include_confirmation
        else two_year["development_targets"]
    )
    return {
        "conversion_1y": {
            "feature_rows": one_year["feature_rows"],
            "matrices": one_year["matrices"],
            "split_registry": one_year["split_registry"],
            "outcomes": _outcome_lookup(one_targets, "conversion_1y"),
        },
        "conversion_2y": {
            "feature_rows": two_year["feature_rows"],
            "matrices": two_year["matrices"],
            "split_registry": two_year["split_registries"]["conversion_2y"],
            "outcomes": _outcome_lookup(two_targets, "conversion_2y"),
        },
        "noncontinuous_persistence_2y": {
            "feature_rows": two_year["feature_rows"],
            "matrices": two_year["matrices"],
            "split_registry": two_year["split_registries"]
            ["noncontinuous_persistence_2y"],
            "outcomes": _outcome_lookup(
                two_targets, "noncontinuous_persistence_2y"
            ),
        },
    }


def _load_reference_explicit_config(
    endpoint: str,
    axis_id: str,
    *,
    phase72b_frozen_dir: Path,
    phase72_two_year_frozen_dir: Path,
) -> dict[str, object]:
    if endpoint == "conversion_1y":
        frozen = phase72b_frozen_dir
        selected_name = "phase72b_selected_models.json"
    else:
        frozen = phase72_two_year_frozen_dir
        selected_name = "phase72_two_year_selected_models.json"
    selected = load_hashed_json(frozen / selected_name)
    matches = []
    for raw_record in selected.get("bundle_records", []):
        record = dict(raw_record)
        if (
            str(record.get("axis_id")) == axis_id
            and str(record.get("variant_id"))
            == PHASE72_RESIDUAL_EXPLICIT_VARIANT
            and (
                endpoint == "conversion_1y"
                or str(record.get("endpoint")) == endpoint
            )
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Phase 72 residual explicit source bundle mismatch: {endpoint}/{axis_id}"
        )
    record = matches[0]
    path = frozen / str(record["bundle_path"])
    if _file_sha256(path) != str(record["bundle_sha256"]):
        raise ValueError("Phase 72 residual explicit source bundle hash mismatch")
    bundle = joblib.load(path)
    if (
        str(bundle.get("axis_id")) != axis_id
        or str(bundle.get("variant_id"))
        != PHASE72_RESIDUAL_EXPLICIT_VARIANT
    ):
        raise ValueError("Phase 72 residual explicit source bundle identity mismatch")
    return dict(bundle["estimator_params"])


def cross_fitted_explicit_probability(
    explicit_features: np.ndarray,
    outcome: np.ndarray,
    spatial_groups: Sequence[object],
    *,
    candidate_config: Mapping[str, object],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    features = np.asarray(explicit_features, dtype=np.float32)
    target = np.asarray(outcome, dtype=np.int8)
    if (
        features.ndim != 2
        or target.ndim != 1
        or len(features) != len(target)
        or set(np.unique(target).tolist()) != {0, 1}
    ):
        raise ValueError("Phase 72 residual explicit cross-fit inputs are invalid")
    assignments = spatial_group_cross_fit_assignments(spatial_groups, folds=folds)
    probability = np.full(len(target), np.nan, dtype=np.float64)
    fold_rows = []
    for fold_id in range(int(folds)):
        held_out = assignments == fold_id
        train = ~held_out
        if set(np.unique(target[train]).tolist()) != {0, 1}:
            raise ValueError("Phase 72 residual explicit cross-fit fold lacks a class")
        scaler, estimator = _fit_estimator(
            features[train],
            target[train],
            candidate_config,
            seed=int(seed),
        )
        probability[held_out] = _raw_probability(
            scaler, estimator, features[held_out]
        )
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "train_rows": int(train.sum()),
                "held_out_rows": int(held_out.sum()),
                "train_positive_rows": int(target[train].sum()),
                "held_out_positive_rows": int(target[held_out].sum()),
            }
        )
    if not np.isfinite(probability).all():
        raise ValueError("Phase 72 residual explicit cross-fit is incomplete")
    return probability, {
        "folds": int(folds),
        "assignment_sha256": _array_sha256(assignments),
        "prediction_sha256": _array_sha256(probability),
        "rows": fold_rows,
        "group_exclusive": True,
    }


def _residual_candidate_id(
    *, l2_strength: float, class_weight: str
) -> str:
    return canonical_json_sha256(
        {
            "model_family": "offset_logistic_residual",
            "l2_strength": float(l2_strength),
            "class_weight": str(class_weight),
            "residual_intercept": False,
        }
    )[:16]


def _fit_select_residual_bundle(
    *,
    base_bundle: Mapping[str, object],
    train_explicit_probability: np.ndarray,
    train_residual_features: np.ndarray,
    train_y: np.ndarray,
    validation_explicit_features: np.ndarray,
    validation_residual_features: np.ndarray,
    validation_y: np.ndarray,
    endpoint: str,
    axis_id: str,
    variant_id: str,
    control_seed: int | None,
    protocol: Mapping[str, object],
    cross_fit_audit: Mapping[str, object],
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    residual = dict(protocol["residual"])
    calibration = dict(protocol["calibration"])
    base_validation_probability = _raw_probability(
        base_bundle.get("scaler"),
        base_bundle["estimator"],
        np.asarray(validation_explicit_features, dtype=np.float32),
    )
    choices = []
    validation_rows = []
    for class_weight in residual["class_weight"]:
        initial_coefficient = None
        for l2_strength in sorted(
            (float(value) for value in residual["l2_strength"]), reverse=True
        ):
            fitted = fit_offset_logistic_residual(
                train_explicit_probability,
                train_residual_features,
                train_y,
                l2_strength=float(l2_strength),
                class_weight=str(class_weight),
                max_iter=int(residual["max_iter"]),
                tolerance=float(residual["tolerance"]),
                initial_coefficient=initial_coefficient,
            )
            initial_coefficient = np.asarray(
                fitted["coefficient"], dtype=np.float64
            )
            raw_probability = predict_offset_logistic_residual(
                fitted,
                base_validation_probability,
                validation_residual_features,
            )
            candidate_id = _residual_candidate_id(
                l2_strength=float(l2_strength), class_weight=str(class_weight)
            )
            candidate_choices = []
            for method in calibration["methods"]:
                calibrator = _fit_calibrator(
                    str(method), raw_probability, validation_y
                )
                probability = _apply_calibrator(
                    str(method), calibrator, raw_probability
                )
                threshold = _best_f1_threshold(validation_y, probability)
                budget_thresholds = _budget_thresholds(
                    probability, protocol["budgets"]
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    metrics = phase72b_metrics(
                        validation_y,
                        probability,
                        threshold=threshold,
                        budgets=protocol["budgets"],
                        ece_bins=int(calibration["ece_bins"]),
                        budget_thresholds=budget_thresholds,
                    )
                validation_rows.append(
                    {
                        "endpoint": endpoint,
                        "axis_id": axis_id,
                        "variant_id": variant_id,
                        "control_seed": ""
                        if control_seed is None
                        else int(control_seed),
                        "candidate_id": candidate_id,
                        "l2_strength": float(l2_strength),
                        "class_weight": str(class_weight),
                        "calibration_method": str(method),
                        **metrics,
                    }
                )
                candidate_choices.append(
                    {
                        "calibration_method": str(method),
                        "calibrator": calibrator,
                        "probability": probability,
                        "threshold": threshold,
                        "budget_thresholds": budget_thresholds,
                        "metrics": metrics,
                    }
                )
            calibration_choice = min(
                candidate_choices,
                key=lambda item: (
                    float(item["metrics"]["brier"]),
                    float(item["metrics"]["ece"]),
                    str(item["calibration_method"]),
                ),
            )
            choices.append(
                {
                    "candidate_id": candidate_id,
                    "residual_model": fitted,
                    **calibration_choice,
                }
            )
    selected = min(
        choices,
        key=lambda item: (
            -float(item["metrics"]["average_precision"]),
            float(item["metrics"]["brier"]),
            float(item["metrics"]["ece"]),
            str(item["candidate_id"]),
        ),
    )
    return {
        "fit_implementation_id": "phase72_explicit_residual_v2_warm_start",
        "endpoint": endpoint,
        "axis_id": axis_id,
        "variant_id": variant_id,
        "control_seed": "" if control_seed is None else int(control_seed),
        "model_family": "offset_logistic_residual",
        "candidate_id": selected["candidate_id"],
        "base_bundle": dict(base_bundle),
        "residual_model": selected["residual_model"],
        "calibration_method": selected["calibration_method"],
        "calibrator": selected["calibrator"],
        "f1_threshold": float(selected["threshold"]),
        "budget_thresholds": dict(selected["budget_thresholds"]),
        "validation_metrics": dict(selected["metrics"]),
        "cross_fit_audit": dict(cross_fit_audit),
        "train_index_sha256": _array_sha256(
            np.asarray(train_indexes, dtype=np.int64)
        ),
        "validation_index_sha256": _array_sha256(
            np.asarray(validation_indexes, dtype=np.int64)
        ),
        "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
    }, validation_rows


def predict_phase72_explicit_residual_bundle(
    bundle: Mapping[str, object],
    explicit_features: np.ndarray,
    residual_features: np.ndarray,
) -> np.ndarray:
    base = dict(bundle["base_bundle"])
    base_probability = _raw_probability(
        base.get("scaler"),
        base["estimator"],
        np.asarray(explicit_features, dtype=np.float32),
    )
    raw = predict_offset_logistic_residual(
        dict(bundle["residual_model"]), base_probability, residual_features
    )
    return _apply_calibrator(
        str(bundle["calibration_method"]), bundle.get("calibrator"), raw
    )


def _residual_feature_matrix(
    endpoint_data: Mapping[str, object],
    indexes: Sequence[int],
    *,
    variant_id: str,
    axis_id: str,
    split_name: str,
    control_seed: int | None,
    protocol: Mapping[str, object],
) -> np.ndarray:
    matrices = dict(endpoint_data["matrices"])
    selected = np.asarray(indexes, dtype=np.int64)
    if variant_id == PHASE72_RESIDUAL_PRIMARY_VARIANT:
        return np.asarray(
            matrices["geofm_temporal_full"][selected], dtype=np.float32
        )
    if variant_id not in PHASE72_RESIDUAL_CONTROL_VARIANTS:
        raise ValueError(f"Unknown Phase 72 residual variant: {variant_id}")
    if control_seed is None:
        raise ValueError("Phase 72 residual control seed is required")
    feature_rows = list(endpoint_data["feature_rows"])
    subset_rows = [feature_rows[int(index)] for index in selected]
    control = build_phase72b_control_features(
        PHASE72_RESIDUAL_CONTROL_VARIANTS[variant_id],
        matrices["embedding_history"][selected],
        matrices["history_mask"][selected],
        subset_rows,
        partition_ids=[f"{axis_id}:{split_name}"] * len(selected),
        seed=int(control_seed),
        output_dim=int(protocol["controls"]["random_projection_dim"]),
        learned_transform_fit_scope=str(
            protocol["controls"]["learned_transform_fit_scope"]
        ),
    )
    return np.asarray(control["matrix"], dtype=np.float32)


def _bundle_key(
    endpoint: str, axis_id: str, variant_id: str, control_seed: int | None
) -> str:
    seed = "noseed" if control_seed is None else f"seed{int(control_seed)}"
    return "__".join((endpoint, axis_id, variant_id, seed))


def _load_fit_progress(
    output: Path, *, prepared_sha256: str, protocol_sha256: str
) -> dict[str, object]:
    path = output / "phase72_explicit_residual_fit_progress.json"
    if not path.exists():
        return {
            "status": "phase72_explicit_residual_fit_in_progress",
            "fit_implementation_id": "phase72_explicit_residual_v2_warm_start",
            "prepared_sha256": prepared_sha256,
            "protocol_sha256": protocol_sha256,
            "entries": [],
            "validation_rows": [],
        }
    progress = load_hashed_json(path)
    if (
        progress.get("prepared_sha256") != prepared_sha256
        or progress.get("protocol_sha256") != protocol_sha256
        or progress.get("fit_implementation_id")
        != "phase72_explicit_residual_v2_warm_start"
    ):
        raise ValueError("Phase 72 residual fit progress binding mismatch")
    return progress


def _write_fit_progress(output: Path, progress: Mapping[str, object]) -> None:
    write_hashed_json(
        output / "phase72_explicit_residual_fit_progress.json", progress
    )


def _resume_bundle(
    output: Path,
    progress: Mapping[str, object],
    *,
    key: str,
) -> tuple[dict[str, object], dict[str, object]] | None:
    matches = [
        dict(record)
        for record in progress.get("entries", [])
        if str(record.get("key")) == key
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Duplicate Phase 72 residual fit record: {key}")
    record = matches[0]
    path = output / str(record["bundle_path"])
    if _file_sha256(path) != str(record["bundle_sha256"]):
        raise ValueError(f"Phase 72 residual resumed bundle hash mismatch: {key}")
    bundle = joblib.load(path)
    if _bundle_key(
        str(bundle.get("endpoint")),
        str(bundle.get("axis_id")),
        str(bundle.get("variant_id")),
        None
        if bundle.get("control_seed", "") == ""
        else int(bundle["control_seed"]),
    ) != key:
        raise ValueError(f"Phase 72 residual resumed bundle identity mismatch: {key}")
    return bundle, record


def _checkpoint_bundle(
    output: Path,
    progress: dict[str, object],
    *,
    bundle: Mapping[str, object],
    validation_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    seed = bundle.get("control_seed", "")
    control_seed = None if seed == "" else int(seed)
    key = _bundle_key(
        str(bundle["endpoint"]),
        str(bundle["axis_id"]),
        str(bundle["variant_id"]),
        control_seed,
    )
    bundle_dir = output / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / f"{key}.joblib"
    if path.exists():
        raise ValueError(f"Untracked Phase 72 residual bundle already exists: {path}")
    joblib.dump(dict(bundle), path, compress=3)
    record = {
        "key": key,
        "endpoint": str(bundle["endpoint"]),
        "axis_id": str(bundle["axis_id"]),
        "variant_id": str(bundle["variant_id"]),
        "control_seed": "" if control_seed is None else int(control_seed),
        "model_family": str(bundle["model_family"]),
        "candidate_id": str(bundle.get("candidate_id", "")),
        "calibration_method": str(bundle["calibration_method"]),
        "validation_average_precision": float(
            bundle["validation_metrics"]["average_precision"]
        ),
        "validation_brier": float(bundle["validation_metrics"]["brier"]),
        "validation_ece": float(bundle["validation_metrics"]["ece"]),
        "bundle_path": path.relative_to(output).as_posix(),
        "bundle_sha256": _file_sha256(path),
    }
    progress["entries"] = [*progress.get("entries", []), record]
    progress["validation_rows"] = [
        *progress.get("validation_rows", []),
        *[dict(row) for row in validation_rows],
    ]
    _write_fit_progress(output, progress)
    return dict(bundle), record


def _baseline_bundle(
    *,
    explicit_matrix: np.ndarray,
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    train_y: np.ndarray,
    validation_y: np.ndarray,
    endpoint: str,
    axis_id: str,
    candidate_config: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bundle, rows = fit_fixed_phase72b_model(
        explicit_matrix[np.asarray(train_indexes, dtype=np.int64)],
        train_y,
        explicit_matrix[np.asarray(validation_indexes, dtype=np.int64)],
        validation_y,
        variant_id=PHASE72_RESIDUAL_EXPLICIT_VARIANT,
        axis_id=axis_id,
        protocol=protocol,
        candidate_config=candidate_config,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
    )
    bundle.update(
        {
            "endpoint": endpoint,
            "control_seed": "",
            "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
        }
    )
    return bundle, [{"endpoint": endpoint, **row} for row in rows]


def fit_freeze_phase72_explicit_residual_models(
    *,
    prepared_dir: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    phase72_two_year_prepared_dir: Path | str,
    phase72_two_year_frozen_dir: Path | str,
    phase72_two_year_confirmation_dir: Path | str,
    output_dir: Path | str,
) -> tuple[dict[str, object], dict[str, Path]]:
    prepared = load_verified_phase72_explicit_residual_prepared(
        prepared_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72b_confirmation_dir=phase72b_confirmation_dir,
        phase72_two_year_prepared_dir=phase72_two_year_prepared_dir,
        phase72_two_year_frozen_dir=phase72_two_year_frozen_dir,
        phase72_two_year_confirmation_dir=phase72_two_year_confirmation_dir,
    )
    protocol = dict(prepared["protocol"])
    sources = _load_endpoint_sources(
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72_two_year_prepared_dir=phase72_two_year_prepared_dir,
        include_confirmation=False,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress = _load_fit_progress(
        output,
        prepared_sha256=str(prepared["manifest_sha256"]),
        protocol_sha256=str(prepared["manifest"]["protocol_sha256"]),
    )
    records: list[dict[str, object]] = []
    selected_control_seeds: dict[str, dict[str, int]] = {}
    for endpoint in PHASE72_RESIDUAL_ENDPOINTS:
        endpoint_data = sources[endpoint]
        rows = list(endpoint_data["feature_rows"])
        matrices = dict(endpoint_data["matrices"])
        registry = dict(endpoint_data["split_registry"])
        outcomes = dict(endpoint_data["outcomes"])
        explicit_matrix = np.asarray(
            matrices["explicit_history"], dtype=np.float32
        )
        selected_control_seeds[endpoint] = {}
        endpoint_control_bundles: dict[str, list[dict[str, object]]] = {
            variant: [] for variant in PHASE72_RESIDUAL_CONTROL_VARIANTS
        }
        for axis_id in PHASE72_RESIDUAL_AXES:
            if axis_id not in registry:
                raise ValueError(f"Phase 72 residual axis is missing: {endpoint}/{axis_id}")
            axis = dict(registry[axis_id])
            train_indexes = [int(value) for value in axis["train"]]
            validation_indexes = [int(value) for value in axis["validation"]]
            train_y = _y_for_indexes(outcomes, train_indexes)
            validation_y = _y_for_indexes(outcomes, validation_indexes)
            candidate_config = _load_reference_explicit_config(
                endpoint,
                axis_id,
                phase72b_frozen_dir=Path(phase72b_frozen_dir),
                phase72_two_year_frozen_dir=Path(
                    phase72_two_year_frozen_dir
                ),
            )
            baseline_key = _bundle_key(
                endpoint,
                axis_id,
                PHASE72_RESIDUAL_EXPLICIT_VARIANT,
                None,
            )
            resumed_baseline = _resume_bundle(
                output, progress, key=baseline_key
            )
            if resumed_baseline is None:
                baseline, baseline_rows = _baseline_bundle(
                    explicit_matrix=explicit_matrix,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                    train_y=train_y,
                    validation_y=validation_y,
                    endpoint=endpoint,
                    axis_id=axis_id,
                    candidate_config=candidate_config,
                    protocol=protocol,
                )
                baseline, baseline_record = _checkpoint_bundle(
                    output,
                    progress,
                    bundle=baseline,
                    validation_rows=baseline_rows,
                )
            else:
                baseline, baseline_record = resumed_baseline
            records.append(baseline_record)

            residual_specs = [(PHASE72_RESIDUAL_PRIMARY_VARIANT, None)]
            if axis_id == "pooled_temporal":
                residual_specs.extend(
                    (variant_id, int(seed))
                    for variant_id in PHASE72_RESIDUAL_CONTROL_VARIANTS
                    for seed in protocol["controls"]["seeds"]
                )
            missing_specs = [
                (variant_id, seed)
                for variant_id, seed in residual_specs
                if _resume_bundle(
                    output,
                    progress,
                    key=_bundle_key(endpoint, axis_id, variant_id, seed),
                )
                is None
            ]
            cross_fit_probability = None
            cross_fit_audit = None
            if missing_specs:
                train_positions = np.asarray(train_indexes, dtype=np.int64)
                groups = [
                    str(rows[int(index)]["spatial_block_id"])
                    for index in train_positions
                ]
                cross_fit_probability, cross_fit_audit = (
                    cross_fitted_explicit_probability(
                        explicit_matrix[train_positions],
                        train_y,
                        groups,
                        candidate_config=candidate_config,
                        folds=int(protocol["residual"]["cross_fit_folds"]),
                        seed=int(protocol["seed"]),
                    )
                )
            for variant_id, control_seed in residual_specs:
                key = _bundle_key(
                    endpoint, axis_id, variant_id, control_seed
                )
                resumed = _resume_bundle(output, progress, key=key)
                if resumed is None:
                    assert cross_fit_probability is not None
                    assert cross_fit_audit is not None
                    train_residual = _residual_feature_matrix(
                        endpoint_data,
                        train_indexes,
                        variant_id=variant_id,
                        axis_id=axis_id,
                        split_name="train",
                        control_seed=control_seed,
                        protocol=protocol,
                    )
                    validation_residual = _residual_feature_matrix(
                        endpoint_data,
                        validation_indexes,
                        variant_id=variant_id,
                        axis_id=axis_id,
                        split_name="validation",
                        control_seed=control_seed,
                        protocol=protocol,
                    )
                    bundle, validation_rows = _fit_select_residual_bundle(
                        base_bundle=baseline,
                        train_explicit_probability=cross_fit_probability,
                        train_residual_features=train_residual,
                        train_y=train_y,
                        validation_explicit_features=explicit_matrix[
                            np.asarray(validation_indexes, dtype=np.int64)
                        ],
                        validation_residual_features=validation_residual,
                        validation_y=validation_y,
                        endpoint=endpoint,
                        axis_id=axis_id,
                        variant_id=variant_id,
                        control_seed=control_seed,
                        protocol=protocol,
                        cross_fit_audit=cross_fit_audit,
                        train_indexes=train_indexes,
                        validation_indexes=validation_indexes,
                    )
                    bundle, record = _checkpoint_bundle(
                        output,
                        progress,
                        bundle=bundle,
                        validation_rows=validation_rows,
                    )
                else:
                    bundle, record = resumed
                records.append(record)
                if variant_id in endpoint_control_bundles:
                    endpoint_control_bundles[variant_id].append(bundle)

        for variant_id, bundles in endpoint_control_bundles.items():
            if len(bundles) != len(protocol["controls"]["seeds"]):
                raise ValueError(
                    f"Phase 72 residual control seed coverage mismatch: {endpoint}/{variant_id}"
                )
            selected = min(
                bundles,
                key=lambda bundle: (
                    -float(bundle["validation_metrics"]["average_precision"]),
                    float(bundle["validation_metrics"]["brier"]),
                    float(bundle["validation_metrics"]["ece"]),
                    int(bundle["control_seed"]),
                ),
            )
            selected_control_seeds[endpoint][variant_id] = int(
                selected["control_seed"]
            )

    records = sorted(records, key=lambda record: str(record["key"]))
    if len(records) != 123 or len({str(record["key"]) for record in records}) != 123:
        raise ValueError("Phase 72 residual frozen bundle coverage mismatch")
    selected_models = {
        "status": "phase72_explicit_residual_models_frozen",
        "prepared_sha256": prepared["manifest_sha256"],
        "protocol_sha256": prepared["manifest"]["protocol_sha256"],
        "selected_control_seeds": selected_control_seeds,
        "bundle_records": records,
        "bundle_count": len(records),
        "confirmation_targets_opened": False,
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
    }
    validation_path = output / "phase72_explicit_residual_validation_metrics.csv"
    pd.DataFrame(progress["validation_rows"]).to_csv(
        validation_path, index=False
    )
    selected_path, selected_hash = write_hashed_json(
        output / "phase72_explicit_residual_selected_models.json",
        selected_models,
    )
    progress["status"] = "phase72_explicit_residual_fit_complete"
    progress["selected_models_sha256"] = _read_sha256(selected_hash)
    _write_fit_progress(output, progress)
    return selected_models, {
        "validation_metrics": validation_path,
        "selected_models": selected_path,
        "selected_models_sha256": selected_hash,
    }


def _metric_delta(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float]:
    return {
        "ap_delta": round(
            float(candidate["average_precision"])
            - float(baseline["average_precision"]),
            12,
        ),
        "brier_delta": round(
            float(baseline["brier"]) - float(candidate["brier"]), 12
        ),
        "ece_delta": round(
            float(baseline["ece"]) - float(candidate["ece"]), 12
        ),
    }


def confirm_phase72_explicit_residual_screen(
    *,
    prepared_dir: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    phase72_two_year_prepared_dir: Path | str,
    phase72_two_year_frozen_dir: Path | str,
    phase72_two_year_confirmation_dir: Path | str,
    frozen_dir: Path | str,
) -> dict[str, object]:
    prepared = load_verified_phase72_explicit_residual_prepared(
        prepared_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72b_confirmation_dir=phase72b_confirmation_dir,
        phase72_two_year_prepared_dir=phase72_two_year_prepared_dir,
        phase72_two_year_frozen_dir=phase72_two_year_frozen_dir,
        phase72_two_year_confirmation_dir=phase72_two_year_confirmation_dir,
    )
    protocol = dict(prepared["protocol"])
    frozen = Path(frozen_dir)
    selected = load_hashed_json(
        frozen / "phase72_explicit_residual_selected_models.json"
    )
    selected_sha256 = _read_sha256(
        frozen / "phase72_explicit_residual_selected_models.sha256"
    )
    if selected.get("status") != "phase72_explicit_residual_models_frozen":
        raise ValueError("Phase 72 explicit residual models are not frozen")
    if (
        selected.get("prepared_sha256") != prepared["manifest_sha256"]
        or selected.get("protocol_sha256")
        != prepared["manifest"]["protocol_sha256"]
        or selected.get("confirmation_targets_opened") is not False
    ):
        raise ValueError("Phase 72 explicit residual frozen binding mismatch")
    sources = _load_endpoint_sources(
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72_two_year_prepared_dir=phase72_two_year_prepared_dir,
        include_confirmation=True,
    )
    bundles: dict[tuple[str, str, str, int | None], dict[str, object]] = {}
    bundle_hashes = {}
    for raw_record in selected["bundle_records"]:
        record = dict(raw_record)
        path = frozen / str(record["bundle_path"])
        actual_hash = _file_sha256(path)
        if actual_hash != str(record["bundle_sha256"]):
            raise ValueError(
                f"Phase 72 explicit residual frozen bundle hash mismatch: {record['key']}"
            )
        bundle = joblib.load(path)
        seed = record.get("control_seed", "")
        key = (
            str(record["endpoint"]),
            str(record["axis_id"]),
            str(record["variant_id"]),
            None if seed == "" else int(seed),
        )
        if key in bundles:
            raise ValueError("Duplicate Phase 72 explicit residual bundle")
        bundles[key] = bundle
        bundle_hashes[str(record["key"])] = actual_hash
    if len(bundles) != 123:
        raise ValueError("Phase 72 explicit residual bundle count mismatch")

    metrics_rows = []
    prediction_rows = []
    bootstrap_rows = []
    control_rows = []
    transfer_rows = []
    spatial_rows = []
    endpoint_results = {}
    for endpoint in PHASE72_RESIDUAL_ENDPOINTS:
        endpoint_data = sources[endpoint]
        rows = list(endpoint_data["feature_rows"])
        matrices = dict(endpoint_data["matrices"])
        registry = dict(endpoint_data["split_registry"])
        outcomes = dict(endpoint_data["outcomes"])
        explicit_matrix = np.asarray(
            matrices["explicit_history"], dtype=np.float32
        )
        groups = {}
        for key, bundle in bundles.items():
            bundle_endpoint, axis_id, variant_id, control_seed = key
            if bundle_endpoint != endpoint:
                continue
            indexes = np.asarray(registry[axis_id]["test"], dtype=np.int64)
            y = _y_for_indexes(outcomes, indexes)
            explicit = explicit_matrix[indexes]
            if variant_id == PHASE72_RESIDUAL_EXPLICIT_VARIANT:
                probability = predict_phase72b_bundle(bundle, explicit)
            else:
                residual_features = _residual_feature_matrix(
                    endpoint_data,
                    indexes,
                    variant_id=variant_id,
                    axis_id=axis_id,
                    split_name="test",
                    control_seed=control_seed,
                    protocol=protocol,
                )
                probability = predict_phase72_explicit_residual_bundle(
                    bundle, explicit, residual_features
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                metric = phase72b_metrics(
                    y,
                    probability,
                    threshold=float(bundle["f1_threshold"]),
                    budgets=protocol["budgets"],
                    ece_bins=int(protocol["calibration"]["ece_bins"]),
                    budget_thresholds=bundle["budget_thresholds"],
                )
            row = {
                "endpoint": endpoint,
                "axis_id": axis_id,
                "variant_id": variant_id,
                "control_seed": ""
                if control_seed is None
                else int(control_seed),
                "rows": len(indexes),
                "positives": int(y.sum()),
                "prevalence": round(float(y.mean()), 12),
                "model_family": str(bundle["model_family"]),
                "calibration_method": str(bundle["calibration_method"]),
                **metric,
            }
            metrics_rows.append(row)
            groups[(axis_id, variant_id, control_seed)] = {
                "indexes": indexes,
                "outcome": y,
                "probability": probability,
                "metric": row,
            }
            for position, local_index in enumerate(indexes):
                source = rows[int(local_index)]
                prediction_rows.append(
                    {
                        "endpoint": endpoint,
                        "sample_index": int(local_index),
                        "source_sample_index": int(
                            source.get("source_sample_index", local_index)
                        ),
                        "axis_id": axis_id,
                        "variant_id": variant_id,
                        "control_seed": ""
                        if control_seed is None
                        else int(control_seed),
                        "outcome": int(y[position]),
                        "probability": round(
                            float(probability[position]), 12
                        ),
                        "region_id": str(source["region_id"]),
                        "spatial_block_id": str(source["spatial_block_id"]),
                        "origin_year": int(source["origin_year"]),
                    }
                )

        def group(
            axis_id: str, variant_id: str, control_seed: int | None = None
        ) -> dict[str, object]:
            try:
                return groups[(axis_id, variant_id, control_seed)]
            except KeyError as exc:
                raise ValueError(
                    "Missing Phase 72 explicit residual confirmation group: "
                    f"{endpoint}/{axis_id}/{variant_id}/{control_seed}"
                ) from exc

        pooled_explicit = group(
            "pooled_temporal", PHASE72_RESIDUAL_EXPLICIT_VARIANT
        )
        pooled_primary = group(
            "pooled_temporal", PHASE72_RESIDUAL_PRIMARY_VARIANT
        )
        pooled_delta = _metric_delta(
            pooled_explicit["metric"], pooled_primary["metric"]
        )
        pooled_indexes = pooled_primary["indexes"]
        pooled_bootstrap = paired_block_bootstrap(
            pooled_primary["outcome"],
            pooled_explicit["probability"],
            pooled_primary["probability"],
            [rows[int(index)] for index in pooled_indexes],
            iterations=int(protocol["bootstrap"]["iterations"]),
            seed=int(protocol["bootstrap"]["seed"]),
        )
        bootstrap_rows.append(
            {
                "endpoint": endpoint,
                "comparison_id": "residual_vs_explicit",
                "axis_id": "pooled_temporal",
                **pooled_bootstrap,
            }
        )
        endpoint_controls = []
        for variant_id, control_id in PHASE72_RESIDUAL_CONTROL_VARIANTS.items():
            selected_seed = int(
                selected["selected_control_seeds"][endpoint][variant_id]
            )
            control_group = group(
                "pooled_temporal", variant_id, selected_seed
            )
            control_row = {
                "endpoint": endpoint,
                "control_id": control_id,
                "variant_id": variant_id,
                "selected_seed": selected_seed,
                **_metric_delta(
                    control_group["metric"], pooled_primary["metric"]
                ),
            }
            endpoint_controls.append(control_row)
            control_rows.append(control_row)
        endpoint_transfers = []
        for axis_id in (
            "bishan_to_dongxing",
            "dongxing_to_bishan",
        ):
            explicit = group(axis_id, PHASE72_RESIDUAL_EXPLICIT_VARIANT)
            primary = group(axis_id, PHASE72_RESIDUAL_PRIMARY_VARIANT)
            transfer_row = {
                "endpoint": endpoint,
                "axis_id": axis_id,
                "source_region": axis_id.split("_to_")[0],
                "target_region": axis_id.split("_to_")[1],
                "rows": len(primary["indexes"]),
                **_metric_delta(explicit["metric"], primary["metric"]),
            }
            endpoint_transfers.append(transfer_row)
            transfer_rows.append(transfer_row)
        endpoint_spatial = []
        for axis_id in sorted(
            name for name in registry if name.startswith("spatial_")
        ):
            explicit = group(axis_id, PHASE72_RESIDUAL_EXPLICIT_VARIANT)
            primary = group(axis_id, PHASE72_RESIDUAL_PRIMARY_VARIANT)
            region_id = axis_id.removeprefix("spatial_").split("_fold", 1)[0]
            spatial_row = {
                "endpoint": endpoint,
                "axis_id": axis_id,
                "region_id": region_id,
                "rows": len(primary["indexes"]),
                **_metric_delta(explicit["metric"], primary["metric"]),
            }
            endpoint_spatial.append(spatial_row)
            spatial_rows.append(spatial_row)
        gate = build_phase72b_gate(
            pooled_delta=pooled_delta,
            pooled_bootstrap=pooled_bootstrap,
            control_rows=endpoint_controls,
            transfer_rows=endpoint_transfers,
            spatial_rows=endpoint_spatial,
            leakage_ok=True,
            gates=protocol["gates"],
        )
        endpoint_results[endpoint] = {
            **gate,
            "pooled_delta": pooled_delta,
            "pooled_bootstrap": pooled_bootstrap,
        }

    status = phase72_explicit_residual_overall_status(endpoint_results)
    return {
        "phase": "phase72_explicit_residual_exhaustion_screen",
        "phase72_explicit_residual_status": status,
        "decision_rule": protocol["decision_rule"],
        "phase72c_allowed": False,
        "endpoint_results": endpoint_results,
        "metrics_rows": metrics_rows,
        "prediction_rows": prediction_rows,
        "bootstrap_rows": bootstrap_rows,
        "control_rows": control_rows,
        "transfer_rows": transfer_rows,
        "spatial_rows": spatial_rows,
        "counts": {
            "endpoints": len(endpoint_results),
            "metric_rows": len(metrics_rows),
            "prediction_rows": len(prediction_rows),
            "bundle_count": len(bundles),
        },
        "prepared_sha256": prepared["manifest_sha256"],
        "selected_models_sha256": selected_sha256,
        "bundle_hashes": bundle_hashes,
        "next_action": (
            "Record this residual result in the Phase 72 exhaustion analysis. "
            "Do not enter Phase 72C or revise the formal manuscript from this screen."
        ),
        "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
    }


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"Phase 72 explicit residual CSV rows are empty: {path.name}")
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _confirmation_markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Phase 72 Explicit Residual Exhaustion Screen",
        "",
        f"Status: `{result['phase72_explicit_residual_status']}`",
        "",
        "| Endpoint | Status | AP delta | Brier delta | ECE delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for endpoint, endpoint_result in result["endpoint_results"].items():
        delta = endpoint_result["pooled_delta"]
        lines.append(
            f"| `{endpoint}` | `{endpoint_result['phase72b_status']}` | "
            f"{delta['ap_delta']:.6f} | {delta['brier_delta']:.6f} | "
            f"{delta['ece_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(result["next_action"]),
            "",
            "## Claim Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_phase72_explicit_residual_confirmation_artifacts(
    result: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Phase 72 explicit residual confirmation output must be new or empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics": output / "phase72_explicit_residual_metrics.csv",
        "predictions": output / "phase72_explicit_residual_predictions.csv",
        "bootstrap": output
        / "phase72_explicit_residual_bootstrap_deltas.csv",
        "controls": output
        / "phase72_explicit_residual_control_comparison.csv",
        "transfers": output
        / "phase72_explicit_residual_transfer_summary.csv",
        "spatial": output / "phase72_explicit_residual_spatial_summary.csv",
        "result": output / "phase72_explicit_residual_screen.json",
        "markdown": output / "phase72_explicit_residual_screen.md",
    }
    row_fields = {
        "metrics": "metrics_rows",
        "predictions": "prediction_rows",
        "bootstrap": "bootstrap_rows",
        "controls": "control_rows",
        "transfers": "transfer_rows",
        "spatial": "spatial_rows",
    }
    for artifact_key, result_key in row_fields.items():
        _write_csv(artifacts[artifact_key], result[result_key])
    artifacts["result"].write_text(
        json.dumps(
            _json_ready(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {
                        "metrics_rows",
                        "prediction_rows",
                        "bootstrap_rows",
                        "control_rows",
                        "transfer_rows",
                        "spatial_rows",
                    }
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    artifacts["markdown"].write_text(
        _confirmation_markdown(result), encoding="utf-8"
    )
    receipt = {
        "status": "phase72_explicit_residual_confirmation_receipt",
        "phase72_explicit_residual_status": result[
            "phase72_explicit_residual_status"
        ],
        "prepared_sha256": result["prepared_sha256"],
        "selected_models_sha256": result["selected_models_sha256"],
        "artifacts": [
            {"name": path.name, "sha256": _file_sha256(path)}
            for path in artifacts.values()
        ],
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_RESIDUAL_CLAIM_BOUNDARY,
    }
    receipt_path, receipt_hash = write_hashed_json(
        output / "phase72_explicit_residual_confirmation_receipt.json",
        receipt,
    )
    artifacts["receipt"] = receipt_path
    artifacts["receipt_sha256"] = receipt_hash
    return artifacts
