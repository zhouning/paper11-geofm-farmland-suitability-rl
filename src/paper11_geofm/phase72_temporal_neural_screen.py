from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .phase72_explicit_residual_screen import (
    _array_sha256,
    _load_npz,
    _load_reference_explicit_config,
    _metric_delta,
    _outcome_lookup,
    _read_sha256,
    _y_for_indexes,
    cross_fitted_explicit_probability,
)
from .phase72b_geofm_features import (
    _derived_seed,
    build_phase72b_random_projection,
)
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


PHASE72_TEMPORAL_NEURAL_PROTOCOL_SHA256 = (
    "f62448b84a3f640c78a12cbe7a2e1cd8891fdc640a408d3ff64d277c2a56da95"
)
PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY = (
    "This bounded Phase 72 exhaustion experiment tests a compact gated "
    "temporal neural residual for one-year product-label conversion. It does "
    "not enter Phase 72C, train a two-year neural model, run planning, alter "
    "rewards, establish agronomic suitability, or revise the formal manuscript."
)
PHASE72_TEMPORAL_NEURAL_ENDPOINT = "conversion_1y"
PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT = (
    "explicit_plus_gated_temporal_neural_residual"
)
PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT = "explicit_history"
PHASE72_TEMPORAL_NEURAL_CONTROL_VARIANTS = {
    "neural_temporal_order_shuffle": "temporal_order_shuffle",
    "neural_spatial_shuffle": "spatial_shuffle",
    "neural_random_projection": "random_projection",
}
PHASE72_TEMPORAL_NEURAL_AXES = (
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
PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID = (
    "phase72_temporal_neural_residual_v1"
)


def validate_phase72_temporal_neural_protocol(
    payload: Mapping[str, object],
) -> dict[str, object]:
    protocol = json.loads(json.dumps(dict(payload)))
    if protocol.get("phase") != "phase72_temporal_neural_exhaustion_screen":
        raise ValueError("Phase 72 temporal neural protocol phase mismatch")
    if canonical_json_sha256(protocol) != PHASE72_TEMPORAL_NEURAL_PROTOCOL_SHA256:
        raise ValueError("Phase 72 temporal neural protocol is not frozen")
    return protocol


def load_phase72_temporal_neural_protocol(
    path: Path | str,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 72 temporal neural protocol must be an object")
    return validate_phase72_temporal_neural_protocol(payload)


def validate_prefix_history_mask(history_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(history_mask, dtype=bool)
    if mask.ndim != 2 or len(mask) == 0:
        raise ValueError("Phase 72 temporal history mask must be a matrix")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Phase 72 temporal history mask must be non-empty")
    if np.any(np.diff(mask.astype(np.int8), axis=1) > 0):
        raise ValueError("Phase 72 temporal history mask must be a valid prefix")
    return mask


def _validate_history(
    embedding_history: np.ndarray, history_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(embedding_history, dtype=np.float32)
    mask = validate_prefix_history_mask(history_mask)
    if (
        history.ndim != 3
        or history.shape[:2] != mask.shape
        or not np.isfinite(history[mask]).all()
    ):
        raise ValueError("Phase 72 temporal embedding history is invalid")
    return history, mask


def fit_history_standardizer(
    embedding_history: np.ndarray, history_mask: np.ndarray
) -> dict[str, object]:
    history, mask = _validate_history(embedding_history, history_mask)
    observed = np.asarray(history[mask], dtype=np.float64)
    mean = observed.mean(axis=0)
    scale = observed.std(axis=0)
    scale[scale == 0] = 1.0
    return {
        "mean": np.asarray(mean, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "fit_rows": int(len(history)),
        "fit_valid_steps": int(mask.sum()),
        "fit_history_sha256": _array_sha256(history),
        "fit_mask_sha256": _array_sha256(mask),
        "scope": "training_valid_history_entries_only",
    }


def transform_history_with_standardizer(
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
    standardizer: Mapping[str, object],
) -> np.ndarray:
    history, mask = _validate_history(embedding_history, history_mask)
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    scale = np.asarray(standardizer["scale"], dtype=np.float32)
    if (
        mean.shape != (history.shape[2],)
        or scale.shape != mean.shape
        or np.any(scale <= 0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("Phase 72 temporal history standardizer is invalid")
    transformed = (history - mean[None, None, :]) / scale[None, None, :]
    transformed[~mask] = 0.0
    if not np.isfinite(transformed).all():
        raise ValueError("Phase 72 standardized temporal history is invalid")
    return np.asarray(transformed, dtype=np.float32)


def build_phase72_temporal_control_history(
    control_id: str,
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
    sample_rows: Sequence[Mapping[str, object]],
    *,
    partition_ids: Sequence[str],
    seed: int,
) -> dict[str, object]:
    history, mask = _validate_history(embedding_history, history_mask)
    if len(sample_rows) != len(history) or len(partition_ids) != len(history):
        raise ValueError("Phase 72 temporal control rows are not aligned")
    partitions = np.asarray([str(value) for value in partition_ids], dtype=object)
    if any(not value.strip() for value in partitions.tolist()):
        raise ValueError("Phase 72 temporal control partitions contain blanks")
    try:
        sample_ids = [int(row["sample_index"]) for row in sample_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Phase 72 temporal controls require integer sample indexes"
        ) from exc
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Phase 72 temporal control sample indexes are duplicate")

    control_name = str(control_id)
    controlled_history = history.copy()
    controlled_mask = mask.copy()
    source_by_target = list(range(len(history)))
    if control_name == "temporal_order_shuffle":
        for row_index in range(len(history)):
            valid_indexes = np.flatnonzero(mask[row_index])
            earlier_positions = valid_indexes[:-1]
            if len(earlier_positions) <= 1:
                continue
            row_rng = np.random.default_rng(
                _derived_seed(
                    int(seed),
                    control_name,
                    partitions[row_index],
                    sample_ids[row_index],
                )
            )
            permutation = row_rng.permutation(len(earlier_positions))
            if np.array_equal(permutation, np.arange(len(earlier_positions))):
                permutation = np.roll(permutation, 1)
            original = history[row_index, earlier_positions].copy()
            controlled_history[row_index, earlier_positions] = original[permutation]
    elif control_name == "spatial_shuffle":
        groups: dict[tuple[str, str, int], list[int]] = {}
        for index, row in enumerate(sample_rows):
            key = (
                str(partitions[index]),
                str(row["region_id"]),
                int(row["origin_year"]),
            )
            groups.setdefault(key, []).append(index)
        for group_key, indexes in groups.items():
            targets = np.asarray(indexes, dtype=np.int64)
            group_rng = np.random.default_rng(
                _derived_seed(int(seed), control_name, *group_key)
            )
            sources = group_rng.permutation(targets)
            if len(targets) > 1 and np.array_equal(targets, sources):
                sources = np.roll(sources, 1)
            controlled_history[targets] = history[sources]
            controlled_mask[targets] = mask[sources]
            for target, source in zip(targets.tolist(), sources.tolist()):
                source_by_target[target] = source
    elif control_name == "random_projection":
        projection = build_phase72b_random_projection(
            input_dim=history.shape[2],
            output_dim=history.shape[2],
            seed=int(seed),
        )
        controlled_history = np.asarray(history @ projection, dtype=np.float32)
    else:
        raise ValueError(f"Unknown Phase 72 temporal control: {control_name}")

    validate_prefix_history_mask(controlled_mask)
    cross_partition_count = sum(
        partitions[target] != partitions[source]
        for target, source in enumerate(source_by_target)
    )
    if cross_partition_count:
        raise ValueError("Phase 72 temporal control crossed a split partition")
    return {
        "history": np.asarray(controlled_history, dtype=np.float32),
        "mask": np.asarray(controlled_mask, dtype=bool),
        "manifest": {
            "control_id": control_name,
            "seed": int(seed),
            "partition_ids": sorted(set(partitions.tolist())),
            "source_index_by_target": source_by_target,
            "cross_partition_count": int(cross_partition_count),
            "shape_matched": controlled_history.shape == history.shape,
            "temporal_shuffle_preserves_current": (
                control_name != "temporal_order_shuffle"
                or all(
                    np.array_equal(
                        controlled_history[index, np.flatnonzero(mask[index])[-1]],
                        history[index, np.flatnonzero(mask[index])[-1]],
                    )
                    for index in range(len(history))
                )
            ),
        },
    }


class Phase72GatedTemporalResidual(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        projection_channels: int,
        maximum_history_steps: int,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.projection_channels = int(projection_channels)
        self.maximum_history_steps = int(maximum_history_steps)
        if min(
            self.input_channels,
            self.projection_channels,
            self.maximum_history_steps,
        ) <= 0:
            raise ValueError("Phase 72 temporal neural dimensions must be positive")
        self.projection = nn.Linear(
            self.input_channels, self.projection_channels, bias=False
        )
        self.content_gate = nn.Linear(self.projection_channels, 1, bias=False)
        self.relative_position_weight = nn.Parameter(torch.zeros(()))
        self.residual_head = nn.Linear(self.projection_channels, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        explicit_logit: torch.Tensor,
    ) -> torch.Tensor:
        if (
            history.ndim != 3
            or history.shape[2] != self.input_channels
            or history.shape[1] > self.maximum_history_steps
            or history_mask.shape != history.shape[:2]
            or explicit_logit.shape != (history.shape[0],)
        ):
            raise ValueError("Phase 72 temporal neural tensor shape mismatch")
        mask = history_mask.to(dtype=torch.bool)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Phase 72 temporal neural history is empty")
        projected = torch.tanh(self.projection(history))
        lengths = mask.sum(dim=1)
        steps = torch.arange(
            history.shape[1], dtype=history.dtype, device=history.device
        )[None, :]
        last = (lengths - 1).to(dtype=history.dtype)[:, None]
        denominator = torch.clamp(last, min=1.0)
        relative_position = (steps - last) / denominator
        gate_logit = self.content_gate(projected).squeeze(-1)
        gate_logit = gate_logit + self.relative_position_weight * relative_position
        weights = torch.sigmoid(gate_logit) * mask.to(dtype=history.dtype)
        pooled = (projected * weights[..., None]).sum(dim=1)
        pooled = pooled / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        residual = self.residual_head(pooled).squeeze(-1)
        return explicit_logit + residual


def _state_dict_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in model.state_dict().items()
    }


def _load_model_from_residual(
    residual: Mapping[str, object],
) -> Phase72GatedTemporalResidual:
    model = Phase72GatedTemporalResidual(
        input_channels=int(residual["input_channels"]),
        projection_channels=int(residual["projection_channels"]),
        maximum_history_steps=int(residual["maximum_history_steps"]),
    )
    state = {
        str(name): torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for name, value in dict(residual["state_dict"]).items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(value / (1.0 - value))


def fit_gated_temporal_residual(
    train_explicit_probability: np.ndarray,
    train_history: np.ndarray,
    train_mask: np.ndarray,
    train_outcome: np.ndarray,
    validation_explicit_probability: np.ndarray,
    validation_history: np.ndarray,
    validation_mask: np.ndarray,
    validation_outcome: np.ndarray,
    *,
    architecture: Mapping[str, object],
    training: Mapping[str, object],
    max_epochs_override: int | None = None,
) -> dict[str, object]:
    train_history, train_mask = _validate_history(train_history, train_mask)
    validation_history, validation_mask = _validate_history(
        validation_history, validation_mask
    )
    train_y = np.asarray(train_outcome, dtype=np.float32)
    validation_y = np.asarray(validation_outcome, dtype=np.float32)
    train_probability = np.asarray(train_explicit_probability, dtype=np.float64)
    validation_probability = np.asarray(
        validation_explicit_probability, dtype=np.float64
    )
    if (
        len(train_history) != len(train_y)
        or len(validation_history) != len(validation_y)
        or len(train_probability) != len(train_y)
        or len(validation_probability) != len(validation_y)
        or set(np.unique(train_y).tolist()) != {0.0, 1.0}
        or set(np.unique(validation_y).tolist()) != {0.0, 1.0}
        or np.any(train_probability <= 0)
        or np.any(train_probability >= 1)
        or np.any(validation_probability <= 0)
        or np.any(validation_probability >= 1)
    ):
        raise ValueError("Phase 72 temporal neural fit inputs are invalid")
    if str(training["class_weight"]) != "none":
        raise ValueError("Phase 72 temporal neural class weight is not frozen")

    standardizer = fit_history_standardizer(train_history, train_mask)
    train_x = transform_history_with_standardizer(
        train_history, train_mask, standardizer
    )
    validation_x = transform_history_with_standardizer(
        validation_history, validation_mask, standardizer
    )
    seed = int(training["seed"])
    torch.manual_seed(seed)
    model = Phase72GatedTemporalResidual(
        input_channels=int(architecture["input_channels"]),
        projection_channels=int(architecture["projection_channels"]),
        maximum_history_steps=int(architecture["maximum_history_steps"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_tensor = torch.from_numpy(train_x)
    train_mask_tensor = torch.from_numpy(train_mask)
    train_offset_tensor = torch.from_numpy(_logit(train_probability).astype(np.float32))
    train_y_tensor = torch.from_numpy(train_y)
    validation_tensor = torch.from_numpy(validation_x)
    validation_mask_tensor = torch.from_numpy(validation_mask)
    validation_offset_tensor = torch.from_numpy(
        _logit(validation_probability).astype(np.float32)
    )
    batch_size = int(training["batch_size"])
    maximum_epochs = int(
        training["max_epochs"]
        if max_epochs_override is None
        else max_epochs_override
    )
    if maximum_epochs <= 0 or batch_size <= 0:
        raise ValueError("Phase 72 temporal neural epoch or batch budget is invalid")
    minimum_epochs = min(int(training["minimum_epochs"]), maximum_epochs)
    patience = int(training["early_stopping_patience"])
    minimum_delta = float(training["minimum_delta"])
    rng = np.random.default_rng(seed)
    best_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, np.ndarray] | None = None
    epochs_without_improvement = 0
    history_rows = []
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            permutation = rng.permutation(len(train_y))
            epoch_loss = 0.0
            for start in range(0, len(permutation), batch_size):
                indexes = torch.from_numpy(permutation[start : start + batch_size])
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    train_tensor[indexes],
                    train_mask_tensor[indexes],
                    train_offset_tensor[indexes],
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits, train_y_tensor[indexes]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(indexes)
            model.eval()
            with torch.no_grad():
                validation_logits = model(
                    validation_tensor,
                    validation_mask_tensor,
                    validation_offset_tensor,
                )
                validation_raw = torch.sigmoid(validation_logits).cpu().numpy()
            brier = float(np.mean((validation_raw - validation_y) ** 2))
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(epoch_loss / len(train_y)),
                    "validation_brier": brier,
                }
            )
            if brier < best_brier - minimum_delta or best_state is None:
                best_brier = brier
                best_epoch = epoch
                best_state = _state_dict_to_numpy(model)
                epochs_without_improvement = 0
            elif epoch >= minimum_epochs:
                epochs_without_improvement += 1
            if epoch >= minimum_epochs and epochs_without_improvement >= patience:
                break
    finally:
        torch.set_num_threads(previous_threads)
    if best_state is None:
        raise RuntimeError("Phase 72 temporal neural training produced no model")
    return {
        "model_family": "compact_gated_temporal_neural_residual",
        "input_channels": int(architecture["input_channels"]),
        "projection_channels": int(architecture["projection_channels"]),
        "maximum_history_steps": int(architecture["maximum_history_steps"]),
        "residual_intercept": False,
        "state_dict": best_state,
        "standardizer": standardizer,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_rows)),
        "best_validation_brier": float(best_brier),
        "training_history": history_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "seed": seed,
        "fit_implementation_id": PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID,
    }


def predict_gated_temporal_residual(
    residual: Mapping[str, object],
    explicit_probability: np.ndarray,
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
) -> np.ndarray:
    history, mask = _validate_history(embedding_history, history_mask)
    probability = np.asarray(explicit_probability, dtype=np.float64)
    if len(probability) != len(history) or np.any(probability <= 0) or np.any(
        probability >= 1
    ):
        raise ValueError("Phase 72 temporal neural prediction offset is invalid")
    transformed = transform_history_with_standardizer(
        history, mask, dict(residual["standardizer"])
    )
    model = _load_model_from_residual(residual)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(transformed),
            torch.from_numpy(mask),
            torch.from_numpy(_logit(probability).astype(np.float32)),
        )
        result = torch.sigmoid(logits).cpu().numpy()
    return np.clip(np.asarray(result, dtype=np.float64), 1e-6, 1 - 1e-6)


def _verify_source_hashes(
    protocol: Mapping[str, object],
    *,
    phase72b_prepared_dir: Path,
    phase72b_frozen_dir: Path,
    phase72b_confirmation_dir: Path,
) -> None:
    load_hashed_json(phase72b_frozen_dir / "phase72b_selected_models.json")
    load_hashed_json(
        phase72b_confirmation_dir / "phase72b_confirmation_receipt.json"
    )
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
    }
    expected = {str(key): str(value) for key, value in protocol["source_bindings"].items()}
    if actual != expected:
        raise ValueError("Phase 72 temporal neural source binding mismatch")


def prepare_phase72_temporal_neural_screen(
    *,
    protocol_path: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
) -> dict[str, object]:
    protocol = load_phase72_temporal_neural_protocol(protocol_path)
    _verify_source_hashes(
        protocol,
        phase72b_prepared_dir=Path(phase72b_prepared_dir),
        phase72b_frozen_dir=Path(phase72b_frozen_dir),
        phase72b_confirmation_dir=Path(phase72b_confirmation_dir),
    )
    prepared = load_verified_phase72b_prepared(
        phase72b_prepared_dir,
        deferred_names={"phase72b_confirmation_targets.npz"},
    )
    history, mask = _validate_history(
        prepared["matrices"]["embedding_history"],
        prepared["matrices"]["history_mask"],
    )
    architecture = dict(protocol["architecture"])
    if history.shape[1:] != (
        int(architecture["maximum_history_steps"]),
        int(architecture["input_channels"]),
    ):
        raise ValueError("Phase 72 temporal neural history shape mismatch")
    pooled = dict(prepared["split_registry"]["pooled_temporal"])
    return {
        "status": "phase72_temporal_neural_inputs_prepared",
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "source_bindings": dict(protocol["source_bindings"]),
        "counts": {
            "feature_rows": len(prepared["feature_rows"]),
            "development_rows": len(pooled["train"]) + len(pooled["validation"]),
            "confirmation_rows": len(pooled["test"]),
            "valid_history_steps": int(mask.sum()),
            "minimum_history_length": int(mask.sum(axis=1).min()),
            "maximum_history_length": int(mask.sum(axis=1).max()),
        },
        "history_sha256": _array_sha256(history),
        "history_mask_sha256": _array_sha256(mask),
        "confirmation_targets_opened": False,
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
    }


def write_phase72_temporal_neural_prepared_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Phase 72 temporal neural prepared output must be new or empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest, sidecar = write_hashed_json(
        output / "phase72_temporal_neural_prepared.json", package
    )
    return {"manifest": manifest, "manifest_sha256": sidecar}


def load_verified_phase72_temporal_neural_prepared(
    prepared_dir: Path | str,
    *,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    manifest = load_hashed_json(
        prepared / "phase72_temporal_neural_prepared.json"
    )
    if manifest.get("status") != "phase72_temporal_neural_inputs_prepared":
        raise ValueError("Phase 72 temporal neural prepared status mismatch")
    protocol = validate_phase72_temporal_neural_protocol(manifest["protocol"])
    if manifest.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("Phase 72 temporal neural prepared protocol mismatch")
    if manifest.get("confirmation_targets_opened") is not False:
        raise ValueError("Phase 72 temporal neural prepare opened confirmation")
    _verify_source_hashes(
        protocol,
        phase72b_prepared_dir=Path(phase72b_prepared_dir),
        phase72b_frozen_dir=Path(phase72b_frozen_dir),
        phase72b_confirmation_dir=Path(phase72b_confirmation_dir),
    )
    return {
        "manifest": manifest,
        "manifest_sha256": _read_sha256(
            prepared / "phase72_temporal_neural_prepared.sha256"
        ),
        "protocol": protocol,
    }


def _load_development_source(
    phase72b_prepared_dir: Path | str,
) -> dict[str, object]:
    prepared_path = Path(phase72b_prepared_dir)
    prepared = load_verified_phase72b_prepared(
        prepared_path,
        deferred_names={"phase72b_confirmation_targets.npz"},
    )
    verify_phase72b_prepared_artifact(
        prepared_path,
        prepared["manifest"],
        "phase72b_development_targets.npz",
    )
    targets = _load_npz(prepared_path / "phase72b_development_targets.npz")
    return {
        "feature_rows": prepared["feature_rows"],
        "matrices": prepared["matrices"],
        "split_registry": prepared["split_registry"],
        "outcomes": _outcome_lookup(targets, PHASE72_TEMPORAL_NEURAL_ENDPOINT),
        "manifest": prepared["manifest"],
        "manifest_sha256": prepared["manifest_sha256"],
    }


def _fit_explicit_baseline(
    *,
    source: Mapping[str, object],
    endpoint: str,
    axis_id: str,
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    train_y: np.ndarray,
    validation_y: np.ndarray,
    phase72b_frozen_dir: Path,
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = _load_reference_explicit_config(
        endpoint,
        axis_id,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72_two_year_frozen_dir=phase72b_frozen_dir,
    )
    explicit = np.asarray(source["matrices"]["explicit_history"], dtype=np.float32)
    bundle, validation_rows = fit_fixed_phase72b_model(
        explicit[np.asarray(train_indexes, dtype=np.int64)],
        train_y,
        explicit[np.asarray(validation_indexes, dtype=np.int64)],
        validation_y,
        variant_id=PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT,
        axis_id=axis_id,
        protocol=protocol,
        candidate_config=config,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
    )
    bundle.update(
        {
            "endpoint": endpoint,
            "control_seed": "",
            "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
        }
    )
    return bundle, [{"endpoint": endpoint, **row} for row in validation_rows]


def _fit_select_neural_bundle(
    *,
    base_bundle: Mapping[str, object],
    train_explicit_probability: np.ndarray,
    train_history: np.ndarray,
    train_mask: np.ndarray,
    train_y: np.ndarray,
    validation_explicit_features: np.ndarray,
    validation_history: np.ndarray,
    validation_mask: np.ndarray,
    validation_y: np.ndarray,
    endpoint: str,
    axis_id: str,
    variant_id: str,
    control_seed: int | None,
    protocol: Mapping[str, object],
    cross_fit_audit: Mapping[str, object],
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    max_epochs_override: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    explicit_validation = _raw_probability(
        base_bundle.get("scaler"),
        base_bundle["estimator"],
        np.asarray(validation_explicit_features, dtype=np.float32),
    )
    residual = fit_gated_temporal_residual(
        train_explicit_probability=train_explicit_probability,
        train_history=train_history,
        train_mask=train_mask,
        train_outcome=train_y,
        validation_explicit_probability=explicit_validation,
        validation_history=validation_history,
        validation_mask=validation_mask,
        validation_outcome=validation_y,
        architecture=dict(protocol["architecture"]),
        training=dict(protocol["training"]),
        max_epochs_override=max_epochs_override,
    )
    raw_probability = predict_gated_temporal_residual(
        residual,
        explicit_validation,
        validation_history,
        validation_mask,
    )
    calibration = dict(protocol["calibration"])
    choices = []
    validation_rows = []
    for method in calibration["methods"]:
        calibrator = _fit_calibrator(str(method), raw_probability, validation_y)
        probability = _apply_calibrator(str(method), calibrator, raw_probability)
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
                "control_seed": "" if control_seed is None else int(control_seed),
                "calibration_method": str(method),
                "best_epoch": int(residual["best_epoch"]),
                "epochs_completed": int(residual["epochs_completed"]),
                **metrics,
            }
        )
        choices.append(
            {
                "calibration_method": str(method),
                "calibrator": calibrator,
                "probability": probability,
                "threshold": threshold,
                "budget_thresholds": budget_thresholds,
                "metrics": metrics,
            }
        )
    selected = min(
        choices,
        key=lambda item: (
            -float(item["metrics"]["average_precision"]),
            float(item["metrics"]["brier"]),
            float(item["metrics"]["ece"]),
            str(item["calibration_method"]),
        ),
    )
    return {
        "fit_implementation_id": PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID,
        "endpoint": endpoint,
        "axis_id": axis_id,
        "variant_id": variant_id,
        "control_seed": "" if control_seed is None else int(control_seed),
        "model_family": "compact_gated_temporal_neural_residual",
        "base_bundle": dict(base_bundle),
        "residual_model": residual,
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
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
    }, validation_rows


def predict_phase72_temporal_neural_bundle(
    bundle: Mapping[str, object],
    explicit_features: np.ndarray,
    embedding_history: np.ndarray,
    history_mask: np.ndarray,
) -> np.ndarray:
    base = dict(bundle["base_bundle"])
    explicit_probability = _raw_probability(
        base.get("scaler"),
        base["estimator"],
        np.asarray(explicit_features, dtype=np.float32),
    )
    raw = predict_gated_temporal_residual(
        dict(bundle["residual_model"]),
        explicit_probability,
        embedding_history,
        history_mask,
    )
    return _apply_calibrator(
        str(bundle["calibration_method"]), bundle.get("calibrator"), raw
    )


def _bundle_key(
    axis_id: str, variant_id: str, control_seed: int | None
) -> str:
    seed = "noseed" if control_seed is None else f"seed{int(control_seed)}"
    return "__".join((PHASE72_TEMPORAL_NEURAL_ENDPOINT, axis_id, variant_id, seed))


def _load_fit_progress(
    output: Path, *, prepared_sha256: str, protocol_sha256: str
) -> dict[str, object]:
    path = output / "phase72_temporal_neural_fit_progress.json"
    if not path.exists():
        return {
            "status": "phase72_temporal_neural_fit_in_progress",
            "fit_implementation_id": PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID,
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
        != PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID
    ):
        raise ValueError("Phase 72 temporal neural fit progress binding mismatch")
    return progress


def _write_fit_progress(output: Path, progress: Mapping[str, object]) -> None:
    write_hashed_json(output / "phase72_temporal_neural_fit_progress.json", progress)


def _resume_bundle(
    output: Path, progress: Mapping[str, object], *, key: str
) -> tuple[dict[str, object], dict[str, object]] | None:
    matches = [
        dict(record)
        for record in progress.get("entries", [])
        if str(record.get("key")) == key
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Duplicate Phase 72 temporal neural fit record: {key}")
    record = matches[0]
    path = output / str(record["bundle_path"])
    if _file_sha256(path) != str(record["bundle_sha256"]):
        raise ValueError(f"Phase 72 temporal neural resumed bundle hash mismatch: {key}")
    bundle = joblib.load(path)
    if _bundle_key(
        str(bundle["axis_id"]),
        str(bundle["variant_id"]),
        None if bundle.get("control_seed", "") == "" else int(bundle["control_seed"]),
    ) != key:
        raise ValueError(f"Phase 72 temporal neural resumed identity mismatch: {key}")
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
    key = _bundle_key(str(bundle["axis_id"]), str(bundle["variant_id"]), control_seed)
    bundle_dir = output / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / f"{key}.joblib"
    if path.exists():
        raise ValueError(f"Untracked Phase 72 temporal neural bundle exists: {path}")
    joblib.dump(dict(bundle), path, compress=3)
    record = {
        "key": key,
        "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
        "axis_id": str(bundle["axis_id"]),
        "variant_id": str(bundle["variant_id"]),
        "control_seed": "" if control_seed is None else int(control_seed),
        "model_family": str(bundle["model_family"]),
        "calibration_method": str(bundle["calibration_method"]),
        "validation_average_precision": float(
            bundle["validation_metrics"]["average_precision"]
        ),
        "validation_brier": float(bundle["validation_metrics"]["brier"]),
        "validation_ece": float(bundle["validation_metrics"]["ece"]),
        "best_epoch": int(bundle["residual_model"]["best_epoch"]),
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


def _history_for_split(
    source: Mapping[str, object], indexes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    matrices = source["matrices"]
    selected = np.asarray(indexes, dtype=np.int64)
    return (
        np.asarray(matrices["embedding_history"][selected], dtype=np.float32),
        np.asarray(matrices["history_mask"][selected], dtype=bool),
    )


def _control_split_history(
    source: Mapping[str, object],
    indexes: Sequence[int],
    *,
    axis_id: str,
    split_name: str,
    variant_id: str,
    control_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    selected = np.asarray(indexes, dtype=np.int64)
    rows = [source["feature_rows"][int(index)] for index in selected]
    history, mask = _history_for_split(source, selected)
    control = build_phase72_temporal_control_history(
        PHASE72_TEMPORAL_NEURAL_CONTROL_VARIANTS[variant_id],
        history,
        mask,
        rows,
        partition_ids=[f"{axis_id}:{split_name}"] * len(selected),
        seed=int(control_seed),
    )
    return control["history"], control["mask"], control["manifest"]


def fit_freeze_phase72_temporal_neural_models(
    *,
    prepared_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    output_dir: Path | str,
) -> tuple[dict[str, object], dict[str, Path]]:
    prepared = load_verified_phase72_temporal_neural_prepared(
        prepared_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72b_confirmation_dir=phase72b_confirmation_dir,
    )
    protocol = dict(prepared["protocol"])
    source = _load_development_source(phase72b_prepared_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress = _load_fit_progress(
        output,
        prepared_sha256=str(prepared["manifest_sha256"]),
        protocol_sha256=str(prepared["manifest"]["protocol_sha256"]),
    )
    records: list[dict[str, object]] = []
    selected_control_seeds = {PHASE72_TEMPORAL_NEURAL_ENDPOINT: {}}
    control_bundles: dict[str, list[dict[str, object]]] = {
        variant_id: [] for variant_id in PHASE72_TEMPORAL_NEURAL_CONTROL_VARIANTS
    }
    explicit_matrix = np.asarray(source["matrices"]["explicit_history"], dtype=np.float32)
    for axis_id in PHASE72_TEMPORAL_NEURAL_AXES:
        if axis_id not in source["split_registry"]:
            raise ValueError(f"Phase 72 temporal neural axis is missing: {axis_id}")
        axis = dict(source["split_registry"][axis_id])
        train_indexes = [int(value) for value in axis["train"]]
        validation_indexes = [int(value) for value in axis["validation"]]
        train_y = _y_for_indexes(source["outcomes"], train_indexes)
        validation_y = _y_for_indexes(source["outcomes"], validation_indexes)
        baseline_key = _bundle_key(axis_id, PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT, None)
        resumed_baseline = _resume_bundle(output, progress, key=baseline_key)
        if resumed_baseline is None:
            baseline, baseline_rows = _fit_explicit_baseline(
                source=source,
                endpoint=PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                axis_id=axis_id,
                train_indexes=train_indexes,
                validation_indexes=validation_indexes,
                train_y=train_y,
                validation_y=validation_y,
                phase72b_frozen_dir=Path(phase72b_frozen_dir),
                protocol=protocol,
            )
            baseline, baseline_record = _checkpoint_bundle(
                output, progress, bundle=baseline, validation_rows=baseline_rows
            )
        else:
            baseline, baseline_record = resumed_baseline
        records.append(baseline_record)

        residual_specs: list[tuple[str, int | None]] = [
            (PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT, None)
        ]
        if axis_id == "pooled_temporal":
            residual_specs.extend(
                (variant_id, int(seed))
                for variant_id in PHASE72_TEMPORAL_NEURAL_CONTROL_VARIANTS
                for seed in protocol["controls"]["seeds"]
            )
        missing_specs = [
            (variant_id, control_seed)
            for variant_id, control_seed in residual_specs
            if _resume_bundle(
                output,
                progress,
                key=_bundle_key(axis_id, variant_id, control_seed),
            )
            is None
        ]
        cross_fit_probability = None
        cross_fit_audit = None
        if missing_specs:
            groups = [
                str(source["feature_rows"][int(index)]["spatial_block_id"])
                for index in train_indexes
            ]
            candidate_config = _load_reference_explicit_config(
                PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                axis_id,
                phase72b_frozen_dir=Path(phase72b_frozen_dir),
                phase72_two_year_frozen_dir=Path(phase72b_frozen_dir),
            )
            cross_fit_probability, cross_fit_audit = cross_fitted_explicit_probability(
                explicit_matrix[np.asarray(train_indexes, dtype=np.int64)],
                train_y,
                groups,
                candidate_config=candidate_config,
                folds=int(protocol["explicit_baseline"]["cross_fit_folds"]),
                seed=int(protocol["seed"]),
            )
        for variant_id, control_seed in residual_specs:
            key = _bundle_key(axis_id, variant_id, control_seed)
            resumed = _resume_bundle(output, progress, key=key)
            if resumed is None:
                assert cross_fit_probability is not None
                assert cross_fit_audit is not None
                if variant_id == PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT:
                    train_history, train_mask = _history_for_split(
                        source, train_indexes
                    )
                    validation_history, validation_mask = _history_for_split(
                        source, validation_indexes
                    )
                else:
                    assert control_seed is not None
                    train_history, train_mask, _ = _control_split_history(
                        source,
                        train_indexes,
                        axis_id=axis_id,
                        split_name="train",
                        variant_id=variant_id,
                        control_seed=control_seed,
                    )
                    validation_history, validation_mask, _ = _control_split_history(
                        source,
                        validation_indexes,
                        axis_id=axis_id,
                        split_name="validation",
                        variant_id=variant_id,
                        control_seed=control_seed,
                    )
                bundle, validation_rows = _fit_select_neural_bundle(
                    base_bundle=baseline,
                    train_explicit_probability=cross_fit_probability,
                    train_history=train_history,
                    train_mask=train_mask,
                    train_y=train_y,
                    validation_explicit_features=explicit_matrix[
                        np.asarray(validation_indexes, dtype=np.int64)
                    ],
                    validation_history=validation_history,
                    validation_mask=validation_mask,
                    validation_y=validation_y,
                    endpoint=PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                    axis_id=axis_id,
                    variant_id=variant_id,
                    control_seed=control_seed,
                    protocol=protocol,
                    cross_fit_audit=cross_fit_audit,
                    train_indexes=train_indexes,
                    validation_indexes=validation_indexes,
                )
                bundle, record = _checkpoint_bundle(
                    output, progress, bundle=bundle, validation_rows=validation_rows
                )
            else:
                bundle, record = resumed
            records.append(record)
            if axis_id == "pooled_temporal" and variant_id in control_bundles:
                control_bundles[variant_id].append(bundle)

    expected_count = len(PHASE72_TEMPORAL_NEURAL_AXES) * 2 + 15
    records = sorted(records, key=lambda record: str(record["key"]))
    if len(records) != expected_count or len({str(record["key"]) for record in records}) != expected_count:
        raise ValueError("Phase 72 temporal neural frozen bundle coverage mismatch")
    for variant_id, bundles in control_bundles.items():
        if len(bundles) != len(protocol["controls"]["seeds"]):
            raise ValueError(
                f"Phase 72 temporal neural control seed coverage mismatch: {variant_id}"
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
        selected_control_seeds[PHASE72_TEMPORAL_NEURAL_ENDPOINT][variant_id] = int(
            selected["control_seed"]
        )
    validation_path = output / "phase72_temporal_neural_validation_metrics.csv"
    pd.DataFrame(progress["validation_rows"]).to_csv(validation_path, index=False)
    selected_models = {
        "status": "phase72_temporal_neural_models_frozen",
        "prepared_sha256": prepared["manifest_sha256"],
        "protocol_sha256": prepared["manifest"]["protocol_sha256"],
        "fit_implementation_id": PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID,
        "selected_control_seeds": selected_control_seeds,
        "bundle_records": records,
        "bundle_count": len(records),
        "confirmation_targets_opened": False,
        "phase72c_allowed": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
    }
    selected_path, selected_hash = write_hashed_json(
        output / "phase72_temporal_neural_selected_models.json",
        selected_models,
    )
    progress["status"] = "phase72_temporal_neural_fit_complete"
    progress["selected_models_sha256"] = _read_sha256(selected_hash)
    _write_fit_progress(output, progress)
    return selected_models, {
        "validation_metrics": validation_path,
        "selected_models": selected_path,
        "selected_models_sha256": selected_hash,
    }


def benchmark_phase72_temporal_neural_model(
    *,
    prepared_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
) -> dict[str, object]:
    prepared = load_verified_phase72_temporal_neural_prepared(
        prepared_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72b_confirmation_dir=phase72b_confirmation_dir,
    )
    protocol = dict(prepared["protocol"])
    source = _load_development_source(phase72b_prepared_dir)
    axis_id = "pooled_temporal"
    axis = dict(source["split_registry"][axis_id])
    train_indexes = [int(value) for value in axis["train"]]
    validation_indexes = [int(value) for value in axis["validation"]]
    train_y = _y_for_indexes(source["outcomes"], train_indexes)
    validation_y = _y_for_indexes(source["outcomes"], validation_indexes)
    baseline, _ = _fit_explicit_baseline(
        source=source,
        endpoint=PHASE72_TEMPORAL_NEURAL_ENDPOINT,
        axis_id=axis_id,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
        train_y=train_y,
        validation_y=validation_y,
        phase72b_frozen_dir=Path(phase72b_frozen_dir),
        protocol=protocol,
    )
    explicit_matrix = np.asarray(source["matrices"]["explicit_history"], dtype=np.float32)
    groups = [
        str(source["feature_rows"][int(index)]["spatial_block_id"])
        for index in train_indexes
    ]
    candidate_config = _load_reference_explicit_config(
        PHASE72_TEMPORAL_NEURAL_ENDPOINT,
        axis_id,
        phase72b_frozen_dir=Path(phase72b_frozen_dir),
        phase72_two_year_frozen_dir=Path(phase72b_frozen_dir),
    )
    cross_fit_probability, audit = cross_fitted_explicit_probability(
        explicit_matrix[np.asarray(train_indexes, dtype=np.int64)],
        train_y,
        groups,
        candidate_config=candidate_config,
        folds=int(protocol["explicit_baseline"]["cross_fit_folds"]),
        seed=int(protocol["seed"]),
    )
    train_history, train_mask = _history_for_split(source, train_indexes)
    validation_history, validation_mask = _history_for_split(source, validation_indexes)
    started = time.perf_counter()
    bundle, _ = _fit_select_neural_bundle(
        base_bundle=baseline,
        train_explicit_probability=cross_fit_probability,
        train_history=train_history,
        train_mask=train_mask,
        train_y=train_y,
        validation_explicit_features=explicit_matrix[
            np.asarray(validation_indexes, dtype=np.int64)
        ],
        validation_history=validation_history,
        validation_mask=validation_mask,
        validation_y=validation_y,
        endpoint=PHASE72_TEMPORAL_NEURAL_ENDPOINT,
        axis_id=axis_id,
        variant_id=PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT,
        control_seed=None,
        protocol=protocol,
        cross_fit_audit=audit,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
        max_epochs_override=int(protocol["training"]["runtime_benchmark_epochs"]),
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "phase72_temporal_neural_runtime_benchmark_complete",
        "axis_id": axis_id,
        "train_rows": len(train_indexes),
        "validation_rows": len(validation_indexes),
        "epochs": int(protocol["training"]["runtime_benchmark_epochs"]),
        "parameter_count": int(bundle["residual_model"]["parameter_count"]),
        "elapsed_seconds": float(elapsed),
        "confirmation_targets_opened": False,
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
    }


def _load_confirmation_source(
    phase72b_prepared_dir: Path | str,
) -> dict[str, object]:
    prepared_path = Path(phase72b_prepared_dir)
    prepared = load_verified_phase72b_prepared(prepared_path)
    verify_phase72b_prepared_artifact(
        prepared_path,
        prepared["manifest"],
        "phase72b_confirmation_targets.npz",
    )
    targets = _load_npz(prepared_path / "phase72b_confirmation_targets.npz")
    return {
        "feature_rows": prepared["feature_rows"],
        "matrices": prepared["matrices"],
        "split_registry": prepared["split_registry"],
        "outcomes": _outcome_lookup(targets, PHASE72_TEMPORAL_NEURAL_ENDPOINT),
        "manifest": prepared["manifest"],
        "manifest_sha256": prepared["manifest_sha256"],
    }


def phase72_temporal_neural_status(gate_status: str) -> str:
    statuses = {
        "phase72b_inputs_not_ready": "phase72_temporal_neural_inputs_not_ready",
        "geofm_information_not_supported": (
            "temporal_neural_information_not_supported"
        ),
        "geofm_information_mixed": "temporal_neural_information_mixed",
        "geofm_information_supported": "temporal_neural_information_supported",
    }
    try:
        return statuses[str(gate_status)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Phase 72 temporal neural gate status: {gate_status}"
        ) from exc


def confirm_phase72_temporal_neural_screen(
    *,
    prepared_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    phase72b_frozen_dir: Path | str,
    phase72b_confirmation_dir: Path | str,
    frozen_dir: Path | str,
) -> dict[str, object]:
    prepared = load_verified_phase72_temporal_neural_prepared(
        prepared_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        phase72b_frozen_dir=phase72b_frozen_dir,
        phase72b_confirmation_dir=phase72b_confirmation_dir,
    )
    protocol = dict(prepared["protocol"])
    frozen = Path(frozen_dir)
    selected = load_hashed_json(
        frozen / "phase72_temporal_neural_selected_models.json"
    )
    selected_sha256 = _read_sha256(
        frozen / "phase72_temporal_neural_selected_models.sha256"
    )
    if selected.get("status") != "phase72_temporal_neural_models_frozen":
        raise ValueError("Phase 72 temporal neural models are not frozen")
    if (
        selected.get("prepared_sha256") != prepared["manifest_sha256"]
        or selected.get("protocol_sha256")
        != prepared["manifest"]["protocol_sha256"]
        or selected.get("fit_implementation_id")
        != PHASE72_TEMPORAL_NEURAL_FIT_IMPLEMENTATION_ID
        or selected.get("confirmation_targets_opened") is not False
        or selected.get("phase72c_allowed") is not False
    ):
        raise ValueError("Phase 72 temporal neural frozen binding mismatch")
    expected_count = len(PHASE72_TEMPORAL_NEURAL_AXES) * 2 + 15
    if int(selected.get("bundle_count", -1)) != expected_count:
        raise ValueError("Phase 72 temporal neural frozen bundle count mismatch")

    bundles: dict[tuple[str, str, int | None], dict[str, object]] = {}
    bundle_hashes = {}
    for raw_record in selected["bundle_records"]:
        record = dict(raw_record)
        path = frozen / str(record["bundle_path"])
        actual_hash = _file_sha256(path)
        if actual_hash != str(record["bundle_sha256"]):
            raise ValueError(
                f"Phase 72 temporal neural frozen bundle hash mismatch: {record['key']}"
            )
        bundle = joblib.load(path)
        seed = bundle.get("control_seed", "")
        key = (
            str(bundle["axis_id"]),
            str(bundle["variant_id"]),
            None if seed == "" else int(seed),
        )
        if key in bundles:
            raise ValueError("Duplicate Phase 72 temporal neural frozen bundle")
        bundles[key] = bundle
        bundle_hashes[str(record["key"])] = actual_hash
    if len(bundles) != expected_count:
        raise ValueError("Phase 72 temporal neural loaded bundle count mismatch")

    # Confirmation labels are first loaded after all frozen identities pass.
    source = _load_confirmation_source(phase72b_prepared_dir)
    rows = list(source["feature_rows"])
    matrices = dict(source["matrices"])
    registry = dict(source["split_registry"])
    outcomes = dict(source["outcomes"])
    explicit_matrix = np.asarray(matrices["explicit_history"], dtype=np.float32)
    metrics_rows = []
    prediction_rows = []
    groups: dict[tuple[str, str, int | None], dict[str, object]] = {}
    for key, bundle in bundles.items():
        axis_id, variant_id, control_seed = key
        indexes = np.asarray(registry[axis_id]["test"], dtype=np.int64)
        y = _y_for_indexes(outcomes, indexes)
        explicit = explicit_matrix[indexes]
        if variant_id == PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT:
            probability = predict_phase72b_bundle(bundle, explicit)
        else:
            if variant_id == PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT:
                history, mask = _history_for_split(source, indexes)
            else:
                if control_seed is None:
                    raise ValueError(
                        "Phase 72 temporal neural confirmation control seed missing"
                    )
                history, mask, _ = _control_split_history(
                    source,
                    indexes,
                    axis_id=axis_id,
                    split_name="test",
                    variant_id=variant_id,
                    control_seed=control_seed,
                )
            probability = predict_phase72_temporal_neural_bundle(
                bundle, explicit, history, mask
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
            "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
            "axis_id": axis_id,
            "variant_id": variant_id,
            "control_seed": "" if control_seed is None else int(control_seed),
            "rows": len(indexes),
            "positives": int(y.sum()),
            "prevalence": round(float(y.mean()), 12),
            "model_family": str(bundle["model_family"]),
            "calibration_method": str(bundle["calibration_method"]),
            **metric,
        }
        metrics_rows.append(row)
        groups[key] = {
            "indexes": indexes,
            "outcome": y,
            "probability": probability,
            "metric": row,
        }
        for position, sample_index in enumerate(indexes):
            source_row = rows[int(sample_index)]
            prediction_rows.append(
                {
                    "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                    "sample_index": int(sample_index),
                    "axis_id": axis_id,
                    "variant_id": variant_id,
                    "control_seed": ""
                    if control_seed is None
                    else int(control_seed),
                    "outcome": int(y[position]),
                    "probability": round(float(probability[position]), 12),
                    "region_id": str(source_row["region_id"]),
                    "spatial_block_id": str(source_row["spatial_block_id"]),
                    "origin_year": int(source_row["origin_year"]),
                }
            )

    def group(
        axis_id: str, variant_id: str, control_seed: int | None = None
    ) -> dict[str, object]:
        try:
            return groups[(axis_id, variant_id, control_seed)]
        except KeyError as exc:
            raise ValueError(
                "Missing Phase 72 temporal neural confirmation group: "
                f"{axis_id}/{variant_id}/{control_seed}"
            ) from exc

    pooled_explicit = group(
        "pooled_temporal", PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT
    )
    pooled_primary = group(
        "pooled_temporal", PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT
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
    bootstrap_rows = [
        {
            "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
            "comparison_id": "temporal_neural_vs_explicit",
            "axis_id": "pooled_temporal",
            **pooled_bootstrap,
        }
    ]
    control_rows = []
    for variant_id, control_id in PHASE72_TEMPORAL_NEURAL_CONTROL_VARIANTS.items():
        selected_seed = int(
            selected["selected_control_seeds"][PHASE72_TEMPORAL_NEURAL_ENDPOINT][
                variant_id
            ]
        )
        control = group("pooled_temporal", variant_id, selected_seed)
        control_rows.append(
            {
                "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                "control_id": control_id,
                "variant_id": variant_id,
                "selected_seed": selected_seed,
                **_metric_delta(control["metric"], pooled_primary["metric"]),
            }
        )
    transfer_rows = []
    for axis_id in ("bishan_to_dongxing", "dongxing_to_bishan"):
        explicit = group(axis_id, PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT)
        primary = group(axis_id, PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT)
        transfer_rows.append(
            {
                "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                "axis_id": axis_id,
                "source_region": axis_id.split("_to_")[0],
                "target_region": axis_id.split("_to_")[1],
                "rows": len(primary["indexes"]),
                **_metric_delta(explicit["metric"], primary["metric"]),
            }
        )
    spatial_rows = []
    for axis_id in sorted(name for name in registry if name.startswith("spatial_")):
        explicit = group(axis_id, PHASE72_TEMPORAL_NEURAL_EXPLICIT_VARIANT)
        primary = group(axis_id, PHASE72_TEMPORAL_NEURAL_PRIMARY_VARIANT)
        region_id = axis_id.removeprefix("spatial_").split("_fold", 1)[0]
        spatial_rows.append(
            {
                "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
                "axis_id": axis_id,
                "region_id": region_id,
                "rows": len(primary["indexes"]),
                **_metric_delta(explicit["metric"], primary["metric"]),
            }
        )
    gate = build_phase72b_gate(
        pooled_delta=pooled_delta,
        pooled_bootstrap=pooled_bootstrap,
        control_rows=control_rows,
        transfer_rows=transfer_rows,
        spatial_rows=spatial_rows,
        leakage_ok=True,
        gates=protocol["gates"],
    )
    status = phase72_temporal_neural_status(str(gate["phase72b_status"]))
    return {
        "phase": "phase72_temporal_neural_exhaustion_screen",
        "phase72_temporal_neural_status": status,
        "endpoint": PHASE72_TEMPORAL_NEURAL_ENDPOINT,
        "endpoint_result": {
            **gate,
            "pooled_delta": pooled_delta,
            "pooled_bootstrap": pooled_bootstrap,
        },
        "decision_rule": protocol["decision_rule"],
        "stopping_rule": protocol["stopping_rule"],
        "phase72c_allowed": False,
        "confirmation_targets_opened": True,
        "metrics_rows": metrics_rows,
        "prediction_rows": prediction_rows,
        "bootstrap_rows": bootstrap_rows,
        "control_rows": control_rows,
        "transfer_rows": transfer_rows,
        "spatial_rows": spatial_rows,
        "counts": {
            "metric_rows": len(metrics_rows),
            "prediction_rows": len(prediction_rows),
            "bundle_count": len(bundles),
            "confirmation_rows": len(pooled_indexes),
        },
        "prepared_sha256": prepared["manifest_sha256"],
        "selected_models_sha256": selected_sha256,
        "bundle_hashes": bundle_hashes,
        "next_action": (
            "Record this one-year temporal-neural result in the Phase 72 "
            "exhaustion analysis. Do not enter Phase 72C, train a post hoc "
            "two-year neural extension, run planning, or revise the formal manuscript."
        ),
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
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
        raise ValueError(f"Phase 72 temporal neural CSV rows are empty: {path.name}")
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _confirmation_markdown(result: Mapping[str, object]) -> str:
    endpoint = result["endpoint_result"]
    delta = endpoint["pooled_delta"]
    bootstrap = endpoint["pooled_bootstrap"]
    checks = endpoint["checks"]
    lines = [
        "# Phase 72 Temporal Neural Exhaustion Screen",
        "",
        f"Status: `{result['phase72_temporal_neural_status']}`",
        "",
        "Phase 72C allowed: `false`",
        "",
        "## One-Year Confirmation",
        "",
        "| Endpoint | Gate status | AP delta | Brier delta | ECE delta |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| `{result['endpoint']}` | `{endpoint['phase72b_status']}` | "
        f"{delta['ap_delta']:.6f} | {delta['brier_delta']:.6f} | "
        f"{delta['ece_delta']:.6f} |",
        "",
        "```text",
        f"AP CI95: [{bootstrap['ap_delta_ci_low']:.6f}, {bootstrap['ap_delta_ci_high']:.6f}]",
        f"Brier CI95: [{bootstrap['brier_delta_ci_low']:.6f}, {bootstrap['brier_delta_ci_high']:.6f}]",
        f"practical: {checks['practical']}",
        f"statistical: {checks['statistical']}",
        f"controls: {checks['controls']}",
        f"transfer: {checks['transfer']}",
        f"spatial: {checks['spatial']}",
        "```",
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
    return "\n".join(lines)


def write_phase72_temporal_neural_confirmation_artifacts(
    result: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Phase 72 temporal neural confirmation output must be new or empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics": output / "phase72_temporal_neural_metrics.csv",
        "predictions": output / "phase72_temporal_neural_predictions.csv",
        "bootstrap": output / "phase72_temporal_neural_bootstrap_deltas.csv",
        "controls": output / "phase72_temporal_neural_control_comparison.csv",
        "transfers": output / "phase72_temporal_neural_transfer_summary.csv",
        "spatial": output / "phase72_temporal_neural_spatial_summary.csv",
        "result": output / "phase72_temporal_neural_screen.json",
        "markdown": output / "phase72_temporal_neural_screen.md",
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
        "status": "phase72_temporal_neural_confirmation_receipt",
        "phase72_temporal_neural_status": result[
            "phase72_temporal_neural_status"
        ],
        "prepared_sha256": result["prepared_sha256"],
        "selected_models_sha256": result["selected_models_sha256"],
        "bundle_count": result["counts"]["bundle_count"],
        "artifacts": [
            {"name": path.name, "sha256": _file_sha256(path)}
            for path in artifacts.values()
        ],
        "phase72c_allowed": False,
        "claim_boundary": PHASE72_TEMPORAL_NEURAL_CLAIM_BOUNDARY,
    }
    receipt_path, receipt_hash = write_hashed_json(
        output / "phase72_temporal_neural_confirmation_receipt.json", receipt
    )
    artifacts["receipt"] = receipt_path
    artifacts["receipt_sha256"] = receipt_hash
    return artifacts
