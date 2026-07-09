from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from paper11_geofm.phase63_set_policy_oracle_pretraining import _round_float
from paper11_geofm.tiled_inputs import TiledVariantInput


PHASE70_CLAIM_BOUNDARY = (
    "Phase 70 is a standardized set-policy rerun under the existing Bishan "
    "base-reward protocol. It standardizes model inputs with train-tile-fitted "
    "parameters while preserving original features for reward and oracle "
    "scoring. It does not alter rewards, enable B2/B3, validate suitability, "
    "prove PCA optimality, or justify formal submission-level claims."
)

PHASE70_STATUS_GEOFM = "standardization_improves_geofm_set_policy_route"
PHASE70_STATUS_ARCHITECTURE = "standardization_improves_architecture_not_geofm"
PHASE70_STATUS_NOT_SUFFICIENT = "standardization_not_sufficient"
PHASE70_STATUS_INCOMPLETE = "standardized_rerun_incomplete"


@dataclass(frozen=True)
class Phase70StandardizedTiledInput:
    tiled_input: TiledVariantInput
    model_matrix: np.ndarray
    reward_matrix: np.ndarray
    standardization: Mapping[str, object]


def fit_phase70_standardization(tiled_input: TiledVariantInput) -> dict[str, object]:
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Phase 70 standardization requires a non-empty 2D matrix")
    means = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    return {
        "variant_id": str(tiled_input.variant_id),
        "tile_id": str(tiled_input.tile_id),
        "feature_columns": list(tiled_input.feature_columns),
        "means": [_round_float(value) for value in means.tolist()],
        "scales": [_round_float(value) for value in safe_scales.tolist()],
        "claim_boundary": PHASE70_CLAIM_BOUNDARY,
    }


def apply_phase70_standardization(
    tiled_input: TiledVariantInput,
    params: Mapping[str, object],
) -> Phase70StandardizedTiledInput:
    feature_columns = tuple(str(value) for value in params.get("feature_columns", []))
    if feature_columns != tuple(tiled_input.feature_columns):
        raise ValueError("Phase 70 standardization feature columns do not match input")
    matrix = np.asarray(tiled_input.state_matrix, dtype=np.float32)
    means = np.asarray(params.get("means", []), dtype=np.float32)
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    if means.shape[0] != matrix.shape[1] or scales.shape[0] != matrix.shape[1]:
        raise ValueError("Phase 70 standardization parameter length does not match input")
    safe_scales = np.where(np.isfinite(scales) & (np.abs(scales) >= 1.0e-8), scales, 1.0)
    model_matrix = (matrix - means) / safe_scales
    model_matrix = np.nan_to_num(model_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    return Phase70StandardizedTiledInput(
        tiled_input=tiled_input,
        model_matrix=model_matrix,
        reward_matrix=matrix.astype(np.float32, copy=True),
        standardization=dict(params),
    )
