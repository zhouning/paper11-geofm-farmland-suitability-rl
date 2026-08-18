from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper11_geofm.phase72_explicit_residual_screen import (
    PHASE72_RESIDUAL_ENDPOINTS,
    fit_offset_logistic_residual,
    load_phase72_explicit_residual_protocol,
    phase72_explicit_residual_overall_status,
    predict_offset_logistic_residual,
    spatial_group_cross_fit_assignments,
    validate_phase72_explicit_residual_protocol,
)
from paper11_geofm.phase72b_models import _best_f1_threshold


PROTOCOL = (
    ROOT
    / "experiments"
    / "phase72_explicit_residual_screen"
    / "phase72_explicit_residual_protocol.json"
)
FREEZE_RECEIPT = (
    ROOT
    / "experiments"
    / "phase72_explicit_residual_screen"
    / "phase72_explicit_residual_freeze_receipt.json"
)


def test_phase72_explicit_residual_protocol_is_frozen_and_keeps_phase72c_closed():
    protocol = load_phase72_explicit_residual_protocol(PROTOCOL)

    assert tuple(protocol["endpoints"]) == PHASE72_RESIDUAL_ENDPOINTS
    assert protocol["residual"]["residual_intercept"] is False
    assert protocol["residual"]["explicit_training_prediction"] == (
        "spatial_block_group_cross_fit"
    )
    assert protocol["decision_rule"] == (
        "all_three_endpoints_must_pass_all_frozen_gates"
    )
    assert protocol["phase72c_allowed"] is False


def test_phase72_explicit_residual_freeze_receipt_precedes_confirmation():
    receipt = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))

    assert receipt["status"] == (
        "phase72_explicit_residual_confirmation_frozen"
    )
    assert receipt["implementation_commit"] == (
        "e3b2144ae906349f6a6d520200b17e16359c64c6"
    )
    assert receipt["prepared_sha256"] == (
        "184ade17e02aa86aac2cd3ccb372d1d245be5bd149b734a88fb3df1a9235f396"
    )
    assert receipt["selected_models_sha256"] == (
        "d49d4e0c57fcf75b668d3a30c1177e2b2600ca02195697ad991a4afbf4762628"
    )
    assert receipt["bundle_count"] == 123
    assert receipt["residual_bundle_count"] == 84
    assert receipt["bundle_hash_mismatches"] == 0
    assert receipt["invalid_cross_fit_audits"] == 0
    assert receipt["confirmation_targets_opened"] is False
    assert receipt["phase72c_allowed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("residual", "residual_intercept"), True),
        (("residual", "cross_fit_folds"), 4),
        (("gates", "ap_vs_explicit"), 0.0),
        (("controls", "seeds"), [72]),
        (("phase72c_allowed",), True),
    ],
)
def test_phase72_explicit_residual_protocol_rejects_mutation(path, value):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="protocol"):
        validate_phase72_explicit_residual_protocol(mutated)


def test_spatial_group_cross_fit_assignments_are_group_exclusive_and_stable():
    groups = [f"block_{index // 3}" for index in range(30)]
    first = spatial_group_cross_fit_assignments(groups, folds=5)
    second = spatial_group_cross_fit_assignments(groups, folds=5)

    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == set(range(5))
    for group in set(groups):
        positions = [i for i, value in enumerate(groups) if value == group]
        assert len(set(first[positions].tolist())) == 1


def test_offset_residual_zero_signal_preserves_fixed_explicit_probability():
    explicit_probability = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=float)
    residual_features = np.zeros((4, 2), dtype=float)
    bundle = {
        "feature_mean": np.zeros(2),
        "feature_scale": np.ones(2),
        "coefficient": np.zeros(2),
    }

    actual = predict_offset_logistic_residual(
        bundle, explicit_probability, residual_features
    )

    np.testing.assert_allclose(actual, explicit_probability, atol=1e-12)


def test_offset_residual_can_recover_signal_beyond_explicit_baseline():
    rng = np.random.default_rng(72)
    residual_features = rng.normal(size=(800, 2))
    explicit_probability = np.full(800, 0.5)
    outcome = (residual_features[:, 0] > 0).astype(np.int8)

    bundle = fit_offset_logistic_residual(
        explicit_probability,
        residual_features,
        outcome,
        l2_strength=0.001,
        class_weight="none",
        max_iter=500,
        tolerance=1e-7,
    )
    probability = predict_offset_logistic_residual(
        bundle, explicit_probability, residual_features
    )

    assert probability[outcome == 1].mean() > 0.9
    assert probability[outcome == 0].mean() < 0.1
    assert bundle["residual_intercept"] is False


def test_offset_residual_accepts_a_finite_warm_start():
    rng = np.random.default_rng(73)
    features = rng.normal(size=(300, 3))
    probability = np.full(300, 0.4)
    outcome = (features[:, 0] + features[:, 1] > 0).astype(np.int8)
    first = fit_offset_logistic_residual(
        probability,
        features,
        outcome,
        l2_strength=0.01,
        class_weight="none",
        max_iter=500,
        tolerance=1e-7,
    )

    second = fit_offset_logistic_residual(
        probability,
        features,
        outcome,
        l2_strength=0.001,
        class_weight="none",
        max_iter=500,
        tolerance=1e-7,
        initial_coefficient=first["coefficient"],
    )

    assert second["optimizer_iterations"] < 500
    assert np.isfinite(second["coefficient"]).all()


def test_phase72_explicit_residual_overall_requires_all_endpoints():
    supported = {
        endpoint: {"phase72b_status": "geofm_information_supported"}
        for endpoint in PHASE72_RESIDUAL_ENDPOINTS
    }
    assert phase72_explicit_residual_overall_status(supported) == (
        "explicit_residual_information_supported"
    )

    supported["conversion_2y"] = {
        "phase72b_status": "geofm_information_not_supported"
    }
    assert phase72_explicit_residual_overall_status(supported) == (
        "explicit_residual_information_mixed"
    )

    negative = {
        endpoint: {"phase72b_status": "geofm_information_not_supported"}
        for endpoint in PHASE72_RESIDUAL_ENDPOINTS
    }
    assert phase72_explicit_residual_overall_status(negative) == (
        "explicit_residual_information_not_supported"
    )


def test_fast_f1_threshold_matches_exhaustive_definition():
    rng = np.random.default_rng(72)
    outcome = rng.integers(0, 2, size=200, dtype=np.int8)
    probability = np.round(rng.uniform(size=200), 2)
    candidates = np.unique(np.concatenate([probability, [0.5]]))
    expected = max(
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
    )[2]

    assert _best_f1_threshold(outcome, probability) == expected
