from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paper11_geofm.phase72_temporal_neural_screen as temporal_neural
from paper11_geofm.phase72_temporal_neural_screen import (
    Phase72GatedTemporalResidual,
    build_phase72_temporal_control_history,
    fit_history_standardizer,
    load_phase72_temporal_neural_protocol,
    phase72_temporal_neural_status,
    transform_history_with_standardizer,
    validate_prefix_history_mask,
)


PROTOCOL = (
    ROOT
    / "experiments"
    / "phase72_temporal_neural_screen"
    / "phase72_temporal_neural_protocol.json"
)


def test_temporal_neural_protocol_is_frozen_and_keeps_phase72c_closed():
    protocol = load_phase72_temporal_neural_protocol(PROTOCOL)

    assert protocol["endpoint"]["target"] == "conversion_1y"
    assert protocol["architecture"]["projection_channels"] == 16
    assert protocol["architecture"]["residual_head_bias"] is False
    assert protocol["decision_rule"] == (
        "conversion_1y_must_pass_all_frozen_gates"
    )
    assert protocol["phase72c_allowed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("architecture", "projection_channels"), 32),
        (("architecture", "residual_head_bias"), True),
        (("training", "max_epochs"), 100),
        (("controls", "seeds"), [72]),
        (("gates", "ap_vs_explicit"), 0.0),
        (("phase72c_allowed",), True),
    ],
)
def test_temporal_neural_protocol_rejects_mutation(path, value):
    mutated = copy.deepcopy(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="protocol"):
        temporal_neural.validate_phase72_temporal_neural_protocol(mutated)


def test_history_mask_must_be_a_nonempty_prefix():
    valid = np.asarray(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    np.testing.assert_array_equal(validate_prefix_history_mask(valid), valid)

    with pytest.raises(ValueError, match="prefix"):
        validate_prefix_history_mask(
            np.asarray([[True, False, True]], dtype=bool)
        )
    with pytest.raises(ValueError, match="non-empty"):
        validate_prefix_history_mask(np.zeros((1, 3), dtype=bool))


def test_history_standardizer_uses_only_valid_supplied_training_entries():
    history = np.asarray(
        [
            [[1.0, 10.0], [3.0, 30.0], [9999.0, 9999.0]],
            [[5.0, 50.0], [9999.0, 9999.0], [9999.0, 9999.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[True, True, False], [True, False, False]])

    standardizer = fit_history_standardizer(history, mask)
    transformed = transform_history_with_standardizer(
        history, mask, standardizer
    )

    np.testing.assert_allclose(standardizer["mean"], [3.0, 30.0])
    np.testing.assert_allclose(
        standardizer["scale"], np.std([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], axis=0)
    )
    assert np.all(transformed[~mask] == 0.0)
    np.testing.assert_allclose(transformed[mask].mean(axis=0), 0.0, atol=1e-7)


def test_gated_temporal_residual_is_bias_free_and_zero_branch_preserves_offset():
    model = Phase72GatedTemporalResidual(
        input_channels=4, projection_channels=3, maximum_history_steps=4
    )
    assert model.projection.bias is None
    assert model.content_gate.bias is None
    assert model.residual_head.bias is None
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    history = torch.randn(3, 4, 4)
    mask = torch.tensor(
        [[True, False, False, False], [True, True, False, False], [True] * 4]
    )
    explicit_logit = torch.tensor([-2.0, 0.0, 1.5])

    actual = model(history, mask, explicit_logit)

    torch.testing.assert_close(actual, explicit_logit)


def test_masked_padding_cannot_change_temporal_residual():
    torch.manual_seed(72)
    model = Phase72GatedTemporalResidual(
        input_channels=2, projection_channels=3, maximum_history_steps=3
    )
    first = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    second = first.clone()
    second[:, 1:, :] = 1.0e6
    mask = torch.tensor([[True, False, False]])
    offset = torch.tensor([0.25])

    torch.testing.assert_close(
        model(first, mask, offset), model(second, mask, offset)
    )


def test_temporal_controls_are_deterministic_shape_matched_and_partition_local():
    rng = np.random.default_rng(72)
    history = rng.normal(size=(8, 4, 4)).astype(np.float32)
    mask = np.asarray(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
        * 2,
        dtype=bool,
    )
    rows = [
        {
            "sample_index": index,
            "region_id": "bishan",
            "origin_year": 2019,
        }
        for index in range(8)
    ]
    partitions = ["train"] * 4 + ["validation"] * 4

    first = build_phase72_temporal_control_history(
        "spatial_shuffle",
        history,
        mask,
        rows,
        partition_ids=partitions,
        seed=72,
    )
    second = build_phase72_temporal_control_history(
        "spatial_shuffle",
        history,
        mask,
        rows,
        partition_ids=partitions,
        seed=72,
    )
    np.testing.assert_array_equal(first["history"], second["history"])
    np.testing.assert_array_equal(first["mask"], second["mask"])
    assert first["manifest"]["cross_partition_count"] == 0
    for target, source in enumerate(
        first["manifest"]["source_index_by_target"]
    ):
        assert partitions[target] == partitions[source]

    projected = build_phase72_temporal_control_history(
        "random_projection",
        history,
        mask,
        rows,
        partition_ids=partitions,
        seed=72,
    )
    assert projected["history"].shape == history.shape
    np.testing.assert_allclose(
        np.linalg.norm(projected["history"][mask], axis=1),
        np.linalg.norm(history[mask], axis=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_prepare_keeps_confirmation_targets_deferred(monkeypatch):
    observed = {}

    monkeypatch.setattr(temporal_neural, "_verify_source_hashes", lambda *args, **kwargs: None)

    def fake_load(_path, *, deferred_names):
        observed["deferred_names"] = set(deferred_names)
        return {
            "feature_rows": [{"sample_index": index} for index in range(6)],
            "matrices": {
                "embedding_history": np.zeros((6, 8, 64), dtype=np.float32),
                "history_mask": np.asarray(
                    [[True] + [False] * 7] * 6
                ),
            },
            "split_registry": {
                "pooled_temporal": {
                    "train": [0, 1, 2],
                    "validation": [3],
                    "test": [4, 5],
                }
            },
        }

    monkeypatch.setattr(
        temporal_neural, "load_verified_phase72b_prepared", fake_load
    )

    package = temporal_neural.prepare_phase72_temporal_neural_screen(
        protocol_path=PROTOCOL,
        phase72b_prepared_dir=ROOT / "unused",
        phase72b_frozen_dir=ROOT / "unused",
        phase72b_confirmation_dir=ROOT / "unused",
    )

    assert observed["deferred_names"] == {
        "phase72b_confirmation_targets.npz"
    }
    assert package["confirmation_targets_opened"] is False
    assert package["phase72c_allowed"] is False


@pytest.mark.parametrize(
    ("gate_status", "expected"),
    [
        (
            "geofm_information_not_supported",
            "temporal_neural_information_not_supported",
        ),
        ("geofm_information_mixed", "temporal_neural_information_mixed"),
        (
            "geofm_information_supported",
            "temporal_neural_information_supported",
        ),
    ],
)
def test_temporal_neural_status_preserves_gate_strength(gate_status, expected):
    assert phase72_temporal_neural_status(gate_status) == expected


def test_checkpoint_record_allows_explicit_bundle_without_neural_epoch(
    tmp_path,
):
    bundle = {
        "axis_id": "pooled_temporal",
        "variant_id": "explicit_history",
        "control_seed": "",
        "model_family": "logistic_regression",
        "calibration_method": "none",
        "validation_metrics": {
            "average_precision": 0.5,
            "brier": 0.25,
            "ece": 0.1,
        },
    }
    progress = {"entries": [], "validation_rows": []}

    _, record = temporal_neural._checkpoint_bundle(
        tmp_path,
        progress,
        bundle=bundle,
        validation_rows=[],
    )

    assert record["best_epoch"] == ""
    assert record["variant_id"] == "explicit_history"
