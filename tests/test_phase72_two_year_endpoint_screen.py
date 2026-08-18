from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper11_geofm.phase72_two_year_endpoint_screen import (
    PHASE72_TWO_YEAR_ENDPOINTS,
    _overall_status,
    fit_freeze_phase72_two_year_models,
    load_phase72_two_year_protocol,
    prepare_phase72_two_year_endpoint_screen,
    validate_phase72_two_year_protocol,
    write_phase72_two_year_confirmation_artifacts,
    write_phase72_two_year_prepared_artifacts,
)
from paper11_geofm.phase72b_terrain import _file_sha256


PROTOCOL = (
    ROOT
    / "experiments"
    / "phase72_two_year_endpoint_screen"
    / "phase72_two_year_protocol.json"
)


def test_phase72_two_year_protocol_is_frozen_and_keeps_phase72c_closed():
    protocol = load_phase72_two_year_protocol(PROTOCOL)

    assert protocol["endpoints"] == PHASE72_TWO_YEAR_ENDPOINTS
    assert protocol["years"] == {
        "train": [2017, 2018, 2019, 2020],
        "validation": [2021],
        "test": [2022],
    }
    assert protocol["decision_rule"] == (
        "both_endpoints_must_pass_all_frozen_gates"
    )
    assert protocol["model_selection"]["strategy"] == (
        "reuse_phase72b_frozen_candidate_configs"
    )
    assert protocol["phase72c_allowed"] is False


def test_phase72_two_year_protocol_rejects_post_hoc_year_change():
    protocol = load_phase72_two_year_protocol(PROTOCOL)
    changed = copy.deepcopy(protocol)
    changed["years"]["test"] = [2021]

    with pytest.raises(ValueError, match="years mismatch"):
        validate_phase72_two_year_protocol(changed)


def _source_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    import paper11_geofm.phase72_two_year_endpoint_screen as module

    package_dir = tmp_path / "phase72a"
    package_dir.mkdir()
    package_path = package_dir / "phase72a_temporal_label_package.json"
    package_path.write_text(
        json.dumps({"phase72a_status": "phase72a_label_inputs_ready"}),
        encoding="utf-8",
    )
    rows = []
    for region_index, region_id in enumerate(("bishan", "dongxing")):
        for year in range(2017, 2024):
            sample_index = len(rows)
            has_2y = year <= 2022
            rows.append(
                {
                    "sample_index": sample_index,
                    "region_id": region_id,
                    "unit_id": f"u{sample_index}",
                    "row": region_index * 8,
                    "col": year - 2017,
                    "spatial_block_id": f"{region_id}_br{region_index:03d}_bc000",
                    "origin_year": year,
                    "y_2y": (sample_index % 2) if has_2y else "",
                    "y_continuous_2y": ((sample_index + 1) % 2) if has_2y else "",
                }
            )
    pd.DataFrame(rows).to_csv(
        package_dir / "phase72a_temporal_sample_index.csv", index=False
    )
    np.savez_compressed(
        package_dir / "phase72a_temporal_samples.npz",
        placeholder=np.arange(len(rows)),
    )
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
        for row in rows
    ]
    fake_verified = {
        "protocol_hash": "1" * 64,
        "manifest_sha256": "2" * 64,
        "feature_rows": feature_rows,
        "matrices": {
            "explicit_history": np.ones((len(rows), 2), np.float32),
            "geofm_temporal_full": np.ones((len(rows), 3), np.float32),
            "embedding_history": np.ones((len(rows), 2, 1), np.float32),
            "history_mask": np.ones((len(rows), 2), bool),
        },
    }
    monkeypatch.setattr(
        module, "load_verified_phase72b_prepared", lambda _: fake_verified
    )
    monkeypatch.setattr(
        module,
        "build_phase72b_split_registry",
        lambda sample_rows, **_: {
            "pooled_temporal": {
                "train": [
                    index
                    for index, row in enumerate(sample_rows)
                    if int(row["origin_year"]) <= 2020
                ],
                "validation": [
                    index
                    for index, row in enumerate(sample_rows)
                    if int(row["origin_year"]) == 2021
                ],
                "test": [
                    index
                    for index, row in enumerate(sample_rows)
                    if int(row["origin_year"]) == 2022
                ],
            }
        },
    )
    monkeypatch.setattr(
        module,
        "audit_phase72b_splits",
        lambda *_, **__: {"status": "leakage_audit_passed", "errors": []},
    )
    protocol = load_phase72_two_year_protocol(PROTOCOL)
    protocol["source_bindings"] = {
        "phase72a_package_sha256": _file_sha256(package_path),
        "phase72a_sample_index_sha256": _file_sha256(
            package_dir / "phase72a_temporal_sample_index.csv"
        ),
        "phase72a_temporal_samples_sha256": _file_sha256(
            package_dir / "phase72a_temporal_samples.npz"
        ),
        "phase72b_frozen_protocol_sha256": "1" * 64,
        "phase72b_prepared_artifacts_sha256": "2" * 64,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return package_dir, protocol_path, fake_verified


def test_phase72_two_year_prepare_derives_locked_targets(tmp_path, monkeypatch):
    package_dir, protocol_path, _ = _source_fixture(tmp_path, monkeypatch)

    package = prepare_phase72_two_year_endpoint_screen(
        protocol_path=protocol_path,
        phase72a_package_dir=package_dir,
        phase72b_prepared_dir=tmp_path / "source-prepared",
    )

    assert len(package["feature_rows"]) == 12
    assert all(
        int(row["origin_year"]) <= 2022 for row in package["feature_rows"]
    )
    assert package["development_targets"]["sample_index"].shape == (10,)
    assert package["confirmation_targets"]["sample_index"].shape == (2,)
    assert np.array_equal(
        package["development_targets"]["conversion_2y"],
        1
        - np.asarray(
            [index % 2 for index in range(5)]
            + [index % 2 for index in range(7, 12)],
            dtype=np.int8,
        ),
    )
    assert set(package["split_registries"]) == set(PHASE72_TWO_YEAR_ENDPOINTS)

    paths = write_phase72_two_year_prepared_artifacts(
        package, tmp_path / "prepared"
    )
    assert paths["manifest"].is_file()
    assert paths["manifest_sha256"].is_file()


def test_phase72_two_year_overall_requires_both_endpoints_to_pass():
    supported = {"phase72b_status": "geofm_information_supported"}
    negative = {"phase72b_status": "geofm_information_not_supported"}

    assert _overall_status(
        {name: supported for name in PHASE72_TWO_YEAR_ENDPOINTS}
    ) == "two_year_geofm_information_supported"
    assert _overall_status(
        {
            "conversion_2y": supported,
            "noncontinuous_persistence_2y": negative,
        }
    ) == "two_year_geofm_information_mixed"
    assert _overall_status(
        {name: negative for name in PHASE72_TWO_YEAR_ENDPOINTS}
    ) == "two_year_geofm_information_not_supported"


def test_phase72_two_year_fit_orchestrates_all_frozen_axes(
    tmp_path, monkeypatch
):
    import paper11_geofm.phase72_two_year_endpoint_screen as module

    protocol = load_phase72_two_year_protocol(PROTOCOL)
    base_axis = {
        "train": [0, 1],
        "validation": [2, 3],
        "test": [],
    }
    registry = {
        "pooled_temporal": copy.deepcopy(base_axis),
        "bishan_to_dongxing": copy.deepcopy(base_axis),
        "dongxing_to_bishan": copy.deepcopy(base_axis),
        **{
            f"spatial_{region}_fold{fold}": copy.deepcopy(base_axis)
            for region in ("bishan", "dongxing")
            for fold in range(5)
        },
    }
    prepared = {
        "manifest_sha256": "3" * 64,
        "protocol": protocol,
        "feature_rows": [
            {
                "sample_index": index,
                "region_id": "bishan" if index < 2 else "dongxing",
                "origin_year": 2017 + index,
            }
            for index in range(4)
        ],
        "matrices": {
            "explicit_history": np.ones((4, 2), np.float32),
            "geofm_temporal_full": np.ones((4, 3), np.float32),
            "embedding_history": np.ones((4, 2, 1), np.float32),
            "history_mask": np.ones((4, 2), bool),
        },
        "split_registries": {
            endpoint: copy.deepcopy(registry)
            for endpoint in PHASE72_TWO_YEAR_ENDPOINTS
        },
        "development_targets": {
            "sample_index": np.arange(4, dtype=np.int32),
            "origin_year": np.arange(2017, 2021, dtype=np.int16),
            "conversion_2y": np.asarray([0, 1, 0, 1], np.int8),
            "noncontinuous_persistence_2y": np.asarray(
                [1, 0, 1, 0], np.int8
            ),
        },
    }
    monkeypatch.setattr(
        module,
        "load_verified_phase72_two_year_prepared",
        lambda *_, **__: prepared,
    )
    monkeypatch.setattr(
        module,
        "_control_matrices",
        lambda *_, train_indexes, validation_indexes, **__: (
            np.ones((len(train_indexes), 5), np.float32),
            np.ones((len(validation_indexes), 5), np.float32),
        ),
    )
    reference_configs = {
        (axis_id, variant_id, seed): {
            "model_family": "logistic",
            "C": 1.0,
            "class_weight": None,
        }
        for axis_id in registry
        for variant_id in (
            "explicit_history",
            "explicit_plus_geofm_temporal_full",
            "explicit_plus_temporal_order_shuffle",
            "explicit_plus_spatial_shuffle",
            "explicit_plus_random_projection",
        )
        for seed in (
            (None,)
            if variant_id
            in {
                "explicit_history",
                "explicit_plus_geofm_temporal_full",
            }
            else (72, 73, 74, 75, 76)
        )
    }
    monkeypatch.setattr(
        module,
        "_load_phase72b_reference_configs",
        lambda *_, **__: reference_configs,
    )

    def fake_fit(
        train_x,
        train_y,
        validation_x,
        validation_y,
        *,
        variant_id,
        axis_id,
        candidate_config=None,
        **kwargs,
    ):
        config = candidate_config or {
            "model_family": "logistic",
            "C": 1.0,
            "class_weight": None,
        }
        bundle = {
            "axis_id": axis_id,
            "variant_id": variant_id,
            "estimator_params": config,
            "feature_count": train_x.shape[1],
            "validation_metrics": {
                "average_precision": 0.6,
                "brier": 0.2,
                "ece": 0.1,
            },
        }
        return bundle, [{"axis_id": axis_id, "variant_id": variant_id}]

    monkeypatch.setattr(module, "fit_fixed_phase72b_model", fake_fit)

    selected, paths = fit_freeze_phase72_two_year_models(
        prepared_dir=tmp_path / "prepared",
        phase72a_package_dir=tmp_path / "phase72a",
        phase72b_prepared_dir=tmp_path / "phase72b",
        phase72b_reference_frozen_dir=tmp_path / "phase72b-frozen",
        output_dir=tmp_path / "frozen",
    )

    assert selected["bundle_count"] == 142
    assert len(selected["bundle_records"]) == 142
    assert paths["selected_models"].is_file()


def test_phase72_two_year_confirmation_writer_maps_all_row_artifacts(
    tmp_path,
):
    endpoint_results = {
        endpoint: {
            "phase72b_status": "geofm_information_not_supported",
            "pooled_delta": {
                "ap_delta": 0.0,
                "brier_delta": 0.0,
                "ece_delta": 0.0,
            },
        }
        for endpoint in PHASE72_TWO_YEAR_ENDPOINTS
    }
    result = {
        "phase": "phase72_two_year_endpoint_screen",
        "phase72_two_year_status": "two_year_geofm_information_not_supported",
        "endpoint_results": endpoint_results,
        "metrics_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "prediction_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "bootstrap_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "control_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "transfer_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "spatial_rows": [{"endpoint": "conversion_2y", "value": 1}],
        "prepared_sha256": "1" * 64,
        "selected_models_sha256": "2" * 64,
        "next_action": "Keep Phase 72C closed.",
        "claim_boundary": "Test boundary.",
    }

    paths = write_phase72_two_year_confirmation_artifacts(
        result, tmp_path / "confirmation"
    )

    assert set(paths) == {
        "metrics",
        "predictions",
        "bootstrap",
        "controls",
        "transfers",
        "spatial",
        "result",
        "markdown",
        "receipt",
        "receipt_sha256",
    }
    assert all(path.is_file() for path in paths.values())
