from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from .phase72b_geofm_features import build_phase72b_control_features
from .phase72b_metrics import (
    build_phase72b_gate,
    paired_block_bootstrap,
    phase72b_metrics,
)
from .phase72b_models import (
    fit_fixed_phase72b_model,
    fit_select_phase72b_model,
    predict_phase72b_bundle,
)
from .phase72b_prepared import load_verified_phase72b_prepared
from .phase72b_protocol import (
    PHASE72B_BOOTSTRAP,
    PHASE72B_BUDGETS,
    PHASE72B_CALIBRATION,
    PHASE72B_CONTROLS,
    PHASE72B_GATES,
    PHASE72B_MODELS,
    PHASE72B_SPATIAL,
    canonical_json_sha256,
    load_hashed_json,
    write_hashed_json,
)
from .phase72b_splits import (
    audit_phase72b_splits,
    build_phase72b_split_registry,
)
from .phase72b_terrain import _file_sha256


PHASE72_TWO_YEAR_CLAIM_BOUNDARY = (
    "This Phase 72 exhaustion experiment tests two-year product-label "
    "prediction with the frozen Phase 72B controls and gates. It does not "
    "enter Phase 72C, establish agronomic suitability, run constrained "
    "planning, revise the formal manuscript, or override the official "
    "one-year Phase 72B result."
)

PHASE72_TWO_YEAR_ENDPOINTS = {
    "conversion_2y": "1-y_2y",
    "noncontinuous_persistence_2y": "1-y_continuous_2y",
}
PHASE72_TWO_YEAR_YEARS = {
    "train": [2017, 2018, 2019, 2020],
    "validation": [2021],
    "test": [2022],
}
PHASE72_TWO_YEAR_VARIANTS = [
    "explicit_history",
    "explicit_plus_geofm_temporal_full",
    "explicit_plus_temporal_order_shuffle",
    "explicit_plus_spatial_shuffle",
    "explicit_plus_random_projection",
]
PHASE72_TWO_YEAR_DECISION_RULE = (
    "both_endpoints_must_pass_all_frozen_gates"
)
_BASE_VARIANTS = (
    "explicit_history",
    "explicit_plus_geofm_temporal_full",
)
_CONTROL_VARIANTS = {
    "explicit_plus_temporal_order_shuffle": "temporal_order_shuffle",
    "explicit_plus_spatial_shuffle": "spatial_shuffle",
    "explicit_plus_random_projection": "random_projection",
}
_SEARCH_AXES = (
    "pooled_temporal",
    "bishan_to_dongxing",
    "dongxing_to_bishan",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PREPARED_FILES = (
    "phase72_two_year_feature_rows.csv",
    "phase72_two_year_split_registry.json",
    "phase72_two_year_development_targets.npz",
    "phase72_two_year_confirmation_targets.npz",
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
            str(name): _array_sha256(np.asarray(arrays[name]))
            for name in sorted(arrays)
        }
    )


def _require_equal(actual: object, expected: object, label: str) -> None:
    if canonical_json_sha256({"value": actual}) != canonical_json_sha256(
        {"value": expected}
    ):
        raise ValueError(f"Phase 72 two-year frozen {label} mismatch")


def validate_phase72_two_year_protocol(
    payload: Mapping[str, object],
) -> dict[str, object]:
    protocol = dict(payload)
    required = {
        "phase",
        "seed",
        "source_bindings",
        "endpoints",
        "decision_rule",
        "years",
        "controls",
        "spatial",
        "bootstrap",
        "models",
        "calibration",
        "budgets",
        "variants",
        "gates",
        "phase72c_allowed",
    }
    if set(protocol) != required:
        raise ValueError("Phase 72 two-year protocol fields mismatch")
    if protocol.get("phase") != "phase72_two_year_endpoint_screen":
        raise ValueError("Invalid Phase 72 two-year protocol phase")
    _require_equal(protocol.get("seed"), 72, "seed")
    _require_equal(protocol.get("endpoints"), PHASE72_TWO_YEAR_ENDPOINTS, "endpoints")
    _require_equal(
        protocol.get("decision_rule"),
        PHASE72_TWO_YEAR_DECISION_RULE,
        "decision rule",
    )
    _require_equal(protocol.get("years"), PHASE72_TWO_YEAR_YEARS, "years")
    _require_equal(protocol.get("controls"), PHASE72B_CONTROLS, "controls")
    _require_equal(protocol.get("spatial"), PHASE72B_SPATIAL, "spatial")
    _require_equal(protocol.get("bootstrap"), PHASE72B_BOOTSTRAP, "bootstrap")
    _require_equal(protocol.get("models"), PHASE72B_MODELS, "models")
    _require_equal(protocol.get("calibration"), PHASE72B_CALIBRATION, "calibration")
    _require_equal(protocol.get("budgets"), PHASE72B_BUDGETS, "budgets")
    _require_equal(protocol.get("variants"), PHASE72_TWO_YEAR_VARIANTS, "variants")
    _require_equal(protocol.get("gates"), PHASE72B_GATES, "gates")
    if protocol.get("phase72c_allowed") is not False:
        raise ValueError("Phase 72C must remain forbidden")
    bindings = protocol.get("source_bindings")
    expected_bindings = {
        "phase72a_package_sha256",
        "phase72a_sample_index_sha256",
        "phase72a_temporal_samples_sha256",
        "phase72b_frozen_protocol_sha256",
        "phase72b_prepared_artifacts_sha256",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise ValueError("Phase 72 two-year source bindings mismatch")
    if any(
        _HEX64.fullmatch(str(value).lower()) is None
        for value in bindings.values()
    ):
        raise ValueError("Phase 72 two-year source binding is not SHA256")
    return protocol


def load_phase72_two_year_protocol(path: Path | str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Phase 72 two-year protocol must be a JSON object")
    return validate_phase72_two_year_protocol(payload)


def _verify_source_bindings(
    protocol: Mapping[str, object],
    *,
    phase72a_package_dir: Path,
    verified_source: Mapping[str, object],
) -> None:
    bindings = dict(protocol["source_bindings"])
    actual = {
        "phase72a_package_sha256": _file_sha256(
            phase72a_package_dir / "phase72a_temporal_label_package.json"
        ),
        "phase72a_sample_index_sha256": _file_sha256(
            phase72a_package_dir / "phase72a_temporal_sample_index.csv"
        ),
        "phase72a_temporal_samples_sha256": _file_sha256(
            phase72a_package_dir / "phase72a_temporal_samples.npz"
        ),
        "phase72b_frozen_protocol_sha256": str(
            verified_source["protocol_hash"]
        ).lower(),
        "phase72b_prepared_artifacts_sha256": str(
            verified_source["manifest_sha256"]
        ).lower(),
    }
    mismatches = [
        name
        for name, value in actual.items()
        if value.lower() != str(bindings[name]).lower()
    ]
    if mismatches:
        raise ValueError(
            "Phase 72 two-year source binding mismatch: "
            + ", ".join(mismatches)
        )


def prepare_phase72_two_year_endpoint_screen(
    *,
    protocol_path: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
) -> dict[str, object]:
    protocol = load_phase72_two_year_protocol(protocol_path)
    package_dir = Path(phase72a_package_dir)
    verified = load_verified_phase72b_prepared(phase72b_prepared_dir)
    _verify_source_bindings(
        protocol,
        phase72a_package_dir=package_dir,
        verified_source=verified,
    )
    package = json.loads(
        (package_dir / "phase72a_temporal_label_package.json").read_text(
            encoding="utf-8"
        )
    )
    if package.get("phase72a_status") != "phase72a_label_inputs_ready":
        raise ValueError("Phase 72A package is not ready")
    source_rows = pd.read_csv(
        package_dir / "phase72a_temporal_sample_index.csv",
        keep_default_na=False,
    ).to_dict(orient="records")
    feature_rows = list(verified["feature_rows"])
    if len(source_rows) != len(feature_rows):
        raise ValueError("Phase 72 two-year source row count mismatch")
    identity_fields = (
        "sample_index",
        "region_id",
        "unit_id",
        "row",
        "col",
        "spatial_block_id",
        "origin_year",
    )
    for index, (label_row, feature_row) in enumerate(
        zip(source_rows, feature_rows)
    ):
        if any(
            str(label_row.get(field, "")) != str(feature_row.get(field, ""))
            for field in identity_fields
        ):
            raise ValueError(
                f"Phase 72 two-year source row identity mismatch at {index}"
            )

    eligible_rows = []
    outcome_values = {name: [] for name in PHASE72_TWO_YEAR_ENDPOINTS}
    for source_row in source_rows:
        if str(source_row.get("y_2y", "")) == "" or str(
            source_row.get("y_continuous_2y", "")
        ) == "":
            continue
        local_index = len(eligible_rows)
        origin_year = int(source_row["origin_year"])
        if origin_year not in range(2017, 2023):
            raise ValueError(
                f"Unexpected eligible two-year origin: {origin_year}"
            )
        eligible_rows.append(
            {
                "sample_index": local_index,
                "source_sample_index": int(source_row["sample_index"]),
                "region_id": str(source_row["region_id"]),
                "unit_id": str(source_row["unit_id"]),
                "row": int(source_row["row"]),
                "col": int(source_row["col"]),
                "spatial_block_id": str(source_row["spatial_block_id"]),
                "origin_year": origin_year,
            }
        )
        outcome_values["conversion_2y"].append(
            1 - int(source_row["y_2y"])
        )
        outcome_values["noncontinuous_persistence_2y"].append(
            1 - int(source_row["y_continuous_2y"])
        )
    if not eligible_rows:
        raise ValueError("Phase 72 two-year endpoint rows are empty")

    years = dict(protocol["years"])
    validation_year = int(years["validation"][0])
    test_year = int(years["test"][0])
    origins = np.asarray(
        [int(row["origin_year"]) for row in eligible_rows], dtype=np.int16
    )
    indexes = np.arange(len(eligible_rows), dtype=np.int32)
    development_mask = origins <= validation_year
    confirmation_mask = origins == test_year
    development_targets = {
        "sample_index": indexes[development_mask],
        "origin_year": origins[development_mask],
    }
    confirmation_targets = {
        "sample_index": indexes[confirmation_mask],
        "origin_year": origins[confirmation_mask],
    }
    split_registries = {}
    leakage_audits = {}
    endpoint_counts = {}
    for endpoint, values in outcome_values.items():
        outcome = np.asarray(values, dtype=np.int8)
        if not np.isin(outcome, (0, 1)).all():
            raise ValueError(f"Phase 72 two-year endpoint is not binary: {endpoint}")
        development_targets[endpoint] = outcome[development_mask]
        confirmation_targets[endpoint] = outcome[confirmation_mask]
        split_rows = [
            {**row, "conversion_1y": int(outcome[position])}
            for position, row in enumerate(eligible_rows)
        ]
        registry = build_phase72b_split_registry(
            split_rows,
            train_years=years["train"],
            validation_year=validation_year,
            test_year=test_year,
            folds=int(protocol["spatial"]["folds"]),
            buffer_rings=int(protocol["spatial"]["buffer_rings"]),
        )
        audit = audit_phase72b_splits(
            split_rows,
            registry,
            train_years=years["train"],
            validation_year=validation_year,
            test_year=test_year,
            spatial_folds=int(protocol["spatial"]["folds"]),
            control_partition_local=True,
            reuse_phase8_d4_tables=False,
        )
        if audit["status"] != "leakage_audit_passed":
            raise ValueError(
                f"Phase 72 two-year leakage audit failed for {endpoint}: "
                + " | ".join(audit["errors"])
            )
        split_registries[endpoint] = registry
        leakage_audits[endpoint] = audit
        endpoint_counts[endpoint] = {
            "eligible_rows": len(outcome),
            "positive_rows": int(outcome.sum()),
            "development_rows": int(development_mask.sum()),
            "development_positive_rows": int(outcome[development_mask].sum()),
            "confirmation_rows": int(confirmation_mask.sum()),
            "confirmation_positive_rows": int(outcome[confirmation_mask].sum()),
        }
    return {
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "source_bindings": dict(protocol["source_bindings"]),
        "feature_rows": eligible_rows,
        "development_targets": development_targets,
        "confirmation_targets": confirmation_targets,
        "split_registries": split_registries,
        "leakage_audits": leakage_audits,
        "endpoint_counts": endpoint_counts,
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_phase72_two_year_prepared_artifacts(
    package: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_rows": output / _PREPARED_FILES[0],
        "split_registry": output / _PREPARED_FILES[1],
        "development_targets": output / _PREPARED_FILES[2],
        "confirmation_targets": output / _PREPARED_FILES[3],
        "manifest": output / "phase72_two_year_prepared.json",
    }
    _write_csv(paths["feature_rows"], package["feature_rows"])
    paths["split_registry"].write_text(
        json.dumps(package["split_registries"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        paths["development_targets"], **package["development_targets"]
    )
    np.savez_compressed(
        paths["confirmation_targets"], **package["confirmation_targets"]
    )
    manifest = {
        "status": "phase72_two_year_inputs_prepared",
        "protocol": package["protocol"],
        "protocol_sha256": package["protocol_sha256"],
        "source_bindings": package["source_bindings"],
        "endpoint_counts": package["endpoint_counts"],
        "leakage_audits": package["leakage_audits"],
        "artifact_sha256": {
            name: _file_sha256(output / name) for name in _PREPARED_FILES
        },
        "development_targets_sha256": _target_arrays_sha256(
            package["development_targets"]
        ),
        "confirmation_targets_sha256": _target_arrays_sha256(
            package["confirmation_targets"]
        ),
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
    }
    manifest_path, hash_path = write_hashed_json(paths["manifest"], manifest)
    paths["manifest"] = manifest_path
    paths["manifest_sha256"] = hash_path
    return paths


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def load_verified_phase72_two_year_prepared(
    prepared_dir: Path | str,
    *,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    include_confirmation: bool = False,
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    manifest = load_hashed_json(prepared / "phase72_two_year_prepared.json")
    if manifest.get("status") != "phase72_two_year_inputs_prepared":
        raise ValueError("Phase 72 two-year prepared status mismatch")
    protocol = validate_phase72_two_year_protocol(manifest["protocol"])
    if canonical_json_sha256(protocol) != str(manifest["protocol_sha256"]):
        raise ValueError("Phase 72 two-year protocol semantic hash mismatch")
    verified_source = load_verified_phase72b_prepared(phase72b_prepared_dir)
    _verify_source_bindings(
        protocol,
        phase72a_package_dir=Path(phase72a_package_dir),
        verified_source=verified_source,
    )
    deferred = (
        set()
        if include_confirmation
        else {"phase72_two_year_confirmation_targets.npz"}
    )
    for name in _PREPARED_FILES:
        if name in deferred:
            continue
        if _file_sha256(prepared / name) != str(
            manifest["artifact_sha256"][name]
        ):
            raise ValueError(f"Phase 72 two-year artifact hash mismatch: {name}")
    rows = pd.read_csv(
        prepared / "phase72_two_year_feature_rows.csv", keep_default_na=False
    ).to_dict(orient="records")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Phase 72 two-year local indexes are not contiguous")
    source_indexes = np.asarray(
        [int(row["source_sample_index"]) for row in rows], dtype=np.int64
    )
    source_rows = list(verified_source["feature_rows"])
    if any(index < 0 or index >= len(source_rows) for index in source_indexes):
        raise ValueError("Phase 72 two-year source index is out of range")
    for row, source_index in zip(rows, source_indexes):
        source = source_rows[int(source_index)]
        for field in (
            "region_id",
            "unit_id",
            "row",
            "col",
            "spatial_block_id",
            "origin_year",
        ):
            if str(row[field]) != str(source[field]):
                raise ValueError(
                    "Phase 72 two-year prepared row identity mismatch"
                )
    matrices = {
        name: np.asarray(value)[source_indexes]
        for name, value in verified_source["matrices"].items()
    }
    splits = json.loads(
        (prepared / "phase72_two_year_split_registry.json").read_text(
            encoding="utf-8"
        )
    )
    development = _load_npz(
        prepared / "phase72_two_year_development_targets.npz"
    )
    if _target_arrays_sha256(development) != str(
        manifest["development_targets_sha256"]
    ):
        raise ValueError("Phase 72 two-year development target hash mismatch")
    result = {
        "manifest": manifest,
        "manifest_sha256": (
            prepared / "phase72_two_year_prepared.sha256"
        ).read_text(encoding="ascii").strip(),
        "protocol": protocol,
        "feature_rows": rows,
        "matrices": matrices,
        "split_registries": splits,
        "development_targets": development,
    }
    if include_confirmation:
        confirmation = _load_npz(
            prepared / "phase72_two_year_confirmation_targets.npz"
        )
        if _target_arrays_sha256(confirmation) != str(
            manifest["confirmation_targets_sha256"]
        ):
            raise ValueError(
                "Phase 72 two-year confirmation target hash mismatch"
            )
        result["confirmation_targets"] = confirmation
    return result


def _variant_matrix(
    variant_id: str, matrices: Mapping[str, np.ndarray]
) -> np.ndarray:
    if variant_id == "explicit_history":
        return np.asarray(matrices["explicit_history"])
    if variant_id == "explicit_plus_geofm_temporal_full":
        return np.concatenate(
            [matrices["explicit_history"], matrices["geofm_temporal_full"]],
            axis=1,
        )
    raise ValueError(f"Unknown Phase 72 two-year base variant: {variant_id}")


def _control_matrices(
    variant_id: str,
    matrices: Mapping[str, np.ndarray],
    feature_rows: Sequence[Mapping[str, object]],
    *,
    train_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    axis_id: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    control_id = _CONTROL_VARIANTS[variant_id]
    result = []
    for split_name, raw_indexes in (
        ("train", train_indexes),
        ("validation", validation_indexes),
    ):
        indexes = np.asarray(raw_indexes, dtype=np.int64)
        subset_rows = [feature_rows[int(index)] for index in indexes]
        control = build_phase72b_control_features(
            control_id,
            matrices["embedding_history"][indexes],
            matrices["history_mask"][indexes],
            subset_rows,
            partition_ids=[f"{axis_id}:{split_name}"] * len(indexes),
            seed=int(seed),
            output_dim=matrices["geofm_temporal_full"].shape[1],
        )
        result.append(
            np.concatenate(
                [matrices["explicit_history"][indexes], control["matrix"]],
                axis=1,
            )
        )
    return result[0], result[1]


def _outcomes(
    targets: Mapping[str, np.ndarray], endpoint: str
) -> dict[int, int]:
    indexes = np.asarray(targets["sample_index"])
    values = np.asarray(targets[endpoint])
    if indexes.ndim != 1 or values.ndim != 1 or len(indexes) != len(values):
        raise ValueError("Phase 72 two-year target alignment mismatch")
    if len(set(int(value) for value in indexes)) != len(indexes):
        raise ValueError("Phase 72 two-year target indexes are duplicate")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Phase 72 two-year target must be binary")
    return {
        int(index): int(value) for index, value in zip(indexes, values)
    }


def _y_for_indexes(
    outcomes: Mapping[int, int], indexes: Sequence[int]
) -> np.ndarray:
    missing = [int(index) for index in indexes if int(index) not in outcomes]
    if missing:
        raise ValueError(
            f"Phase 72 two-year outcomes missing indexes: {missing[:5]}"
        )
    return np.asarray([outcomes[int(index)] for index in indexes], np.int8)


def _bundle_key(
    endpoint: str, axis_id: str, variant_id: str, seed: int | None
) -> str:
    seed_id = "base" if seed is None else f"seed{int(seed)}"
    return "__".join((endpoint, axis_id, variant_id, seed_id))


def _load_progress(
    output: Path, prepared_sha256: str, protocol_sha256: str
) -> dict[str, object]:
    path = output / "phase72_two_year_fit_progress.json"
    if not path.exists():
        return {
            "status": "phase72_two_year_fit_in_progress",
            "prepared_sha256": prepared_sha256,
            "protocol_sha256": protocol_sha256,
            "entries": {},
            "validation_rows": [],
        }
    progress = load_hashed_json(path)
    if (
        progress.get("prepared_sha256") != prepared_sha256
        or progress.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("Phase 72 two-year fit progress binding mismatch")
    return progress


def _checkpoint_bundle(
    *,
    output: Path,
    progress: dict[str, object],
    endpoint: str,
    axis_id: str,
    variant_id: str,
    seed: int | None,
    bundle: Mapping[str, object],
    validation_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    key = _bundle_key(endpoint, axis_id, variant_id, seed)
    bundles_dir = output / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundles_dir / f"{key}.joblib"
    persisted = {
        **dict(bundle),
        "endpoint": endpoint,
        "control_seed": "" if seed is None else int(seed),
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
    }
    joblib.dump(persisted, bundle_path)
    record = {
        "key": key,
        "endpoint": endpoint,
        "axis_id": axis_id,
        "variant_id": variant_id,
        "control_seed": "" if seed is None else int(seed),
        "bundle_path": str(Path("bundles") / bundle_path.name),
        "bundle_sha256": _file_sha256(bundle_path),
        "feature_count": int(bundle["feature_count"]),
        "validation_average_precision": float(
            bundle["validation_metrics"]["average_precision"]
        ),
        "validation_brier": float(bundle["validation_metrics"]["brier"]),
        "validation_ece": float(bundle["validation_metrics"]["ece"]),
    }
    entries = dict(progress.get("entries", {}))
    entries[key] = record
    progress["entries"] = entries
    stored_rows = list(progress.get("validation_rows", []))
    stored_rows.extend(dict(row) for row in validation_rows)
    progress["validation_rows"] = stored_rows
    write_hashed_json(output / "phase72_two_year_fit_progress.json", progress)
    return persisted, record


def _resume_bundle(
    output: Path,
    progress: Mapping[str, object],
    endpoint: str,
    axis_id: str,
    variant_id: str,
    seed: int | None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    key = _bundle_key(endpoint, axis_id, variant_id, seed)
    record = dict(progress.get("entries", {}).get(key, {}))
    if not record:
        return None
    path = output / str(record["bundle_path"])
    if _file_sha256(path) != str(record["bundle_sha256"]):
        raise ValueError(f"Phase 72 two-year bundle hash mismatch: {key}")
    bundle = joblib.load(path)
    expected = (endpoint, axis_id, variant_id)
    actual = (
        str(bundle.get("endpoint")),
        str(bundle.get("axis_id")),
        str(bundle.get("variant_id")),
    )
    if actual != expected:
        raise ValueError(f"Phase 72 two-year bundle identity mismatch: {key}")
    return bundle, record


def fit_freeze_phase72_two_year_models(
    *,
    prepared_dir: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    output_dir: Path | str,
) -> tuple[dict[str, object], dict[str, Path]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared = load_verified_phase72_two_year_prepared(
        prepared_dir,
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        include_confirmation=False,
    )
    protocol = dict(prepared["protocol"])
    prepared_hash = str(prepared["manifest_sha256"])
    protocol_hash = canonical_json_sha256(protocol)
    progress = _load_progress(output, prepared_hash, protocol_hash)
    matrices = dict(prepared["matrices"])
    feature_rows = list(prepared["feature_rows"])
    bundle_records = []
    selected_control_seeds = {}
    selected_configs = {}

    for endpoint in PHASE72_TWO_YEAR_ENDPOINTS:
        outcomes = _outcomes(prepared["development_targets"], endpoint)
        registry = dict(prepared["split_registries"][endpoint])
        selected_control_seeds[endpoint] = {}
        selected_configs[endpoint] = {}
        for axis_id in _SEARCH_AXES:
            axis = registry[axis_id]
            train_indexes = [int(value) for value in axis["train"]]
            validation_indexes = [int(value) for value in axis["validation"]]
            train_y = _y_for_indexes(outcomes, train_indexes)
            validation_y = _y_for_indexes(outcomes, validation_indexes)
            selected_configs[endpoint][axis_id] = {}
            for variant_id in _BASE_VARIANTS:
                matrix = _variant_matrix(variant_id, matrices)
                resumed = _resume_bundle(
                    output, progress, endpoint, axis_id, variant_id, None
                )
                if resumed is None:
                    bundle, validation_rows = fit_select_phase72b_model(
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
                    bundle, record = _checkpoint_bundle(
                        output=output,
                        progress=progress,
                        axis_id=axis_id,
                        variant_id=variant_id,
                        seed=None,
                        bundle=bundle,
                        validation_rows=[
                            {"endpoint": endpoint, **row}
                            for row in validation_rows
                        ],
                    )
                else:
                    bundle, record = resumed
                bundle_records.append(record)
                selected_configs[endpoint][axis_id][variant_id] = dict(
                    bundle["estimator_params"]
                )
            selected_control_seeds[endpoint][axis_id] = {}
            for variant_id in _CONTROL_VARIANTS:
                seed_records = []
                for seed in protocol["controls"]["seeds"]:
                    train_matrix, validation_matrix = _control_matrices(
                        variant_id,
                        matrices,
                        feature_rows,
                        train_indexes=train_indexes,
                        validation_indexes=validation_indexes,
                        endpoint=endpoint,
                        axis_id=axis_id,
                        seed=int(seed),
                    )
                    resumed = _resume_bundle(
                        output,
                        progress,
                        endpoint,
                        axis_id,
                        variant_id,
                        int(seed),
                    )
                    if resumed is None:
                        bundle, validation_rows = fit_select_phase72b_model(
                            train_matrix,
                            train_y,
                            validation_matrix,
                            validation_y,
                            variant_id=variant_id,
                            axis_id=axis_id,
                            protocol=protocol,
                            train_indexes=train_indexes,
                            validation_indexes=validation_indexes,
                        )
                        bundle, record = _checkpoint_bundle(
                            output=output,
                            progress=progress,
                            endpoint=endpoint,
                            axis_id=axis_id,
                            variant_id=variant_id,
                            seed=int(seed),
                            bundle=bundle,
                            validation_rows=[
                                {
                                    "endpoint": endpoint,
                                    "control_seed": int(seed),
                                    **row,
                                }
                                for row in validation_rows
                            ],
                        )
                    else:
                        bundle, record = resumed
                    bundle_records.append(record)
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
                selected_control_seeds[endpoint][axis_id][variant_id] = int(
                    best_record["control_seed"]
                )
                selected_configs[endpoint][axis_id][variant_id] = dict(
                    best_bundle["estimator_params"]
                )

        for axis_id, axis in registry.items():
            if not axis_id.startswith("spatial_"):
                continue
            train_indexes = [int(value) for value in axis["train"]]
            validation_indexes = [int(value) for value in axis["validation"]]
            if not train_indexes or not validation_indexes:
                continue
            train_y = _y_for_indexes(outcomes, train_indexes)
            validation_y = _y_for_indexes(outcomes, validation_indexes)
            if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
                continue
            for variant_id in _BASE_VARIANTS:
                matrix = _variant_matrix(variant_id, matrices)
                resumed = _resume_bundle(
                    output, progress, endpoint, axis_id, variant_id, None
                )
                if resumed is None:
                    bundle, validation_rows = fit_fixed_phase72b_model(
                        matrix[train_indexes],
                        train_y,
                        matrix[validation_indexes],
                        validation_y,
                        variant_id=variant_id,
                        axis_id=axis_id,
                        protocol=protocol,
                        candidate_config=selected_configs[endpoint][
                            "pooled_temporal"
                        ][variant_id],
                        train_indexes=train_indexes,
                        validation_indexes=validation_indexes,
                    )
                    bundle, record = _checkpoint_bundle(
                        output=output,
                        progress=progress,
                        endpoint=endpoint,
                        axis_id=axis_id,
                        variant_id=variant_id,
                        seed=None,
                        bundle=bundle,
                        validation_rows=[
                            {"endpoint": endpoint, **row}
                            for row in validation_rows
                        ],
                    )
                else:
                    bundle, record = resumed
                bundle_records.append(record)

    selected = {
        "status": "phase72_two_year_models_frozen",
        "prepared_sha256": prepared_hash,
        "protocol_sha256": protocol_hash,
        "selected_control_seeds": selected_control_seeds,
        "selected_configs": selected_configs,
        "bundle_records": bundle_records,
        "bundle_count": len(bundle_records),
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
    }
    validation_path = output / "phase72_two_year_validation_metrics.csv"
    pd.DataFrame(progress["validation_rows"]).to_csv(validation_path, index=False)
    selected_json, selected_hash = write_hashed_json(
        output / "phase72_two_year_selected_models.json", selected
    )
    progress["status"] = "phase72_two_year_fit_complete"
    progress["selected_models_sha256"] = selected_hash.read_text(
        encoding="ascii"
    ).strip()
    write_hashed_json(output / "phase72_two_year_fit_progress.json", progress)
    return selected, {
        "validation_metrics": validation_path,
        "selected_models": selected_json,
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


def _overall_status(endpoint_results: Mapping[str, Mapping[str, object]]) -> str:
    statuses = {
        str(result.get("phase72b_status", ""))
        for result in endpoint_results.values()
    }
    if "phase72b_inputs_not_ready" in statuses:
        return "phase72_two_year_inputs_not_ready"
    if statuses == {"geofm_information_supported"} and len(
        endpoint_results
    ) == len(PHASE72_TWO_YEAR_ENDPOINTS):
        return "two_year_geofm_information_supported"
    if statuses == {"geofm_information_not_supported"}:
        return "two_year_geofm_information_not_supported"
    return "two_year_geofm_information_mixed"


def confirm_phase72_two_year_endpoint_screen(
    *,
    prepared_dir: Path | str,
    phase72a_package_dir: Path | str,
    phase72b_prepared_dir: Path | str,
    frozen_dir: Path | str,
) -> dict[str, object]:
    prepared = load_verified_phase72_two_year_prepared(
        prepared_dir,
        phase72a_package_dir=phase72a_package_dir,
        phase72b_prepared_dir=phase72b_prepared_dir,
        include_confirmation=True,
    )
    frozen = Path(frozen_dir)
    selected = load_hashed_json(
        frozen / "phase72_two_year_selected_models.json"
    )
    selected_hash = (
        frozen / "phase72_two_year_selected_models.sha256"
    ).read_text(encoding="ascii").strip()
    if selected.get("status") != "phase72_two_year_models_frozen":
        raise ValueError("Phase 72 two-year models are not frozen")
    if selected.get("prepared_sha256") != prepared["manifest_sha256"]:
        raise ValueError("Phase 72 two-year selected/prepared binding mismatch")
    protocol = dict(prepared["protocol"])
    if selected.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("Phase 72 two-year selected/protocol binding mismatch")
    matrices = dict(prepared["matrices"])
    feature_rows = list(prepared["feature_rows"])
    bundles = {}
    bundle_hashes = {}
    for raw_record in selected["bundle_records"]:
        record = dict(raw_record)
        path = frozen / str(record["bundle_path"])
        actual_hash = _file_sha256(path)
        if actual_hash != str(record["bundle_sha256"]):
            raise ValueError(
                f"Phase 72 two-year frozen bundle hash mismatch: {record['key']}"
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
            raise ValueError("Duplicate Phase 72 two-year bundle")
        bundles[key] = bundle
        bundle_hashes[str(record["key"])] = actual_hash

    metrics_rows = []
    prediction_rows = []
    bootstrap_rows = []
    control_rows = []
    transfer_rows = []
    spatial_rows = []
    endpoint_results = {}
    confirmation_targets = dict(prepared["confirmation_targets"])
    for endpoint in PHASE72_TWO_YEAR_ENDPOINTS:
        outcomes = _outcomes(confirmation_targets, endpoint)
        registry = dict(prepared["split_registries"][endpoint])
        groups = {}
        for key, bundle in bundles.items():
            bundle_endpoint, axis_id, variant_id, seed = key
            if bundle_endpoint != endpoint:
                continue
            indexes = np.asarray(registry[axis_id]["test"], dtype=np.int64)
            if not len(indexes):
                continue
            y = _y_for_indexes(outcomes, indexes)
            if variant_id in _CONTROL_VARIANTS:
                subset_rows = [feature_rows[int(index)] for index in indexes]
                control = build_phase72b_control_features(
                    _CONTROL_VARIANTS[variant_id],
                    matrices["embedding_history"][indexes],
                    matrices["history_mask"][indexes],
                    subset_rows,
                    partition_ids=[f"{axis_id}:test"] * len(indexes),
                    seed=int(seed),
                    output_dim=matrices["geofm_temporal_full"].shape[1],
                )
                x = np.concatenate(
                    [matrices["explicit_history"][indexes], control["matrix"]],
                    axis=1,
                )
            else:
                x = _variant_matrix(variant_id, matrices)[indexes]
            probability = predict_phase72b_bundle(bundle, x)
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
                "control_seed": "" if seed is None else int(seed),
                "rows": len(indexes),
                "positives": int(y.sum()),
                "prevalence": round(float(y.mean()), 12),
                "model_family": str(bundle["model_family"]),
                "calibration_method": str(bundle["calibration_method"]),
                **metric,
            }
            metrics_rows.append(row)
            groups[(axis_id, variant_id, seed)] = {
                "indexes": indexes,
                "outcome": y,
                "probability": probability,
                "metric": row,
            }
            for position, local_index in enumerate(indexes):
                source = feature_rows[int(local_index)]
                prediction_rows.append(
                    {
                        "endpoint": endpoint,
                        "sample_index": int(local_index),
                        "source_sample_index": int(source["source_sample_index"]),
                        "axis_id": axis_id,
                        "variant_id": variant_id,
                        "control_seed": "" if seed is None else int(seed),
                        "outcome": int(y[position]),
                        "probability": round(float(probability[position]), 12),
                        "region_id": str(source["region_id"]),
                        "spatial_block_id": str(source["spatial_block_id"]),
                        "origin_year": int(source["origin_year"]),
                    }
                )

        def group(axis_id: str, variant_id: str, seed: int | None = None):
            try:
                return groups[(axis_id, variant_id, seed)]
            except KeyError as exc:
                raise ValueError(
                    "Missing Phase 72 two-year confirmation group: "
                    f"{endpoint}/{axis_id}/{variant_id}/{seed}"
                ) from exc

        pooled_explicit = group("pooled_temporal", "explicit_history")
        pooled_primary = group(
            "pooled_temporal", "explicit_plus_geofm_temporal_full"
        )
        pooled_delta = _metric_delta(
            pooled_explicit["metric"], pooled_primary["metric"]
        )
        pooled_indexes = pooled_primary["indexes"]
        pooled_bootstrap = paired_block_bootstrap(
            pooled_primary["outcome"],
            pooled_explicit["probability"],
            pooled_primary["probability"],
            [feature_rows[int(index)] for index in pooled_indexes],
            iterations=int(protocol["bootstrap"]["iterations"]),
            seed=int(protocol["bootstrap"]["seed"]),
        )
        bootstrap_rows.append(
            {
                "endpoint": endpoint,
                "comparison_id": "primary_vs_explicit",
                "axis_id": "pooled_temporal",
                **pooled_bootstrap,
            }
        )
        endpoint_controls = []
        for variant_id, control_id in _CONTROL_VARIANTS.items():
            selected_seed = int(
                selected["selected_control_seeds"][endpoint][
                    "pooled_temporal"
                ][variant_id]
            )
            control_group = group(
                "pooled_temporal", variant_id, selected_seed
            )
            control_row = {
                "endpoint": endpoint,
                "control_id": control_id,
                "variant_id": variant_id,
                "selected_seed": selected_seed,
                **_metric_delta(control_group["metric"], pooled_primary["metric"]),
            }
            endpoint_controls.append(control_row)
            control_rows.append(control_row)
        endpoint_transfers = []
        for axis_id in (
            "bishan_to_dongxing",
            "dongxing_to_bishan",
        ):
            explicit = group(axis_id, "explicit_history")
            primary = group(axis_id, "explicit_plus_geofm_temporal_full")
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
            explicit = group(axis_id, "explicit_history")
            primary = group(axis_id, "explicit_plus_geofm_temporal_full")
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

    status = _overall_status(endpoint_results)
    return {
        "phase": "phase72_two_year_endpoint_screen",
        "phase72_two_year_status": status,
        "decision_rule": PHASE72_TWO_YEAR_DECISION_RULE,
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
        "selected_models_sha256": selected_hash,
        "bundle_hashes": bundle_hashes,
        "next_action": (
            "Record this result in the Phase 72 exhaustion analysis. Do not "
            "enter Phase 72C or revise the formal manuscript from this screen."
        ),
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
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


def _markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Phase 72 Two-Year Endpoint Screen",
        "",
        f"Status: `{result['phase72_two_year_status']}`",
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


def write_phase72_two_year_confirmation_artifacts(
    result: Mapping[str, object], output_dir: Path | str
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Phase 72 two-year confirmation output must be new or empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics": output / "phase72_two_year_metrics.csv",
        "predictions": output / "phase72_two_year_predictions.csv",
        "bootstrap": output / "phase72_two_year_bootstrap_deltas.csv",
        "controls": output / "phase72_two_year_control_comparison.csv",
        "transfers": output / "phase72_two_year_transfer_summary.csv",
        "spatial": output / "phase72_two_year_spatial_summary.csv",
        "result": output / "phase72_two_year_endpoint_screen.json",
        "markdown": output / "phase72_two_year_endpoint_screen.md",
    }
    for key in (
        "metrics",
        "predictions",
        "bootstrap",
        "controls",
        "transfers",
        "spatial",
    ):
        _write_csv(artifacts[key], result[f"{key[:-1]}_rows"] if key.endswith("s") else result[f"{key}_rows"])
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
    artifacts["markdown"].write_text(_markdown(result), encoding="utf-8")
    receipt = {
        "status": "phase72_two_year_confirmation_receipt",
        "phase72_two_year_status": result["phase72_two_year_status"],
        "prepared_sha256": result["prepared_sha256"],
        "selected_models_sha256": result["selected_models_sha256"],
        "artifacts": [
            {"name": path.name, "sha256": _file_sha256(path)}
            for path in artifacts.values()
        ],
        "claim_boundary": PHASE72_TWO_YEAR_CLAIM_BOUNDARY,
    }
    receipt_json, receipt_hash = write_hashed_json(
        output / "phase72_two_year_confirmation_receipt.json", receipt
    )
    artifacts["receipt"] = receipt_json
    artifacts["receipt_sha256"] = receipt_hash
    return artifacts
