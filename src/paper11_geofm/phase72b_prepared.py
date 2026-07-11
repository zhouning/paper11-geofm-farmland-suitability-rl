from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .phase72b_protocol import canonical_json_sha256, load_hashed_json
from .phase72b_splits import audit_phase72b_splits
from .phase72b_terrain import _file_sha256


PREPARED_ARTIFACT_NAMES = (
    "phase72b_terrain_manifest.csv",
    "phase72b_feature_manifest.csv",
    "phase72b_feature_registry.json",
    "phase72b_feature_rows.csv",
    "phase72b_feature_matrices.npz",
    "phase72b_development_targets.npz",
    "phase72b_confirmation_targets.npz",
    "phase72b_split_registry.json",
    "phase72b_row_alignment_audit.csv",
    "phase72b_leakage_audit.json",
    "phase72b_frozen_protocol.json",
    "phase72b_frozen_protocol.sha256",
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _manifest_records(manifest: Mapping[str, object]) -> dict[str, str]:
    records = {}
    for raw_record in manifest.get("artifacts", []):
        record = dict(raw_record)
        name = str(record.get("name", ""))
        if name in records:
            raise ValueError(f"Duplicate Phase 72B prepared artifact: {name}")
        records[name] = str(record.get("sha256", "")).lower()
    if set(records) != set(PREPARED_ARTIFACT_NAMES):
        raise ValueError("Phase 72B prepared artifact manifest mismatch")
    return records


def verify_phase72b_prepared_artifact(
    prepared_dir: Path | str,
    manifest: Mapping[str, object],
    name: str,
) -> None:
    prepared = Path(prepared_dir)
    records = _manifest_records(manifest)
    actual = _file_sha256(prepared / str(name)).lower()
    if actual != records.get(str(name), ""):
        raise ValueError(
            f"Phase 72B prepared artifact hash mismatch: {name}"
        )


def load_verified_phase72b_prepared(
    prepared_dir: Path | str,
    *,
    deferred_names: Iterable[str] = (),
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    manifest_path = prepared / "phase72b_prepared_artifacts.json"
    manifest_hash_path = prepared / "phase72b_prepared_artifacts.sha256"
    manifest = load_hashed_json(manifest_path, manifest_hash_path)
    if manifest.get("status") != "phase72b_prepared_artifacts_frozen":
        raise ValueError("Phase 72B prepared artifact manifest is not frozen")
    records = _manifest_records(manifest)
    deferred = {str(name) for name in deferred_names}
    if not deferred.issubset(records):
        raise ValueError("Unknown deferred Phase 72B prepared artifact")
    for name in PREPARED_ARTIFACT_NAMES:
        if name not in deferred:
            verify_phase72b_prepared_artifact(prepared, manifest, name)

    frozen_protocol = load_hashed_json(
        prepared / "phase72b_frozen_protocol.json",
        prepared / "phase72b_frozen_protocol.sha256",
    )
    protocol_hash = (
        prepared / "phase72b_frozen_protocol.sha256"
    ).read_text(encoding="ascii").strip().lower()
    if str(manifest.get("frozen_protocol_sha256", "")).lower() != protocol_hash:
        raise ValueError("Phase 72B prepared manifest protocol hash mismatch")
    protocol = dict(frozen_protocol["tracked_protocol"])

    split_registry = json.loads(
        (prepared / "phase72b_split_registry.json").read_text(encoding="utf-8")
    )
    if canonical_json_sha256(split_registry) != str(
        frozen_protocol["split_registry_sha256"]
    ):
        raise ValueError("Phase 72B split registry semantic hash mismatch")
    feature_registry = json.loads(
        (prepared / "phase72b_feature_registry.json").read_text(encoding="utf-8")
    )
    if canonical_json_sha256(feature_registry) != str(
        frozen_protocol["feature_registry_sha256"]
    ):
        raise ValueError("Phase 72B feature registry semantic hash mismatch")
    feature_rows = pd.read_csv(
        prepared / "phase72b_feature_rows.csv", keep_default_na=False
    ).to_dict(orient="records")
    if [int(row["sample_index"]) for row in feature_rows] != list(
        range(len(feature_rows))
    ):
        raise ValueError("Phase 72B feature row alignment failed")
    with np.load(prepared / "phase72b_feature_matrices.npz") as loaded:
        matrices = {name: loaded[name] for name in loaded.files}
    expected_matrices = {
        str(row["matrix_id"]): str(row["sha256"])
        for row in frozen_protocol["feature_manifest_rows"]
    }
    if set(matrices) != set(expected_matrices):
        raise ValueError("Phase 72B feature matrix manifest mismatch")
    for matrix_id, expected_hash in expected_matrices.items():
        matrix = np.asarray(matrices[matrix_id])
        if len(matrix) != len(feature_rows):
            raise ValueError(
                f"Phase 72B feature matrix row mismatch: {matrix_id}"
            )
        if _array_sha256(matrix) != expected_hash:
            raise ValueError(
                f"Phase 72B feature matrix semantic hash mismatch: {matrix_id}"
            )
    for axis_id, axis in split_registry.items():
        for split_name in ("train", "validation", "test"):
            indexes = [int(value) for value in axis.get(split_name, [])]
            if any(index < 0 or index >= len(feature_rows) for index in indexes):
                raise ValueError(
                    f"Phase 72B split index out of range: {axis_id}/{split_name}"
                )
    recomputed_audit = audit_phase72b_splits(
        feature_rows,
        split_registry,
        train_years=protocol["years"]["train"],
        validation_year=int(protocol["years"]["validation"][0]),
        test_year=int(protocol["years"]["test"][0]),
    )
    stored_audit = json.loads(
        (prepared / "phase72b_leakage_audit.json").read_text(encoding="utf-8")
    )
    if canonical_json_sha256(recomputed_audit) != canonical_json_sha256(
        stored_audit
    ):
        raise ValueError("Phase 72B leakage audit recomputation mismatch")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_hash_path.read_text(
            encoding="ascii"
        ).strip().lower(),
        "frozen_protocol": frozen_protocol,
        "protocol_hash": protocol_hash,
        "feature_registry": feature_registry,
        "feature_rows": feature_rows,
        "matrices": matrices,
        "split_registry": split_registry,
        "leakage_audit": recomputed_audit,
    }
