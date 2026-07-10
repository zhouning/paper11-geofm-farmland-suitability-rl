from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


PHASE72B_CLAIM_BOUNDARY = (
    "Phase 72B is a leakage-free low-cost information-gain screen using "
    "independent annual product labels. It does not implement GeoFM-STaR, "
    "alter planning rewards, run planning, or revise the formal manuscript."
)


@dataclass(frozen=True)
class Phase72BProtocol:
    seed: int
    terrain_source_id: str
    terrain_collection: str
    terrain_band: str
    terrain_scale_m: int
    terrain_features: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    control_seeds: tuple[int, ...]
    random_projection_dim: int
    spatial_block_size: int
    spatial_folds: int
    buffer_rings: int
    bootstrap_iterations: int
    bootstrap_seed: int
    gates: dict[str, float]
    raw: dict[str, object]


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_hashed_json(
    path: Path | str, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(dict(payload)))
    json_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8"
    )
    hash_path = json_path.with_suffix(".sha256")
    hash_path.write_text(
        canonical_json_sha256(normalized) + "\n", encoding="ascii"
    )
    return json_path, hash_path


def load_hashed_json(
    json_path: Path | str, hash_path: Path | str | None = None
) -> dict[str, object]:
    source = Path(json_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    digest_path = (
        Path(hash_path) if hash_path is not None else source.with_suffix(".sha256")
    )
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    actual = canonical_json_sha256(payload)
    if expected != actual:
        raise ValueError(
            "Phase 72B frozen JSON hash mismatch: "
            f"expected {expected}, got {actual}"
        )
    return payload


def load_phase72b_protocol(path: Path | str) -> Phase72BProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("phase") != "phase72b_geofm_information_gain_screen":
        raise ValueError("Invalid Phase 72B protocol phase")

    terrain = dict(payload["terrain"])
    years = dict(payload["years"])
    controls = dict(payload["controls"])
    spatial = dict(payload["spatial"])
    bootstrap = dict(payload["bootstrap"])
    gates = {
        str(key): float(value)
        for key, value in dict(payload["gates"]).items()
    }
    partition = [
        int(value)
        for name in ("train", "validation", "test")
        for value in years[name]
    ]
    if sorted(partition) != list(range(2017, 2024)):
        raise ValueError("Phase 72B years must partition origins 2017-2023")
    if tuple(int(value) for value in controls["seeds"]) != (
        72,
        73,
        74,
        75,
        76,
    ):
        raise ValueError("Phase 72B control seeds are frozen")

    return Phase72BProtocol(
        seed=int(payload["seed"]),
        terrain_source_id=str(terrain["source_id"]),
        terrain_collection=str(terrain["collection"]),
        terrain_band=str(terrain["band"]),
        terrain_scale_m=int(terrain["scale_m"]),
        terrain_features=tuple(str(value) for value in terrain["feature_names"]),
        train_years=tuple(int(value) for value in years["train"]),
        validation_years=tuple(
            int(value) for value in years["validation"]
        ),
        test_years=tuple(int(value) for value in years["test"]),
        control_seeds=tuple(int(value) for value in controls["seeds"]),
        random_projection_dim=int(controls["random_projection_dim"]),
        spatial_block_size=int(spatial["block_size"]),
        spatial_folds=int(spatial["folds"]),
        buffer_rings=int(spatial["buffer_rings"]),
        bootstrap_iterations=int(bootstrap["iterations"]),
        bootstrap_seed=int(bootstrap["seed"]),
        gates=gates,
        raw=payload,
    )
