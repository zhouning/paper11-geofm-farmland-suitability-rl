from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .drl_smoke_env import Phase4InputContractEnv
from .tiled_inputs import load_tiled_variant_input


PHASE17_CLAIM_BOUNDARY = (
    "Phase 17 is a tiled MaskablePPO readiness smoke check; it does not "
    "train, tune, evaluate, or compare a useful DRL policy, does not enable "
    "suitability reward, and does not report planning performance."
)


def build_phase17_tiled_contract_summary(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    tile_id: str | None = None,
    tile_selection: str = "largest",
    seed: int = 0,
    total_timesteps: int = 8,
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")

    selected = _select_tile(Path(tile_index_csv), tile_id, tile_selection)
    try:
        tiled = load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            selected["tile_id"],
            variant_id=variant_id,
        )
    except ValueError as exc:
        if "suitability reward variants are disabled" in str(exc):
            raise ValueError(
                "Phase 17 suitability reward variants are disabled by default; "
                "use a representation-only variant such as B0 or B1"
            ) from exc
        raise

    env = Phase4InputContractEnv(tiled, max_steps=int(total_timesteps))
    obs, info = env.reset(seed=int(seed))
    initial_mask = env.action_masks()
    return {
        "phase": "phase17_tiled_maskableppo_readiness",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "tile_id": tiled.tile_id,
        "tile_selection": selected["selection"],
        "variant_id": str(info["variant_id"]),
        "seed": int(seed),
        "learn_timesteps": int(total_timesteps),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "reward_mode": str(info["reward_mode"]),
        "initial_valid_actions": int(initial_mask.sum()),
        "claim_boundary": PHASE17_CLAIM_BOUNDARY,
    }


def run_phase17_tiled_maskableppo_readiness(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    variant_id: str = "B1",
    tile_id: str | None = None,
    tile_selection: str = "largest",
    total_timesteps: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 17 tiled MaskablePPO readiness requires "
            "stable-baselines3 and sb3-contrib"
        ) from exc

    selected = _select_tile(Path(tile_index_csv), tile_id, tile_selection)
    tiled = _load_tiled_variant_for_phase17(
        phase2_output_dir,
        tile_index_csv,
        selected["tile_id"],
        variant_id,
    )
    env = Phase4InputContractEnv(tiled, max_steps=int(total_timesteps))
    obs, info = env.reset(seed=int(seed))
    initial_mask = env.action_masks()
    masking_supported = bool(is_masking_supported(env))
    if not masking_supported:
        raise ValueError(
            "Phase 17 tiled env does not expose action_masks for sb3-contrib"
        )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="XPU device count is zero!.*",
            category=UserWarning,
        )
        model.learn(total_timesteps=int(total_timesteps))

    obs, _ = env.reset(seed=int(seed))
    action_masks = env.action_masks()
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=action_masks,
    )
    predicted_action = int(action)
    predicted_action_valid = bool(action_masks[predicted_action])
    if not predicted_action_valid:
        raise ValueError("Phase 17 predicted action is not valid under the action mask")

    return {
        "phase": "phase17_tiled_maskableppo_readiness",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "tile_index_csv": str(Path(tile_index_csv)),
        "tile_id": tiled.tile_id,
        "tile_selection": selected["selection"],
        "variant_id": str(info["variant_id"]),
        "seed": int(seed),
        "learn_timesteps": int(total_timesteps),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "reward_mode": str(info["reward_mode"]),
        "initial_valid_actions": int(initial_mask.sum()),
        "masking_supported": masking_supported,
        "device": "cpu",
        "predicted_action": predicted_action,
        "predicted_action_valid": predicted_action_valid,
        "selected_block_id": str(env.block_ids[predicted_action]),
        "dependencies": _dependency_metadata(),
        "readiness_status": "passed_tiled_maskableppo_smoke",
        "recommendation": (
            "tiled_maskableppo_contract_ready_for_larger_controlled_smokes"
        ),
        "claim_boundary": PHASE17_CLAIM_BOUNDARY,
    }


def write_phase17_tiled_maskableppo_readiness_artifact(
    summary: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "phase17_tiled_maskableppo_readiness.json"
    artifact_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def _select_tile(
    tile_index_csv: Path,
    tile_id: str | None,
    tile_selection: str,
) -> dict[str, object]:
    rows = _read_tile_rows(tile_index_csv)
    if tile_id is not None and str(tile_id).strip():
        normalized_tile_id = str(tile_id).strip()
        for row in rows:
            if row["tile_id"] == normalized_tile_id:
                return {"tile_id": normalized_tile_id, "selection": "explicit"}
        raise ValueError(f"Tile ID not found in tile index: {normalized_tile_id}")

    selection = str(tile_selection).strip().lower()
    if selection != "largest":
        raise ValueError(f"Unknown Phase 17 tile selection: {tile_selection}")
    largest = max(rows, key=lambda row: int(row["n_blocks"]))
    return {"tile_id": str(largest["tile_id"]), "selection": "largest"}


def _load_tiled_variant_for_phase17(
    phase2_output_dir: Path | str,
    tile_index_csv: Path | str,
    tile_id: object,
    variant_id: str,
):
    try:
        return load_tiled_variant_input(
            phase2_output_dir,
            tile_index_csv,
            str(tile_id),
            variant_id=variant_id,
        )
    except ValueError as exc:
        if "suitability reward variants are disabled" in str(exc):
            raise ValueError(
                "Phase 17 suitability reward variants are disabled by default; "
                "use a representation-only variant such as B0 or B1"
            ) from exc
        raise


def _read_tile_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 17 tile index CSV: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in ("tile_id", "block_ids") if field not in fieldnames]
        if missing:
            raise ValueError(f"Phase 17 tile index is missing columns: {missing}")
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            block_ids = [
                part.strip()
                for part in str(row.get("block_ids", "")).split(";")
                if part.strip()
            ]
            if not tile_id:
                raise ValueError("Phase 17 tile index contains a row without tile_id")
            if not block_ids:
                raise ValueError(f"Tile {tile_id} contains no block IDs")
            rows.append(
                {
                    "tile_id": tile_id,
                    "block_ids": block_ids,
                    "n_blocks": len(block_ids),
                }
            )
    if not rows:
        raise ValueError("Phase 17 tile index contains no tile rows")
    return rows


def _dependency_metadata() -> dict[str, dict[str, object]]:
    return {
        "stable_baselines3": _package_metadata("stable-baselines3"),
        "sb3_contrib": _package_metadata("sb3-contrib"),
    }


def _package_metadata(distribution_name: str) -> dict[str, object]:
    try:
        package_version = version(distribution_name)
    except PackageNotFoundError:
        return {"available": False, "version": None}
    return {"available": True, "version": package_version}
