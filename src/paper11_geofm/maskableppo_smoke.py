from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .drl_smoke_env import make_phase4_smoke_env


PHASE7_CLAIM_BOUNDARY = (
    "Phase 7 is a MaskablePPO compatibility smoke check; it does not train, "
    "tune, evaluate, or report a useful DRL policy."
)


def run_phase7_maskableppo_smoke(
    phase2_output_dir: Path | str,
    variant_id: str = "B3",
    total_timesteps: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import is_masking_supported
    except ImportError as exc:
        raise RuntimeError(
            "Phase 7 MaskablePPO smoke requires stable-baselines3 and sb3-contrib"
        ) from exc

    env = make_phase4_smoke_env(phase2_output_dir, variant_id)
    obs, info = env.reset(seed=seed)
    initial_mask = env.action_masks()
    masking_supported = bool(is_masking_supported(env))
    if not masking_supported:
        raise ValueError("Phase 7 env does not expose action_masks for sb3-contrib")

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

    obs, _ = env.reset(seed=seed)
    action_masks = env.action_masks()
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=action_masks,
    )
    predicted_action = int(action)
    predicted_action_valid = bool(action_masks[predicted_action])

    return {
        "phase": "phase7_maskableppo_smoke",
        "phase2_output_dir": str(Path(phase2_output_dir)),
        "variant_id": str(info["variant_id"]),
        "seed": int(seed),
        "n_blocks": int(info["n_blocks"]),
        "n_features": int(info["n_features"]),
        "observation_shape": int(obs.shape[0]),
        "action_space_n": int(env.action_space.n),
        "reward_mode": str(info["reward_mode"]),
        "masking_supported": masking_supported,
        "initial_valid_actions": int(initial_mask.sum()),
        "learn_timesteps": int(total_timesteps),
        "device": "cpu",
        "predicted_action": predicted_action,
        "predicted_action_valid": predicted_action_valid,
        "selected_block_id": (
            str(env.block_ids[predicted_action]) if predicted_action_valid else None
        ),
        "dependencies": _dependency_metadata(),
        "claim_boundary": PHASE7_CLAIM_BOUNDARY,
    }


def write_phase7_maskableppo_artifact(
    summary: Mapping[str, object],
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "phase7_maskableppo_smoke.json"
    artifact_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


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
