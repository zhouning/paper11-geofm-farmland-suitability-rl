from __future__ import annotations

from collections.abc import Mapping, Sequence


EMBEDDING_COLUMNS = [f"embedding_mean_{idx:02d}" for idx in range(64)]
EXPLICIT_FEATURE_COLUMNS = [f"explicit_feature_{idx:02d}" for idx in range(17)]
SUITABILITY_COLUMNS = ["suitability_proxy"]


VARIANT_DEFINITIONS = {
    "B0": {
        "description": "Explicit planning feature baseline.",
        "state_groups": ["explicit_planning_features"],
        "reward": "base_planning_reward",
        "required_columns": EXPLICIT_FEATURE_COLUMNS,
    },
    "B1": {
        "description": "Explicit planning features plus raw GeoFM embeddings.",
        "state_groups": ["explicit_planning_features", "geofm_embedding"],
        "reward": "base_planning_reward",
        "required_columns": EXPLICIT_FEATURE_COLUMNS + EMBEDDING_COLUMNS,
    },
    "B2": {
        "description": "Explicit planning features plus the suitability proxy.",
        "state_groups": ["explicit_planning_features", "suitability_proxy"],
        "reward": "base_plus_suitability_reward",
        "required_columns": EXPLICIT_FEATURE_COLUMNS + SUITABILITY_COLUMNS,
    },
    "B3": {
        "description": (
            "Explicit planning features, raw GeoFM embeddings, and the suitability "
            "proxy."
        ),
        "state_groups": [
            "explicit_planning_features",
            "geofm_embedding",
            "suitability_proxy",
        ],
        "reward": "base_plus_suitability_reward",
        "required_columns": EXPLICIT_FEATURE_COLUMNS
        + EMBEDDING_COLUMNS
        + SUITABILITY_COLUMNS,
    },
}

VARIANT_CLAIM_BOUNDARY = (
    "These variants define feature-table readiness for later DRL experiments; "
    "they do not report trained-policy performance or prove planning improvement."
)


def summarize_phase2_readiness(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Report which Paper11 B0/B1/B2/B3 feature variants are table-ready."""
    has_embedding = _rows_have_columns(rows, EMBEDDING_COLUMNS)
    has_suitability = _rows_have_columns(rows, ["suitability_proxy"])
    has_explicit = _rows_have_columns(rows, EXPLICIT_FEATURE_COLUMNS)

    return {
        "B0": _readiness(
            ready=has_explicit,
            missing=_missing(has_explicit, has_embedding, has_suitability),
        ),
        "B1": _readiness(
            ready=has_explicit and has_embedding,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_embedding=True,
            ),
        ),
        "B2": _readiness(
            ready=has_explicit and has_suitability,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_suitability=True,
            ),
        ),
        "B3": _readiness(
            ready=has_explicit and has_embedding and has_suitability,
            missing=_missing(
                has_explicit,
                has_embedding,
                has_suitability,
                require_embedding=True,
                require_suitability=True,
            ),
        ),
    }


def build_phase2_variant_manifest(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe B0/B1/B2/B3 feature contracts for later DRL experiments."""
    readiness = summarize_phase2_readiness(rows)
    variants = {}
    for variant_id, definition in VARIANT_DEFINITIONS.items():
        variants[variant_id] = {
            "description": definition["description"],
            "state_groups": list(definition["state_groups"]),
            "reward": definition["reward"],
            "required_columns": list(definition["required_columns"]),
            "ready": readiness[variant_id]["ready"],
            "missing": list(readiness[variant_id]["missing"]),
        }

    return {
        "claim_boundary": VARIANT_CLAIM_BOUNDARY,
        "variants": variants,
    }


def _rows_have_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> bool:
    if not rows:
        return False
    return all(all(column in row for column in columns) for row in rows)


def _readiness(ready: bool, missing: list[str]) -> dict[str, object]:
    return {"ready": ready, "missing": missing}


def _missing(
    has_explicit: bool,
    has_embedding: bool,
    has_suitability: bool,
    require_embedding: bool = False,
    require_suitability: bool = False,
) -> list[str]:
    missing: list[str] = []
    if not has_explicit:
        missing.append("explicit_features_incomplete")
    if require_embedding and not has_embedding:
        missing.append("geofm_embedding_columns_missing")
    if require_suitability and not has_suitability:
        missing.append("suitability_proxy_missing")
    return missing
