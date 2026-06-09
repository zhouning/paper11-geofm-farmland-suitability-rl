# Phase 3 DRL Input Adapter Design

## Goal

Build a lightweight, deterministic adapter that turns Phase 2 ready-only
variant CSV exports into numeric matrices suitable for later Paper11 DRL
experiments.

This phase does not train a policy, evaluate action quality, or claim planning
improvement. It only validates and exposes the input contract between Phase 2
feature assembly and later DRL environments.

## Scope

Phase 3 reads a Phase 2 output directory containing:

- `experiment_variants.json`
- one or more ready-only feature tables such as `variant_B3_features.csv`

The adapter loads one requested variant (`B0`, `B1`, `B2`, or `B3`) and returns a
structured object with:

- `variant_id`
- `block_ids`
- `feature_columns`
- `state_matrix`
- `reward_mode`
- `state_groups`
- `source_table`

The returned matrix must be `numpy.float32` with shape
`(n_blocks, n_features)`. Column order must exactly follow the manifest
`required_columns` for the requested variant.

## Non-Goals

The phase intentionally avoids:

- creating a new Gymnasium environment;
- modifying legacy Paper3/Paper4/Paper8 training code;
- simulating block transitions or action masks;
- running Stable-Baselines3 training;
- reporting DRL performance or agronomic validity.

## Public API

Create `src/paper11_geofm/drl_inputs.py` with:

```python
@dataclass(frozen=True)
class VariantInput:
    variant_id: str
    block_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    state_matrix: np.ndarray
    reward_mode: str
    state_groups: tuple[str, ...]
    source_table: Path


def load_variant_input(phase2_output_dir: Path | str, variant_id: str) -> VariantInput:
    ...
```

The function normalizes variant IDs to uppercase and rejects unknown variants.

## Validation Rules

The loader should fail with clear `ValueError` or `FileNotFoundError` messages
when:

- `experiment_variants.json` is missing;
- the requested variant is absent;
- the requested variant is not ready;
- `feature_table` is null or empty;
- the referenced CSV is missing;
- the CSV lacks `block_id` or any manifest `required_columns`;
- any required feature value is blank or non-numeric.

The loader should preserve `block_id` order from the CSV and should not sort
rows. Duplicate block IDs are rejected because later action spaces need a
one-to-one mapping between row index and planning unit.

## CLI

Create:

```text
experiments/phase3_drl_input_adapter/inspect_variant_inputs.py
```

The command:

```powershell
python experiments\phase3_drl_input_adapter\inspect_variant_inputs.py --phase2-output-dir path\to\outputs --variant B3
```

prints a compact summary:

- variant ID;
- source table;
- row count;
- feature count;
- matrix shape;
- reward mode;
- state groups;
- claim boundary.

The command exits non-zero for invalid or incomplete variants.

## Documentation

Update `README.md`, `reproducibility/REPRODUCTION_GUIDE.md`, and
`reproducibility/FILE_MANIFEST.tsv` to describe Phase 3 as a reviewer-facing
input-contract inspection step.

## Test Strategy

Use TDD. Add tests that:

- generate Phase 2 fixture outputs and load `B3`;
- verify B0/B1/B2/B3 feature dimensions are 17/81/18/82;
- verify row order and manifest column order are preserved;
- verify incomplete default Phase 2 outputs fail clearly;
- verify duplicate block IDs fail;
- verify non-numeric required values fail;
- verify the CLI prints the expected summary.

The verification suite remains CPU-only and offline.
