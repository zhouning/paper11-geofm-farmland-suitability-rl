# Phase 1 Results Package

This folder records the current Phase 1 evidence for Paper11. It translates the executable Bishan GeoFM baseline into reviewer-facing interpretation and defines the next experiment matrix needed before manuscript-level performance claims.

## Files

- `01_phase1_result_interpretation.md`: what the Phase 1 output shows, what it does not show, and how it should be cited in the manuscript workflow.
- `02_next_experiment_matrix.md`: the next Paper11 experiments required to test representation, suitability reward, ablation, and transfer claims.

## Reproduction Link

Run the Phase 1 baseline from the repository root:

```powershell
python experiments\phase1_bishan_baseline\run_phase1.py
```

The command writes generated artifacts under:

```text
experiments/phase1_bishan_baseline/outputs/
```

These artifacts are ignored by Git, but the result interpretation here is based on the current generated `summary.json`.

## Claim Boundary

Phase 1 verifies that the repository can produce deterministic region-level GeoFM features and a bounded `suitability_proxy` from the included Bishan AlphaEarth sample. It does not demonstrate DRL planning performance, farmland layout improvement, soil quality, fertility, or irrigation access.
