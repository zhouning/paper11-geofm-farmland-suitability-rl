# Phase 28 Results Package

This folder records the current Phase 28 representation-control evidence for
Paper11. Phase 28 tests whether the raw B1 GeoFM representation is
distinguishable from explicit-only B0 and from random, shuffled, and
PCA-compressed representation controls under the same padded held-out Bishan
base-reward protocol.

## Files

- `01_phase28_representation_control_diagnosis.md`: reviewer-facing
  interpretation of the 1024-step and 4096-step Phase 28 representation-control
  diagnostics.

## Reproduction Link

Run the diagnostic from the repository root after the Phase 11/13 real Bishan
outputs and Phase 8 D-control feature tables exist:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\real_bishan_4096
```

The generated artifacts are ignored by Git, but the interpretation here is
based on the current generated Phase 28 1024-step and 4096-step comparison
JSON files.

## Claim Boundary

Phase 28 is diagnostic only. It does not enable suitability reward, does not
test B2/B3, does not test cross-region transfer, and does not support final
submission-level planning-performance claims.
