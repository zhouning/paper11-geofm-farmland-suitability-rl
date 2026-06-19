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
- `02_phase28_compression_diagnosis.md`: read-only follow-up diagnosis of why
  D4P8/D4P16 can exceed raw B1 in the current 4096-step run.

## Reproduction Link

Run the diagnostic from the repository root after the Phase 11/13 real Bishan
outputs and Phase 8 D-control feature tables exist:

```powershell
python experiments\phase28_representation_controls\run_phase28_representation_controls.py --mode run-and-analyze --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --tile-index-csv experiments\phase13_tiled_real_contract\outputs\real_bishan\phase13_tile_index.csv --variants B0,B1,D2,D3,D4P8,D4P16 --total-timesteps 4096 --eval-max-steps 8 --seeds 0,1,2 --max-eval-tiles 3 --output-dir experiments\phase28_representation_controls\outputs\real_bishan_4096
```

The generated artifacts are ignored by Git, but the interpretation here is
based on the current generated Phase 28 1024-step and 4096-step comparison
JSON files.

The compression diagnosis is read-only. It recomputes selected-block overlap,
base-reward component means, and PCA variance summaries from the generated
4096-step Phase 28 summary CSV plus the existing Phase 2/8 feature tables. It
does not run additional policy training.

Run the reproducible compression follow-up from the repository root:

```powershell
python experiments\phase28_compression_diagnosis\run_phase28_compression_diagnosis.py --summary-csv experiments\phase28_representation_controls\outputs\real_bishan_4096\phase28_representation_control_summary.csv --phase2-b1-features-csv experiments\phase11_bishan_dltb_real\outputs\phase2_real\variant_B1_features.csv --d4p8-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P8_features.csv --d4p16-features-csv experiments\phase8_ablation_controls\outputs\real_bishan_controls\variant_D4P16_features.csv --output-dir experiments\phase28_compression_diagnosis\outputs\real_bishan_4096
```

Expected local read-only artifacts:

- `phase28_compression_overlap.csv`
- `phase28_compression_reward_components.csv`
- `phase28_compression_diagnosis.json`
- `phase28_compression_diagnosis.md`

## Claim Boundary

Phase 28 is diagnostic only. It does not enable suitability reward, does not
test B2/B3, does not test cross-region transfer, and does not support final
submission-level planning-performance claims.
