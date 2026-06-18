# Phase 27 Results Package

This folder records the current Phase 27 diagnostic evidence for Paper11. It
compares the existing 1024-step and 4096-step Phase 26 B0/B1 learned-policy
result sets and explains why the current evidence should remain negative
evidence rather than a positive GeoFM-performance claim.

## Files

- `01_phase27_stability_diagnosis.md`: reviewer-facing interpretation of the
  Phase 27 budget and tile-seed stability diagnosis.

## Reproduction Link

Run the diagnosis from the repository root after the macOS Phase 26 artifacts
exist:

```powershell
python experiments\phase27_stability_diagnosis\run_phase27_stability_diagnosis.py --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main\phase26_analysis\phase26_main_comparison.json --phase26-comparison-json experiments\phase26_main_experiment\outputs\macos_main_4096\phase26_analysis\phase26_main_comparison.json --output-dir experiments\phase27_stability_diagnosis\outputs\macos_1024_vs_4096
```

The generated artifacts are ignored by Git, but the interpretation here is
based on the current generated Phase 27 summary.

## Claim Boundary

Phase 27 is diagnostic only. It does not run training, does not enable
suitability reward, does not test B2/B3, and does not support cross-region
transfer or final manuscript claims.
