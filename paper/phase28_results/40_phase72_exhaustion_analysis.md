# Phase 72 Exhaustion Analysis

Status: `phase72_exhaustion_criteria_not_fully_evaluated`

Route decision: `phase72_route_closed_at_phase72b_gate`

## Purpose

This read-only analysis audits the completed Phase 72A, receipt-bound Phase
72B, separately frozen two-year endpoint evidence, the confirmation-frozen
explicit residual screen, and the compact temporal neural residual screen
against the exhaustion criteria in the Phase 72 master design. It separates
criteria that were evaluated and failed or mixed from criteria that remain
unresolved. It does not train Phase 72C, alter rewards, run planning, or modify
the formal manuscript.

## Integrity and Coverage

All nine artifacts bound by the official Phase 72B confirmation receipt, all
eight artifacts bound by the two-year receipt, all eight artifacts bound by
the explicit residual receipt, and all eight artifacts bound by the temporal
neural receipt matched their recorded SHA256 hashes. All four receipts matched
their canonical-JSON SHA256 sidecars. The analysis found zero integrity
blockers across 37 receipt and artifact checks.

The audited evidence contains:

- one independent annual land-cover product: ESRI Global LULC;
- two regions: Bishan and Dongxing;
- `31,627` Phase 72A temporal samples;
- assembled `1y`, `2y`, and `continuous_2y` label horizons;
- `560` manual-review-frame rows, with zero completed reviews;
- `153` Phase 72B metric rows across nine variants and two model families;
- `142` two-year metric rows across two endpoints and `142` frozen bundles;
- `123` explicit residual metric rows across three endpoints and `123` frozen
  bundles, including `84` residual bundles with valid cross-fit audits;
- `41` temporal neural metric rows and `41` frozen bundles, including `28`
  neural bundles with valid five-fold spatial cross-fit audits;
- three representation-control axes;
- two bidirectional transfer axes;
- 10 buffered spatial folds across two regions;
- zero confirmation blockers.

## Exhaustion-Criteria Audit

| Criterion | Status | Result |
| --- | --- | --- |
| Independent annual products | `data_gap` | One product was evaluated; the design asks for at least two where accessible. |
| One- and two-year endpoints | `evaluated_complete` | Phase 72B evaluated one year and the separate frozen screen evaluated both two-year endpoints. |
| Two-year prediction outcome gate | `evaluated_negative` | Both two-year endpoints returned `geofm_information_not_supported`. |
| Explicit residual model | `evaluated_mixed` | The one-year pooled residual passed practical/statistical/control gates, but failed transfer and spatial stability; both two-year residual endpoints were negative. |
| Temporal neural model | `evaluated_negative` | Pooled AP improved, but Brier/ECE, strict controls, both transfers, and fold-level spatial stability failed. |
| Bidirectional cross-region transfer | `evaluated_negative` | Both Bishan-to-Dongxing and Dongxing-to-Bishan transfer axes failed. |
| Buffered spatial validation | `evaluated_mixed` | All 10 folds were evaluable, but regional and fold directions were heterogeneous. |
| Temporal and representation controls | `evaluated_negative` | Temporal-order and spatial-shuffle controls failed the frozen gate. |
| Label disagreement/noise sensitivity | `not_evaluated` | Only one product was available and all 560 review decisions remain blank. |
| Prediction outcome gate | `evaluated_negative` | Official status remains `geofm_information_not_supported`. |
| Constrained planning outcomes | `not_evaluated` | No hidden-outcome constrained planning run exists. |

The one- and two-year receipt-bound low-cost screens and the temporal neural
screen can therefore be reported as negative, while the explicit residual
result must be reported as mixed. The broader claim that every future-aware
GeoFM design has been scientifically exhausted remains blocked because three
criteria are unresolved: a second independent annual product, label
disagreement/noise sensitivity, and constrained planning outcomes.

## Transition Decision

Do not begin Phase 72C. Preserve the Phase 72B, two-year, and temporal neural
negative gates, treat the explicit residual evidence as mixed rather than
stable support, and record the remaining product, label-noise, and planning
evidence as unresolved limitations. This decision does not permit post hoc
changes to thresholds, metrics, regions, seeds, or folds.

## Generated Artifacts

Ignored real-run artifacts are under:

```text
experiments/phase72_exhaustion_analysis/outputs/real_bishan_dongxing_residual_neural
```

The directory contains:

```text
phase72_exhaustion_criteria.csv
phase72_exhaustion_claim_boundary.csv
phase72_exhaustion_artifact_hashes.csv
phase72_exhaustion_analysis.json
phase72_exhaustion_analysis.md
```

## Verification

```text
focused temporal-neural and exhaustion tests: 27 passed
full repository: 599 passed, 84 existing sklearn warnings
smoke check: passed
git diff --check: passed
paper/submission/final/*: unchanged
```

## Reproduction

Run from the repository root:

```powershell
python experiments\phase72_exhaustion_analysis\run_phase72_exhaustion_analysis.py --phase72a-json experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_temporal_label_package.json --phase72a-summary-csv experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_package_summary.csv --phase72a-review-csv experiments\phase72a_temporal_label_package\outputs\bishan_dongxing_esri_2017_2024\phase72a_manual_review_frame.csv --phase72b-json experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_information_gain_screen.json --phase72b-protocol-json experiments\phase72b_geofm_information_gain_screen\phase72b_protocol.json --phase72b-metrics-csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_metrics.csv --phase72b-control-csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_control_comparison.csv --phase72b-transfer-csv experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_transfer_summary.csv --phase72b-receipt-json experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_confirmation_receipt.json --phase72b-receipt-sha256 experiments\phase72b_geofm_information_gain_screen\outputs\confirmation\phase72b_confirmation_receipt.sha256 --phase72b-confirmation-dir experiments\phase72b_geofm_information_gain_screen\outputs\confirmation --phase72-two-year-json experiments\phase72_two_year_endpoint_screen\outputs\confirmation_fixed_configs\phase72_two_year_endpoint_screen.json --phase72-two-year-receipt-json experiments\phase72_two_year_endpoint_screen\outputs\confirmation_fixed_configs\phase72_two_year_confirmation_receipt.json --phase72-two-year-receipt-sha256 experiments\phase72_two_year_endpoint_screen\outputs\confirmation_fixed_configs\phase72_two_year_confirmation_receipt.sha256 --phase72-two-year-confirmation-dir experiments\phase72_two_year_endpoint_screen\outputs\confirmation_fixed_configs --phase72-residual-json experiments\phase72_explicit_residual_screen\outputs\confirmation_v2\phase72_explicit_residual_screen.json --phase72-residual-receipt-json experiments\phase72_explicit_residual_screen\outputs\confirmation_v2\phase72_explicit_residual_confirmation_receipt.json --phase72-residual-receipt-sha256 experiments\phase72_explicit_residual_screen\outputs\confirmation_v2\phase72_explicit_residual_confirmation_receipt.sha256 --phase72-residual-confirmation-dir experiments\phase72_explicit_residual_screen\outputs\confirmation_v2 --phase72-neural-json experiments\phase72_temporal_neural_screen\outputs\confirmation\phase72_temporal_neural_screen.json --phase72-neural-receipt-json experiments\phase72_temporal_neural_screen\outputs\confirmation\phase72_temporal_neural_confirmation_receipt.json --phase72-neural-receipt-sha256 experiments\phase72_temporal_neural_screen\outputs\confirmation\phase72_temporal_neural_confirmation_receipt.sha256 --phase72-neural-confirmation-dir experiments\phase72_temporal_neural_screen\outputs\confirmation --output-dir experiments\phase72_exhaustion_analysis\outputs\real_bishan_dongxing_residual_neural
```

## Claim Boundary

Phase 72 exhaustion analysis is a read-only audit of the completed Phase 72A,
Phase 72B, and separately frozen exhaustion evidence. It does not train Phase
72C, run planning, alter rewards, revise the formal manuscript, or establish a
complete scientific exhaustion of every future-aware GeoFM design.
