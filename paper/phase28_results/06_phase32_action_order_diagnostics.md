# Phase 32 Action-Order Diagnostics

## One-Sentence Argument

Phase 32 is a read-only follow-up to Phase 31 that explains selected failure
cases by comparing focal/comparator action order, cumulative reward trajectory,
and selected-block composition within the local tile block pool.

## What This Phase Does

Phase 32 reads:

```text
experiments/phase31_case_diagnostics/outputs/.../phase31_ranked_cases.csv
experiments/phase30_normalized_b1_ablation/outputs/.../phase30_normalized_b1_traces.json
experiments/phase28_representation_controls/outputs/.../phase28_representation_control_traces.json
experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv
experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv
```

It writes:

```text
phase32_step_alignment.csv
phase32_case_summary.csv
phase32_tile_pool_composition.csv
phase32_action_order_diagnostics.json
phase32_action_order_diagnostics.md
```

## Reproduction Command

```text
python experiments/phase32_action_order_diagnostics/run_phase32_action_order_diagnostics.py --ranked-cases-csv experiments/phase31_case_diagnostics/outputs/real_bishan_4096/phase31_ranked_cases.csv --focal-traces-json experiments/phase30_normalized_b1_ablation/outputs/real_bishan_4096_incremental/phase30_normalized_b1_traces.json --comparator-traces-json experiments/phase28_representation_controls/outputs/real_bishan_4096/phase28_representation_control_traces.json --phase2-features-csv experiments/phase11_bishan_dltb_real/outputs/phase2_real/block_geofm_features.csv --tile-index-csv experiments/phase13_tiled_real_contract/outputs/real_bishan/phase13_tile_index.csv --output-dir experiments/phase32_action_order_diagnostics/outputs/real_bishan_4096 --top-k 6
```

## Current Real-Run Snapshot

The current real Bishan `top_k=6` run splits into three descriptive/positive
cases and three negative/failure-side cases.

The strongest negative high-overlap case is `tile_r002_c003|2|N1ZR|B1`.
Phase 32 ranks it as `overlap_with_negative_gap` with selected-block Jaccard
`0.7777777778`, mean shared-step displacement `0.4285714286`, and cumulative
reward gap `-0.3638798802`. This matters because the failure is not explained
by a low-overlap selected set alone.

For this case, the step alignment shows:

- step 1: both variants select `dltb_5372912`, cumulative gap stays `0.0`;
- steps 2-3: `dltb_3871952` and `dltb_822221` are swapped, but the gap returns
  to `0.0` after step 3;
- step 5: N1ZR selects `dltb_819715` while B1 selects `dltb_5316709`, pushing
  the cumulative gap to `-0.2468150215`;
- step 6: N1ZR selects `dltb_822310` while B1 selects `dltb_819715`, pushing
  the cumulative gap to `-0.3638798802`;
- steps 7-8: both variants match again, but the gap does not recover.

The local tile-pool summary for `tile_r002_c003|2|N1ZR|B1` also shows that the
tile itself has low low-slope-farmland prevalence (`0.0443327239`), while both
selected sets have low-slope-farmland mean `0.0`. Even under that shared local
constraint, the focal and comparator differ in average base reward
(`-0.0369255798` vs `0.0085594052`), which is consistent with one or two late
block substitutions dominating the reward gap.

The other negative cases are mostly low-overlap selections rather than
high-overlap reorderings:

- `tile_r005_c003|0|N1ZR|B1`: Jaccard `0.2307692308`, cumulative gap
  `-0.4077216427`, first-step gap `-0.6853969041`;
- `tile_r005_c003|1|N1ZR|B1`: Jaccard `0.3333333333`, cumulative gap
  `-0.322342556`.

This keeps the current experiment conclusion narrow: some N1ZR losses versus
B1 come from clearly different selected sets, but at least one important
failure case survives even when the two selected sets overlap heavily, and the
remaining loss is concentrated in late action choices.

## Claim Boundary

Phase 32 is a read-only action-order diagnostic over existing Phase 28, Phase
30, and Phase 31 artifacts. It does not run new policy training, does not alter
rewards, does not enable suitability reward, does not test B2/B3, and does not
support final submission-level planning-performance claims.
