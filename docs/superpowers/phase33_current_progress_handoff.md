# Phase 33 Current Progress Handoff

## Current State

Paper11 remains experiment-first. Phase 33 completed the bounded `3 tiles x 3
seeds` matched robustness check over the Phase 30 normalized-B1 branch. Phase
34 and Phase 35 then localized the result with spatial-composition and
action-overlap diagnostics. Phase 36 added a read-only suitability-proxy
validation gate before any B2/B3 reward work, Phase 37 added a read-only
decision-alignment audit over Phase 34, Phase 35, and Phase 36 artifacts,
and Phase 38 added a leakage-aware proxy-rebuild diagnostic over the real
Phase 2, Phase 8, and Phase 30 feature tables. Phase 39 added an
independent-label audit over the real Phase 2 feature table and found that
independent label inputs are still missing.
Phase 40 then added a hard independent-label go/no-go gate and confirmed
that the current no-registry real run remains blocked.

## Phase 39 Final Merge / Window-Close Save

Phase 39 has been fast-forward merged back to local `main`. The temporary
`phase39-independent-label-audit` worktree was removed and the local feature
branch was deleted after merge.

Latest Phase 39 merged head before this handoff-doc save:

```text
237b789 fix: close Phase 39 label audit review gaps
```

Current Phase 39 status remains `independent_label_inputs_missing`: the real
Bishan Phase 2 table still contains only leakage-risk labels or source metadata,
not a defensible independent non-DLTB validation label. Phase 38 therefore
cannot yet be rerun with a stronger non-leakage label, and B2/B3 suitability
reward remains blocked.

Post-merge verification on `main`:

```text
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_main_merge -p no:cacheprovider
28 passed, 84 warnings

python scripts\smoke_check.py
Paper11 smoke check passed.

git diff --check HEAD~1..HEAD
clean
```

The review fix closed two Phase 39 contract gaps:

- `valid` is no longer treated as an evaluation split; only `test`, `eval`,
  `evaluation`, `validation`, and `val` count as eval roles.
- label registries now support CSV and JSON, including JSON object lists and
  objects keyed by `label_column`, with the same provenance validation as CSV.

Repository state when the Phase 36 continuation started:

- branch: `main`
- remote relation: `main...origin/main`
- latest commit before Phase 36 continuation: `6f94881 docs: record Phase 33 full-grid budget result`
- starting tracked edits: none

## Window-Close Save State

Before closing this window, the Phase 36 implementation was committed and
pushed as:

```text
461683a feat: add Phase 36 suitability proxy validation
```

The current Phase 36 conclusion remains `proxy_signal_not_supported`, so the
current B2/B3 suitability reward remains blocked. The next continuation should
start from the Phase 37 proxy-rebuild or decision-alignment branch described
below. Local generated outputs under `experiments/**/outputs/` remain ignored
and should not be committed.

## Phase 37 Window-Close Save State

Phase 37 has been merged back to `main` locally. The temporary
`phase37-decision-alignment` worktree was removed and the local feature branch
was deleted after the fast-forward merge.

Latest Phase 37 commits on `main`:

```text
f82a9db fix: align Phase 37 artifact contract
0930280 docs: record Phase 37 decision-alignment result
005cb81 feat: add Phase 37 decision-alignment runner
```

Post-merge verification on `main`:

```text
python -m pytest tests\test_phase37_decision_alignment.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase37_main_merge -p no:cacheprovider
13 passed

python scripts\smoke_check.py
Paper11 smoke check passed.
```

Current Phase 37 real-run status is `decision_alignment_not_supported` with
`54` joined case rows and `37` summary rows. Phase 36 remains
`proxy_signal_not_supported`, so B2/B3 suitability reward remains blocked.

## Phase 38 Proxy-Rebuild Continuation

Phase 38 completed the leakage-aware proxy-rebuild diagnostic on the
`phase38-proxy-rebuild` worktree. It rebuilds lightweight diagnostic proxy
classifiers over existing feature tables, writes full rebuilt proxy scores to
CSV, and keeps B2/B3 reward blocked unless non-leakage labels clear the control
gate.

Latest Phase 38 commits on the worktree branch:

```text
25db7cf fix: cap Phase 38 JSON score preview
d07810c feat: add Phase 38 proxy rebuild runner
ce2dce3 fix: harden Phase 38 artifact writer
f56ef0f feat: write Phase 38 proxy rebuild artifacts
41ae5f1 fix: prevent leakage label promotion
```

Local ignored Phase 38 output:

```text
experiments/phase38_proxy_rebuild/outputs/real_bishan
```

Artifacts:

```text
phase38_label_summary.csv
phase38_model_summary.csv
phase38_rebuilt_proxy_scores.csv
phase38_proxy_rebuild.json
phase38_proxy_rebuild.md
```

Status and row counts:

- status: `proxy_rebuild_diagnostic_only`
- block rows: `64984`
- feature families: `11`
- label summary rows: `3`
- model rows: `99`
- rebuilt proxy score rows: `6433416`
- JSON score preview rows: `20`, with full rows in `phase38_rebuilt_proxy_scores.csv`

Interpretation: Phase 38 remains diagnostic only: either evaluated labels were
explicit leakage risks or GeoFM-derived rebuilt proxies did not clear the
control thresholds.

All current real labels are `explicit_label_leakage_risk`:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`

Therefore B2/B3 suitability reward remains blocked. Phase 38 does not support
a bounded B2/B3 smoke claim with the current label set, does not run PPO, does
not alter rewards, and does not support final planning-performance claims.

Verification and real run:

```text
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_json_contract_after_review -p no:cacheprovider
15 passed, 84 warnings

python experiments\phase38_proxy_rebuild\run_phase38_proxy_rebuild.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --phase8-output-dir experiments\phase8_ablation_controls\outputs\real_bishan_controls --normalized-controls-dir experiments\phase30_normalized_b1_ablation\outputs\real_bishan_4096_incremental\derived_normalized_controls --output-dir experiments\phase38_proxy_rebuild\outputs\real_bishan --label-columns current_farmland_label,farmland_or_orchard_label,low_slope_farmland_label --model-families logistic_elastic_net,random_forest,hist_gradient_boosting
Phase 38 proxy-rebuild status: proxy_rebuild_diagnostic_only

python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_final2 -p no:cacheprovider
15 passed, 84 warnings

python scripts\smoke_check.py
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```
## Phase 38 Window-Close Save State

Phase 38 has been merged back to `main` locally. The temporary
`phase38-proxy-rebuild` worktree was removed and the local feature branch was
deleted after the fast-forward merge.

Latest Phase 38 commits on `main` before this handoff save:

```text
96dc762 docs: record Phase 38 proxy rebuild result
25db7cf fix: cap Phase 38 JSON score preview
d07810c feat: add Phase 38 proxy rebuild runner
ce2dce3 fix: harden Phase 38 artifact writer
f56ef0f feat: write Phase 38 proxy rebuild artifacts
```

Post-merge verification on `main`:

```text
python -m pytest tests\test_phase38_proxy_rebuild.py tests\test_phase36_suitability_proxy_validation.py -q --basetemp=.pytest_tmp_phase38_main_merge -p no:cacheprovider
15 passed, 84 warnings

python scripts\smoke_check.py
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```

Current Phase 38 real-run status is `proxy_rebuild_diagnostic_only` with
`64984` block rows, `99` model rows, and `6433416` rebuilt proxy score rows.
All current real labels remain `explicit_label_leakage_risk`, so B2/B3
suitability reward remains blocked.

## Phase 39 Independent-Label Audit Continuation

Phase 39 completed the independent-label audit on the
`phase39-independent-label-audit` worktree. It inventories available default
label-like columns in the Phase 2 real feature table, writes a registry
template for future independent label sources, and keeps Phase 38 rerun work
blocked until a stronger non-leakage label exists.

Local ignored Phase 39 output:

```text
experiments/phase39_independent_label_audit/outputs/real_bishan
```

Artifacts:

```text
phase39_label_inventory.csv
phase39_label_readiness.csv
phase39_label_registry_template.csv
phase39_independent_label_audit.json
phase39_independent_label_audit.md
```

Status and row counts:

- status: `independent_label_inputs_missing`
- block rows: `64984`
- label inventory rows: `7`
- label readiness rows: `7`
- registry rows: `0`

Audited default labels:

- `current_farmland_label`
- `farmland_or_orchard_label`
- `low_slope_farmland_label`
- `source_bsm`
- `source_category`
- `source_dlbm`
- `source_dlmc`

Verification and real run:

```text
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan
Phase 39 independent-label audit status: independent_label_inputs_missing
```

Docs-branch verification:

```text
$pattern = 'T' + 'BD|TO' + 'DO|REPLACE_' + 'WITH|PLACE' + 'HOLDER'
rg -n $pattern README.md paper\phase28_results\README.md paper\phase28_results\13_phase39_independent_label_audit.md docs\superpowers\phase33_current_progress_handoff.md
no matches
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_docs_final -p no:cacheprovider
24 passed, 84 warnings
python scripts\smoke_check.py
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```

Interpretation: Phase 39 does not run PPO, alter rewards, enable B2/B3, prove
agronomic validity, or support planning-performance claims. Phase 38 cannot yet
be rerun with a stronger non-leakage label and B2/B3 remains blocked.

Next step: obtain or register defensible independent labels, then rerun the
independent-label audit before any Phase 38 proxy-rebuild rerun.

## Phase 40 Independent-Label Gate

Phase 40 adds the hard go/no-go gate requested by the reviewer critique. It
does not try to rescue B2/B3 by adding another ordinary diagnostic stage.
Instead, it requires a registered independent, non-leakage label before any
Phase 38 rerun or suitability-reward smoke.

Current real no-registry status:

```text
independent_label_inputs_missing
```

Real run counts:

- feature rows: `64,984`
- registry rows: `0`
- label gate rows: `0`

Decision: B2/B3 remains blocked. The next scientifically valid action is to
provide an external independent label registry and rerun Phase 40. Without
that, Paper11 should be framed as a reproducible diagnostic platform with
negative suitability-reward readiness evidence.

Phase 40 does not run PPO, alter rewards, enable B2/B3, prove suitability, or
support planning-performance claims.

## What Was Run This Window

The earlier Phase 33 state only had a positive local pilot:

- `tile_r002_c003`, `seed=0`, `5120` train timesteps
- local status: `budget_closes_compressed_gap`

This window completed the remaining bounded coverage for the current Phase 30
held-out set:

- tiles: `tile_r002_c003`, `tile_r005_c003`, `tile_r005_c004`
- seeds: `0,1,2`
- train timesteps: `5120`
- eval max steps: `8`
- trained variants: `N1Z`, `N1ZR`
- reused matched comparators: `B1`, `D4P8`, `D4P16` from the existing `4096`
  Phase 28/30 artifacts

The single-run outputs are under:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_seed2_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_seed2_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed0_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed1_matched
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_seed2_matched
```

The aggregate outputs are under:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r002_c003_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c003_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tile_r005_c004_3seed_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_tiles_r002c003_r005c003_6run_aggregate
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate
```

## Main Result

The current full bounded Phase 33 status is:

```text
budget_not_explanatory
```

The full `3 tiles x 3 seeds` aggregate is:

```text
experiments/phase33_budget_robustness/outputs/real_bishan_5120_pilot_3tiles_3seeds_9run_aggregate/phase33_budget_robustness.json
```

Key focal aggregate deltas:

| Gap | 4096 mean delta | 5120 mean delta | 5120 positive count |
|---|---:|---:|---:|
| `N1Z - B1` | `0.3008963657` | `-0.1185345091` | `5 / 9` |
| `N1Z - D4P16` | `-0.3402976578` | `-0.7597285327` | `4 / 9` |
| `N1Z - D4P8` | `-0.0759554690` | `-0.4953863438` | `3 / 9` |
| `N1ZR - B1` | `0.2266106233` | `-0.0246975093` | `5 / 9` |
| `N1ZR - D4P16` | `-0.4145834002` | `-0.6658915329` | `3 / 9` |
| `N1ZR - D4P8` | `-0.1502412114` | `-0.4015493440` | `4 / 9` |

Tilewise aggregate statuses:

- `tile_r002_c003`: `budget_closes_compressed_gap`
- `tile_r005_c003`: `budget_not_explanatory`
- `tile_r005_c004`: `budget_not_explanatory`

Interpretation: the positive `tile_r002_c003 seed0` pilot was real but local.
It does not support a general budget-rescue claim for normalized B1. The
broader bounded result weakens the Phase 29 optimization-budget explanation and
pushes the next experimental priority toward spatial/action diagnostics and
suitability-proxy validation.

## Phase 34 Case-Map Continuation

Phase 34 completed the first next-step priority from the previous handoff. It
is read-only and uses the nine existing Phase 33 matched pilot directories; it
does not run new policy training.

Local ignored Phase 34 output:

```text
experiments/phase34_case_map_diagnostics/outputs/real_bishan_5120_phase33_9run
```

Artifacts:

```text
phase34_case_map_cases.csv
phase34_case_map_blocks.csv
phase34_case_map_diagnostics.json
phase34_case_map_diagnostics.md
```

Status and row counts:

- status: `case_map_diagnostics_ready`
- case rows: `54`
- case-map block rows: `857`
- selected Phase 2 feature rows: `294`
- Phase 33 matched directories: `9`

Main diagnostic result:

- positive Phase 33 cases: `24`, mean higher delta `0.5822555613`, mean
  base-reward gap `0.0727819453`, all classified as
  `variant_selects_higher_base_reward_blocks`
- failure Phase 33 cases: `30`, mean higher delta `-1.2055407806`, mean
  base-reward gap `-0.1506925976`, all classified as
  `variant_selects_lower_base_reward_blocks`
- `tile_r002_c003`: `15` higher / `3` lower spatial-pattern split
- `tile_r005_c003`: `0` higher / `18` lower; clearest negative spatial
  counterexample
- `tile_r005_c004`: `9` higher / `9` lower

Interpretation: Phase 34 supports the existing bounded Phase 33 conclusion and
makes the spatial split concrete. Positive cases are associated with selected
block sets that have higher deterministic base-planning reward than their
comparators; negative cases are associated with lower base-reward selected
block sets. This still does not support a general budget-rescue claim for
normalized B1.

## Phase 35 Action-Overlap Continuation

Phase 35 completed the second next-step priority: extending action-overlap
diagnostics to the completed Phase 33 `5120` matched outputs. It is
read-only and uses the same nine Phase 33 matched pilot directories as Phase
34; it does not run new policy training.

Local ignored Phase 35 output:

```text
experiments/phase35_phase33_action_overlap_diagnostics/outputs/real_bishan_5120_phase33_9run
```

Artifacts:

```text
phase35_action_overlap_cases.csv
phase35_action_overlap_steps.csv
phase35_action_overlap_diagnostics.json
phase35_action_overlap_diagnostics.md
```

Status and row counts:

- status: `action_overlap_diagnostics_ready`
- case rows: `54`
- step rows: `432`
- Phase 33 matched directories: `9`

Main diagnostic result:

- `disjoint_positive_gap`: `20` cases
- `partial_overlap_positive_gap`: `4` cases
- `disjoint_negative_gap`: `27` cases
- `partial_overlap_negative_gap`: `3` cases
- positive Phase 33 cases: mean selected-block Jaccard `0.0111111111`,
  mean summary reward gap `0.5822555613`, nonzero-overlap cases `4 / 24`
- failure Phase 33 cases: mean selected-block Jaccard `0.0066666667`,
  mean summary reward gap `-1.2055407806`, nonzero-overlap cases `3 / 30`
- `tile_r005_c003`: mean Jaccard `0.0074074074`, mean summary reward gap
  `-1.3403609350`, still the clearest negative tile counterexample

Interpretation: Phase 35 supports the Phase 34 spatial-composition diagnosis.
The Phase 33 positive and negative outcomes are not mainly same-block
reorderings; they are almost always nearly disjoint selected-block sets.
Comparator step rewards are not fully available in the current matched
artifacts, so Phase 35 uses summary selected-block order for comparators and
keeps full step-reward trajectory claims out of scope.

## Phase 36 Suitability-Proxy Validation Continuation

Phase 36 completed the third next-step priority: suitability-proxy validation
before any B2/B3 reward integration. It is read-only and consumes the existing
Phase 11/Phase 2 real feature tables, Phase 8 representation controls, and
Phase 30 normalized controls. It does not run policy training and does not
enable suitability reward.

Local ignored Phase 36 output:

```text
experiments/phase36_suitability_proxy_validation/outputs/real_bishan
```

Artifacts:

```text
phase36_label_summary.csv
phase36_model_summary.csv
phase36_suitability_proxy_validation.json
phase36_suitability_proxy_validation.md
```

Status and row counts:

- status: `proxy_signal_not_supported`
- input rows: `64,984`
- train/evaluation rows: `45,460 / 19,524`
- feature families evaluated: `11`
- usable labels: `current_farmland_label`, `farmland_or_orchard_label`,
  `low_slope_farmland_label`
- label leakage flag: all three labels are `explicit_label_leakage_risk`

Main diagnostic result:

- `explicit_only` reaches ROC AUC, average precision, and balanced accuracy of
  `1.0` for all three labels because the weak labels are DLTB/slope-derived and
  encoded by explicit planning features.
- `suitability_proxy_only` is near random: ROC AUC `0.5081982029` for current
  farmland, `0.5124973908` for farmland/orchard, and `0.4979564572` for
  low-slope farmland.
- `raw_geofm_only` has one weak positive association with
  `low_slope_farmland_label` (ROC AUC `0.6490064144`, AP `0.1695914498`), but
  this does not clear the leakage boundary.

Interpretation: Phase 36 keeps the current B2/B3 suitability reward blocked.
The next branch should obtain independent labels or rebuild a supervised or
semi-supervised suitability proxy under spatial held-out validation. More PPO
budget is not the right next move until that proxy gate improves.


## Phase 37 Decision-Alignment Continuation

Phase 37 completed the decision-alignment audit over the existing Phase 34
case-map outputs, Phase 35 action-overlap cases, and Phase 36 suitability-proxy
diagnosis. It is read-only, does not run policy training, and does not alter
reward logic.

Local ignored Phase 37 output:

```text
experiments/phase37_decision_alignment/outputs/real_bishan_5120_phase33_9run
```

Artifacts:

```text
phase37_decision_alignment_cases.csv
phase37_decision_alignment_summary.csv
phase37_decision_alignment.json
phase37_decision_alignment.md
```

Status and row counts:

- status: `decision_alignment_not_supported`
- Phase 36 status: `proxy_signal_not_supported`
- Phase 34 case rows: `54`
- Phase 34 selected-block rows: `857`
- Phase 35 case rows: `54`
- Phase 37 joined case rows: `54`
- Phase 37 summary rows: `37`

Main diagnostic result:

- all joined cases: mean summary reward gap `-0.4109646286`, mean
  suitability-proxy gap `-0.0066127380`, mean low-slope farmland-label gap
  `-0.0486111111`, and `39 / 54` proxy-or-label alignment cases
- positive Phase 33 cases: `24` cases, mean summary reward gap
  `0.5822555613`, and `22 / 24` proxy-or-label alignment cases
- failure Phase 33 cases: `30` cases, mean summary reward gap
  `-1.2055407806`, and `17 / 30` proxy-or-label alignment cases

Interpretation: Phase 37 did not find conservative decision-alignment support under the failure-subgroup gate: at least one failure-case group also showed a positive status-gate signal.
The positive cases often align with available proxy or weak environmental
diagnostics, but many failure cases also show proxy-or-label alignment while
remaining strongly negative. This keeps Phase 37 diagnostic-only and does not
support a proxy-rebuild success claim. Phase 36 still blocks B2/B3 suitability
reward because its status remains `proxy_signal_not_supported`.

## Tracked Docs Updated

Updated in the Phase 33 window:

```text
README.md
paper/phase26_results/02_next_experiment_matrix.md
paper/phase28_results/07_phase33_budget_robustness.md
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 34 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/08_phase34_case_map_diagnostics.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 35 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/09_phase35_phase33_action_overlap_diagnostics.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 36 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/10_phase36_suitability_proxy_validation.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
docs/superpowers/specs/2026-06-25-phase36-suitability-proxy-validation-design.md
docs/superpowers/plans/2026-06-25-phase36-suitability-proxy-validation.md
src/paper11_geofm/phase36_suitability_proxy_validation.py
experiments/phase36_suitability_proxy_validation/run_phase36_suitability_proxy_validation.py
tests/test_phase36_suitability_proxy_validation.py
```

Updated in the Phase 37 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/11_phase37_decision_alignment.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 38 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/12_phase38_proxy_rebuild.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 39 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/13_phase39_independent_label_audit.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Updated in the Phase 40 continuation:

```text
README.md
paper/phase28_results/README.md
paper/phase28_results/14_phase40_independent_label_gate.md
paper/submission/01_ijaeog_submission_readiness.md
paper/submission/02_draft_titles_highlights_declarations.md
reproducibility/FILE_MANIFEST.tsv
docs/superpowers/phase33_current_progress_handoff.md
```

Do not describe Phase 33 as generally `budget_closes_compressed_gap`. That is
only true for the `tile_r002_c003` three-seed aggregate, not for the complete
bounded aggregate.

Do not describe Phase 37 as supporting B2/B3, suitability reward, reward
changes, policy training, or final planning-performance claims. Its real status
is `decision_alignment_not_supported`.

Do not describe Phase 38 as supporting B2/B3, suitability reward, reward
changes, policy training, agronomic validity, or final planning-performance
claims. Its real status is `proxy_rebuild_diagnostic_only`.

Do not describe Phase 39 as supporting B2/B3, suitability reward, reward
changes, policy training, agronomic validity, or final planning-performance
claims. Its real status is `independent_label_inputs_missing`, and Phase 38
cannot yet be rerun with a stronger non-leakage label.

Do not describe Phase 40 as supporting B2/B3, suitability reward, reward
changes, policy training, agronomic validity, or final planning-performance
claims. Its real status is `independent_label_inputs_missing`, and it is a
hard stop for the suitability-reward route until an external independent
label registry passes the gate.

## Verification Run

Fresh verification completed after the full 9-run aggregate:

```powershell
python -m pytest tests\test_phase33_budget_robustness.py tests\test_phase32_action_order_diagnostics.py tests\test_phase30_normalized_b1_ablation.py -q --basetemp=.pytest_tmp_phase33_full_cover -p no:cacheprovider
```
Result:

```text
13 passed in 0.61s
```

Smoke check:

```powershell
python scripts\smoke_check.py
```

Result:

```text
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```

Fresh Phase 36 verification completed after the suitability-proxy continuation:

```powershell
python -m pytest tests\test_phase36_suitability_proxy_validation.py tests\test_phase9_proxy_validation.py tests\test_phase10_reward_readiness.py -q --basetemp=.pytest_tmp_phase36_final -p no:cacheprovider
```

Result:

```text
15 passed in 10.23s
```

Smoke check rerun after the Phase 36 continuation:

```powershell
python scripts\smoke_check.py
```

Result:

```text
Paper11 smoke check passed.
Sample years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Embedding shape: (67, 70, 64)
```

## Next Experimental Priority

Do not move directly to manuscript claims. Next work should stay experiment-first:

1. Obtain or construct defensible non-DLTB labels, such as high-standard
   farmland, retention/productivity, irrigation/water-proximity, external
   soil/yield proxy, or another independent label source.
2. Re-run the proxy-rebuild gate with those independent labels under spatial
   held-out validation, then test whether Phase 33/34/35 selected block sets
   are ordered by the rebuilt proxy.
3. Use Phase 34/35 as spatial and action-overlap diagnostic support only; do
   not convert them into manuscript-level performance claims.
4. Treat `8192` only as a later, chunked/resumable experiment if the rebuilt
   proxy branch produces a reason to test longer budgets. The previous `8192`
   attempts did not finish within the execution window and are not evidence.

## Useful Commands For Next Window

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
Get-Content docs\superpowers\phase33_current_progress_handoff.md
Get-Content paper\phase28_results\13_phase39_independent_label_audit.md
Get-Content paper\phase28_results\12_phase38_proxy_rebuild.md
Get-ChildItem -Force experiments\phase39_independent_label_audit\outputs\real_bishan
Get-Content experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_independent_label_audit.md
Get-Content experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_independent_label_audit.json
Import-Csv experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_label_inventory.csv | Format-Table -AutoSize
Import-Csv experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_label_readiness.csv | Format-Table -AutoSize
Import-Csv experiments\phase39_independent_label_audit\outputs\real_bishan\phase39_label_registry_template.csv | Measure-Object
python experiments\phase39_independent_label_audit\run_phase39_independent_label_audit.py --phase2-output-dir experiments\phase11_bishan_dltb_real\outputs\phase2_real --output-dir experiments\phase39_independent_label_audit\outputs\real_bishan
Get-Content paper\phase28_results\07_phase33_budget_robustness.md
Get-ChildItem -Force experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run
Get-ChildItem -Force experiments\phase35_phase33_action_overlap_diagnostics\outputs\real_bishan_5120_phase33_9run
Import-Csv experiments\phase34_case_map_diagnostics\outputs\real_bishan_5120_phase33_9run\phase34_case_map_cases.csv | Group-Object eval_tile_id,spatial_pattern | Select-Object Name,Count
Import-Csv experiments\phase35_phase33_action_overlap_diagnostics\outputs\real_bishan_5120_phase33_9run\phase35_action_overlap_cases.csv | Group-Object eval_tile_id,action_overlap_pattern | Select-Object Name,Count
python -m pytest tests\test_phase39_independent_label_audit.py tests\test_phase38_proxy_rebuild.py -q --basetemp=.pytest_tmp_phase39_resume -p no:cacheprovider
python scripts\smoke_check.py
```
