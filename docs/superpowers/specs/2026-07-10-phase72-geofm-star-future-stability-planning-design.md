# Phase 72 GeoFM-STaR Future-Stability Planning Design

## Goal

Build a new evidence chain that tests whether temporally truncated AlphaEarth
representations provide independent, forward-looking information about future
farmland persistence and whether calibrated uncertainty from that model can
improve constrained spatial planning on future observed outcomes.

Phase 72 is not a textual revision of the Phase 46 submission package. It is a
scientific redesign motivated by the failure of the current deterministic base
reward to expose a GeoFM-specific target and by the matched-dimension results in
Phases 59 and 62. Formal manuscript revision begins only after the prediction
and planning gates in this design have been evaluated.

## One-Sentence Argument Target

In multi-region farmland planning, we test whether a temporally truncated,
residual GeoFM survival model predicts one- and two-year farmland persistence
beyond explicit GIS and current land-cover baselines and whether its calibrated
lower-confidence-bound risk estimates improve future observed persistence under
base-planning and spatial-compactness constraints.

This sentence is a target argument, not a current result. It may be promoted to
a manuscript claim only if the confirmation gates below pass.

## Scientific Motivation

The current Paper11 `base_planning_reward` is almost completely aligned with
explicit planning variables. Phase 66 reports an explicit-feature proxy R2 of
approximately `0.9974`, whereas the GeoFM representation-only proxy R2 is about
`0.0295`. Better optimization of that target cannot, by itself, establish an
independent GeoFM contribution.

The original D4P8/D4P16 result is also confounded by input dimension and
conditioning. Phase 59 does not support D4 against same-dimension random and
shuffled controls, and Phase 62 reports negative D4-minus-random-orthonormal
projection means. Phase 63 and Phase 71 further show that architecture and
training-target changes improve decision performance without distinguishing D4
from B0 or D6.

The new task therefore changes the scientific target rather than continuing to
tune the same explicit-feature reward. It asks whether information available in
AlphaEarth through year `t` predicts independently observed land-cover outcomes
after `t`, then evaluates whether that information changes planning decisions in
a way that improves those future outcomes.

## Terminology Ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| GeoFM | geospatial foundation model | Use `GeoFM` after first expansion. |
| AlphaEarth | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` annual embedding product | Do not describe individual channels as soil, irrigation, fertility, or yield measurements. |
| GeoFM-STaR | spatiotemporal risk-aware GeoFM model | Working model name for Phase 72; use consistently in code, results, and the later manuscript. |
| explicit baseline | model using terrain, geometry, current land cover, and available planning attributes without GeoFM coordinates | Must be a strong baseline, not a deliberately weak comparator. |
| farmland persistence | a unit classified as farmland at time `t` remaining farmland at the specified future horizon | Keep distinct from agronomic suitability and policy success. |
| farmland conversion | a farmland unit at time `t` changing to a non-farmland class at the specified future horizon | Treat product-label change and manual-reference change separately. |
| one-year endpoint | farmland state at `t+1` conditional on farmland at `t` | Primary predictive endpoint. |
| two-year endpoint | farmland state at `t+2`, with continuous persistence reported separately when available | Secondary robustness endpoint. |
| LCB | lower confidence bound of calibrated persistence probability | Use as the conservative planning score. |
| product label | annual land-cover label from ESRI Global LULC or Dynamic World | Independent of DLTB/base reward, but not equivalent to manual ground truth. |
| confirmation set | region-year or spatial block not used for preprocessing, model selection, calibration, or threshold choice | Open only after the model and analysis choices are frozen. |

## Scope and Regions

The minimum region set is:

- Bishan, with existing annual AlphaEarth arrays and existing ESRI Global LULC
  2023/2024 same-grid product labels;
- Dongxing, with `134,369` DLTB units, a slope-enriched candidate source,
  sparse adjacency assets, and annual AlphaEarth arrays for 2017-2024.

The Dongxing static DLTB proxy labels are explicitly excluded as future
outcomes. The Paper13 label audits correctly classify them as circular and not
independent. Dongxing requires newly acquired annual product labels and an
audited temporal alignment before it can contribute outcome evidence.

A third region may be added as a final external confirmation set if a
compatible public annual land-cover product, AlphaEarth coverage, and stable
spatial units can be assembled. The third region must not be used to choose the
model or decision threshold.

## Data Contract

### Stable Prediction Units

The primary prediction analysis uses spatial units that remain stable across
years. Preferred units are a common product grid or fixed aggregation grid.
DLTB parcels are used for planning applications and parcel-level external
analysis only after their annual labels are derived from a documented spatial
aggregation rule.

The data adapter must emit one row per:

```text
region_id x unit_id x prediction_origin_year
```

Required fields include:

- stable unit identifier and region identifier;
- spatial-block identifier for buffered validation;
- prediction origin year `t` and outcome year;
- AlphaEarth coordinates for every available year from 2017 through `t`;
- explicit features available no later than `t`;
- current product-label class at `t`;
- one-year and, when defined, two-year outcomes;
- label source, source version, resolution, aggregation rule, and confidence;
- flags for mixed pixels, boundary uncertainty, missing years, and manual
  review status.

### Independent Outcome Labels

The primary product-label source is ESRI Global LULC or Dynamic World. Where
both products are available, the adapter retains source-specific labels and an
agreement flag rather than silently merging disagreements.

The primary binary cohort contains units classified as farmland at `t`:

```text
y_1y = 1 if farmland at t+1, else 0
y_2y = 1 if farmland at t+2, else 0
y_continuous_2y = 1 if farmland at both t+1 and t+2, else 0
```

Conversion type is an exploratory multiclass target. Product labels are not
described as observed policy outcomes, agronomic suitability, yield, or field
survey truth.

### Manual or High-Resolution Review

A stratified sample is drawn by region, year, model-confidence band, product
agreement, and predicted conversion risk. Review records must contain the image
source/date, reviewer decision, uncertainty, and provenance. These records
estimate product-label error and support sensitivity analyses; they do not
silently replace the full product-label dataset.

## Leakage-Free Validation Protocol

### Temporal Separation

All model inputs for origin year `t` stop at `t`. Labels after `t` are not used
to derive features. Early origin years form training data, an intermediate
origin year forms validation data, and the latest usable origin year is held
for final testing.

No test-year rows may be used to fit:

- PCA or any learned projection;
- means, scales, imputers, or feature filters;
- class weights selected from outcomes;
- model hyperparameters;
- probability calibration;
- decision thresholds or planning weights.

### Spatial Separation

Within-region evaluation uses spatial blocks with an exclusion buffer around
test blocks. Units in the buffer are removed from training. The buffer width is
recorded in physical units and tested in sensitivity analysis.

### Cross-Region Separation

The required transfer directions are:

- Bishan training and validation, Dongxing testing;
- Dongxing training and validation, Bishan testing.

Region-specific calibration may be reported only as a separate adaptation
experiment. It cannot replace zero-shot cross-region results.

### Confirmation Lock

Before opening the latest-year confirmation labels, the run writes a frozen
JSON contract containing:

- training, validation, and test regions/years/blocks;
- feature definitions;
- model hyperparameters;
- primary baselines and ablations;
- calibration method;
- primary metrics and hypotheses;
- planning budget and non-inferiority margins;
- random seeds and software versions.

The contract hash is included in all confirmation artifacts.

## Prediction Models

### Strong Explicit Baselines

The prediction package includes at least:

- regularized logistic regression;
- gradient-boosted trees;
- a small multilayer perceptron under the same split.

Inputs include terrain, geometry, current product-label class, available DLTB
attributes, and other information available by `t`. The strongest validation
baseline becomes the primary explicit comparator.

### GeoFM-STaR Architecture

GeoFM-STaR has four components.

1. **Explicit branch.** Encodes the same inputs as the strong explicit
   baseline.
2. **Temporal GeoFM branch.** Encodes AlphaEarth history through `t`, including
   current state, annual differences, long-term trend, and variability. The
   first implementation compares a compact gated temporal network with a small
   temporal Transformer and selects between them using validation data only.
3. **Residual discrete-time risk head.** Predicts GeoFM residual logits on top
   of the explicit baseline and emits one- and two-year hazards or persistence
   probabilities. The residual construction makes the independent information
   question explicit.
4. **Decision-aware auxiliary objective.** Adds a ranking loss on candidate
   units near the planning budget boundary so that probability improvements are
   tested where they can change planning decisions.

The training objective may include class-balanced prediction loss, temporal
consistency, domain-invariant regularization, and the decision-aware ranking
term. Each optional term requires an ablation and cannot be introduced after
the confirmation labels are opened.

### Calibration and Uncertainty

Calibration uses validation data only. Candidate methods are temperature
scaling for neural logits and isotonic regression for non-parametric baselines.
The selected method is frozen before testing.

Uncertainty is estimated with a small deep ensemble or an equivalent
validation-approved ensemble. The planning score is:

```text
LCB_i = calibrated_mean_persistence_i - k * predictive_std_i
```

The coefficient `k` is chosen on validation data under the planning objective
and then frozen.

## Required Prediction Baselines and Ablations

The confirmation matrix includes:

- explicit baseline;
- explicit baseline plus current land cover;
- single-year GeoFM;
- temporal mean GeoFM;
- full temporal GeoFM;
- temporal-order-shuffled GeoFM;
- spatially shuffled GeoFM;
- same-dimension random projection;
- GeoFM-STaR without the residual construction;
- GeoFM-STaR without domain-invariant regularization;
- GeoFM-STaR without the decision-aware ranking term.

All learned projections and standardization steps are fitted separately inside
each training fold. A control that sees more regions, years, or labels than the
primary model is invalid.

## Prediction Metrics and Gates

The primary endpoint is class-imbalance-aware one-year farmland persistence or
conversion performance. Primary metrics are:

- average precision / precision-recall AUC;
- Brier score;
- expected calibration error;
- decision-curve or budget-specific net benefit.

ROC AUC, F1, balanced accuracy, and two-year metrics are secondary. All metrics
include region-year or spatial-block uncertainty intervals rather than treating
every pixel or parcel as independent.

The prediction gate passes only if:

1. GeoFM-STaR improves at least two of average precision, Brier score, and
   calibration relative to the primary explicit baseline;
2. GeoFM-STaR exceeds temporal-order, spatial-shuffle, and random-projection
   controls;
3. improvement direction is consistent in both Bishan and Dongxing or the
   cross-region heterogeneity is explicitly resolved in a predeclared analysis;
4. the result survives label-source and spatial-buffer sensitivity analyses;
5. the confirmation set has complete expected coverage and no leakage audit
   failures.

Failure of this gate prevents a positive GeoFM prediction claim and prevents
the learned risk from becoming a manuscript planning reward.

## Planning Problem

### Candidate Score

For each eligible planning unit, the future-stability score is the calibrated
LCB of the required horizon. The score is not computed from embedding norm,
PCA variance, DLTB class, or base reward.

### Objectives and Constraints

The primary planner maximizes future-stability LCB subject to:

- total selected-area or action budget;
- action validity;
- minimum base-planning reward or a predeclared non-inferiority margin;
- slope and legal constraints already represented by Paper11;
- spatial compactness or maximum-fragment constraints;
- optional risk-concentration constraint.

A reproducible constrained ranking or combinatorial optimizer is the primary
decision method. The scalable set-policy/ranker is a learned planning method,
not the only planning baseline. PPO remains optional and cannot be the sole
evidence source.

### Planning Baselines

The comparison set includes:

- seeded random selection;
- explicit-rule greedy selection;
- current-land-cover ranking;
- explicit-risk-model planning;
- single-year GeoFM planning;
- GeoFM-STaR LCB planning;
- a deterministic upper-bound oracle using test outcomes, marked non-deployable;
- an integer, genetic, or NSGA-II comparator when computationally feasible.

### Planning Outcomes

Planning is evaluated on future product labels that were hidden from the
planner. Primary outputs are:

- observed future farmland persistence among selected units;
- lift over explicit-risk planning at the same budget;
- base-reward non-inferiority;
- compactness, connected-component count, and fragmentation;
- worst region-year result;
- calibration and uncertainty coverage of selected units;
- constrained net benefit and Pareto-frontier summaries.

The planning gate passes only when GeoFM-STaR improves future observed
persistence, respects the frozen base-reward and compactness margins, and is
not driven by a single region, year, or seed.

## Implementation Phases

### Phase A: Label Acquisition and Alignment

- inventory existing Bishan product labels and provenance;
- acquire the matching annual product labels for Dongxing;
- build stable region-year-unit sample tables;
- audit grid orientation, CRS, dates, aggregation, mixed pixels, and product
  agreement;
- create the stratified high-resolution/manual review frame.

Exit condition: an independent-label intake validator passes for both regions.

### Phase B: Low-Cost Information-Gain Screen

- train explicit logistic and boosted-tree baselines;
- add single-year, temporal-summary, shuffle, and random-projection features;
- run buffered spatial, temporal, and cross-region evaluation;
- quantify calibration and decision net benefit.

Exit condition: GeoFM shows independent information against strict controls. A
failed screen triggers label-resolution, horizon, and cohort audits before any
deep model is trained.

### Phase C: GeoFM-STaR

- implement the temporal encoder, residual risk head, domain regularizer,
  decision-aware ranking term, calibration, and ensemble uncertainty;
- run the full ablation matrix;
- freeze the confirmation contract and evaluate the held-out labels once.

Exit condition: the prediction gate passes.

### Phase D: Risk-Aware Planning

- implement constrained LCB planning and strong non-learning baselines;
- compare future persistence, base reward, compactness, and robustness;
- run budget and non-inferiority sensitivity analyses.

Exit condition: the planning gate passes.

### Phase E: External Confirmation

- evaluate an untouched region, origin year, or spatial block set;
- add a third region when compatible data can be obtained without changing the
  frozen primary model;
- run label-source and manual-review sensitivity analyses.

Exit condition: the main direction survives external confirmation.

### Phase F: Manuscript Reconstruction

Only after Phases C-D produce results does the formal manuscript change. The
new argument will be organized as:

1. future farmland persistence is a target not determined by the current base
   reward;
2. GeoFM-STaR predicts that target beyond explicit GIS under leakage-free
   temporal and spatial validation;
3. calibrated uncertainty changes constrained planning decisions;
4. selected units achieve better future observed persistence without violating
   planning constraints;
5. cross-region and control experiments bound the claim.

The revised package must add a study-region map, data timeline, model diagram,
calibration plot, cross-region comparison, planning maps, Pareto plots, complete
method details, immutable code version, data provenance, and final submission
metadata.

### Program Decomposition

This document is the master scientific design. It must not be converted into
one monolithic implementation plan. Execution is split into six separately
specified and accepted cycles:

1. `Phase 72A`: independent annual label acquisition, intake validation, and
   stable region-year-unit assembly;
2. `Phase 72B`: leakage-free low-cost information-gain screen;
3. `Phase 72C`: GeoFM-STaR training, calibration, uncertainty, and confirmation;
4. `Phase 72D`: constrained future-stability planning and planning baselines;
5. `Phase 72E`: untouched external confirmation and label-noise sensitivity;
6. `Phase 72F`: figures, claim registry, and formal manuscript reconstruction.

Each cycle receives its own specification, implementation plan, tests, real
run, result note, and gate decision. The next implementation cycle is Phase
72A only. A later cycle may be designed and implemented only after the previous
cycle has produced the required artifacts and its exit condition has been
evaluated.

## Expected Software Boundaries

Implementation should use focused modules with explicit contracts:

- label acquisition/intake and provenance validation;
- stable unit and region-year table assembly;
- leakage-free split and preprocessing bundle;
- low-cost baseline and information-gain screen;
- GeoFM-STaR model and training;
- calibration and uncertainty;
- constrained planning and baselines;
- statistical analysis and gate classification;
- figure/source-data export;
- manuscript claim registry and submission preflight.

Each module requires unit tests and a small fixture path before real-data runs.
New behavior follows red-green-refactor TDD.

## Result Status Model

Phase 72 uses conservative statuses:

- `label_inputs_not_ready`: independent aligned outcomes are unavailable;
- `geofm_information_not_supported`: strict low-cost controls show no
  independent GeoFM information;
- `geofm_temporal_prediction_supported`: the prediction gate passes but the
  planning gate has not passed;
- `geofm_planning_not_supported`: prediction passes but constrained planning
  does not improve future outcomes or violates non-inferiority margins;
- `geofm_future_stability_planning_supported`: both prediction and planning
  gates pass with complete confirmation coverage;
- `confirmation_incomplete`: the frozen confirmation contract is not fully
  evaluated.

Only `geofm_future_stability_planning_supported` permits the strongest target
manuscript argument.

## Exhaustion Criteria

The original future-aware GeoFM idea is considered unsupported under the
available evidence only after evaluating:

- at least two independent annual land-cover products where accessible;
- one-year and two-year endpoints;
- an explicit residual model and a temporal neural model;
- Bishan-to-Dongxing and Dongxing-to-Bishan transfer;
- buffered spatial and strict temporal splits;
- temporal shuffle, spatial shuffle, and same-dimension random controls;
- label-resolution, disagreement, and noise sensitivity;
- both prediction and constrained-planning outcomes.

Failure must be reported as a scientific result. These criteria do not permit
post hoc threshold changes, selective region removal, or metric switching to
manufacture a positive finding.

## Risks and Mitigations

### Product-Label Noise

Annual land-cover products may contain classification flicker. Use
source-specific outcomes, agreement masks, continuous-persistence endpoints,
and stratified high-resolution review.

### Resolution Mismatch

The current Dongxing AlphaEarth arrays are sampled at `500 m`, whereas land
cover and DLTB operate at finer scales. The primary prediction grid must avoid
pseudoreplication from assigning one embedding vector to many parcels. Cluster
uncertainty at the embedding-cell level and report grid-level primary results.

### Regional Domain Shift

Different land-use systems may reduce transfer. Report zero-shot transfer
first, use domain-invariant regularization only with an ablation, and separate
adaptation experiments from the primary claim.

### Rare Conversion Events

If conversion is rare, use average precision, calibrated class weighting, and
spatial-cluster uncertainty. Do not rely on accuracy or ROC AUC alone.

### Planning Metric Gaming

Do not evaluate the planner only on its own risk score. Use hidden future
product labels, fixed budgets, explicit non-inferiority constraints, and a
non-deployable oracle upper bound.

## Non-Goals

Phase 72 does not:

- claim that AlphaEarth directly measures soil, irrigation, fertility, yield,
  legal quality, or agronomic suitability;
- use Dongxing static DLTB proxy labels as future outcomes;
- reuse the Phase 8 whole-region PCA tables as leakage-free evidence;
- claim causal effects of planning actions on land conversion;
- treat product labels as manual ground truth;
- revise the formal submission before prediction and planning evidence exists;
- guarantee a positive GeoFM result.

## Completion Criteria for the Design Program

The program is complete when it has:

- an audited independent multi-region temporal label package;
- leakage-free reproducible prediction runs and strict controls;
- calibrated GeoFM-STaR risk and uncertainty outputs;
- constrained planning comparisons on hidden future outcomes;
- external confirmation or a documented confirmation limitation;
- a claim registry that maps every manuscript statement to an artifact;
- a revised formal manuscript whose title, abstract, methods, results,
  discussion, conclusion, figures, data availability, and code availability
  match the completed evidence.
