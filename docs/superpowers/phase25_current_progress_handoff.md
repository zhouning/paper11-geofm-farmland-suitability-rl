# Phase 25 Current Progress Handoff

Last updated: 2026-06-12.

## Repository State

- Repository: `D:\test\paper11-geofm-farmland-suitability-rl`
- Branch: `main`
- Remote: `origin/main`
- Latest pushed commit before this handoff: `04ed419 Add Phase 25 padded held-out policy design`
- Working tree was clean before creating this handoff.

## Completed Work

Phase 24 was completed, tested, committed, and pushed:

- Commit: `43f9c5e Add Phase 24 IJAEOG evidence package`
- Purpose: synthesize Phase 22/23 pilot outputs into IJAEOG claim-readiness
  artifacts.
- Verified locally: smoke check passed; full pytest reported `141 passed`.

Phase 25 design was completed, committed, and pushed:

- Commit: `04ed419 Add Phase 25 padded held-out policy design`
- Spec file:
  `docs/superpowers/specs/2026-06-12-phase25-padded-heldout-policy-design.md`
- Purpose: implement the first learned-policy held-out tile experiment with a
  padded variable-size MaskablePPO environment.

## Current Decision

Proceed with Phase 25:

```text
padded variable-size MaskablePPO held-out tile experiment
```

The goal is to move beyond Phase 20/23 same-tile pilots and produce learned
policy evidence on distinct held-out Bishan tiles for B0/B1 under the
deterministic base planning reward.

## Important Claim Boundaries

Phase 25 should only test B0/B1 under `base_planning_reward`.

Do not enable B2/B3 or suitability reward yet because Phase 10 reports:

```text
status: not_ready_for_suitability_reward
recommendation: do_not_enable_suitability_reward
```

Phase 25 may support held-out Bishan tile learned-policy evidence if results
are positive. It must not claim:

- suitability-reward benefit;
- B2/B3 full-model superiority;
- cross-region transfer beyond Bishan held-out tiles;
- final IJAEOG submission-level planning performance.

## Windows Hardware Assessment

Checked in the local Windows workspace:

- OS: Windows 11 Home China, 64-bit;
- CPU: Intel Core Ultra 9 185H, 16 cores, 22 logical processors;
- RAM: 31.43 GiB total, about 14.32 GiB available during the check;
- workspace disk: `D:` had about 88.82 GiB free;
- GPU: Intel Arc Graphics; no NVIDIA CUDA device detected;
- Python: `C:\Python314\python.exe`;
- RL stack installed: `gymnasium 1.2.3`, `stable-baselines3 2.7.1`,
  `sb3-contrib 2.7.1`, `torch 2.10.0+xpu`;
- observed direct `torch` import took about 83 seconds once.

Conclusion:

- Windows is suitable for implementation, unit tests, schema checks, and short
  real Bishan smoke runs such as `total_timesteps <= 128`.
- Google Colab Pro+ should be used for the main Phase 25 multi-seed,
  multi-held-out-tile long-budget run.
- macOS is secondary unless it has clearly better PyTorch/MPS performance.

## Next Step

Use `superpowers:writing-plans` to create:

```text
docs/superpowers/plans/2026-06-12-phase25-padded-heldout-policy.md
```

The plan should use TDD and cover:

1. padded environment unit tests;
2. Phase 25 module implementation;
3. CLI runner under `experiments/phase25_padded_heldout_policy/`;
4. comparison JSON aggregation;
5. README, reproduction guide, file manifest, and IJAEOG evidence updates;
6. Windows short real Bishan smoke run;
7. Colab Pro+ main-run command recipe.

After the plan is written, execute it with `superpowers:executing-plans` and
`superpowers:test-driven-development`.

## Suggested Initial Commands Next Session

```powershell
cd D:\test\paper11-geofm-farmland-suitability-rl
git status --short --branch
git log --oneline --max-count=5
Get-Content docs\superpowers\specs\2026-06-12-phase25-padded-heldout-policy-design.md
Get-Content docs\superpowers\phase25_current_progress_handoff.md
```

Then write the Phase 25 implementation plan before editing code.
