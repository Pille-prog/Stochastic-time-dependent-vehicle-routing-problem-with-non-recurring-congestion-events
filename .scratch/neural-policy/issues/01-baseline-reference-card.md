# 01 — The baseline reference card

**What to build:** Run the linear `MonteCarloPolicy` on the real Chengdu
dataset and freeze its **per-seed** costs to disk as the fixed opponent every
later run is measured against. Retire the scalar `static_policy_mean_cost`.

This is the first full training run of the repaired lab. The number does not
exist yet — `.scratch/simulator-correctness/spec.md` deliberately left it out
("the first experiment of the repaired lab, not part of repairing it"). Nothing
downstream of this ticket can be measured until it does.

**Blocked by:** —

**Status:** resolved

## Why per-seed and not a mean

`static_policy_mean_cost: 2490` is one hardcoded YAML scalar feeding one red
line on `training_plot.png`. A scalar only supports comparing *means*, and with
this cost distribution's variance two means hide a real 3% effect. The
per-seed vector supports the **paired** comparison the acceptance contract
needs (spec.md, "Why the paired comparison is valid") — same demand, same
congestion schedule, seed by seed.

## Work

- [x] Run the baseline at **three training budgets**: 100, 500, 2000 episodes,
      each with the evaluation cadence scaled per spec.md's frozen parameters
      (~every 50 episodes, *not* `test_frequency: 10` — at 2000 episodes that
      would cost 10 000 evaluation episodes, more than the training).
      `learning_rate` was tuned for 100 episodes; longer runs may improve or
      diverge. **Measure it, do not assume it.**
- [x] For each budget, run the existing `final_test` over `test_seeds`
      (100..153) at every `test_action_count` (2, 10, 20, 30, 40, 50).
- [x] Freeze a **reference card** artifact: for the winning cell (best budget ×
      best action count) the per-seed `total_cost` vector over `test_seeds`,
      plus the per-seed vector over `evaluation_seeds` (which is what the live
      training report in ticket 07 prints against). Include the config
      snapshot, the `best_w`, wall-clock, and which cell won and by how much.
- [x] Retire `static_policy_mean_cost` from `ExperimentConfig` and from every
      shipped YAML. `write_training_plot` takes the reference card instead.
      **`test_config_sweep.py` and `test_experiment_config.py` both touch this
      field** — check before deleting.
- [x] Record the wall-clock per phase, so ticket 03's transformer measurement
      has something to be a multiple of.

## Acceptance

- [x] The reference card exists, is committed, and names its winning cell.
- [x] The three budgets' results are reported side by side — including if a
      longer budget made the baseline *worse*, which is a real possibility and
      a result in itself.
- [x] Predicted self-golden diff: **zero.** This ticket runs the existing code
      and deletes an unused-by-the-simulator config scalar. If it moves a float
      in the self-golden, something else changed.

## Comments

**2026-07-31, resolved.** Built and ran `scripts/run_baseline_reference_card.py`
against the real Chengdu dataset (`experiments/chengdu/config.yaml`, warm world
cache reused from a prior real-data run — 36-45s per load rather than the ~21m
cold estimate). Committed artifact: `experiments/chengdu/reference_card.json`.

**Correction to the ticket's own text**: `test_config_sweep.py` does not
reference `static_policy_mean_cost` directly — it loads `ExperimentConfig` via
the shared `tests/fixtures/chengdu_mini/config.yaml` fixture, so the actual
touch point was that YAML (which `test_trainer_smoke.py` also shares), not a
code reference in the test file itself. `test_experiment_config.py` does
reference it directly, as the ticket said.

**Side-by-side results, all 18 (budget x test_action_count) cells, mean
total_cost over the 50 `test_seeds`:**

| action_count | budget 100 | budget 500 | budget 2000 |
|---|---|---|---|
| 2  | 3458.4 | 3564.2 | 3564.2 |
| 10 | 3485.9 | 3542.7 | 3542.7 |
| 20 | 3489.3 | 3657.9 | 3657.9 |
| 30 | 3429.4 | 3713.0 | 3713.0 |
| 40 | **3384.8 (winner)** | 3523.6 | 3523.6 |
| 50 | 3523.4 | 3602.6 | 3602.6 |

**Winning cell: budget=100, test_action_count=40, mean_cost=3384.8184**
(margin over the next-best cell, budget=100/action_count=30 at 3429.3737:
44.5553). Budget 100 wins **every** action count outright over budgets 500 and
2000 — the ticket's flagged possibility ("a longer budget made the baseline
*worse*") is what happened, unambiguously, not a marginal effect: budget 100's
own worst cell (3523.4 @2) still beats budget 500/2000's own best cell
(3523.6 @40) by a hair, and its best cell beats their best by 3.94%
((3523.6 - 3384.8) / 3523.6).

**Mechanism, verified in the log, not just inferred**: budgets 500 and 2000
are **bit-identical to each other** on every final-test cell, because both
select the exact same `best_w` — the evaluation-block minimum (mean cost
2420.6396) is reached at episode 400 and never beaten again through episode
2000. The full evaluation-block trajectory (`sweep.log`, budget 2000's run)
bottoms at episode 400 and then rises almost monotonically from episode 650
(2439.4) to episode 2000 (2807.8, +16% over the minimum) — the tuned
`learning_rate` (1e-5, warm-up 1e-6) **diverges** past roughly episode 600 on
this config, exactly the risk the ticket's own Work item flagged
("longer runs may improve or diverge").

**Methodological finding for whoever builds ticket 07/09** (same "pick best W
via `evaluation_seeds`, verdict-test on `test_seeds`" protocol this ticket
used): the eval-selected optimum does not generalize here. Episode 400's W
scores *better* on `evaluation_seeds` than episode 50's W (2420.64 vs 2483.24,
-2.5%) but *worse* on the held-out `test_seeds` verdict set (3523.6 vs 3384.8,
+4.1%) — a real train/eval-set generalization gap on this baseline, not a bug
to fix here, but worth watching for when the transformer's own best-checkpoint
selection is built.

**Wall-clock per phase** (winning budget 100, recorded in the reference card's
`wall_clock_seconds`; all three budgets in `sweep.log`):

| budget | world_load | train | train s/ep | final_test (300 ep) | final_test s/ep |
|---|---|---|---|---|---|
| 100  | 36.0s | 313.1s | 3.131 | 428.5s | 1.428 |
| 500  | 37.8s | 1568.7s | 3.137 | 421.0s | 1.403 |
| 2000 | 44.7s | 5936.8s | 2.968 | 464.6s | 1.549 |

Total wall-clock across the three budgets: 9251.2s (2h34m11s).

**Self-golden**: verified exactly zero diff (`tests/test_self_golden.py`,
6/6 passed unchanged) — as predicted, since nothing in this ticket touches the
linear baseline's execution path.

**New code**: `ReferenceCard` (`src/stdvrp/training/reference_card.py`, TDD,
11 unit tests) — per-seed `test_seed_costs`/`evaluation_seed_costs`, `best_w`,
config snapshot, `wall_clock_seconds`, JSON save/load round trip.
`Trainer.run`/`write_training_plot` take `reference_card: ReferenceCard | None`
in place of the scalar. `scripts/run_baseline_reference_card.py` drives the
sweep and picks the winning cell; it needed a `--data-dir` override not
originally anticipated — running from a git worktree (this branch's standard
concurrent-session isolation), the config's relative `data_dir: ../../..`
resolves against the worktree's nesting depth under `.claude/worktrees/`
rather than the repo root, so it no longer reaches the "Mega city" data
folder. The override keeps the *committed* config YAMLs untouched (still
relative, still correct for a normal checkout) and only patches the path this
run itself loads from; the reference card's own `config` snapshot uses the
portable (relative) config, not the absolute override, so the committed
artifact carries no machine-specific path.
