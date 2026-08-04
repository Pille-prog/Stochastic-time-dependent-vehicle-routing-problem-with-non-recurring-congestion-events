# 01 — Diff the legacy's W trajectory against ours, episode by episode

**Status:** resolved

**What to build:** The first comparison in this effort that is not two black-box
cost totals. Capture the legacy monolith's W after each of ~200 training
episodes, capture ours over the same episode count, and diff the 24 components.

## Why this and not another sweep

Everything measured so far compares *episode cost*, which is one number summing
a whole trajectory. Under that measure the search has already exhausted its
cheap moves: `spec.md` records five hypotheses ruled out by measurement (the
objective function, the cost rates, world hardness, the clock discretization,
B10) and seven more ruled out by source audit (reward indexing, `U_t`
accumulation, gradient sign, lr warm-up, snapshot fields, feature count,
`action_features` symmetry). What is left is a residual that shows up as **W
rotating into a bad direction while ‖W‖ keeps growing** — and no cost total can
say *which component* rotates.

The legacy's W stays effective across 500 episodes (its cost is flat in
[2 798, 4 316]); ours does not (‖W‖ 5 460 → 21 760, cost oscillating to
40 077). Twenty-four numbers per episode, side by side, is the measurement that
localises that. It is also cheap: no new machinery.

## The machinery already exists

`scripts/capture_golden_master.py` (generic-stdvrp-refactor ticket 04) imports
the monolith **unmodified** — cwd-redirection onto the data folder plus a
builtins shim — and its `run_capture` already loops training episodes and
appends `copy.deepcopy(w)` to a `w_trajectory`. It also pickles the pristine
post-`__init__` world, which turns the legacy's ~28-minute cold load (88 speed
files plus a pure-Python parse of the 907 MB `all_shortest_paths.csv`) into
~32 seconds warm. **The cache does not exist on this machine yet**
(`%LOCALAPPDATA%/stdvrp/golden_world_cache.pkl`): budget one cold run.

Ours comes from `scripts/legacy_fidelity_bench.py`, which already prints W and
can be asked for it per block.

## Two things that will bite

1. **Legacy training is nondeterministic by construction.** `policy.__init__`
   (monolith 1677-78) creates two *unseeded* `random.Random()` instances used by
   `select_epsilon_greedy_action_train`. Ticket 04 solved this at driver level —
   it seeds them per episode right after constructing the policy, without
   touching the legacy file. Reuse that convention (`train_exploration_seed_offset`
   / `train_repair_seed_offset`, offsets 10M/20M).
2. **Same seed integers, different worlds.** Training seeds start at 1000 on
   both sides and evaluation seeds are 100000-100049 on both, but ticket 13
   (ADR-0001 phase 2) replaced the legacy's two shared global streams (Mersenne
   Twister) with four independent PCG64 streams. Seed 1000 draws a *different*
   client set, congestion and velocity field on each side. Measured over the 50
   evaluation seeds: legacy mean 148.94 Clients / 5.78 vehicles, ours 147.18 /
   5.76 — same distribution, different realisations. So **do not** expect
   per-episode equality; compare the *shape* of each component's trajectory
   (sign, growth rate, which components saturate), not its values.

   If per-episode equality is wanted, that is a separate and larger ticket:
   feed the legacy's demand into our runner (the legacy already has
   `ClientGenerator.write_clients_to_file`, monolith 1580) and freeze
   congestion and velocities on both sides.

## Done when

- [x] A legacy W trajectory over ~200 training episodes at
      `config_linear_congestion_10k.yaml`'s parameters (lr 1e-3, ε 0.1,
      congestion 0.1-0.4, duration 120, 150 clients, TW 60), stored as JSON
- [x] The same for the current code, clip included, same episode count
- [x] A per-component comparison: for each of the 24 weights, both trajectories'
      growth over the run, and specifically **which components diverge in ours
      and stay bounded in the legacy**
- [x] The answer written into `spec.md`: either a named component (and therefore
      a named feature) that misbehaves, or an explicit statement that all 24
      diverge together, which would point at the target rather than the features

## Leads worth carrying in

- Both the clip-only and full-fidelity arms spike catastrophically in the **same
  block, episodes 251-300** (40 077 and 38 306). Same training seeds. If one
  episode injects the rotation, it is in that window — logging ‖ΔW‖ per episode
  would find it in one run.
- In that block ‖W‖ *falls* (6 071 → 5 268) while cost rises to 38 306. Whatever
  it is, it is not step size.
- `general[13]` is a live regressor here and structurally 0 in **both** legacy
  sources — but reverting it measured worse, once, noisily. If the per-component
  diff shows W[13] misbehaving, that resolves the tension `spec.md` records.

## Answer

### First, a correction the diff forced out

Every Chengdu config in this repo set `time_window_spread: 60`; the reference
runs used **150** (`capture_golden_master.py`'s `diff_TW`, and the `Tw150` in the
legacy's own result filenames). The configs are now 150 —
`issues/04-retire-the-tw60-measurements.md` carries the fallout.

It is not cosmetic. `diff_TW` sets each Client's window width *and* the latest
minute a window can open, so 60 is strictly harder, and correcting it moves the
two sides in opposite directions: the legacy's `‖W‖` after 200 episodes goes
3 732 -> 2 895 while ours goes 7 502 -> 15 754. TW 60 was masking half the gap.

The legacy's own numbers say which is right: at TW 150 its training cost is mean
3 534 / median 2 695, inside the [2 798, 4 316] band `spec.md` records for it and
matching its stored result file for lr 0.001 / ε 0 / lc 0.1 / uc 0.4 /
duration 120 (mean 4 182). At TW 60 it comes out at 6 671, outside it.

Numbers below are at **TW 150**. The TW 60 pair is kept
(`w_diff_200.md`, `*_w_trajectory.json`) and says the same thing with smaller
numbers, so the finding is not an artifact of either setting.

### The finding

**All 22 components the legacy trains are inflated here — none is bounded.**
That is this ticket's second branch: there is no named feature to blame, so the
fault is in the target or the conditioning of the update, not in any one
regressor. Full table in `../w_diff_tw150.md`; narrative in `../spec.md` under
"Inside W".

Peak `|W[i]|` over 200 episodes, ours against the legacy's: median 4.84x, range
2.64x - 13.81x. Most inflated are `distance_cost/100` (13.8x),
`earliness_cost/60` (12.3x) and `earliness_bin1` (9.9x); least inflated are
`earliness_bin0` (2.6x), `mean_earliness_diff` (3.0x) and `delay_cost/60` (3.1x).
(At TW 60: median 3.97x, range 2.22x - 9.79x.)

### On the leads carried in

- **`general[13]`.** The diff does show `W[13]` as a real divergence — peak 3 653
  here against a structural 0 there, and the feature averages 0.242 per update,
  a substantial share of our +20% `‖X‖`. But it carries a fraction of a percent
  of the final `‖W‖` while 22 other components are inflated 2.6-13.8x, so it is
  not the cause. That resolves the tension `spec.md` recorded without making the
  B10 revert the fix.
- **The episode-251-300 spike.** No single episode injects it, and that block is
  not special. Checked where the lead pointed — `repo_w_trajectory_500.json`, the
  TW 60 500-episode run, which is the only trajectory here that reaches those
  episodes. Its largest `‖ΔW‖` in 251-300 is 16 832 at episode 288, **5th of
  500**, and the block's *mean* `‖ΔW‖` (1 995) is *below* the run's overall mean
  (2 473). The top of the table is spread right across the run — episodes 154,
  232, 355, 475, 288 — and at TW 150 the largest lands at episode 43 instead.
  Big steps are the steady state, not an event: `‖ΔW‖` is 17 951 at its worst
  against a legacy maximum of 1 844 (its episode 2). Whatever made the
  episode-251-300 *cost* block spike, it is not a `‖ΔW‖` outlier.
- **"Not step size."** Confirmed, and sharpened. Decomposing the update over 25
  episodes (wrapping `actualize_W` / `learn` on both sides, each instrumented run
  reproducing its uninstrumented `‖W‖` exactly), no input is off by more than
  1.74x — `T` 377.8 vs 413.4, mean `‖X‖` 1.803 vs 2.170, mean `|residual|`
  1 380 vs 2 398 — and our steps are *less* coherent (0.868 vs 0.784). Yet
  `‖ΔW‖` comes out 4.68x and `‖W‖` after 25 episodes 9.16x (1 658 vs 15 189).

### What is actually different

Stability, not magnitude. The legacy's `‖W‖` rises to 4 249 by episode 50 and
then comes back down (3 134 at 100, 2 796 at 150, 2 895 at 200). Ours never
settles: 15 189 at 25, 11 875 at 50, 7 910 at 100, 12 784 at 150, 15 754 at 200.
Episode 1, with `W = 0` on both sides, has ours the *cheaper* world (29 424
against 45 480), so the refactor's world is not harder.

### Follow-ups this opens

1. `issues/02` — why the legacy's iteration is stable and ours is not, given
   per-update inputs within 1.74x. Measure the spectrum of `E[XXᵀ]` on both
   sides rather than reason about it.
2. `issues/03` — over training seeds 1000-1199 the legacy draws mean 146.28
   Clients / 5.70 vehicles and ours 153.38 / 5.96, ~3σ apart on the mean and
   wider than the evaluation-seed comparison in `spec.md`.
3. `issues/04` — everything measured at TW 60: `reference_card.json`, the Gate A
   / Gate A' results paired against it, `runs/`, and `spec.md`'s own tables.

### Reproducing

The legacy world cache is warm
(`%LOCALAPPDATA%/stdvrp/legacy_w_world_c8aadb53ae7a.pkl`), so the 778s cold parse
is paid: a re-run loads in ~15s and costs ~4s/episode. `diff_TW` is not part of
the cache key — it shapes per-episode demand, not the world — so switching width
still hits the cache.

```
.venv/Scripts/python.exe -u scripts/capture_legacy_w_trajectory.py     --data-dir "C:/Users/ferna/OneDrive/Documentos/Mega city"     --diff-tw 150 --episodes 200     --out .scratch/linear-policy-learning/legacy_w_trajectory_tw150.json
.venv/Scripts/python.exe -u scripts/capture_repo_w_trajectory.py     experiments/chengdu/config_linear_congestion_10k.yaml --episodes 200     --out .scratch/linear-policy-learning/repo_w_trajectory_tw150.json
.venv/Scripts/python.exe scripts/compare_w_trajectories.py     .scratch/linear-policy-learning/legacy_w_trajectory_tw150.json     .scratch/linear-policy-learning/repo_w_trajectory_tw150.json     --out .scratch/linear-policy-learning/w_diff_tw150.md
```

(The repo config now says 150, so `--time-window-spread` is only needed to
reproduce the superseded TW 60 pair.)
