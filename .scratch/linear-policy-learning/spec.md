# Spec: the linear policy stopped learning — why, and what is still divergent

Status: **open.** One necessary cause found and fixed (the missing update-time
feature clip). It is *not sufficient* — with the clip the Policy still fails to
converge over 500 episodes while the legacy is flat from episode 50. The
catalogue at the bottom is now the suspect list for the residual instability,
not an optional-cleanup list.

Ticket 01 has since found two things. First, **the time-window width was wrong**:
every Chengdu config said `time_window_spread: 60` where the reference runs used
150, and correcting it moves the two sides in opposite directions — see
"Corrected first" below. Second, at the corrected width, **no single component
misbehaves — all 22 that the legacy trains are inflated here, by 2.6x to 13.8x.**
Those two sections supersede both the "which component rotates" framing and every
TW 60 number the rest of this document was written against.

Reference implementation (the one that *does* learn):
`c:\Users\ferna\OneDrive\Documentos\Mega city\Main_Chengdu_Sirve_Con_Clip.py`
— 7431 lines, run as `python Main_Chengdu_Sirve_Con_Clip.py
"<iters>,<test_freq>,<lr>,<eps>,<lc>,<uc>,<duration>,<clients>,<diff_TW>,<test_iters>"`.
Read it with care: many methods are defined more than once and several live
inside triple-quoted blocks (fences at 2022/2081, 2084/2152, 2154/2211,
2212/2225, 4185/4303, 4763/4791, 4983/5043). The live training path is
`main` 7380 → `training_and_testing.training_model` 6906 →
`model.create_monte_carlo_episode_train` 5977 → `policy.monte_carlo_policy_train`
1957 → `select_epsilon_greedy_action_train` 2237 →
`generate_best_Q_pred_for_1_vehicle` 2285 → `extract_general_state_features` 2859
/ `extract_state_action_features` 3573 → `policy.actualize_W` 4318.

## Handoff — where this stands

**Shipped to `src/`** (the only production change this effort has made):
`UPDATE_FEATURE_CEILING = 3` in `src/stdvrp/policies/monte_carlo.py`, applied
inside `learn`, plus two regression tests in
`tests/unit/test_monte_carlo_policy.py::TestWUpdate`. 3907 unit tests pass;
ruff and mypy clean.

**Tooling left behind:** `scripts/legacy_fidelity_bench.py` — trains the linear
Policy with any combination of candidate legacy reverts monkeypatched on
(`clip`, `bin3`, `legacydepot`, `unserved`, `nofifo`, `evalpool15`, `step6`) and
prints the evaluation-cost and ‖W‖ trajectory. Every table in this document was
produced with it. Nothing in `src/` was reverted to measure any of them.

Ticket 01 added three more, plus the trajectories they produced:

- `scripts/capture_legacy_w_trajectory.py` — imports `Main_Chengdu_Sirve_Con_Clip.py`
  unmodified (ticket 04's shim) and captures its W after every training episode.
  The world cache is now warm at
  `%LOCALAPPDATA%/stdvrp/legacy_w_world_c8aadb53ae7a.pkl`, so a re-run loads in
  ~15s instead of the 778s cold parse, and runs ~4s/episode.
- `scripts/capture_repo_w_trajectory.py` — ours, nothing monkeypatched, ~1.3s/episode.
- `scripts/compare_w_trajectories.py` — the per-component diff. Its pure analysis
  is unit-tested in `tests/unit/test_w_trajectory_comparison.py`.
- At the corrected TW 150: `legacy_w_trajectory_tw150.json`,
  `repo_w_trajectory_tw150.json` (200 episodes each) and `w_diff_tw150.md`.
- At the old TW 60, superseded but kept because the tables below were produced
  at it: `legacy_w_trajectory.json`, `repo_w_trajectory.json`,
  `repo_w_trajectory_500.json` (500 episodes) and `w_diff_200.md`.

Both drivers take the width as an override (`--diff-tw` /
`--time-window-spread`) so a comparison run never has to edit a committed config.

The repo driver is validated against the numbers already in this document: at
TW 60 its trajectory reproduces every ‖W‖ in the table below to the digit (5460
at 50, 6961 at 100, 6253 at 150, 7502 at 200, 13646 at 250, 22909 at 400, 21760
at 500). The legacy driver is validated against the legacy's own stored result
file — see "Corrected first".

**Next step:** ticket 01 is resolved. It opens three:
`issues/02-why-is-the-iteration-unstable.md` (why the legacy's ‖W‖ settles and
ours does not, given per-update inputs within 1.74x),
`issues/03-training-demand-draw-gap.md`, and
`issues/04-retire-the-tw60-measurements.md` (everything pinned at TW 60).

**Known repo defect, unrelated but in the way:** `pytest tests/unit` cannot
collect — `stdvrp.policies` imported before `stdvrp.simulation` hits a circular
import (`policies/__init__` → `monte_carlo` → `action_set` →
`simulation.state` → `simulation/__init__` → `episode` → `monte_carlo`,
half-initialised). Pre-existing on this branch, confirmed by stashing this
effort's changes. Work around it with
`python -c "import stdvrp.simulation, sys, pytest; sys.exit(pytest.main(['tests/unit','-q']))"`.

## The symptom

`runs/linear_congestion_0.1-0.4_10k_eps0_lr1e-5` degrades monotonically, 5866 →
7199 over 1350 episodes. `runs/linear_congestion_0.1-0.4_10k` oscillates and
ends worse than it started. `runs/linear_congestion_0.1-0.4_10k_restored_features`
— the newest, after the restored depot-distance / congestion / `norm_future**2`
features — goes 10531 → 19975.

## The root cause

`actualize_W` line 4374 clips the assembled feature vector before it computes
anything:

```python
self.X = np.clip(self.X, a_min=None, a_max=3)
```

and it clips **only there** — `generate_best_Q_pred_for_1_vehicle` (2285-2308)
prices candidates raw. The port dropped it. `MonteCarloPolicy.learn` had no
feature clip and neither did `FeatureExtractor`.

That asymmetry is load-bearing, and the file's own name says so ("Con_Clip").
Most components sit below 1, but two do not: `X[22] = (future_delay/2500)**2`
— restored in the uncommitted feature work, measured at 7.76 — and
`X[16] = late_count/13`, measured at 3.46. Unclipped, the update multiplies a
residual of thousands by a feature of tens, W walks away, and the greedy argmin
follows it.

Measured on the real Chengdu data, `config_linear_congestion_10k.yaml`
(lr 1e-3, ε 0.1), 25 training Episodes:

| | ‖W‖ | eval mean cost |
| --- | --- | --- |
| without the clip | 16 104 725 | 57 357 |
| with the clip | 4 376 | 6 940 |

## The fix (shipped)

`src/stdvrp/policies/monte_carlo.py`: `UPDATE_FEATURE_CEILING = 3`, applied
inside `learn` between `action_features` and `Q_pred`, nowhere else. Two
regression tests in `tests/unit/test_monte_carlo_policy.py::TestWUpdate` pin
both halves of the asymmetry: that the update clips, and that the decision
argmin does not. Full unit suite: 3907 passed.

## What was measured, not argued

150 training episodes, `config_linear_congestion_10k.yaml`, evaluation on 10
seeds at vehicles+2. Single runs — the spread between adjacent blocks is wide
enough that only the first row is a real signal.

| arm | 25 | 50 | 75 | 100 | 125 | 150 | final ‖W‖ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no fixes | 57 357 | | | | | | 1.6e7 |
| clip | 6 940 | 5 850 | 6 404 | 6 941 | 4 850 | 4 311 | 6 253 |
| clip + legacy dead bin 3 | 5 512 | 6 329 | 4 635 | 6 135 | 4 617 | 43 036 | 14 688 |
| clip + legacydepot + nofifo + unserved | 6 280 | 6 407 | 5 029 | 4 448 | 4 944 | 5 282 | 4 518 |

Reading: the clip is the fix. Reverting the B10 "fourth earliness bin" fix is
actively harmful. The three ADR reverts in the last row do not beat the clip
alone on cost at this horizon, but they are the only arm where ‖W‖ *stops
growing* — worth a longer run before deciding.

### Against the legacy, at the legacy's own measurement width

Same 50 evaluation seeds, same `vehicles + 15` pool. The demand matches: over
those seeds the legacy draws mean 148.94 Clients / 5.78 vehicles and the repo
147.18 / 5.76 (both `N(150, 30)` floored at 60, nodes sampled without
replacement from `range(1, 1900)` — note `client_universe_size` is *not* read by
either demand generator).

| episodes | repo (clip) | ‖W‖ | legacy |
| --- | --- | --- | --- |
| 50 | 14 777 | 5 460 | 3 385 |
| 100 | 7 486 | 6 961 | 3 457 |
| 150 | 4 328 | 6 253 | 4 316 |
| 200 | 8 189 | 7 502 | 3 795 |
| 250 | 5 165 | 13 646 | |
| 300 | 40 077 | 13 691 | |
| 350 | 12 744 | 13 203 | |
| 400 | 17 284 | 22 909 | |
| 450 | 5 890 | 20 829 | |
| 500 | 6 043 | 21 760 | |

**The clip is necessary, not sufficient — this is the correction that matters.**
A first read of the 150-episode window (14 777 → 7 486 → 4 328) looked like
convergence. It was not: extended to 500 episodes the run oscillates between
4 328 and 40 077 and **‖W‖ grows monotonically throughout**, 5 460 → 21 760. The
legacy, at the same lr, the same ε and a matched demand distribution, stays
bounded in [3 385, 4 316] across the same span. Do not quote the episode-150
number as evidence of learning.

Also worth keeping: widening the evaluation pool makes the repo *worse*, not
better — 14 777 at `+15` against 5 850 at `+2` for the identical W (‖W‖ matches
the `+2` clip arm to the digit; `evalpool15` changes only the measurement). The
narrow pool was hiding how badly this W generalizes to a wider action set, not
hiding learning.

The two curves differ in *shape*, not offset: the legacy is flat from episode 50
and the repo never settles. That points at a divergence which changes what is
being fit — the target, the feature semantics, or the reachable action set —
rather than at a step-size or conditioning effect.

## Corrected first: the time-window width was wrong (`diff_TW`)

Every Chengdu config in this repo said `time_window_spread: 60`. The reference
runs used **150** — `capture_golden_master.py`'s protocol says `diff_TW: 150`,
and the legacy's own result filenames say `Tw150`. The configs are now 150.

This is not a detail. `diff_TW` sets each Client's time-window width *and* the
latest minute a window can open (`randint(300, 780 - diff_TW)`), so 60 is a
strictly harder problem than 150. Measured over 200 training episodes:

| | legacy `‖W‖` @200 | legacy train cost (mean/median) | repo `‖W‖` @200 | repo train cost |
| --- | --- | --- | --- | --- |
| TW 60 (what this repo was running) | 3 732 | 6 671 / 6 126 | 7 502 | 11 229 / 10 008 |
| TW 150 (what the reference used) | 2 895 | 3 534 / 2 695 | 15 754 | 9 450 / 6 749 |

The legacy's own numbers are the check: at TW 150 its training cost lands at
mean 3 534 / median 2 695, inside the [2 798, 4 316] band this document records
for it and matching the legacy's own stored result file for
lr 0.001 / ε 0 / lc 0.1 / uc 0.4 / duration 120 (mean cost 4 182). At TW 60 it
comes out at 6 671, outside that band. **TW 150 is the setting the legacy
numbers in this document were produced at.**

And the correction moves the two sides in *opposite* directions: widening the
windows helps the legacy (3 732 -> 2 895) and hurts us (7 502 -> 15 754). TW 60
was masking half the gap.

**Everything measured at TW 60 needs re-reading**, including every table above
this section, `runs/`, the frozen `experiments/chengdu/reference_card.json`, and
the neural effort's Gate A / Gate A' results. See
`issues/04-retire-the-tw60-measurements.md`.

## Inside W: all 22 trained components are inflated, none is bounded

Ticket 01's measurement. 200 training episodes per side at this config's
parameters (lr 1e-3, ε 0.1, congestion 0.1-0.4, duration 120, 150 clients,
**TW 150**), W captured after every episode. Full table in `w_diff_tw150.md`;
`w_diff_200.md` is the superseded TW 60 pair, kept only because the rest of this
document was written against it.

Peak |W[i]| over the run, ours against the legacy's:

| | components | ratio |
| --- | --- | --- |
| most inflated | `distance_cost/100` 13.8x, `earliness_cost/60` 12.3x, `earliness_bin1` 9.9x | 9.9x - 13.8x |
| middle | `future_delay/2500`, `(future_delay)^2`, `congestion_signal`, `time^2*clients_left^2`, `overtime_cost/180` | 6.2x - 8.1x |
| least inflated | `delay_cost/60`, `mean_earliness_diff`, `earliness_bin0` | 2.6x - 3.1x |
| structurally absent there | `earliness_bin3` (legacy peak exactly 0, ours 3 653) | — |
| dead on both sides | `zero_pad` | — |

Median 4.84x over the 22 the legacy trains; range 2.64x - 13.81x. **Not one
component is bounded here.** This is the ticket's second branch: there is no
named feature to blame, so the fault is in the target or the conditioning of the
update, not in any one regressor. (The TW 60 pair says the same thing with
smaller numbers — median 3.97x, range 2.22x - 9.79x — so the finding is not an
artifact of either setting.)

Two things this settles:

- **The B10 tension is resolved, and B10 is not the answer.** `general[13]` *is*
  a live regressor here (peak 3 653 against a structural 0, feature value
  averaging 0.242 per update) and structurally zero in both legacy sources. So it
  is a real divergence and it does put mass into every update. But it carries a
  fraction of a percent of the final `‖W‖` while the other 22 components are
  inflated 2.6-13.8x. Reverting B10 cannot fix an inflation spread across the
  whole vector — consistent with the measured revert being *worse*, not better.
- **It is not step size per update.** Decomposing
  `W += lr·(U_t − acquired − Q_pred)·X` over the first 25 episodes:

  | per-update statistic | legacy | repo | ratio |
  | --- | --- | --- | --- |
  | updates per episode (`T`) | 377.8 | 413.4 | 1.09 |
  | mean `‖X‖` (post-clip) | 1.803 | 2.170 | 1.20 |
  | mean non-zero components of `X` | 15.93 | 17.07 | — |
  | mean `X[13]` (`earliness_bin3`) | 0.000 | 0.242 | live vs structurally 0 |
  | mean \|residual\| | 1 380.1 | 2 398.1 | 1.74 |
  | mean *signed* residual | +939.9 | +1 267.9 | 1.35 |
  | step coherence `‖ΔW‖ / Σ‖step‖` | 0.868 | 0.784 | 0.90 |
  | `‖ΔW‖` per episode | 300.1 | 1 405.4 | 4.68 |
  | `‖W‖` after 25 episodes | 1 657.5 | 15 188.7 | 9.16 |

  No single input is off by more than 1.74x, and our steps are *less* coherent
  than the legacy's, yet `‖ΔW‖` comes out 4.7x and `‖W‖` 9.2x. The two inputs
  that are genuinely larger are `‖X‖` (+20%, of which `X[13]` — live here,
  identically zero there — is a substantial share) and the residual (+74%).
  A linear Monte-Carlo update's stability margin scales with
  `lr·λmax(E[XXᵀ])`, so a uniformly heavier `X` is exactly the shape of thing
  that turns a converging iteration into a drifting one, and it inflates *every*
  weight rather than one.

  Measured by wrapping `actualize_W` / `learn` without editing either side; both
  instrumented runs reproduce their uninstrumented `‖W‖` exactly.

**The difference is stability, not magnitude.** The legacy's `‖W‖` rises to
4 249 by episode 50 and then comes *back down* — 3 134 at 100, 2 796 at 150,
2 895 at 200. Ours does not settle: 15 189 at 25, 11 875 at 50, 7 910 at 100,
12 784 at 150, 15 754 at 200. The training costs agree: legacy mean 3 534 to our
9 450 (2.67x) — while at episode 1, with W = 0 on both sides, ours is the
*cheaper* world (29 424 against 45 480), so this is not "our world is harder".

Worst single episodes: our largest `‖ΔW‖` is 17 951 (episode 43), which moves W
further in one episode than the legacy's entire trained norm; the legacy's
largest is 1 844, at episode 2. The episodes-251-300 block this document
flagged is *not* where it comes from — in the only run that reaches those
episodes (`repo_w_trajectory_500.json`, TW 60) the block's biggest step ranks
5th of 500 and its mean `‖ΔW‖` is *below* the run's overall mean.

*Caveat on the demand draw:* over training seeds 1000-1199 the legacy draws mean
146.28 Clients / 5.70 vehicles and ours 153.38 / 5.96 — unchanged by `diff_TW`,
which does not touch the count draw. Same distribution, different realisations
(ticket 13 replaced the shared global streams with per-Episode `SeedSequence`
spawning), but at ~3σ of the mean it is wider apart than the evaluation-seed
comparison recorded above. See `issues/03-training-demand-draw-gap.md`.

## Ruled out: the clock discretization is not the explanation either

The simulation clock advances every 2 minutes (`DECISION_EPOCH_MINUTES`,
legacy `tau_multiplicator_difference`); decisions fire only when
`(tau + 180 - 2) % 6 == 0`, i.e. every 6 minutes, one clock step in three. Both
sides agree — this is a faithful port. Coarsening the *clock* to 6 minutes
(third of the velocity resamples, same decision instants: start
`next_decision_tau` at 302, not 306, or the gate never fires at all) changes
nothing qualitative. 500 episodes, same measurement as the table above:

| episodes | 2-minute clock | 6-minute clock |
| --- | --- | --- |
| 50 | 14 777 | 8 387 |
| 100 | 7 486 | 4 614 |
| 150 | 4 328 | 5 920 |
| 200 | 8 189 | 5 509 |
| 250 | 5 165 | 5 269 |
| 300 | 40 077 | 7 359 |
| 350 | 12 744 | 5 526 |
| 400 | 17 284 | 15 826 |
| 450 | 5 890 | 18 478 |
| 500 | 6 043 | 7 991 |
| final ‖W‖ | 21 760 | 25 163 |

Both oscillate, both grow ‖W‖ without bound. The 6-minute clock dodges the
episode-300 spike and grows its own at 450.

## Ruled out: the objective function is not the explanation

The cost function *was* changed (B3/ADR-0004: the legacy charges only unserved
Clients with `tau > due`, at the live `tau` on the horizon path and at 1150 on
the all-back path; the refactor charges every unserved Client at
`max(0, max(1150, tau) - due)`). `docs/simulator-review.md` B3 shows that
mattering by four orders of magnitude on a one-vehicle fixture, so it is a fair
suspect. It is not the cause here.

Measured by pricing the *same* evaluation episodes under both rules — the
terminal charge does not feed back into the trajectory, so the pair is one
identical episode priced twice, and `distance_cost` comes out bit-identical
across the pair, proving it:

| | repo pricing | legacy pricing | difference |
| --- | --- | --- | --- |
| trained W, total | 6 047.83 | 6 047.71 | **0.12** (0.002%) |
| W = 0, total | 36 334.65 | 36 286.37 | 48.28 (0.13%) |

It vanishes because at this config the all-back path essentially never fires
(1 premature ending in ~500 legacy episodes) and horizon terminations stop at
`tau ≈ 1148`, so `max(1150, tau) - due` and `tau - due` differ by ~2 minutes.
B3's dramatic example lives on the path this config does not take.

The rates are identical on both sides (earliness 0.1, distance 1, delay 1,
overtime 5/6, service time 5, shift end 780, decision epochs every 2 minutes —
legacy `model.__init__` 5222-5231 vs `CostLedger.__init__`).

**Episode shape also matches**, so the gap is not "the refactor's world is
harder": with the trained W the repo finishes early on 48 of 50 evaluation
seeds (mean `tau` 975) and leaves 0.06 Clients unserved, against ~496 of 500
legacy episodes finishing before the horizon.

What is left is genuine policy quality. The repo's trained policy serves 93.6
of ~147 Clients *late*, and delay is 78% of its cost (4 691 of 6 048; distance
674, overtime 640, earliness 42). With identical rates and a legacy total of
2 798–4 316, the legacy is routing substantially better, not being charged less.
*(Caveat: the legacy run was stopped before `test_model`, which is what writes
its component breakdown, so its delay share is inferred from its total, not
observed.)*

## Still divergent from the reference (confirmed, undecided)

Every item below was verified against the legacy source by an adversarial pass;
all are real. None is required to make the policy learn. They are listed because
the stated goal is to match this legacy file, and because several were shipped
as deliberate fixes under an ADR — reverting them is a decision, not a bug fix.

1. **`vehicle_standing` (ADR-0005, B1a/B1b) at seven sites.** The legacy has no
   standing concept: every depot test is `vehicle_position == depot` alone. The
   refactor's `is_parked_at_depot` is a strictly smaller set, so a vehicle
   mid-arc past the depot (an interior node on 6.8% of cached paths) is treated
   as *out* where the legacy treats it as *home*. Sites:
   `monte_carlo._already_acquired_cost`, `feature_extraction._classify_closest_clients`,
   `action_set` (the 350 lock and both 310 cutoffs),
   `model._every_vehicle_home_and_no_clients_left`,
   `model.terminate_state_passing_horizon`. This one predicate moves the
   training target, the future-delay features, the candidate set and the episode
   length at once.
2. **Unserved-Client pricing (ADR-0004, B3/B14).** The legacy filters
   `if tau_episode > due` in both terminators and prices the premature-ending
   path at the fixed 1150 clock and the horizon path at the live `tau`. The
   refactor charges *every* remaining Client at `max(0, max(1150, tau) - due)`.
   On a premature ending that is a one-directional inflation of `rewards[T]`,
   which enters every epoch's `U_t`.
3. **The congestion-expiry (FIFO) branch.** Commented out in the legacy
   (5723-5736 — the run filename says `_SinFIFO_`), live in
   `Model.transition_function`. It resamples every vehicle's arc on expiry,
   so the 3-slot `observed_velocity` window the Policy reads advances on a
   different schedule than the legacy's.
4. **Evaluation action pool.** `training_model` builds its evaluation policy
   with `number_vehicles + 15` (line 6983) while training uses `+2`. The
   Trainer's evaluation blocks use `+2`. Every number in `runs/` — and
   `best_w`'s selection — is therefore measured on a candidate set 13 wide
   narrower than the legacy's. **Not changed here**: the frozen
   `reference_card.json` and the neural effort's Gate A/A' numbers are all
   pinned at the current width, so moving it invalidates them.
5. **B10, the fourth earliness bin.** `client_counts_earliness[3]` is never
   assigned in the legacy (2938-2947, three branches, no fourth), so
   `general[13]` is structurally 0 and `W[13]` never trains. The refactor fills
   it with the residual `remaining_count - sum(bins 0..2)`.

   Two facts in tension, both worth carrying forward. Against the fix: the
   verification pass also checked `Main_Chengdu_Sirve_2_Acciones.py` — the
   source `feature_extraction.py`'s own docstring names as the port's origin —
   and its *live* `extract_general_state_features` has the same dead
   three-branch chain. Only its dead `_routes` variant assigns bin 3, and as
   `elif 600 <= i < 700`, a fourth band rather than a residual. So the
   refactor's residual matches **neither** source, and the partition invariant
   B10 was justified by restoring was never there to restore. For the fix: the
   measured revert above was *worse* (43 036 at episode 150, ‖W‖ 14 688) —
   though that is one noisy run and the blow-up sits in the last block. This
   needs a repeated run before anyone acts on it either way.

   **Resolved by ticket 01** (see "Inside W"): `W[13]` is a live regressor here
   (peak 3 653 against a structural 0) and the feature averages 0.242 per
   update, so it does add real mass to `X`. But it carries a fraction of a
   percent of the final ‖W‖ while the other 22 components are inflated
   2.6-13.8x, so it cannot be the cause. Both facts stand; neither makes
   reverting B10 the fix. (Figures at TW 150. The measured revert quoted above
   was run at TW 60 and is one of the numbers `issues/04` retires.)
6. **B11 (`< 3` Clients endgame forbidden-action filter)**, **B5 (the empty
   `distances` guard)**, **B15 (the overtime guard)**, **B20 / ADR-0008
   (`fleet.is_at_node` on the depot-park branch)**, the double
   `total_cost` commit on the horizon path, and the three-RNG-streams collapse.
   All confirmed, all small, all deliberate.

## Not reproducible against the legacy run

The legacy was run for comparison (`500,50,0.001,0.1,0.1,0.4,120,150,60,1`). Its
per-episode trajectory is *not* comparable episode-by-episode: ticket 13 retired
the legacy's two shared global RNG streams for per-Episode `SeedSequence`
spawning, so episode k of one is a different world from episode k of the other.
Only the shape of the learning curve can be compared, not the numbers.
