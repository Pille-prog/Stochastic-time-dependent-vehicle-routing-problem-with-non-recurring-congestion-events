# 06 — Training snapshots without deepcopy

**What to build:** Replace the per-step `copy.deepcopy(self.state)` /
`copy.deepcopy(action)` in `Model.run_training_episode` with a purpose-built
snapshot capturing exactly what `MonteCarloPolicy.update_W` replays.

**Blocked by:** 01.

**Status:** resolved

- [x] Identify (and pin with a test) the exact State surface `update_W` reads
      via `_calculate_already_acquired_cost`, `_extract_general_state_features`
      and `_extract_state_action_features`: `tau_episode`,
      `clients_not_visited`, `vehicle_position`, `observed_velocity`, plus any
      field the ticket's audit finds. The snapshot copies only that, shallowly
      where aliasing is safe, and is immutable (frozen dataclass or equivalent).
- [x] `update_W` accepts the snapshot type; the `self.state` rebinding to
      historical snapshots keeps working (or is redesigned away — the rebind is
      hidden temporal coupling; dissolving it counts as the elegance goal, but
      stays behavior-identical).
- [x] Tier 1 gate: W trajectories bit-identical to self-golden on the fixture.
- [x] Benchmark note in Comments: training-episode throughput before/after.

## Comments

### Resolution (2026-07-23)

`TrainingSnapshot` (`src/stdvrp/simulation/state.py`) is a frozen, `slots=True`
dataclass over exactly the four fields the audit found `update_W`'s replay path
reads — traced through `_calculate_already_acquired_cost`,
`_extract_general_state_features` (and the `_classify_delayed_clients` it calls)
and `_extract_state_action_features`: `tau_episode`, `clients_not_visited`,
`vehicle_position`, `observed_velocity`. No other `State` field is touched
anywhere in that call graph — confirmed by reading every method in the chain,
not just grepping for `self.state`. `TrainingSnapshot.capture(state)` copies
these (tuples, so the copy and the immutability come from the same conversion)
instead of aliasing them, because `State` mutates `clients_not_visited`,
`vehicle_position` and `observed_velocity` **in place** across the Episode —
that in-place mutation is exactly why the original code needed a *deep* copy at
all, not just a fresh top-level object;
`tests/unit/test_training_snapshot.py::test_is_immune_to_later_mutation_of_the_source_state`
pins it directly (mutate the source `State` after capture, assert the snapshot
is unaffected).

`Model.run_training_episode` (`src/stdvrp/simulation/model.py`) now appends
`TrainingSnapshot.capture(self.state)` instead of `copy.deepcopy(self.state)`,
and `list(action)` instead of `copy.deepcopy(action)` — `action` is a flat
`list[int]`, and it's the Policy's own mutable `self.action` handed back by
reference each decision, so it only needs a shallow copy to stop aliasing.
`MonteCarloPolicy.update_W` (`src/stdvrp/policies/monte_carlo.py`) now types
`states: list[TrainingSnapshot]`; the `self.state` rebind-per-epoch design is
**kept as-is** (first parenthetical option in this ticket) — `self.state` is
typed `State | TrainingSnapshot`, and every method the replay path calls reads
only the four shared fields, so the duck-typed rebind needs no other change.
Redesigning the rebind away is left to ticket 09 (Model decomposition), which
already owns broader Policy/Model seam changes; doing it here would widen this
ticket's blast radius for no behavior or measured-speed gain.
`tests/unit/test_monte_carlo_policy.py`'s `TestWUpdate` cases now build a
`TrainingSnapshot` explicitly instead of passing a raw `State`, matching what
real callers do.

**Tier 1 gate:** `uv run pytest tests/test_self_golden.py` — all 5 tests pass on
this machine (environment-live, not skipped):
`test_training_w_trajectory_is_bit_exact` confirms the per-seed W trajectories
on the mini fixture are bit-identical to the pre-ticket capture; the other four
self-golden tests (metrics, final W, evaluation) pass unchanged as expected,
since evaluation episodes never touch this code path.

**Benchmark (training-episode throughput, before/after).** This machine had
heavy concurrent load from other sessions working other tickets in the same
branch during measurement, which makes wall-clock timing noisy (world load
alone varied 20–27s across runs with no code changes). `cProfile` isolates
per-function CPU time from that noise and is the more reliable signal here:

```
cProfile, 5 training episodes on the fixture (world built outside the profiler)

before (ticket 01 baseline, this machine)   1.191s / 5 ep
  copy.py:128 deepcopy            0.062  5.2%
  copy.py:201 _deepcopy_list      0.018  1.5%
  copy.py:227 _deepcopy_dict      0.009  0.8%
  (deepcopy family total)         0.089  7.5% of profiled time

after (this ticket)                          1.124s / 5 ep
  copy.py:* deepcopy              absent from the profile entirely
```

Same call counts throughout (e.g. `_extract_state_action_features`: 7224 calls
both before and after) confirm the change is call-graph-identical, not just
same-total-cost — nothing upstream changed how many times anything runs, only
what `run_training_episode` does with the snapshot. The ~5.6% total-time drop
(1.191s → 1.124s) is consistent with removing the measured 7.5%-of-profiled-time
`deepcopy` family, net of normal run-to-run profiler variance. Reproduce:
`uv run pytest tests/test_self_golden.py` for correctness, or profile
`scripts.benchmark_episodes.time_training` directly with `cProfile` for timing.
