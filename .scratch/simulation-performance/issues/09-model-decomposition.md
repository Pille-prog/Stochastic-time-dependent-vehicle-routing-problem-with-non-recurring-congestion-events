# 09 — Model decomposition into concrete collaborators

**What to build:** The elegance ticket. `Model` is a god class mixing the
event loop, cost accounting, velocity sampling and episode running; `State`
carries parallel per-vehicle lists; `end_transition_function = 1/2` is an int
flag. Decompose into cohesive *concrete* classes (ADR-0002: no new seams, no
interfaces) without changing behavior. This runs last on the hot path so it
refactors the already-optimized code, not code tickets 05–07 are about to
replace.

**Blocked by:** 05, 06, 07.

**Status:** resolved

- [x] Cost accounting (the `total_*` accumulators and per-transition charging)
      extracted into a concrete ledger collaborator; the transition function
      reads as event dispatch, not bookkeeping.
- [x] Velocity sampling (`create_random_velocity`, memoization,
      `generate_normal_velocity`, congestion interaction) extracted as a
      concrete collaborator owning `velocity_rng` and the memo dict.
- [x] The `end_transition_function` int flag becomes an explicit control-flow
      construct (bool/enum/loop restructure — pick what reads best).
- [x] Per-vehicle parallel lists (`node_time_arrival`, `departure_tau`,
      `vehicles_shortest_path`, …) regrouped coherently — struct-of-arrays is
      fine (ADR-0003), scattered instance attributes are not.
- [x] Preserved-quirk documentation (the 1150/1198 constants, the epoch-gate
      arithmetic) moves with the code it describes; no quirk is fixed here.
- [x] Tier 1 gate: bit-exact self-golden pass. Benchmark note in Comments
      proving no measurable regression (>2% episode throughput).

## Comments

### Resolution (2026-07-27)

`Model` keeps one job — deciding *which event happens next* and what it does to
the world — and hands the other three to concrete collaborators it constructs
(ADR-0002: no new seams, no interfaces):

- **`CostLedger`** (`simulation/cost_ledger.py`, `model.costs`) — the four rate
  constants, the ten accumulators, and one named charge per priced event.
- **`EpisodeVelocities`** (`simulation/episode_velocities.py`,
  `model.velocities`) — `velocity_rng`, the per-arc-minute memo, the congestion
  event book, and the sampling that threads between them. In Powell's terms this
  is the Episode's *exogenous information*, which is why it lives under
  `simulation` and not under `traffic` or `congestion`: it draws from the
  `TravelTimeModel`'s distributions and holds the `CongestionGenerator`'s
  events, so it belongs to neither.
- **`FleetRoutes`** (`simulation/fleet_routes.py`, `model.fleet`) — the six
  per-vehicle lists as one struct-of-arrays value (ADR-0003), plus the three
  queries the loop asked of them inline (`earliest_arrival`, `all_parked`,
  `is_travelling`). Plain Python lists, not arrays: the transition function
  reads them one vehicle at a time, which is the case where a list index beats
  a numpy scalar index (the measurement that shaped `EpisodeGeometry`).

**The transition function is now a dispatch table.** Five branches, each one
line, each naming an event: a congestion expires, the Episode completes, the
fleet is all home, a vehicle arrives, a decision epoch begins. Every handler is
its own method. The 40-line `elif` pyramid that inlined cost arithmetic,
velocity re-sampling and route surgery between the branch conditions is gone.

**`end_transition_function = 1/2` is `self._transition_ended: bool`** and the
loop is `while not self._transition_ended`. It stays an attribute rather than a
`break` because it is set three call levels down (`vehicle_reaches_node`,
`terminate_state_*`, reached via `begin_arc`), and threading a "did you
terminate?" return value back up through those would be a bigger change than the
flag it replaces — with more places to get it wrong.

**Two control-flow quirks the flag was hiding, now commented where they live:**

- `transition_function` resets the flag *after* rerouting, so a termination
  raised during `_reroute_for` is discarded and the loop runs anyway. Only
  reachable past `CLOCK_CEILING`, which `EMERGENCY_HORIZON` beats by ~24
  simulated minutes.
- `vehicle_reaches_node`'s "this Client was already served" branch is the one
  transition end that never commits its cost to the Episode total. The reward
  the Policy learns from still carries it; only the report misses it. Pinned by
  `test_an_uncommitted_transition_is_priced_but_never_totalled`.

**The float accumulation order is load-bearing, and now says so.** Two ledger
charges take a whole collection and sum it into a local before touching a total
(`charge_unserved_delays`, `charge_fleet_overtime`) because `total + (a + b)`
and `(total + a) + b` differ in the last bits and the Tier-1 gate is `==`. Both
carry the warning at the definition and a unit test that fails if someone
"simplifies" them into per-item calls.

**Magic numbers became named constants with their quirk attached**:
`EMERGENCY_HORIZON` (1150), `CLOCK_CEILING` (1198), `DECISION_EPOCH_MINUTES`,
`SERVICE_MINUTES`, `ABORT_PENALTY` / `ABORT_PENALTY_PER_SERVED_CLIENT`, and
`fleet_routes.PARKED` for the `float("inf")` sentinel that meant "retired at the
depot" in six comparisons. The two epoch gates became named predicates
(`_congestion_epoch_due`, `_epoch_ends_the_transition`) whose docstrings carry
the `+ 180 - 2` arithmetic — kept as two operations rather than folded to
`+ 178`, because it rounds twice and the gate is bit-exact.

**Renames, all Model-internal** (nothing outside read them): `work_time` →
`shift_end_minute`, `tau_multiplicator` → `next_decision_tau`,
`vehicle_distance_transition_cost` → `advance_fleet_to`,
`create_and_actualize_state_velocity` → `begin_arc`,
`time_horizon_actualization` → `resample_arc`, `calculate_action_route` →
`_reroute_for`. The four method names ADR-0001's change log refers to
(`terminate_state_*`, `vehicle_reaches_*`) are unchanged, and every renamed
method's docstring still names the legacy method it ports.

`_next_congestion_end` also lost its second return value — the vehicle index,
which no caller ever read — and is now `_next_congestion_expiry`.

**Test surface.** `tests/unit/test_cost_ledger.py`, `test_fleet_routes.py` and
`test_episode_velocities.py` (47 cases) cover the new collaborators directly;
the velocity tests script the rng rather than seeding it, so the caps, the
even-minute bucketing and the congestion interaction are exact assertions rather
than statistical ones. `test_invariants.py`'s `RecordingModel` wraps the Model's
`EpisodeVelocities` instead of overriding `create_random_velocity` — a wrapper,
not an injected seam, since ADR-0002 forbids adding one for a test.
`tests/unit/test_model_termination.py` now assembles a `CostLedger` on its
`__new__`-built Model instead of six loose float attributes.

### Code review (2026-07-27) applied

Both review axes cleared the two things that mattered most — no documented
standard is breached, and an expression-by-expression comparison against
`HEAD:src/stdvrp/simulation/model.py` found **no change to float arithmetic
order or operand order** anywhere, which is what the Tier-1 gate is protecting.
Five judgement calls were raised and taken:

- **`EpisodeVelocities.congestion_expiry` had no production caller** — the
  Model hand-rolled the same `.get(...)[1]` on the collaborator's dict. Both
  reviewers flagged it (dead code on one axis, Feature Envy on the other).
  `_next_congestion_expiry` now calls the method, which kills both — and stops
  the Model knowing that an event is a `[multiplier, end]` pair.

  Calling it per vehicle per loop pass did cost ~1% of evaluation throughput
  (measured, not guessed: the interleaved sweep below was re-run to catch it).
  `EpisodeVelocities.any_congestion` buys it back by short-circuiting the whole
  scan, which most epochs of most Episodes want anyway — the event book is
  usually empty, and an empty book cannot expire.
- **`horizon_change_tau = tau; arc_distance_travelled = 0` appeared at three
  arrival sites** → `FleetRoutes.settle_at_node(vehicle, tau)`, which is a real
  domain event ("the vehicle is standing at a node") rather than a code-motion
  helper.
- **`begin_arc` and `resample_arc` held identical mid-service blocks** (same
  four writes, different order — the writes are independent, so unifying the
  order is bit-neutral) → `Model._hold_for_service`.
- **`CostLedger.distance_travelled` was write-only**, as `total_distance_travelled`
  had been on the Model. Dead code re-homed is still dead code, so it is gone.
- **`advance_clock_to` under-promised**: it also charges the whole fleet for the
  distance covered → `advance_fleet_to`.
- **`simulation/__init__.py` had been widened** to re-export the three
  collaborators. Reverted: they are reached through the Model that owns them,
  and the only direct importers are tests, which name their modules. The
  `__init__` docstring now says so, so the omission does not read as an oversight.

**Deferred, for whoever touches `State` next.** `state.observed_velocity[v].pop(0)`
+ `.append(x)` is an n-arc shift register open-coded at four sites, and
`state.total_vehicle_distance_travelled` is written every clock advance and read
nowhere. Both are real, and both live on `State` — the Policy-facing half of the
per-vehicle picture, which this ticket deliberately did not touch (its checklist
names the *Model's* lists). A `State.observe_velocity` would also add a bound
call per vehicle per epoch on the hot path, which is budget this ticket has no
reason to spend.

### Benchmark (2026-07-27, this machine: 16 logical cores, 31.3 GB RAM)

Tier 1 holds: `tests/test_self_golden.py` passes bit-exact (5 passed, no skips —
the gate is live on this environment), so every Episode metric and W trajectory
in `mini_fixture.json` is reproduced float-for-float. The full suite is 3024
passed / 6 deselected.

Throughput was measured **interleaved A/B against `HEAD` in a `git worktree`** —
alternate before/after runs in one session, so machine drift hits both sides
equally instead of only one. This is the sweep of the *final* code, re-run after
the code-review changes above (the first sweep measured an intermediate state
and is not what shipped):

    uv run python scripts/benchmark_episodes.py --train 40 --eval 40 --no-cache

```
round      training ep/s          evaluation ep/s
        before    after          before    after
  1       8.59     8.54           10.75    10.90
  2       8.55     8.51           10.79    10.58
  3       8.71     8.59           10.85    10.83
  4       8.55     8.55           10.85    10.83
  5       8.51     8.69           10.84    10.65
  6       8.69     7.72 *         10.63    10.29 *
best      8.71     8.69           10.85    10.90
```

**Best-of-6 — the statistic to read, since noise only ever costs time — is
-0.2% on training and +0.5% on evaluation.** Round 6's after-run (*) is a
machine outlier, ~11% below the band on both phases at once; excluding it the
means are -0.1% and -0.5%, and including it -1.9% and -1.0%. Every reading is
inside the ticket's 2% gate, and inside the before column's own 2.3% spread.

This is the expected shape: the transition function was ~5% of episode time at
ticket 01's profile, so even a real few-percent change inside it barely shows at
the Episode level. Ledger-wise the extraction adds one bound call per charge and
per sample, and gives back two list copies per reroute, a tuple repack per memo
hit, and — via `any_congestion` — a whole per-vehicle scan on every epoch with no
active congestion.

