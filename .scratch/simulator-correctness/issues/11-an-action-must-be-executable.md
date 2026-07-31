# 11 — B20: an action must be executable

**What to build:** Close **B20**, a crash in the shared routing code reached by
any Policy whose action set can name the node a vehicle is currently on — and
the invariant that would have caught it. Found by `neural-policy` ticket 08
while training the transformer on Chengdu; **not** introduced by that ticket's
diff.

**Blocked by:** —

**Status:** resolved

**Reopens this effort.** Closed at ticket 10 (2026-07-30) on the criterion
*"the next defect of this shape fails a test instead of surviving until
someone reads the code"*. This defect is exactly that shape and failed no
test: it crashed a training run. The three things it needs edited — the
catalogue (`docs/simulator-review.md`), an ADR-0005 consequence, and
`tests/test_invariants.py` — are all this effort's own deliverables.

---

## The defect

`Model._reroute_for`'s **at-a-node** branch routes straight from the vehicle's
position:

```python
if fleet.departure_tau[vehicle] == self.state.tau_episode:
    fleet.route[vehicle] = list(
        self.shortest_path_cache.path_between(last_node_reached, vehicle_destination).nodes
    )
    self.begin_arc(vehicle)
```

`all_shortest_paths.csv` contains all 45 self-rows (`0,0,0,0.0,0.0`), so
`path_between(n, n)` **does not raise** — it returns a well-formed, one-node,
zero-length path. `FleetRoutes.current_arc` then reads `route[1]` and dies with
`IndexError: list index out of range`.

**Trigger.** All four at once:

1. the vehicle is *on* a node with zero arc progress (`departure_tau == tau`);
2. `vehicle_standing` is `False` — `begin_arc` has already launched it, so the
   ticket-04 `is_parked_at_depot` guard does not fire;
3. the decision names that same node;
4. `fleet.destination != action` (otherwise no reroute happens at all).

Today only the **depot** is reachable, because it is the only node always in
the action set (ADR-0007). The **client** case — a vehicle standing on a
pending Client it crossed without serving — is reachable by construction and
crashes identically.

### Reproduction

`tests/fixtures/chengdu_mini`, seed **1131**, untrained network, vehicle 0:

| τ | event |
|---|---|
| 302.0 | heading to Client 10, mid-arc `0→4`. Decision flips to **depot**. Mid-arc splice `[last] + path(4→0)` → route `[0, 4, 0]` |
| 303.28 | crosses node 4 → route `[4, 0]` |
| 308.0 | decision flips to **Client 13**. Splice `[4] + path(0→13)` → route `[4, 0, 4, 16, 13]` — **the depot is now an interior waypoint** |
| 308.7466 | reaches the depot as a waypoint; `begin_arc` sets `departure_tau = τ`, `standing = False`. **Same instant**, vehicle 2 parks at the depot → transition ends |
| 308.7466 | `_reroute_for`: at-a-node branch, `last_node_reached = 0`, action = `0` → `path_between(0,0)` → `[0.0]` → 💥 |

### Two corrections to the received story

- **This is not the review's 6.8 % cached-path statistic.** `path(0→13)` is
  `0→4→16→13`; the depot became interior because the vehicle was *heading to
  the depot* and got re-targeted mid-arc, so `_reroute_for`'s
  finish-the-current-arc splice sewed the depot into the middle of the route.
  A Policy that flip-flops depot↔Client manufactures these crossings itself,
  at a rate unrelated to 6.8 %.
- **The coincidence in step 4 is structural, not luck.** Vehicles 0 and 2 were
  in **lockstep**: launched together, shortest paths sharing a prefix, and
  `EpisodeVelocities` memoises velocity per (arc, minute) — so they travel
  identically and arrive together. One vehicle's arrival routinely ends the
  transition at the exact instant another is crossing a node.

### Measured

| measurement | sample | result |
|---|---|---|
| Crash rate, transformer (untrained, ε-greedy training episodes) | 80 episodes | **3** (seeds 1116, 1131, 1134) — all at τ = 308.7466 |
| Crash / precondition, linear `MonteCarloPolicy` | 600 episodes (200 seeds × fleets 1/3/6) | **0 / 0** |
| Degenerate routes, linear | 600 episodes | **0** |
| "Depot recorded, not standing" (the state a Policy-side gate would have to exclude) | 2 229 transformer decisions | 15 (0.67 %) — **all** genuinely mid-arc, none zero-progress |
| …of those, with zero pending Clients (would empty the action set, B5) | — | **0** |
| Crashes, uniform-random full-action-space policy | 300 episodes | **0** |
| Crashes, hand-built adversarial "lockstep flip-flop" policy | 120 episodes | **0** |

**The last two rows are the important finding.** The trigger needs
*correlated fleet behaviour* — vehicles launched together on prefix-sharing
paths, kept synchronised by the per-(arc, minute) memo. Uniform randomness
**destroys** that correlation; a real greedy Policy produces it for free. So
this defect is structurally invisible to fuzzing, and widening the invariant
suite the obvious way would not have caught it either. That is why the
red-before test below is a hand-built unit test and an end-to-end regression
on the observed seed, not a random-policy sweep.

### Why the ticket-04 fix did not prevent it

ADR-0005 defines `vehicle_standing` to flip to `False` **the moment
`begin_arc` launches the vehicle**. That leaves an instant — *"I am on the
node, I have not moved a metre"* — where the simulator simultaneously says:

- *at the node* (`departure_tau == tau`) → so the at-a-node routing branch runs;
- *not standing* (`vehicle_standing == False`) → so the park branch does not.

The gap between those two readings is the bug. Note that `test_invariants.py`
already defines "genuinely mid-arc" with a **strict** inequality
(`departure_tau < tau < arrival_tau`), so the zero-progress instant is
deliberately *outside* it — parking there does not violate ADR-0005's own
invariant.

---

## The decision

**An action must be executable.** Concretely:

- **The depot is always in the action set**, but it means two different things:
  **park** when the vehicle is on the depot node, **travel** when it is not.
  Only the simulator can tell those apart (the Policy cannot see
  `FleetRoutes`), so the simulator makes the distinction.
- **A node the vehicle is already on is not a travel destination.** For a
  pending Client that is a Policy-side feasibility rule, not a heuristic: there
  is nothing to travel to.

"Can park" is therefore **positional presence** — on the node, zero arc
progress — *not* `vehicle_standing`.

### Considered and rejected

- **Gate the depot Policy-side on `vehicle_standing`.** Literal, cheapest,
  leaves `model.py` untouched, and covers the Client case for free. Rejected:
  it also removes the depot from the action set for vehicles *genuinely rolling
  past it*, where "go home" is a perfectly well-defined travel action the
  simulator already executes correctly — 0.67 % of decisions, measured, and
  precisely the end-of-shift decision where being wrong costs overtime. It is
  the mirror of B1a. It also leaves the action set empty when no Clients are
  pending (B5; latent, 0 observed).
- **A no-op reroute** when the target is the current node. Rejected: silently
  discards a decision the model can execute.
- **A well-formed zero-length self arc `[n, n]`.** Rejected: new physics
  (travel time 0, congestion lookup on a link that does not exist) and an
  arrival-at-τ spin risk.
- **A new `State` fact visible to both sides**, so offer and execution use
  literally one predicate. Sound, and the purist reading of ADR-0005's own
  lesson — but it changes what the Policy sees, touches `State`,
  `TrainingSnapshot` and ADR-0005, for a case the simulator can settle alone.

---

## The fix

| where | what |
|---|---|
| `simulation/fleet_routes.py` | `is_at_node(vehicle, tau) -> bool` = `departure_tau[vehicle] >= tau`. The missing **fact**, named where progress lives |
| `simulation/model.py` `_reroute_for` | park branch: `action[v] == depot and last_node_reached[v] == depot and fleet.is_at_node(v, tau)`, replacing `is_parked_at_depot(...)`. **Also sets `vehicle_standing[v] = True`** |
| `policies/transformer_policy.py` `_sweep` | a pending Client equal to `last_node_reached[v]` is not a candidate — greedy branch **and** ε-exploration branch. The depot is never filtered |
| `policies/monte_carlo.py` | **untouched** |

Only the *fact* gets a name; the condition is composed inline at its single
call site, so the difference from `is_parked_at_depot` is visible where it is
read rather than hidden behind a second, confusingly similar predicate.

**`vehicle_standing = True` on park is not optional.** The widened branch will
now be reached with `standing == False`, and two downstream sites read
`is_parked_at_depot`: `terminate_state_passing_horizon` would count the parked
vehicle in `vehicles_out` and **charge overtime to a vehicle sitting at the
depot**, and `_every_vehicle_home_and_no_clients_left` would never fire,
diverting termination to its sibling path with different accounting. Setting
the flag keeps all seven other `is_parked_at_depot` call sites correct
untouched — and mirrors what `_vehicle_parks_at_depot` already does on arrival.

**Known, accepted artefact.** `begin_arc` has already pushed a velocity sample
for the outgoing arc into `observed_velocity` before the park. It never becomes
an accounting error — `advance_fleet_to` skips `PARKED` vehicles, so it is
never charged as distance — it only lingers as one observation in the window of
a retired vehicle. Documented, not reverted.

---

## What pins it

- `tests/unit/test_model_reroute.py` — a third case in
  `TestDepotParkForeverGuard`: *on the depot node with zero arc progress →
  parks, not routed to itself.* Hand-built state, deterministic, torch-free.
  **This carries the red-before/green-after** (decision 7).
- `tests/unit/test_fleet_routes.py` — `is_at_node`.
- `tests/test_invariants.py` — route well-formedness at every reroute point
  (`len(route) >= 2`), scoped the way the review scoped its own measurement.
  Green from day one under the linear Policy; it is the net, not the catch.
- End-to-end regression on **seed 1131** in the torch-gated neural tests — the
  only thing that reproduces the real failure.

## Documentation

- **ADR-0008** — the rule above, both halves, plus the second notion of "at the
  node". Cross-reference pointers are mandatory in **ADR-0005** (positional
  presence vs. `vehicle_standing`) and **ADR-0007**, whose *"every pending
  Client not already claimed, plus the depot. That is the whole rule"* is no
  longer the whole rule.
- `docs/simulator-review.md` — the entry moves out of **"Hipótesis
  descartadas"** and becomes **B20**. The review's number stands (re-measured
  at 5× the sample: 0/600); the *inference* was wrong — it is not a discarded
  hypothesis but a live defect whose reachability depends on the Policy's
  action set. Record why fuzzing cannot see it.

## Gates

- `tests/test_self_golden.py` (Tier-1 bit-exact) — expected **zero diff**: the
  widened predicate's trigger set is measured at 0/600 for the linear Policy.
  **Verify by running, not by assuming.**
- `tests/test_invariants.py`, `test_config_sweep.py`, `test_data_spine.py`,
  `test_benchmark_*`, full unit suite.
- `tests/test_golden_master.py` — **not applicable**: it re-runs the frozen
  legacy monolith, which this change does not touch.
- No golden-master re-baseline exists to do: `chengdu_full_phase2.json` was
  orphaned when simulator-correctness ticket 03 deleted
  `test_new_package_vs_golden_master.py`. The review's "any fix requires a
  golden-master re-baseline" advice predates that and is stale.

## Consequence for `neural-policy`

The Client filter changes the action set, so **every** transformer decision
changes. Any Gate A number gathered before this lands is void; ticket 08 reruns
from zero. `neural-policy/issues/08` gains a `Blocked by:` pointer to this
ticket.

## Scope

Extra latent defects surfaced while widening the suite are fixed **inside this
ticket**, not deferred (user decision, 2026-07-31). Measured risk: a
uniform-random full-action-space policy found **zero** new defects in 300
episodes.

## Comments

### Resolution (2026-07-31)

Landed exactly as scoped in "The fix":

- `simulation/fleet_routes.py` — `FleetRoutes.is_at_node(vehicle, tau) ->
  bool`, `departure_tau[vehicle] >= tau`. One call site.
- `simulation/model.py` `_reroute_for` — the park branch's condition is now
  `action[v] == depot and last_node_reached[v] == depot and
  fleet.is_at_node(v, tau)`, replacing `is_parked_at_depot(...)`, and sets
  `vehicle_standing[v] = True` on park so the seven other `is_parked_at_depot`
  call sites (overtime accounting, both termination paths) stay correct.
- `policies/transformer_policy.py` `_sweep` — a pending Client equal to
  `last_node_reached[v]` is excluded from both the greedy argmin and the
  ε-exploration candidate list, via the same `infeasible_mask` both branches
  now share (`claimed_mask | (pending_array == vehicle_position)`). The depot
  is never filtered.
- `policies/monte_carlo.py` — untouched, as predicted: its own candidate
  construction already excludes the vehicle's current node by construction.

**Red-before/green-after** (decision 7): `tests/unit/test_model_reroute.py`
gained a third `TestDepotParkForeverGuard` case that reproduces the exact
crash — a hand-built `ShortestPathCache` with the `(depot, depot)` self-row
the real CSV carries, `departure_tau == tau` (zero arc progress),
`vehicle_standing = False`. Verified against the pre-fix code (`git stash` on
the three touched source files) that it raises the documented
`IndexError: list index out of range` at `FleetRoutes.current_arc`'s
`route[1]`, then verified green after restoring the fix. One pre-existing
fixture in the same file (`test_a_vehicle_genuinely_parked_at_the_depot_is_left_parked`)
needed its `departure_tau` corrected from an arbitrary `0` to `tau` itself
(360.0) — under the old `is_parked_at_depot` check that value was irrelevant,
but the new positional-presence check reads it directly, so the fixture had
to actually represent "at the node."

**End-to-end regression**: `tests/unit/test_neural_episode.py` gained
`TestSeed1131Regression`, running `run_neural_training_episode` with seed
1131 against `chengdu_mini` (untrained network, the fixture's `epsilon: 0.1`).
Same `git stash` verification: crashes with the identical `IndexError`
pre-fix (vehicle 0, `route[1]`), passes post-fix.

**The net, not the catch**: `tests/test_invariants.py`'s `RecordingModel._reroute_for`
gained a route-well-formedness assertion (`len(route) >= 2`) for every
travelling vehicle after each reroute, scoped like the review's own
measurement. Green before and after this ticket's fix under the linear
Policy (confirms the review's 0/600), by design — the trigger needs
correlated fleet behaviour a Hypothesis-drawn or uniform-random Policy does
not reproduce (measured: 0/300 uniform-random, 0/120 hand-built adversarial
lockstep-flip-flop), so this invariant could never have caught B20 on its
own; it is documented as the net, and the hand-built unit test plus the
seed-1131 regression as the catch.

**Also added**: `tests/unit/test_fleet_routes.py::TestIsAtNode` (the new
fact in isolation), and `tests/unit/test_transformer_policy.py::TestSelfNodeNotACandidate`
(three cases: greedy exclusion via a rigged `_score` that forces the
self-node to the global minimum Q, so the assertion is on the mask, not on
the network happening not to prefer it; ε-exploration exclusion over 30 rng
seeds; and a control proving the depot itself is never filtered).

**Documentation**: `docs/adr/0008-an-action-must-be-executable.md` (the
decision, both halves, the rejected alternatives, and the "structurally
invisible to fuzzing" measurement). Cross-reference pointers added to
ADR-0005 (the `vehicle_standing`/positional-presence gap) and ADR-0007
("that is the whole rule" is no longer the whole rule). `docs/simulator-review.md`:
the "Ruta degenerada inicial `[0,0]`" bullet moved out of *Hipótesis
descartadas* into a full **B20** entry (table row + section, placed after
B19) — the review's own 120-episode measurement was correct, re-measured at
5× the sample (600 episodes, still 0), but the *inference* ("never
reachable") was wrong: reachability depends on the Policy's action set, not
on `_reroute_for` alone.

`neural-policy/issues/08` already carried its `Blocked by:` pointer to this
ticket — filed that way at ticket creation, no edit needed.

### Gates

- `tests/test_self_golden.py`: 6/6 passed, zero diff — as predicted (the
  widened predicate's trigger set is 0/600 for the linear Policy).
- `tests/test_invariants.py`: 3/3 passed (Hypothesis, `derandomize=True`).
- `tests/test_config_sweep.py`: 434/434 passed.
- `tests/test_data_spine.py`: 21/21 passed.
- `tests/*benchmark*`: 10/10 passed.
- `tests/test_golden_master.py`: not applicable, as predicted — it re-runs
  the frozen legacy monolith, untouched here.
- Full suite (`uv run pytest -q`, project default `-m "not golden"`): **4218
  passed, 3 deselected**, ~6m20s, including the `neural`-marked torch-gated
  tests (torch is installed in this environment).
- `uv run mypy src/stdvrp`: clean.
- `uv run ruff check` / `ruff format --check` on every file this ticket
  touched: clean except four pre-existing violations this ticket's diff does
  not introduce (verified via `git stash` on the touched source files and
  re-running ruff against the unstashed base) — three are the same `E501`
  gap ticket 10 already recorded (`feature_extraction.py`, `monte_carlo.py`,
  `model.py`, now at shifted line numbers) plus one pre-existing `RUF059` in
  `neural_episode.py`, a file this ticket never touches. Not this ticket's to
  fix, per the same precedent ticket 10 set.

### Scope: no extra latent defects surfaced

The widened suite (config sweep, full suite, self-golden) stayed green
throughout; nothing beyond the scoped fix and its own tests was needed.
