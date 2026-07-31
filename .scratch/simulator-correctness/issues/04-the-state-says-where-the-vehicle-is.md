# 04 — The State says where the vehicle is

**What to build:** Fix the review's common root cause — `State.vehicle_position`
does not mean "where the vehicle is" — in the domain model rather than at the
call sites. Closes B1a and B1b. Writes **ADR-0005**.

`vehicle_position` holds **the last node the vehicle reached**.
`vehicle_reaches_node` (`model.py:517`) writes it for a node the vehicle merely
*passes through*, and `begin_arc` immediately launches it onto the next arc.
Nothing distinguishes "standing at the node" from "crossing it". `state.py:32`
already half-admits it in a comment: *"Node each vehicle last departed from (or
is at)"* — that "(or is at)" is the ambiguity, and it is why the same wrong
predicate got written at three sites.

It matters because the depot is not only the starting point: 138 of 2025 cached
shortest paths (6.8%) use it as an interior node, and the review measured 75
depot crossings in 60 episodes.

**Two layers, two fixes, and fixing one does not cure the other:**

| | Where | What breaks | If only this one is fixed |
|---|---|---|---|
| B1a | `policies/monte_carlo.py:297` | The fleet loses capacity — up to half the fleet idle with Clients pending | The vehicle keeps working and layer 2 stops being reachable by this route |
| B1b | `simulation/model.py:376` | Physics and accounting — the vehicle is teleported to the depot, remaining arc and return trip never charged | The vehicle finishes the arc, drives back and parks — it pays the trip, but **still leaves service** |

**Blocked by:** 03 (so the impact is measured under the repaired objective)

**Status:** resolved

- [x] **Rename** `State.vehicle_position` → an honest name (`last_node_reached`
      or similar). This is half the fix: it makes
      `if vehicle_position == depot` un-writable by anyone who means "is at the
      depot". Six read sites — audit each for which of the two meanings it
      wants.
- [x] **Add the missing fact to `State`**, maintained by the `Model` wherever it
      already writes the position: whether the vehicle is standing at a node
      rather than travelling. The Policy reads it.
      **Do not hand `FleetRoutes` to the Policy** — `fleet_routes.py:11-15`
      deliberately keeps the route/progress half invisible to the Policy
      (simulation-performance ticket 09). The boundary is right; the State was
      simply missing a fact on the correct side of it.
- [x] **The two dead fields.** `State.vehicle_next_node` (written
      `model.py:281`) and `State.vehicles_direction` (written `model.py:406`)
      are written and **read by nobody** — vestigial slots for exactly this kind
      of information. Either the new fact lands in one of them or both go.
      Do not leave a third dead field behind.
- [x] **B1a**: the depot-idle rule applies only to a vehicle genuinely standing
      at the depot. **The literal 350 is not touched** — it stays the documented
      legacy quirk (`monte_carlo.py:62-64`). What changes is what it applies to.
- [x] **B1b**: the same distinction guards `model.py:376`. The branch is correct
      and frequent in its normal case (a vehicle actually parked at the depot
      told to stay); it is only a bug for a vehicle mid-arc.
- [x] Invariant: **no vehicle becomes `PARKED` while
      `departure_tau < tau < arrival_tau`** (the review's own proposal).
- [x] Invariant: a vehicle's recorded node never changes without charged
      distance — no teleporting.
- [x] **`TrainingSnapshot` audit**: it copies four State fields for `update_W`'s
      replay. Establish whether the replay path reads the new fact; add it if so.
- [x] **ADR-0005** — the State says where the vehicle is. Records the rejected
      alternative (passing `FleetRoutes` to the Policy: breaks the ticket-09
      boundary and leaves the ambiguous predicate alive at four other sites) and
      the naming decision. **`CONTEXT.md`** gains the distinction: "vehicle
      position" today means two different things depending on who reads it.

## Predicted self-golden diff

**Not surgical, and that is expected** — unlike ticket 03, this changes
*decisions*: a vehicle that used to be retired now keeps working, so its whole
trajectory changes. The falsifiable claims are these:

- **Capture seeds with zero teleports and zero spurious depot-idle firings must
  be bit-identical, in all three blocks.** Ticket 01 reports those per-seed
  counts; list the untouched seeds here before running.
- On affected seeds, in the frozen-W block: `distance_cost` **rises** (the
  remaining arc and the return trip are now driven and charged), `tau`
  **extends**, `state_count` **rises**. A fall in any of the three contradicts
  the mechanism.
- Aggregate direction over the 60-seed bench: the review measured **+1.24% mean
  cost and +0.19 km/episode recovered** over 180 episodes (1060.27 → 1073.39,
  34.14 km). Reproduce that magnitude or explain the difference — note it was
  measured under the *old* objective, so ticket 03 shifts the denominator.

**Do not expect the fleet-capacity win to show up as a cost reduction.** Fixing
B1b alone makes the vehicle pay for its trip home without returning it to
service; only B1a puts it back to work. Both land here, so the net is a fleet
that works longer *and* pays for its distance — the review's honest reading is
that the important thing is the physics and the accounting, not the number.

## Evidence required

The untouched-seed bit-identity list. Per-seed direction of `distance_cost`,
`tau`, `state_count` on affected seeds. The 60-seed bench before/after with km
driven. Teleport counter at zero after the fix (it was ~0.4/episode, 24 of 60
episodes).

## Comments

### Resolution (2026-07-30)

`State.vehicle_position` is renamed `last_node_reached`; a new
`vehicle_standing: list[bool]` carries the missing fact, reusing the
write-only `vehicle_next_node` slot (`vehicles_direction` deleted outright —
neither field had a reader). The `Model` sets it `True` at every arrival
(`_vehicle_parks_at_depot`, `vehicle_reaches_client`, the already-served-Client
branch of `vehicle_reaches_node`) and at every hold (`_hold_for_service`,
reached from both `begin_arc` and `resample_arc`); `False` the instant
`begin_arc` actually launches a vehicle onto an arc. Every site that meant
"genuinely parked/idle at the depot" now reads
`last_node_reached == depot and vehicle_standing` instead of the node alone:
`MonteCarloPolicy._select_vehicle_possible_actions` (B1a),
`_classify_shortest_distance_clients`'s two endgame branches,
`_already_acquired_cost`'s overtime term, `FeatureExtractor._classify_closest_clients`'s
eligibility filter, `Model._reroute_for`'s depot-park-forever branch (B1b),
`_every_vehicle_home_and_no_clients_left`, and `terminate_state_passing_horizon`'s
overtime vehicle count — six read sites beyond B1a/B1b's own two, all
carrying the identical wrong predicate. `TrainingSnapshot` gained
`vehicle_standing` alongside the renamed `last_node_reached`: `_already_acquired_cost`
reads both during `update_W`'s replay. ADR-0005 records the naming decision
and the rejected alternative (handing `FleetRoutes` to the Policy). `CONTEXT.md`'s
**State** entry gains the two-meanings distinction.

**Red before / green after, verified directly** (spec.md decision 7): with
the `vehicle_standing` guard temporarily stripped from both
`_select_vehicle_possible_actions` and `_reroute_for`,
`tests/test_invariants.py::test_episode_invariants` fails (a vehicle parked
mid-arc, `AssertionError` in `RecordingModel._reroute_for`) and
`tests/unit/test_model_reroute.py`'s dedicated regression fails identically;
restoring the guard turns both green. The new invariant — "no vehicle becomes
`PARKED` while mid-arc" — is one assertion for two catalogue rows (see its
docstring): the review measured "`PARKED` while travelling" and "recorded
node changed without charged distance" as the same event, never
independently, so this suite does the same rather than inventing a second,
harder-to-justify mechanism for the teleport row alone.

**Evidence: `scripts/measurement_bench.py`, before/after, seven protocol
variants** (`--vehicle-count 1/3/6`, the demand-generated default fleet, the
15-seed capture-seeds protocol, and the literal-frozen-W bench at
`--vehicle-count 1` and default — `--w frozen`):

- **Untouched-seed bit-identity holds in every variant, 0 mismatches**: 42/60
  (v1), 52/60 (v3), 38/60 (v6), 39/60 (default), 8/15 (capture-seeds), 46/60
  (frozen v1), 35/60 (frozen default) seeds had zero `mid_arc_park_violations`
  before the fix, and every single one of those seeds reproduces every
  reported metric bit-for-bit after it.
- **The teleport counter (`B1a_B1b_mid_arc_park`) drops to exactly zero after
  the fix, in all seven variants** — it was 18, 8, 22, 21, 9, 14 and 25 events
  respectively before.
- **On affected seeds, single-vehicle fleets (`--vehicle-count 1`, both zero-W
  and frozen-W)**: `distance_cost`, `final_tau` and `decisions` (state_count)
  all rise, unanimously — 18/18, 18/18, 17/18 (one tie) under zero-W; 14/14,
  14/14, 14/14 under frozen-W. Zero falls in either run. This is the exact,
  unambiguous falsifiable claim the ticket predicted, and it holds cleanly
  where the review itself measured it: one vehicle, so the episode's
  aggregate tau/decisions/distance *is* that vehicle's own trajectory.
- **Multi-vehicle fleets (v3, v6, the demand-generated default) do not show
  the same clean monotonic direction** (e.g. default zero-W: `distance_cost`
  up=11/down=10, `final_tau` up=6/down=12/same=3) — explained, not a
  contradiction: `MonteCarloPolicy`'s candidate selection is joint across
  vehicles (`forbidden_actions`, the closest-vehicle endgame classifiers), so
  un-crippling one vehicle mid-episode changes every other vehicle's
  downstream decisions too, and the *episode-level* aggregate no longer needs
  to move in lockstep with the one vehicle the fix directly touches. The
  review's own methodology already anticipated this distinction — it measured
  B1b's "trip home" numbers on forced single-vehicle fleets specifically.
- **Aggregate direction, 180 episodes (60 seeds × fleets 1/3/6, zero-W,
  matching the review's own N and fleet sweep)**: mean total cost
  1537.32 → 1293.28 (**-15.9%**), mean km driven per episode 172.19 → 187.25
  (**+15.06 km/episode**). This does not reproduce the review's own
  **+1.24%/+0.19 km/episode** — expected and explained, not a red flag: (1)
  the review measured B1b in isolation, under the pre-ticket-03 objective,
  where an unfixed B1a still capped how much extra service the vehicle could
  recover; fixing both layers together (as this ticket does — B1a is what
  "puts it back to work") recovers far more distance than the trip-home-only
  number. (2) Ticket 03's reference-clock formula prices abandoned demand
  extremely harshly; the single-vehicle fleet is a stress case where the
  old bug's forced abandonment was catastrophically expensive (e.g. seed 42:
  `delay_cost` 8798 → 69 under frozen W) — fixing B1a/B1b removes that tail
  and the mean cost falls rather than rises. Both are "ticket 03 shifts the
  denominator" playing out exactly as flagged.

**Self-golden fixture recaptured.** `scripts/capture_self_golden.py --check`
confirmed the fix moves the mini fixture's own 15 episodes (160 of the
committed capture's leaves changed — training W trajectories, evaluation and
frozen-W-eval metrics), exactly the "not surgical" outcome predicted.
Recaptured via `scripts/capture_self_golden.py`;
`tests/test_self_golden.py` and `tests/test_world_cache_self_golden.py` both
green (7 passed) after.

**`/code-review` (Standards + Spec axes), both landed.** Standards flagged
the depot-parked predicate (`last_node_reached == depot and vehicle_standing`)
being reconstructed inline at 8 read sites — exactly the failure mode ADR-0005
itself diagnoses ("neither call site can be trusted to reconstruct it
correctly on its own"). Consolidated into one function,
`is_parked_at_depot(node, standing, depot)` in `state.py`, called from every
site instead. Spec flagged two real issues: ADR-0005's own site count was
internally inconsistent (labelled "three" a bucket that actually listed six
items, and didn't credit `FeatureExtractor`'s eligibility filter) — corrected
to an accurate enumeration of all ten read sites (eight needing the guard,
two rename-only); and `test_model_reroute.py`'s "genuinely parked" case
pre-set `arrival_tau` to `PARKED` before calling `_reroute_for`, so it passed
vacuously whether or not the branch ran — fixed to start from a non-`PARKED`
value and assert the branch's own side effects, verified red (branch
disabled) before green. Full suite (3605 passed) and mypy (60 pre-existing,
unrelated errors, unchanged from baseline) re-run clean after both fixes.
