---
status: accepted
---

# The State says where the vehicle is

The simulator-review's B1a/B1b (`docs/simulator-review.md`) trace to one shared
root cause: `State.vehicle_position` held **the last node a vehicle reached**,
not where it is. `vehicle_reaches_node` wrote it for a node the vehicle merely
*passed through* on the way somewhere else, and `begin_arc` immediately
launched the vehicle onto the next arc — nothing distinguished "standing at
the node" from "crossing it". `state.py`'s own comment half-admitted the
ambiguity: "Node each vehicle last departed from (**or is at**)".

This is not a corner case: the depot is not only the fleet's starting point —
138 of 2025 cached shortest paths (6.8%) use it as an *interior* node, and the
review measured 75 depot crossings in 60 episodes. Every one of them left
`vehicle_position == depot` for a vehicle that was, physically, mid-arc
somewhere past it. Three call sites read that ambiguity as "idle at the
depot" and acted on it:

- `MonteCarloPolicy._select_vehicle_possible_actions` (B1a) offered the depot
  as the vehicle's *only* candidate action once `tau > 350`, taking a working
  vehicle out of service by accident — up to half the fleet, measured.
- `Model._reroute_for` (B1b) read the same ambiguity and set
  `fleet.arrival_tau = PARKED` outright: the remaining arc and the drive home
  were never driven or charged, and the vehicle could never revive
  (`is_travelling` stays `False` for good).
- `MonteCarloPolicy._classify_shortest_distance_clients`'s two endgame
  branches, `FeatureExtractor._classify_closest_clients`'s eligibility filter,
  `_already_acquired_cost`'s overtime term, `Model._every_vehicle_home_and_no_clients_left`
  and `terminate_state_passing_horizon`'s overtime vehicle count all carried
  the same "`position == depot`" shorthand for "idle" or "home".

Fixing B1a alone stops the fleet from losing capacity but leaves B1b
independently reachable (the "no Clients left" branch sends every vehicle to
the depot regardless of the 350 literal). Fixing B1b alone makes the vehicle
pay for the trip home without returning it to service. Both need the same
underlying fact; neither call site can be trusted to reconstruct it
correctly on its own, which is why three of them had already gotten it wrong.

## Decision

Fix the domain model, not the call sites. `State` gains two honestly-named
things instead of one ambiguous one:

- **`last_node_reached`** (renamed from `vehicle_position`) means exactly what
  it says: the last node the vehicle reached, nothing more. It is *not*
  "where the vehicle is" — a vehicle can be strictly mid-arc with this
  field unchanged since the last node it passed.
- **`vehicle_standing`** (`list[bool]`, new) is the missing fact: is the
  vehicle actually standing at `last_node_reached` — parked, serving, or
  holding — rather than travelling past it? The `Model` maintains it at every
  transition that changes either fact: `True` at every arrival
  (`_vehicle_parks_at_depot`, `vehicle_reaches_client`, the already-served-Client
  branch of `vehicle_reaches_node`) and at every hold (`_hold_for_service`,
  reached from both `begin_arc` and `resample_arc`); `False` the moment
  `begin_arc` actually launches the vehicle onto an arc. `vehicle_reaches_node`'s
  crossing branch writes `last_node_reached` but deliberately leaves
  `vehicle_standing` for the `begin_arc` call immediately after to decide —
  which is the exact moment the old code had no such decision to make.

Every site above now calls one function instead of reconstructing
`last_node_reached == depot and vehicle_standing` inline:
`is_parked_at_depot(node, standing, depot)`, a plain module-level function on
`state.py` (not a `State` method — it takes the two facts and the depot id
directly, so it reads the same off a live `State`, a `TrainingSnapshot`, or
an already-extracted array row). The ADR's own diagnosis ("neither call site
can be trusted to reconstruct it correctly on its own") applies exactly as
much to a *correct* boolean expression copy-pasted eight times as it did to
the original wrong one; one function closes that risk instead of relocating
it.
Two previously write-only fields disappear rather than leaving a third dead
slot behind: `vehicles_direction` is deleted outright (nothing ever read it);
`vehicle_next_node`'s slot is reused for `vehicle_standing` (nothing read the
old contents either — repurposing costs nothing a fresh field would not,
and keeps `State` from accumulating a second corpse).

`TrainingSnapshot` gains `vehicle_standing` alongside the renamed
`last_node_reached`: `_already_acquired_cost`'s overtime term reads both
during `update_W`'s replay, so the fact belongs on the snapshot exactly as
the four fields already there do.

The literals this touches (350, 310) are untouched — spec.md decision 3 (fix
what misclassifies or crashes, never re-tune what is tuned). What changes is
what they apply to, not their values.

## Considered options

- **Hand `FleetRoutes` to the Policy**, so it can read `departure_tau`/
  `arrival_tau` directly instead of a State-side fact. Rejected: `FleetRoutes`
  deliberately keeps the route/progress half of the per-vehicle picture
  invisible to the Policy (simulation-performance ticket 09) — the Policy
  sees positions and directions, the Model alone sees routes and progress.
  Piercing that boundary here would fix B1a's own read site but leave the
  identical ambiguous predicate alive at the four other sites listed above,
  which do not all have a natural reason to depend on `FleetRoutes`. The
  boundary was right; the State was simply missing a fact on the correct
  side of it.
- **Infer standing from `departure_tau >= tau`** at each read site instead of
  a stored fact. Rejected for the same reason as above (it still requires
  Policy-side access to `FleetRoutes`) and because it duplicates the same
  derivation at every call site rather than computing it once where the
  transition already knows it.
- **Rename only, no new fact.** Makes `if last_node_reached == depot` an
  honest but still-wrong question at the sites that actually meant "is
  standing there" — renaming alone does not fix B1a or B1b, only makes the
  bug legible.

## Consequences

- Every read site was audited (the rename's blast radius). Eight want
  "genuinely parked at the depot" and needed `is_parked_at_depot`: B1a
  (`_select_vehicle_possible_actions`), B1b (`_reroute_for`'s
  depot-park-forever branch), `_classify_shortest_distance_clients`'s two
  endgame branches, `_already_acquired_cost`, `_every_vehicle_home_and_no_clients_left`,
  `terminate_state_passing_horizon`'s overtime vehicle count, and
  `FeatureExtractor._classify_closest_clients`'s eligibility filter. Two want
  "the node to route from" and needed only the rename: `_reroute_for`'s own
  routing reads (the at-a-node and mid-arc branches, once the vehicle is
  already known not to be the depot-park-forever case above), and
  `FeatureExtractor`'s distance/ETA features (`vehicle_minutes`,
  `vehicle_length`) — the latter is B6's separate, already-documented defect:
  measuring ETA from a stale node the simulator does not honor is not this
  ticket's fix.
- Fixing B1a stops the fleet capacity loss; fixing B1b stops the teleport and
  the uncharged trip home. Doing only one leaves the other reachable
  independently (see the "no Clients left" case in the review), so both land
  together here.
- Not surgical, by design: a vehicle that used to be retired by accident now
  keeps working, so its whole trajectory changes downstream. `CONTEXT.md`
  gains the distinction between the two meanings "vehicle position" used to
  conflate.
- New invariant (the review's own proposal, `tests/test_invariants.py`): no
  vehicle becomes `PARKED` while mid-arc. One check stands for two catalogue
  rows — "`PARKED` while travelling" and "recorded node changed without
  charged distance" — because the review measured them as the same event,
  never independently.

**See also ADR-0008.** `vehicle_standing` flips to `False` the instant
`begin_arc` launches a vehicle — the same instant `departure_tau == tau` also
holds, zero arc progress. `is_parked_at_depot` (which requires `standing`)
therefore misses that one instant; `_reroute_for`'s own park branch answers
"can this vehicle park here" from positional presence
(`FleetRoutes.is_at_node`) directly instead of from this ADR's predicate.
Every other `is_parked_at_depot` call site is unaffected — this is a gap in
one read site, not in the fact itself.
