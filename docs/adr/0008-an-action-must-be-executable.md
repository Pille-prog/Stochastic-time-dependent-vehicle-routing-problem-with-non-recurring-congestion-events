---
status: accepted
---

# An action must be executable

Ticket 11 (simulator-correctness, B20) closed a crash in `Model._reroute_for`'s
at-a-node branch, reached by any Policy whose action set can name the node a
vehicle is currently on. Found by `neural-policy` ticket 08 while training the
transformer on Chengdu; not introduced by that ticket's diff — it is a gap
this effort's own earlier fixes (ADR-0005, ADR-0007) left open.

## The gap

ADR-0005 defines `vehicle_standing` to flip to `False` the moment `begin_arc`
launches the vehicle onto an arc. That leaves one instant — *"I am on the
node, I have not moved a metre"* (`departure_tau == tau`, zero arc progress)
— where the simulator simultaneously says:

- *at the node* (`departure_tau == tau`) → the at-a-node routing branch runs;
- *not standing* (`vehicle_standing == False`) → the depot-park-forever guard
  (`is_parked_at_depot`, ADR-0005) does not fire, because it requires
  `standing`.

If the decision at that exact instant names the node the vehicle is already
on, `_reroute_for` falls through to the mid-arc/at-a-node reroute branch and
asks the `ShortestPathCache` for the path from that node to itself.
`all_shortest_paths.csv` contains a self-row for every node (`0,0,0,0.0,0.0`),
so `path_between(n, n)` does not raise — it returns a well-formed, one-node,
zero-length path. `FleetRoutes.current_arc` then reads `route[1]` off that
one-node route and dies with `IndexError`.

Today only the depot is reachable this way, because it is the only node
always in the action set (ADR-0007). A Policy whose action set can also name
a pending Client the vehicle is standing on (having crossed it without
serving it) reaches the identical crash for that Client.

**Note on `test_invariants.py`:** the invariant already defines "genuinely
mid-arc" with a strict inequality (`departure_tau < tau < arrival_tau`), so
this zero-progress instant is deliberately outside it — parking there does
not violate ADR-0005's own invariant. The gap is real, just orthogonal to
what that invariant checks.

## Decision

**An action must be executable.** Concretely:

- **The depot is always in the action set, but it means two different
  things**: *park* when the vehicle is on the depot node, *travel* when it is
  not. Only the simulator can tell those apart (the Policy cannot see
  `FleetRoutes`), so the simulator makes the distinction.
- **A node the vehicle is already on is not a travel destination.** For a
  pending Client that is a Policy-side feasibility rule, not a heuristic:
  there is nothing to travel to.

"Can park" is therefore **positional presence** — on the node, zero arc
progress — *not* `vehicle_standing`. `FleetRoutes.is_at_node(vehicle, tau)` =
`departure_tau[vehicle] >= tau` names that fact where the progress it reads
already lives.

### Where each half lands

| where | what |
|---|---|
| `simulation/fleet_routes.py` | `is_at_node(vehicle, tau) -> bool`, the missing fact |
| `simulation/model.py` `_reroute_for` | park branch: `action[v] == depot and last_node_reached[v] == depot and fleet.is_at_node(v, tau)`, replacing `is_parked_at_depot(...)`. Also sets `vehicle_standing[v] = True` |
| `policies/transformer_policy.py` `_sweep` | a pending Client equal to `last_node_reached[v]` is not a candidate — greedy branch **and** ε-exploration branch. The depot is never filtered |
| `policies/monte_carlo.py` | untouched — its own candidate rules already exclude a vehicle's current node by construction |

Only the *fact* gets a name (`is_at_node`); the condition itself is composed
inline at its single call site, so the difference from `is_parked_at_depot`
stays visible where it is read rather than hidden behind a second,
confusingly similar predicate — the same lesson ADR-0005 already drew about
reconstructing a compound condition at more than one site.

**`vehicle_standing = True` on park is not optional.** The widened branch is
now reached with `standing == False`, and several `is_parked_at_depot` call
sites downstream read it: `terminate_state_passing_horizon` would count the
parked vehicle in `vehicles_out` and charge overtime to a vehicle sitting at
the depot, and `_every_vehicle_home_and_no_clients_left` would never fire for
it. Setting the flag keeps every other `is_parked_at_depot` call site correct
untouched, and mirrors what `_vehicle_parks_at_depot` already does on
arrival.

**Known, accepted artefact.** `begin_arc` has already pushed a velocity
sample for the outgoing arc into `observed_velocity` before the park. It
never becomes an accounting error — `advance_fleet_to` skips `PARKED`
vehicles, so it is never charged as distance — it only lingers as one
observation in the window of a retired vehicle. Documented, not reverted.

## Considered and rejected

- **Gate the depot Policy-side on `vehicle_standing`.** Literal, cheapest,
  leaves `model.py` untouched, and covers the Client case for free. Rejected:
  it also removes the depot from the action set for vehicles *genuinely
  rolling past it*, where "go home" is a perfectly well-defined travel action
  the simulator already executes correctly — 0.67% of decisions, measured,
  and precisely the end-of-shift decision where being wrong costs overtime.
  It is the mirror of B1a. It also leaves the action set empty when no
  Clients are pending (B5; latent, 0 observed).
- **A no-op reroute** when the target is the current node. Rejected:
  silently discards a decision the model can execute.
- **A well-formed zero-length self arc `[n, n]`.** Rejected: new physics
  (travel time 0, a congestion lookup on a link that does not exist) and an
  arrival-at-τ spin risk.
- **A new `State` fact visible to both sides**, so offer and execution use
  literally one predicate. Sound, and the purist reading of ADR-0005's own
  lesson — but it changes what the Policy sees, touches `State`,
  `TrainingSnapshot` and ADR-0005, for a case the simulator can settle alone.

## Consequences

- `TransformerMonteCarloPolicy`'s action set (ADR-0007: "every pending Client
  not already claimed, plus the depot — that is the whole rule") is no
  longer the whole rule: a pending Client equal to the vehicle's own node is
  excluded too. ADR-0007 is amended by cross-reference rather than rewritten
  — the feasibility-not-heuristic framing still holds, it now has a second
  clause.
- ADR-0005's `is_parked_at_depot` (`last_node_reached == depot and
  vehicle_standing`) stays correct for every read site it already served —
  the seven other call sites are untouched. Its one gap was `_reroute_for`'s
  own park branch, which no longer calls it; every other site's question
  ("is this vehicle idle/home *right now*, as opposed to merely having
  arrived and possibly since moved on") is still exactly what
  `vehicle_standing` answers, and this ADR does not change that.
- Structurally invisible to fuzzing: the trigger needs *correlated* fleet
  behaviour — vehicles launched together on prefix-sharing shortest paths,
  kept synchronised by `EpisodeVelocities`' per-(arc, minute) memo, arriving
  at the same node at the same instant. Measured: 0 crashes in 300
  uniform-random full-action-space episodes and 120 hand-built adversarial
  "lockstep flip-flop" episodes — uniform randomness destroys the
  correlation a real greedy Policy produces for free. `tests/test_invariants.py`
  gained a general net (route well-formedness, `len(route) >= 2`, at every
  reroute point) precisely because a targeted Hypothesis strategy would not
  have caught this either; the actual catch is a hand-built unit test
  (`tests/unit/test_model_reroute.py`) plus an end-to-end regression on the
  observed seed.
- `tests/test_self_golden.py` (Tier-1 bit-exact) is unaffected: the widened
  predicate's trigger set is measured at 0/600 for the linear `MonteCarloPolicy`
  (re-measured at 5× `docs/simulator-review.md`'s original sample).
- Every transformer decision changes once the Client filter lands (the action
  set itself changed), so any `neural-policy` Gate A number gathered before
  this ticket is void.
