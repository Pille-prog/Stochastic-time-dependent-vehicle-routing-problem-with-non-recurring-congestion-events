---
status: accepted
---

# What the Policy is allowed to see

`neural-policy/spec.md` replaces the linear `MonteCarloPolicy`'s approximator
(`Q = X * W` over 19 hand-engineered features) with a transformer that reads raw
`State` facts and produces `Q(s, a)` directly. That substitution is only a fair
test of the approximator if both Policies observe the same world. The linear
Policy's 19 features — and the candidate-set heuristic that built its action
list — are already, structurally, incapable of seeing anything past the static
`ShortestPathCache`: no live velocity, no congestion event, ever reaches
`FeatureExtractor`. A transformer that reads raw `State` facts *could* reach
further, because nothing about "raw facts" limits it to what the legacy
happened to compute. Left unconstrained, ticket 04's tokenizer would eventually
gain a "just add the live congestion field, it obviously helps" — and at that
point a win would prove nothing about the approximator, only that one side was
handed information the other was not.

## Decision

The Policy reads **only**: this Episode's `State` (`tau_episode`,
`clients_not_visited`, `last_node_reached`, `vehicle_standing`,
`vehicle_completing_service`, `observed_velocity`), the Episode's `TimeWindows`,
the static `EpisodeGeometry` matrices, and configuration clocks
(`horizon_start_minute`, `shift_end_minute`, `episode_end_minute`). Explicitly
forbidden, in any form: `EpisodeVelocities`, `congested_arcs`, `TravelTimeModel`
evaluated at `tau`, `FleetRoutes`, and any velocity the fleet has not itself
observed.

`tokenize`'s signature (`src/stdvrp/policies/tokenizer.py`) is the rule made
executable: those five inputs and nothing else. `tests/unit/test_tokenizer.py::TestObservabilityRule`
pins both the signature (an exact allow-list — a future ticket that "just adds"
a field must edit that test, not slip past it) and the module's own imports
(nothing from `stdvrp.simulation.episode_velocities`,
`stdvrp.traffic.travel_time_model` or `stdvrp.simulation.fleet_routes`).

### Why the live traffic feed was rejected

`docs/research/rl-methodology-for-stdvrp.md` ranks live congestion awareness as
the single highest-leverage change available (#1, F1) — and that ranking is not
disputed here. It probably would help. It is rejected from this Policy's inputs
anyway, because the effort's acceptance contract (spec.md, Gate B) is a
head-to-head comparison against the linear baseline, and that comparison is
only informative if a win can be attributed to the approximator. Handing the
transformer live congestion the linear Policy structurally cannot see would
make any win ambiguous between two different causes — a strictly better function
approximator, or strictly better information — and the whole point of this
effort is to answer which one the observation set actually is (the research
note's own central claim, which spec.md commits to testing rather than
assuming). The transformer is deliberately exactly as congestion-blind as the
linear baseline, so that question stays answerable.

### Why `EpisodeGeometry.average_minutes` is permitted

It looks like it should be forbidden — it is, after all, "the travel time
between two nodes", and the model that produces live travel times
(`TravelTimeModel`) is on the forbidden list. The distinction is *when* the
number was computed, not what it measures: `average_minutes` is baked into the
`ShortestPathCache` CSV once, offline, from historical traffic — it is a fixed
prior about the road network's typical geometry, not an observation of this
Episode's actual congestion. It is also the identical object the linear
baseline reads (`FeatureExtractor`'s every distance/ETA feature). Forbidding it
would not make the comparison fairer, since both Policies already share it
equally; it would only leave the network with no notion of distance at all,
which is not congestion-blindness, it is geometry-blindness.

### The cost constants are configuration, not observation (2026-08-01)

The 2026-08-01 amendment to spec.md decision 1 (ticket 08) put the four
projected components of the simulator's cost function on the `(client,
vehicle)` arc tokens. That amendment lives in decision 1 — the *purity* rule —
and changes nothing here, but it leans on one clarification this ADR should
state explicitly: the cost **rate constants** (`EARLINESS_COST_RATE`,
`DELAY_COST_RATE`, `OVERTIME_COST_RATE`, `SERVICE_MINUTES` — the values
`cost_ledger.py` and `Model` hardcode) are in the same class as the
configuration clocks already on the allow-list. They are fixed problem
definition, known before any Episode runs, identical for both Policies — not
an observation of this Episode's world. Every projected cost is arithmetic
over `tau`, the time windows and `EpisodeGeometry.average_minutes` (all
already permitted) times those constants; `tokenize`'s five-argument
signature is unchanged and the structural test still pins it. Nothing from
the forbidden list — `EpisodeVelocities`, `congested_arcs`,
`TravelTimeModel` at `tau`, `FleetRoutes` — is any closer to the Policy than
before.

### The only admissible congestion-aware arm

If Gate B fails on the congestion-blind Policy (ticket 09), ticket 10's
conditional arm 2 is the one form of congestion awareness the observability
rule still allows: the fleet's own **shared observation memory** — pooling
what this Episode's vehicles have themselves measured
(`state.observed_velocity`), not reading the world's velocity field. A
dispatcher may aggregate its own vehicles' reports; it may not see congestion
no vehicle of theirs has driven through. This keeps arm 2 in the same
observability class as arm 1 — richer pooling of the fleet's own measurements,
never a new channel into `EpisodeVelocities`.

## Considered options

- **Let the tokenizer take `EpisodeVelocities`/`congested_arcs` and leave it to
  code review to keep them unused.** Rejected: "unused today" is not a
  guarantee, and the ticket's own stated failure mode is a *future* ticket
  adding a congestion read because it would obviously help — exactly the
  pressure a live traffic feed would keep applying every time someone tuned
  Gate A calibration. A field that exists on the signature will eventually be
  read; the fix is to not hand it in.
- **Forbid `EpisodeGeometry.average_minutes` too, for the cleanest possible
  "zero congestion information" story.** Rejected: it is not congestion
  information (see above), and the linear baseline depends on it identically —
  removing it from one side while the other keeps it would itself have broken
  the parity this ADR exists to protect.
- **Enforce the rule by convention/code review only.** Rejected for the same
  reason ADR-0005 gives for `is_parked_at_depot`: a correct rule enforced only
  by everyone remembering it is the exact failure mode that produces a
  violation eventually. A structural test that inspects the tokenizer's own
  signature and imports closes that risk instead of relocating it to reviewer
  vigilance.

## Consequences

- `tokenize` (`src/stdvrp/policies/tokenizer.py`) takes exactly five arguments;
  adding a sixth is a deliberate, reviewable act, not a silent widening.
- Ticket 05's network and ticket 06's Policy can only ever be as
  congestion-aware as this module lets them be — the rule is enforced once,
  upstream of every consumer, rather than re-checked at each one.
- Gate B's result (ticket 09), win or lose, is attributable to the
  approximator change alone. A loss cannot be blamed on "it wasn't given
  enough information" without that becoming ticket 10's explicit, pre-committed
  next question — not a retroactive excuse.
