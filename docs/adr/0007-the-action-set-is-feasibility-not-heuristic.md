---
status: accepted
---

# The action set is feasibility, not heuristic

`neural-policy/spec.md` (ticket 06) builds `TransformerMonteCarloPolicy`: the
`TrainablePolicy` that decides one vehicle at a time using the ticket-05
`TokenEncoder`/`QHead` in place of the linear `MonteCarloPolicy`'s `X · W`.
`MonteCarloPolicy`'s candidate set — `_select_vehicle_possible_actions`,
`_classify_shortest_distance_clients`, the `delayed_clients` classifier, the
`350`/`310` depot-idle literals, `number_actions_test` — is hand-engineered
ranking, not a rule about legality: it exists to shrink a large action space
down to a handful of plausible candidates for a *linear* model with 19
features, where scoring every pending Client would be both expensive and
uninformative to a model that cannot represent the relevant nonlinearities
anyway. None of that reasoning applies to a network that scores every
candidate directly.

## Decision

`TransformerMonteCarloPolicy`'s action set is **feasibility, not heuristic**:
every pending Client not already claimed by another vehicle *this decision*,
plus the depot, which is always legal. That is the whole rule
(`src/stdvrp/policies/transformer_policy.py`, `_sweep`). No nearest-`k`
shortlist, no delayed-Client classifier, no depot-idle cutoff.

- **The no-double-booking rule survives as a constraint, not a heuristic
  side-effect.** `_sweep` tracks which pending Clients earlier-indexed
  vehicles have already claimed *this decision* and excludes them from the
  argmin outright — the B11 invariant ("two vehicles never receive the same
  non-depot Client in one decision") is enforced structurally, the same way
  it is checked. `claimed` is *also* fed to the network as an ordinary input
  (spec.md decision 6: "claimed enters at the head, not the encoder"), so a
  trained network's predictions can learn to route around contention — but
  legality never depends on what the network outputs for an already-claimed
  candidate, only on the hard mask.
- **`number_actions_test` has no meaning here.** It is an experimental axis
  for the linear baseline's candidate-pool width; this Policy always scores
  every pending Client. The effort's comparison (spec.md, "Comparison
  budget") therefore runs the baseline's own **best** `test_action_count`
  against this Policy — the baseline's best shot, not its narrowest.
- **The depot needs its own Q value, and gets one the same way every other
  candidate does.** `QHead` scores `Embeddings.clients` — one row per pending
  Client — so the depot, never a Client, has no row to score. `_score` builds
  one: `encoder.arc_embed([minutes_to_depot, length_to_depot] /
  horizon_length)` for the arc half (the same computation every real
  candidate's arc embedding goes through, using `EpisodeGeometry` directly —
  permitted under the observability rule, ADR-0006, since it is the same
  offline prior the linear baseline already reads), concatenated with the
  vehicle's own context embedding for the other half (the depot's meaning —
  "return to base" — is a fact about the vehicle, not about a destination
  with time windows or attention-refined context of its own). At
  initialization this reproduces the myopic warm start (ticket 05) exactly:
  `Q(v, depot) == minutes_to_depot / horizon_length`, so the untrained greedy
  policy is "go to whichever feasible target — Client or depot — is nearest",
  not "nearest Client, with the depot as an unscored afterthought".

## Considered options

- **Keep a shrunk candidate set for the transformer too, for a closer
  architectural parity with the baseline.** Rejected: the whole point of
  spec.md's Gate B is to attribute a win (or a loss) to the *approximator*.
  A shrunk, hand-tuned candidate list is exactly the kind of hand-engineering
  this effort is trying to remove — keeping one would leave any result
  ambiguous between "the network is a better approximator" and "the network
  inherited a candidate list built for the old one, which happened to help
  (or hurt) here too."
- **Give the depot no Q value; treat it as a fallback used only when no
  Client is feasible.** Rejected: spec.md's own pseudocode lists the depot
  as part of the feasible set every decision, not a last resort — and a
  network that cannot express "go home now, even though Clients remain" (a
  real decision the linear baseline can already make) is strictly less
  capable than the Policy it is meant to replace.
- **Synthesize the depot's "context" half from the episode's global token
  instead of the vehicle's own embedding.** Considered and rejected on
  grounds of fit, not correctness — either choice is invisible at
  initialization (row 0 of `QHead.layer1` reads only the arc dimension) and
  only starts mattering once training moves the background rows. The
  vehicle's own state (how deep into the shift it is, how loaded its
  itinerary is) is a more directly relevant signal for "should I go home"
  than an Episode-wide summary shared by every vehicle.

## Consequences

- `TransformerMonteCarloPolicy` never imports `monte_carlo.py`, and
  `monte_carlo.py` is untouched by this ticket — the predicted self-golden
  diff is exactly zero.
- `_already_acquired_cost` (the Monte Carlo target's sunk-cost baseline) is
  duplicated from `MonteCarloPolicy`, not shared, to keep that file
  untouched — see `transformer_policy.py`'s module docstring for the full
  reasoning.
- Every existing invariant that applies to any Policy — in particular B11
  (no double booking) and B5 (a legal action for every vehicle at every
  `tau`, since the depot is always feasible even with zero pending Clients)
  — holds for this Policy by construction, not by inheriting
  `MonteCarloPolicy`'s own guards against them.
