---
status: accepted
---

# The action set is shared, not owned

Reverses **ADR-0007** ("the action set is feasibility, not heuristic").

## The gap ADR-0007 left, and what closed it

ADR-0007 retired `MonteCarloPolicy._select_vehicle_possible_actions` for
`TransformerMonteCarloPolicy` as "hand-engineered ranking built for the linear
baseline's small candidate pool", keeping only feasibility (no double booking,
no self-node, the depot as a last resort). The argument was never measured
against the baseline's own numbers until `neural-policy` ticket 01's sweep was
read against ticket 08's diagnosis ("the likely cause is the candidate count").

Measured directly on the 50 `evaluation_seeds`
(`.scratch/neural-policy/results/baseline_null_50.py`): the action count is
worth **12.68%** to the linear baseline itself — `m+40` **2168.39** against
`m+2` **2483.24**, cheaper on 36/50 seeds, Wilcoxon **p = 8.24e-05**. On
`test_seeds` the same axis shows 2.1% (3458.4 → 3384.8, ticket 01): the
*effect* reproduces on both seed sets, its *magnitude* does not transfer. A
`k`-nearest shortlist is a regularizer against long myopic hauls (`c(s, a)`
can send a vehicle far across the map to a Client about to breach its window —
myopically correct, globally ruinous), not a crutch for a linear model that
cannot see far.

**A hypothesis carried for one afternoon and killed by measurement, recorded
because it is the kind that gets re-derived.** `MonteCarloPolicy._create_W` is
`np.zeros(19)`, so it was argued `W = 0` must be "go to the nearest allowed
Client" — every candidate scores 0, `argmin` returns index 0,
`_closest_allowed_clients` orders nearest-first. **Measured: `W = 0` @ `m+2`
scores 30 791.43.** `_select_vehicle_possible_actions`'s branch 3 runs
`possible_actions = list(set(possible_actions))` *after* the nearest-first
sort, and node ids are arbitrary ints, so the dedup returns hash-table order:
`W = 0` picks an **arbitrary** feasible Client, not the nearest. The linear
baseline has no cheap myopic null, and the candidate-set-versus-ranking
decomposition could not be had before ticket 14 existed — comparing
baseline@`m+2` against cost-greedy@~151 varies the action set *and* the
ranking rule at once.

**The second reason is fairness, not performance.** spec.md decision 1's
amendment (2026-08-01) handed the four projected costs to the network on the
explicit ground that the baseline's seven state-action features already carry
them, so sharing them *levels* the two Policies' inputs rather than tilting
them. The action set was the last un-levelled axis; while it differed, a
Gate B result could not be attributed to the approximator at all.

## Decision

Both Policies call one definition —
`src/stdvrp/policies/action_set.py`'s `select_vehicle_possible_actions`
(ticket 13's verbatim extraction of the linear baseline's own function, its
two collaborators beside it: `_closest_allowed_clients`,
`_classify_shortest_distance_clients`). `TransformerMonteCarloPolicy._sweep`
calls it at **`m + 2`, in both `decide()` and `decide_train()`**
(`src/stdvrp/policies/transformer_policy.py`). `episode.py` trains the linear
baseline at `m+2` but evaluates it at `m+40` over the swept action counts;
this Policy is fixed at `m+2` throughout and does not inherit that mismatch —
whatever estimator eventually consumes these candidates sees one action
distribution, never one it was fitted on and a wider one it is evaluated
against.

It comes back **whole**: the three branches (parked-past-350, `<3`-Clients
endgame, the normal `k`-nearest sweep), the delayed-Client classifier, the
`350`/`310` literals that disagree by 40 minutes, the `list(set(...))` dedup
and its insertion-order dependence, the `< 3 clients` branch — all preserved
exactly, per ADR-0001's rule (fix what crashes, never re-tune what is tuned).
Importing it by halves would have been worse than not importing it.

### What retires

- **`_is_retired` retires.** Branch 1 of `select_vehicle_possible_actions`
  offers a vehicle parked at the depot past `tau > 350` only the depot, so it
  can no longer claim a Client it cannot serve — the same protection, now one
  arm of the shared candidate computation. One behavior difference is
  deliberate: `_is_retired` gated on `horizon_start_minute` (a config clock);
  the shared branch gates on the literal `350` instead. A vehicle parked at
  the depot between the two (e.g. `tau = horizon_start_minute + 2`) is now
  eligible for a real Client rather than forced home — adopting the
  *identical* set means adopting that literal too, which is why this is a
  reversal of ADR-0007 rather than an amendment to it.
- **`_depot_is_feasible` retires.** Its condition 2 (the return leg already
  breaches the shift) *is* branch 3's own depot-append condition, the same
  formula; its condition 1 (no Client feasible) is subsumed by every branch's
  fallback to `[depot]` when nothing else survives.
- **The `claimed_mask` defect closes, as a side effect rather than as its own
  change.** Ticket 08 left it open: the transformer's `claimed_mask` used to
  be rebuilt fresh every `_sweep` call, so it only ever knew about vehicles
  already decided *this* pass — never a not-yet-processed vehicle's in-flight
  target from the previous decision epoch, the way `MonteCarloPolicy` seeds
  `forbidden_actions` from `self.action` for every other vehicle
  (`Model.run_training_episode`'s own comment already documents `policy.action`
  being "mutated on every later decision" generically, for any Policy).
  `TransformerMonteCarloPolicy` now carries its own `self.action`, threaded
  through as `current_action` — adopting the shared module adopts this
  behavior for free.

### What stays Policy-side

Two things `action_set.py` does not know about, because they are not part of
the linear baseline's own rule:

1. **No double booking within one decision pass** (the B11 invariant) — now
   arising from `select_vehicle_possible_actions` excluding `current_action`'s
   other entries, rather than from a locally rebuilt mask.
2. **No self-node** (`simulator-correctness` ticket 11, B20, ADR-0008): a
   pending Client the vehicle is already standing on is not a candidate.
   ADR-0008 leaves `monte_carlo.py` untouched because "its own candidate rules
   already exclude a vehicle's current node by construction" — an argument
   about how the linear baseline happens to call the shared function, not a
   property of the function itself, so `TransformerMonteCarloPolicy._sweep`
   still filters it explicitly (both the greedy and the ε-exploration
   branch), falling back to the depot if nothing survives. The depot is never
   filtered by either rule.

### `is_depot` / `DEPOT_WARM_START_PENALTY`: kept, decided by measurement

With the depot now entering the candidate list only where
`select_vehicle_possible_actions` itself admits it, the warm-start penalty
that kept it from winning at decision epoch 1 (ADR-0007, "the depot's Q
value") might be doing nothing any more. Measured, not argued
(`.scratch/neural-policy/results/action_set_m2_50.py`): the untrained `cost`
warm start at `m+2` over 50 `evaluation_seeds` reads **3365.09** at
`DEPOT_WARM_START_PENALTY = 1.0` (as shipped) against **3364.52** at `0.0` —
**-0.02%, 1/50 seeds differ, Wilcoxon p = 0.317**. A null result, confirming
the structural prediction. The penalty stays: it costs nothing where it no
longer matters (most decisions now, since the depot rarely enters the
shortlist) and is still correct where it does (the shift-breach window, where
the depot genuinely competes against real Clients in the argmin).

## The measurement this decision rests on: the 2×2

Zero training. All on the 50 `evaluation_seeds`, never `test_seeds` (a `W`
*selected* on `evaluation_seeds` reads 36% better there than on `test_seeds`
while an arithmetic rule reads 3% better — F12's winner's curse, measured
policy-dependent — so the two seed sets are not interchangeable here).

| ranking ↓ / candidate set → | baseline's set (`m+2`) | all ~151 |
|---|---|---|
| arbitrary (`W = 0` of the linear) | 30 791.4 — *degenerate, not a null* | — |
| **cost-greedy (`c(s, a)`)** | **3365.09** ← the frozen null | 3693.2 |
| trained `W` of the linear | 2168.4 (`m+40`) / 2483.2 (`m+2`) | — |

The new cell (cost-greedy @ `m+2`) is the one that separates candidate set
from ranking rule: it holds the ranking rule fixed at cost-greedy and moves
only the candidate set, against the 3693.2 already measured at ~151. Cost-greedy
at `m+2` scores **8.9% better** than cost-greedy at ~151 — restricting the
candidate set to the `k` nearest helps the myopic dispatcher too, consistent
with "a regularizer against long myopic hauls" above.

**The branch declared before the number existed, so it could not be
rationalised afterwards:** if cost-greedy @ `m+2` had come in below `2168.4` —
the linear baseline's best cell on this same seed set — the myopic base would
have beaten a tuned 19-weight VFA at zero training. It did not (3365.09 >
2168.4); say so plainly rather than re-framing the question. This is a claim
about `evaluation_seeds` until `neural-policy` ticket 09 measures the verdict
set, and given the ×1.56 / ×1.03 spread `baseline_null_50.py` and
`warm_start_50.py` already measured between the two seed sets, it may not
survive the trip unchanged.

## Considered and rejected

- **Keep the transformer's own feasibility-only action set, restricted to
  `k` nearest by some new rule.** Rejected: a second, independently-tuned
  shortlist reintroduces exactly the ambiguity Gate B exists to remove — any
  result would be attributable to "the approximator" or "a different
  candidate heuristic that happened to help (or hurt) here too", never
  cleanly to one or the other.
- **Compose: hold a `MonteCarloPolicy` instance purely to ask it for
  candidates.** Rejected in ticket 13: drags `W`, the exploration RNG and a
  rebound `self.state` along for one method call, and needs `self.action`
  kept in sync every epoch — a hidden coupling in place of a visible one.
- **Retire `DEPOT_WARM_START_PENALTY` now that it measures as a null.**
  Rejected: a null result is a reason not to touch working code, not a reason
  to remove it — see above.

## Consequences

- `src/stdvrp/policies/action_set.py` (ticket 13) is the single definition
  both Policies call; `MonteCarloPolicy._select_vehicle_possible_actions`
  stays as a thin delegate, `TransformerMonteCarloPolicy._sweep` calls it
  directly.
- `TransformerMonteCarloPolicy` gains a `feature_extractor`
  (`FeatureExtractor.state_features` only — the twelve general-state features
  are computed as a byproduct, but only `delayed_clients` is read, and neither
  it nor any of the twelve reaches `QHead`; spec.md decision 1's "the state
  features stay out" is untouched) and a persistent `self.action`, mutated in
  place by `_sweep` exactly as `MonteCarloPolicy.action` already is.
- `tests/unit/test_transformer_policy.py`'s `TestHomeIsAnExitNotADestination`
  no longer calls the retired `_is_retired`/`_depot_is_feasible` directly;
  their branch arithmetic is pinned once, directly, in
  `tests/unit/test_action_set.py` (ticket 13). A new
  `TestCurrentActionIsSelfAction` pins that `current_action` passed to the
  shared function is `self.action` itself, not a copy — the mechanism the
  `claimed_mask` defect closure rests on.
- Predicted and measured self-golden diff: **zero** — this ticket does not
  touch `monte_carlo.py` or `action_set.py`, only `transformer_policy.py` and
  its tests (`tests/test_self_golden.py`, 6/6 passed).
