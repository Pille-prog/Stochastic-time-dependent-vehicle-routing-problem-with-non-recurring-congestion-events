# 06 — The Policy: decide, decide_train, learn

**What to build:** `TransformerMonteCarloPolicy` — the `TrainablePolicy` that
scores every pending Client, decides one vehicle at a time, and learns from the
Monte Carlo return. Writes **ADR-0007**.

**Blocked by:** 02, 05

**Status:** open

## `decide` — greedy, all feasible actions

```
emb = encoder(tokens)                    # once per decision epoch
for v in range(m):                       # index order, as the baseline does
    q = head(emb.vehicle[v], emb.clients, claimed)
    action[v] = argmin(q over feasible)
    claimed[action[v]] = True
```

- **Feasible** = every pending Client not already claimed by another vehicle in
  this decision, plus the depot (always legal). That is the whole action rule.
- The mask is the **B11 invariant** (`tests/test_invariants.py`: "two vehicles
  never receive the same non-depot Client in one decision") expressed as a
  constraint rather than reconstructed by a candidate-selection heuristic. It
  must still pass.
- `_select_vehicle_possible_actions`, `_classify_shortest_distance_clients`,
  `delayed_clients`, the `350`/`310` literals and `number_actions_test` are
  **not used by this Policy**. They stay untouched in `monte_carlo.py`, which
  is the opponent.

## `decide_train` — ε-greedy over the same set

Same sweep, with an ε gate drawing from the injected `exploration_rng`. Note
that ε over ~150 Clients is a far more diffuse exploration than ε over
`vehicles + 2` candidates — an ε schedule is a hyperparameter to tune on the
evaluation seeds, and `docs/research/rl-methodology-for-stdvrp.md` **F4** notes
the baseline never decays it at all while Chen/Ulmer/Thomas decay 1.0 → 0.01.

## `learn` — one batch per episode

```
X = tokens of the episode's ~400 decisions
y = U_t - acquired_cost                  # the SAME target update_W uses
for _ in range(K):
    for mb in shuffled(X, y, batch_size):
        adam.step(huber(net(mb.X)[mb.action], mb.y))
discard
```

- The target is **unchanged from `update_W`**: backward accumulation of the
  realised return, minus `_already_acquired_cost`. Same statistical object; only
  the approximator and the optimizer differ. That is what makes the comparison
  against the baseline interpretable.
- Strictly **on-policy**: samples are used once and discarded. A replay buffer
  would be more sample-efficient (Chen/Ulmer/Thomas use one) but the targets
  would come from policies that no longer exist, which is no longer Monte Carlo
  policy evaluation — and the baseline has no buffer, so the comparison would
  mix approximator with data regime. Record it as a knob to measure later, not
  as the default.
- **Huber, not squared error.** The episode-cost distribution has a brutal right
  tail (research note **F10**: a truncated episode injects a target 2-3 orders of
  magnitude larger into *every* `t` of that episode's backward pass). Squared
  error would let one such episode dominate the batch.
- Target scaling: standardize `y` with fixed, config-derived scales, same
  discipline as ticket 04's token normalization and for the same reason — no
  running statistics, or an Episode's gradients depend on which Episodes came
  before it.

## ADR-0007 — The action set is feasibility, not heuristic

Records why this Policy has no candidate set: the shortlist is hand-engineered
ranking (nearest-k by *static* travel time, a bespoke delayed-Client classifier,
two disagreeing depot-idle literals that `simulator-correctness` decision 3
explicitly refused to re-tune), and leaving it in place would mean the network
chooses inside a list a heuristic built for it. Records what survives and why:
the no-double-booking mask is a **constraint**, not a preference. Records that
`number_actions_test` — an experimental axis for the baseline — has no meaning
here, and therefore that the comparison uses the baseline's **best** action
count, giving the opponent its best shot.

## Acceptance

- [ ] Every existing invariant that applies to any Policy still passes, in
      particular B11 (no double booking) and B5 (a legal action for every
      vehicle at every tau under every valid config).
- [ ] One encoder pass per decision epoch, asserted — not `m`.
- [ ] Predicted self-golden diff: **zero.** `monte_carlo.py` is not touched.

## Comments
