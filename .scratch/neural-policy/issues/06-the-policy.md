# 06 — The Policy: decide, decide_train, learn

**What to build:** `TransformerMonteCarloPolicy` — the `TrainablePolicy` that
scores every pending Client, decides one vehicle at a time, and learns from the
Monte Carlo return. Writes **ADR-0007**.

**Blocked by:** 02, 05

**Status:** resolved

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

- [x] Every existing invariant that applies to any Policy still passes, in
      particular B11 (no double booking) and B5 (a legal action for every
      vehicle at every tau under every valid config).
- [x] One encoder pass per decision epoch, asserted — not `m`.
- [x] Predicted self-golden diff: **zero.** `monte_carlo.py` is not touched.

## Comments

Implemented as `TransformerMonteCarloPolicy`
(`src/stdvrp/policies/transformer_policy.py`) + ADR-0007
(`docs/adr/0007-the-action-set-is-feasibility-not-heuristic.md`).

**The depot needed its own Q value, which the ticket's pseudocode didn't
spell out.** `QHead` only scores `Embeddings.clients` — one row per pending
Client — but the action rule says the depot is always feasible too. Resolved
by synthesizing a depot "candidate" row the same way every real candidate's
arc half is built: `encoder.arc_embed([minutes_to_depot, length_to_depot] /
horizon_length)` (computed directly from `EpisodeGeometry`, permitted under
the observability rule) concatenated with the vehicle's own context embedding
(the depot has no client-like context of its own). Verified numerically, not
just by construction: at initialization `Q(v, depot) ==
minutes_to_depot / horizon_length` exactly (`TestDepotWarmStart`, Hypothesis
over 30 random worlds, `atol=1e-4`) — the myopic warm start (ticket 05) now
covers "should I go home" on the same footing as "which Client is nearest",
not as an unscored fallback. Full reasoning and rejected alternatives in
ADR-0007.

**`_already_acquired_cost` is duplicated from `MonteCarloPolicy`, not
shared** — ten lines of arithmetic over two hardcoded legacy cost factors,
reimplemented here rather than imported so `monte_carlo.py` stays completely
untouched (checked structurally: `TestDoesNotDependOnMonteCarlo` AST-parses
this module's own imports, not its prose, since the module docstring names
`monte_carlo.py` freely to explain why it isn't imported).

**Feasibility is enforced as a hard mask on the argmin, not trusted to the
network's `claimed` input** — at initialization `claimed` is provably
init-inert (ticket 05's `TestClaimedIsWired`), so relying on it alone to keep
B11 would fail on the very first decisions of a fresh Policy. `_sweep` tracks
claimed pending Clients directly and excludes them from the candidate set
before computing any argmin; `claimed` still flows into the network as an
ordinary feature so a trained network's predictions can account for
contention.

**Learning-time inefficiency, acknowledged, not fixed here:** `learn` does
not attempt to share one encoder pass across a decision epoch's `m`
vehicle-samples during replay (unlike the acting path's asserted "one encoder
pass per decision epoch, not `m`") — shuffled minibatches routinely split
those samples apart. Every training sample re-tokenizes and re-encodes its
snapshot from scratch: correct, not maximally efficient, recorded as a future
measurement rather than the default (mirrors spec.md decision 9's own
treatment of the replay-buffer question).

New `ExperimentConfig` fields (not anticipated by the ticket text, needed to
drive `learn`): `neural_learning_rate` (seeds Adam — a different scale from
the linear baseline's SGD `learning_rate`, and what ticket 07's patience-based
convergence stopping will multiply by 0.3), `neural_learn_passes` (`K`,
ticket 03's compute-budget measurement used 4), `neural_batch_size`.

Verified: 11 new tests (`tests/unit/test_transformer_policy.py`) — B11 and B5
as Hypothesis properties over 0..8 pending Clients and 1..5 vehicles (60
examples each), one-encoder-pass-per-decision-epoch via a call-counting
monkeypatch on `TokenEncoder.transformer.forward`, the depot warm start,
`decide_train`'s epsilon=1.0 exploration still respecting B11 over 20 trials,
`learn` moving weights / staying finite / being a no-op on an empty episode /
being bit-reproducible given the same `learn_rng` seed. Full suite (excluding
`golden`) after landing: 4115 passed, 3 deselected — the 14-test delta from
`tests/unit/test_trainer.py`'s and `test_experiment_config.py`'s own new
`neural_*` cases, plus this ticket's 11. `mypy`/`ruff check`/`ruff format
--check` clean on every file this ticket touched (four pre-existing E501
violations elsewhere, documented in `simulator-correctness` ticket 10's own
Comments, untouched by this ticket). Self-golden not re-run: this ticket
predicted and structurally guarantees a zero diff (`monte_carlo.py` untouched,
confirmed by `TestDoesNotDependOnMonteCarlo`), so there is nothing for it to
measure.
