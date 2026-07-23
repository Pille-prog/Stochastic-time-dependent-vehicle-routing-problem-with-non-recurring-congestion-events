# 08 — Training Episode and W update

**What to build:** The training side of the tracer bullet: exploration-mode policy path and W creation/update, producing a W sequence identical to the legacy for fixed seeds.

**Blocked by:** 07.

**Status:** resolved

- [x] Training-mode policy path and the W update rule ported
- [x] W after N fixture training episodes matches the golden master exactly — per the ADR-0001 addendum the golden trajectory lives on the full local dataset (`test_w_trajectory_matches_exactly`, marker `golden`, all 4 episodes bit-exact); the CI-runnable equivalent chains 3 training episodes on the 44-copy fixture world against the in-process legacy, also bit-exact
- [x] The legacy warm-up quirk (first iteration runs with the hardcoded tiny learning rate before the configured one) is preserved and explicitly documented as legacy behavior pending ticket 12 triage

## Comments

**2026-07-22 — resolved.** Delivered: `MonteCarloPolicy.decide_train` (ports `monte_carlo_policy_train` → `select_epsilon_greedy_action_train`), `update_W` (ports `actualize_W`) and `_calculate_already_acquired_cost`; constructor gained keyword-only `number_actions_train` / `learning_rate` plus the two legacy-faithful **unseeded** `local_rng` / `local_rng_2` attributes. `Model.run_training_episode` ports `create_monte_carlo_episode_train`. `run_training_episode` + `TrainingEpisodeResult` in `simulation/episode.py` mirror the capture protocol's per-train-seed block.

Findings for ticket 09 (Trainer):

- **Training RNG order per decision epoch** (bit-exact): repair pass draws from `local_rng_2` (one `choice` per infeasible carried-over action), then the ε-gate draws one `local_rng.random()` per vehicle, and an exploratory action draws from the **global** `random.choice` — interleaved with the transition function's velocity `gauss` draws. `update_W` consumes no randomness. Reproducibility requires the capture convention: seed `local_rng`/`local_rng_2` right after policy construction (offsets 10 000 000 / 20 000 000 + train seed, stored in the golden protocol); `run_training_episode(exploration_seed=, repair_seed=)` does this — `None` keeps legacy nondeterminism.
- **Warm-up quirk lives at the caller** exactly as in the legacy: `training_model` sets `lr = 0.000001` before its loop and only assigns the configured rate after constructing the first episode's policy, so episode 1 always updates W with 1e-6. Documented in `episode.py`'s docstring and exercised by both tests; the Trainer (ticket 09) must own this sequence. Pending ticket 12 triage.
- **Quirk preserved**: the training loop zeroes `total_distance_cost` after every step, so a training Episode ends with distance cost 0 (asserted in the CI test). Pending ticket 12 triage.
- **Not ported (dead in the live path)**: policy diagnostics `rewards`/`Q_preds`/`error` (assigned in `actualize_W`, never read; no RNG/W effect), constructor counters `cont_1`/`cont_2`/`x`/`o`, and model `total_cost_2` (consistent with the ticket 07 eval port).
- **For the Trainer**: legacy eval block runs seeds 100000..100049 every `test_frequency` episodes with `Newest_W`; `Best_W` updates when the eval mean beats `Q_pred` (init 1e11, so the first block always wins); the `mean_static_policy` plot lookup table and hardcoded loop constants move to `ExperimentConfig` per the spec.
- **Verification**: `tests/test_training_episode_vs_legacy.py` — 3 chained episodes (W from `None`, warm-up first) bit-identical to the in-process legacy incl. costs, tau, state counts, global stream positions, and proof the repair/exploration RNG paths fired (~56 s, in CI); full suite 160 passed; `uv run pytest -m golden` **5 passed in 26 min 13 s** — the stored 4-episode W trajectory reproduced bit-for-bit on the full local dataset.
