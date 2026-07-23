# 02 — Deduplicate the final test's identical episodes

**What to build:** Stop re-running bit-identical episodes in
`Trainer.final_test`. With per-seed Generators (ticket 13 of the refactor),
every one of the `test_episodes: 50` iterations for a given (seed, vehicles,
action count) is deterministic and identical — the trainer docstring already
admits "the mean equals a single episode's value". 15,000 episodes → 300.

**Blocked by:** 01.

**Status:** resolved

- [x] `final_test` runs each (action count, seed) episode **once**; the
      reported per-seed metrics are that episode's values (the mean of k
      identical values is the value itself — division-order float noise from
      the legacy `sum/k` must not leak into the report, so compare against the
      self-golden and document if the old report differed in last-bit ulps).
- [x] `test_episodes` stays in `ExperimentConfig` for config compatibility but
      is documented as inert (or validated == deduplicated behavior); decide
      and record which in the ticket Comments.
- [x] Tier 1 gate: `results.json` for a fixture run is bit-identical to the
      self-golden capture (or the documented ulp-level report difference is
      pinned by a test).
- [x] Benchmark note in Comments: measured final-test speedup.

## Comments

### Resolution (2026-07-23)

`Trainer.final_test` (`src/stdvrp/training/trainer.py`) no longer loops
`config.test_episodes` times per (action count, seed): it calls
`run_evaluation_episode` once and reports that episode's nine metrics
directly, instead of summing `test_episodes` bit-identical repeats and
dividing. Docstrings updated (module docstring, `SeedTestResult`,
`final_test`) to record the decision.

**`test_episodes` decision: inert, not validated.** The field stays in
`ExperimentConfig` (still required by every YAML, still validated positive)
but `final_test` never reads it — documented inline at the field
(`src/stdvrp/config.py`) and in `trainer.py`. Chose "inert" over "validated ==
1" to avoid forcing every committed/production config file
(`experiments/chengdu/config.yaml: 50`, `tests/fixtures/chengdu_mini/config.yaml:
10`, etc.) to change just to keep loading — lower blast radius, and the field
now reads as a harmless historical relic rather than something that must be
edited to stay valid.

**Division-order float noise: real, bounded to 1 ULP, pinned by a test.**
Reimplemented the retired sum/k loop locally in
`tests/test_final_test_dedup.py` (production code no longer has it) and ran
it against the real mini-fixture world alongside the new
single-episode `final_test`, with `test_episodes=7` (deliberately not a power
of two). Confirmed empirically: `delay_cost`, `overtime_cost`, `state_count`,
`delay_clients`, `earliness_clients` matched the legacy mean bit-for-bit;
`total_cost`, `distance_cost`, `tau`, and (for one seed) `earliness_cost`
differed from the legacy sum/7 mean by **exactly 1 ULP** — e.g. seed 100
`total_cost`: legacy `441.2652910076495` vs new `441.2652910076496`. The test
asserts the drift is always ≤ 1 ULP and that both an exact match and a 1-ULP
drift actually occur (so the assertion isn't vacuously true). This is the
"Tier 1 gate" deliverable — the existing self-golden capture
(`tests/fixtures/self_golden/mini_fixture.json`) never exercised `final_test`
at all (it captures `run_training_episode`/`run_evaluation_episode` output
directly), so there was no pre-existing bit-exact reference for the `test`
section of `results.json` to diff against; this ticket adds the first one, as
a reproducible test rather than a static fixture (the legacy computation no
longer exists in production code to regenerate against).

**Measured final-test speedup.** Mini fixture, real chengdu
`test_episodes: 50`, `test_action_counts: (2, 50)`, 2 test seeds, trained W:

```
new (deduplicated) final_test: 0.233s
legacy (test_episodes=50) final_test:  9.462s
speedup: 40.6x
```

(Short of the theoretical 50× because per-episode setup isn't the only cost
at this tiny fixture scale; the ratio converges toward 50× as episodes get
more expensive — see the real per-episode timings below.) Also updated
`scripts/benchmark_episodes.py`'s `full_run_shape()` to hardcode
`test_episodes_per_cell=1` (was `config.test_episodes`) so future
`--project` runs reflect this fix instead of the pre-ticket-02 15,000-episode
assumption. Ticket 01's projection (`## Comments` there) is now stale in the
"final test" line and is superseded by this ticket: full-run final test goes
from a projected 15,000 episodes (58h36m) to 300 episodes — reproduce with
`uv run python scripts/benchmark_episodes.py --config
experiments/chengdu/baseline_scaled.yaml --project experiments/chengdu/config.yaml`
once ticket 01's scaled baseline config is re-run (not re-run here, to avoid
the ~18-minute world-load cost and to stay clear of concurrent work on ticket
03's world cache).

**Note on concurrent work.** This branch had uncommitted, in-flight work from
another session on tickets 03/04/05/06 (`world_cache.py`,
`episode_geometry.py`, `monte_carlo.py` geometry refactor, training snapshots)
while this ticket was worked. Changes here were scoped and committed via
explicit pathspec to avoid interfering; one unrelated pre-existing test
failure (`test_main_dispatches_default_args_to_the_fixture_benchmark`,
ticket-03 `cache_dir` wiring) was observed and left untouched as out of
scope.
