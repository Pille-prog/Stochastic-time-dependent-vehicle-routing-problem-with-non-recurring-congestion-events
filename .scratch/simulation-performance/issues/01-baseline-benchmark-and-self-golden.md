# 01 — Baseline benchmark + self-golden capture

**What to build:** The measurement foundation every other ticket gates on: a
committed episode benchmark, a one-off full-run baseline measurement, and the
Tier-1 self-golden capture of the current package's exact outputs.

**Blocked by:** —

**Status:** resolved

- [x] `scripts/benchmark_episodes.py`: builds the mini-fixture world once, then
      times N training + N evaluation episodes (wall-clock per episode and
      episodes/sec), plus world-load time. Deterministic seeds; prints a
      compact table. CI runs it as smoke (no timing asserts).
- [x] A **scaled** real-dataset baseline — never the full experiment (hours):
      the real Chengdu config reduced to `total_train_iterations: 5`,
      one evaluation block of 5 seeds, `test_action_counts: [2, 50]`,
      3 test seeds (100/101/102, fleets 6/5/5), `test_episodes: 1`
      (~16 real episodes, minutes not hours). Timed per phase: world load,
      per-training-episode, per-evaluation-episode, per-test-episode at each
      action count. The full-run denominator is **projected** from those
      per-episode means (`load + 100·t_train + 500·t_eval +
      50·50·Σ t_test(action counts, interpolated between 2 and 50)`);
      measured numbers + projection recorded in this ticket's Comments.
- [x] Self-golden capture: per-seed episode metrics (all nine `EPISODE_METRICS`)
      and per-episode W vectors for a fixed seed set on the mini fixture,
      written under `tests/fixtures/self_golden/` with a capture script in
      `scripts/`. A pytest gate compares the live package against it bit-exactly
      (floats compared with `==`, no tolerance).
- [x] Profile snapshot (cProfile top-20 by tottime on the fixture) stored in
      Comments — the stopping-rule reference (`no site >10%`).

## Comments

### Resolution (2026-07-23)

All four deliverables landed. Files:

- `scripts/benchmark_episodes.py` — fixture benchmark (default) + scaled
  real-dataset baseline & full-run projection (`--config` / `--project`). The
  projection math (`project_full_run` / `interpolate_test_time`) is pure and
  unit-tested (`tests/test_benchmark_projection.py`).
- `tests/test_benchmark_smoke.py` — CI smoke: the fixture path runs, per-episode
  times are finite and positive, no wall-clock thresholds.
- `scripts/capture_self_golden.py` + `tests/fixtures/self_golden/mini_fixture.json`
  + `tests/test_self_golden.py` — the Tier-1 bit-exact gate.
- `experiments/chengdu/baseline_scaled.yaml` — the scaled real config.

### Mini-fixture episode benchmark (deliverable 1)

Committed world (21 clients / 6 vehicles, day 601 copied across the 44 legacy
traffic days so per-day speed std is 0.0, not NaN). Wall-clock, this machine
(Windows/AMD64, numpy 2.4.6, python 3.11.9):

```
world load (s)        20.2
training  (5 ep)       0.092 s/ep    10.9 ep/s
evaluation (5 ep)      0.060 s/ep    16.6 ep/s
```

### Scaled real-dataset baseline + full-run projection (deliverable 2)

`experiments/chengdu/baseline_scaled.yaml` — the real Chengdu archive (44 days,
907 MB path cache), episode counts cut to 5 train + 5 eval + (2 action counts ×
3 seeds × 1) = 16 real episodes. Measured per-phase per-episode means:

```
world load (s)     1105.965   (0h18m26s, no binary cache yet — that is ticket 03)
train  s/ep           7.113
eval   s/ep           5.266
test s/ep @2          4.118
test s/ep @50        24.577
```

Projected full run (`config.yaml`: 100 train, 10 eval blocks × 50 seeds = 500
eval, 6 action counts × 50 seeds × 50 test episodes = 15000 test; middle action
counts linearly interpolated between the @2 and @50 means):

```
world load                   1106.0s   ( 0h18m25s)
training (100 ep)             711.3s   ( 0h11m51s)
evaluation (500 ep)         2632.8s   ( 0h43m52s)
final test (15000 ep)     210948.2s   (58h35m48s)
TOTAL                     215398.2s   (59h49m58s)   ≈ 2.5 days
```

**The projection is the effort's denominator, and it already names the biggest
prize: the final test is 58h36m of the 59h50m total (98%).** Those 15000 test
episodes are only 300 distinct results (50 identical `test_episodes` per
seed/action cell, deterministic per seed) — exactly what ticket 02 removes; a
50× cut there projects to ~1h10m of test, ~1h20m total. World load (18m×
per-process) is ticket 03's binary cache. The per-episode decision path (train
7.1 s/ep, eval 5.3 s/ep) is tickets 04–07.

Reproduce: `uv run python scripts/benchmark_episodes.py --config
experiments/chengdu/baseline_scaled.yaml --project experiments/chengdu/config.yaml`.

### Self-golden capture (deliverable 3)

`tests/fixtures/self_golden/mini_fixture.json` pins, on the mini fixture: the
per-episode W vector + all nine `EPISODE_METRICS` for training seeds
1000–1004 (carried W, warm-up lr on the first), and the nine metrics for
evaluation seeds 100000–100009 run with the final trained W. `float()` values
JSON-round-trip exactly, so `tests/test_self_golden.py` re-runs the identical
protocol and asserts `==` with no tolerance.

**Environment guard (user-ratified 2026-07-23; ADR-0003).** numpy's `Generator`
guarantees a reproducible *integer* stream per seed but **not** bit-identical
float draws — the per-arc velocity `normal` draws are Ziggurat-based, and
numpy's own compatibility policy excludes cross-CPU/libm/version float identity
(confirmed via DeepWiki against numpy/numpy). The gate therefore records a
`{numpy, python, system, machine}` fingerprint and asserts `==` only when the
running environment matches the capture; otherwise it **skips** rather than
falsely fail. It is live on the capture machine (where optimization tickets are
worked) and inert elsewhere; an always-run test
(`test_self_golden_gate_is_active_on_this_environment`) **warns loudly** when
inert so a green CI never hides a dormant gate, and CI relies on the Tier-2
statistical gate meanwhile. To make Tier-1 live on the CI image, re-run
`scripts/capture_self_golden.py` there and commit the result.

### Profile snapshot (deliverable 4 — stopping-rule reference)

`cProfile`, top 20 by tottime, 5 training episodes on the fixture (world built
outside the profiler). Total 1.191s / 5 ep. Percentages are tottime ÷ total.

```
   ncalls  tottime  %tot  cumtime  function
     7224    0.454  38.1%   0.697  policies/monte_carlo.py:453 _extract_state_action_features
   781768    0.203  17.0%   0.203  network/shortest_path_cache.py:47 path_between
     1075    0.077   6.5%   0.105  policies/monte_carlo.py:309 _classify_delayed_clients
48926/710    0.062   5.2%   0.126  copy.py:128 deepcopy
     1613    0.054   4.5%   0.726  policies/monte_carlo.py:217 _select_best_q_action_for_vehicle
     3595    0.024   2.0%   0.080  policies/monte_carlo.py:241 _select_vehicle_possible_actions
    44414    0.021   1.8%   0.033  policies/monte_carlo.py:480 <genexpr>
   177881    0.019   1.6%   0.019  {method 'append' of 'list'}
      715    0.018   1.5%   0.091  policies/monte_carlo.py:387 _extract_general_state_features
8199/6414    0.018   1.5%   0.065  copy.py:201 _deepcopy_list
    15450    0.014   1.2%   0.057  {built-in sum}
     2549    0.014   1.2%   0.022  policies/monte_carlo.py:267 <listcomp>
      355    0.014   1.2%   0.874  policies/monte_carlo.py:151 _select_epsilon_greedy_actions_train
    98953    0.011   0.9%   0.011  {method 'get' of 'dict'}
      355    0.011   0.9%   0.071  simulation/model.py:176 transition_function
 1065/355    0.009   0.8%   0.112  copy.py:227 _deepcopy_dict
     1286    0.009   0.8%   0.009  simulation/model.py:402 vehicle_distance_transition_cost
     7224    0.009   0.8%   0.009  policies/monte_carlo.py:470 <listcomp>
     2787    0.009   0.8%   0.016  {heapq.nsmallest}
     7352    0.008   0.7%   0.008  policies/monte_carlo.py:474 <genexpr>
```

Stopping rule (`no site > 10% of episode time`) is **not** met at baseline —
two sites exceed it: `_extract_state_action_features` (38%, cumulative 58% via
`_select_best_q_action_for_vehicle`) and its `path_between` dict lookups (17%,
781,768 lookups / 5 ep). `deepcopy` per training step is 5% (ticket 06). This is
the reference the effort optimizes down toward; tickets 04/05/07 target the top
two sites, ticket 06 the deepcopy.
