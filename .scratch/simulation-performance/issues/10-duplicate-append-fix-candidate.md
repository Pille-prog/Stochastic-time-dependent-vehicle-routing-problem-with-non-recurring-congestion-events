# 10 — Duplicate-append fix candidate: measure, then the user decides

**What to build:** The Tier-3 candidate. Implement the fixed
`_classify_delayed_clients` semantics (one append per client, after the
closest-vehicle scan finishes) behind a config or constructor switch, run the
comparison study, and present the evidence. **Adoption is the user's call —
this ticket ends with a decision request, not a merge of the fix as default.**
Also the effort's closing measurement.

**Blocked by:** 05, 08, 09.

**Status:** resolved

- [x] The fixed classification implemented alongside the faithful quirk
      (ticket 05's vectorized construction), switchable per run; default
      remains the quirk.
- [x] Comparison study on the real dataset: full training + final test under
      quirk vs fix — mean costs with spread, W trajectories, and episode
      throughput (the fix removes ~V× work from `future_delay`). Results
      recorded in Comments.
- [x] Decision request to the user with the evidence; if adopted: the fix
      becomes default, ADR-0001-style change-log entry, statistical
      re-baseline, and the quirk path retires. If rejected: the switch is
      removed and the quirk stays documented as deliberate.
- [x] Effort closing measurement: rerun the ticket-01 fixture benchmark and
      the ticket-01 *scaled* real-dataset protocol (same ~16-episode config,
      apples-to-apples per-phase comparison); Record final speedups vs baseline and the closing
      profile.
      Stopping rule check: no single site >10% of episode time, or the residual
      sites are listed as future-work notes in the spec.

## Comments

### Resolution (2026-07-28)

**The fix, implemented.** `FeatureExtractor._classify_closest_clients` gained a
`duplicate_append_fix: bool = False` constructor switch (threaded through
`MonteCarloPolicy` → `run_evaluation_episode`/`run_training_episode` →
`EpisodeWorld.episode_kwargs()` → `ExperimentConfig.duplicate_append_fix`, a new
required YAML field, `false` in every committed config). Ticket 05's handoff was
exactly right: the fixed semantics are the *same* `owner`/`running_minimum`
arrays the quirk already computes, restricted to their last row (the scan's
final state — each Client's true closest eligible vehicle, at its true minimum
travel time) instead of every row (every vehicle that briefly held that title
during the scan). Both branches feed the same counting/lexsort/grouping code
that follows, so the fix cost one `if` and two array slices, not a second
implementation — `counts` comes out 0/1 automatically since each Client now
contributes to exactly one vehicle's bucket.

**Tests.** `tests/unit/test_feature_extraction.py` gained a `FixedLoopReference`
(the same duplicate-append loop with the
`vehicle_to_clients[assigned_vehicle].append(...)` line dedented out of the
vehicle scan — one append per Client) and two parity test classes mirroring the
existing quirk ones: `TestFixedClassificationParity` (delayed-Client lists,
0/1-only counts, bit-exact against `FixedLoopReference`) and
`TestFixedStateActionFeatureParity` (the fixed multiplicities flow into
`future_delay` correctly). `tests/unit/test_monte_carlo_policy.py` pinned that
the constructor switch reaches the extractor. All green
(`uv run pytest tests/unit -q` → 5648 passed with the switch in place), Tier-1
self-golden stayed bit-exact throughout (the default is `False`, and the quirk
branch's arithmetic is untouched — same operations, same order, just addressed
through renamed variables).

**Comparison study.** Two runs, both through the full `Trainer` pipeline
(`scripts/compare_duplicate_append.py`, written for this ticket and removed
with the switch — see below), reusing one loaded world across both variants
(the world-cache key does not depend on `duplicate_append_fix`).

*Mini fixture* (6 vehicles, ~20 Clients, 30 train + 30 eval + 10 test): evaluation-block
and final-test mean costs came out **bit-identical** between quirk and fix to 4
decimal places; only the W trajectory differs, by a small amount
(`‖W_quirk − W_fix‖ = 0.057` against `‖W‖ ≈ 12.6`, ~0.4%). Expected — the quirk's
effect scales with vehicle count (ticket 05's Comments), and the fixture's 6
vehicles rarely produce enough running-minimum churn during the closest-vehicle
scan to change which candidates get selected.

*Full real Chengdu dataset* (`experiments/chengdu/config.yaml`, unscaled: 100
training + 500 evaluation (10 blocks × 50 seeds) + 300 final-test episodes
(6 action counts × 50 seeds, post ticket-02 dedup), both variants from one
warm-cached world):

```
                          quirk (default)      fix
evaluation-block mean cost   2040–2115             2325–2407     (+13% to +18%, every one of 10 blocks)
final test @2   actions      2967.95 (σ 2057)      3425.59 (σ 2256)   +15.4%
final test @10  actions      2803.98 (σ 2101)      3512.05 (σ 2722)   +25.3%
final test @20  actions      2957.32 (σ 2448)      3594.36 (σ 2718)   +21.5%
final test @30  actions      2827.11 (σ 2186)      3545.05 (σ 2831)   +25.4%
final test @40  actions      2882.99 (σ 2342)      3524.28 (σ 2751)   +22.2%
final test @50  actions      2931.45 (σ 2472)      3525.28 (σ 2710)   +20.3%
‖W_quirk − W_fix‖            181.9   (‖W_quirk‖ = 890.7, ‖W_fix‖ = 976.1, ~20%)
throughput (900 ep total)    825.86s (1.09 ep/s)    795.90s (1.13 ep/s)   fix ~3.6% faster
```

Consistent and large-N (50 seeds per cell, 10 independent evaluation blocks, all
one direction): the fix makes the learned policy **worse**, not better — every
evaluation block and every final-test action count regressed by double digits,
for a throughput gain far short of the "~V× work removed from `future_delay`"
the ticket's premise expected (the classifier was never the dominant profile
site at this scale; `candidate_features` is, per the closing profile below).
Reading of *why*: the duplicate-append quirk's inflated multiplicities are an
implicit importance weighting on `future_delay` toward Clients that stayed
"contested" (near several vehicles' running-nearest) during the scan; removing
that weighting changes what the linear Q-approximation's gradient steps
optimize for, and empirically the legacy's accidental weighting learns a better
policy under this reward shaping than the "corrected" uniform one does.

**Decision (user, 2026-07-28): reject.** Evidence presented via `AskUserQuestion`;
the user chose "keep the quirk." Per the ticket's rejection branch: the switch is
removed — `duplicate_append_fix` reverted out of `ExperimentConfig`,
`FeatureExtractor`, `MonteCarloPolicy`, `run_evaluation_episode`/
`run_training_episode`, `EpisodeWorld.episode_kwargs()`, every YAML config, and
the three direct `ExperimentConfig(...)` construction sites; the fixed-mode
tests and `scripts/compare_duplicate_append.py` removed with it (`git checkout --
<files>` back to HEAD, confirmed by unit-suite count returning to the pre-ticket
2946 and Tier-1 self-golden staying bit-exact). The quirk stays the sole
implementation; `feature_extraction.py`'s module docstring now records the
measured-and-rejected finding (numbers above) at the quirk's definition instead
of pointing at an open ticket, so a future reader hits the evidence before
reopening this.

**Effort closing measurement.**

*Ticket-01 fixture benchmark* (`--train 30 --eval 30 --no-cache`, this machine):
world load 20.7s (baseline 20.2s, unchanged — the fixture always rebuilds into a
fresh temp dir so the binary cache never hits here), training 0.123 s/ep (8.14
ep/s, baseline 0.092 s/ep / 10.9 ep/s), evaluation 0.096 s/ep (10.46 ep/s,
baseline 0.060 s/ep / 16.6 ep/s). **Still a regression on the fixture** — the
fourth ticket in a row to record this (04, 05/07, 09): 6 vehicles / ~20 Clients
is the least favorable size for numpy dispatch overhead, and the effort's real
target was always the 150-Client configuration.

*Ticket-01 scaled real-dataset protocol* (`baseline_scaled.yaml`, warm-cached
world): world load 31.6s, train 1.817 s/ep, eval 0.802 s/ep, test @2 0.750 s/ep,
test @50 0.955 s/ep.

*Apples-to-apples projected full run* (ticket-01's baseline per-episode times
vs. this measurement, **both** run through `full_run_shape(config.yaml)` —
ticket 02's dedup'd final-test shape, so the comparison isolates per-episode
speed from the test-dedup win already banked):

```
                  baseline (ticket 01)   closing (ticket 10)   speedup
world load             1106.0s (18m26s)       31.6s (0m32s)     35.03x
training  (100 ep)      711.3s (11m51s)      181.7s (3m02s)      3.91x
evaluation (500 ep)    2633.0s (43m53s)      401.0s (6m41s)      6.57x
final test (300 ep)    4219.0s (1h10m19s)    254.9s (4m15s)     16.55x
TOTAL                  8669.3s (2h24m29s)    869.2s (14m29s)     9.97x
```

Add ticket 02's already-banked dedup win (15,000 → 300 test episodes, 40.6×
measured on the fixture) on top and the true baseline-to-now full-experiment
speedup is far larger than 9.97×; this number isolates what tickets 03–09 (world
cache, geometry matrices, feature vectorization, parallel pool, decomposition)
bought on top of ticket 02, which is the fair per-phase comparison the ticket
asked for. **9.97× lands one hundredth short of the spec's ≥10× target** — close
enough that machine noise alone could tip it either way (see the interleaved-A/B
guidance in the effort's own benchmarking notes); not worth chasing further
given the closing profile below shows why.

*Closing profile* (cProfile, 5 training episodes, mini fixture, world built
outside the profiler): 0.914s total (baseline 1.191s, ticket 06 1.124s — steady
decline). Top by tottime: `FeatureExtractor.candidate_features` 0.194s (**21.2%**
— exceeds the 10% stopping-rule threshold), `_arrival_costs`'s nested `costs`
0.075s (8.2%), `_overtime_costs`'s nested `homecoming` 0.049s (5.4%),
`_classify_closest_clients` 0.040s (4.4%), everything else under 4%. **Stopping
rule not strictly met** — recorded as a future-work note in `spec.md`'s new
"Closing status" section per the rule's alternative clause:
`candidate_features` is already the ticket-05 vectorized per-candidate matrix
builder; shrinking its tottime further would mean restructuring that hot loop
again, which is out of scope for this ticket.

**Effort status: closed.** Tickets 01–10 all resolved. `duplicate_append_fix`
does not exist in the shipped code — the classifier is exactly ticket 05's
vectorized quirk, now with the ticket-10 finding documented at its definition.
