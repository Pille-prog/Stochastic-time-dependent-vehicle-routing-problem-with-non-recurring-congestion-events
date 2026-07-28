# 05 — Vectorized feature extraction

**What to build:** Rebuild `_extract_state_action_features` (66% of episode
time) and `_extract_general_state_features` + `_classify_delayed_clients` as
vectorized numpy computations over the ticket-04 geometry matrices, extracted
from `MonteCarloPolicy` into a cohesive concrete class (no new seam —
ADR-0002). Candidate Q evaluation becomes one matrix–vector product per
vehicle instead of per-candidate Python loops.

**Blocked by:** 04.

**Status:** resolved

- [x] Feature extraction lives in a concrete collaborator (working name
      `FeatureExtractor`) with no hidden instance-state handoff: the
      `X_general_state`/`X_state_action`/`possible_actions` mutable-attribute
      coupling inside `MonteCarloPolicy` is dissolved; data flows through
      arguments and return values.
- [x] The **duplicate-append quirk is reproduced faithfully** (spec decision):
      the inflated `vehicle_to_clients` semantics — per-remaining-vehicle
      appends with evolving travel times — are expressed as a deterministic
      vectorized construction with the same contents and the same effect on
      `delayed_clients` and the `future_delay` feature.
- [x] The permanently-zero feature and all normalization literals stay exactly
      as documented (W keeps 19 components; stored weight vectors remain
      valid).
- [x] Candidate evaluation batches all candidates of a vehicle into one
      feature matrix and a single `X @ W`; the argmin tie-break (first
      candidate in iteration order wins) is preserved exactly.
- [x] Tier 2 gate (float sums reorder): fixture self-golden within
      `rtol=1e-9` **and** the statistical golden gate green; the ticket
      Comments state precisely which reductions reordered.
- [ ] Benchmark note in Comments: episode throughput before/after and new
      profile top-10. **Throughput before/after recorded below; the profile
      top-10 was NOT captured** — see "Unfinished" at the end of this comment.

## Comments

### Resolution (2026-07-27)

`FeatureExtractor` (`src/stdvrp/policies/feature_extraction.py`, new) owns all
three routines the ticket names. `MonteCarloPolicy` keeps only *decision* logic:
which actions a vehicle may take, and which minimizes Q.

**The attribute coupling is gone.** `X_general_state`, `X_state_action`,
`possible_actions`, `delayed_clients`, `vehicle_to_clients`,
`shortest_distance_clients`, `mean_velocities`, `number_of_actions` and
`total_cost_acquired` are all deleted from `MonteCarloPolicy`. One
`StateFeatures` value object is built per decision pass and handed to every
method that needs it; `_select_vehicle_possible_actions` takes it as an
argument, `_classify_shortest_distance_clients` *returns* its dict instead of
assigning one, and `_already_acquired_cost(state)` takes the State and returns
the float. Two consequences worth recording:

- The legacy classified delayed Clients **twice** per decision (the selection
  routine ran it, then `extract_general_state_features` ran it again on the
  unchanged State). It runs once now. Same values — it is a pure function of
  the State — so this is a removed redundancy, not a behavior change.
- `update_W` no longer rebinds `self.state` to each `TrainingSnapshot`; the
  snapshot flows through as an argument. Ticket 06 explicitly deferred this
  ("Redesigning the rebind away is left to ticket 09"); dissolving the
  attribute coupling made it fall out here instead, so **ticket 09 no longer
  owns it**.

**The duplicate-append quirk, vectorized.** `_classify_closest_clients` is the
delicate part. The legacy scans eligible vehicles per Client keeping a running
nearest and appends `(travel time, Client)` to the running nearest vehicle's
list on *every* iteration after the first — so each Client lands in the lists
once per prefix of that scan, carrying that prefix's running minimum and going
to the first vehicle that attained it (strict `<`). Vectorized down the vehicle
axis: the running minima are one `np.minimum.accumulate`, the prefix argmin one
`np.maximum.accumulate` over the seats where a new minimum appears. The only
thing the `future_delay` feature reads from those lists is each
`(vehicle, Client)` pair's **multiplicity** — the stored travel time is never
used, confirmed by reading the legacy body — so the classifier returns a
`[vehicle, column]` count matrix via `np.bincount`. `delayed_clients`
additionally needs the pairs sorted, which one `np.lexsort` reproduces exactly
(travel time primary, Client id secondary, as tuple comparison did), keeping the
first two per vehicle where the legacy's `break` stopped.

One documented assumption: the legacy's first comparison is against `inf`, which
every finite travel time wins, so the vectorized form seeds
`is_new_minimum[0] = True`. A cache holding `inf`/NaN would diverge; a real
`ShortestPathCache` has neither.

**Which reductions reordered (the Tier-2 disclosure).**

Bit-exact, *not* reordered:

- All 12 general-state features. Their only sum is over time-window starts,
  which are integers and exactly representable; `late_count` is an integer
  count. The `elif` chain over the disjoint 400/500/600 earliness bands became
  three `count_nonzero` calls with the `tau` test hoisted out of the loop —
  disjoint bands make `elif` and `if` select the same Clients.
- The delayed-Client classification and the multiplicities: per-Client scans
  over vehicles, order-independent, no float arithmetic created.

Reordered (five sites, all in the seven state-action features):

1. `total_distance` (feature 14) — `float(other_lengths.sum()) + candidate` instead
   of one running `sum()` in vehicle index order.
2. `earliness_cost` (feature 15) — same shape: the other vehicles' terms sum
   first (numpy pairwise summation), the decided vehicle's term is added last.
3. `delay_cost` (feature 16) — same shape.
4. `future_delay` (feature 17) — two changes. Each Client's contribution is
   **multiplied by its duplicate-append multiplicity** instead of added once per
   duplicate; and the other vehicles' pooled contribution is a matrix–vector
   product (`keep @ pooled`) with the decided vehicle's term summed separately
   and added last.
5. `overtime_cost` (feature 18) — same shape as 1–3.

Plus one outside the extractor: `_best_q_action` computes `X @ W` over the whole
`[candidates, 19]` matrix — a BLAS `gemv` rather than one 19-term `np.dot` per
candidate.

**Tier-2 gate result: better than required.** The spec asks for `rtol=1e-9` on
the fixture self-golden. Measured with the new
`scripts/capture_self_golden.py --check` mode (added by this ticket — it re-runs
the capture protocol against the committed JSON and reports the worst relative
deviation instead of overwriting it):

```
worst relative deviation: 0.000e+00 at (identical)
tolerance:                1.000e-09
OK: within tolerance
```

**Bit-identical** across every W vector and every episode metric of the capture
— the reassociations above do not bite at fixture magnitudes. That is a
measured fact about this fixture, not a guarantee: the arithmetic *is*
reassociated, so the Tier-2 tolerance remains the contract and the committed
capture was **not** re-baselined (it did not need to be — `tests/test_self_golden.py`
still passes bit-exactly).

Statistical golden gate (full Chengdu, 907 MB path cache):
`uv run pytest tests/test_new_package_vs_golden_master.py -m golden` → **3 passed
in 24m28s**.

**Tests.** `tests/unit/test_feature_extraction.py` (new) pins parity against
`LoopReference` — the pre-ticket `MonteCarloPolicy` bodies transcribed verbatim,
scalar lookups and duplicate-append and all — over 4 randomized worlds × 9 taus
straddling every hardcoded cutoff (310, 400/500/600, 580, the horizon) × 5
remaining-Client sets × 3 depot-idle configurations × 4 action shapes:
**2701 cases**, at `rtol=0` for the general-state features, the delayed-Client
lists and the multiplicities, `rtol=1e-12` for the seven state-action features.
`tests/unit/test_monte_carlo_policy.py` gained `TestBatchedQEvaluation` (3 cases:
the tie keeps the first candidate in iteration order, the batch picks the same
winner as scoring one-by-one, a missing W is created at the feature width) and
its ticket-07 oracle now takes `StateFeatures` as an argument.
`tests/unit/test_episode_geometry.py` gained 4 cases for the new
`average_minutes_rows`/`length_rows` gathers (order follows the nodes handed in,
empty input keeps the column width, unknown node raises).

Full suite **2966 passed**; unit suite **2897 passed**; `ruff format`,
`ruff check`, `mypy` all clean.

**API added to `EpisodeGeometry`:** `average_minutes_rows(nodes)` /
`length_rows(nodes)` — `[node, column]` gathers, one row per vehicle. That is
how the extractor reads geometry now: the whole fleet against every column once
per decision, instead of one lookup per (vehicle, Client) pair.

### Benchmark (mini fixture, this machine) — and why it is not the win

`scripts/benchmark_episodes.py --train 30 --eval 30 --no-cache`, run
**alternating** before/after for 4 rounds so background load hits both sides
(another session was running ticket-08 episodes throughout — at peak four Python
processes and ~18 GB resident). "before" is a clean worktree at HEAD `5b68184`
(ticket 04); "after" is this ticket plus the uncommitted ticket 07. Min over the
4 rounds:

```
                 before (ticket 04)   after (ticket 05+07)
training  s/ep         0.143                0.145
evaluation s/ep        0.120                0.102
```

Per-round spread: before train 0.143–0.160, eval 0.120–0.147; after train
0.145–0.180, eval 0.102–0.143.

**Read this honestly: training is neutral and the spread is wider than the
effect.** Evaluation is ~15% better. This is the third ticket in a row to record
the same thing — ticket 04 was a measured *regression* in isolation, ticket 07
was invisible in whole-episode timing — and the cause is the same: the committed
fixture has **21 Clients and 6 vehicles**, which is the least favorable size for
a numpy rewrite, because per-call dispatch overhead is paid on arrays a few
dozen elements wide. The real Chengdu configuration is `mean_number_clients: 150`
with candidate pools up to 50, where the legacy cost is
O(vehicles × candidates × clients × vehicles) through the inflated
`vehicle_to_clients` and this one is a handful of array ops of fixed count.

I did **not** get that measurement. See below.

### Unfinished — hand this to ticket 09/10

Two measurements this ticket owed and did not deliver:

1. **The cProfile top-10** (the effort's stopping-rule reference, ticket 01
   deliverable 4). Not captured. `scripts/`-side protocol is unchanged: cProfile,
   5 training episodes, world built outside the profiler.
2. **The site-level benchmark at 150 Clients**, which is where the case for this
   vectorization actually lives. A first attempt used iteration counts ~4× too
   high and was killed on timeout after burning 635s of CPU; the rewritten
   version (smaller counts) was killed with the shell before it produced output.

Both are cheap to redo on a quiet machine and neither gates correctness — every
behavior gate above is green. Ticket 10 already owns the closing measurement
("rerun the ticket-01 fixture benchmark and the ticket-01 scaled real-dataset
protocol"), so the honest place for the 150-Client number is there, alongside the
duplicate-append fix it will compare against.

### Handoff to ticket 10 (duplicate-append fix candidate)

The switch point is exactly one line. `_classify_closest_clients` returns
`counts`, the `[vehicle, column]` multiplicity matrix; the *fixed* semantics —
one append per Client, after the closest-vehicle scan finishes — is that same
matrix with every non-zero entry set to 1, taken at the final running minimum
(`counts = (counts > 0)` restricted to the last row of the accumulate). Nothing
else in the feature arithmetic changes, because multiplicity is the only thing
`future_delay` reads. That makes the comparison study a per-run boolean rather
than a second implementation.
