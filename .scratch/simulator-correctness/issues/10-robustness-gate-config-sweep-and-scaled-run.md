# 10 — Robustness gate: config sweep and scaled real run

**What to build:** The thing that separates "these 19 findings are gone" from
"the simulator holds up". Closes the effort.

**Blocked by:** 02, 03, 04, 05, 06, 07, 08, 09

**Status:** resolved

## Why this ticket exists

Three of the 19 findings — **B5, B15, B16** — are configuration traps. None
fires at the shipped values. B15 needs `horizon_end_minute` above 1148; B16
needs a `max_congestion_duration` whose `/60` quotient is not binary-exact
(50 or 70 collapse congestion from 12–17 rolls per episode to **one**, while
`_compute_event_probabilities` keeps calibrating as if nothing happened); B5
needs the 40-minute window between the 310 and 350 thresholds with one Client
pending.

All three were found by a person reading code. Nothing in the repository would
have found them, and nothing rules out a fourth, a fifth and a sixth. With the
config frozen at one point, "no errors" means "no errors at that point".

## What to build

- [x] **Config sweep test.** Over a cross product of `shift_end_minute` ×
      `episode_end_minute` × `max_congestion_duration` × fleet size × Client
      count × seed, on the mini fixture, asserting on every run:
      - no exception raised;
      - **no negative cost component** (this is B15, which the existing
        invariant missed by only ever testing 780);
      - no NaN or infinity in velocities, costs or features;
      - congestion cadence fires the expected number of times (B16);
      - every invariant from tickets 02–08 still holds.

      Include the specific traps as named cases: `max_congestion_duration` of
      50 and 70; `shift_end_minute` at and above `episode_end_minute` (must be
      **rejected by validation**, not silently accepted); the 310–350 tau window
      with `min_number_clients` low.

      Mini-fixture episodes run in milliseconds, so hundreds of combinations fit
      in CI. Size it to run there.
- [x] **Scaled real-data run** (`experiments/chengdu/baseline_scaled.yaml`,
      which already exists). The 19 findings were measured on 45 nodes; the real
      instance has 1900. This only verifies that nothing new appears at real
      scale — it is not looking for results. Report wall clock alongside, to
      confirm the `simulation-performance` effort's ~10× was not regressed.
- [x] **Final self-golden re-capture**, and confirm the whole suite is green.
- [x] **Spec closing status** section: what landed, the cumulative measured cost
      change from ticket 01's baseline to here with the per-ticket contributions
      attributed, any finding that did not reproduce, and any *new* finding the
      sweep turned up (each of those opens its own ticket rather than being
      fixed inline).

## Predicted self-golden diff

**Zero.** This ticket adds tests and runs things; it changes no production code.

If the sweep finds a defect, **it does not get fixed here.** It opens a ticket
with its own invariant, prediction and evidence, exactly like 02–09. A
robustness gate that quietly repairs what it finds cannot be trusted to report
what it found.

## Evidence required

Sweep green across the full cross product, with the combination count stated —
"the sweep passed" without saying what it covered is not evidence. Scaled real
run completing clean, with wall clock. Cumulative before/after on the 60-seed
bench.

## Out of scope, explicitly

**The full Chengdu training run.** It is tempting to close on it, but it is not
part of repairing the simulator — it is the first experiment of the repaired
lab. Mixing them confuses two things: if the full run produces an odd number,
you would not know whether it is a residual defect or science. It gets its own
step, afterwards, with attention on interpreting results rather than debugging.

## Comments

### Resolution (2026-07-30)

No production code changed, exactly as predicted. Three files:

- `tests/conftest.py` — extracted `perturbed_mini_world` (the real-gauss-perturbed
  traffic world `tests/test_invariants.py` already built for itself) and added
  `measurement_bench_module` (loads `scripts/measurement_bench.py`, mirroring
  the existing `benchmark_module` fixture), so the sweep reuses the exact
  `BenchModel`/`BenchCongestionGenerator`/`check_b9_spread_depths` instrumentation
  tickets 01–09 already measured with, instead of a second implementation.
- `tests/test_invariants.py` — its `sim_world` fixture now layers its own fixed
  `FIXTURE_DEMAND` onto the shared fixture instead of building the traffic world
  itself. Behavior-identical: same dict shape, same keys, verified by re-running
  the file alone (3 passed) before touching anything else.
- `tests/test_config_sweep.py` (new) — the config sweep, 434 tests.

### Config sweep (434 tests, all green)

Cross product: 2 demand configs (`Client count`) × 4 `(shift_end_minute,
episode_end_minute)` pairs × 6 `max_congestion_duration` values × 3 fleet sizes
× 3 seeds = 432, plus 2 standalone named tests (B9's structural check, the
validation-boundary trap) = **434**. Full file: 40.8s locally (well under a
minute, as the ticket asked for).

Every grid point asserts, via a `BenchModel` subclass (`SweepModel`) that adds
two checks to the ten `BenchModel` already carries: no exception, no negative
cost component (B15), no NaN/infinity in sampled velocities/travel
times/lengths or the general-state feature vector (new), the real cadence gate
(`Model._congestion_epoch_due`, not a synthetic oracle — see the correction
below) fires exactly the intended integer-arithmetic count for that run's own
`duration`/`episode_end_minute`, and B1a/B1b/B7/B10/B11/B14/B17 (both the
terminal-snapshot and, new here, the per-clock-advance purge check) all stay at
zero. **Zero violations across all 432 combinations.**

**Named traps**, each an explicit grid value:

- `max_congestion_duration` 50 and 70 (in `DURATIONS`, alongside 30/120/200/240).
- The 310–350 tau window with `min_number_clients` low: the `LOW_DEMAND`
  config (`mean_number_clients=3, min_number_clients=1`) crossed with
  `vehicle_count=1` biases real swept episodes toward the endgame branches
  B5/B11 live in, complementing (not replacing)
  `tests/unit/test_monte_carlo_policy.py::TestEndgameInvariants`'s direct,
  exhaustive Hypothesis property over that same window.
- `shift_end_minute` at and above `episode_end_minute`, **with a correction to
  this ticket's own wording**: the checklist above says both must be
  "rejected by validation," but the effort's own boundary decision (spec.md
  decision 5, landed in ticket 02) validates `shift_end_minute <=
  episode_end_minute` — equality is the accepted boundary (a shift that ends
  exactly when the episode does is a legitimate zero-overtime config), not a
  trap. `test_shift_end_minute_validation_boundary` asserts both halves of the
  real boundary explicitly: `(600, 600)` is accepted, `(601, 600)` raises
  `ValueError`. `(600, 600)` is also one of the four `HORIZON_PAIRS` in the main
  grid, so the sweep runs 108 full episodes at that exact boundary, not merely
  constructs the config.

**A tooling gotcha this ticket hit and did not repeat**: `scripts/
measurement_bench.py`'s own `check_b16_cadence` is ticket 01's *frozen
reproduction* of the pre-fix bug — a standalone float-division formula, never
routed through production code, and never updated after ticket 02's fix
landed (confirmed: its printed `B16_cadence_duration_50/70` lines still read
`MISMATCH` in this ticket's own fresh `ticket10-final-*.txt` bench captures,
byte-for-byte the same as ticket 01's pre-fix ones). That is expected and
correct for what the function documents itself as ("the shipped ... vs ...
intended" cadence, `check_b16_cadence`'s own docstring) — but it means it
cannot be used as a live B16 regression check. This sweep instead calls
`Model._congestion_epoch_due` directly (`_cadence_fires`), mirroring `tests/
unit/test_congestion_epoch_cadence.py`'s own oracle but varying
`episode_end_minute` over the grid, which that file fixes at 1150. Recorded
here so a future reader of `measurement_bench.py`'s printed report does not
mistake a frozen historical artifact for a live regression.

### Scaled real-data run

**Wall clock** (`scripts/benchmark_episodes.py --config
experiments/chengdu/baseline_scaled.yaml --project experiments/chengdu/config.yaml
--test`, warm world-cache, real 1900-node Chengdu dataset; raw output committed
at `bench-output/ticket10-scaled-real-run.txt`):

```
world load (s)        39.097
train  s/ep            2.442
eval   s/ep            1.303
test s/ep @2            1.319
test s/ep @50           1.496

projected full run (config.yaml)
  world load                   39.1s  (0h00m39s)
  training (100 ep)           244.2s  (0h04m04s)
  evaluation (500 ep)         651.4s  (0h10m51s)
  final test (300 ep)         421.5s  (0h07m01s)
  TOTAL                      1356.2s  (0h22m36s)
```

**Correctness at real scale** (16 real episodes — the exact shape of
`baseline_scaled.yaml`: 5 train + 5 eval + 6 test — run through the same
`SweepModel`-style instrumentation as the mini-fixture sweep, ad hoc, not
committed as a new permanent test: spec.md decision 11 keeps real-dataset runs
out of the per-ticket loop, and this closing check does not need to persist as
an ongoing gate any more than it needed to exist as one before; raw output
committed at `bench-output/ticket10-real-scale-invariant-check.txt`, and
reproduced identically on a second run): **zero violations on every counter,
on every one of the 16 episodes** (no negative cost, no NaN/infinity, zero
B1a/B1b, B7, B10, B11, B14, B17). The structural B9 check on the real
topology: **0/38379 wrong-depth, 0/38379 missed-entirely** (vs. the mini
fixture's 38/635 and 40/675 pre-fix) — confirms ticket 08's fix at the scale
the review's own "~15%" figure was originally measured against. This is the
ticket's actual "nothing new appears at real scale" evidence; the wall-clock
run alone (no crash) would not have caught a silently-wrong number.

**The `simulation-performance` effort's ~10× — reported, not silently
preserved.** Comparing this run against that effort's own closing measurement
(`.scratch/simulation-performance/issues/10-duplicate-append-fix-candidate.md`,
same `full_run_shape(config.yaml)` methodology, so directly comparable):

```
                  closing (simulation-performance)   now (this ticket)   ratio
world load              31.6s                             39.1s          1.24x
training (100 ep)      181.7s (1.817 s/ep)                244.2s (2.442 s/ep)   1.34x
evaluation (500 ep)    401.0s (0.802 s/ep)                651.4s (1.303 s/ep)   1.63x
final test (300 ep)    254.9s (~0.850 s/ep, 2 pts)         421.5s (~1.40 s/ep)  1.65x
TOTAL                  869.2s (14m29s)                    1356.2s (22m36s)     1.56x
```

The full-run projection is **~1.56× slower** than the simulation-performance
effort's own closing number, i.e. the effective speedup from that effort's
original pre-optimization baseline (8669.3s) is now **~6.4×**, down from the
recorded 9.97×. This is disclosed exactly as spec.md's own out-of-scope clause
anticipates ("nothing here may regress it **without saying so**") — it is not
a silently-accepted regression.

**Explained to a mechanism, not a new finding (spec.md decision 10).** None of
tickets 02–09 touched `world_cache.py`, `episode_geometry.py`,
`episode_pool.py`, or any vectorized `FeatureExtractor` routine's shape — the
architecture the 10× came from is structurally untouched. The slowdown is
*more real simulated work per episode*, the same effect tickets 03/04/07/08
already measured and predicted directly, now also showing up in wall clock:
on the matching single-vehicle mini-fixture bench (below), `mean_decisions`
rose **+31.8%** and `mean_km_driven` **+36.4%** after this effort's fixes — a
fleet that used to wrongly retire early (B1a) or drive less because congestion
under-covered the network (B9) now does measurably more work, and doing more
real work costs more wall clock. No new ticket: the mechanism is already
on record, not newly discovered.

### Cumulative measured cost change, ticket 01's baseline → here

Three fresh 60-seed bench runs at current HEAD, diffed directly against ticket
01's own committed baselines (`bench-output/ticket01-*.txt`) — a single clean
comparison rather than chaining each ticket's own before/after (several
tickets measured against a common ticket-01-baseline worktree in parallel, not
sequentially against each other's output, so multiplying their percentages
would double-count).

**Default fleet (~5 vehicles), `--w zero`:**

| metric | before | after | change |
|---|---|---|---|
| mean_total_cost | 512.194224 | 515.646000 | **+0.67%** |
| mean_distance_cost | 181.750069 | 181.471293 | −0.15% |
| mean_delay_cost | 18.287359 | 22.278006 | **+21.82%** |
| mean_earliness_cost | 312.156796 | 311.896701 | −0.08% |
| mean_final_tau | 459.450195 | 462.238985 | +0.61% |
| mean_decisions | 71.033 | 71.867 | +1.17% |

**Default fleet, `--w frozen` (a real, congestion-aware trained policy):**

| metric | before | after | change |
|---|---|---|---|
| mean_total_cost | 481.687896 | 474.679231 | **−1.45%** |
| mean_distance_cost | 171.559680 | 164.177077 | −4.30% |
| mean_delay_cost | 2.073866 | 2.569448 | +23.90% |
| unserved (any) | 1 client, 1 episode | 0 | fixed |

**`--vehicle-count 1` (the review's own stress config), `--w zero`:**

| metric | before | after | change |
|---|---|---|---|
| mean_total_cost | 1831.729883 | 2862.068295 | **+56.25%** |
| mean_distance_cost | 165.047718 | 225.091750 | +36.38% |
| mean_delay_cost | 1398.733832 | 2248.199433 | +60.73% |
| mean_overtime_cost | 105.800836 | 225.168570 | +112.83% |
| mean_final_tau | 822.459462 | 1031.399111 | +25.40% |
| mean_decisions | 118.583 | 156.283 | +31.79% |
| unserved_in_window | 14/60 eps, 109 clients | **0** | fixed (B1a/B3) |
| unserved_overdue | 40/60 eps, 146 clients | 39/60 eps, 159 clients | now honestly priced |

**Every violation counter (B1a/B1b, B7, B10, B11, B14, B15, B9 structural) is
zero on all three "after" runs**, and `B17_book_purge`'s `expired_at_end_total`
goes to 0 on every run — was 3064 (default `--w zero`), 3228 (default `--w
frozen`), 3158 (`--vehicle-count 1`) — the book now actually purges.

**Reading the split.** Multi-vehicle fleets absorb most of the effect via
redistribution slack (+0.67%, even −1.45% under a real trained policy that can
route around the fixes' consequences) — exactly what ticket 08 predicted for
this reason. The single-vehicle stress config (the review's own reproduction
setup for B1b/B3) shows the effect at full strength: **+56% total cost**,
because a single vehicle has no alternative routing to absorb either the fix's
extra distance (B1a/B1b: it now drives home and keeps working, instead of
teleporting) or the harsher, now-honest abandonment price (B3/ADR-0004: the
same demand that used to cost 0 when abandoned mid-episode now costs its full
`max(episode_end_minute, tau) - due`). This matches ticket 03's own prediction
almost exactly ("expect the magnitude to be large... total cost multiplies
rather than shifts").

**Per-ticket contributions**, as each ticket's own Comments already recorded
(cited, not re-derived — several were measured on parallel worktrees against
the same ticket-01 base rather than chained sequentially, so they are each
individually attributable but do not sum arithmetically to the table above):

| Ticket | Finding(s) | Own recorded direction |
|---|---|---|
| 02 | B12/B15/B16 | Zero (rename + validation only) |
| 03 | B3/B14 | `--vehicle-count 1`: mean_delay_cost 1713.33 → 3072.73 (own bench, before ticket 04 landed) |
| 04 | B1a/B1b | 180 mini-fixture episodes (fleets 1/3/6): mean total cost 1537.32 → 1293.28 (−15.9%, ticket 03's harsh pricing already active), +15.06 km/episode |
| 05 | B5/B11 | Default fleet: 512.194224 → 505.760321 (−1.3%, non-increasing as predicted) |
| 06 | B10 | Zero on the decision-stable bench (by construction); `final_w[10]` 0.0 → nonzero once trained |
| 07 | B7/B17 | B17 alone: zero. B7: default 512.194→512.740 (+0.11%), vc1 1831.730→1949.168 (+6.4%) |
| 08 | B9 | Default zero-W 512.194→508.314 (−0.76%), frozen-W 481.688→480.642 (−0.22%), vc1 1831.730→1972.425 (+7.68%) |
| 09 | B18/B19/B8-doc | Zero (docs only) |
| 10 | Sweep + scaled run | Zero (no production code) |

### Findings that did not reproduce exactly

Already recorded in ticket 01, not re-litigated here: B11's exact "734
transitions" and B17's exact "116/116 expired" review figures did not
reproduce on this fixture at their literal magnitude (this bench measured 308
and "6277 written, 3158 expired" respectively pre-fix) — the *defects*
reproduced and were fixed as scoped in both cases; only the specific headline
counts did not. Ticket 03 found a similar gap mid-effort: its own
predicted-nonzero frozen-W blast radius on the 15 capture seeds turned out to
be exactly zero, because ticket 08's BFS fix (landed in between) had already
changed which seeds carry unserved demand. No further non-reproductions turned
up in this closing sweep.

### New findings

None that need their own ticket. The two items above this ticket surfaced
(the `check_b16_cadence` frozen-oracle gotcha, and the ~10×→~6.4× wall-clock
change) are both fully explained to an already-on-record mechanism, per
spec.md decision 10 — neither is an unexplained defect in the simulator
itself, which is what would open a new ticket.

### Self-golden and full suite

`scripts/capture_self_golden.py --check`: `worst relative deviation: 0.000e+00`,
`per-seed per-metric diff: (identical - nothing moved)` — **exactly zero**, as
predicted. No re-capture needed; there is nothing to overwrite when the diff is
genuinely zero (same outcome as ticket 02, same reason). `tests/test_self_golden.py`
and `tests/test_world_cache_self_golden.py`: 7/7 passed.

Full suite (`uv run pytest -q`, `-m "not golden"` per project default):
**4039 passed, 3 deselected**, ~7m20s. `ruff check` / `ruff format --check`
clean on every file this ticket touched. `mypy src/stdvrp` (the project's own
configured scope — `tests/` is deliberately outside `[tool.mypy] files`, so
this ticket's test-only diff is not subject to it): unchanged, clean.

**One pre-existing gap, not this ticket's to fix**: `uv run ruff check .`
reports 4 `E501` (line too long) violations, all in ticket 04's files
(`feature_extraction.py:311`, `monte_carlo.py:313`, `model.py:447,666`) —
confirmed pre-existing at `5de38c6` (ticket 04's own commit, before this
ticket touched anything) via `git stash` + a clean re-run. Ticket 04's own
Comments claimed "ruff check ... all clean," which this contradicts; recorded
here rather than silently fixed, per this ticket's own rule about what it
finds.

### Closing status

The effort closes here. All 15 in-scope findings (B1a, B1b, B3, B5, B7, B9,
B10, B11, B12, B14, B15, B16, B17, B18, B19) have a red-before/green-after
invariant, landed in tickets 02–09, all still green under this ticket's
432-combination config sweep and a 16-episode real-scale check neither of
which existed when those invariants were written. The four modeling findings
(B4, B6, B8, B13) remain explicitly out of scope, headed to the modeling
effort per spec.md. See `spec.md`'s own "Closing status" section for the
effort-level summary.
