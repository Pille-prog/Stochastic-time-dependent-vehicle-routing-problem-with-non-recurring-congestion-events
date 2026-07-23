# Spec: Simulation and Monte Carlo training performance

Status: approved (grilling session 2026-07-23)

## Goal

Make the Chengdu experiment measurably fast — episode throughput and full-run
wall-clock — by optimizing the data structures and algorithms of the simulation
(Model) and the Monte Carlo training path (Policy, Trainer, world loading). The
OOP/SOLID redesign is the *means* to do it cleanly, not the goal: success is
measured on a committed benchmark, and structural elegance must never cost
measured performance.

This effort takes up the "performance scalability work" the
`generic-stdvrp-refactor` spec deliberately left out of scope. That refactor
(tickets 01–14) is complete: the package is behavior-pinned by the phase-2
statistical golden master (`chengdu_full_phase2.json`, ±40% mean-cost
tolerance) and all RNG is injected per-Episode Generators.

## Measured baseline (profile, mini fixture: 21 clients, 6 vehicles)

- **~82% of episode time is the Policy's decision path**, not the Model:
  `_extract_state_action_features` alone is 66% (8,536 calls in 5 episodes),
  driven by **1.13M `path_between` dict lookups**. The transition function is ~5%.
- The preserved duplicate-append quirk in `_classify_delayed_clients` inflates
  `vehicle_to_clients` to O(clients × vehicles) entries, and the `future_delay`
  feature iterates that inflated list per candidate — a legacy behavior quirk
  that is also the hot path's biggest work multiplier.
- `copy.deepcopy(state)` per training step: ~7% on the fixture, grows with
  clients/vehicles.
- The final test runs `test_episodes: 50` bit-identical episodes per seed
  (per-seed Generators make them deterministic): 15,000 episodes where 300
  produce the identical report.
- World load: 21.6s on the mini fixture; the real dataset (88 speed files +
  907 MB `all_shortest_paths.csv`) costs minutes per run, per process.

## Decisions (from the grilling session)

| Decision | Choice |
|---|---|
| Primary objective | Measured speed (benchmark-driven); OOP/SOLID redesign is the means |
| Behavior contract | **Tiered**: default gate is bit-exact per-seed vs a self-golden capture of the current package; float-reordering optimizations (vectorization) drop to the existing statistical gate, documented per ticket; behavior fixes are separate ticket-12-style tickets with own test + re-baseline |
| Scope | Policy hot path + training snapshots, Trainer redundancy, world loading, Model transition loop (the last for structural elegance, not speed) |
| Abstraction seams | ADR-0002 stands: exactly three seams, no new interfaces. Elegance via cohesive *concrete* classes and value objects |
| Hot-path representation | **Arrays inside, OO outside** (ADR-0003): per-Episode numpy geometry matrices, struct-of-arrays state, vectorized features — wrapped in concrete domain-named facades |
| Duplicate-append quirk | Preserved and vectorized faithfully now; a separate candidate ticket implements the fix, measures its statistical impact on costs and W, and the user decides adoption with that evidence |
| Techniques / dependencies | Anything goes (numba, cython, polars, …) decided tactically per ticket, justified by benchmark in the ticket and documented there |
| Parallelism | Evaluation blocks and final test on a persistent `ProcessPoolExecutor` (16 logical cores available), seed-ordered aggregation to stay bit-identical to serial; training stays sequential (W is a serial dependency — parallel training would be new science, not optimization) |
| Success / stopping rule | Ticket 01 captures the baseline: a *scaled* real-dataset run (~16 episodes: 5 train, 5 eval, 2 action counts × 3 seeds × 1 test) whose per-phase per-episode times *project* the full-run denominator — never a full experiment run — plus the committed fixture episode benchmark in `scripts/`. Target: ≥10× serial episode throughput. Stop polishing when no single profile site exceeds 10% of episode time. CI runs the benchmark as smoke, no timing asserts |
| Process | This spec + tickets under `issues/`, worked via `/implement`; one new ADR (0003); `CONTEXT.md` only gains genuine domain terms, never implementation vocabulary |

## Behavior contract, precisely

1. **Tier 1 — self-golden bit-exact (default).** Ticket 01 captures the current
   package's outputs (per-seed episode metrics and W trajectories on the mini
   fixture, plus the statistical summary on the full dataset where available).
   Every pure-mechanical optimization must reproduce Tier 1 exactly.
2. **Tier 2 — statistical (documented fallback).** An optimization that
   inherently reorders float arithmetic (vectorized sums, BLAS batching)
   validates against the existing ±40% statistical golden gate *plus* the
   fixture self-golden at a tight numeric tolerance (e.g. `rtol=1e-9`) — the
   ticket must state why Tier 1 is unattainable.
3. **Tier 3 — behavior change (separate tickets only).** Ticket-12-style:
   explicit user triage, own test, ADR-0001-style change-log entry,
   re-baseline. In this effort only ticket 10 (duplicate-append fix candidate)
   sits here, and it ships *measurement first, adoption maybe*.

## Tickets

Published to `issues/01`–`issues/10` (2026-07-23). Critical path:
01 → 04 → 05 → 10; parallelism lands via 02/03 → 08.

| # | Ticket | Blocked by |
|---|---|---|
| 01 | Baseline benchmark + self-golden capture | — |
| 02 | Deduplicate the final test's identical episodes | 01 |
| 03 | Binary world cache (TravelTimeModel + ShortestPathCache) | 01 |
| 04 | Per-Episode geometry matrices replace path-cache dict lookups | 01 |
| 05 | Vectorized feature extraction | 04 |
| 06 | Training snapshots without deepcopy | 01 |
| 07 | Vectorized candidate-action selection | 04 |
| 08 | Parallel evaluation and final test | 02, 03 |
| 09 | Model decomposition (concrete collaborators) | 05, 06, 07 |
| 10 | Duplicate-append fix candidate: measure, then user decides | 05, 08, 09 |

## Out of scope (deliberately)

- Parallel or asynchronous *training* (A3C-style shared W) — new research.
- Fixing any legacy quirk other than the measured ticket-10 candidate
  (hardcoded 1150/1198 horizons, 350/310 cutoffs, the dead zero feature that
  pads W to 19, the epoch-gate arithmetic): all preserved; each would need its
  own Tier-3 triage in a future effort.
- New abstraction seams (ADR-0002) and the database `DataSource`.
- GPU acceleration.
