# 10 — Simulation invariant suite (property-based)

**What to build:** The Hypothesis suite that makes simulations trustworthy: physical and accounting invariants asserted across randomly generated configs and seeds on the fixture, regardless of policy quality.

**Blocked by:** 07.

**Status:** resolved

- [x] Simulated clock never decreases; the Episode terminates according to the Horizon rules
- [x] Every Client ends the Episode served or penalized exactly once
- [x] Travel times are strictly positive; congestion factors stay within the configured bounds
- [x] Episode total cost equals distance + delay + earliness + overtime exactly
- [x] Suite runs in CI within acceptable time (bounded examples/deadlines)

## Comments

**2026-07-22 — resolved.** Delivered `tests/test_invariants.py`: two Hypothesis properties. `test_episode_invariants` (25 examples, `derandomize=True`, `deadline=None`) runs full evaluation Episodes on a module-scoped fixture world and asserts all four invariant families at once; `test_congestion_generator_stays_in_bounds` (80 examples) hammers `ArcProbabilityCongestionGenerator` on synthetic graphs. Module runs in ~24 s locally (≈20 s is the one-time TravelTimeModel build). Validated beyond the pinned examples with an 80-config randomized sweep (incl. single-vehicle fleets and horizon terminations): zero violations.

Findings the next tickets need:

- **Bug for ticket 12 triage — terminating transition cost is double-added.** When the Episode ends via `terminate_state_passing_horizon` inside the decision-epoch branch (`tau_multiplicator >= 1150`, so the clock sits at 1148 for horizon start 300), execution falls through to the `(tau + 178) % 6 == 0` epoch-end branch, which adds the same `transition_cost` to `total_cost` a second time — the gate always fires at 1148 (1326 = 6·221). Verified bit-exactly: `total_cost == components + final transition_cost` (diff 0.0). Unserved Clients' termination penalties therefore count twice in `total_cost` but once in the component totals. Preserved per ADR-0001; the invariant test documents and asserts this shape.
- **The cost identity holds only to float accumulation order**, not bit-exactly: `total_cost` accumulates per-transition sums while component totals accumulate per-increment, so ULP drift appears in most episodes (only 20/80 random configs were bit-equal). Asserted with `math.isclose(rel_tol=1e-9)`.
- **Congestion factor bound is `upper_bound / 0.78`, not `upper_bound`.** Spread arcs divide the drawn penalization by the depth damping, and reachable depths are only 0–2 because `generate` passes `max_depth - 1` to `_reachable_nodes` — the 0.73 (depth 3) branch is dead code under the default `max_depth=3`.
- **Single-day fixture worlds cannot exercise stochastic velocities** (stds are NaN, a preserved legacy quirk). The suite builds a temp world of 8 deterministically perturbed copies of fixture day 601 (`np.random.default_rng(day)`, ±20% speed noise) so every arc gets a positive std and real `random.gauss` draws happen. Reusable pattern for any test needing live randomness on the fixture.
- **Instrumentation pattern:** behavior-identical recording subclasses (`TauRecordingState.__setattr__` logs every clock write; `RecordingModel` logs velocity samples and termination calls) swapped into `stdvrp.simulation.episode` via a module fixture, plus a `RecordingCongestionGenerator` wrapper around the live generator — episodes still run through the real `run_evaluation_episode`, no production seams added (ADR-0002 respected).
