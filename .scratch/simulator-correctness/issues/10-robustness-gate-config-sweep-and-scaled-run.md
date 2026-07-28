# 10 — Robustness gate: config sweep and scaled real run

**What to build:** The thing that separates "these 19 findings are gone" from
"the simulator holds up". Closes the effort.

**Blocked by:** 02, 03, 04, 05, 06, 07, 08, 09

**Status:** open

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

- [ ] **Config sweep test.** Over a cross product of `shift_end_minute` ×
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
- [ ] **Scaled real-data run** (`experiments/chengdu/baseline_scaled.yaml`,
      which already exists). The 19 findings were measured on 45 nodes; the real
      instance has 1900. This only verifies that nothing new appears at real
      scale — it is not looking for results. Report wall clock alongside, to
      confirm the `simulation-performance` effort's ~10× was not regressed.
- [ ] **Final self-golden re-capture**, and confirm the whole suite is green.
- [ ] **Spec closing status** section: what landed, the cumulative measured cost
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
