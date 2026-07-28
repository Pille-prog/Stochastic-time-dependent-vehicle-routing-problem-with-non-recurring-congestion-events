# 02 — Name the two clocks

**What to build:** Split the one config field that carries two concepts into two
named, validated, configurable fields — **without moving a single value**.
Closes B12, B15 and B16.

`horizon_end_minute` does three jobs (overtime threshold, latest window opening,
the Policy's homecoming guard) and does *not* do the one its name promises:
bound the episode. That is `EMERGENCY_HORIZON = 1150`, hardcoded in
`model.py:76` and repeated as a literal in the feature normalizer
`time_left = (1150 - tau) / 850` (`feature_extraction.py:197`).

These are two legitimately different things — a shift that ends (and after which
overtime accrues) and a hard stop — and the code already calls the first one
`shift_end_minute` internally (`model.py:141`). Only the config name lies.

**Blocked by:** 01

**Status:** open

- [ ] `horizon_end_minute` → **`shift_end_minute`** in `ExperimentConfig` and
      every committed YAML. Same value (780), same three uses.
- [ ] New **`episode_end_minute`**, default 1150. `EMERGENCY_HORIZON` and the
      feature normalizer's literal read it instead of hardcoding.
- [ ] Validation `shift_end_minute <= episode_end_minute`. **This is what kills
      B15**: negative overtime becomes unconfigurable rather than merely
      unreached. Add the `tau > shift_end` guard in
      `terminate_state_passing_horizon` anyway, for symmetry with
      `_vehicle_parks_at_depot:306` — it is free.
- [ ] **B16**: `_congestion_epoch_due` drops the `/60` division that produced
      the non-representable quotient. **Verify before asserting**: `tau_episode`
      is a float, so integer arithmetic only applies if decision epochs land on
      integer minutes. Establish that first; if they do not, the fix is
      "eliminate the division", not "use integers".
- [ ] Invariant: no cost component is negative **for any valid config** — a
      parametrized run over several `shift_end_minute` / `episode_end_minute`
      pairs. The existing suite asserts `component_total >= 0` only at 780,
      which is exactly why B15 survived.
- [ ] Invariant: congestion cadence fires
      `⌊(episode_end − horizon_start) / max_congestion_duration⌋` times, for
      every integer duration in a swept range — the review's 50 and 70 included,
      which today collapse to one roll.

## Predicted self-golden diff

**Exactly zero.** Renaming a field and exposing an identical default cannot move
a float. Every committed config keeps 780 and gets 1150, which is what
`EMERGENCY_HORIZON` already is.

**The one risky part is B16.** `(tau + 180 - 2)` is written as two operations
because it rounds twice and the Tier-1 gate is bit-exact (the existing docstring
says so). Any reassociation can move the firing set by a last-bit rounding. The
prediction stays **zero** — under `max_congestion_duration: 120` the review
measured the gate firing at the intended cadence — but if it is non-zero, the
mechanism is a float-rounding edge at a specific tau and that is outcome 2:
explain it to the tau, record it, do **not** edit this prediction.

## Evidence required

Zero diff on all three capture blocks. Bench output identical to ticket 01's
baseline. The B16 integrality finding stated explicitly.

## Comments
