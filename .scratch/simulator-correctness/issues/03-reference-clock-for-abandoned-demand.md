# 03 — The reference clock for abandoned demand

**What to build:** Price unserved Clients at termination against a **fixed
reference clock** instead of whatever clock the fleet happened to stop at.
Closes B3 and B14. Writes **ADR-0004**. Retires the statistical baseline.

This is the yardstick, not a measurement. While episode cost depends on *when*
the fleet stopped — the same 19 abandoned Clients cost 0.00 or 11 245.00 in the
review's seed 14, four orders of magnitude apart — no other fix in this effort
can be measured, and any mean that mixes horizon-terminated with all-back
episodes is averaging two different cost functions.

`cost(t) = max(0, t − due)` is **correct while the episode runs**: a pending
Client whose window is still open should cost nothing, because a vehicle could
still make it. The defect is reusing that same live formula, unchanged, for a
different event — termination, where the outcome is no longer open and the
Client will never be served.

**Blocked by:** 02 (needs `episode_end_minute` to exist)

**Status:** open

- [ ] `_charge_unserved_delays` prices **every** unserved Client at
      `max(0, max(episode_end_minute, tau) − due)`. The live in-episode formula
      is untouched — it is correct.
- [ ] **B14**: charges increment their counters. `unserved_clients` is its own
      metric, **not folded into `late_clients`** — under the new clock every
      abandoned Client generates a charge, so lumping them together would
      conflate "served late" with "never served". `charge_fleet_overtime`'s
      `overtime_vehicles` undercount gets the same treatment.
- [ ] Invariant: **the termination charge is a function of `due` and config
      only, never of `tau`.** A property test that terminates the same
      unserved set at several clocks and asserts one price. This invariant *is*
      the definition of comparability across episodes.
- [ ] Invariant: money charged > 0 ⟺ its counter > 0, for every component.
- [ ] **ADR-0004** — the episode clock and the price of abandoned demand.
      Records the rejected alternatives and why:
      - `shift_end` (780) as the reference: under it, serving a Client at 900
        costs 120 while abandoning one due at 700 costs 80 — **abandoning stays
        cheaper than serving**, and `best_w` is selected by minimum mean cost.
      - A fixed no-service charge: conceptually cleaner (lost demand ≠ late
        arrival) but introduces a free parameter with nothing to calibrate it,
        and moving that number moves the optimum. Recorded as a deliberate
        extension for the modeling effort.
      - Why `episode_end` works: it is by construction the maximum achievable
        lateness, so abandoning is never cheaper than serving however late; and
        it leaves horizon-terminated episodes essentially unmoved
        (`1148 − due` → `1150 − due`), concentrating the entire change on the
        all-back path, which is where the defect lives.
- [ ] **Retire the statistical gate.** Delete the three ±40% tolerance tests in
      `tests/test_new_package_vs_golden_master.py`. Their purpose was continuity
      with the legacy and this ticket severs it deliberately; a baseline
      computed under a different cost function cannot gate a run under this one.
      `chengdu_full_phase2.json` **stays in the repo** as historical evidence,
      with no test reading it. Recorded in ADR-0004.
- [ ] **`CONTEXT.md`**: fix **Horizon**, which is currently false — it claims
      all decisions and events happen within (start, end) while episodes run to
      1148 with an end of 780. Add the two clocks and a term for demand that is
      abandoned rather than served late.

## Predicted self-golden diff

Written before running, per the effort's contract. Magnitudes come from ticket
01's per-capture-seed unserved counts; **the shape below is fixed now and is the
falsifiable part.**

**Frozen-W block — the surgical claim.** With W fixed, no cost change can move a
decision, so the entire trajectory is identical and only the terminal charge
moves:

- `delay_cost` and `total_cost` **rise** on every seed that ends with unserved
  Clients, by exactly `Σ (1150 − due)` over them;
- `tau`, `distance_cost`, `earliness_cost`, `overtime_cost`, `state_count`,
  `delay_clients`, `earliness_clients` are **bit-identical on every seed**.

That invariance is checkable regardless of magnitude, and it is the whole point
of the frozen-W block. If any of those seven moves, the change was not confined
to termination pricing.

**Training and final-eval blocks — full divergence, predicted and expected.**
The termination charge enters the Monte Carlo return, so W moves from the first
affected training episode. All five training seeds end all-back (tau 430–495),
so divergence from seed 1000 onward is the expectation, and the final-eval block
inherits it. No per-metric prediction is possible here and none is claimed.

**Structural change:** the capture gains an `unserved_clients` key in every
metrics block. That is a new key, not a changed value.

**Expect the magnitude to be large.** With windows due in [360, 780], each
abandoned Client is priced at 370–790 against episode totals of 300–640 in the
current capture. If these episodes abandon several Clients, total cost multiplies
rather than shifts. That is the correct reading of a defect that was pricing
abandonment at zero — but **report it explicitly**, because a delay component
that dominates the other three changes the optimization landscape and the
modeling effort needs to know.

## Evidence required

The seven-metric invariance above, verified seed by seed. Mean cost and the four
components over the 60-seed bench, before and after. The count of Clients that
went from unpriced to priced, and the resulting component balance.

## Comments
