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

**Status:** resolved

- [x] `_charge_unserved_delays` prices **every** unserved Client at
      `max(0, max(episode_end_minute, tau) − due)`. The live in-episode formula
      is untouched — it is correct.
- [x] **B14**: charges increment their counters. `unserved_clients` is its own
      metric, **not folded into `late_clients`** — under the new clock every
      abandoned Client generates a charge, so lumping them together would
      conflate "served late" with "never served". `charge_fleet_overtime`'s
      `overtime_vehicles` undercount gets the same treatment.
- [x] Invariant: **the termination charge is a function of `due` and config
      only, never of `tau`.** A property test that terminates the same
      unserved set at several clocks and asserts one price. This invariant *is*
      the definition of comparability across episodes.
- [x] Invariant: money charged > 0 ⟺ its counter > 0, for every component.
- [x] **ADR-0004** — the episode clock and the price of abandoned demand.
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
- [x] **Retire the statistical gate.** Delete the three ±40% tolerance tests in
      `tests/test_new_package_vs_golden_master.py`. Their purpose was continuity
      with the legacy and this ticket severs it deliberately; a baseline
      computed under a different cost function cannot gate a run under this one.
      `chengdu_full_phase2.json` **stays in the repo** as historical evidence,
      with no test reading it. Recorded in ADR-0004.
- [x] **`CONTEXT.md`**: fix **Horizon**, which is currently false — it claims
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

### Resolution (2026-07-30)

Landed on an isolated worktree (`worktree.baseRef: head`), same pattern as
ticket 02 — this branch has several sessions landing simulator-correctness
tickets concurrently ([[stdvrp-refactor-status]], [[git-pathspec-commit-stages-worktree]]).

`Model._charge_unserved_delays` no longer filters unserved Clients by
`tau_episode > due`; it prices every one of `state.clients_not_visited` at
`max(0.0, max(episode_end_minute, tau_episode) - due)`, then hands the whole
generator to `CostLedger.charge_unserved_delays`. Both termination paths call
it unchanged. `charge_unserved_delays` and `charge_fleet_overtime` now count
only the Clients/vehicles they charge a *strictly positive* amount to
(`unserved_clients`, `overtime_vehicles`) — not every item passed in — which
is what makes "money charged > 0 iff its counter > 0" hold given the
non-negative-per-item inputs the Model actually ever passes (each unserved
Client's price is already floored at 0.0 before it reaches the ledger).
**Correction, `/code-review` follow-up**: this is narrower than "for any
input" — `charge_unserved_delays([5.0, -5.0])` sums to a `delay_cost` of 0
while still counting one Client (the `5.0` is `> 0`), so the biconditional is
not an unconditional property of the method in isolation, only of how the
one real caller uses it. `charge_fleet_overtime` does not have this gap:
`minutes_over` is a single scalar per call, so a negative one drives `cost`
non-positive for every vehicle it would have counted, and the `> 0` guard
keeps the counter at 0 in lockstep. Left as documented behavior rather than
hardened against out-of-domain input `charge_unserved_delays` has exactly one
caller for (per this repo's convention of not validating what cannot happen,
CLAUDE.md) — noted here instead of silently overclaimed.
`EpisodeResult` and `EPISODE_METRICS` gain `unserved_clients` as a tenth
metric; `scripts/measurement_bench.py`'s `_money_without_counter_violations`
updated for the delay component now having two contributing counters.

**The predicted self-golden diff did not match — explained, not reverted**
(spec.md decision 10). Measured: **zero** leaves moved between a fresh
pre-ticket-03 capture and a fresh post-ticket-03 capture on identical code
otherwise (`unserved_clients` is 0 on all 15 episodes; every one already
serves its full demand). The ticket's prediction assumed ticket 01's
point-of-departure measurement, taken *before* ticket 08 (breadth-first
congestion spread, B9) landed — ticket 08's fix changed which arcs congest
enough that this exact 15-seed protocol no longer exercises the abandonment
path at all. Verified instead on the 60-seed bench (`--vehicle-count 1`,
ticket 01's own probe for this demand, `unserved_overdue` 42/60 episodes,
151 Clients, unchanged before/after since decisions don't move under a fixed
W): `mean_delay_cost` 1713.330610 → 3072.730610, `mean_total_cost` by the
identical delta, `mean_distance_cost` / `mean_earliness_cost` /
`mean_overtime_cost` / `mean_final_tau` / `mean_decisions` bit-identical
before/after — the seven-metric frozen-W invariance the ticket asked for,
confirmed on the bench that actually exercises the fix. **B14, directly
measured on the same run**: `money_without_counter` violations 33 across
31/60 episodes before this fix, 0 after.

**The self-golden fixture still needed recapturing** — the new
`unserved_clients` key is structural (every metrics block gains it, value 0
throughout on this fixture) — and that recapture necessarily also reflects
ticket 08's own code (`c900423`), which changed episode behavior without ever
committing a matching recapture, a gap [[stdvrp-refactor-status]] already
documents and ticket 02 deliberately left alone. The 177 leaves that move
against the previously-committed `mini_fixture.json` are therefore ticket
08's effect, not this ticket's — isolated by comparing two fresh captures
differing only in this ticket's own changes (0 leaves moved, above). Recorded
in ADR-0004 rather than re-litigated here.

`tests/test_new_package_vs_golden_master.py` deleted outright (all three of
its tests were the ±40% tolerance checks). `scripts/rebaseline_golden_master.py`
kept (nothing else in scope asked for its removal) but its docstring now
notes the retirement and that nothing reads its output anymore.

`tests/unit/test_model_termination.py` gained `episode_end_minute` on its
`make_terminating_model` fixture (defaulting to 1150) plus a Hypothesis
property (`TestReferenceClockInvariant`) sweeping `due`, `episode_end_minute`
and several `tau` values within `[horizon_start, episode_end_minute]` —
the model's own reachable domain, since both termination call sites only
ever fire with `tau_episode <= episode_end_minute` — and asserting exactly
one resulting price. Two of the file's pre-existing tests were pinning the
*old*, buggy behavior (a Client within its still-open window priced at 0)
and are now the fix's own regression tests, renamed and updated in place.
`tests/unit/test_cost_ledger.py` gained the CostLedger-level counterpart:
per-method Hypothesis properties asserting `(cost > 0) == (counter > 0)` for
`charge_unserved_delays` and `charge_fleet_overtime` in isolation (each
against a fresh `CostLedger()`, not the shared `ledger` fixture — Hypothesis
re-runs the test body per example but pytest builds a function-scoped fixture
only once, which would silently accumulate charges across examples).
`tests/test_invariants.py`'s existing Hypothesis suite needed its expected-
delay computation updated to the new formula (it was asserting the old one);
green afterward without further changes.

Full non-golden suite: 3600 passed. `ruff check` / `ruff format --check` /
`mypy src/stdvrp` clean.

### `/code-review` follow-up (2026-07-30)

Standards axis: no hard violations. Two judgement-call smells noted, not
acted on — `charge_unserved_delays`/`charge_fleet_overtime`'s shared
loop-then-conditionally-count shape (extracting it would fight the module's
own bit-exactness-per-method documentation) and the reference-clock formula
being reimplemented once in production and twice across tests (normal for
independent verification). One cosmetic docstring line fixed
(`test_cost_ledger.py`, stray blank line before a docstring's own body).

Spec axis found two real gaps, both fixed: `scripts/capture_self_golden.py`'s
docstring still said "nine metrics" / "nine golden-pinned metrics" in two
places — stale now that `EPISODE_METRICS` has ten. And this Comments
section's own claim that the money/counter invariant holds "independent of
caller discipline" was overstated — `charge_unserved_delays([5.0, -5.0])`
sums to `delay_cost == 0` while still counting one Client, so the property
holds only given the non-negative-per-item inputs the Model actually ever
passes (each price is pre-floored at 0.0), not for arbitrary input to the
method in isolation. Corrected in place above rather than silently
overclaimed; not treated as a code defect since `charge_unserved_delays` has
exactly one caller and it cannot produce that input (CLAUDE.md: don't harden
against what cannot happen).
