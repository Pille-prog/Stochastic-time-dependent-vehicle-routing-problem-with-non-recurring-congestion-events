# 10 — Duplicate-append fix candidate: measure, then the user decides

**What to build:** The Tier-3 candidate. Implement the fixed
`_classify_delayed_clients` semantics (one append per client, after the
closest-vehicle scan finishes) behind a config or constructor switch, run the
comparison study, and present the evidence. **Adoption is the user's call —
this ticket ends with a decision request, not a merge of the fix as default.**
Also the effort's closing measurement.

**Blocked by:** 05, 08, 09.

**Status:** open

- [ ] The fixed classification implemented alongside the faithful quirk
      (ticket 05's vectorized construction), switchable per run; default
      remains the quirk.
- [ ] Comparison study on the real dataset: full training + final test under
      quirk vs fix — mean costs with spread, W trajectories, and episode
      throughput (the fix removes ~V× work from `future_delay`). Results
      recorded in Comments.
- [ ] Decision request to the user with the evidence; if adopted: the fix
      becomes default, ADR-0001-style change-log entry, statistical
      re-baseline, and the quirk path retires. If rejected: the switch is
      removed and the quirk stays documented as deliberate.
- [ ] Effort closing measurement: rerun the ticket-01 fixture benchmark and
      the ticket-01 *scaled* real-dataset protocol (same ~16-episode config,
      apples-to-apples per-phase comparison); run one full Chengdu experiment
      only if its projected wall-clock is now acceptable (expected: it is —
      that is the point). Record final speedups vs baseline and the closing
      profile.
      Stopping rule check: no single site >10% of episode time, or the residual
      sites are listed as future-work notes in the spec.
