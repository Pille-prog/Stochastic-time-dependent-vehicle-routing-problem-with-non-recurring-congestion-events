# 09 — Arm 1: the real run and the verdict

**What to build:** Train the blind transformer to convergence on the real
Chengdu dataset and answer the effort's objective question against the frozen
protocol.

**Blocked by:** 08

**Status:** open

## The comparison

Transformer vs. the linear baseline's **best cell** — best training budget
(100/500/2000, from ticket 01) × best `test_action_count` (2..50). The baseline
gets its best shot; if the transformer still wins, it won.

Paired over `test_seeds` (100..153), each with its fleet size from
`test_vehicle_counts`.

> **Gate B: Wilcoxon signed-rank p < 0.05 and ≥ 3% mean total-cost reduction.**

Report regardless of outcome: mean and median delta, win count, p, effect size
with a confidence interval, wall-clock and training budget for both sides, and
the per-cost-component breakdown (distance / delay / earliness / overtime). A
policy that wins on total cost by trading all its earliness for overtime is a
different result from one that wins evenly, and the components say which.

## The pairing is sound (verified in spec.md)

Same seed ⇒ same demand and the same congestion schedule under any policy; only
the velocity realisation differs, because different policies drive different
arcs. Verified against `ClientGenerator.generate` and
`ArcProbabilityCongestionGenerator.generate` (one uniform per arc key
regardless of outcome, rolls at deterministic clock values, no fleet input).

## The anti-p-hacking clause applies here more than anywhere

The protocol is frozen. Not the seeds, not the metric, not the test, not the 3%,
not the stopping rule. If the result is 2.6% at p = 0.04, **that is a loss** and
it gets written down as one. Wanting to move the threshold after seeing the
number is precisely what the clause exists to prevent.

## Outcomes, and only these three

1. **Wins** (p < 0.05, ≥ 3%): recorded, and ticket 11 closes the effort with the
   result. Ticket 10 does not run.
2. **Loses**: **ticket 10 runs** — the fleet shared observation memory arm, the
   one pre-committed escalation. This is not moving the goalposts; it is the
   second arm declared before any number was measured, and it is the arm that
   tests the research note's central claim.
3. **Fails to converge** (the safety cap fired): not a result. Diagnose and
   re-run, or record explicitly that the question was not answered within the
   budget. Never presented as a loss — a run that did not converge measured
   nothing.

## Acceptance

- [ ] The verdict, with every number above, in this ticket's Comments.
- [ ] The training curve and the reference curve on one plot.
- [ ] Predicted self-golden diff: **zero.**

## Comments
