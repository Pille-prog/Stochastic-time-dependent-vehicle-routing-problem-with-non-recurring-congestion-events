# 09 — Gate B: the verdict

**What to build:** Answer the effort's objective question against the frozen
protocol, with every contender on the table — including the ones that need no
training.

**Blocked by:** 17 (both encoder arms; the policies this ticket judges come
from the same runs Gate A′ grades — one compute budget, not two)

**Status:** open

## The comparison

> **Gate B: Wilcoxon signed-rank p < 0.05 **and** ≥ 3% mean total-cost
> reduction against 3384.82**, paired over `test_seeds` (100..153), each with
> its fleet size from `test_vehicle_counts`.

`3384.82` is the linear baseline's best cell — best training budget × best
`test_action_count` (budget 100, `m + 40`, from ticket 01). The baseline gets
its best shot; if the transformer still wins, it won.

## The contender table — all of it, always

Ticket 08 closed with the best *policy* in the effort being an untrained one.
A verdict that omits it reports the wrong contender.

| policy | training | mean, 50 `test_seeds` |
|---|---|---|
| linear baseline, best cell (`m+40`) | 100 episodes | **3384.82** ← the verdict is against this |
| linear baseline, `m+2` — same action set as ours | 100 episodes | 3458.4 (like-for-like, reported beside) |
| linear baseline, `W = 0` | **none** | ticket 14 (1) |
| myopic base, shared action set, `W = 0` | **none** | ticket 14 (2) — the null |
| residual, **frozen** encoder | ridge only | ticket 17 |
| residual, **trained** encoder | ridge + SGD | ticket 17 |
| *(historical)* neural, `minutes` warm start, trained | 1150 episodes | 4423.73 |

### Why the verdict is against 3384.82 and not 3458.4

This Policy is confined to `m + 2` by ticket 14, and the baseline's best cell is
`m + 40`, so the action sets are **not** identical at verdict time — which
partially gives back what ticket 14 bought. Both numbers already exist in ticket
01's sweep, so there is no compute argument either way, and the choice was made
deliberately:

- Judging against **3384.82** cannot be accused of anything. Lowering the
  opponent's number after seeing the data is exactly what the anti-p-hacking
  clause exists to stop, even where a legitimate argument for it exists.
- **3458.4** is reported beside it as the attribution number: identical action
  set, identical inputs, only the approximator differs.

The counter-argument is recorded rather than dismissed: a Policy forced to
`m+2` by design is being judged against a cell it was not allowed to reach.
If the verdict lands between the two numbers, **say so in exactly those
words** — "wins like-for-like, loses against the best cell" is a real and
reportable outcome, not a tie to be spun either way.

## What is reported, regardless of outcome

Mean **and** median delta, win count, `p`, effect size with a confidence
interval, wall-clock and training budget for both sides, and the
**per-cost-component breakdown** (distance / delay / earliness / overtime). A
policy that wins on total cost by trading all its earliness for overtime is a
different result from one that wins evenly, and the components say which.

Plus, from ticket 17 and cheap to carry through: `r =
sd_candidates(W·φ)/sd_candidates(c)` at the evaluated checkpoint. It says
whether the winning policy is the base with a light correction or something
that overwrote it — two very different claims about what was learned.

## The pairing is sound (verified in spec.md)

Same seed ⇒ same demand and the same congestion schedule under any policy; only
the velocity realisation differs, because different policies drive different
arcs. Verified against `ClientGenerator.generate` and
`ArcProbabilityCongestionGenerator.generate` (one uniform per arc key
regardless of outcome, rolls at deterministic clock values, no fleet input).

## The anti-p-hacking clause applies here more than anywhere

The protocol is frozen. Not the seeds, not the metric, not the test, not the
3%, not the stopping rule, and **not which baseline number the verdict is taken
against**. If the result is 2.6% at p = 0.04, that is a loss and it gets
written down as one.

## Outcomes, and only these three

1. **Wins** (p < 0.05, ≥ 3% against 3384.82): recorded, and ticket 11 closes
   the effort. Ticket 10 does not run.
2. **Loses**: **ticket 10 runs** — the fleet shared observation memory, on
   whichever encoder arm did better. The one pre-committed escalation, declared
   before any number was measured, and the arm that tests the research note's
   central claim.
3. **Fails to converge** (the safety cap fired): not a result. Diagnose and
   re-run, or record explicitly that the question was not answered within the
   budget. Never presented as a loss — a run that did not converge measured
   nothing.

**A fourth case declared in advance because ticket 14 can produce it:** if the
untrained myopic base over the shared action set already beats 3384.82, Gate B
is won with **zero training** and the effort's remaining question is whether
learning adds anything on top. Report it exactly that way. It is a stronger
result than a trained win and a weaker one for the thesis, and both halves of
that sentence get written.

## Acceptance

- [ ] The verdict, with every number above, in this ticket's Comments.
- [ ] The full contender table filled in — no row left as "not measured".
- [ ] The training curve and the reference curve on one plot.
- [ ] Predicted self-golden diff: **zero.**

## Comments
