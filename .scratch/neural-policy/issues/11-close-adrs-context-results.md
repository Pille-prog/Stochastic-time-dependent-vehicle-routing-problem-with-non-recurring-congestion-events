# 11 — Close: ADRs, CONTEXT.md, results

**What to build:** Close the effort. Write **ADR-0008**, finish the CONTEXT.md
terms, and write the closing status into `spec.md` — with whatever the verdict
turned out to be.

**Blocked by:** 09 (or 10, if arm 2 ran)

**Status:** open

## ADR-0008 — The reference card and the paired protocol

- Retires `static_policy_mean_cost`: one hardcoded scalar supported only
  comparing means; the per-seed vector supports the paired test the acceptance
  contract rests on.
- Records the **verified independence argument**: same seed ⇒ same demand and
  the same congestion schedule under any policy (one uniform per arc key
  regardless of outcome, rolls at deterministic clock values, no fleet input,
  purge timing does not change the resulting book). Only velocities diverge.
  This is why pairing is legitimate here and should not be re-derived by the
  next person who wonders.
- Records **the verdict**, including a negative one, with its numbers.
- Records the anti-p-hacking clause as having been honoured: the protocol as
  frozen in spec.md is the protocol that was executed. If anything deviated,
  say what and why — an unrecorded deviation makes every number in the effort
  unciteable.

## CONTEXT.md

Glossary entries only — no implementation detail, no spec content:

- **Approximator** — what maps an observation to `Q`. The variation point
  *inside* Policy: linear weights or a neural network, same decision rule.
  _Avoid_: model, network, estimator
- **Reference card** — a completed Policy's frozen per-seed costs; the fixed
  opponent every later run is measured against. _Avoid_: baseline (overloaded —
  it also names the sunk-cost term subtracted from the Monte Carlo target)
- **Null policy** — the same Approximator untrained. The floor a trained Policy
  must clear before "it learned" means anything. _Avoid_: random policy (it is
  not random — the myopic warm start makes it nearest-Client)
- Confirm the **Policy** observability clause from ticket 04 landed and still
  reads true.

## Closing status in spec.md

Following `.scratch/simulator-correctness/spec.md`'s closing section as the
model:

- [ ] What landed, ticket by ticket.
- [ ] Gate A's three numbers.
- [ ] Gate B's verdict for every arm that ran, with components broken out.
- [ ] **The measured compute cost against ticket 03's estimate** — the budget
      in spec.md was a napkin calculation; say what it actually was.
- [ ] Any performance effect on the linear path, disclosed. The
      `simulation-performance` effort's ~6.4× may not be regressed silently.
- [ ] What the effort did **not** answer, and which effort would.

## Follow-on efforts this one names but does not do

- **Move training out of `Model` into `Trainer`** (debt recorded in ticket 02;
  research note §6.3).
- **Multiagent rollout** over this Policy as base (research note §5.1,
  Bertsekas Prop. 2.1). The one-agent-at-a-time structure preserved in ticket 06
  is exactly what it builds on, and it comes with a *guaranteed* improvement
  over its base policy — which is a stronger claim than anything this effort
  can make.
- **The four modeling findings** B4/B6/B8/B13 that `simulator-correctness`
  deliberately excluded. This effort connects B4 (`observed_velocity`, computed
  and discarded) almost incidentally, by putting it in a vehicle token; it does
  not touch the other three.

## Acceptance

- [ ] ADR-0008 written, CONTEXT.md updated, spec.md closed.
- [ ] Predicted self-golden diff: **zero.**

## Comments
