# 11 — Close: ADRs, CONTEXT.md, results

**What to build:** Close the effort. Write **ADR-0009**, **ADR-0010** and
**ADR-0011**, finish the CONTEXT.md terms, and write the closing status into
`spec.md` — with whatever the verdict turned out to be.

**Blocked by:** 09 (or 10, if rung 3 ran)

**Status:** open

## ADR-0009 — The reference card and the paired protocol

**Renumbered 2026-07-31 from ADR-0008**, taken first by
`simulator-correctness`/11 (`docs/adr/0008-an-action-must-be-executable.md`).

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
  frozen in spec.md is the protocol that was executed. **Where it was amended
  — decisions 4, 5, 9, 10, 11 and 14, all mid-effort — say what changed, what
  forced it, and that no threshold was ever set from `test_seeds` data.** An
  unrecorded deviation makes every number in the effort unciteable. Include the
  Gate A′ threshold that was amended and reverted within a day; a criterion
  that moved and moved back is part of the record too.
- Records **F12's winner's curse as measured, and as policy-dependent** — the
  finding most likely to mislead the next effort. A `W` selected on
  `evaluation_seeds` reads 2168.39 there against 3384.82 on `test_seeds`
  (×1.56); a policy with no selected parameters reads 3693.23 against 3811.28
  (×1.03). Any quantity defined across the two seed sets is therefore not
  well-defined, and "the gap between the neural policy and the baseline"
  computes to 41.3% or 11.2% depending on which set you ask.
- **One clause on the device** (ticket 12): with `device: "auto"` the config no
  longer pins a result — CPU and CUDA do not agree bit for bit — so the run's
  own record does, resolved once and written into the checkpoint and the
  results. The device itself gets no ADR: where a tensor runs is operational,
  not architectural. What is permanent is that every reported number names the
  device it was produced on.

## ADR-0010 — The approximator is a residual over a frozen myopic base

The architectural decision of the second half of this effort. All three tests
are met: hard to reverse (it is the shape of `Q`), surprising without context
(*why is the cost function hardcoded into the value function?*), and the result
of a real trade-off with four measured alternatives.

- `Q(s, v, a) = c(s, v, a) + W · φ(s, v, a)`, with `c` computed by the
  tokenizer and **structurally unreachable by the gradient** — not a
  `Parameter`, not in any `state_dict`, not a config flag.
- **The evidence:** ticket 08 §4's four attempts (dueling `V` + centred
  advantage, the Huber knee at 0.02, the level gain, and the `cost` warm start
  itself) share one signature — *each makes the optimizer more effective, and
  each only helps when the starting policy is bad.* One finding about where
  `c` lived, not four about three techniques.
- **And the estimator that goes with it:** normal equations accumulated across
  Episodes with exponential forgetting, solved by ridge. `learn` previously
  fitted ~400 samples from a *single* Episode and discarded them, where `U_t`
  is a suffix sum that 595k parameters fit by reading the clock. Research note
  #3, in the effort that had skipped straight to #5.
- **What it costs:** `W = 0` is the myopic base exactly, so "the null" and "the
  initialization" become the same object, and Gate A stops being able to ask
  "is this better than nothing" — it can only ask "is this better than the best
  thing we already had". That is a harder and more useful question, and it is
  the trade being made.
- Record the two rejected alternatives with *why*, since both will be suggested
  again: the dueling decomposition (centring puts the correction where the
  argmin reads, converting a level error into ranking damage) and the Huber
  knee (rejected for a reason specific to Adam, which the new estimator
  removes — so it is *available again*, not settled).

## ADR-0011 — The action set is shared, not owned

Reverses **ADR-0007** ("the action set is feasibility, not heuristic").

- Both Policies call one definition (`action_set.py`, ticket 13). `m + 2` in
  training and evaluation for this Policy; the baseline keeps its own
  `m + 2` / `m + action_count` mismatch untouched.
- **The evidence:** the candidate count is worth **12.68%** to the linear
  baseline (`m+40` 2168.39 vs `m+2` 2483.24 over 50 `evaluation_seeds`, 36/50,
  p = 8.24e-05), reproducing at 2.1% on `test_seeds`; budget 100 beats budgets
  500 and 2000 in every cell; and ticket 08's own "the likely cause is the
  candidate count". The restriction is a regularizer against long myopic hauls,
  and ADR-0007 discarded it by argument rather than by measurement.
- **The second reason is fairness, not performance:** decision 1's amendment
  levelled the two Policies' *cost features* on exactly this ground. The action
  set was the last un-levelled axis, and while it differed a Gate B result was
  not attributable to the approximator.
- **Record the hypothesis that was killed**, because it is the kind that gets
  re-derived: `_create_W` is `np.zeros(19)`, so `W = 0` looks like it must be
  "nearest allowed Client" — but branch 3's `list(set(...))` runs *after* the
  nearest-first sort and returns hash-table order, so `W = 0` picks an
  arbitrary Client and measures **30 791.43**. The linear baseline has no cheap
  myopic null, and the candidate-set-versus-ranking decomposition needed ticket
  14 to exist at all.
- Record what came back with it and was *not* cleaned up: the `350`/`310`
  literals, the `list(set(...))` dedup, the duplicate-append quirk, the
  `< 3 clients` branch. Importing it by halves would be worse than not
  importing it, and ADR-0001's rule holds — never re-tune what is tuned.
- Record what it retired: `_is_retired` and `_depot_is_feasible`, both
  superseded; and that it closed ticket 08's open `claimed_mask` defect as a
  side effect, because `forbidden_actions` is seeded from the other vehicles'
  in-flight commitments.

## CONTEXT.md

Glossary entries only — no implementation detail, no spec content:

- **Approximator** — what maps an observation to `Q`. The variation point
  *inside* Policy: linear weights or a neural network, same decision rule.
  _Avoid_: model, network, estimator
- **Reference card** — already landed (ticket 01). Confirm it still reads true.
- **Myopic base** — the projected cost of one assignment, computed from the
  static `EpisodeGeometry` prior and added to `Q` outside the approximator's
  parameters. _Avoid_: **warm start** (a warm start is a point you move away
  from; nothing moves away from this), **immediate cost** (it is a projection,
  not what the simulator will charge), **post-decision state** (Powell's term,
  reserved — this is a projection, not the state after a decision)
- **Residual approximator** — the only learned term: what the network adds to
  the myopic base. _Avoid_: correction, delta
- **Null policy** — **corrected.** No longer "the same Approximator untrained"
  in the abstract: it is the Approximator at `W = 0`, which under the residual
  decomposition *is* the myopic base. _Avoid_: random policy; "untrained
  network" without naming which base it sits on
- Confirm the **Policy** observability clause from ticket 04 landed and still
  reads true — ADR-0006 was not touched by any of tickets 13-17.

## Closing status in spec.md

Following `.scratch/simulator-correctness/spec.md`'s closing section as the
model:

- [ ] What landed, ticket by ticket.
- [ ] Gate A′'s numbers, **both encoder arms**, and the decomposition
      `delta(trained) − delta(frozen)` as a signed number.
- [ ] Gate B's verdict with the full contender table, components broken out,
      and both baseline numbers (3384.82 and the like-for-like 3458.4).
- [ ] **Ticket 08's `n = 1`, stated plainly** wherever the effort's history is
      summarised. It was closed incomplete and its headline was never
      reproduced.
- [ ] **The measured compute cost against ticket 03's estimate** — the budget
      in spec.md was a napkin calculation; say what it actually was. Include
      what the LSMC path cost against the 15 s/ep the SGD path was paying.
- [ ] Any performance effect on the linear path, disclosed. The
      `simulation-performance` effort's ~6.4× may not be regressed silently.
- [ ] What the effort did **not** answer, and which effort would.

## Follow-on efforts this one names but does not do

- **Multiagent rollout** over this Policy as base (research note §5.1,
  Bertsekas Prop. 2.1) — now the *first* named successor, not the third. The
  one-agent-at-a-time structure preserved in ticket 06 is exactly what it
  builds on, and it comes with a *guaranteed* improvement over its base policy,
  which is a stronger claim than anything this effort can make. If ticket 17's
  failure table lands on "the residual fits and the policy still does not
  improve", this is what that row points at.
- **F14 — the action-space decomposition.** The per-vehicle greedy argmin
  treats other vehicles' targets as fixed. Untouched by this effort, and
  Bertsekas's result is the reason it is sound *only* when the inner evaluation
  is a rollout.
- **Move training out of `Model` into `Trainer`** (debt recorded in ticket 02;
  research note §6.3).
- **The four modeling findings** B4/B6/B8/B13 that `simulator-correctness`
  deliberately excluded. This effort connects B4 (`observed_velocity`, computed
  and discarded) almost incidentally, by putting it in a vehicle token; it does
  not touch the other three.

## Acceptance

- [ ] ADR-0009, ADR-0010 and ADR-0011 written; CONTEXT.md updated; spec.md
      closed.
- [ ] Predicted self-golden diff: **zero.**

## Comments
