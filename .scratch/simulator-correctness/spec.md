# Spec: Simulator correctness — the simulator does what it says it does

Status: closed (ticket 10, 2026-07-30 — see "Closing status" below)

## Goal

Close the correctness findings of [`docs/simulator-review.md`](../../docs/simulator-review.md)
and leave behind a machine that would have caught them. The deliverable is **not
thirteen bug fixes** — it is the invariant suite, the measurement bench and the
config sweep. The fixes are what happens when you write the invariants down and
run them.

Success is not "these 19 findings are gone". Success is that the next defect of
this shape fails a test instead of surviving until someone reads the code.

## Point of departure

The review audited `simulation/`, `traffic/`, `congestion/`, `policies/`,
`demand/` and `network/` across seven dimensions, put every finding through an
adversarial refutation pass that re-implemented the proposed fix and measured
it, and instrumented hundreds of real episodes on `tests/fixtures/chengdu_mini`.
22 of 23 findings survived; 1 was refuted. **The core worry did not confirm** —
distance/velocity integration, clock monotonicity and the ledger are correct.
19 findings remain.

**The whole suite passes today, with all 19 present.** That is not a criticism
of the tests: the ±40% statistical gate and the bit-exact self-golden were built
to *pin behavior during a refactor*, and they did that job. But a
behavior-pinning gate defines "correct" as "what it did yesterday", and what it
did yesterday *is* the 19 findings. None of the three existing gates could have
detected any of them.

Two measurements taken while writing this spec, both load-bearing:

- **`final_w[10] == 0.0` exactly** — B10's untrainable weight, confirmed in the
  committed capture.
- **All 15 self-golden episodes terminate all-back between tau 396 and 502** —
  none reaches the shift end (780), let alone the episode end (1150). The entire
  capture sample therefore sits inside the blast radius of B3 (abandonment
  priced at zero) *and* B1a (the depot-idle rule is active above tau 350) at
  once. Ticket 01 must report per-seed unserved-Client counts to size this.

## Decisions (from the grilling session)

| # | Decision | Choice |
|---|---|---|
| 1 | What "no errors" means | **Two efforts.** This one: *the simulator does what it says* (internal contradictions between stated intent and code). Modeling fidelity — what the simulator *should* say — is a separate downstream effort |
| 2 | B3's exception | In scope despite being a modeling decision: it is **the yardstick, not a measurement**. While episode cost depends on when the fleet happened to stop, no other fix can be measured |
| 3 | Is the Policy in scope | **Yes, with a strict test**: fix what *misclassifies* or *crashes*; never re-tune what is *tuned*. Literals 350/310 stay; what changes is what they apply to |
| 4 | Root cause of B1a/B1b | **Fix the domain model, not the call sites.** `State` gains an explicit, Model-maintained "standing at a node" fact and `vehicle_position` gets an honest name. Rejected: handing `FleetRoutes` to the Policy (breaks the ticket-09 boundary and leaves the ambiguous predicate alive at four other sites) |
| 5 | The horizon | **Two named concepts, no value moved**: `shift_end_minute` (780, overtime threshold) and `episode_end_minute` (1150, hard stop), both configurable, with `shift_end_minute <= episode_end_minute` validated. Rejected: making the configured end actually bound the episode (that is a model change, and it would delete the concept of overtime) |
| 6 | B3's reference clock | `max(episode_end_minute, tau) − due`, floored at 0, **no fixed abandonment charge**. It is the only candidate under which abandoning is never cheaper than serving late, it introduces no free parameter, and it leaves horizon-terminated episodes essentially untouched |
| 7 | The acceptance contract | **Invariant before fix, same ticket.** Each finding becomes an executable property that fails before the fix and passes after — that is what proves the bug existed, rather than trusting the review |
| 8 | The statistical gate | `test_new_package_vs_golden_master`'s three ±40% tolerance tests **retire** in ticket 03. Their purpose was continuity with the legacy, and B3 severs that deliberately; comparing against a baseline computed under a different cost function is noise wearing the costume of a guarantee. `chengdu_full_phase2.json` stays in the repo as evidence, with no test reading it |
| 9 | The self-golden | **Not re-captured once at the end — diffed every ticket.** Predict the diff in the ticket *before* running it; accept only if what moved is what was predicted. Plus a **new third capture block: evaluation with a frozen literal W**, which is the only decision-stable block and therefore the only one where attribution is surgical |
| 10 | When a diff surprises | Three outcomes, only three: **matches** → accept and re-capture; **explained to a mechanism** → record the explanation *and the original prediction, unedited*; **unexplained** → revert, the ticket does not land. A prediction that can be rewritten after seeing the result was never a prediction |
| 11 | Evidence per ticket | Invariant red→green, plus mean cost and its 4 components over a **fixed 60-seed set** on the mini fixture (the review's own N, so the numbers stay comparable to the report). Physics tickets also report km driven. No real-dataset run per ticket |
| 12 | What closes the effort | **The config sweep** (three of the 19 were configuration traps invisible at the shipped values) plus one scaled real-data run. The full training run is explicitly **out** — that is the first experiment of the repaired lab, not part of repairing it |

## Scope: the 19 findings, allocated

**In (15).** Every one is an internal contradiction: stated intent versus code.

| Finding | What is fixed | Ticket |
|---|---|---|
| B1a, B1b | The depot chain, at the root: `State` gains the missing fact, `vehicle_position` is renamed | 04 |
| B3 | Termination prices against a fixed reference clock | 03 |
| B5 | The empty-`min()` crash — **the fallback only**, not the 310/350 unification | 05 |
| B7 | Congestion writes compose instead of replacing | 07 |
| B9 | Spread is breadth-first, so depths are minimal | 08 |
| B10 | The fourth earliness bin exists — **the cuts are not redesigned** | 06 |
| B11 | The endgame branch filters `forbidden_actions` | 05 |
| B12, B15, B16 | Two named clocks, validation, integer cadence arithmetic | 02 |
| B14 | Charges increment their counters; `unserved_clients` is its own | 03 |
| B17 | The congestion book purges | 07 |
| B18 | Docstring + parameter rename (`n_observed_arcs` promises arcs, delivers observations) | 09 |
| B19 | **The docstring, not the anchors** — ADR-0001 fix 2 already chose to preserve 418/542 | 09 |

**Out (4), to the modeling effort.** The code does what it says; whether what it
says is right is a science question.

- **B4** — `mean_velocities` is computed and discarded. Connecting it is an
  experimental-design decision, and `docs/research/rl-methodology-for-stdvrp.md`
  will reopen it.
- **B6** — features measure from the node the vehicle left. Not a miswritten
  predicate: *what the Policy should observe*. Same note, same effort.
- **B8** — multiplier saturation makes distance damping inert. **The most
  arguable exclusion**: ADR-0001's fix 7 states its purpose was to *resurrect*
  the 0.73 damping, and the saturation it introduced kills it again under the
  shipped 0.3/0.4 bounds — a documented fix that contradicts itself. Excluded
  anyway because *how* to correct it (saturate? rescale? move the bounds?) has
  no obvious answer, and `docs/research/congestion-no-recurrente-y-rl.md` will
  reopen the generator wholesale. Ticket 09 records the inertness as an ADR-0001
  addendum so nobody reads fix 7 and believes the damping is live.
- **B13** — the 60 km/h cap censors the distribution onto an atom. The cap does
  what it says; that capping collapses mass is a *consequence* of the modeling
  choice. Its sibling (the `0.001` floor, which is not a slow vehicle but a
  frozen one) goes with it.

## Behavior contract, precisely

Three instruments, three different questions. None substitutes for another.

| Instrument | Question it answers | Where |
|---|---|---|
| **Self-golden, frozen-W block** (new) | Did *exactly* what I predicted change, and nothing else? | `scripts/capture_self_golden.py`, ticket 01 |
| Self-golden, training + final-eval blocks | Did the learned outcome move, and from which episode? | existing |
| **Measurement bench** (new) | Why did it move, and by how much, with N large enough to mean something? | `scripts/`, ticket 01 |
| **Config sweep** (new) | Does it hold up away from the shipped parameter point? | ticket 10 |

**Why the frozen-W block is necessary.** The existing protocol trains 5 seeds
carrying W, then evaluates 10 seeds *with the final trained W*. Both blocks hang
off learning. B3's termination charge enters the Monte Carlo return, so it moves
W from the first affected training episode, so **everything diverges** — and
against a total divergence you cannot verify that only the intended thing
changed. A block run with a fixed literal W has no learning feedback: cost
changes do not move decisions, so the diff is the fix's direct effect and
nothing else.

**The diff cascades.** From ticket 03 on, each ticket diffs against a capture
that already contains its predecessors. Re-capturing an unvalidated diff poisons
every downstream ticket.

**The gate is local.** `test_self_golden` skips by design when the environment
differs from the capture (numpy's float draws are not cross-platform bit-exact).
Verified 2026-07-28: this machine matches the capture exactly (numpy 2.4.6,
Python 3.11.9, Windows, AMD64). It is a mandatory local step; CI cannot enforce
it.

## The invariant catalogue

One per finding, written before its fix, in its ticket:

| Invariant | Finding | Ticket |
|---|---|---|
| No vehicle becomes `PARKED` while `departure_tau < tau < arrival_tau` | B1a/B1b | 04 |
| A vehicle's recorded node never changes without charged distance (no teleporting) | B1b | 04 |
| The termination charge depends only on `due` and config — **never on `tau`** | B3 | 03 |
| Money charged > 0 ⟺ its counter > 0 | B14 | 03 |
| The Policy returns a legal action for every vehicle, at every tau, under every valid config | B5 | 05 |
| Two vehicles never receive the same non-depot Client in one decision | B11 | 05 |
| The four earliness bins partition pending demand | B10 | 06 |
| A congestion write never shortens an active event's expiry | B7 | 07 |
| The book holds no expired entries after a clock advance | B17 | 07 |
| Every node within `max_depth` is congested, at its true minimum depth, independent of arc-table order | B9 | 08 |
| No cost component is negative **under any valid config** | B15 | 02 |
| Congestion cadence fires the expected number of times for every integer duration | B16 | 02 |

The existing suite already asserts the last one's neighbours only at
`horizon_end = 780`, which is exactly why B15 went unseen.

## Tickets

Critical path: 01 → 02 → 03 → 04. Tickets 05–09 are independent of each other
and of 04; 10 closes.

| # | Ticket | Blocked by | Predicted self-golden diff |
|---|---|---|---|
| 01 | Measurement bench + frozen-W capture block | — | **zero** on existing blocks |
| 02 | Name the two clocks | 01 | **zero** |
| 03 | The reference clock for abandoned demand | 02 | predicted in-ticket |
| 04 | The State says where the vehicle is | 03 | predicted in-ticket |
| 05 | Endgame: no crash, no double booking | 01 | predicted in-ticket |
| 06 | The fourth earliness bin | 01 | **zero** in the frozen-W block (`final_w[10] == 0.0`) |
| 07 | Congestion book: compose and purge | 01 | predicted in-ticket |
| 08 | Breadth-first spread | 01 | predicted in-ticket |
| 09 | Honest documentation | 01 | **zero** |
| 10 | Robustness gate: config sweep + scaled real run | 02–09 | **zero** (no production change) |

Three tickets predict an exactly-zero diff (02, 09, and 06's frozen-W block).
Those are the strongest claims in the effort: if a rename or a docstring moves a
single float, it was not a rename.

## ADRs this effort writes

Each is written *inside its ticket*, when the decision is executed — not up
front.

- **ADR-0004** (ticket 03) — the episode clock and the price of abandoned
  demand; records why `shift_end` and a fixed no-service charge were rejected,
  and retires the statistical baseline.
- **ADR-0005** (ticket 04) — the State says where the vehicle is; records why
  handing `FleetRoutes` to the Policy was rejected.
- **ADR-0001 addendum** (ticket 09) — under the shipped bounds, distance damping
  is inert (B8).

`CONTEXT.md` gains terms in tickets 03 and 04. It also needs a correction found
while writing this spec: it currently defines **Horizon** as *"the simulated time
interval (start minute, end minute) within which all decisions and events
happen"* — false, since with horizon (300, 780) episodes demonstrably run to
1148. The bug of decision 5 is written into the ubiquitous language, not only
into the code.

## Out of scope (deliberately)

- The four modeling findings (B4, B6, B8, B13) — separate effort, informed by
  the two research notes.
- Redesigning the earliness bin cuts, the 310/350 threshold disagreement, moving
  the 418/542 std anchors, splitting congestion cadence from duration. All are
  "the tuning is wrong" rather than "the code contradicts itself".
- The full Chengdu training run. It is the first experiment of the repaired lab.
- Performance. The `simulation-performance` effort closed at ~10× throughput;
  nothing here may regress it without saying so.

## Closing status (ticket 10, 2026-07-30)

**What landed.** All 15 in-scope findings — B1a, B1b, B3, B5, B7, B9, B10, B11,
B12, B14, B15, B16, B17, B18, B19 — have a red-before/green-after invariant,
written before its fix in the same ticket per decision 7, all landed across
tickets 02–09. Three ADRs written inside their tickets as planned: 0004 (the
episode clock and the price of abandoned demand), 0005 (the State says where
the vehicle is), and an 0001 addendum (B8's inertness). `CONTEXT.md` gained the
two-clock and abandoned-demand terms (tickets 02/03) and the State's
two-meanings distinction (ticket 04). The statistical gate retired as planned
(ticket 03); `chengdu_full_phase2.json` stays as evidence nothing reads. This
ticket closes the effort with a 432-combination config sweep (`tests/
test_config_sweep.py`) and a 16-episode real-scale correctness check plus a
wall-clock baseline against the real 1900-node dataset — the "machine that
would have caught them" decision 0's success criterion asked for, not just the
19 fixes. No production code changed in ticket 10 itself.

**Cumulative measured cost, ticket 01's baseline → now** (60-seed mini-fixture
bench, three configurations — full detail and per-ticket attribution in ticket
10's own Comments):

| Config | mean_total_cost before | after | change |
|---|---|---|---|
| Default fleet (~5 vehicles), `--w zero` | 512.194224 | 515.646000 | +0.67% |
| Default fleet, `--w frozen` (real trained policy) | 481.687896 | 474.679231 | −1.45% |
| `--vehicle-count 1` (the review's own stress config) | 1831.729883 | 2862.068295 | **+56.25%** |

Multi-vehicle fleets absorb almost all of the effect via redistribution slack;
the single-vehicle config the review used to measure B1b/B3 shows it at full
strength, because B1a/B1b's extra driving and B3's now-honest abandonment price
have no alternative capacity to hide behind. Every violation counter (B1a/B1b,
B7, B9, B10, B11, B14, B15, B17) is zero on every configuration measured.

**Real-scale confirmation.** 16 real episodes on the true 1900-node Chengdu
network (the exact `baseline_scaled.yaml` shape): zero violations on every
counter; B9's structural check shows 0/38379 wrong-depth and 0/38379
missed-entirely, confirming ticket 08's fix at the scale the review's own
"~15%" figure was originally measured against.

**Performance, disclosed as the out-of-scope clause above requires.** The
`simulation-performance` effort's ~10× projected full-run speedup is now
**~6.4×** measured the same way (8669.3s baseline → 1356.2s now, vs. 869.2s at
that effort's own close) — a real, ~1.56× wall-clock increase, not machine
noise alone. Explained to a mechanism, not a new finding (decision 10): none of
tickets 02–09 touched the vectorized architecture itself (world cache, episode
geometry, feature vectorization all structurally unchanged); the increase is
more real simulated work per episode — on the matching single-vehicle bench,
`mean_decisions` rose +31.8% and `mean_km_driven` +36.4%, the same effect
tickets 03/04/07/08 already measured in cost terms, now also visible in wall
clock. A fleet that used to wrongly retire early or under-cover the network
with congestion did less work per episode; fixing that costs more wall clock
by design.

**Findings that did not reproduce exactly** (recorded in ticket 01, not
new): B11's "734 transitions" and B17's "116/116 expired" review headline
figures did not reproduce at their literal magnitude on this fixture — the
underlying defects did, and both were fixed as scoped regardless.

**New findings from the sweep: none that need their own ticket.** Two
observations surfaced and are both explained to an already-on-record
mechanism rather than opened as new tickets: `scripts/measurement_bench.py`'s
`check_b16_cadence` is a frozen, never-updated reproduction of the *pre-fix*
formula (ticket 10's Comments); and the performance change above. Neither is
an unexplained defect in the simulator.

**Self-golden: exactly zero diff**, as predicted — this ticket adds tests and
runs things, and changes no production code. No re-capture needed. Full suite:
4039 passed, 3 deselected (golden-marked, need the real dataset).

**Out of scope, unchanged**: the four modeling findings (B4, B6, B8, B13) go to
a separate, downstream effort informed by `docs/research/`. The full Chengdu
training run is the first experiment of the repaired lab, not part of
repairing it, and is deliberately not run here.
