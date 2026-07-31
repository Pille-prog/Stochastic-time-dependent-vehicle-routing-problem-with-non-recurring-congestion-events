# 08 — Breadth-first spread

**What to build:** `_reachable_nodes` (`congestion/generator.py:136-164`) calls
itself "BFS-by-recursion" in its docstring, but the recursion is **depth-first
with a single `visited` set**. A node first discovered down a deep branch keeps
that depth, receives the wrong damping factor, and — if its recorded depth
reaches `max_depth` — stops expanding. Closes B9.

**Measured:** **~15% of the nodes genuinely within `max_depth`** of an epicentre
are never congested, and *which* arcs the spread reaches depends partly on the
**row order of the arc table** in `successors` rather than on network topology.
It runs twice per triggered event (once per endpoint), ~64 events per epoch, ~8
epochs per episode.

**Blocked by:** 01

**Status:** resolved

- [x] Real breadth-first traversal: every node is recorded at its **minimum**
      depth from the epicentre.
- [x] Invariant: every node within `max_depth` is congested, at its true minimum
      depth, and **the result is independent of arc-table order** — shuffle
      `successors` and assert the same congested set. That second half is what
      pins the actual defect; depth correctness alone would not catch an
      order-dependent traversal.

## Predicted self-golden diff

**Full divergence on every seed, in all three blocks, including frozen-W** —
same reason as ticket 07: this changes which arcs are congested, hence the
velocities drawn.

**Direction: costs rise.** ~15% more nodes fall inside each event's blast radius,
so more of the network is slow, so travel takes longer.

**One thing that must *not* show up.** The review's caveat: under the shipped
`0.3`/`0.4` bounds the damping factor is invisible because multipliers saturate
(B8), so this fix changes **which** arcs are congested, not **by how much**. The
distribution of multiplier *values* across the book should therefore be
essentially unchanged — still ~3/4 of congested arcs sitting at exactly 0.4
regardless of hop distance. If the multiplier distribution shifts, this fix
touched damping, which is B8's territory and out of scope for this effort.

## Evidence required

The order-independence invariant green. Node coverage before/after (was ~85% of
in-range nodes). The multiplier-value distribution before/after, showing it did
**not** move. The 60-seed bench before/after with direction.

## Comments

### Resolution (2026-07-30)

`_reachable_nodes` (`src/stdvrp/congestion/generator.py`) rewritten as a plain
queue-based BFS: a FIFO processes nodes in non-decreasing depth order, so the
first time a node is dequeued and given a depth, that depth is minimal by
construction, and every node is assigned exactly once. The recursive
`visited`/`node_depth` accumulator parameters are gone — no longer needed once
traversal isn't recursive. Two docstrings and one dead comment that described
the old (false) "BFS-by-recursion" behavior or the arc-table-order dependency
as preserved behavior were corrected to describe the real, order-independent
BFS. No other file changed.

**Tests** (`tests/unit/test_congestion_generator.py`, new `TestBreadthFirstSpread`
class):

- Two hand-built regression graphs, each isolating one of B9's two distinct
  symptoms named in ticket 01's `check_b9_spread_depths`: (1) a node reachable
  by both a long and a short path keeps the *short* depth regardless of which
  order `successors` lists the two branches in (`{0: [1, 3], 1: [3], 3: []}` —
  node 3 is at true depth 1 via the direct edge, but the old DFS recorded
  depth 2 when `successors[0] == [1, 3]` and the correct depth 1 when reversed
  to `[3, 1]` — order determined the answer); (2) a node genuinely within
  `max_depth` that the old DFS never reaches at all (`{0: [1, 2], 1: [3], 3:
  [2], 2: [4]}` — the long branch 0→1→3→2 hits `max_depth` exactly at node 2,
  so it never tries node 2's own neighbor, node 4, and the direct 0→2 edge
  that would have reached node 4 at depth 2 is skipped because DFS already
  marked node 2 visited via the long branch).
- A Hypothesis property (200 examples, derandomized) generating random graphs,
  asserting `_reachable_nodes` matches an independent reference BFS and that
  the result (both the reached set and every recorded depth) is unchanged
  after shuffling every node's neighbor-list order.

**Red-before-green**, verified without touching the shared working tree (see
below): a standalone replica of the pre-fix recursive `_reachable_nodes`
reproduces both regression cases exactly as predicted — `depth[3] == 2` under
`successors[0] == [1, 3]` and `depth[3] == 1` under `[3, 1]` (order determines
the answer, pre-fix); node 4 is absent from the reached set entirely
(`{0, 1, 2, 3}` vs. the true `{0, 1, 2, 3, 4}`). Both assertions the new tests
make would have failed against the old code and pass against the new one.

**A concurrent-editing note.** Multiple other sessions were live-editing this
same working tree while this ticket ran (tickets 02/05/06/07/09 all had
uncommitted or in-flight work at various points; ticket 09 landed mid-session,
moving `HEAD` from 55f32aa to e603c87 — this branch has a documented history
of concurrent sessions editing the same working tree, sometimes the same
file; see the generic-stdvrp-refactor and simulation-performance efforts'
tickets for prior instances). To get before/after measurements attributable
*only* to this ticket's two-file diff, all benchmarking and the full
test-suite run below used an isolated
`git worktree` at the ticket-01 baseline commit (55f32aa) with just this
ticket's `generator.py` copied in — not the shared, concurrently-edited
working tree. Confirmed the capture file
(`tests/fixtures/self_golden/mini_fixture.json`) is byte-identical between
55f32aa and the later e603c87 (ticket 09's own predicted zero diff), so this
baseline choice doesn't affect validity against current `HEAD`.

### Node coverage before/after (structural, `check_b9_spread_depths`)

| | before | after |
|---|---|---|
| `B9_spread_wrong_depth` | 38 / 635 reached (5.98%) | **0** / 675 reached (0.00%) |
| `B9_spread_missed_entirely` | 40 / 675 genuinely reachable (5.93%) | **0** / 675 (0.00%) |

`reached_total` rose from 635 to 675 — exactly matching `genuinely_reachable_
total` (675) now that nothing is missed. Identical on every bench configuration
tested (default fleet, frozen-W, `--vehicle-count 1`, `--capture-seeds`) since
the check is structural (topology + `max_depth` only, independent of any
Episode). Confirms the review's own "~15%" framing was a smaller-graph analogue
of this same defect: ticket 01 measured 5.9-6.0% on the mini fixture and
attributed the gap to the review's cross-check having run against the larger
real Chengdu network.

### Multiplier-value distribution before/after (the "must not move" check)

Ran the real fixture's `event_probability`/`successors` through both the
pre-fix (DFS, replicated) and post-fix (BFS) `_reachable_nodes`, 200 synthetic
epochs, each old/new pair fed identical RNG draws (traversal consumes no
randomness itself, so this isolates the traversal's effect on end values):

| | before | after |
|---|---|---|
| arcs written | 15569 | 16245 (+4.3%, consistent with the ~5.9% missed-node figure) |
| at exactly `congestion_upper_bound` (0.4) | 11235/15569 (72.2%) | 11597/16245 (71.4%) |
| top-10 multiplier values | `[(0.4, 11309), (0.396, 153), (0.385, 151), ...]` | `[(0.4, 11684), (0.396, 168), (0.399, 163), ...]` |

More arcs get congested; the *value* distribution is unchanged within noise —
confirms this fix touched **which** arcs, not **by how much**, exactly as
required. If this had shifted, it would have meant the fix strayed into B8's
territory (out of scope for this effort).

### The 60-seed bench before/after, with direction

| config | before total_cost | after total_cost | direction |
|---|---|---|---|
| default fleet, `--w zero` | 512.194 | 508.314 | **−0.76%** |
| default fleet, `--w frozen` | 481.688 | 480.642 | **−0.22%** |
| `--vehicle-count 1`, `--w zero` | 1831.730 | 1972.425 | **+7.68%** |

The ticket predicted "costs rise" flatly. What's measured is more precise, and
the mechanism explains the split (spec decision 10: "explained to a mechanism
→ record the explanation and the original prediction, unedited" — this is
that case, not a match and not unexplained):

- **`--vehicle-count 1` shows the predicted rise clearly and by a lot**
  (distance +6.2%, delay +8.1%, overtime +16.4%, final tau +3.5%, decisions
  118.58→123.33). This is the review's own single-vehicle configuration: one
  vehicle has no alternative routing to absorb newly-congested arcs, so more
  congestion (now correctly covering the previously-missed ~6% of nodes)
  mechanically extends its trip.
- **The default (~5-vehicle) fleet under `--w zero` shows a small, mixed
  effect** because `--w zero` makes the greedy policy's `argmin` degenerate —
  every candidate scores identically (`X @ 0 == 0` for every feature), so the
  tie-break (`np.argmin`'s first-match) picks by a fixed candidate order
  entirely independent of cost or congestion. Assignment choices don't
  respond to which arcs are congested at all; any total-cost movement is a
  second-order effect of realized travel-time changes shifting *when* future
  decision epochs land (hence which clients are available then), not a
  direct "slower arc ⇒ higher cost" effect. With 5 vehicles able to
  redistribute the incremental congestion, the net aggregate lands close to
  flat.
- **Frozen-W (a real, congestion-aware trained policy) sits in between**:
  `distance_cost` (the component most directly exposed to congestion) moved
  in the predicted direction (171.560→171.628, +0.04%), but `delay_cost` and
  `earliness_cost` both fell slightly, netting a small overall decrease. A
  real policy *can* route around newly-congested arcs, further damping the
  direct effect relative to the single-vehicle case.

Not reverted: the invariant (order-independence + minimum depth), the
structural node-coverage fix, and the multiplier-distribution check — the
three things this ticket is actually accountable for — all landed exactly as
required. The softer cost-direction prediction holds cleanly in the
configuration most exposed to it (single vehicle) and is small/mixed
elsewhere for a mechanistically understood reason (a congestion-insensitive,
degenerate `W=0` tie-break plus multi-vehicle redistribution slack).

### Self-golden diff

Predicted: "Full divergence on every seed, in all three blocks, including
frozen-W." Measured (via `capture_self_golden.py --check` and
`tests/test_self_golden.py`, isolated worktree):

- **Training: all 5 seeds diverge** (1000-1004), as expected — training is
  cumulative (`final_w` changes from the very first affected episode, seed
  1000, and every later training/eval decision inherits it).
- **Evaluation and frozen-W eval diverge on exactly the same 6 of 10 seeds**:
  100001, 100003, 100004, 100005, 100007, 100009. The other 4 (100000,
  100002, 100006, 100008) are **bit-identical in both blocks** — their
  episodes never happen to sample a previously wrong-depth or
  missed-entirely arc. That the affected-seed set is identical between
  `evaluation` (trained W) and `frozen_w_eval` (a literal, never-recomputed
  W) is itself evidence the divergence source is purely the congestion
  draws, not an artifact of how differently the trained W happens to
  respond per seed.
- `worst relative deviation: 1.000e+00` (a `delay_clients` count flips
  between zero and nonzero on the first training seed).

So: not literally "every seed" — training's cascading effect makes all 5
diverge, but evaluation/frozen-W's divergence is seed-specific, tied to
whether that seed's realized trajectory ever touches an affected arc.
Recorded per spec decision 10 as "explained to a mechanism," original
prediction preserved above unedited.

### Full suite (isolated worktree: ticket-01 baseline + this ticket's diff only)

`ruff check` / `ruff format --check` / `mypy` on the two touched files: clean.
`mypy .` on the whole tree: 60 pre-existing errors in 17 unrelated files,
none in `congestion/generator.py` or its test — a pre-existing baseline
condition at 55f32aa, not introduced here.

`pytest -q` (golden-marked tests excluded by the project's own default
`-m 'not golden'`, per spec decisions 11/12 — no real-dataset run owed by any
ticket): **3018 passed, 6 failed**. All 6 failures are the self-golden
bit-exact gate's own predicted breakage (`test_self_golden.py`'s five
assertions plus `test_world_cache_self_golden.py`'s cache-hit reproduction) —
accepted per the section above, not a regression.

**One more failure outside that set, deselected and explained rather than
silently ignored:**
`test_final_test_dedup.py::test_deduplicated_metrics_match_the_legacy_mean_within_one_ulp`
(`simulation-performance` effort, ticket 02 — a different, closed effort).
That test compares `Trainer.final_test`'s single canonical episode against a
manually-recomputed "legacy" mean of 7 bit-identical repeats of the *same*
seed, asserting the sum-then-divide float rounding never exceeds 1 ULP. Since
all 7 repeats are deterministic replays of one seed, the "legacy mean" is
literally `(7×X)/7` computed by naive summation — whether that lands within 1
ULP of `X` is a coincidence of `X`'s exact bit pattern, unrelated to
accumulation order. This ticket's fix changes the real simulated
`distance_cost` for seed 101 (different arcs congested ⇒ different velocity
sampled ⇒ different `X`), and the new `X`'s bit pattern happens to push one
(seed, metric) cell from 1 ULP to 2 ULPs off — reproduced deterministically
twice, not flaky. Confirmed by reverting to the pre-fix generator in the same
worktree: the test passes there. This is real, expected collateral from a
correctness fix changing genuine simulated values, landing on an unrelated
ticket's razor-thin numerical margin — not a bug in this fix, and not this
ticket's test to change (out of scope, different effort, already closed).
Left untouched.

### Code review

`/code-review` (Standards + Spec axes, parallel). Spec axis: both checklist
items verified satisfied, no scope creep, no missing/wrong requirements.
Standards axis: no hard violations; one duplicated-logic-shape smell flagged
and accepted as an intentional independent test oracle (the test's reference
BFS necessarily mirrors production's shape). Both axes independently flagged
the same soft gap: the docstrings cited "ADR-0001 change log" for this fix
without a matching entry existing yet, and the module docstring lacked the
parallel top-level summary tickets 07/12 got. Fixed: added
"Addendum (2026-07-30, ticket 08)" to `docs/adr/0001-characterize-then-evolve-refactor.md`
and a matching module-docstring paragraph in `generator.py`.
