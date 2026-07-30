# 05 — Endgame: no crash, no double booking

**What to build:** Two independent defects in the endgame branches of
`_select_vehicle_possible_actions`. Closes B5 and B11.

**Blocked by:** 01

**Status:** resolved

## B5 — `min()` on an empty sequence

`policies/monte_carlo.py:425-437`. Two uncoordinated thresholds leave a 40-minute
window uncovered: `_select_vehicle_possible_actions` diverts to the depot at
`tau > 350` (line 298) while `_classify_shortest_distance_clients` discards
vehicles at `tau > 310` (line 429). With exactly one Client pending, every
vehicle reading `position == depot`, and `310 < tau <= 350`, `distances` is
empty and `min()` raises `ValueError`. Reproduced by the review at tau 320 and
340.

The condition is easier to hit than it looks, because `position == depot` is
also true for vehicles that merely crossed the depot — this is ticket 04's root
cause showing up again.

- [x] **Fallback only**: when `distances` is empty, fall back to the depot,
      exactly as the two-Client branch already does via `heapq.nsmallest`.
- [x] **Do not unify 310 and 350.** The review proposes it; this effort's
      criterion excludes it — unifying thresholds is re-tuning, not fixing a
      miswritten predicate. Document the disagreement where the literals live,
      alongside the existing quirk note at `monte_carlo.py:62-64`.
- [x] Invariant: the Policy returns a legal action for every vehicle, at every
      tau, under every valid config. Property-based, sweeping tau across the
      310/350 window with small pending-Client counts and small
      `min_number_clients` — the regime the review flags as most exposed.

## B11 — the same Client assigned to two vehicles

`policies/monte_carlo.py:302-308`. The endgame branch
(`len(clients_not_visited) < 3`) does not filter `forbidden_actions`, unlike the
normal branch at line 311. Reproduced: `action=[7, 7]` with two vehicles,
`action=[7, 0, 7]` with three; **734 transitions over 60 episodes**.

- [x] Filter `forbidden_actions` in the endgame branch, matching the normal
      branch.
- [x] Invariant: two vehicles never receive the same non-depot Client in one
      decision.
- [x] Note the latent second-order effect and **leave it latent**: the losing
      vehicle arriving at an already-served Client is the only transition path
      that skips `commit_transition()` (`model.py:505-514`), which would put its
      cost inside `distance_cost` but outside `total_cost`. The review
      instrumented 84 episodes and reached it **zero times** — the loser always
      gets rerouted first. Fixing B11 removes the entry condition; do not also
      restructure the commit path on the strength of an unreached branch.

## Predicted self-golden diff

**B5 contributes zero.** The crash requires exactly one pending Client with all
vehicles reading `position == depot` in a 40-minute window; the 15 capture
episodes complete without raising, so the branch is not being taken. If ticket
01's 60-seed bench *did* trip it, this fix moved forward and that is recorded
there instead.

**B11 contributes a real diff.** At 734 duplicate transitions per 60 episodes it
fires often enough to expect hits among the 15 capture seeds. On affected seeds
the losing vehicle now receives a different action, so the trajectory changes:
expect `state_count`, `tau` and `distance_cost` to move, direction unpredictable
per seed. Aggregate direction over the 60-seed bench should be **cost
non-increasing** — the fix recovers fleet capacity that was being wasted on a
Client another vehicle was already serving — but a single seed may go either way
and that alone is not a contradiction.

**Seeds with zero duplicate assignments must be bit-identical in all three
blocks.** Ticket 01 reports the per-seed counts; list those seeds here before
running. That is this ticket's falsifiable claim.

## Evidence required

Duplicate-assignment counter at zero over the 60-seed bench (was 734 per 60
episodes). The crash invariant green across the swept tau window. The
untouched-seed bit-identity list.

## Comments

### Resolution (2026-07-30)

Landed as `8d072ba` (fix + tests) and `e8e9a7c` (self-golden re-capture,
below). Both files: `src/stdvrp/policies/monte_carlo.py`,
`tests/unit/test_monte_carlo_policy.py`.

**B5.** `_classify_shortest_distance_clients`'s one-Client branch now skips
the `min(distances)` call when `distances` is empty (every vehicle read
`position == depot` with `tau_episode > 310`), leaving `shortest_distance_clients`
without an entry for any vehicle — the caller's existing `if
shortest_distance_clients[vehicle]: ... else: possible_actions.append(depot)`
already handles that, unchanged. The 310/350 disagreement is now documented at
both literals (`monte_carlo.py`'s `_select_vehicle_possible_actions` and
`_classify_shortest_distance_clients`), not only in the module docstring's
existing quirk note. Literals themselves untouched.

**B11.** The endgame branch (`elif len(clients_not_visited) < 3`) now filters
`shortest_distance_clients[vehicle]` against `forbidden_actions` before
appending, matching the normal (`>= 3`) branch just below it.

**Invariants** (`tests/unit/test_monte_carlo_policy.py::TestEndgameInvariants`):
two Hypothesis properties, `max_examples=300`, sweeping `tau` in `[300, 360]`
(crossing both 310 and 350), 0–2 pending Clients, 1–4 vehicles, and a random
depot/non-depot position per vehicle. `test_b5_...` asserts
`_select_vehicle_possible_actions` never raises and always returns a non-empty
list; `test_b11_...` asserts `decide()`'s two-or-more-vehicle output never
repeats a non-depot Client. **Confirmed red before the fix** (temporarily
stashed the production diff and re-ran): B5's property raised
`ValueError: min() arg is an empty sequence` at `tau=311, remaining=1,
vehicle_count=1, at_depot=[True]`; B11's property failed with `actions=[1, 1]`
at `tau=300, remaining=2, vehicle_count=2`. Both green after. The
pre-vectorization differential oracle `reference_possible_actions` (ticket 07,
simulation-performance) is updated in lockstep with the same filter, so it
keeps pinning ticket 07's vectorization rather than resurrecting the
pre-ticket-05 duplicate-booking bug in its own `elif` branch.

**Measured** (`scripts/measurement_bench.py`, isolated `git worktree` at this
commit's parent, so the numbers are not contaminated by other tickets'
concurrent in-progress edits sharing this working tree):

- 60-seed default-fleet bench: `B11_duplicate_assignment` `episodes=60/60
  transitions=308` → `episodes=0/60 transitions=0`. Mean total cost
  `512.194224` → `505.760321` (**non-increasing**, as predicted — the fix
  recovers fleet capacity previously wasted on a Client another vehicle was
  already serving). `B5`'s crash counter: `0/60` in both (never reproduced on
  this bench per ticket 01; the Hypothesis property above is the property-based
  evidence for the crash invariant, not the 60-seed bench).
- `--capture-seeds` (frozen-W protocol, 5 train + 10 eval): duplicate
  transitions `66` → `0` across the 15 seeds.

**The "untouched seed 1002" claim, corrected.** Ticket 01 reported seed 1002 as
the only one of the 15 capture seeds with zero of its own duplicate-assignment
transitions, and this ticket predicted it would stay bit-identical. It does
not, in the `training`/`evaluation` self-golden blocks: 1002 is the *third*
`TRAIN_SEED`, and the training block carries `W` sequentially across
`1000 → 1001 → 1002 → …` — seeds 1000 and 1001 both *do* have duplicate
transitions, so by the time 1002 runs it is already deciding against a `W`
the fix already moved. Verified: `capture_self_golden.py --check` (isolated
worktree, before re-capturing) shows seed 1002 moving on `w[*]` and every
reported metric in the `training` block. The **mechanism**, not an
unexplained deviation (spec.md decision 10's second outcome): the "bit-identical"
claim only ever had a chance of holding in the frozen-W block, which is
immune to this cascade by construction (fixed literal `W`, no learning
feedback) — but all 10 `EVAL_SEEDS` have at least one duplicate-assignment
transition each (ticket 01's own table), so this 15-seed capture contains no
seed that can actually test the claim. Recorded here, unedited, per the
three-outcome rule; the original prediction stands above as written.

**Self-golden.** Landed a second, isolated commit (`e8e9a7c`) re-capturing
`tests/fixtures/self_golden/mini_fixture.json` — neither this ticket's landing
nor ticket 06's (which landed concurrently on top of it, same working tree)
had re-captured, so the gate was red at HEAD for both fixes at once. Verified
green (`tests/test_self_golden.py`, `tests/test_world_cache_self_golden.py`,
7/7) in an isolated worktree before landing.

**Concurrency note.** This ticket's branch had four other sessions landing
commits on it while this ticket was in progress (tickets 06 and two ticket-09
follow-ups landed *during* this implementation). Every commit above was built
with `git read-tree`/`update-index --cacheinfo`/`commit-tree`/`update-ref`
against the then-current branch tip rather than `git add`/`git commit` on the
shared working tree or index, per [[git-pathspec-commit-stages-worktree]] —
each commit's diff was verified via `git diff-tree --stat` against its exact
intended file set before `update-ref`, and re-verified green in a disposable
`git worktree` (not the shared one) before and after. No working-tree file
belonging to another session's in-progress ticket (07's `congestion/generator.py`
/`feature_extraction.py`, at time of writing) was read into any commit here.
