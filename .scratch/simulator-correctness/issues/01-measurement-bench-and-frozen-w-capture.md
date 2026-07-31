# 01 — Measurement bench and the frozen-W capture block

**What to build:** The instrumentation every other ticket depends on, plus a
reproduction of the review's headline numbers at HEAD. No production behavior
changes.

The review's measurements exist today as ad-hoc `Model` subclasses pasted into a
report. This ticket turns them into a command, and adds the one capture block
that makes per-ticket diff attribution possible at all.

**Blocked by:** —

**Status:** resolved

- [x] **Measurement bench** in `scripts/` — runs a fixed 60-seed set on
      `tests/fixtures/chengdu_mini` and reports, per seed and aggregated: mean
      total cost and its 4 components, km driven, final tau, decision count,
      **unserved-Client count split by in-window / overdue**, and a violation
      counter per invariant in the spec's catalogue. Reusable: tickets 02–09
      call it for their before/after evidence, so its output must be diffable
      text, not prose.
- [x] **Frozen-W evaluation block** added to `scripts/capture_self_golden.py`:
      the same `EVAL_SEEDS`, run with a fixed literal W committed in the
      protocol (use the current `final_w`), written to the capture under a new
      key. The two existing blocks are **not touched**.
      `tests/test_self_golden.py` asserts the new block float-for-float like the
      others.
- [x] **Diff reporter**: extend `--check` so it reports *what* changed per seed
      per metric, not only the worst relative deviation. This is the tool the
      three-outcome rule is enforced with — a bare "worst rtol 3e-2" cannot be
      checked against a prediction.
- [x] **Reproduce the review at HEAD.** Confirm or refute, with numbers:
      - B1b: ~0.4 mid-arc teleports per episode; 24 of 60 episodes affected
      - B3: 15 of 60 single-vehicle episodes leave ≥1 in-window Client unpriced
        (116 Clients total)
      - B11: 734 duplicate-assignment transitions over 60 episodes
      - B17: the book reaches 116 arcs, all 116 expired
      Any finding that does not reproduce gets its ticket re-scoped or dropped
      **before** it costs work.
- [x] **Size the blast radius of tickets 03 and 04 on the capture seeds.** All
      15 capture episodes terminate all-back between tau 396 and 502 — inside
      both B3's and B1a's regime. Report per capture seed how many Clients are
      left unserved and how many teleports fire, so tickets 03 and 04 can write
      honest predictions instead of guesses.

## Predicted self-golden diff

**Exactly zero on the two existing blocks.** This ticket adds a key to the
capture and changes no production code. The new frozen-W block has no prior
value to diff against.

If the existing blocks move, something in the bench leaked into production code
— revert and find it.

## Evidence required

The four reproduction numbers above, plus the per-capture-seed unserved and
teleport counts. Recorded in Comments.

## Contingency

If the bench trips B5's `min()` crash on any of the 60 seeds, pull B5's
two-line fallback forward into this ticket and note it here; ticket 05 then
carries only B11.

## Comments

### Resolution (2026-07-28)

All five deliverables landed. Files:

- `scripts/measurement_bench.py` — the reusable 60-seed bench, the
  `--capture-seeds` blast-radius mode, and two structural checks (B9, B16)
  independent of any seed.
- `scripts/capture_self_golden.py` — `FROZEN_W` constant + `run_frozen_w_eval`
  (new `"frozen_w_eval"` capture key) + `diff_report`/`_diff_line` (per-seed,
  per-metric `--check` reporting).
- `tests/test_self_golden.py` — `test_frozen_w_eval_metrics_are_bit_exact`.
- `tests/fixtures/self_golden/mini_fixture.json` — re-captured; `final_w`,
  `training`, `evaluation` byte-identical to the prior commit (verified below);
  `frozen_w_eval` is the only new content.

Full pytest suite: 3025 passed. `ruff check .` / `ruff format --check .` /
`mypy` all clean.

### Measurement bench (deliverable 1)

`scripts/measurement_bench.py`. Reports, per seed and aggregated, over the
fixed `BENCH_SEEDS = range(60)`: the four cost components + total, km driven
(equal to `distance_cost` under the shipped `distance_rate = 1` — verified
identical on every row), final tau, decision count, the unserved-Client split
(in-window / overdue), and a counter for every invariant in the spec's
catalogue that is directly countable pre-fix. Output is tab-separated,
seed-sorted plain text — diffable with `diff` between a "before" and "after"
run, per the ticket's brief. The four runs cited below are saved at
`.scratch/simulator-correctness/bench-output/*.txt`.

Decision-stable by construction — no training happens inside the bench:
`--w zero` (default, the zero vector; matches how the review's own ad hoc
probes ran and needs no training dependency) or `--w frozen` (the literal
`FROZEN_W`, see below). `--vehicle-count` controls fleet size: default `None`
uses the demand-generated fleet (~5 vehicles on this fixture, needed for B11 —
duplicates require ≥2 vehicles); `--vehicle-count 1` forces the single-vehicle
configuration the review used for B1b's and B3's headline numbers.

Ten counters cover eleven of the twelve catalogue rows: nine findings
(B5, B7, B9, B10, B11, B14, B15, B16, B17) each get their own distinct
counter, and one counter (`mid_arc_park_violations`) covers **both** B1a's row
("no vehicle becomes `PARKED` while travelling") and B1b's ("a vehicle's
recorded node never changes without charged distance") — not two independent
measurements folded together for convenience, but because the review itself
found these are the same event at HEAD and never observed apart ("comparten
disparador y en las mediciones nunca aparecen por separado", 71/71 mid-arc
retirements traced to the one depot-idle trigger). Ticket 04 still owes each
row its own property test once the State carries the fact this bench cannot
observe directly (standing at a node vs. crossing it).

The twelfth row — B3, "the termination charge is a function of `due` and
config only, never of `tau`" — is reported as `NOT_DIRECTLY_COUNTABLE`: there
is only one clock to evaluate the charge at before the fix exists, so nothing
can disagree with itself yet. The bench points at the unserved-split line
instead (exactly the demand B3 prices at zero); ticket 03 writes the real
property test once a second clock exists to compare against.

**B17's counter is a terminal snapshot, not a continuous check.** The
catalogue row reads "the book holds no expired entries *after a clock
advance*" — every advance, not only the last one. `congestion_book_expired_at_end`
samples the book once, immediately before `release()` clears it at Episode
end. That is real evidence (and exactly what the ticket's bullet 4 asks for:
the arcs-written / arcs-expired count at termination) but it is narrower than
the catalogue text; ticket 07 still needs a per-advance property once it
composes and purges.

Reproduce: `uv run python scripts/measurement_bench.py [--vehicle-count N]
[--w zero|frozen] [--capture-seeds] [--output PATH]`.

### Frozen-W capture block (deliverable 2)

`FROZEN_W` is a literal 19-float snapshot of this protocol's own `final_w`, as
committed 2026-07-28 — not a live read of the capture file, so it stays fixed
even when the training block above it is re-captured for an unrelated reason.
`run_frozen_w_eval` runs `EVAL_SEEDS` greedily against it and writes the result
under a new `"frozen_w_eval"` key.

**Verified: exactly zero diff on the two existing blocks**, matching the
prediction exactly. Byte-for-byte comparison of the pre-ticket capture against
the post-ticket one:

```
keys before: ['meta', 'final_w', 'training', 'evaluation']
keys after : ['meta', 'final_w', 'training', 'evaluation', 'frozen_w_eval']
final_w identical: True
training identical: True
evaluation identical: True
meta identical: True
frozen_w_eval entries: 10
```

`tests/test_self_golden.py::test_frozen_w_eval_metrics_are_bit_exact` asserts
the new block float-for-float; all 6 tests in the file pass, including the
always-on environment-activity check.

### Diff reporter (deliverable 3)

`capture_self_golden.diff_report` walks `final_w`, then `training` /
`evaluation` / `frozen_w_eval` entry-by-entry (seed by seed), reporting every
`w[i]` and every `metrics.<name>` that moved — not only the single worst
relative deviation `worst_deviation` already reported. `check()` prints both.
Self-check against the freshly captured golden:

```
worst relative deviation: 0.000e+00 at (identical)
tolerance:                1.000e-09
per-seed per-metric diff: (identical - nothing moved)
OK: within tolerance
```

This is the tool tickets 02–10 use to verify their "Predicted self-golden
diff" sections against what actually moved (spec.md decision 10's
three-outcome rule).

### Reproducing the review at HEAD (deliverable 4)

`--vehicle-count 1` for B1b/B3 (the review's own single-vehicle
reproduction); default (generated) fleet for B11, since duplicate assignment
needs ≥2 vehicles.

| # | Review's claim | Measured at HEAD | Verdict |
|---|---|---|---|
| B1b | ~0.4 mid-arc teleports/episode; 24 of 60 episodes | **24/60 episodes, 0.400/episode** (`--vehicle-count 1`) | **Confirmed, near-exact** |
| B3 | 15 of 60 single-vehicle episodes leave ≥1 in-window Client unpriced (116 Clients) | **14/60 episodes, 109 Clients** (`--vehicle-count 1`) | **Confirmed, close** (within ~7%) |
| B11 | 734 duplicate-assignment transitions over 60 episodes | **308** transitions (`--w zero`, default fleet) / **362** (`--w frozen`) — present in **60/60** episodes either way | **Confirmed as real and frequent; magnitude does not reproduce** (~2× low, either W) |
| B17 | the book reaches 116 arcs, all 116 expired | over 60 episodes (`--vehicle-count 1`): **6277 arcs written, 3158 (50.3%) still expired-but-present at termination**; 14/60 episodes individually reach a fully-expired book (max book size seen: 112) | **Confirmed qualitatively** (book never purges mid-episode); **the literal "116/116" does not reproduce** |

On the two that did not reproduce exactly — before this cost real work on
tickets 05/07, per the ticket's own contingency clause:

- **B11.** The defect (the endgame branch does not filter `forbidden_actions`)
  is unambiguously present and fires in **every one** of the 60 default-fleet
  episodes tested, under both `--w zero` and `--w frozen`. **Ticket 05
  proceeds as scoped** — this is real and frequent, just not exactly 734
  transitions under this bench's parameters. The review's ad hoc script (not
  preserved in the repo) evidently ran under different parameters (a
  different W, or a different generated-fleet distribution); the exact count
  was never reproducible from the report text alone.
- **B17.** The invariant violation itself — congestion events sit in the book
  past their expiry until `release()`, never purged mid-episode — is
  unambiguous and appears in every configuration tested. The specific "116
  arcs, all expired" reading most likely came from one particular probe run
  in the review's own ad hoc script (a longer-running or differently-seeded
  episode where the book had stabilized); the committed 60-seed set does not
  reproduce that exact reading under either fleet configuration in this
  bench. **Ticket 07 proceeds as scoped** — the qualitative defect is what
  the fix targets, not the specific figure.

`B9` and `B16` (structural, independent of any seed — `check_b9_spread_depths`
/ `check_b16_cadence`):

- **B9** confirmed on both symptoms the review names: **38/635 (6.0%)**
  reached nodes get a non-minimal depth, and **40/675 (5.9%)** of the nodes
  genuinely within `max_depth` of a trigger are never congested at all (the
  review's own "~15%" framing) — order-of-magnitude confirmed, lower on the
  mini fixture's smaller graph than the review's figure (which the review's
  Method section says was cross-checked against the real Chengdu network;
  topology density plausibly explains the gap).
- **B16** confirmed exactly as scoped: fires at the intended cadence for
  **every duration the repo ships, tests, or has captured** (30, 45, 60, 90,
  120, 150, 180, 240 all match the integer-arithmetic reference) and
  **breaks for 50, 70, 200** (the review's own two examples plus one more) —
  e.g. duration 70 fires 2 times instead of the intended 12. Confirms ticket
  02's premise that decision epochs land on integer minutes (verified: `tau`
  at every `_congestion_epoch_due` check is `next_decision_tau`, which starts
  at an int and is incremented by an int).

`B5`'s `min()` crash: **0/60 episodes crashed**, in every configuration tested
(default fleet, `--vehicle-count 1`, both W choices). **Contingency not
triggered** — ticket 05 carries the full B5 fix as scoped, and its "if it did
trip" fallback-pull-forward clause drops out.

`B14` (money charged without its counter incrementing): **0 violations** at
the default fleet, **26/60 episodes (27 violations)** at `--vehicle-count 1` —
consistent with the review's framing: it fires specifically on unserved
Clients charged at termination, which only the single-vehicle configuration
produces enough of on this fixture.

`B15` (negative cost component): **0 violations**, as expected — the shipped
`shift_end_minute = 780 <= EMERGENCY_HORIZON`, so the trap ticket 02 closes is
not yet reachable.

### Sizing tickets 03/04/05's blast radius on the capture seeds (deliverable 5)

`uv run python scripts/measurement_bench.py --capture-seeds` replays the exact
frozen-W protocol (`TRAIN_SEEDS` 1000–1004 carrying W, `EVAL_SEEDS`
100000–100009 with the final trained W, generated fleet), instrumented the
same way.

**All 15 capture episodes terminate all-back between tau 396.54 and 501.72** —
confirms the spec's own measurement ("tau 396 and 502") exactly.

**Ticket 03 (B3): zero blast radius on the capture seeds.** All 15 episodes
end with **zero unserved Clients** — `unserved_in_window` and
`unserved_overdue` are both 0 on every one of the 15. This is a
**load-bearing correction to ticket 03's plan as written**: with nothing
unserved to reprice, ticket 03's frozen-W block prediction should not be "the
seven listed metrics stay identical while `delay_cost` / `total_cost` rise on
affected seeds" — on this capture set, the frozen-W block's predicted diff is
**exactly zero on all nine metrics, on every one of the 15 seeds**, because no
seed has any unserved demand for the new formula to reprice. The training and
final-eval blocks can still diverge (the termination charge still enters every
training Episode's Monte Carlo return in principle, and training seeds could
in principle hit the branch even though these particular ones don't at the
zero/frozen W tested here) — but the frozen-W block's "affected seeds" set is
empty on this capture, and ticket 03 should write that explicitly rather than
guess a nonzero frozen-W movement.

**Ticket 04 (B1a/B1b): 7 of 15 capture seeds have ≥1 mid-arc teleport, 8 have
zero.**

```
teleport (mid-arc-park) count per capture seed:
  1000: 1   1001: 1   1002: 0   1003: 0   1004: 0
  100000: 1  100001: 1  100002: 1  100003: 0  100004: 1
  100005: 0  100006: 0  100007: 0  100008: 0  100009: 1
```

Untouched-seed list (must stay bit-identical in all three blocks under ticket
04): **1002, 1003, 1004, 100003, 100005, 100006, 100007, 100008**. Affected:
**1000, 1001, 100000, 100001, 100002, 100004, 100009**.

**Ticket 05 (B11): 14 of 15 capture seeds have ≥1 duplicate-assignment
transition; only one has zero.**

```
duplicate-assignment transitions per capture seed:
  1000: 10  1001: 4   1002: 0   1003: 5   1004: 3
  100000: 2  100001: 5  100002: 6  100003: 1  100004: 7
  100005: 7  100006: 4  100007: 1  100008: 7  100009: 4
```

Untouched-seed list for ticket 05: **only 1002**. Every other capture seed is
expected to move once B11 is fixed.

Full per-seed tables for all four runs (default fleet, default fleet with
frozen W, `--vehicle-count 1`, `--capture-seeds`):
`.scratch/simulator-correctness/bench-output/`.

### Contingency

Not triggered — see B5 above (0/60 crashes in every configuration tested).
Ticket 05 carries the full two-part fix (fallback + `forbidden_actions`
filter) as originally scoped; ticket 05's own file is unchanged by this
finding.
