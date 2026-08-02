# 14 — The Policy adopts the baseline's action set (reverses ADR-0007)

**What to build:** `TransformerMonteCarloPolicy` stops sweeping every pending
Client and takes the **identical** candidate set the linear baseline takes,
through ticket 13's shared module. `m + 2` in training **and** in evaluation.
Then measure the 2×2 that separates *candidate set* from *ranking rule* — the
attribution this effort has never had.

**Blocked by:** 13

**Status:** resolved

## The evidence that reverses ADR-0007

ADR-0007 retired `_select_vehicle_possible_actions` as "hand-engineered ranking
built for the linear baseline's small candidate pool", keeping only feasibility.
Ticket 01's own sweep, read after ticket 08's diagnosis, says that was wrong:

| `test_action_count` | budget 100 | budget 500 | budget 2000 |
|---|---|---|---|
| 2  | 3458.4 | 3564.2 | 3564.2 |
| 40 | **3384.8 (winner)** | 3523.6 | 3523.6 |

Measured directly on the 50 `evaluation_seeds`
(`results/baseline_null_50.py`), the action count is worth **12.68%** to the
linear baseline: `m+40` **2168.39** against `m+2` **2483.24**, cheaper on 36/50
seeds, Wilcoxon **p = 8.24e-05**. On `test_seeds` the same axis shows 2.1%
(3458.4 → 3384.8). The *effect* reproduces on both seed sets; its *magnitude*
does not transfer, which is worth knowing before any number here is used as a
target.

**The mechanism, stated so it is not mistaken for a coincidence:** `c(s, a)` can
send a vehicle far across the map to a Client about to breach its window —
myopically correct, globally ruinous. Restricting to the `k` nearest is a
regularizer against long myopic hauls, and the neural Policy threw it away by
decree. It is also, restated, ticket 08's own "the likely cause is the candidate
count", now with the linear side's numbers attached.

**And the second reason, which is fairness rather than performance.** spec.md's
decision-1 amendment handed the four projected costs to the network on the
explicit ground that the baseline's seven state-action features already carry
them, so sharing them *levels* the input sets rather than tilting them. The
action set is the last un-levelled axis. While it differs, a Gate B result
cannot be attributed to the approximator at all — which is the effort's entire
brief.

### A hypothesis this ticket carried for one afternoon, and lost

`MonteCarloPolicy._create_W` is `np.zeros(19)`, so it was argued that `W = 0`
must be "go to the nearest allowed Client" — every candidate scores 0, `argmin`
returns index 0, `_closest_allowed_clients` orders nearest-first — and that
`test_action_count = 2` scoring 3458.4 against cost-greedy-over-151's 3811.3
therefore showed most of the gap was the candidate heuristic rather than the
learned weights.

**Measured: `W = 0` @ `m+2` scores 30 791.43.** Branch 3 runs
`possible_actions = list(set(possible_actions))` *after* the nearest-first sort,
and node ids are arbitrary ints, so the dedup returns hash-table order. `W = 0`
picks an **arbitrary** feasible Client. The quirk ADR-0001 preserves eats the
tie-break, and the baseline has no cheap myopic null — the same trap ticket 08
fell into when it measured its own null at 81 701.

Two things follow, and neither weakens the reversal above:

1. **The `test_seeds` comparison confounds two changes.** baseline@`m+2` versus
   cost-greedy@151 varies the action set *and* the ranking rule at once, and
   with two candidates `W` still decides which of the two — 30 791 says it
   decides a great deal. "Two candidates, nothing left to decide" was
   hand-waving.
2. **The decomposition cannot be had before this ticket lands.** Which is
   precisely what the 2×2 below is for; there was no shortcut to it.

## What is adopted

`_sweep` calls ticket 13's `select_vehicle_possible_actions` and takes its
argmin over **that list only**. All three branches, the delayed-Client
classifier, the `350`/`310` literals, the `list(set(...))` dedup — identical, or
the word means nothing. Importing it by halves would be worse than not
importing it.

**`m + 2` in training and `m + 2` in evaluation.** The count is not `40`:
`episode.py` hardcodes `number_actions = number_vehicles + 2` and passes it to
*both* `number_actions_train` and `number_actions_test`, while `trainer.py:404`
evaluates at `vehicle_count + action_count` over the swept counts. The baseline
therefore **trains on `m+2` and evaluates on `m+40`**; this Policy does not
inherit that mismatch — the LSMC accumulator (ticket 16) is fitted and applied
on the same action distribution, with no extrapolation onto candidates it never
observed.

**Its cost, recorded rather than glossed:** on the only evidence available,
`m+2` at evaluation costs the linear baseline 2.2% (3458.4 against 3384.8).
This Policy is choosing a configuration that is measurably worse *for the
baseline*, and buying zero extrapolation gap with it. That is a trade, not an
improvement.

**Its benefit, also measured:** ticket 08's standing worry was "`decide` takes
an argmin over ~151 candidates while `learn` attaches a gradient to exactly one
of them — the other 150 are unconstrained extrapolation". Under `m+2` that
becomes roughly **1 observed in ~10** (the `k` nearest after the other
vehicles' targets are excluded, plus the depot, plus up to two delayed
Clients). An order of magnitude better-posed.

## What retires, and what is fixed for free

- **`_is_retired` retires.** Branch 1 offers a parked vehicle past `tau > 350`
  only the depot, so it can no longer claim a Client it cannot serve.
- **`_depot_is_feasible` retires.** Its condition 2 *is* branch 3's depot gate
  (`average_minutes(position, depot) + tau > end_of_horizon`) and its condition
  1 is branch 1. Note the deliberate consequence: the config-clock version this
  effort built is replaced by the literal `350` it was introduced to avoid.
  That is what adopting the identical set means, and it is why this is a
  reversal of ADR-0007 rather than an amendment to it.
- **The `claimed_mask` defect closes.** Ticket 08 left it open: "`claimed_mask`
  is rebuilt per epoch so it ignores the other vehicles' in-flight commitments
  (`MonteCarloPolicy` seeds `forbidden_actions` from `self.action` for *every*
  other vehicle)". Adopting the shared module adopts `self.action`, so the fix
  arrives as a side effect of the reversal rather than as its own change.

`is_depot` / `DEPOT_WARM_START_PENALTY` need a decision inside this ticket: the
depot now enters the candidate list only where the baseline admits it, so the
warm-start penalty that kept it from winning at epoch 1 may be unnecessary.
Decide by measurement on `evaluation_seeds`, not by argument.

## A purity note, so it is not read as a leak

The action set needs `features.delayed_clients` and
`features.closest_client_counts`, so this Policy's pipeline now calls
`FeatureExtractor.state_features(state)` — which computes all 12 of the
baseline's engineered general-state features. **Only those two are read, and
neither reaches `QHead`.** spec.md decision 1's "the state features stay out"
governs what enters the network, and it is untouched: `tokenize`'s signature is
the same five arguments and ADR-0006's structural test still pins it.
Computing a feature is not feeding it to `Q`.

## The measurement this ticket exists to produce

Zero training. All on `evaluation_seeds`, never `test_seeds`.

| ranking ↓ / candidate set → | baseline's set (`m+2`) | all ~151 |
|---|---|---|
| arbitrary (`W = 0` of the linear) | 30 791.4 — *degenerate, not a null* | — |
| **cost-greedy (`c(s, a)`)** | **(1)** ← the new frozen null | 3693.2 |
| trained `W` of the linear | 2168.4 (`m+40`) / 2483.2 (`m+2`) | — |

**(1) is the one missing number, and it is the only cell that separates action
set from ranking rule** — it holds the ranking rule fixed at cost-greedy and
moves only the candidate set, against the 3693.2 already measured. Everything
else in the table varies two things at once.

The whole table is on `evaluation_seeds`, including the 3693.2 and the two
linear cells, so it is internally comparable. Do not mix in the `test_seeds`
numbers (3811.3 / 3384.8 / 3458.4): a `W` *selected* on `evaluation_seeds`
reads 36% better there, while an arithmetic rule reads 3% better, so the two
seed sets are not interchangeable for this comparison.

**A branch declared before the number exists, so it cannot be rationalised
afterwards:** if (1) comes in below **2168.4** — the linear baseline's best cell
*on this same seed set* — then the myopic base beats a tuned 19-weight VFA at
**zero training**, and the effort's question becomes "and on top of that, does
learning add anything?". Say so if it happens; do not re-frame it as expected.
It is a claim about `evaluation_seeds` until ticket 09 measures it on the
verdict set, and given the ×1.56 / ×1.03 spread above it may not survive the
trip.

## Work

- [x] `_sweep` takes its candidates from ticket 13's module; `_replay_joint_q`
      unchanged (it scores the action taken, it does not sweep).
- [x] Retire `_is_retired` and `_depot_is_feasible`, with their docstrings'
      reasoning moved into this ticket's record rather than deleted.
- [x] Measure (1) and (2) on the 50 `evaluation_seeds`. Freeze (2) as the null.
- [x] Decide `is_depot` / `DEPOT_WARM_START_PENALTY` on that measurement.
- [x] Develop on the mini fixture; measure on the real dataset.

## Acceptance

- [x] The 2×2 filled in and recorded in this ticket's Comments.
- [x] The new null frozen and named, per spec.md's obligation: *report the null
      alongside the trained number, always, and name which one produced it.*
- [x] **ADR-0011** drafted (written in ticket 11): *the action set is shared,
      not owned* — reversing ADR-0007 with the table above as its evidence.
- [x] Predicted self-golden diff: **zero** (ticket 13 already paid that cost).

## Comments

**Correction to "A purity note" above:** it claims the action set needs both
`features.delayed_clients` and `features.closest_client_counts`. Checked
against the landed `action_set.py` (ticket 13): `select_vehicle_possible_actions`
reads only `features.delayed_clients`; `closest_client_counts` belongs to
`FeatureExtractor.candidate_features`/`action_features`, which this Policy
never calls. The claim in the paragraph above is wrong; `transformer_policy.py`'s
own docstring and this Comments entry are the correct record.

Landed as planned: `TransformerMonteCarloPolicy._sweep` now calls ticket 13's
`select_vehicle_possible_actions` at `m + 2` (`self._number_actions`), in both
`decide()` and `decide_train()`, with a persistent `self.action` threaded
through as `current_action` — mutated in place vehicle by vehicle, mirroring
`MonteCarloPolicy.action`. `_is_retired` and `_depot_is_feasible` are deleted;
their reasoning (the `350` cutoff replacing the `horizon_start_minute` one,
the shift-breach depot append, the no-Clients fallback) moved into
`transformer_policy.py`'s module docstring ("ADR-0011" section) and into
`docs/adr/0011-the-action-set-is-shared-not-owned.md`. Two constraints stay
Policy-side because `action_set.py` does not know about either: no double
booking within one pass (now arising from `current_action` exclusion rather
than a locally rebuilt mask) and no self-node (ADR-0008, B20), both applied
after the shared function returns, with a depot fallback if nothing survives.
`TransformerMonteCarloPolicy` gained a `FeatureExtractor` (`state_features`
only; only `delayed_clients` reaches `select_vehicle_possible_actions`, none
of the twelve general-state features reach `QHead`).

**The 2×2, all on the 50 `evaluation_seeds`, zero training:**

| ranking ↓ / candidate set → | baseline's set (`m+2`) | all ~151 |
|---|---|---|
| arbitrary (`W = 0` of the linear) | 30 791.4 — *degenerate, not a null* | — |
| **cost-greedy (`c(s, a)`)** | **3365.09** ← the frozen null | 3693.2 |
| trained `W` of the linear | 2168.4 (`m+40`) / 2483.2 (`m+2`) | — |

(`.scratch/neural-policy/results/action_set_m2_50.py`, log and JSON alongside
it.) The missing cell (cost-greedy @ `m+2`) is 8.9% better than cost-greedy at
~151 — restricting to the `k` nearest helps the myopic dispatcher too, not
only a linear model that cannot see far. **The pre-declared branch did not
fire**: 3365.09 is above 2168.4 (the linear baseline's best cell on this seed
set), so the myopic base does not beat a tuned 19-weight VFA at zero training
here — recorded as measured, not re-framed.

**`is_depot` / `DEPOT_WARM_START_PENALTY`: kept.** Measured with vs. without
the penalty at `m+2` over the same 50 seeds: 3365.09 (penalty=1.0, as shipped)
against 3364.52 (penalty=0.0) — **-0.02%, 1/50 seeds differ, Wilcoxon
p = 0.317**. A null result confirming the structural prediction (the depot
rarely enters the `m+2` shortlist any more): the penalty stays, since it costs
nothing where it no longer matters and is still correct in the shift-breach
window where it does.

**Verification:** `mypy`/`ruff` clean on the changed files; full suite
(`uv run pytest`) 4274 passed, 3 `golden`-marked deselected; `tests/test_self_golden.py`
run explicitly, 6/6 passed (predicted-zero self-golden diff confirmed —
this ticket never touches `monte_carlo.py` or `action_set.py`); `-m neural`,
114 passed. `tests/unit/test_transformer_policy.py`'s
`TestHomeIsAnExitNotADestination` rewritten around the `350` literal
(replacing the retired `horizon_start_minute` gate) and the shared function's
own branch tests (`test_action_set.py`, ticket 13) rather than the deleted
private methods; new `TestCurrentActionIsSelfAction` pins that
`current_action` is `self.action` itself, not a copy, and that the stale
in-flight value of a not-yet-processed vehicle reaches an earlier vehicle's
candidate computation — the claimed_mask defect closing, demonstrated rather
than asserted by outcome.
