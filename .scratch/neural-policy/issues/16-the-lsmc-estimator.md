# 16 — The estimator: accumulated least squares, not per-episode SGD

**What to build:** Replace the learning rule. `W` is solved in closed form from
normal equations accumulated **across** episodes with exponential forgetting.
No learning rate, no level to walk, no per-episode batch discarded.

**Blocked by:** 15

**Status:** resolved

## The defect this fixes, which ticket 08 never named

`learn` receives one Episode's ~400 decision epochs, runs `learn_passes = 4`
shuffled passes in minibatches of 32 (~50 gradient steps), and **discards the
batch.** There is no buffer across Episodes.

Within one Episode, `U_t = Σ_{k≥t} rewards[k+1]` is a **suffix sum** — a
monotone function of `t` — and `_backward_returns` subtracts
`_already_acquired_cost`, itself monotone in `t`. The global token carries
`tau_episode` and `clients_not_visited`.

> With 595k parameters, `U_t` is fitted almost perfectly by reading the clock.
> The action carries no incremental explanatory power *within* one Episode,
> because one Episode gives exactly one action per epoch against a target that
> is deterministic given `t`. **The action→return signal exists only across
> Episodes, and the batch is thrown away every Episode.**

That reconciles every measurement in ticket 08 at once: `loss` at `1e-4` while
the policy degrades below its own null (it is fitting each Episode's suffix sum,
~50 steps at a time); why 19 weights survive the same target (they cannot
overfit one Episode, and `W` is carried across all of them); and why every
optimizer improvement helped from a bad start and hurt from a good one (each
accelerates the per-Episode overfit).

Quantified: **595k parameters against ~400 samples from one Episode** becomes
**515 parameters against ~20 000 samples from ~50 Episodes.**

The rejected `huber_delta = 0.02` deserves a note: it was rejected because
near-linear gradients "under Adam is a much larger effective step". **That
reason evaporates without Adam.** A finding rejected for optimizer-specific
reasons does not transfer to a different estimator.

## It is exactly linear in the solvable parameters

`QHead` is `linear(x) + layer2(ReLU(layer1(x)))`. Holding `layer1` and the
encoder fixed, define per candidate

```
φ(s, v, a) = [ x ; ReLU(layer1(x)) ; 1 ]        386 + 128 + 1 = 515
W          = [ linear.weight ; layer2.weight ; linear.bias + layer2.bias ]
```

and per decision epoch, summing the row actually taken:

```
Φ_t = Σ_v φ(s, v, a_v)                          the regression feature
ỹ_t = (U_t − acquired) / return_scale  −  Σ_v c(s, v, a_v)
```

so `W · Φ_t ≈ ỹ_t` is a plain linear regression. `layer2`'s bias and `linear`'s
bias collapse into one intercept, which is correct — they were never separately
identifiable.

**`Φ_t` is free.** The head already computed `x` and `ReLU(layer1(x))` while
scoring that candidate during the sweep. Accumulating it costs no
re-tokenization, no re-encode and no backward pass — the three things
`transformer_policy.py`'s docstring calls "not maximally efficient". Expect the
frozen-encoder arm to run several times faster than the 15 s/ep this effort has
been paying.

## The accumulator

```
A ← γ·A + Σ_t Φ_t Φ_tᵀ
b ← γ·b + Σ_t Φ_t · ỹ_t
W ← (A + λI)⁻¹ b                        re-solved every N episodes
```

`γ` is exponential forgetting — effective window ≈ `1/(1−γ)` Episodes — which
is what keeps the fit on the *recent* policy's returns while still spanning
many Episodes. Memory is `O(d²)` at `d = 515`, so no sample buffer is kept.

**One currency.** `c` is divided by `return_scale` like the target, so `c`, `Q`
and `y` are all the same physical quantity: accumulated episode cost. Note what
that gets for free — ticket 08's level pathology (`Q_joint` at 0.3–0.9 against
a target of 0.03, corrected by same-signed steps 10–30× the ranking signal) does
not need a fix here. It cannot arise: both sides are the same measurement, and
the solve sets the level exactly on the first solve rather than walking to it
over ~100 Episodes.

## The heavy tail (F10)

`ABORT_PENALTY = 40000`, rebated 200 per served Client, charged whenever the
clock reaches `CLOCK_CEILING = 1198`. It enters `U_t` for **every** epoch of
that Episode — ~400 targets at 10–30× normal, which in an accumulator weighs
like ~12 000 ordinary epochs. One aborted Episode would dominate a window of
fifty. Ticket 08's log has Episodes at 75 760.

**Aborted Episodes are excluded from the accumulator.** The penalty is
`40000 − 200·served`, dominated by a constant: it says the Episode ended badly,
not which candidate was better, so it carries no ranking information to buy
with that leverage.

The accepted risk is that the value term then has no gradient *away* from the
abort region. The mitigation costs nothing and is mandatory:

- [x] Log the exclusion count and rate per evaluation block.
- [x] **A rising exclusion rate is a stop signal.** If the policy starts
      aborting, the estimator is blind to exactly that, and the run diverges
      silently — the same "small loss, destroyed policy" signature as ticket
      08's failure, with a different cause. Do not let it reproduce.

## The main technical risk, stated up front

`A` is dominated by the state columns of `Φ`; the action columns are the ones
the argmin reads. **With `λ` too large the action weights shrink toward zero,
`W·φ` goes constant across candidates, and you have rebuilt the exact failure
this effort has been measuring for two months** — with a clean-looking solve
and no divergence to warn you.

- [x] Standardize columns before accumulating (running scale, frozen once so
      the solve stays stationary).
- [x] Sweep `λ` and `γ` on `evaluation_seeds`, **never** `test_seeds`.
- [ ] Ticket 17's `r = sd_candidates(W·φ) / sd_candidates(c)` is the instrument
      that catches over-shrinkage. `r ≈ 0` means the ridge ate the ranking. It
      is reported every block, not only at the gate. **Not built here** — see
      Comments: this reads `decide()`-time candidate spread, which is ticket
      17's own live-report surface, and its own issue file carries the
      identical work item under "Part 3's companion diagnostic". Left for
      ticket 17 to implement where the rest of the live report lives.

## Two timescales (the trained-encoder arm)

For ticket 17's arm B the encoder and `layer1` still train by SGD on the same
residual loss, with `W` held at its last solve. `neural_learn_passes` /
`neural_batch_size` / `neural_learning_rate` survive for that path only; the
frozen arm uses none of them.

## What this amends in spec.md

- **Decision 5** (learning rule): still `Q(s,a)` on the Monte Carlo return —
  the same statistical object — but fitted by regularized least squares over a
  residual, not by Adam over `net(tokens)`.
- **Decision 9** (learning unit): "one batch per Episode, K passes, then
  discarded, strictly on-policy" becomes accumulated normal equations with
  forgetting. It remains Monte Carlo policy evaluation; it stops being a
  per-Episode fit. This is research note **#3** ("batch least-squares MC /
  LSTD-Q, removes lr tuning, scaling pathology, most variance") arriving inside
  the effort that had skipped straight to **#5**.

## Acceptance

- [x] `W = 0` before the first solve reproduces ticket 15's null exactly.
- [x] `γ`, `λ` and the solve cadence `N` chosen on `evaluation_seeds`, with the
      sweep recorded.
- [x] Exclusion rate logged per block; stop rule implemented.
- [x] Predicted self-golden diff: **zero.**

## Comments

**Landed as planned.** `learn` no longer runs any gradient step. A new module,
`stdvrp/policies/ridge_estimator.py`, holds `RidgeAccumulator`: plain numpy
(the accumulator is a 515x515 matrix and a 515-vector at the shipped
architecture — no benefit from GPU placement, and independent of whatever
device the encoder/head live on), accumulating `A`/`b`/`raw_sum_sq`/
`effective_n` with exponential forgetting applied once per `observe_episode`
call, freezing a per-column standardization scale on the first call that has
data, and solving `W = (A + λI)⁻¹b` in physical (unstandardized) units.
`QHead` (`network.py`) gained `features`/`w_vector`/`load_w_vector`:
`features(...)` returns `φ(s,v,a) = [x; ReLU(layer1(x)); 1]`, sharing the new
`_candidate_inputs` helper with `forward` so the two can never disagree about
what a candidate row is; `w_vector`/`load_w_vector` read and write the
combined `[linear.weight; layer2.weight; linear.bias+layer2.bias]` row
`RidgeAccumulator` solves for. `linear`/`layer2` are the only parameters this
class ever touches — `encoder`/`layer1` are never trained on this
(frozen-encoder) arm, matching the ticket's own scope: "the frozen arm uses
none of them" (`neural_learn_passes`/`batch_size`/`neural_learning_rate`),
left declared for ticket 17's trained-encoder arm to pick up.

`TransformerMonteCarloPolicy.__init__` drops `optimizer`/`learn_rng`/
`learn_passes`/`batch_size`/`grad_clip_norm`/`huber_delta` (nothing left for
any of them to do) and gains `ridge: RidgeAccumulator`/`solve_cadence: int`.
`learn` now: detects an aborted Episode off the final reward
(`rewards[-1] >= ABORT_PENALTY - ABORT_PENALTY_PER_SERVED_CLIENT *
number_clients`, a guaranteed floor under any real abort's penalty for this
Episode's demand — the only signal available under the frozen
`TrainablePolicy.learn(snapshots, actions, rewards)` signature, which this
ticket does not touch); replays each decision epoch once
(`_replay_joint_features`, mirroring `_replay_joint_q`'s existing replay
exactly, verified directly against it — see Verification) to build `Φ_t`/
`ỹ_t`; folds the whole Episode into `self.ridge` in one call; and re-solves
once `episodes_since_solve >= solve_cadence`, writing the result back via
`QHead.load_w_vector`. `NeuralPolicyState` (`neural_episode.py`) carries the
accumulator alongside `encoder`/`head`, and `neural_checkpoint.py` persists
it — required on load, not defaulted, since a pre-ticket-16 checkpoint was
trained under a different learning rule entirely and there is no coherent way
to "resume" it under this one. `neural_report.py`/`trainer.py` add the
mandatory exclusion mitigation: `EvaluationReport.training_episodes`/
`excluded_episodes`/`exclusion_rate`, printed every block, and
`exclusion_rate_stop_signal` (10%, a protocol constant alongside
`PATIENCE_BLOCKS`) stops the run — `NeuralTrainingResult.
stopped_for_exclusion_rate`, distinct from `converged` — the moment a block
crosses it, per the ticket's "do not let it reproduce."

`spawn_neural_episode_rngs` drops the fourth (`learn_rng`) stream — the ridge
solve shuffles nothing — verified that the remaining three streams are
bit-identical to `spawn(4)`'s first three
(`test_dropping_the_fourth_stream_does_not_change_the_first_three`), not just
asserted in prose.

**The sweep — four rounds, on `evaluation_seeds`, on the real Chengdu
dataset, never `test_seeds`**
(`.scratch/neural-policy/results/ridge_sweep*.{py,json,log}`). A pilot at the
originally-chosen `neural_solve_cadence: 1` (solving after a single Episode,
against 515 parameters) surfaced the first finding before the sweep itself
even started: two training Episodes contributed ~20-40 samples total, the
frozen scale was estimated from that same tiny sample, and the resulting
policy scored *worse* than an aborting one on the very next evaluation seed —
motivating `N` as a swept parameter in its own right, not a fixed "1" the
ticket's sketch implied.

| round | fixed | swept | best cell | best mean | vs. null (3365.09) |
|---|---|---|---|---|---|
| 1-2 | γ=0.98, N=50 | λ ∈ {1, 10, 100, 1e3, 1e4, 1e5} | λ=1 | 9152.26 | +172.0% |
| 3 | λ=1, γ=0.98 | N ∈ {50, 150} | N=50 | 9152.26 (N=150: 9039.89) | +172.0% |
| 4 | λ=1, N=50 | γ ∈ {0.90, 0.95, 0.99} | γ=0.99 | 6978.45 | +107.4% |

**No cell, in any round, beat the untrained null.** Round 1-2's λ sweep was
not monotonic (worst around λ=100-1,000; mildly recovering at 1e4; worse
again at 1e5 alongside a sharp rise in excluded Episodes, 10/60 there against
2/60 everywhere else) — ruled out "just needs more regularization" as the
whole story, since none of {1..1e5} shrank the standardized diagonal
(`effective_n` ≈ 12,000-20,000 at N=50) by more than ~10%. Round 3's N=150
essentially matched N=50 at the same λ, directly against this ticket's own
"50 Episodes ≈ 20,000 samples should be enough" framing. Round 4's γ sweep
was the one clean, monotonic result: larger γ (more effective history)
straightforwardly helped, and γ=0.99 (the largest tried, effective window
~100 Episodes) is the best cell measured across all four rounds — still
+107.4% worse than doing nothing. **Shipped defaults: `neural_ridge_gamma:
0.99`, `neural_ridge_lambda: 1.0`, `neural_solve_cadence: 50`** — the
least-bad cell measured, stated as exactly that in both `config.py` and
`experiments/chengdu/config.yaml`, not as a validated-good configuration.
γ above 0.99 was never tried and is the obvious next point on round 4's own
trend.

A working hypothesis for *why*, recorded rather than chased further within
this ticket's scope: several columns of `φ` are at or near zero variance in
the accumulated data (`claimed` is exactly zero for every chosen action —
`select_vehicle_possible_actions` excludes every candidate an argmin could
have read as "claimed" — and other components of the random `ReLU(layer1(x))`
projection may be near-constant for a given world). `_freeze_scale`'s floor
(`_SCALE_FLOOR = 1e-3`) keeps such a column's *standardized* coefficient
finite, but a small, largely-noise standardized coefficient divided by a
floor-sized scale can still produce a disproportionately large *physical*
coefficient — one that ridge's uniform standardized-space shrinkage does not
obviously control. Not confirmed (would need inspecting the solved `W`'s
per-column contribution on a real run, not done here), but consistent with
γ helping (more data narrows a near-zero variance estimate) while λ does not
(monotonically) — recorded in `config.py`/`experiments/chengdu/config.yaml`
for ticket 17 to pick up, since its own Gate A′ pass needs `λ`/`γ`/`N` chosen
on `evaluation_seeds` before "does training add value" can be answered on
this arm, and this sweep's raw numbers are exactly the evidence that question
needs.

**`/code-review`, two axes, both run and both fed back.** The Spec pass
caught a real, confirmed implementation bug:
`RidgeAccumulator.observe_episode` decayed `A`/`b`/`effective_n` by
`forgetting` on every call but left `raw_sum_sq` undecayed, so
`_freeze_scale`'s `mean_square = raw_sum_sq / effective_n` divided an
unbounded-growing numerator by a properly-windowed denominator — inflating
the frozen scale by an estimated ~26% at the shipped `γ=0.98`/`N=50` (worse
the longer training ran before the first solve: ~78% at `N=150`), equivalent
to solving with a larger effective `λ` than configured. Fixed (`raw_sum_sq`
now decays identically to `A`/`b`/`effective_n`), with a dedicated regression
test (`test_raw_sum_sq_decays_exactly_like_effective_n`) checking the
recurrence directly rather than trusting a plausible-looking final answer.
Round 4 (the γ sweep, and the shipped defaults) was run after the fix; rounds
1-3's absolute numbers predate it and are approximate — the estimated ~26-78%
scale inflation is far short of explaining a 170%+ gap from the null on its
own, so the qualitative finding (no configuration tested beats the null)
stands, but is noted as approximate everywhere it's cited. The Spec pass also
flagged that `γ` had not actually been swept in the first three rounds
despite the ticket asking for both `λ` and `γ`; round 4 closes that gap. The
Standards pass found no hard violations and two judgement calls, both
accepted as-is: `ExclusionStub`/inline test doubles in `test_neural_trainer.py`
reach into `RidgeAccumulator`'s counters directly rather than through
`observe_episode` (documented as deliberate — the real `learn()` never runs
under those stubs); and the DEAD-config-knob pattern already established for
`neural_warm_start`/`neural_huber_delta`/`neural_level_gain` now covers four
more fields (`neural_learning_rate`/`neural_learn_passes`/
`neural_batch_size`/`neural_grad_clip_norm`), judged proportionate given a
concrete future consumer (ticket 17) rather than speculative.

**Verification:** `mypy` clean (project-wide, 41 source files); `ruff check`/
`ruff format --check` clean on every file this ticket touches (two pre-existing,
untouched files elsewhere in the tree — `feature_extraction.py`, `model.py` —
already failed `ruff format --check` before this ticket, unrelated drift, left
alone); full suite (`uv run pytest -q -m "not golden"`) 4343 passed, 3
`golden`-marked deselected (up from ticket 15's 4279 — every new test net of
the ones replaced); `tests/test_self_golden.py` run explicitly, 6/6 passed —
this ticket never touches `monte_carlo.py`, `action_set.py`, or `tokenizer.py`,
and `decide()`/`_score()`/`QHead.__init__` are unmodified, so `W = 0`
reproducing ticket 15's null holds both by construction and by
`TestWZeroReproducesTicket15sNull`'s direct check (a `solve_cadence` the
training budget never reaches leaves `W` at exactly zero and `decide()`'s
argmin unchanged from before any data was ever observed).

`tests/unit/test_transformer_policy.py`: `TestGradClip`/`TestHuberDelta`
retired (the mechanisms are gone, not merely inert) in favour of `TestLearn`
(rewritten around the ridge solve), `TestWZeroReproducesTicket15sNull`,
`TestSolveCadence`, `TestAbortedEpisodesAreExcluded`, and
`TestReplayJointFeaturesMatchesReplayJointQ` (ties the new feature-replay path
directly to the already-tested `_replay_joint_q`, rather than trusting the two
to agree by construction). `tests/unit/test_ridge_estimator.py` is new (28
tests): OLS-recovery at a negligible ridge, forgetting behaviour, the
standardization-protects-small-scale-columns property with an unstandardized
control proving the risk is real at the tested scale/λ, abort exclusion,
state-dict round trips, and the `raw_sum_sq` regression test above.
`tests/unit/test_network.py` gained `TestFeaturesAndWVector`, pinning
`forward(...) == features(...) @ w_vector()` both at init and after
`load_w_vector`. `tests/fixtures/chengdu_mini/config.yaml` pins
`neural_solve_cadence: 1` explicitly (unlike the real config) so the existing
fast single-Episode wiring tests still see an immediate solve regardless of
the shipped default.
