# 08 — Gate A: does it learn

**What to build:** The three-part acceptance gate that decides whether the
network learned anything at all. This is the **hard landing gate** of the
effort — independent of whether it beats the baseline.

**Blocked by:** 07; **12** (`device: cuda` end to end — the three runs this
gate needs cost ~11.4 s/ep on CPU, so all three would hit the 24 h cap and be
recorded *"did not converge"*, which is not a Gate A result); and
`simulator-correctness`/11 (B20 — a crash in the shared
`Model._reroute_for` reached by this Policy's action set; it blocks the real
run, and its Client filter changes every decision, so any Gate A number
gathered before it lands is void)

**Status:** open

## The three parts

All three, on the held-out `test_seeds` (100..153). **Never on the evaluation
seeds** — those selected the checkpoint and the hyperparameters.

| Part | Test | Threshold |
|---|---|---|
| **Null model** | Trained vs. the same architecture **untrained**, paired per seed | Wilcoxon signed-rank **p < 0.05** and **≥ 5%** mean cost reduction |
| **Reproducibility** | ≥ **3** independent network-init seeds | Improvement reported as mean ± sd |
| **Calibration** | Spearman ρ(predicted `Q`, realised `U_t`) on held-out episodes | **≥ 0.5**, against ≈ 0 untrained |

## Why each one is there

**The null model is nearest-neighbour, not noise.** Ticket 05's myopic warm
start means the untrained network already goes to the nearest feasible Client.
Beating that by 5% is a real claim. Do not weaken the warm start to make this
gate easier — that games the null.

**Reproducibility** guards against one lucky init. A single run that clears the
bar is not evidence; three runs with a reported spread are. If the spread
straddles zero, it did not learn, whatever the best run says.

**Calibration is the part that cannot be faked.** A policy can improve by
accident — a shifted argmin that happens to route better without the value
function meaning anything. Correlation between predicted `Q` and the realised
cost-to-go measures whether the network learned *the thing it was trained to
learn*, independent of whether the policy improved. Spearman and not Pearson
because the cost distribution's right tail is brutal (research note **F10**).

## Work

- [ ] Develop and debug on the **mini fixture** (20 Clients, ~72 decisions/ep,
      no 8 GB world load). Run the gate itself on the real dataset.
- [ ] Wilcoxon signed-rank over the 50 paired seeds. Report **both** mean and
      median improvement — with this distribution they will differ, and the
      difference is informative.
- [ ] Report the numbers **whatever they are**, including a failure. Ticket 09
      cannot be interpreted without them.

## If the gate fails

The effort does not proceed to a verdict on a network that did not learn. The
failure is diagnosed first: the calibration number distinguishes "the value
function is not fitting" (a learning-rule or target-scaling problem) from "the
value function fits but the policy does not improve" (a decision-structure or
action-space problem). Those have different fixes and the gate is designed to
tell them apart.

## Acceptance

- [ ] All three parts pass, with the numbers recorded in this ticket's
      Comments.
- [ ] Predicted self-golden diff: **zero.**

## Comments

### 2026-08-01 — the first Gate A run did not measure learning; it measured five bugs

The run in `runs/gate_a/log.txt` (1 338+ episodes, arm `init_seed=0`) reported
untrained mean cost **81 701.58** against the linear baseline reference card's
**2 483.2**, `wins on 0/50 seeds` at every block, and the *identical* eval mean
`85170.5` at eight separate blocks. A nearest-neighbour null does not cost 33×
a tuned linear VFA, and a stochastic simulator does not return the same mean
eight times. Both are signatures of a degenerate policy, not of a hard problem.

Five defects, each reproduced on the mini fixture before it was touched.

**1. The fleet retired at decision epoch 1 of every Episode.** The synthetic
depot candidate got the same myopic warm start as a real one, so
`Q(v, depot) == average_minutes(position, depot) / horizon_length`, which is
exactly `0` for a vehicle standing on the depot — the smallest value the
warm-started head can produce. Every vehicle starts parked there, so every
vehicle's argmin was the depot; `Model.transition_function` saw
`fleet.all_parked()` and terminated. Measured on the mini fixture: action
`[0, 0, 0, 0]`, `state_count=1`, 15/15 Clients unserved, cost 8 350, all of it
delay. `81 701 ≈ 150 clients × ~545 min` of unserved delay is the same number
on the real dataset. Compounding it, `Model._reroute_for` has no branch that
fires for a parked vehicle, so **retiring is irreversible**.

**2. `learn` divided both the prediction and the target by `_return_scale`.**
That is arithmetically identical to regressing `Q` on the *raw* return, while
the warm start puts `Q` in normalized-minutes units — five orders of magnitude
apart, with the gradient reaching the network divided by 127 500. Least squares
did the only thing available: fit the mean, discard the ranking. That is what
`loss 0.000` from episode 3 onward alongside an unmoving eval cost was saying.

**3. `learn` emitted `m` samples per decision epoch, each regressed
individually onto the same `U_t`.** For one epoch the squared loss decomposes
as `Σ_v (Q_tv − y_t)² = m(Q̄_t − y_t)² + Σ_v (Q_tv − Q̄_t)²`, and the second
term is an explicit penalty on the spread of `Q` across candidates — the one
quantity the argmin reads. The objective was *paying* the optimizer to become
action-blind. The linear baseline never had this: `action_features` builds
**one** joint-action vector per epoch whose cost components sum over the whole
fleet, so `Q = W·general(s) + Σ_v W·f(s, a_v)` is already additive over
vehicles, with the matching coordinate-wise argmin.

**4. The warm start died as a ReLU inside one training episode.** `hidden[0] =
ReLU(w0·x)` carried a signal of `minutes/horizon_length ∈ [0.002, 0.061]` on a
row whose other 384 columns multiply a prefix of L2 norm ~2.9. Adam moves every
weight by ~`lr` per step whatever its gradient, so the perturbation dwarfed the
signal. Measured: pre-activation `[+0.003, +1.000]` → `[−0.546, −0.077]` after
one episode — **0/16 candidates alive**, zero gradient, unrecoverable. `Q`'s
spread across candidates collapsed 0.036 → 0.0007 and the argmin stopped
picking the nearest Client. This is the defect that made the other four
un-diagnosable from the log.

**5. Nothing selected the best network.** `ConvergenceState` tracks
`best_mean_cost`/`best_episode` and only ever *prints* them; `run_gate_a`
measures `training.policy_state`, i.e. whatever the last training episode left.
At `lr 3.0e-4` that is the difference between −6.3% and +90%.

### The fixes, and what each is measured against

| # | Fix | Where |
|---|---|---|
| 1 | `is_depot` flag into `QHead`; warm start prices home at one whole horizon. Plus `_depot_is_feasible`: home is legal only with nothing left to travel to, or once the return leg breaches `shift_end_minute` — the baseline's own gate minus its `350`/`310` literals. **Amends ADR-0007.** | `network.py`, `transformer_policy.py` |
| 2 | Scale the target only; the prediction is already normalized by the warm start | `transformer_policy.py::learn` |
| 3 | Regress `Q_joint(s, a) = Σ_v Q(s, v, a_v)`, one sample per epoch — the linear baseline's own additive-over-vehicles shape. Also 4× cheaper: one encoder pass per epoch of replay | `transformer_policy.py::_replay_joint_q` |
| 4 | Warm start moves to a trainable **linear** path, `Q = linear(x) + layer2(ReLU(layer1(x)))`, with `layer2` zeroed at init. Same weights, same trainability, no kink to fall off | `network.py::QHead` |
| 5 | Snapshot the weights of the best evaluation block, restore before returning; persisted in the checkpoint | `trainer.py`, `neural_checkpoint.py` |
| — | `_is_retired`: a parked vehicle claims no Client. Its action is discarded by `_reroute_for`, but its *claim* is not — it starved that Client for the rest of the Episode. Latent before fix 1, reachable after it | `transformer_policy.py` |
| — | `loss` printed in scientific notation: at `%.3f` every healthy value is `0.000` | `neural_report.py` |
| — | `neural_learning_rate` 3.0e-4 → 3.0e-5; 3.0e-4 measurably diverges (below) | both configs |

### Measurements (mini fixture, 4 vehicles, ~15 Clients)

**The null model is now the null model the spec describes.** An independently
written nearest-neighbour policy and the untrained network agree to the last
decimal on seeds 100/101 — `477.5` and `327.8`, 0 unserved. Before: `8 350` and
`10 293`, everything unserved, 1 decision epoch.

**It learns.** Trained vs. the untrained null, paired over ten held-out seeds:

| lr | best block | typical band | end of 200 episodes |
|---|---|---|---|
| 3.0e-4 | **−6.26%** @ep30, 8/10 wins | −5%..−6% to ep 70 | diverges: +17% @90, +91% @100, +255% @130 |
| 3.0e-5 | **−5.69%** @ep30, 7/10 wins | −2%..−5.7% throughout | −4.5% @180, +0.1% @200 |

Ticket 08's own thresholds are for the real dataset over `test_seeds` 100..153,
not for this fixture — these numbers say the machinery learns, not that the
gate passes.

### 2026-08-01 (later) — first real-dataset run: it learns, then it diverges

Gate A relaunched on the real Chengdu dataset, `--device cpu` (ticket 12
measured CUDA slower here), protocol frozen, no cap overrides. **Arm 0 was
stopped by hand at episode 653** once the picture was unambiguous; arms 1 and 2
never ran. 15.0 s/episode.

**The null model, in the real harness: `Gate A: untrained mean cost 5299.4792`**
over the 50 `test_seeds`, against 81 701.58 before. Independently reproduced by
a standalone probe to the second decimal. That part is closed.

**Arm 0's live report looked like a total failure and was partly a reading
error.** The blocks pair against the linear reference card on
`evaluation_seeds`; the 5 299 null is on `test_seeds`. Comparing the two is
comparing different seed sets, and the first read of this log did exactly that
before catching it. The comparison Gate A part 1 actually makes, best-block
checkpoint vs. the null **on the same eight `test_seeds`**:

| | null | trained (best block, ep 50) |
|---|---|---|
| mean cost | 6 259.0 | **5 962.6 (−4.7%)** |
| unserved Clients | 0 | **0** |
| overtime | 0–84 | 0–109 |

Six of eight seeds improved. **The Policy does learn on the real dataset.**
−4.7% on eight seeds against a ≥5% threshold on fifty is not a pass, but it is
not the negative result the live log appeared to show either.

**What actually fails is stability.** The best block is ep 50; by ep 100 the
evaluation mean is 4× worse and it never recovers, through two patience-driven
lr cuts (3.0e-5 → 9.0e-6 → 2.7e-6). Best-network selection (fix 5) is the only
reason anything usable survived the run.

**The depot gate is not the cause** — worth recording because it was the
leading hypothesis. Instrumented over both networks: zero unserved Clients,
overtime marginal, and every depot action taken under condition 1 (that vehicle
had no unclaimed Client left, 4–6 pending held by others). Condition 2 (return
leg past the shift end) opens on this dataset and is not being abused.
`_depot_is_feasible` stays as it is.

**The likely cause is the candidate count.** `decide` takes an argmin over
~151 candidates on Chengdu against ~16 on the mini fixture, while `learn`
attaches a gradient to exactly one of them per (epoch, vehicle) — the other 150
`Q` values are unconstrained extrapolation, and the minimum of 151 noisy
estimates is far more biased than the minimum of 16. That is the difference
between the fixture where `3.0e-5` is stable and the real dataset where it is
not, and it is why the fixture result did not transfer. Being measured now as
an lr/gradient-clipping sweep on `evaluation_seeds` (never `test_seeds`).

### Still open

- **The gate has not been completed.** Arm 0 stopped at ep 653, arms 1 and 2
  not run. No Gate A verdict exists.
- Two verified defects left unfixed, both `degrades-learning`, neither blocking:
  `claimed_mask` is rebuilt per epoch so it ignores the other vehicles'
  in-flight commitments (`MonteCarloPolicy` seeds `forbidden_actions` from
  `self.action` for *every* other vehicle); and ~24-33% of `learn`'s samples
  carry an action the simulator discarded because the vehicle was mid-service.
  *(A fix for the second was tried and measured **worse** — see the 2026-08-01
  comment below, rejected row. Both remain open, the second now with evidence
  that the naive fix is not the fix.)*
- Gate A's calibration statistic pools `(Q, U_t)` pairs whose `U_t` is
  dominated by the state's cost-to-go, so a `Q` that has collapsed to `V(s)`
  scores *well* on it. Emitting one joint pair per epoch (fix 3) removes the
  `m`-fold duplication but not the underlying blind spot: **part 3 can pass on
  an action-blind network.** Worth knowing before the number is interpreted.
- `warmup_learning_rate` is in both configs and read only by the linear
  `Trainer.train()`; `train_neural` ignores it.

### 2026-08-01 (later still) — the divergence attacked at its root: the cost function is now an input

The lr/clipping sweep announced above treats the instability as an
optimization problem. The diagnosis says it is a *representation* problem
first: `Q(s, a)` decomposes as `V(s) + A(s, a)`, the Monte Carlo return's
variance is almost entirely `V(s)`, so the network spends its capacity
fitting `V` and leaves `A` — the only quantity the argmin reads — as a
residual rediscovered from noisy returns. On the fixture's ~16-candidate
argmin that residual is learnable; on Chengdu's ~151, the minimum of 151
unanchored extrapolations is dominated by their noise. That is exactly the
"candidate count" observation above, restated as something fixable.

**The fix is the research note's own recommendation #5** ("neural VFA over
*enriched* features", Chen/Ulmer/Thomas, up to +22 %), which spec.md cited
while doing the opposite ("the features are not enriched, they are
*removed*"). **Spec.md decision 1 is amended** (dated section there): each
`(client, vehicle)` arc token now carries the four projected components of
the simulator's own cost function — `earliness_cost`, `delay_cost`,
`future_delay`, `overtime_cost`, per-pair marginals mirroring
`FeatureExtractor.candidate_features` minus its multiplicity classifier —
next to `[minutes, path_length]`, and the synthetic depot candidate gets the
same six-field block (`Tokens.depot_arc_tokens` -> `Embeddings.depot`).
`arc_embed` is linear, so `QHead`'s linear path composed with it spans every
linear function of the six facts: `A(s, a)` is a near-linear readout from
the first gradient step, and `Q` can no longer be arbitrarily abrupt between
neighbouring candidates — the thing the 151-candidate argmin was amplifying.

Two changes landed; a third was tried, measured, and rejected:

| # | Change | Where |
|---|---|---|
| 1 | The four projected costs on every arc token + the depot block | `tokenizer.py`, `network.py`, `transformer_policy.py` |
| 2 | `neural_grad_clip_norm` config knob (default `null` = exact prior behavior) — the sweep's second lever, now reproducible from config | `config.py`, `transformer_policy.py`, both configs |
| ✗ | **Executed-actions-only regression — rejected on measurement.** `learn`/`calibration_pairs` skipping the `Q` terms of vehicles whose action the simulator discards (the ~24-33 % above: mid-service and retired vehicles) removes one bias but introduces a worse one: the skipped vehicle's share of `V(s)` vanishes from that epoch's joint sum while `U_t` still contains its future cost, so the per-term burden varies by epoch and the additive decomposition stops being consistent. A/B on the mini fixture (third column below): mean over 20 blocks fell from −2.42 % to −0.74 %, blocks-worse-than-null rose 3 → 7, end of run fell −2.16 % → +4.55 %. Reverted; recorded in `transformer_policy.py`'s docstring ("A tried-and-rejected variant") so it is not re-tried naively | — |

**What did not move, verified by test and by measurement:** the observability
rule (ADR-0006 — same five `tokenize` arguments, same structural test; the
cost *rate constants* are configuration-class, see the ADR's dated
clarification); the baseline's general-state feature engineering stays out
(polynomials, bins, `late_count/13`, literals, classifier); `decide`/
`decide_train` semantics (evaluation numbers of a given network are
untouched); and the null model — `arc_embed` row 0 reads only minutes, so
untrained `Q` is bit-identical (measured below: null mean `577.22` on both
sides, to the cent).

**Measured on the mini fixture** (this working tree; 200 episodes, greedy
eval every 10 over held-out seeds 100..109, `init_seed=0`, lr `3.0e-5`, paired
against the same-architecture untrained null):

| | control (pre-change) | + cost features (kept) | + exec-filter (rejected) |
|---|---|---|---|
| null mean | 577.22 | 577.22 (identical) | 577.22 |
| best block | −5.27 % @ep110, 7/10 wins | **−5.68 % @ep80, 8/10 wins** | −4.40 % @ep10 |
| mean over 20 blocks | −1.55 % | **−2.42 %** | −0.74 % |
| blocks worse than null | 5/20 | **3/20** | 7/20 |
| mean of last 5 blocks | **+0.47 %** (degrading) | **−1.84 %** (still ahead) | +1.12 % |
| end @ep200 | +2.67 %, 3/10 | **−2.16 %**, 5/10 | +4.55 % |

The late-run degradation — the real dataset's failure mode — is what moves
most, and it moves in the right direction on the fixture where it is
*hardest* to see (16 candidates, not 151). The real test is the relaunched
real-dataset arm (below).

**Relaunched: arm 0 on the real Chengdu dataset** (2026-08-01), protocol
frozen, no cap overrides, `--device cpu` (ticket 12's measurement), clip off
(`null` — one variable at a time), fresh checkpoint dir `runs/gate_a_v2/`
(the old `runs/gate_a/` checkpoints hold the pre-amendment architecture,
`arc_embed` 2-wide, and stay as that run's evidence), log at
`runs/gate_a_v2/log.txt`. Arms 1-2 after arm 0's picture is readable.
