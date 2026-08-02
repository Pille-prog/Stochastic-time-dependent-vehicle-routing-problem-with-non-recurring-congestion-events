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

**Interim read at ep ~823** (run still training; the arm's official numbers
land in `runs/gate_a_v2/results_init0.json` at convergence). The live blocks
pair against the *linear reference card* on `evaluation_seeds`; the null
anchor for that seed set, measured by a standalone probe (untrained
`init_seed=0` network, greedy, 50 seeds): **mean 5484.25** (min 2944, max
8949). Against it: block @50 `5597` (+2 %), spikes @100 `12160` / @150
`37179`, oscillation `6-14k` through two patience lr cuts, then **best @650
`4150` = −24.3 % below the null** — at the lowest lr (2.7e-6) — and back to
`7-8k` after. Reading: (a) *depth is unlocked* — the previous architecture's
best-ever was −4.7 % vs null; the amended one reached −24 % at its best
block, which is what the cost features were for; (b) *stability is not
solved* — 1 of 16 blocks below the null, training episodes spiking to 54-93k
with the minibatch loss jumping ~40× (2e-4 → 8.7e-3) in the same stretches,
and eval means still moving thousands at lr 2.7e-6, so `Q` differences
between candidates remain razor-thin; (c) the best arriving only after two
lr cuts says the remaining binding constraint is the *optimization*, not the
representation — exactly what the announced lr × `neural_grad_clip_norm`
sweep on `evaluation_seeds` is for.

### 2026-08-01 (fourth) — why it is not learning, measured: one large win, two rejected fixes

The interim read above ends on "the remaining binding constraint is the
*optimization*". That was wrong, and the measurements below are what corrected
it. The binding constraint is that **the training objective cannot see the
quantity the policy decides with** — and the largest improvement available was
never in the optimizer at all, it was an initialization left at zero.

#### 1. The failure mode, decomposed instead of inferred

A probe over eight `evaluation_seeds` of the real dataset reporting cost
*components* rather than the total (scratchpad `warm_start_probe.py`):

| policy | total | delay | earliness | overtime | unserved |
|---|---|---|---|---|---|
| untrained null (`init_seed=0`) | 4754 | 3191 | 1070 | 8 | 0.0 |
| `runs/gate_a_v2` best block (@650) | 3794 | 2590 | 318 | 183 | 0.0 |
| `runs/gate_a_v2` latest (@~850) | 8293 | **6604** | 977 | 64 | 0.2 |

The degraded network serves **everybody** — `unserved` stays at zero. So this
is not the fleet retiring early, not a depot-masking regression, not a
simulator effect. It is a *ranking* that got worse: the same Clients, visited
in a worse order, paying double the delay. That rules out the whole class of
explanations the previous comment was still entertaining.

#### 2. Why the ranking is what training damages

`learn` regresses `Q_joint(s,a) = sum_v Q(s,v,a_v)` — one scalar per decision
epoch — onto `U_t`. The gradient reaching every candidate term of that epoch
is the *same* residual, so the loss is mathematically invariant to how a given
sum is split across candidates. The per-vehicle `argmin` reads nothing *but*
that split. The objective and the decision are decoupled.

The linear baseline survives this because 19 weights **cannot** fit `V(s)` well
enough to kill the residual, so its action columns keep receiving signal. At
595k parameters the encoder fits `V(s)` easily, the residual becomes noise, and
the arc's cost weights — the ones the `argmin` reads — random-walk. Loss `1e-4`
with a policy degrading below its own null is exactly that signature, and it is
what the log has shown for 950 episodes.

#### 3. What landed: the warm start was pricing the wrong thing

Since the decision-1 amendment the arc token carries the projected cost
components — and `arc_embed` row 0 was still `[1,0,0,0,0,0]`, so they were
**init-inert by construction**, and gradient descent was being asked to
rediscover from noisy Monte Carlo returns an arithmetic identity the tokenizer
had already computed exactly. Setting that row to price the leg instead:

`Q = (minutes + earliness_cost + delay_cost + overtime_cost) / horizon_length`

— one minute-equivalent currency, every term already scaled the same way, no
free parameter. Chosen over variants on eight `evaluation_seeds`
(cost-without-minutes 34601: with no tie-break, the many zero-cost candidates
are picked arbitrarily far away; `+future_delay` at full weight 5185 and at a
tenth 3778; `future_delay` alone 5413), then confirmed on the **full 50
`evaluation_seeds`, never `test_seeds`**:

| warm start | mean over 50 `evaluation_seeds` |
|---|---|
| `minutes` (the frozen null) | 5484.25 |
| **`cost`** | **3693.23 — −32.7%, wins 47/50, Wilcoxon p = 2.5e-14** |

For scale: the *best block of 650 training episodes* on this seed set was 4150
(−24.3%). **The initialization beats the training, by a wide margin, at zero
compute.** Shipped as `neural_warm_start` (`config.py`; the weight vectors live
in `tokenizer.py::WARM_START_WEIGHTS`, beside the arc-token layout they index
and in the one module that needs no torch); default `minutes`, with
`experiments/chengdu/config.yaml` set to `cost`.

#### 4. Rejected on measurement: the dueling decomposition

§2 has a standard structural remedy — `Q(s,v,a) = V(s,v) + [A − mean over the
sweep's candidates of A]` (Wang et al. 2016) — which makes fitting the level
`V`'s job by construction and identifies the advantage from *within-state*
contrast. It was implemented with the null model provably preserved
(subtracting one constant per sweep cannot move an `argmin`; measured —
identical null means `577.22` and `542.82` with and without it).

It made learning **much worse**, and the reason is worth more than the code
was. The `argmin` picks the *minimum*, so the chosen action's centred advantage
is negative while the target is positive; the residual therefore pushes it *up,
toward the candidate mean* — actively un-learning that this action was the best
one — every step, until `V` catches up. Without centring, that same pressure
raises all candidates near-equally (they share parameters): a level shift the
`argmin` cannot see. **The centring removed a benign escape valve and turned a
level error into ranking damage.** The level error is large for a structural
reason: `Q` starts in `minutes/horizon_length` units because of the warm start
while the return lives on an unrelated scale, so the optimizer's first job is
reconciling two arbitrary scales — and it pays for it with the ranking. That is
the "fitted the mean and threw the ranking away" signature already recorded in
`transformer_policy.py` under "Target scaling", one layer deeper. Anything
revisiting this must fix the scale mismatch *first*. Reverted; kept in
`network.py`'s docstring ("A dueling decomposition of `Q` — tried, measured,
and rejected").

#### 5. Rejected on measurement: the Huber knee

`huber_loss`'s default `delta=1.0` is ~100× every residual this regression
produces, so it is exactly `0.5 * MSE` — verified twice: against torch's own
piecewise definition, and by a test showing `delta=1.0` and `delta=1e6` leave
**bit-identical** parameters after a real episode. The robustness never
engaged, and a truncated episode's terminal penalty (F10) entered squared on
all ~400 of that episode's epochs. But `delta=0.02` was far *worse*, not
better: shrinking the knee also makes *ordinary* samples near-linear, so their
gradients go sign-like and constant-magnitude, which under Adam is a much
larger effective step on precisely the quantity the `argmin` reads. Shipped as
`neural_huber_delta` **defaulted to 1.0** — a knob that records a measurement
rather than one that changes behaviour.

#### 6. All arms, mini fixture (200 episodes, seeds 100..109, `init_seed=0`, lr 3.0e-5)

| arm | null | best block | mean over blocks | last 5 | end | worse than null |
|---|---|---|---|---|---|---|
| control (pre-cost-features) | 577.2 | −5.27% | −1.55% | +0.47% | +2.7% | 5/20 |
| **+ cost features (committed)** | 577.2 | −5.68% | **−2.42%** | **−1.84%** | **−2.2%** | **3/20** |
| + dueling | 577.2 | −5.99% | +10.73% | +31.90% | +34.0% | 15/20 |
| + dueling + `cost` warm start | 542.8 | +0.88% | +19.63% | +58.23% | +27.1% | 20/20 |
| + dueling + `cost` + huber 0.02 | 542.8 | +0.30% | +53.98% | +107.85% | +73.0% | 15/15 |
| `cost` warm start, no dueling | 542.8 | −3.94% | +2.57% | +3.37% | +4.9% | 14/20 |
| dueling, value branch ×10 | 577.2 | **−7.28%** | +2.44% | +15.05% | +23.8% | 9/20 |

Read: the committed learning rule is the most *stable* arm; the `cost` warm
start gives the best *absolute* policy (its untrained null, 542.8, already
beats every trained network in the table — the best trained `minutes` block is
577.2 × (1 − 0.0568) = 544.4); and **no arm shows training reliably adding
value on top of a good initialization.**

#### 7. What this means for the gate

- The `minutes` arm in `runs/gate_a_v2/` is left running to convergence: it is
  the record for the **frozen** null, and its numbers belong here as-is. The
  rejected changes were reverted, so its checkpoints stay loadable.
- `experiments/chengdu/config.yaml` now carries `neural_warm_start: cost`,
  which makes Gate A's null the 3693 policy instead of the 5484 one — a
  **harder** gate, decided on `evaluation_seeds`, which is exactly where
  spec.md puts architecture tuning. A criterion moved in the direction that
  makes it harder to pass is not a criterion rewritten to be passable.
- The honest statement of where this stands, **as written before arm 0
  finished and corrected by it below**: the network looked like it was not
  adding value over its own initialization. Under the `cost` warm start that
  is right. Under `minutes` it is wrong — arm 0's converged measurement is
  +15.5% over its own null, significant. See the official result below.

### 2026-08-01 (fifth) — the level term: the scale mismatch, found and given a fast home

The `cost` arm launched above was watched for 87 episodes and made the previous
section's last bullet concrete, then explained it.

**What it showed.** Null on `test_seeds` **3811.28** (so the warm start's −28%
holds on the verdict set too, not just on `evaluation_seeds`). First evaluation
block, at episode 50: **6168.5 — +62% *worse* than its own null**, with training
episode costs climbing monotonically 6-8k → 15-26k → 22-50k. Training does not
fail to help here. It actively destroys a good policy, monotonically.

**Why, from the first line of the log.** Episode 1's loss is `3.2e-01`, falling
to ~5e-4. That is not noise, it is a level. At init `Q_joint = sum_v Q(s,v,a_v)`
is a sum of minute-normalised quantities landing around **0.3-0.9**, while the
target `U_t / (number_clients * episode_length)` is **0.03 at t=0, decaying to
0**. `0.5 * 0.8² = 0.32` reproduces the logged loss exactly. So before the
network can learn anything about ranking it must walk `Q_joint` down by an
order of magnitude — while the candidate differences the `argmin` reads are
~0.03. The correction is 10-30× the signal **and has the same sign every
step**, so Adam (step ~`lr` regardless of gradient size) walks in a straight
line, dragging every weight it touches. The `cost` warm start makes `Q` larger,
which is exactly why it degrades faster than `minutes`.

**The fix.** `QHead.linear`'s bias is the one weight added identically to every
candidate of every sweep — the only one no `argmin` can see. It already gets
precisely this gradient; it is just as slow as everything else, needing ~2700
steps (~100 episodes at this fixture's ~27 steps/episode) to travel the ~0.08
required. That is the exact window of the damage. `neural_level_gain`
multiplies its effective contribution, so one step moves the level that many
times as far; written as `(gain - 1.0) * linear.bias` so the default `1.0` adds
exactly `0.0` and is bit-identical to the term not existing. This is what the
rejected dueling attempt should have been: both give the level a fast home, but
a centred advantage puts the correction where the `argmin` reads and a bias
puts it in the one direction the `argmin` is blind to.

**Measured, and conditional** (mini fixture, 200 episodes, gain 100):

| arm | null | best | mean over blocks | last 5 | end | worse than null |
|---|---|---|---|---|---|---|
| `minutes` (committed) | 577.2 | −5.68% | −2.42% | −1.84% | −2.16% | 3/20 |
| **`minutes` + gain 100** | 577.2 | **−7.44%** | **−2.79%** | **−3.03%** | −0.13% | **2/20** |
| `cost` | 542.8 | −3.94% | +2.57% | +3.37% | +4.89% | 14/20 |
| `cost` + gain 100 | 542.8 | −2.39% | +5.31% | +8.43% | +12.68% | 16/20 |

On `minutes` it improves every column and is the **best learning behaviour
measured anywhere in this effort**. On `cost` it makes things worse. The gain
does not make the ranking signal less noisy — it removes what was throttling
the optimizer. From a poor ranking (nearest-neighbour) there is something to
find, so that helps; from a good one (cost-greedy) it only lets the noise do
damage sooner.

**Which is the finding, stated plainly: training is not adding value on top of
a *good* initialization, it is spending it.** Absolute numbers on the mini
fixture make it unarguable — the *untrained* `cost` network sits at 542.8, and
no trained configuration in the table sustains anything below that (best
sustained is `minutes` + gain 100 at 577.2 × 0.9697 = 559.7). Note the
qualifier: from the *poor* `minutes` initialization training does genuinely
help, and the official arm-0 result below measures how much.

**Shipped:** `experiments/chengdu/config.yaml` keeps `neural_warm_start: cost`
with `neural_level_gain: 1.0` — the warm start is measured on real data over 50
`evaluation_seeds`, the gain only on the fixture and only helps the other warm
start. A Gate A run whose question is specifically "does it learn" should use
`minutes` + gain 100, where the answer is most clearly yes.

### 2026-08-01 (sixth) — OFFICIAL: Gate A arm 0, converged (`runs/gate_a_v2/results_init0.json`)

The frozen-null arm converged at **1150 episodes** and restored its best
evaluated network (episode 650) for measurement, per protocol. On the 50
held-out `test_seeds`, paired against the *same architecture untrained*:

| | value |
|---|---|
| untrained (null) mean | **5299.48** |
| trained mean | **4423.73** |
| mean per-seed reduction | **+15.49%** (median +21.02%) |
| trained cheaper on | **39 / 50 seeds** |
| Wilcoxon signed-rank | **p = 6.21e-05** |
| calibration Spearman rho, trained | **0.5427** (untrained −0.3487) |

**Part 1 (null model): PASS for this arm** — p < 0.05 and ≥ 5% reduction, both
comfortably. **Part 3 (calibration): PASS** — 0.543 against a required 0.5,
from −0.349 untrained. **Part 2 (reproducibility): NOT MEASURED** — it requires
≥ 3 independent init seeds and exactly one ran, so the reported `sd 0.00%` is
an artefact of `n = 1`, not a result.

`GATE A: FAIL` in the log is therefore **"one of three arms run"**, not "the
network did not learn". It learned, significantly, and the ticket's own
diagnostic instrument agrees: the failure protocol above says calibration is
what distinguishes "the value function is not fitting" from "the value function
fits but the policy does not improve", and here *both* fit — rho 0.543 **and**
a policy 15.5% better than its null.

Two things this corrects in the sections above, written while the arm was still
running:

1. The eval-block trajectory degrading past the null (−24% at ep 650, +74%
   later) is a description of the *trajectory*, not of the outcome. The trainer
   selects and restores the best block, so the measured network is ep 650's.
2. The checkpoint selected on `evaluation_seeds` (−24.3% there) transfers to
   `test_seeds` at +15.5%. That is shrinkage — F12's winner's curse, visible
   and modest — not collapse.

**And the finding that survives all of it,** on one seed set, apples to apples:

| policy on the 50 `test_seeds` | mean cost |
|---|---|
| untrained, `minutes` warm start (the Gate A null) | 5299.48 |
| **trained** to convergence, `minutes` — 1150 episodes | 4423.73 |
| untrained, `cost` warm start — **zero training** | **3811.28** |

**The untrained cost-greedy initialization is 13.8% cheaper than the fully
trained network.** Training works; it is simply worth less than the arithmetic
the tokenizer was already computing and the warm start was throwing away.

#### What closing this ticket now requires

- Arms 1 and 2 (`--init-seeds 1,2`) — ~4-6h each, unchanged config, purely
  mechanical. Part 2 is the only unmeasured part, and part 1 must hold on each.
- The relaunch, verbatim:
  `uv run python -u scripts/run_gate_a.py --config experiments/chengdu/config.yaml --reference-card experiments/chengdu/reference_card.json --data-dir "C:/Users/ferna/OneDrive/Documentos/Mega city" --init-seeds 1,2 --checkpoint-dir runs/gate_a_v2 --results-path runs/gate_a_v2/results_init12.json --device cpu`
  The config now ships `neural_warm_start: minutes` for exactly this reason —
  all three arms must share one architecture or the null differs between them
  and the gate is void. Flip it to `cost` for Gate B, not before.
- Whatever the arms say, Gate B inherits a live question the gate itself does
  not ask: the best *policy* measured here is untrained.

### 2026-08-01 (seventh) — conclusions

Consolidated so the sections above do not have to be re-derived. Every number
here is on the **50 held-out `test_seeds`**, from
`runs/gate_a_v2/results_init0.json` and `experiments/chengdu/reference_card.json`
(`test_seed_costs`, the baseline's own per-seed vector on those same seeds).

#### Where the policy actually stands

| policy | mean cost | vs. the linear baseline |
|---|---|---|
| linear baseline, best cell (budget 100, `test_action_count` 40) | **3384.82** | — |
| neural, untrained, `minutes` warm start (Gate A's null) | 5299.48 | +56.6% |
| neural, **trained** to convergence, 1150 episodes | 4423.73 | +30.7% |
| neural, untrained, `cost` warm start — **zero training** | **3811.28** | **+12.6%** |

The linear baseline still wins, on 39 of the 50 seeds against the trained
network. But the gap went from +56.6% to +12.6%, and **three quarters of that
came from the warm start, one quarter from 1150 episodes of training.**

#### 1. It learns — that question is closed

Arm 0, converged: +15.49% mean reduction over its own null (median +21.02%),
cheaper on 39/50 seeds, Wilcoxon p = 6.21e-05, calibration Spearman 0.543
against −0.349 untrained. Parts 1 and 3 PASS. The `GATE A: FAIL` line means
"one of three arms has run"; part 2 is unmeasured, not failed.

#### 2. But learning is not where the value was

1150 episodes bought 5299 → 4424. One line of initialization bought
5299 → 3811. The untrained network beats the trained one by 13.8%.

#### 3. Why, structurally — this is the part worth keeping

`learn` regresses **one scalar per decision epoch** onto the Monte Carlo
return, so every candidate term of that epoch receives the same residual and
the loss is invariant to how the sum is split between candidates. That split
is the only thing the `argmin` reads. Capacity makes this *worse*: the linear
baseline survives because 19 weights cannot fit `V(s)` well enough to kill the
residual; at 595k the encoder fits it easily and what remains is noise.
Compounding it, `Q_joint` starts at 0.3-0.9 against a target of 0.03, so the
first ~100 episodes are spent walking that gap down with same-signed steps
that drag the ranking with them.

#### 4. The pattern across four attempts — the strongest evidence here

| attempt | result |
|---|---|
| `cost` warm start | **−32.7%** (50 `evaluation_seeds`, p = 2.5e-14) |
| level gain (a fast home for the level) | better from `minutes`, **worse** from `cost` |
| dueling `V` + centred advantage | mean over blocks −2.42% → **+10.73%** |
| Huber knee at 0.02 | **+129%** by episode 140 |

The last three share a property: **each makes the optimizer more effective, and
each only helps when the starting policy is bad.** From a good starting point
they accelerate the damage. That is the signature of an objective that is not
the one we want optimized, and it is stronger evidence than any of the
theoretical arguments in the sections above — which is why the two rejected
changes are kept in the docstrings rather than deleted.

#### 5. What this hands to Gate B

- The binding constraint measured here is the **estimator**, not the
  representation. That points at the research note's ranked #3/#4 —
  least-squares Monte Carlo / LSTD-Q (removes the learning rate and the
  scaling pathology at the root), adaptive stepsize, common random numbers —
  and **not** at #5, which is now done.
- **Gate B must compare the linear baseline against the untrained `cost`
  network as well as the trained one.** The spec frames Gate B as trained
  transformer vs. baseline best cell; on this evidence the best transformer
  *policy* is untrained, and a comparison that omits it would report the wrong
  contender. 3811.28 vs 3384.82 is +12.6%; Gate B needs ≥3% the other way.

#### Status

Arms 1 and 2 launched (`runs/gate_a_v2/log_init12.txt` →
`results_init12.json`). If part 1 holds on both, all three parts are satisfied
and this ticket closes with the numbers above.
