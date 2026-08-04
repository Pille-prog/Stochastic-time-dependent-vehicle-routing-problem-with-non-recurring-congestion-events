# Spec: Neural policy — a transformer over raw State, against the linear baseline

Status: open (grilling session 2026-07-30)

## Goal

Replace the Policy's **approximator**, not its algorithm: keep every-visit Monte
Carlo policy evaluation exactly as it is, and swap the 19-feature linear
`Q = X · W` for a transformer that reads **raw State facts** and produces
`Q(s, a)` directly. No hand-engineered features, no polynomial cross-terms, no
earliness bins, no candidate-set heuristic. *(Amended 2026-08-01, ticket 08:
the cost function's own per-candidate components are inputs — see decision 1's
amendment; the state features stay raw.)*

Two acceptance questions, deliberately separated:

1. **Does it learn?** A hard landing gate. Answered against a null model, three
   network-init seeds, and a calibration check.
2. **Does it beat the linear MonteCarloPolicy?** The effort's *objective*, with
   a pre-committed two-arm escalation and an honest negative-result exit.

Success is not "the transformer wins". Success is that we **know** whether it
wins, on a protocol frozen before the first number was measured.

## Point of departure

### What exists

`MonteCarloPolicy` (`src/stdvrp/policies/monte_carlo.py`) is a linear VFA over
19 features (`FeatureExtractor`): 12 general-state (polynomials of
`clients_left` × `time`, four earliness bins, a mean-earliness gap) and 7
state-action (projected earliness/delay/overtime/future-delay costs). `update_W`
walks each episode backward accumulating the realised return `U_t`, subtracts a
sunk-cost baseline, and takes one constant-step SGD step per decision epoch.

The candidate set (`_select_vehicle_possible_actions`) is itself heavy hand
engineering: the `k` nearest unvisited Clients by *static* travel time, minus
the other vehicles' targets, plus the depot when the return leg breaches the
horizon, plus up to two "delayed" Clients from a bespoke classifier, plus
degenerate branches gated on the literals `350` and `310`.

### What does not exist

**There is no baseline number.** The full Chengdu training run has never been
executed against the repaired simulator — `.scratch/simulator-correctness/spec.md`
declares it "the first experiment of the repaired lab, not part of repairing
it". Ticket 01 of this effort *is* that run. Everything downstream depends on
it.

### What the research says, and why we are doing this anyway

`docs/research/rl-methodology-for-stdvrp.md` reaches a conclusion that is
uncomfortable for this effort and must be recorded here rather than discovered
later:

> **"The algorithm is not the binding constraint; the observation set is."**
> Not one of the 19 features, and not one of the travel times used to build the
> candidate action set, can see the time-dependent stochastic velocities or the
> non-recurring congestion the simulator injects (F1).

It also recommends **against** porting attention/pointer constructive policies
(AM, POMO), citing SVRPBench's finding that they degrade >20% under
distributional shift while classical heuristics stay robust. That recommendation
is about *constructive* policies that rebuild a tour from unit-square Euclidean
coordinates — a different object from what this effort builds, which keeps the
simulator, the decision structure and the Monte Carlo estimator, and changes
only how `Q` is computed. The distinction is real, but the warning is not
dismissed: it is why the acceptance contract below is adversarial about the
comparison rather than optimistic about the architecture.

The research note's ranked recommendation #5 is "neural VFA over enriched
features", with Chen/Ulmer/Thomas reporting **up to +22%** over a tuned
parametric policy on a comparable SDVRP. That is this effort, with one
difference: the features are not enriched, they are *removed*.

## Decisions (from the grilling session)

| # | Decision | Choice |
|---|---|---|
| 1 | What the network observes | **Raw entity tokens**: one per pending Client, one per vehicle, one global. ~~Hard rule: a token never carries a cost formula, a polynomial, or a bin~~ — **amended 2026-08-01 (ticket 08): the per-(client, vehicle) arc tokens carry the four projected components of the simulator's own cost function**; the baseline's *state* feature engineering (polynomials, bins, normalizers, literals) stays out. See the amendment below |
| 2 | Congestion awareness | **Out of the blind arm.** The transformer is exactly as congestion-blind as the baseline, so the comparison measures the approximator — not information one side was handed and the other was not |
| 3 | **The observability rule** | The Policy reads **only** `State` and the static `EpisodeGeometry`. Never `EpisodeVelocities`, never `congested_arcs`, never `TravelTimeModel` at `tau`. The only velocity that exists for the network is `state.observed_velocity` — what its own vehicles measured while traversing real arcs |
| 4 | Action set | ~~**Every pending Client + the depot**, feasibility mask only. `_select_vehicle_possible_actions`, `_classify_shortest_distance_clients`, `delayed_clients`, the `350`/`310` literals and `number_actions_test` all die for this Policy~~ — **reversed 2026-08-02 (tickets 13/14): the action set is the baseline's own, identically, at `m + 2`.** See the amendment below |
| 5 | Learning rule | **`Q(s,a)` + Monte Carlo return.** Same target `U_t − acquired_cost`, undiscounted, same statistical object. ~~Only `X · W` → `net(tokens)` and constant-step SGD → Adam~~ — **amended 2026-08-02 (tickets 15/16): `Q = c(s,a) + W·φ`, with `c` outside the parameters and `W` solved by accumulated ridge.** See the amendment below |
| 6 | Multi-vehicle structure | **Sequential one-agent-at-a-time**: one encoder pass per decision epoch, then `m` cheap head passes. `claimed` enters at the head, not the encoder, so one encode serves the whole sweep |
| 7 | torch | **Optional extra `neural`**, lazily imported, `device` in config. A ticket measures CPU vs CUDA rather than assuming. ~~CPU by default~~ — **amended 2026-07-31 (ticket 12): the default is `"auto"`**, resolved once per run and pinned in that run's record. See below |
| 8 | Training seam | **`TrainablePolicy` protocol**; the loop stays in `Model`. `TrainingSnapshot` gains `vehicle_completing_service` |
| 9 | Learning unit | ~~**One batch per episode**, K shuffled minibatch passes, then discarded. Strictly on-policy~~ — **amended 2026-08-02 (ticket 16): normal equations accumulated *across* Episodes with exponential forgetting.** The per-Episode fit was the effort's largest measured defect. Still Monte Carlo policy evaluation; no longer a per-Episode fit |
| 10 | "It learns" gate | **Three parts**: null model, ≥3 init seeds, calibration — **amended 2026-08-02 (ticket 17)**: the ≥5% threshold **stays**, calibration moves onto the residual, `r` diagnostic added. See "Gate A′" below |
| 11 | Comparison budget | **Each policy at its own best, budget disclosed.** The baseline runs at 100/500/2000 episodes × its full action-count sweep and is compared at its *best* cell. **Clarified 2026-08-02**: the action-count axis is now *shared* rather than each-at-its-own — this Policy is fixed at `m + 2` — so Gate B reports both the best-cell number and the like-for-like one |
| 12 | Stopping | **Train to convergence**: patience → `lr × 0.3` → three reductions → stop. The hard cap is a safety net, not the budget; if it fires the run is reported as **"did not converge"** |
| 13 | Live comparison | **Reference card + paired live prints.** The scalar `static_policy_mean_cost` retires in favour of the baseline's **per-seed** cost vector |
| 14 | If it loses | ~~**Pre-committed two-arm escalation**~~ — **amended 2026-08-02: a three-rung ladder**, the first two unconditional (both encoder arms, so the representation decomposition always exists), the third conditional. Protocol frozen before measuring, closing ADR with the negative result. The Policy ships available but not default |

### Amendment to decision 7 (2026-07-31, ticket 12)

"CPU by default" rested on one stated risk: `EpisodePool` workers each opening
their own CUDA context on an 8 GB laptop GPU (ticket 03). **The neural path
never opens a worker** — ticket 07 runs its evaluation blocks serially and
`EpisodePool` appears nowhere in `trainer.py` — so the risk does not apply to
this effort, and the default was resting on an excluded interaction.

Measured, the device is not a preference but a **precondition**: at ~11.4 s per
training episode on CPU (derived: ~9.6 s network + 1.82 s simulator), 24 h buys
~6 600 episodes, so the frozen 10 000-episode cap is unreachable, the clock cap
always fires first, and every run is recorded *"did not converge"* — three init
seeds, three days, nothing concluded. At ~3.4 s/ep on CUDA the episode cap is
reachable in ~11 h.

`device` therefore defaults to **`"auto"`** (`cuda` if available, else `cpu`).
Because CPU and CUDA do not agree bit for bit, `auto` moves the pinning of a
result out of the config and into the run's own record: it resolves **once**,
is printed, is written into the checkpoint and the results, and a cross-device
resume is an error. An explicit `device: cuda` with no GPU fails loudly rather
than degrading silently. The test suite pins `"cpu"` explicitly so it stays
identical on CI and on GPU machines.

**No frozen acceptance number changes.** Seeds, metric, test, minimum effect
size and stopping rule are untouched; the safety cap keeps both its halves. The
only difference is *which half binds*: the clock on CPU, the episode count on
CUDA.

### Correction to the amendment above (2026-07-31, ticket 12's own measurement)

The "~11.4 s CPU / ~3.4 s CUDA" figures above were a **derivation** from
ticket 03's stub network (no tokenizer, no real per-sample `learn()` loop, no
`torch.use_deterministic_algorithms`), not a measurement of the real thing.
Ticket 12 wired the device end to end and measured the real network against
the real Chengdu dataset for the first time — see "Compute budget" below for
the full table. The result does not match the projection: **CUDA is not
faster than CPU on the reference laptop**, measured against the real
tokenizer/network, on either the acting or the learning path. A likely major
cause, isolated by ablation: `resolve_device("cuda")` correctly enables
`torch.use_deterministic_algorithms(True)` (required by ticket 05's
bit-identical contract) — a cost ticket 03's stub never paid, measured here at
~33% on this hardware. A second likely contributor: the real `learn()` step
re-tokenizes and re-encodes every sample individually rather than as one
batched op (documented as "not maximally efficient" in
`transformer_policy.py`'s own module docstring), favouring CPU's lack of
per-call transfer/launch overhead over CUDA's. Not conclusively separated
from this laptop's own known run-to-run timing drift (ticket 03's Comments)
— this measurement followed roughly 15 minutes of sustained CUDA test
activity in the same process, so thermal/power throttling cannot be ruled
out; recorded as found rather than re-measured under controlled conditions.

**This means the reachability argument above does not hold as stated**: the
measurement does not show CUDA reaching the 10 000-episode cap any faster
than CPU. `device` stays `"auto"` anyway — not because it is faster here (it
measurably is not), but because an `"auto"` that resolves to whichever device
is genuinely faster locally costs nothing, the answer is machine-specific (a
different GPU, a batched `learn()`, or the mixed-precision/TF32 work this
effort declined to do could change it), and the correctness machinery this
ticket built (loud failure on an explicit `cuda` request with no GPU,
cross-device resume refusal, the resolved device recorded in the checkpoint
and results) has value independent of which device turns out faster. Tickets
08/09's real runs should not assume CUDA buys episode-cap headroom; budget
for either device costing close to the frozen safety cap, per the "Compute
budget" table below.

### Amendment to decision 1 (2026-08-01, ticket 08)

The hard rule "a token never carries a cost formula, a polynomial or a bin"
mixed two constraints that this effort's own Gate A run showed must be
separated: **observability** (decision 3, ADR-0006 — what of the *world* the
Policy may see) and **purity** (decision 1 — how much arithmetic a token may
carry). The first is untouched by this amendment. The second is amended on
this evidence:

**The evidence.** Ticket 08's first real-dataset arm (its Comments,
2026-08-01): the raw-facts network *does* learn — −4.7 % against the null on
eight `test_seeds` at its best block — and then diverges to 4× worse by
episode 100, unrecovered through two lr cuts. The diagnosis: `Q(s, a)`
decomposes as `V(s) + A(s, a)`, the Monte Carlo return's variance is almost
entirely `V(s)`, so least squares spends the network on fitting `V` while
`A` — the only quantity the argmin reads — is left as a residual to be
rediscovered from noisy returns. On the mini fixture's ~16-candidate argmin
that residual is learnable; on Chengdu's ~151 candidates, the minimum of 151
noisy, unanchored extrapolations is dominated by the noise (the ticket's
"likely cause is the candidate count").

**The change.** Each `(client, vehicle)` arc token — the action-conditional
pathway — now carries, next to `[minutes, path_length]`, the four **projected
components of the simulator's own cost function** for that assignment:
`earliness_cost`, `delay_cost`, `future_delay`, `overtime_cost`, as per-pair
marginals mirroring `FeatureExtractor.candidate_features`' formulas (minus its
closest-client multiplicity classifier, hand engineering ADR-0007 already
retired). The synthetic depot candidate gets the same six-field block
(`Tokens.depot_arc_tokens`), so its row is built by the identical pathway.
With them, `A(s, a)` is expressible as a near-linear readout of the inputs
from the first gradient step — and `Q` can no longer be arbitrarily abrupt
between neighbouring candidates, which is what the 151-candidate argmin was
amplifying.

**Why this is the literature's own recommendation.** The research note this
spec cites ranks "neural VFA over **enriched** features" as recommendation #5
(Chen/Ulmer/Thomas, up to +22 % over a tuned parametric policy), and this spec
recorded the tension explicitly at the outset: "the features are not enriched,
they are *removed*". The amendment adopts the enrichment for the **cost half
only**: the state representation stays raw (no polynomials, no bins, no
`late_count / 13`, no `350`/`310` literals, no multiplicity classifier — the
transformer's job is still to learn the state), and what is added is only the
arithmetic of the objective the run is scored by.

**Why the comparison stays fair.** The linear baseline's seven state-action
features already carry these exact four costs (`FeatureExtractor`,
`X[:, 15..18]`); handing them to the network *levels* the two Policies' input
sets rather than tilting them. Observability is untouched: every cost is
computed from `tau`, the time windows and `EpisodeGeometry.average_minutes`
(plus the simulator's hardcoded rate constants, which are configuration, not
observation — ADR-0006's clarification), `tokenize`'s signature is the same
five arguments, and the same structural test pins it. The untrained null is
**bit-identical**: the warm start's `arc_embed` row 0 reads the minutes input
only, so at init `Q` equals bare `minutes / horizon_length` whatever the cost
fields hold — Gate A's null model did not move.

### Amendment: the warm start, and the null model it defines (2026-08-01, ticket 08)

The paragraph above ends "Gate A's null model did not move". It moves now, and
deliberately — **upward**.

`arc_embed` row 0 is the whole of `Q` at initialization, and it was reading
`minutes` alone while the four projected cost components sat in the same token,
weighted zero. Ticket 08 measured what that costs: pricing the leg instead
(`Q = (minutes + earliness + delay + overtime) / horizon_length`, one
minute-equivalent currency, no free parameter) takes the untrained network from
**5484 to 3693** over all 50 `evaluation_seeds` — −32.7%, winning 47/50 seeds,
Wilcoxon `p = 2.5e-14`. The *best block of 650 training episodes* of the
`minutes` network on the same seed set was 4150. The initialization beats the
training.

Selected as `neural_warm_start` (`minutes` | `cost`), defaulting to `minutes`;
`experiments/chengdu/config.yaml` runs `cost`.

**What this does to Gate A, stated plainly rather than buried.** The frozen
parameters table says "the null is nearest-neighbour, not random", and the
prose calls the untrained network "a respectable rival". Under `cost` the null
is no longer nearest-neighbour — it is a cost-greedy dispatcher, and a
considerably stronger rival. Two things make that legitimate rather than a
protocol breach:

1. The architecture row of the same table is explicitly *not* frozen — "a
   starting point, tuned on the evaluation seeds only" — and this was chosen on
   `evaluation_seeds`, never on `test_seeds`.
2. It moves the bar **up**. The anti-p-hacking clause exists to stop a
   criterion being rewritten into one the result can pass; a null model that
   makes "≥ 5% better than untrained" harder to reach is the opposite of that.

The obligation it creates: **report the null alongside the trained number,
always, and name which warm start produced it.** A Gate A pass against the 5484
null and a Gate A pass against the 3693 null are not the same claim.

## The redesign (2026-08-02, after ticket 08) — decisions 4, 5, 9, 10, 11, 14

Ticket 08 closed **incomplete** (one of three arms; its "it learns" result
stands at `n = 1`) and its diagnosis reversed the direction of the effort. This
section is the amendment; tickets 13-17 execute it.

### What ticket 08 established, and the one defect it never named

Three claims that the prose above conflates and the measurements separate:

| claim | verdict |
|---|---|
| it learns (beats its own null) | **yes**, +15.49% p=6.21e-05 — but **n = 1** |
| training adds value over a good initialization | **no** — untrained `cost` 3811.28 beats trained 4423.73 by 13.8% |
| it beats the linear baseline | **no** — 3384.82 against 3811.28 |

The unexamined defect is the **unit of estimation**. `learn` receives one
Episode's ~400 epochs, runs 4 passes in minibatches of 32, and discards the
batch — there is no buffer across Episodes. Within one Episode `U_t` is a
*suffix sum*, monotone in `t`, and the global token carries `tau_episode` and
`clients_not_visited`, so **595k parameters fit `U_t` almost perfectly by
reading the clock and the action carries no incremental explanatory power.**
The action→return signal exists only *across* Episodes.

That single fact reconciles everything ticket 08 measured piecewise: `loss` at
`1e-4` beside a collapsing policy; why 19 linear weights survive the identical
target (they cannot overfit one Episode, and `W` is carried across all of
them); and why its §4's four attempts share one signature — dueling, the Huber
knee, the level gain and the `cost` warm start each make the optimizer *more
effective*, and a more effective optimizer overfits one Episode faster. Helpful
from a bad start, damaging from a good one, every time.

### Decision 4, reversed: the action set is the baseline's, identically

The action count is worth **12.68%** to the linear baseline — `m+40` against
`m+2` over the 50 `evaluation_seeds`, cheaper on 36/50, Wilcoxon p = 8.24e-05
(`results/baseline_null_50.py`). On `test_seeds` the same axis shows 2.1%
(3458.4 → 3384.8, ticket 01's sweep), so the *effect* reproduces and its
*magnitude* does not transfer between seed sets. Together with ticket 08's own
"the likely cause is the candidate count", that is enough: ADR-0007 discarded
`_select_vehicle_possible_actions` by argument, and the argument does not hold.

The second reason is fairness rather than performance. The baseline's seven
state-action features already carry the four projected costs, and spec.md's
decision-1 amendment handed those to the network on exactly that ground —
*levelling the two Policies' input sets rather than tilting them.* The action
set is the remaining un-levelled axis, and while it differs a Gate B result
cannot be attributed to the approximator at all.

It comes back **whole** (three branches, the delayed-Client classifier, the
`350`/`310` literals, the `list(set(...))` dedup) through one shared
definition, at `m + 2` in training *and* evaluation. **ADR-0011 reverses
ADR-0007.**

**A hypothesis this section carried on 2026-08-02 and lost the same day**, kept
because the correction is the useful part. `MonteCarloPolicy._create_W` is
`np.zeros(19)`, so it was argued that the baseline at `W = 0` must be "go to
the nearest allowed Client" — `X · W = 0` for every candidate, `argmin` returns
index 0, and `_closest_allowed_clients` orders nearest-first — and therefore
that most of the gap was the candidate heuristic rather than the learned
weights. **Measured, `W = 0` scores 30 791.43.** Branch 3 runs
`possible_actions = list(set(possible_actions))` *after* the nearest-first sort,
and node ids are arbitrary ints, so the dedup returns hash-table order: `W = 0`
picks an **arbitrary** feasible Client, not the nearest. The preserved quirk
eats the tie-break.

Two consequences, both recorded rather than quietly dropped: the linear
baseline **has no cheap myopic null** (`W = 0` is a degenerate policy, the same
trap ticket 08 fell into when it measured its own null at 81 701), and the
candidate-set-versus-learned-ranking decomposition therefore **cannot be had
before tickets 13-14** — comparing baseline@`m+2` with cost-greedy@151 changes
the action set and the ranking rule at once. Ticket 14's 2×2 is the first place
that separation exists.

### Decisions 5 and 9: the residual, and the estimator

```
Q(s, v, a) = c(s, v, a)  +  W · φ(s, v, a)        W = 0 at init
             ↑ tokenizer, not a Parameter,
               structurally unreachable by the gradient

A ← γ·A + Σ Φ Φᵀ ;  b ← γ·b + Σ Φ·ỹ ;  W ← (A + λI)⁻¹ b
515 parameters × ~20 000 samples          (was: 595k × ~400)
```

`W = 0` is the cost-greedy policy exactly, so training can only add. Aborted
Episodes are excluded from the accumulator (F10: `40000 − 200·served` entering
~400 targets of one Episode would weigh like ~12 000 ordinary epochs), with the
exclusion rate logged and a rising rate treated as a stop signal.

This is research note recommendation **#3** — *"removes the learning rate, the
feature-scaling pathology and most of the variance"* — arriving inside the
effort that had skipped straight to **#5**. `neural_warm_start`,
`neural_level_gain` and `neural_huber_delta` become no-ops on this path and are
documented rather than deleted.

### Decision 10: Gate A′

The untrained network is now a genuinely strong dispatcher, so the gate's
question sharpens into the one that matters: **does training add value on top
of a good initialization?**

- **Part 1's threshold stays at ≥ 5%, and stays decoupled from the baseline.**
  It was briefly amended to the gap against 3384.82 and amended back the same
  day, on measurement: that gap is not a stable quantity. `best_w` was
  *selected* on `evaluation_seeds` and reads 2168.39 there against 3384.82 on
  `test_seeds` (×1.56), while a policy with no selected parameters reads 3693.23
  and 3811.28 (×1.03) — F12's winner's curse, measured, and **policy-dependent**,
  so any formula mixing the two seed sets is incoherent. The same gap computes
  to 41.3% on selection data and 11.2% on verdict data. Gate A′ therefore asks
  its own self-contained question — *does training improve on its own null* —
  and Gate B alone decides whether that is enough.
- **Part 3 is redefined** to `ρ(W·φ, ỹ)` — the learned term against the
  residual it is actually regressed onto. `ρ(Q, U_t)` would pass at `W = 0`
  with no parameter having moved: a guaranteed false PASS.
- **A diagnostic is added**, reported every block:
  `r = sd_candidates(W·φ) / sd_candidates(c)`. `r ≈ 0` means the ranking was
  never touched (or `λ` shrank it away); `r ≫ 1` means the base is being
  overwritten — ticket 08's failure mode returning.

### Decisions 11 and 14: the ladder

```
rung 1  blind × frozen encoder    ┐ unconditional — delta(2) − delta(1) is the
rung 2  blind × trained encoder   ┘ value of representation learning
        → Gate B with the better of the two
rung 3  fleet-memory × (the better)   conditional on both losing
        → if it loses: ADR-0009, negative result, close. No rung 4.
```

Both encoder arms run unconditionally: if `trained` only ran when `frozen`
lost, and `frozen` won, the effort's original thesis would never get its
number. Gate B's verdict is taken against **3384.82** — the baseline's best
cell, unweakened — with the like-for-like **3458.4** (`m+2`, identical action
set) reported beside it.

### What did not move

The **observability rule** (decision 3, ADR-0006) — `c` is built from `tau`,
the time windows and `EpisodeGeometry.average_minutes`, all already admitted;
`tokenize`'s five arguments and its structural test are untouched. The **seeds,
the metric, the test, Gate B's 3%, and the stopping rule.** And the discipline
that produced this amendment: every number above is on `evaluation_seeds` or
already-published `test_seeds` measurements, and no threshold was set from the
verdict set.

## The observability rule, precisely

This is the load-bearing constraint of the effort and it is written as an
executable test, not as a docstring (ticket 04).

> A decision may be computed **only** from: this Episode's `State`
> (`tau_episode`, `clients_not_visited`, `last_node_reached`, `vehicle_standing`,
> `vehicle_completing_service`, `observed_velocity`), the Episode's time
> windows, the static `EpisodeGeometry` matrices, and configuration clocks.
>
> Explicitly forbidden: `EpisodeVelocities`, `congested_arcs`,
> `TravelTimeModel` evaluated at `tau`, `FleetRoutes`, and any velocity the
> fleet has not itself observed.

**Boundary case, decided:** `EpisodeGeometry.average_minutes` — the historical
mean travel time precomputed into the CSV — **is permitted**. It is an offline
prior, not an observation of this Episode, and it is the identical object the
linear baseline reads. Without it the network has no notion of distance at all.

**The only admissible version of a congestion-aware arm** (ticket 10) is the
**fleet's shared observation memory**: what this Episode's own vehicles have
already measured, pooled across the fleet. A dispatcher may aggregate its own
vehicles' reports; it may not read the world's velocity field.

## Why the paired comparison is valid (verified, not assumed)

Both acceptance gates are paired per seed. That is only legitimate if a seed
pins the same *problem* under both policies. Verified against the code:

- **Demand** comes from `ClientGenerator.generate(seed)` — no policy input.
- **Congestion** rolls in `ArcProbabilityCongestionGenerator.generate`, which
  consumes **exactly one uniform per arc key** (whether or not an event fires)
  plus two per triggered event, and reads nothing from the fleet. The rolls
  happen at deterministic clock values — every `tau` where
  `(tau + 178) % max_congestion_duration == 0`, reached by
  `_decision_epoch_begins` advancing `next_decision_tau` by exactly 2 per call.
  `_compose_congestion` reads the event book but consumes no randomness, and
  treats an expired entry identically whether or not it was purged — so the
  policy-dependent purge timing does not change the resulting book.
- **Velocities** *do* diverge: different policies traverse different arcs, so
  `velocity_rng` is consumed in a different order.

**Therefore: same seed ⇒ same demand and same congestion schedule under any
policy, on the shared prefix.** Only the velocity realisation differs. Pairing
is sound and the variance reduction is real.

## Acceptance contract

Three gates, three different questions. None substitutes for another.

### Landing gate (hard — a ticket does not land without it)

Tests pass, `mypy` clean, `ruff` clean, and — for every ticket — the predicted
self-golden diff is met. **Every ticket in this effort predicts an exactly-zero
self-golden diff**, because nothing here touches the linear baseline's execution
path. That is a strong claim, not a formality: if a protocol extraction moves a
single float, it was not an extraction.

### Gate A′ — "does training add anything" (hard; ticket 17)

~~Gate A, ticket 08~~ — **superseded 2026-08-02.** Ticket 08's version is
preserved in its own file; it closed incomplete at `n = 1`. All three parts, on
the held-out `test_seeds` (100..153), never on the evaluation seeds:

| Part | Test | Threshold |
|---|---|---|
| **Null model** | Trained vs. `W = 0` (the myopic base), paired per seed | Wilcoxon **p < 0.05** and **≥ 5%** mean cost reduction — unchanged, and deliberately not coupled to the baseline |
| **Reproducibility** | ≥ **3** independent init seeds, **per arm** | Improvement reported as mean ± sd, not a single number |
| **Calibration** | Spearman ρ(`W·φ`, `ỹ`) on held-out episodes — the learned term against the residual it is regressed onto | **≥ 0.5**, against ≈ 0 at `W = 0` |
| *(diagnostic)* | `r = sd_candidates(W·φ) / sd_candidates(c)`, every block | reported, not gated: `≈ 0` the ranking was never touched; `≫ 1` the base is being overwritten |

The null model is **not a random network and no longer a nearest-Client one**:
under the residual decomposition `W = 0` *is* the myopic base, a cost-greedy
dispatcher over the baseline's own candidate set. "It learns" therefore means
"it improves on the best policy we already had for free", which is a strictly
harder claim than the one ticket 08 tested.

Spearman, not Pearson: the episode-cost distribution has a brutal right tail
(F10). Calibration is the part that cannot be faked by a lucky run — **provided
it is computed on the residual.** Against `Q` and `U_t` it passes at `W = 0`.

### Gate B — "it wins" (the objective; tickets 09/10)

Transformer vs. the linear baseline's **best cell** (best training budget × best
`test_action_count` = budget 100, `m + 40`, **3384.82**), paired over
`test_seeds`:

> Wilcoxon signed-rank **p < 0.05** **and** **≥ 3%** mean total-cost reduction.

Below 3% paired over 50 seeds is real but not interesting.

**Reported beside it, never instead of it:** the like-for-like number against
the baseline at `m + 2` (**3458.4**) — identical action set, identical inputs,
only the approximator differing. This Policy is confined to `m + 2` by ticket
14, so the two numbers answer different questions and both get written. "Wins
like-for-like, loses against the best cell" is a real outcome, and it is
reported in exactly those words if it happens.

**And the contender table is complete or it is not a verdict:** the untrained
myopic base is a contender, because ticket 08 closed with the best *policy* in
the effort being an untrained one.

### The anti-p-hacking clause

Copied deliberately from `simulator-correctness` decision 10:

> **The protocol — seeds, metric, test, minimum effect size, stopping rule — is
> frozen in this spec before the first number is measured.** A criterion that
> can be rewritten after seeing the result was never a criterion.

If Gate B fails on arm 1, arm 2 (fleet shared observation memory) runs. If arm 2
also fails, the effort closes with **ADR-0009 recording the negative result**,
the Policy ships available-but-not-default, and the research note's central
claim — that the observation set is the binding constraint — is recorded as
supported by measurement.

## Frozen parameters

| Parameter | Value | Rationale |
|---|---|---|
| Verdict seed set | `test_seeds` 100..153 (50 seeds, per-seed fleet table) | Disjoint from training (1000+) and from checkpoint/hyperparameter selection (100000+) |
| Selection & tuning set | `evaluation_seeds` 100000..100049 | Checkpoint selection *and* hyperparameter search live here — **never on the verdict set** |
| Gate A′ effect | ≥ 5% vs. null, p < 0.05 | **Unchanged.** Briefly amended to the gap against the baseline on 2026-08-02 and amended back the same day: that gap reads 41.3% on `evaluation_seeds` and 11.2% on `test_seeds`, because the winner's curse on a *selected* `W` is policy-dependent (measured, ×1.56 against ×1.03). A threshold that cannot be computed on one seed set is not a threshold. What did change: the null is now the myopic base, a far stronger rival than the nearest-Client placeholder this 5% was written against |
| Gate A′ calibration | Spearman ρ(`W·φ`, `ỹ`) ≥ 0.5 | **Amended 2026-08-02**: on the residual. Against `Q`/`U_t` it passes at `W = 0` |
| Gate B effect | ≥ 3% vs. baseline best cell (3384.82), p < 0.05 | Unchanged. The like-for-like 3458.4 is reported beside it, never in place of it |
| Init seeds | 3 | |
| Patience | 5 evaluation blocks without improvement → `lr × 0.3`; 3 reductions → converged | |
| Safety cap | 10 000 episodes **or** 24 h per run | Not the budget. If it fires, the run is reported **"did not converge"** and never presented as a clean result. Unchanged by ticket 12. Ticket 12's own projection that CUDA moves *which half binds* did not survive its own real-network measurement (see "Compute budget" below) — budget for the clock binding on **either** device, not just CPU |
| Evaluation cadence | scales with run length (~every 50 episodes) | At `test_frequency: 10`, a 2000-episode run costs 10 000 evaluation episodes — more than the training itself |
| Starting architecture | d=128, 3 layers, 4 heads (measured 594,945 params at `dim_feedforward=4*d_model` — see "Compute budget" below, not the ~200k first guessed here) | A starting point, tuned on the evaluation seeds only |

## Compute budget (ticket 03's stub measurement, corrected by ticket 12's real one below)

This effort's planning used a napkin calculation of ~10 s/ep training,
~3.3 s/ep evaluation for PyTorch dispatch overhead. Ticket 03 replaced that
guess with a real measurement: a stub encoder at this table's exact shape
(d=128, 3 layers, 4 heads, dim_feedforward=512, seq_len=159 = 150 client
tokens + 8 vehicle + 1 global), timed on the reference hardware (RTX 4060
Laptop 8 GB / Ryzen 7 8845HS / 32 GB RAM), both the acting path (400
sequential batch=1 forwards/episode) and the learning path (K=4 shuffled
minibatch passes over one episode's batch) — `scripts/benchmark_neural_stub.py`,
full table and analysis in the ticket's Comments.

| | CPU | CUDA |
|---|---|---|
| Training episode (acting + learning) | ~9.5-9.7 s/ep | ~1.5-1.7 s/ep |
| Evaluation episode (acting only) | ~1.0-1.08 s/ep | ~0.75-0.89 s/ep |

**The napkin estimate's training number (~10 s/ep) held up reasonably well
against CPU** (~9.6 s/ep measured); **its evaluation number (~3.3 s/ep) was
roughly 3x too high** (~1.0 s/ep measured, acting path only, no learning
step). **The napkin's own qualitative prediction did not hold**: it expected
CUDA to help only the learning step (kernel-launch overhead dominating the
latency-bound acting path); measured, CUDA won *both* paths on this hardware.
~~`device` still defaults to `"cpu"` (decision 7 is structural, not overturned
by a single-process measurement) — `EpisodePool` worker processes each
wanting their own CUDA context on one 8 GB GPU is a real, untested
interaction the ticket documents but does not measure.~~ **Superseded
2026-07-31 (ticket 12):** the default is `"auto"`. The `EpisodePool`
interaction is not untested-and-risky but *absent* — ticket 07's neural path
is single-process, so no worker ever asks for a CUDA context. See the
amendment to decision 7 above.

These were stub-network numbers, not the real tokenizer/network (tickets
04-05) or the real per-episode simulator cost this effort's other efforts
already measured (`simulation-performance`: ~2.4 s/ep train, ~1.3 s/ep eval on
the real dataset; its closing ticket 10 measured 1.817 s/ep train and
0.802 s/ep eval on the real Chengdu world) — the two costs are additive once
the real Policy exists, not substitutes for one another.

### The real measurement (ticket 12, 2026-07-31)

`scripts/benchmark_neural_real.py` replaces the stub above with the real
thing: `run_neural_training_episode` / `run_neural_evaluation_episode`
(tickets 04-07, the committed d=128/3-layer/4-head architecture) against the
real Chengdu dataset, on the reference hardware. 12 timed episodes per
path per device, one untimed warmup episode first.

Per-episode cost varies far more than the stub anticipated, because it tracks
each episode's **decision-epoch count** — itself highly variable at this
early, largely-untrained stage of the myopic warm start (1 to 409 decisions
across the 12 training episodes sampled; the stub assumed a fixed ~400). The
decision-weighted **ms/decision** figure below is therefore the more
comparable unit against the stub's own per-decision numbers; s/ep is also
given because it is what the safety cap actually counts against.

| | CPU | CUDA |
|---|---|---|
| Training (acting + learning), decision-weighted | ~173 ms/decision | ~191 ms/decision |
| Training (acting + learning), per-episode mean over the 12 sampled (min–max) | 25.8 s/ep (0.16–58.2) | 35.9 s/ep (0.18–76.6) |
| Evaluation (acting only), decision-weighted | ~5.2 ms/decision | ~6.6 ms/decision |
| Evaluation (acting only), per-episode mean over the 12 sampled (min–max) | 1.43 s/ep (0.75–2.06) | 1.97 s/ep (1.35–2.70) |

**CUDA is not faster than CPU here — mildly slower on both paths, the
opposite of the stub's finding above.** See the correction appended to the
decision-7 amendment for the likely causes (the `use_deterministic_algorithms`
cost the stub never paid, verified by ablation at ~33%; `learn()`'s per-sample
non-batched re-tokenize-and-re-encode loop) and the caveat about this
machine's own timing drift. Recorded as found, per this effort's own
discipline (ticket 03's Comments: "not forced into the anticipated shape") —
not smoothed over because it contradicts the ticket that measured it.

## The live report

The scalar `static_policy_mean_cost` (a hardcoded YAML number feeding one red
line on the training plot) retires. It is replaced by the **reference card**:
the baseline's frozen **per-seed** cost vector, which makes every live
comparison paired.

Every evaluation block prints:

```
[ep  348] train seed 1348   cost 2691.3   loss 0.387   lr 3.0e-4   8.9s
[ep  349] train seed 1349   cost 2402.8   loss 0.371   lr 3.0e-4   9.1s
[ep  350] train seed 1350   cost 2555.0   loss 0.369   lr 3.0e-4   8.7s
--------------------------------------------------------------------------
  eval @350   mean 2087.4  |  MC ref 2314.9  |  delta -227.5  (-9.8%)
              wins on 37/50 seeds    Wilcoxon p=0.003
              best so far: -9.8% @ep350     no improvement: 0/5 blocks
--------------------------------------------------------------------------
```

With a positive delta and `wins on 12/50`, you know by the second block that
nothing is happening — instead of finding out after six hours.

## Tickets

Critical path: 01 → 02 → 03 → 04 → 05 → 06 → 07 → **12** → ~~08~~ →
**13 → 14 → 15 → 16 → 17** → 09. Ticket 10 is conditional on 09's verdict; 11
closes. Ticket 12 carries a higher number than the tickets it precedes for a
reason the effort keeps: renumbering would orphan the ticket numbers already in
the commit history — 13-17 follow the same rule.

| # | Ticket | Blocked by | Predicted self-golden diff |
|---|---|---|---|
| 01 | Baseline reference card | — | **zero** |
| 02 | The `TrainablePolicy` seam | — | **exactly zero** |
| 03 | torch as an optional extra + the real cost measurement | — | **zero** |
| 04 | The tokenizer, and the observability rule as a test | 03 | **zero** |
| 05 | The network, the Q head and the myopic warm start | 04 | **zero** |
| 06 | The Policy: decide, decide_train, learn | 02, 05 | **zero** |
| 07 | Trainer: live paired report, convergence stopping, checkpoints | 01, 06 | **zero** |
| 08 | ~~Gate A — does it learn~~ **closed incomplete** (`n = 1`) | 07, **12**, `simulator-correctness`/11 | **zero** |
| **13** | The action set becomes shared code — **the only ticket that edits `monte_carlo.py`** | — | **exactly zero** |
| **14** | The Policy adopts the identical action set (reverses ADR-0007) | 13 | **zero** |
| **15** | The myopic base leaves the network | 14 | **zero** |
| **16** | The estimator: accumulated least squares | 15 | **zero** |
| **17** | Gate A′ — does training add anything | 16 | **zero** |
| 09 | Gate B — the verdict | 17 | **zero** |
| 10 | Rung 3 — fleet shared observation memory (conditional) | 09 | **zero** |
| 11 | Close: ADRs, CONTEXT.md, results | 09 or 10 | **zero** |
| 12 | `device: cuda` end to end (runs before 08 — see the critical path) | 07 | **zero** |

## ADRs this effort writes

Each inside its ticket, when the decision is executed — never up front.

- **ADR-0006** (ticket 04) — *What the Policy is allowed to see.* The
  observability rule, why the live traffic feed was rejected, why
  `average_minutes` is permitted, and what the only admissible congestion-aware
  arm is.
- **ADR-0007** (ticket 06) — *The action set is feasibility, not heuristic.*
  Why `_select_vehicle_possible_actions` and its `350`/`310` literals do not
  apply to this Policy, why the B11 no-double-booking invariant survives as a
  mask, and why `number_actions_test` has no meaning here.
  **Reversed 2026-08-02 by ADR-0011** — measured wrong, not argued wrong: see
  the redesign section above.
- **ADR-0010** (ticket 11) — *The approximator is a residual over a frozen
  myopic base, fitted by accumulated least squares.* Why `c(s, a)` is
  structurally unreachable by the gradient rather than merely frozen by config,
  the four measured attempts whose shared signature forced it, and the two
  rejected alternatives with their reasons — including that the Huber knee was
  rejected for an Adam-specific reason the new estimator removes, so it is
  *available again*, not settled.
- **ADR-0011** (ticket 11) — *The action set is shared, not owned.* Reverses
  ADR-0007 on ticket 01's own sweep: `W = 0` is nearest-allowed-Client, budget
  100 beats 500 and 2000 in every cell, and two candidates already beat a
  cost-greedy argmin over ~151. What came back unclean, and what it retired.
- **ADR-0009** (ticket 11) — *The reference card and the paired protocol.*
  Retires `static_policy_mean_cost`; records the verified independence argument
  above; records the verdict, including a negative one. Also records, in one
  clause, that a run's device is resolved once and pinned in the run's own
  record (ticket 12) — the device gets no ADR of its own, being operational
  rather than architectural.
  **Renumbered 2026-07-31 from ADR-0008**, which `simulator-correctness`/11
  took first (`docs/adr/0008-an-action-must-be-executable.md`).

## CONTEXT.md terms

Added in their tickets (04, 07, 11), as a glossary — never as spec content:

- **Approximator** — what maps an observation to `Q`. The variation point
  *inside* Policy: linear weights or a neural network, same decision rule.
- **Reference card** — a completed Policy's frozen per-seed costs, the fixed
  opponent every later run is compared against.
- **Myopic base** *(added 2026-08-02)* — the projected cost of one assignment,
  added to `Q` outside the Approximator's parameters. Explicitly **not** a warm
  start and **not** Powell's post-decision state.
- **Residual approximator** *(added 2026-08-02)* — the only learned term.
- **Null policy** — ~~the same Approximator untrained~~ **corrected
  2026-08-02**: the Approximator at `W = 0`, which under the residual
  decomposition *is* the myopic base. The floor a trained Policy must clear
  before "it learned" means anything.
- A clause under **Policy** stating the observability rule.

## Out of scope (deliberately)

- **Changing the linear baseline in any way.** It is the opponent; `test_self_golden`
  pins it; every ticket predicts a zero diff against it.
- **Moving the training loop out of `Model` into the `Trainer`.** The research
  note recommends it (§6.3) and it is architecturally right — but it is not a
  precondition for this effort, and it moves the very code path the self-golden
  pins. Recorded as debt in ticket 02, not executed here.
- **Rollout, MCTS, scenario lookahead** (research note §5). The one-agent-at-a-time
  structure this effort preserves is exactly what they build on; that is a
  follow-on, not this.
- **Making the transformer the default Policy.** It ships available. Whether it
  becomes the default is decided by ticket 09's verdict, not by this spec.
- **Performance of the linear path.** The `simulation-performance` effort's
  ~6.4× is not to be regressed without saying so.

## Comments
