# Spec: Neural policy — a transformer over raw State, against the linear baseline

Status: open (grilling session 2026-07-30)

## Goal

Replace the Policy's **approximator**, not its algorithm: keep every-visit Monte
Carlo policy evaluation exactly as it is, and swap the 19-feature linear
`Q = X · W` for a transformer that reads **raw State facts** and produces
`Q(s, a)` directly. No hand-engineered features, no polynomial cross-terms, no
earliness bins, no candidate-set heuristic.

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
| 1 | What the network observes | **Raw entity tokens**: one per pending Client, one per vehicle, one global. Hard rule: a token never carries a cost formula, a polynomial, or a bin |
| 2 | Congestion awareness | **Out of the blind arm.** The transformer is exactly as congestion-blind as the baseline, so the comparison measures the approximator — not information one side was handed and the other was not |
| 3 | **The observability rule** | The Policy reads **only** `State` and the static `EpisodeGeometry`. Never `EpisodeVelocities`, never `congested_arcs`, never `TravelTimeModel` at `tau`. The only velocity that exists for the network is `state.observed_velocity` — what its own vehicles measured while traversing real arcs |
| 4 | Action set | **Every pending Client + the depot**, feasibility mask only. `_select_vehicle_possible_actions`, `_classify_shortest_distance_clients`, `delayed_clients`, the `350`/`310` literals and `number_actions_test` all die for this Policy. The B11 invariant survives *as a mask*, because it is a constraint, not a heuristic |
| 5 | Learning rule | **`Q(s,a)` + Monte Carlo return.** Same target `U_t − acquired_cost`, undiscounted, same statistical object. Only `X · W` → `net(tokens)` and constant-step SGD → Adam |
| 6 | Multi-vehicle structure | **Sequential one-agent-at-a-time**: one encoder pass per decision epoch, then `m` cheap head passes. `claimed` enters at the head, not the encoder, so one encode serves the whole sweep |
| 7 | torch | **Optional extra `neural`**, lazily imported, `device` in config. A ticket measures CPU vs CUDA rather than assuming. ~~CPU by default~~ — **amended 2026-07-31 (ticket 12): the default is `"auto"`**, resolved once per run and pinned in that run's record. See below |
| 8 | Training seam | **`TrainablePolicy` protocol**; the loop stays in `Model`. `TrainingSnapshot` gains `vehicle_completing_service` |
| 9 | Learning unit | **One batch per episode**, K shuffled minibatch passes, then discarded. Strictly on-policy — which is what "Monte Carlo policy evaluation" means and what keeps the comparison interpretable |
| 10 | "It learns" gate | **Three parts**: null model, ≥3 init seeds, calibration |
| 11 | Comparison budget | **Each policy at its own best, budget disclosed.** The baseline runs at 100/500/2000 episodes × its full action-count sweep and is compared at its *best* cell |
| 12 | Stopping | **Train to convergence**: patience → `lr × 0.3` → three reductions → stop. The hard cap is a safety net, not the budget; if it fires the run is reported as **"did not converge"** |
| 13 | Live comparison | **Reference card + paired live prints.** The scalar `static_policy_mean_cost` retires in favour of the baseline's **per-seed** cost vector |
| 14 | If it loses | **Pre-committed two-arm escalation**, protocol frozen before measuring, closing ADR with the negative result. The Policy ships available but not default |

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

### Gate A — "it learns" (hard; ticket 08)

All three, on the held-out `test_seeds` (100..153), never on the evaluation
seeds:

| Part | Test | Threshold |
|---|---|---|
| **Null model** | Trained vs. the *same architecture untrained*, paired per seed | Wilcoxon signed-rank **p < 0.05** and **≥ 5%** mean cost reduction |
| **Reproducibility** | ≥ **3** independent network-init seeds | Improvement reported as mean ± sd, not a single number |
| **Calibration** | Spearman ρ(predicted `Q`, realised `U_t`) on held-out episodes | **≥ 0.5**, against ≈ 0 untrained |

The null model is **not a random network**. Because of the myopic warm start
(ticket 05), the untrained network is a *nearest-feasible-Client* policy — a
respectable rival. "It learns" therefore means "it beats going to the nearest
Client", which is a claim with content.

Spearman, not Pearson: the episode-cost distribution has a brutal right tail
(F10). Calibration is the part that cannot be faked by a lucky run.

### Gate B — "it wins" (the objective; tickets 09/10)

Transformer vs. the linear baseline's **best cell** (best training budget × best
`test_action_count`), paired over `test_seeds`:

> Wilcoxon signed-rank **p < 0.05** **and** **≥ 3%** mean total-cost reduction.

Below 3% paired over 50 seeds is real but not interesting.

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
| Gate A effect | ≥ 5% vs. null, p < 0.05 | The null is nearest-neighbour, not random |
| Gate A calibration | Spearman ρ ≥ 0.5 | Softest of the three; report the number regardless |
| Gate B effect | ≥ 3% vs. baseline best cell, p < 0.05 | |
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

Critical path: 01 → 02 → 03 → 04 → 05 → 06 → 07 → **12** → 08 → 09. Ticket 10
is conditional on 09's verdict; 11 closes. Ticket 12 carries the highest
number but runs before 08: it was opened after 08-11 existed, and renumbering
would orphan the ticket numbers already in the commit history.

| # | Ticket | Blocked by | Predicted self-golden diff |
|---|---|---|---|
| 01 | Baseline reference card | — | **zero** |
| 02 | The `TrainablePolicy` seam | — | **exactly zero** |
| 03 | torch as an optional extra + the real cost measurement | — | **zero** |
| 04 | The tokenizer, and the observability rule as a test | 03 | **zero** |
| 05 | The network, the Q head and the myopic warm start | 04 | **zero** |
| 06 | The Policy: decide, decide_train, learn | 02, 05 | **zero** |
| 07 | Trainer: live paired report, convergence stopping, checkpoints | 01, 06 | **zero** |
| 08 | Gate A — does it learn | 07, **12**, `simulator-correctness`/11 | **zero** |
| 09 | Arm 1 — the real run and the verdict | 08, **12** | **zero** |
| 10 | Arm 2 — fleet shared observation memory (conditional) | 09 | **zero** |
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
- **Null policy** — the same Approximator untrained. The floor a trained
  Policy must clear before "it learned" means anything.
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
