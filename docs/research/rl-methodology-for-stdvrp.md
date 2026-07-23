# RL methodology for the STDVRP: what to do instead of linear-VFA Monte Carlo

> **Status:** research note, not a decision. No ADR has been written from it yet.
> **Scope:** the evaluation/training policy in `src/stdvrp/policies/monte_carlo.py` and the
> legacy training rule it ports.
> **Date of research:** 2026-07-22. All external claims are anchored to primary sources
> (arXiv/PDF, publisher records, official repositories) with URLs in [Sources](#sources).
> **Line anchors** were re-verified against the working tree on 2026-07-22, on branch
> `FEATURE_DEV_total_code_refactorization` with ticket-08 work in progress
> (`monte_carlo.py`, `model.py`, `episode.py` all had uncommitted changes). They will drift
> as Phase 2 proceeds — treat the symbol names as authoritative and the numbers as hints.

---

## Question

*What reinforcement-learning methodology should replace (or repair) the current
every-visit Monte Carlo policy evaluation with a 19-feature linear value-function
approximation, in order to get lower episode cost (earliness + delay + overtime),
better sample efficiency, and less dependence on hand-engineered basis functions?*

---

## Bottom line

**The algorithm is not the binding constraint; the observation set is.** The simulator
draws time-dependent stochastic velocities and injects non-recurring congestion events,
but *not one of the 19 features, and not one of the travel times used to build the
candidate action set, can see any of it* — every travel time in the policy comes from a
static, precomputed, time-independent `average_minutes` column
(`shortest_path_cache.py:47-49`), and the only velocity aggregate the policy computes is
discarded before it reaches the feature vector (`monte_carlo.py:444-450`). No RL
estimator can learn to anticipate congestion from features that are congestion-blind.
Fixing that is cheap and strictly dominates every algorithmic change below.

Ranked, my recommendation is:

1. **Make the state observable** — add time-dependent and congestion-aware travel-time
   estimates plus congestion features; keep the linear model at first so the change is
   measurable in isolation.
2. **Layer one-agent-at-a-time (multiagent) rollout on top of the existing greedy
   policy** — Bertsekas's Proposition 2.1 gives a *guaranteed* cost improvement over the
   base policy at `O(s·m)` rather than `O(s^m)` per decision, and the code already has
   the exact per-vehicle sequential structure the theorem needs
   ([Bertsekas 2019](https://arxiv.org/abs/1910.00120)).
3. **Replace SGD-on-Monte-Carlo with a least-squares estimator (LSTD-Q / LSPI, or batch
   least-squares Monte Carlo)** on the same features — removes the learning rate, the
   feature-scaling pathology and most of the variance, at ~100 lines of numpy
   ([Lagoudakis & Parr 2003](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf)).
4. **Then, and only then**, consider a small neural VFA over the same (enriched)
   features — Ulmer's own group reports up to **+22%** over a tuned parametric policy
   with a 2-hidden-layer MLP Q-network on an SDVRP
   ([Chen, Ulmer & Thomas](https://arxiv.org/abs/1910.11901)).

**What I recommend *against*:** porting an attention/pointer-network constructive policy
(Kool-style AM, POMO). Their published assumptions — unit-square Euclidean coordinates,
fixed instance size, deterministic transitions, a single sequentially-constructed tour —
are all violated here, and the only benchmark I found that stress-tests them under
time-dependent stochastic travel times reports that **POMO and AM degrade by over 20%
under distributional shift while classical heuristics stay robust**
([SVRPBench, arXiv:2505.21887](https://arxiv.org/abs/2505.21887)).

---

## Where we are today

### The decision rule (evaluation path)

`MonteCarloPolicy.decide` (`monte_carlo.py:124-129`) → `_select_greedy_actions`
(`:131-137`). For each vehicle `v` in fixed index order `0..m-1`:

1. Build a candidate set `A_v` (`_select_vehicle_possible_actions`, `:242-308`):
   the `number_actions_test` nearest unvisited clients by **static** average travel time
   (`:268-284`), minus the clients currently targeted by other vehicles
   (`forbidden_actions`, `:249-251`), plus the depot when the return trip would breach
   the horizon (`:289-296`), plus up to two "delayed" clients (`:298-303`). Degenerate
   branches handle `< 3` clients left and a post-350-minute idle depot.
2. Evaluate `Q = W · X` for each candidate with the *joint* action vector holding all
   other vehicles fixed, and take a strict `argmin` with first-wins tie-breaking
   (`_select_best_q_action_for_vehicle`, `:218-236`).

`X` is 19 components: 12 "general state" (`_extract_general_state_features`, `:388-452`)
and 7 "state-action" (`_extract_state_action_features`, `:454-532`), concatenated at
`:226`. `W` is created as `np.zeros(19)` (`_create_W`, `:238-240`).

This per-vehicle-in-sequence structure is *exactly* the "one-agent-at-a-time"
reformulation of a multiagent decision — see [§5](#5-decision-time-methods-rollout-mcts-and-scenario-based-lookahead).

### The learning rule (training path)

`Model.run_training_episode` (`model.py:132-161`) runs an ε-greedy episode
(`MonteCarloPolicy.decide_train`, `monte_carlo.py:141-172`), deep-copying the state
before each transition (`model.py:149`) and appending the per-transition cost to
`episode_rewards`, which is seeded with a leading `0` (`model.py:144`). At termination it
calls `MonteCarloPolicy.update_W` (`monte_carlo.py:174-198`), which ports the legacy
`actualize_W` (`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py:4300-4389`):

```python
U_t = 0
for t in range(T - 1, -1, -1):
    U_t += rewards[t + 1]                       # undiscounted cost-to-go
    ...
    Q_pred = np.dot(X, self.W)
    gradient = lr * ((U_t - self.total_cost_acquired - Q_pred) * X)
    self.W = self.W + gradient                  # constant-step SGD
```

So: **every-visit Monte Carlo policy evaluation, linear VFA, constant-step semi-gradient
SGD, no discounting, one pass per episode, backward over the episode, `W` initialised to
zeros, ε-greedy behaviour policy, target = realised cost-to-go minus a state-dependent
"already-acquired cost" baseline** (`_calculate_already_acquired_cost`,
`monte_carlo.py:200-216`).

The outer loop (`Main_Chengdu...py:6053-6150`) draws a **new client instance per training
episode** (`random_seed` starting at 1000 and incrementing), carries `W` forward
episode-to-episode, uses `lr = 1e-6` for the first episode only and the configured `lr`
thereafter (`:6095`, `:6098`), and every `test_frequency` episodes evaluates the current
`W` greedily on a fixed 50-seed validation set (`range(100000, 100050)`), keeping the
`Best_W` with the lowest mean cost.

### The dead alternative

`actualize_W_Q_learning` (`Main_Chengdu...py:5037-5060`) is a semi-gradient TD(0)-style
update, called only from `create_Q_learning_episode` (`:6035-6043`), which has **no call
sites anywhere in the file**. It also carries two independent defects: the weight update
uses `self.W - lr * (...)` where the target-following direction is `+`, and `state_t` and
`state_t_1` are both bound to `self.state` (`:6038`, `:6040`) — the same mutable object —
so the TD target would always have compared a state to itself. Treat it as an abandoned
sketch, not as a validated alternative.

### The `Policy` seam

`Policy` (`policies/base.py:12-18`) declares only `decide(state) -> list[int]`. Training
lives outside it: `Model.run_training_episode` does
`isinstance(policy, MonteCarloPolicy)` (`model.py:139`) and then calls `decide_train`
and `update_W` directly. Any new methodology has to reckon with this leak — see
[§6](#6-practical-migration-path).

---

## 1. Stay with a linear VFA, but fix the estimator

### 1.1 Monte Carlo vs TD(λ) vs least-squares

The current target `U_t` is the full realised return. That is unbiased for the value of
the *behaviour* policy, but its variance is the variance of an entire episode's cost, and
this episode's cost distribution has a very heavy right tail: `model.py:255-261` adds a
terminal penalty of `40000 - 200 · |visited|` whenever the clock reaches 1198 without
termination. A single such episode injects a target three orders of magnitude larger than
a normal one into *every* `t` in that episode's backward pass.

- **Bias–variance is formally characterised.** Kearns & Singh give the first finite-sample
  bias–variance decomposition of the error of `k`-step / TD(λ) updates and derive
  *schedules* for λ (or `k`) that provably beat any fixed value
  ([Kearns & Singh, COLT 2000, pp. 142-147](https://www.learningtheory.org/colt2000/papers/KearnsSingh.pdf)).
  *(a) That is what the paper demonstrates.* I could not extract the exact form of the
  bound — the PDF did not parse through my tooling — so I am **not** stating the direction
  of each term from the source; see [Open questions](#open-questions--what-i-could-not-verify).
- **Convergence with linear function approximation is only guaranteed on-policy.** The
  canonical result is [Tsitsiklis & Van Roy, IEEE TAC 42(5):674-690, 1997]. *(a)* I
  verified the bibliographic record but could not obtain the full text through open
  channels, so I cite it as the standard reference rather than quoting its theorem.
- **Least-squares methods sidestep the whole question.** LSTD-Q/LSPI solve the projected
  Bellman equation directly by a matrix solve instead of by stochastic descent.
  Lagoudakis & Parr state that LSTD-Q "makes efficient use of data and converges faster
  than other conventional temporal-difference learning methods", that it is a
  "model-free, off-policy method which can use data collected arbitrarily from any
  reasonable sampling distribution", and — decisively for this codebase — that it needs
  **no learning-rate parameter**
  ([JMLR 4:1107-1149, 2003](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf)).

*(b) Inference for this repo:* the cheapest version of this idea is not even LSTD — it is
**batch least-squares Monte Carlo**: keep the existing episode replay, but instead of
stepping `W` once per epoch, accumulate `A += X Xᵀ` and `b += X (U_t − c_acq)` across many
episodes and solve `W = A⁻¹ b` (ridge-regularised). That is a ~30-line change to
`update_W` + a trainer-side accumulator, it is invariant to the arbitrary feature
normalizers (150, 850, 1150, 13, 60, 100, 180, 2500), and it eliminates the learning rate
entirely. It is also *exactly the same statistical object* the current code is trying to
estimate, so a golden-master comparison is meaningful.

*(a) What Lagoudakis & Parr also warn:* combining LSTD with policy iteration is not free —
they note that a *state* value function learned by LSTD "is of no use for policy
improvement when a model of the underlying process is not available". Here that warning
does not bite, because the code already learns a **state-action** function `Q(S, a)`, which
is the right object for a model-free argmin.

### 1.2 Step sizes

The current rule is a constant step size, tuned by hand, applied to unnormalised features
of wildly different magnitude, against a target whose distribution changes as `W` changes
(because the behaviour policy is ε-greedy over `W`).

Ryzhov, Frazier & Powell argue directly against this: fixed step sizes "create volatility
or slow initial convergence"; the `1/n` rule is theoretically convergent but "unusably
slow"; and they emphasise that existing rules "largely ignore the dependence of the
observation on the previous value function approximation" — the exact nonstationarity
present here. They state the Robbins–Monro conditions as `Σ αₙ₋₁ = ∞` and
`Σ α²ₙ₋₁ < ∞` and derive an optimal stepsize formula with a single, insensitive tuning
parameter ([arXiv:1407.2676](https://arxiv.org/abs/1407.2676)). The earlier and more
widely used rule from the same lab is George & Powell's adaptive stepsize / BAKF
([Machine Learning 65(1):167-198, 2006](https://doi.org/10.1007/s10994-006-8365-9) —
bibliographic record verified; the CASTLE Lab PDF links are dead as of this research).
Powell devotes an entire chapter (Ch. 6, "stepsize policies") to this in
[RLSO](https://castle.princeton.edu/rlso/) (Wiley, 2022).

*(a)* The current rule satisfies **neither** Robbins–Monro condition. *(b)* Combined with
raw, unstandardised features, the *effective* per-feature step size differs by orders of
magnitude across the 19 components, which is a plausible explanation for the difficulty of
tuning `lr` reported in the legacy experiment filenames.

### 1.3 The post-decision state, and whether the code already approximates it

Powell's post-decision state `S^x` is the state immediately after the decision but before
the exogenous information arrives; approximating `V(S^x)` "eliminates the expectation"
from the Bellman operator, so the decision becomes a deterministic optimisation over the
current cost plus `V̄(S^x)` rather than an expectation over exogenous outcomes
([Powell, arXiv:2002.06238](https://arxiv.org/abs/2002.06238); see also
[RLSO](https://castle.princeton.edu/rlso/), chapters 14-16).

*(b) My reading of the code:* `MonteCarloPolicy` does **not** implement a post-decision
VFA. It learns `Q(S, a)` over a *joint* action vector with explicit action features
(`_extract_state_action_features`, `monte_carlo.py:454-532`, which computes projected
earliness/delay/overtime for the proposed assignment). That is a state-action VFA, not a
post-decision-state VFA. `_calculate_already_acquired_cost` (`:200-216`) is *not* a
post-decision construct either — it is a sunk-cost baseline subtracted from the target;
because it depends only on `S`, it cannot change the per-state `argmin`, so it acts purely
as a variance-reduction baseline, and its value is never subtracted at decision time
(`:218-236`), making train-time and act-time targets differ by a state-dependent offset.

**Sources disagree on whether moving to a post-decision formulation would help here.**
The textbook OR position (Powell) is that the post-decision state is the canonical fix for
exploding expectations in resource-allocation and routing. But Ulmer's own group tested
this empirically on an SDVRP and found the opposite: in
[Chen, Ulmer & Thomas](https://ar5iv.labs.arxiv.org/html/1910.11901), Figure 3 shows that
their feature set *combining state and action information* "outperforms all other
selections", **including post-decision state features**, which they attribute to explicit
action features guiding decisions "especially in early training runs when the
approximation is still weak". *(a) That is what their experiment shows.* *(b) My
inference:* the current design's choice of state-action features over a post-decision
formulation is defensible and should **not** be the first thing changed.

### Cost to adopt (§1)

| Change | Files touched | Effort |
|---|---|---|
| Standardise features (running mean/var), drop the permanently-zero component | `monte_carlo.py:388-532`, `:238-240` | hours; invalidates stored `W` |
| Adaptive stepsize (BAKF or the OSA formula) | `monte_carlo.py:174-198` | hours |
| Batch least-squares Monte Carlo | `monte_carlo.py:174-200` + a trainer accumulator | ~1 day |
| LSTD-Q / LSPI | new module + a sample-collection loop | ~2-3 days |

All of these **reuse the existing simulator unchanged**.

---

## 2. Ulmer-style ADP for dynamic VRP

This is the closest published line of work to the problem at hand.

### 2.1 What Ulmer's group has actually built

- **Anticipatory time budgeting (ATB)** — offline VFA over a deliberately tiny aggregated
  state: (point of time, free time budget). Simulate many realisations, tabulate the
  value of each cell, act greedily. Ulmer, Mattfeld & Köster,
  [Transportation Science 52(1):20-37](https://doi.org/10.1287/trsc.2016.0719).
- **Offline VFA for dynamic multi-period VRP** — accept/defer decisions, offline VFA,
  "significant improvements compared to benchmark policies". Ulmer, Soeffker & Mattfeld,
  [EJOR 269(3):883-899, 2018](https://www.sciencedirect.com/science/article/abs/pii/S0377221718301711).
- **Offline–online hybrid** — the flagship result. Ulmer, Goodson, Mattfeld & Hennig
  combine "offline value function approximation (VFA) with online rollout algorithms
  resulting in a high-quality, computationally tractable policy", and prove that "rollout
  policies based on deterministic decision rules guarantee performance improvements over
  their base policies".
  [Transportation Science 53(1):185-202, 2019](https://doi.org/10.1287/trsc.2017.0767)
  (abstract verified via [TRID 01695266](https://trid.trb.org/view/1583758); the INFORMS
  full text is paywalled).
- **Neural VFA** — Chen, Ulmer & Thomas replace the tabulated/parametric VFA with a deep
  Q-network on a same-day-delivery SDVRP: three MLPs (2 hidden layers, `10·|V|` nodes),
  ReLU, min-max normalised features, ε decayed 1.0 → 0.01, Adam with an exponentially
  decaying learning rate from 0.01, experience replay, 500 training instances, no target
  network. Reported improvement over the tuned parametric policy: **up to 22%** with
  constrained resources (2 vehicles / 5 drones), shrinking as resources become abundant.
  [arXiv:1910.11901](https://arxiv.org/abs/1910.11901).

### 2.2 What Hildebrandt, Thomas & Ulmer say works and fails

Their survey is the single most on-point source for this decision
([arXiv:2103.00507](https://arxiv.org/abs/2103.00507); the journal version is
Hildebrandt, Thomas & Ulmer, *Computers & Operations Research* 150, 2022,
["Opportunities for reinforcement learning in stochastic dynamic vehicle routing"](https://www.sciencedirect.com/science/article/abs/pii/S030505482200301X) —
*(b) I infer these are the same work from identical authorship and topic; the arXiv
listing carries no journal-ref, so I could not confirm it*).

What they report, verbatim-anchored:

- **On linear VFAs specifically:** the advantage is that a linear VFA "can be plugged into
  an MIP formulation"; the disadvantage is that "we assume that action values depend
  linearly on state information which is not the case for most SDVRPs". *(a) This is a
  direct criticism of the class the current code belongs to.*
- **On state aggregation (the dominant OR approach):** "the biggest challenge is
  fine-tuning the granularity … which is essentially a form of state space reduction."
- **On neural Q-networks:** "it is efficient as the action-values are obtained with only
  one forward pass through the Q-network", and good architectures enable "effective usage
  of more features which is akin to a finer discretization" — but larger networks need
  longer training, "a limiting factor for real-world sized problem instances".
- **On why standard RL breaks on SDVRP:** "the number of solution variables regularly
  explodes" and "each decision point reduces to a vehicle routing problem"; most RL
  methods "typically enumerate all possible actions", which is infeasible.
- **On action decomposition (which this repo does):** restricting actions to make
  enumeration tractable yields "coarse" policies because "the decomposed actions are
  dependent". *(a) That is a real, named limitation of the current per-vehicle greedy
  argmin.*
- **On the CS/deep-RL stream:** "there is no work in CS that addresses an SDVRP with
  complex route constraints"; those researchers focus on evaluation because "searching the
  action space is relatively easy" in their problems — which inverts the actual SDVRP
  challenge.
- **Their two proposed directions:** (i) piecewise-linear neural VFAs embedded in an MIP,
  exploiting recent strong MIP formulations of trained ReLU networks; (ii) iterative
  policy-based route construction that decides "the next tentative stop to visit"
  sequentially, with "an extra head in the neural network" evaluating long-term
  feasibility.

*(b) Inference for this repo:* direction (ii) is structurally *already* what
`MonteCarloPolicy` does — one next stop per vehicle, per decision epoch. That is a strong
signal that the architecture of the decision is right and the approximator/feature layer
is what is wrong.

### Cost to adopt (§2)

- **Offline VFA with better features / aggregation:** low. Reuses the simulator unchanged.
- **Neural VFA (small MLP over the same enriched features):** medium. Requires a
  torch dependency, a replay buffer, and a `Policy` implementation that swaps `np.dot`
  for a forward pass. **Reuses the simulator unchanged** — the Chen/Ulmer/Thomas recipe
  maps almost one-for-one onto `run_training_episode`.
- **Offline–online hybrid:** high, because it needs the rollout machinery of §5.

---

## 3. Replacing hand-engineered features with learned representations

### 3.1 What the constructive neural models actually assume

**Kool, van Hoof & Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019**
([arXiv:1803.08475](https://arxiv.org/abs/1803.08475);
[official repo](https://github.com/wouterkool/attention-learn-to-route)). From the full
text:

- Node coordinates are **sampled uniformly in the unit square `[0,1]²`**; distances are
  **Euclidean**, implicit in the coordinates.
- **Separate models are trained per instance size** (n = 20, 50, 100). Figure 5 shows
  cross-size generalisation degrades "as the difference becomes bigger".
- Transitions are **deterministic**; at test time they use greedy decoding.
- Results (Table 1, greedy decoding): TSP20 0.34% gap, TSP50 1.76%, TSP100 4.53%;
  CVRP20 4.97%, CVRP50 5.86%, CVRP100 7.34%.
- Training: REINFORCE, `∇L(θ|s) = E[(L(π) − b(s)) ∇log p_θ(π|s)]`, with `b(s)` the cost of
  "a deterministic greedy rollout of the policy defined by the best model so far"; the
  baseline is frozen for a full epoch and replaced only when a paired t-test at α = 5% on
  10,000 held-out instances shows improvement.
- **They explicitly note no treatment of non-Euclidean or time-dependent costs.**

**Nazari, Oroojlooy, Snyder & Takáč, "Reinforcement Learning for Solving the Vehicle
Routing Problem"** ([arXiv:1802.04240](https://arxiv.org/abs/1802.04240)). They drop the
RNN encoder for permutation invariance, train with policy gradient against a critic
baseline, and — relevant here — do address stochasticity: Appendix C.6 runs a Stochastic
VRP with Poisson customer arrivals and cancellable demands trained with A3C, satisfying
**28.8%** of total demand versus **19.6%** for the next-best baseline. *(a) That is
stochastic **demand**, not stochastic time-dependent **travel time**.*

**Kwon et al., POMO** ([arXiv:2010.16011](https://arxiv.org/abs/2010.16011)). Rolls out
`N` trajectories from `N` different start nodes and uses their mean as a shared baseline,
`b_shared(s) = (1/N) Σ_j R(τʲ)`. They argue three advantages over a greedy-rollout
baseline: zero-mean advantages (a greedy-rollout baseline "produce[s] mostly negative
advantages"), no separate critic or cloned policy, and resistance to local minima.
Results with ×8 augmentation: TSP100 0.14% gap, CVRP100 0.32% gap. Assumptions are the
same as Kool's: Euclidean unit square, fixed instances, deterministic reward.

### 3.2 Do those assumptions hold here? No.

| Assumption in AM / POMO | This problem |
|---|---|
| Euclidean distance from `[0,1]²` coordinates | Shortest paths over the Chengdu road graph, cached per (node, client) pair (`shortest_path_cache.py`) |
| Fixed instance size per trained model | Client count and fleet size are drawn per episode by `ClientGenerator` |
| Deterministic transition | Velocities drawn per arc-minute (`model.py:564`, `random.gauss`) plus exogenous congestion events |
| One tour constructed left-to-right | `m` vehicles moving asynchronously in continuous time, re-decided at every decision epoch |
| Cost = tour length | Earliness + delay + overtime + distance, with time windows |

### 3.3 The one benchmark that stress-tests this

[SVRPBench (arXiv:2505.21887, 2025)](https://arxiv.org/abs/2505.21887) — *a preprint; I
did not find a peer-reviewed version* — builds >500 urban-scale instances with
time-dependent congestion (Gaussian peaks at μ = 8h and μ = 17h, σ = 1.5), log-normal
multiplicative delays, Poisson accident events of 0.5-2.0h, and empirically grounded time
windows. Reported results
([HTML full text](https://arxiv.org/html/2505.21887v2)):

- Table 2 (mean over 5 stochastic runs): POMO total cost 40,650.4 / feasibility 0.933;
  AM 41,358.3 / 0.910; OR-Tools 40,259.3 / 0.984; NN+2opt 40,707.5 / 0.984.
- Adding time windows: cost increase of **+584.8% for POMO** and **+536.2% for AM**;
  feasibility drops to 87.9% and 85.4% against >96% for the classical methods.
- Instances ≥500 nodes: POMO feasibility 86%, AM 83.5%, versus >97% for classical methods.
- Authors' conclusion: "state-of-the-art RL solvers like POMO and AM degrade by over 20%
  under distributional shift", while classical heuristics remain robust.

### 3.4 What *does* transfer: time-dependency-aware encoders

[SED2AM (arXiv:2503.04085, 2025)](https://arxiv.org/abs/2503.04085) — Mozhdehi, Wang, Sun
& Wang — is the closest neural-architecture match: a Transformer policy with a "temporal
locality inductive bias" in the encoder and a dual decoder (a vehicle-selection decoder
plus a trip-construction decoder), trained with a policy-gradient algorithm with a
self-critic baseline, on real time-dependent travel data from **Edmonton and Calgary**.
They report outperforming OR-Tools, Kool's AM, GCN-NPEC, GAT-Edge and Residual E-GAT.
**Crucially: their travel times are deterministic time-dependent — constant within each
time interval — not stochastic.** *(b) So SED2AM validates the "encode time-of-day into
the representation" idea, but not the "handle exogenous congestion shocks" part.*

### 3.5 Implementation reality check

I queried the two mainstream libraries via DeepWiki
([`wouterkool/attention-learn-to-route`](https://github.com/wouterkool/attention-learn-to-route),
[`ai4co/rl4co`](https://github.com/ai4co/rl4co)). Summary of what came back:

- **No environment supports time-dependent or stochastic travel times.** Travel time is
  `d_ij / speed` with a constant `speed` in `MTVRPEnv`.
- **Distances are Euclidean from `locs`.** A `cost_matrix` exists only in
  `TSPEdgeEmbedding` (a model-side embedding), not in the environment's `_step`,
  `get_action_mask` or `_get_reward`.
- **`CVRPTWEnv`/`MTVRPEnv` support time windows but the reward is negative tour length —
  "time windows are not considered for the calculation of the reward".** Earliness/lateness
  penalties would have to be added by hand.
- **Multi-vehicle is turn-based, not asynchronous continuous-time.** `MTSPEnv` advances an
  `agent_idx`; the `step` method takes a single action at a time.
- To plug in a custom simulator you must subclass `RL4COEnvBase` and reimplement
  `_make_spec`, `_reset`, `_step`, `_get_reward`, `get_action_mask`, plus a `Generator`.

*(b) Inference:* adopting rl4co means rewriting `Model` as a batched, TensorDict-based
environment — that is a rewrite of the simulator, not a reuse of it, and it discards the
golden-master safety net that tickets 04/07/08 were built to provide.

### 3.6 The cheap version of "learned representations"

*(b) My recommendation:* the useful half of this idea is separable from the expensive
half. Keep the decision structure and the simulator; replace only `np.dot(X, W)` with a
small MLP over a *larger, less hand-tuned* feature vector — precisely the Chen/Ulmer/Thomas
recipe. That gets you nonlinearity and removes the pressure to hand-craft polynomial
cross-terms (`monte_carlo.py:401-407`), at a fraction of the cost of a constructive
neural policy. Graph neural networks over the road network are a further step, but *(c)
speculatively* the payoff is limited while the state features carry no per-arc congestion
information for a GNN to propagate.

---

## 4. Policy gradient / actor–critic vs value-based, for this decision structure

The decision structure is: `m` vehicles, a per-step candidate set of ~`m+2` discrete
choices per vehicle, long episodes (hundreds of decision epochs), cost accrued
continuously plus a large terminal penalty.

**Arguments for staying value-based:**

- The action set is *small after decomposition* (`number_actions_test = vehicles + 2`).
  Hildebrandt et al. note the neural-Q advantage precisely here: "action-values are
  obtained with only one forward pass".
- A learned `Q` composes with rollout and MCTS (§5) as a terminal-cost approximation.
  A policy network does not, except as a rollout base policy.
- Every reported SDVRP success from the OR side in the sources above (Ulmer's ATB,
  offline VFA, offline–online, Chen/Ulmer/Thomas DQN) is value-based.

**Arguments for policy gradient:**

- Kool et al. found a greedy-rollout baseline "more efficient than using a value
  function", and Figure 3 shows the rollout baseline improves both quality and convergence
  speed over an exponential-moving-average baseline and over a critic. *(a) Demonstrated —
  on static Euclidean TSP/CVRP.*
- POMO's shared baseline removes the critic entirely and gets the best published gaps on
  those same static benchmarks.

**Why the baseline story does not transfer cleanly here.** *(b) Inference:* both the
greedy-rollout baseline and the POMO shared baseline are **per-instance** baselines that
work because the same static instance can be re-solved from scratch many times. In this
problem an "instance" is a *stochastic episode*: re-running it draws different velocities
and different congestion events. To get the same variance reduction you would need
**common random numbers** — replay the same exogenous sample path under both the sampled
and the baseline policy. That is implementable (`model.py` already memoizes per-arc
velocity draws per episode via `_memoized_normal_velocity`, `:545`), but it is not
free, and *the current code actively destroys it*: training exploration draws from the
**global** `random` stream (`monte_carlo.py:170`), interleaved with the transition
function's `random.gauss` velocity draws (`model.py:564`), so **changing the policy's
exploration changes the traffic realisation**. See
[Flaws](#flaws-in-the-current-setup-independent-of-algorithm-choice), item F5.

**PPO.** *(c) Speculation, flagged as such:* I found no primary source applying PPO to an
SDVRP with time-dependent stochastic travel times in this survey. Given the long episodes
and the enormous terminal-cost variance, my expectation is that PPO would need heavy
reward shaping and per-step advantage normalisation before it beat a well-fit VFA, and
would cost far more implementation effort. I would not start here.

---

## 5. Decision-time methods: rollout, MCTS, and scenario-based lookahead

This is where I think the largest *reliable* gain is, because these methods **improve on
whatever policy you already have** rather than replacing it.

### 5.1 Multiagent (one-agent-at-a-time) rollout — the structural match

Bertsekas, ["Multiagent Rollout Algorithms and Reinforcement Learning" (arXiv:1910.00120,
2019, rev. 2020)](https://arxiv.org/abs/1910.00120):

- **Standard rollout cost improvement:** `J_{k,π̃}(x_k) ≤ J_{k,π}(x_k)` for all states and
  stages — the rollout policy is never worse than the base policy.
- **The multiagent blow-up:** with `m` agents and `s` choices each, standard rollout needs
  `O(s^m)` Q-factor evaluations per stage.
- **The fix:** insert intermediate states `(x_k), (x_k, u¹_k), …, (x_k, u¹_k,…,u^{m-1}_k)`
  so agents choose sequentially. This reduces the per-stage cost to **`O(s·m)`** — linear
  instead of exponential.
- **Proposition 2.1:** the one-agent-at-a-time rollout policy **still satisfies the cost
  improvement property**.
- Agent ordering does not affect the proof; the order may be changed per stage or
  optimised over.

*(b) Inference, and this is the key one:* `_select_greedy_actions`
(`monte_carlo.py:131-137`) already iterates vehicles in order and fixes previously
decided vehicles when evaluating the next — it is the multiagent decomposition, with a
learned `Q` standing in for the rollout's simulated cost-to-go. Swapping that learned `Q`
for an actual **simulated rollout of the current greedy policy** upgrades a heuristic with
no guarantees into a policy with Bertsekas's guaranteed improvement over its base, at
`O(s·m)` simulations per decision epoch. Ulmer, Goodson, Mattfeld & Hennig make exactly
the same move in the SDVRP setting and state that "rollout policies based on deterministic
decision rules guarantee performance improvements over their base policies"
([Transportation Science 53(1), 2019](https://doi.org/10.1287/trsc.2017.0767)).

See also Goodson, Thomas & Ohlmann, "Restocking-Based Rollout Policies for the Vehicle
Routing Problem with Stochastic Demand and Duration Limits", Transportation Science
50(2):591-607, 2016, which reports restocking-based rollout outperforming a-priori-based
rollout on a stochastic VRP. *(I verified title, authors, journal, volume, issue, pages and
year from search-surfaced records; INFORMS returned 403, so I have no DOI string and did
not read the full text.)*
The textbook treatment is Bertsekas, *Rollout, Policy Iteration, and Distributed
Reinforcement Learning* (Athena Scientific, 2020), full book PDF at the author's page:
<https://web.mit.edu/dimitrib/www/Rollout_Complete%20Book.pdf> *(cited as a reference; I
did not read it for this note)*.

### 5.2 Scenario-based / sample-average-approximation lookahead

Bent & Van Hentenryck's Multiple Scenario Approach continuously samples futures, solves
each as a deterministic problem, and aggregates via **consensus** or **regret** decision
functions: "Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic
Customers", [Operations Research 52(6):977-987, 2004](https://doi.org/10.1287/opre.1040.0124);
and "Waiting and Relocation Strategies in Online Stochastic Vehicle Routing",
[IJCAI 2007](https://www.ijcai.org/Proceedings/07/Papers/293.pdf). *(I could not parse
either PDF through my tooling; I verified the bibliographic records and the described
method only — see [Open questions](#open-questions--what-i-could-not-verify).)*

Mercier & Van Hentenryck's **Amsaa** goes further, combining sample-average approximation
from stochastic programming with an MDP search algorithm and a discrete-optimisation
oracle to guide the search; they report it "significantly outperforms state-of-the-art
algorithms" under various time constraints ("An anytime multistep anticipatory algorithm
for online stochastic combinatorial optimization",
[Annals of Operations Research, 2011](https://link.springer.com/article/10.1007/s10479-010-0798-7);
originally CPAIOR 2008). They also characterise the **one-step anticipatory algorithm**
(1s-AA) as making near-optimal decisions on a range of online stochastic combinatorial
problems including dynamic fleet management.

*(b) Inference for this repo:* the natural analogue is to sample `K` congestion/velocity
futures from the existing `CongestionGenerator` + `TravelTimeModel` at each decision epoch,
evaluate each candidate assignment under each future with a cheap deterministic route
evaluator, and pick by consensus or expected cost. This **reuses the simulator's stochastic
components unchanged** but needs a cheap forward evaluator that does not run the full
`transition_function`.

### 5.3 MCTS

Wilbur et al., "An Online Approach to Solve the Dynamic Vehicle Routing Problem with
Stochastic Trip Requests for Paratransit Services", ICCPS 2022
([arXiv:2203.15127](https://arxiv.org/abs/2203.15127)) formulate a DVRP with time windows
as an MDP and "use Monte Carlo tree search to evaluate actions for any given state",
handling the intractable action space by designing "heuristics that can sample promising
actions for the tree search". They report outperforming state-of-the-art approaches on
real partner-agency data "both in terms of performance and robustness". *(I could not
extract per-decision computation times from the abstract page.)*

*(b)* MCTS is strictly more expensive than one-step multiagent rollout and requires the
same simulator-forking capability. I would treat it as a follow-on to §5.1, not a
starting point.

### 5.4 What decision-time methods cost here

All of §5 needs one capability the code does not have: **the ability to fork the Model and
simulate forward from an arbitrary state**. Today `Model` holds mutable per-episode state
across many attributes (`congested_arcs`, `sampled_arc_velocities`, `node_time_arrival`,
`departure_tau`, `vehicles_shortest_path`, `tau_multiplicator`, the cost accumulators).
`copy.deepcopy(self.state)` already works (`model.py:149`), but the `State` object is not
the whole simulation state. See §6.

---

## 6. Practical migration path

### What has to change in the `Policy` interface

`Policy.decide(state) -> list[int]` (`policies/base.py:12-18`) is sufficient for:

- a different linear estimator (LSTD/LSPI/least-squares MC),
- an adaptive stepsize,
- richer features,
- a neural VFA.

It is **not** sufficient for anything in §5. Rollout, scenario lookahead and MCTS all need
the policy to *query the model*, which inverts today's dependency (Model owns Policy,
`model.py:48-118`). The minimal change I would make:

1. **Give `Model` a `fork()`** that returns a deep copy of the full simulation state
   (including `congested_arcs`, `sampled_arc_velocities`, arrival-time bookkeeping and a
   snapshotted RNG), plus a `simulate_to_termination(policy) -> float`. ADR-0002 permits
   this: it is a concrete capability, not a new interface.
2. **Introduce a `RolloutPolicy(Policy)`** holding a `base_policy: Policy` and a
   model factory; `decide` loops vehicles in order, and for each candidate runs `K` forked
   simulations under the base policy, taking the argmin of the mean cost. This preserves
   Bertsekas's Proposition 2.1 structure.
3. **Move training out of `Model`.** `run_training_episode` currently does
   `isinstance(policy, MonteCarloPolicy)` (`model.py:139`) and calls `decide_train` /
   `update_W`. Per CONTEXT.md, that belongs to the **Trainer**. Extracting it removes the
   type check and makes any new trainable policy pluggable without touching `Model`.
4. **Separate exploration randomness from exogenous randomness.** Give the Policy its own
   seeded `random.Random` for exploration (it already has `local_rng`/`local_rng_2`,
   `monte_carlo.py:96-97`) and stop drawing exploratory actions from the global stream
   (`:170`). This is a prerequisite for common random numbers, for any policy-gradient
   baseline, and for meaningful A/B comparison of two `W` vectors.

### Ranked recommendation

| # | Option | Expected gain | Impl. cost | Reuses existing simulator? | Primary evidence |
|---|---|---|---|---|---|
| 1 | **Congestion-aware & time-dependent features** (observed velocities, per-arc congestion on the planned path, time-dependent travel-time estimate replacing static `average_minutes`) | **High** — the policy currently cannot observe the phenomenon it is optimising against | Low–Medium | **Yes, unchanged** | Hildebrandt/Thomas/Ulmer on feature granularity ([arXiv:2103.00507](https://arxiv.org/abs/2103.00507)); Chen/Ulmer/Thomas feature-set ablation, Fig. 3 ([arXiv:1910.11901](https://arxiv.org/abs/1910.11901)) |
| 2 | **Multiagent rollout** with the current greedy policy as base | **High**, and *guaranteed* not worse than base | Medium (needs `Model.fork()`) | Yes — needs a forkable Model, no model changes to physics | Bertsekas Prop. 2.1, `O(s^m)→O(s·m)` ([arXiv:1910.00120](https://arxiv.org/abs/1910.00120)); Ulmer et al. TS 2019 ([DOI](https://doi.org/10.1287/trsc.2017.0767)) |
| 3 | **Batch least-squares MC / LSTD-Q / LSPI** on the same features | Medium — removes lr tuning, scaling pathology, most variance | **Low** | **Yes, unchanged** | Lagoudakis & Parr, JMLR 4 (2003) |
| 4 | **Feature standardisation + adaptive stepsize + common random numbers** | Medium — mostly turns an untunable optimiser into a tunable one | **Very low** | **Yes, unchanged** | Ryzhov/Frazier/Powell ([arXiv:1407.2676](https://arxiv.org/abs/1407.2676)); George & Powell (2006) |
| 5 | **Neural VFA** (2-hidden-layer MLP Q-network, replay, ε-decay, Adam) over enriched features | Medium–High (+22% reported on a comparable SDVRP) | Medium | **Yes, unchanged** | Chen/Ulmer/Thomas ([arXiv:1910.11901](https://arxiv.org/abs/1910.11901)) |
| 6 | **Offline VFA + online rollout hybrid** (2 and 5 combined) | High | High | Yes (needs `Model.fork()`) | Ulmer/Goodson/Mattfeld/Hennig, TS 53(1) 2019 |
| 7 | **Scenario-based / SAA lookahead** over sampled congestion futures | Medium–High | High (needs a cheap forward evaluator) | Yes (reuses `CongestionGenerator`) | Bent & Van Hentenryck, OR 52(6) 2004; Mercier & Van Hentenryck, Ann. OR 2011 |
| 8 | **MCTS at decision time** | Medium | High | Yes (needs `Model.fork()`) | Wilbur et al., ICCPS 2022 ([arXiv:2203.15127](https://arxiv.org/abs/2203.15127)) |
| 9 | **Attention/pointer constructive policy** (AM / POMO, REINFORCE or shared baseline) | **Uncertain, plausibly negative** under this problem's stochasticity | **Very high** | **No** — requires rewriting `Model` as a batched RL4CO env | Kool ICLR 2019; Kwon POMO 2020; SVRPBench degradation >20% ([arXiv:2505.21887](https://arxiv.org/abs/2505.21887)); rl4co env limitations (DeepWiki over [ai4co/rl4co](https://github.com/ai4co/rl4co)) |

**Single highest-leverage change:** #1. Everything else is an estimator improvement on top
of an observation set that omits the problem's defining stochasticity.

---

## Flaws in the current setup independent of algorithm choice

**F1 — The policy is congestion-blind and time-of-day-blind.** *(Severity: critical.)*
Every travel time consumed by the policy — candidate generation (`monte_carlo.py:268-276`),
delayed-client classification (`:321`), and all seven state-action features (`:482-524`) —
comes from `ShortestPathCache.path_between(...).average_minutes`, a static column read
from CSV (`shortest_path_cache.py:38-49`). `_extract_general_state_features` computes
`self.mean_velocities` from `state.observed_velocity` and then **never appends it to the
feature vector** (`monte_carlo.py:444-450`). In the legacy monolith, `mean_velocities` *is*
used in several shadowed/dead feature extractors (e.g. lines 3198, 3303, 3543), and a
time-dependent path evaluator `calculate_real_mean_travel_time` exists at
`Main_Chengdu...py:5064` but appears **only inside commented-out lines** (3355, 3358,
3364). So the live configuration is one in which the simulator models time-dependent
stochastic travel and non-recurring congestion, and the decision rule cannot see either.

**F2 — Zero-initialised `W` makes the first policy arbitrary.** `_create_W` sets
`W = zeros(19)` (`:238-240`), so on the first episode every candidate scores `Q = 0`. The
strict `<` in `_select_best_q_action_for_vehicle` (`:232`) with `min_q_value = inf` means
the *first* candidate always wins — and the candidate list has just been passed through
`list(set(...))` (`:287`), which destroys the nearest-first ordering from
`heapq.nsmallest`. The bootstrap policy is therefore neither "nearest neighbour" nor
anything else principled; it is set-iteration order. A myopic warm start (initialise `W` so
that `Q` equals the immediate projected cost) would be strictly better and costs nothing.

**F3 — Constant step size violates Robbins–Monro.** `lr` is fixed (`monte_carlo.py:89`,
used at `:197`) except for the one-episode warm-up at `1e-6`
(`Main_Chengdu...py:6095, 6098`). Neither `Σα = ∞, Σα² < ∞` condition holds
([Ryzhov/Frazier/Powell](https://arxiv.org/abs/1407.2676)). Combined with unstandardised
features whose magnitudes differ by orders of magnitude, the effective per-component step
size is uncontrolled.

**F4 — The target is non-stationary in two compounding ways.** The behaviour policy is
ε-greedy over `W`, and `W` changes every episode, so `U_t` is drawn from a
policy-dependent distribution that shifts under the estimator; and `ε` never decays
(`monte_carlo.py:167`). This is optimistic approximate policy iteration, which has no
convergence guarantee with function approximation. Chen/Ulmer/Thomas decay ε from 1.0 to
0.01 ([arXiv:1910.11901](https://arxiv.org/abs/1910.11901)); the current code does not
decay it at all.

**F5 — Exploration randomness and exogenous randomness share the global RNG stream.**
`monte_carlo.py:170` draws the exploratory action with `random.choice` from the **global**
`random` module, and `model.py:564` draws arc velocities with `random.gauss` from the same
stream. Consequence: *changing which action is explored changes the traffic realisation for
the remainder of the episode.* This makes common random numbers impossible, makes any
paired comparison of two policies invalid, and injects extra variance into every learning
signal. The module docstring records this as intentional ADR-0001 preservation, which is
correct for Phase 1 — but it must be fixed before any Phase-2 methodology work, and ticket
13 (`.scratch/generic-stdvrp-refactor/issues/13-rng-modernization.md`) is the right home.

**F6 — A permanently-zero feature.** `_extract_state_action_features` appends a literal
`0` as its second component (`monte_carlo.py:479`), padding `W` to 19. Its weight receives
`gradient = lr · (…) · 0 = 0` forever, so `W[13]` is a constant. Harmless numerically,
misleading in any weight-significance analysis.

**F7 — The `future_delay` feature is computed over a duplicated list.** `future_delay`
(`monte_carlo.py:502-515`) iterates `self.vehicle_to_clients[veh]`, which
`_classify_delayed_clients` (`:310-339`) fills *inside* the vehicle loop (`:329-331`), so
each `(travel_time, client)` pair is appended once per remaining vehicle iteration rather
than once per client. The feature is therefore systematically inflated by a factor that
depends on fleet size and on how early the closest vehicle is found — i.e. the feature's
scale is not even consistent across states. The docstring records this as a preserved
quirk; it is a genuine defect in the feature definition, not just in bookkeeping.

**F8 — Redundant recomputation of the most expensive routine.**
`_select_greedy_actions` calls `_classify_delayed_clients()` at `:133`, then
`_extract_general_state_features()` at `:134`, which calls `_classify_delayed_clients()`
again at `:452`. Every decision epoch pays for it twice.

**F9 — Hardcoded normalizers decoupled from the instance.** `clients_left` divides by 150
regardless of the actual client count (`:392`); `late_count` divides by 13 (`:476`);
`time_left`/`time` use literal 1150/850/300 (`:395-396`); the earliness bins use literal
400/500/600 (`:419-424`); the depot-idle cutoffs are `350` in one place (`:254`) and `310`
in three others (`:318`, `:352`, `:374`) while `end_of_horizon` is injected. Any change to
horizon or demand scale silently changes the meaning of the features.

**F10 — Huge terminal penalty dominates the Monte Carlo target.** `model.py:255-261` adds
`40000 - 200·|visited|` on the 1198-minute cutoff. Because `update_W` accumulates that into
`U_t` for **every** epoch in the episode (`monte_carlo.py:190`), one truncated episode
perturbs all `T` gradient steps by a target 2-3 orders of magnitude larger than normal.
This alone can explain unstable `W` trajectories.

**F11 — `total_distance_cost` is zeroed every training step.** `model.py:156` resets it
inside the loop, so a training episode always reports `0` for the distance component.
Recorded as a preserved quirk pending ticket 12; it means distance is invisible in
training diagnostics.

**F12 — Validation-set winner's curse in model selection.** `Best_W` is the checkpoint
minimising mean cost over 50 fixed seeds (`Main_Chengdu...py:6120-6146`), selected across
many checkpoints. With this cost distribution's variance (see F10), selecting the minimum
over many checkpoints on 50 samples will systematically overstate the selected `W`'s true
quality. A held-out test set disjoint from the selection set is needed for any reported
number.

**F13 — The dead TD(0) variant has a sign error and an aliasing bug.** Not live, but worth
recording so nobody resurrects it: `Main_Chengdu...py:5060` updates
`W = W - lr·(R + Q_{t+1} - Q_t)·X` (semi-gradient TD moves `+` toward the target), and
`create_Q_learning_episode` (`:6035-6043`) binds both `state_t` and `state_t_1` to the same
mutable `self.state` object, so the TD target would always have compared a state to itself.
ADR-0001's rule (dead variants stay in the `legacy-monolith` tag) is the right call here.

**F14 — Action-space decomposition is unmodelled.** The per-vehicle greedy argmin treats
other vehicles' current targets as fixed (`monte_carlo.py:249-251`, `:223-224`).
Hildebrandt/Thomas/Ulmer name this exact pattern and call the resulting policies "coarse"
because "the decomposed actions are dependent". *(b) Inference:* this is a real limitation,
but Bertsekas's Proposition 2.1 shows the same decomposition is *sound* when the inner
evaluation is a rollout of a base policy — which is another argument for §5.1.

---

## Open questions / what I could not verify

1. **Kearns & Singh's exact bound.** The COLT 2000 PDF did not parse through my tooling. I
   therefore did **not** state the direction in which the bias and variance terms move with
   `k`/λ from the source. If that direction matters to a decision, read
   <https://www.learningtheory.org/colt2000/papers/KearnsSingh.pdf> directly.
2. **Tsitsiklis & Van Roy (1997)** — bibliographic record verified (IEEE TAC 42, 674-690),
   full text not obtained. I cite it as the standard convergence reference, not for a
   quoted theorem.
3. **Paywalled INFORMS content.** `pubsonline.informs.org` returned HTTP 403 for every
   request. For Ulmer/Goodson/Mattfeld/Hennig (TS 53(1) 2019) I used the TRID record,
   which carries the publisher abstract; for Ulmer/Mattfeld/Köster (TS 52(1) 2018),
   Goodson/Thomas/Ohlmann (TS 50(2) 2016) and Bent & Van Hentenryck (OR 52(6) 2004) I have
   only bibliographic records plus search-surfaced abstracts. **No quantitative result in
   this note is sourced from those four papers.**
4. **Soeffker, Ulmer & Mattfeld, EJOR 298(3):801-820 (2022)** — the prescriptive-analytics
   review — was 403 on ScienceDirect. It would be the best single source for a
   method-selection table; worth obtaining through institutional access.
5. **The 4OR tutorial** "A tutorial on value function approximation for stochastic and
   dynamic transportation" (<https://doi.org/10.1007/s10288-023-00539-3>) redirected to a
   Springer auth wall. *(b)* Given the title it is likely the most directly actionable
   single document for options #3-#6 in the ranking; obtain it.
6. **CASTLE Lab PDF hosting is broken.** `castlelab.princeton.edu` returns "Account
   Suspended" and the `castle.princeton.edu/Papers/...` links listed on the ADP index page
   404. I therefore could not read Powell's *"What you should know about approximate
   dynamic programming"* (NRL 56(3):239-249, 2009) or the George & Powell stepsize paper in
   full; I used Powell's arXiv paper and the RLSO book page instead.
7. **Venue verification for POMO and Nazari et al.** I read the arXiv versions. Both are
   widely cited as NeurIPS 2020 and NeurIPS 2018 respectively, but the pages I read
   identified them only as arXiv preprints, so I cite them as arXiv.
8. **The single closest published match** — "A reinforcement learning approach for the
   dynamic vehicle routing and scheduling problem with stochastic request times and
   time-dependent, stochastic travel times", *Transportation Research Part C*, published
   26 Oct 2025 (<https://www.sciencedirect.com/science/article/pii/S0968090X25003912>) —
   was 403. From the search-surfaced abstract it jointly optimises departure decisions
   under both uncertainties and penalises lateness, waiting and unserved customers, which
   is nearly this repo's objective. **This is the highest-value single paper to obtain.**
9. **SVRPBench is a 2025 arXiv preprint**; I found no peer-reviewed version. Its headline
   result (RL solvers degrade >20% under distribution shift) is the main evidence against
   option #9 and should be weighted accordingly.
10. **Per-decision computation budget.** None of the decision-time methods in §5 can be
    ranked properly without knowing the wall-clock budget per decision epoch in this
    research setting (offline experiments vs. real-time dispatch). That is a question for
    the user, not the literature.

---

## Sources

**Surveys and problem-class framing**

- Hildebrandt, F. D., Thomas, B. W., Ulmer, M. W. (2021). *Where the Action is: Let's make
  Reinforcement Learning for Stochastic Dynamic Vehicle Routing Problems work!*
  arXiv:2103.00507. <https://arxiv.org/abs/2103.00507> · full text via
  <https://ar5iv.labs.arxiv.org/html/2103.00507>
- Hildebrandt, F. D., Thomas, B. W., Ulmer, M. W. (2022). *Opportunities for reinforcement
  learning in stochastic dynamic vehicle routing.* Computers & Operations Research 150.
  <https://www.sciencedirect.com/science/article/abs/pii/S030505482200301X> (403 — record
  only)
- Soeffker, N., Ulmer, M. W., Mattfeld, D. C. (2022). *Stochastic dynamic vehicle routing
  in the light of prescriptive analytics: A review.* EJOR 298(3):801-820.
  <https://www.sciencedirect.com/science/article/pii/S0377221721006093> (403 — record only)

**Powell / stochastic optimisation framework**

- Powell, W. B. (2020). *On State Variables, Bandit Problems and POMDPs.* arXiv:2002.06238.
  <https://arxiv.org/abs/2002.06238>
- Powell, W. B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified
  Framework for Sequential Decisions.* Wiley, 1100 pp. <https://castle.princeton.edu/rlso/>
- Powell, W. B. (2009). *What you should know about approximate dynamic programming.* Naval
  Research Logistics 56(3):239-249. <https://onlinelibrary.wiley.com/doi/10.1002/nav.20347>
  (record only — CASTLE Lab PDFs are offline)
- Ryzhov, I. O., Frazier, P. I., Powell, W. B. (2014). *A New Optimal Stepsize For
  Approximate Dynamic Programming.* arXiv:1407.2676. <https://arxiv.org/abs/1407.2676> ·
  <https://ar5iv.labs.arxiv.org/html/1407.2676>
- George, A. P., Powell, W. B. (2006). *Adaptive stepsizes for recursive estimation with
  applications in approximate dynamic programming.* Machine Learning 65(1):167-198.
  <https://doi.org/10.1007/s10994-006-8365-9> (record only)

**Estimators for linear VFA**

- Lagoudakis, M. G., Parr, R. (2003). *Least-Squares Policy Iteration.* JMLR 4:1107-1149.
  <https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf>
- Kearns, M., Singh, S. (2000). *Bias-Variance Error Bounds for Temporal Difference
  Updates.* COLT 2000, pp. 142-147.
  <https://www.learningtheory.org/colt2000/papers/KearnsSingh.pdf> (PDF did not parse)
- Tsitsiklis, J. N., Van Roy, B. (1997). *An Analysis of Temporal-Difference Learning with
  Function Approximation.* IEEE TAC 42(5):674-690. (record only)
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed.
  MIT Press. <http://incompleteideas.net/book/the-book-2nd.html> (fetch failed: self-signed
  certificate)

**ADP for dynamic/stochastic VRP (Ulmer's line)**

- Ulmer, M. W., Goodson, J. C., Mattfeld, D. C., Hennig, M. (2019). *Offline–Online
  Approximate Dynamic Programming for Dynamic Vehicle Routing with Stochastic Requests.*
  Transportation Science 53(1):185-202. <https://doi.org/10.1287/trsc.2017.0767> · abstract
  via <https://trid.trb.org/view/1583758>
- Ulmer, M. W., Mattfeld, D. C., Köster, F. (2018). *Budgeting Time for Dynamic Vehicle
  Routing with Stochastic Customer Requests.* Transportation Science 52(1):20-37.
  <https://doi.org/10.1287/trsc.2016.0719> (record only)
- Ulmer, M. W., Soeffker, N., Mattfeld, D. C. (2018). *Value function approximation for
  dynamic multi-period vehicle routing.* EJOR 269(3):883-899.
  <https://www.sciencedirect.com/science/article/abs/pii/S0377221718301711> (record only)
- Chen, X., Ulmer, M. W., Thomas, B. W. (2019, rev. 2021). *Deep Q-Learning for Same-Day
  Delivery with Vehicles and Drones.* arXiv:1910.11901. <https://arxiv.org/abs/1910.11901> ·
  <https://ar5iv.labs.arxiv.org/html/1910.11901>
- Ulmer, M. W. et al. *A tutorial on value function approximation for stochastic and dynamic
  transportation.* 4OR (2023). <https://doi.org/10.1007/s10288-023-00539-3> (auth wall)

**Rollout, MCTS, scenario-based lookahead**

- Bertsekas, D. P. (2019, rev. 2020). *Multiagent Rollout Algorithms and Reinforcement
  Learning.* arXiv:1910.00120. <https://arxiv.org/abs/1910.00120> ·
  <https://ar5iv.labs.arxiv.org/html/1910.00120>
- Bertsekas, D. P. (2020). *Rollout, Policy Iteration, and Distributed Reinforcement
  Learning.* Athena Scientific. <https://web.mit.edu/dimitrib/www/Rollout_Complete%20Book.pdf>
  (reference only)
- Goodson, J. C., Thomas, B. W., Ohlmann, J. W. (2016). *Restocking-Based Rollout Policies
  for the Vehicle Routing Problem with Stochastic Demand and Duration Limits.*
  Transportation Science 50(2):591-607. (record only)
- Bent, R., Van Hentenryck, P. (2004). *Scenario-Based Planning for Partially Dynamic
  Vehicle Routing with Stochastic Customers.* Operations Research 52(6):977-987.
  <https://doi.org/10.1287/opre.1040.0124> (record only)
- Bent, R., Van Hentenryck, P. (2007). *Waiting and Relocation Strategies in Online
  Stochastic Vehicle Routing.* IJCAI 2007.
  <https://www.ijcai.org/Proceedings/07/Papers/293.pdf> (PDF did not parse)
- Mercier, L., Van Hentenryck, P. (2011). *An anytime multistep anticipatory algorithm for
  online stochastic combinatorial optimization.* Annals of Operations Research.
  <https://link.springer.com/article/10.1007/s10479-010-0798-7> (record only; orig. CPAIOR
  2008)
- Wilbur, M. et al. (2022). *An Online Approach to Solve the Dynamic Vehicle Routing
  Problem with Stochastic Trip Requests for Paratransit Services.* ICCPS 2022.
  arXiv:2203.15127. <https://arxiv.org/abs/2203.15127>

**Neural constructive policies**

- Kool, W., van Hoof, H., Welling, M. (2019). *Attention, Learn to Solve Routing Problems!*
  ICLR 2019. arXiv:1803.08475. <https://arxiv.org/abs/1803.08475> ·
  <https://ar5iv.labs.arxiv.org/html/1803.08475> · repo
  <https://github.com/wouterkool/attention-learn-to-route>
- Nazari, M., Oroojlooy, A., Snyder, L. V., Takáč, M. (2018). *Reinforcement Learning for
  Solving the Vehicle Routing Problem.* arXiv:1802.04240. <https://arxiv.org/abs/1802.04240> ·
  <https://ar5iv.labs.arxiv.org/html/1802.04240>
- Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Min, S., Gwon, Y. (2020). *POMO: Policy
  Optimization with Multiple Optima for Reinforcement Learning.* arXiv:2010.16011.
  <https://arxiv.org/abs/2010.16011> · <https://ar5iv.labs.arxiv.org/html/2010.16011>
- Mozhdehi, A., Wang, Y., Sun, S., Wang, X. (2025). *SED2AM: Solving Multi-Trip
  Time-Dependent Vehicle Routing Problem using Deep Reinforcement Learning.*
  arXiv:2503.04085. <https://arxiv.org/abs/2503.04085> · <https://arxiv.org/html/2503.04085v1>
- Heakl, A., Shaaban, Y. S., Takac, M., Lahlou, S., Iklassov, Z. (2025). *SVRPBench: A
  Realistic Benchmark for Stochastic Vehicle Routing Problem.* arXiv:2505.21887 (preprint).
  <https://arxiv.org/abs/2505.21887> · <https://arxiv.org/html/2505.21887v2>
- RL4CO. <https://github.com/ai4co/rl4co> — environment capabilities established via
  DeepWiki Q&A over the repository, 2026-07-22.

**Related (not load-bearing)**

- *(authors not verified — the publisher page was 403)* (2025). *A reinforcement learning
  approach for the dynamic vehicle routing and scheduling problem with stochastic request
  times and time-dependent, stochastic travel times.* Transportation Research Part C:
  Emerging Technologies, published 26 Oct 2025.
  <https://www.sciencedirect.com/science/article/pii/S0968090X25003912> (403 — record only;
  see open question 8)
- Horváth, M., Tamási, T. (2024). *A general modeling and simulation framework for dynamic
  vehicle routing.* arXiv:2411.12406. <https://arxiv.org/abs/2411.12406>

---

## Repo files referenced

- `src/stdvrp/policies/monte_carlo.py` — the policy and the MC weight update
- `src/stdvrp/policies/base.py` — the `Policy` interface
- `src/stdvrp/simulation/model.py` — transition function, episode runners, velocity draws
- `src/stdvrp/simulation/episode.py` — evaluation episode wiring
- `src/stdvrp/simulation/state.py` — the State object
- `src/stdvrp/network/shortest_path_cache.py` — the static travel-time source (see F1)
- `Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py` — the legacy monolith
  (`actualize_W` at 4300, the dead TD(0) variant at 5037/6035, `training_model` at 6053)
- `CONTEXT.md`, `docs/adr/0001-*.md`, `docs/adr/0002-*.md` — vocabulary and refactor policy
- `.scratch/generic-stdvrp-refactor/issues/12-deliberate-bug-fixes.md`,
  `13-rng-modernization.md` — the right homes for F1-F13
