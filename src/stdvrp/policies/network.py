"""network: the transformer encoder over ticket-04 tokens, the Q head that scores
(vehicle, Client) pairs, and the myopic base ``c(s, v, a)`` (ticket 05,
superseded by ticket 15, neural-policy).

Ticket 16 adds ``QHead.features``/``w_vector``/``load_w_vector``: the same
``linear(x) + layer2(ReLU(layer1(x)))`` computation :meth:`QHead.forward`
performs, decomposed into a feature vector ``phi(s, v, a)`` and a combined
weight vector ``W`` so that ``forward(...) == features(...) @ w_vector()``
(level term aside -- see "The level term" below, now inert). ``learn``
(``transformer_policy.py``) regresses onto ``features`` with
:class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator`'s closed-form
solve, never onto ``forward``'s autograd graph -- ``linear``/``layer2`` are
never touched by an optimizer any more, only by :meth:`QHead.load_w_vector`.

torch is an optional extra (ticket 03): this module imports it at module scope
because its whole reason to exist is the network, unlike ``torch_support.py``'s
deferred-import boundary. It must therefore never be imported from
``stdvrp.policies.__init__`` or any module reachable from package-import time —
the same discipline :mod:`~stdvrp.policies.tokenizer` follows by *not* needing
torch at all. Callers import it explicitly:
``from stdvrp.policies.network import TokenEncoder, QHead``.

## Shape (spec.md, ticket 05)

``TokenEncoder`` runs once per decision epoch over the concatenated
``{client, vehicle, global}`` token set (a learned type embedding per kind),
producing per-token embeddings. ``QHead`` then runs once per vehicle (the
"sweep") over the client embeddings, scoring every pending Client — cheap,
because the encoder pass is shared. Both classes take an **injected**
``init_rng: np.random.Generator`` (never a global — same discipline ticket 13
of ``generic-stdvrp-refactor`` established for the Episode streams) and use it
for every learned weight, so a fixed seed reproduces bit-identical parameters.

## Why the per-(client, vehicle) "arc" fact gets its own pathway

Ticket 04's ``Tokens.arc_tokens`` is ``[n_pending, m, 6]`` (``m`` = vehicle
count): for every ``(client, vehicle)`` pair, two raw geometry facts
(``minutes_from_vehicle``, ``path_length_from_vehicle``) plus the four
projected cost-function components (``earliness_cost``, ``delay_cost``,
``future_delay``, ``overtime_cost`` — the 2026-08-01 amendment to spec.md
decision 1; see ``tokenizer.py``, "The cost fields"). ``m`` varies across
configs and even across ``test_seeds`` within one trained network's
evaluation sweep (``ExperimentConfig.test_vehicle_counts``), so one wide
``nn.Linear`` over a whole per-client row is not viable — its weight shape
would have to change with ``m``. Instead the 6-wide arc vector is embedded by
a small ``nn.Linear(ARC_TOKEN_WIDTH, d_model)`` applied identically to every
``(client, vehicle)`` pair (weight-shared across ``v``, so parameter count is
independent of ``m``) — an ``m``-agnostic embedding by construction, not a
workaround adopted only for the warm start below. The client's three base
facts go through a separate, ordinary fixed-width embedding and the shared
self-attention transformer; the arc embedding bypasses the transformer
entirely (it is cheap — ``O(n_pending * m)`` linear ops, not attention — and
keeping it un-mixed by cross-token attention is what let the pre-ticket-15
warm start be exact rather than approximate; ticket 15 makes that property
moot by reading ``c`` straight off ``arc_tokens``, before any embedding
happens at all — see "The myopic base" below).

The arc pathway is where the cost features belong structurally: it is the
action-conditional path, and it is **linear**. ``QHead``'s own linear path
composed with ``arc_embed`` spans every linear function of the six raw arc
facts, so the residual ``W · φ(s, v, a)`` can express a near-linear correction
to the projected costs directly, rather than having to rediscover them from
noisy Monte Carlo returns — and since ticket 15, the projected costs
themselves reach ``Q`` by an even shorter, gradient-free path: see "The myopic
base" below.

``Embeddings.clients`` therefore has shape ``[n_pending, m, 2*d_model]``: for
each client, one row per vehicle, formed by concatenating that client's
(vehicle-independent) transformer-refined context embedding with the
(client, vehicle)-specific arc embedding. ``QHead`` takes one vehicle's slice,
``embeddings.clients[:, v, :]``, alongside ``embeddings.vehicles[v]`` — the
per-vehicle sweep (ticket 06) does the slicing, not the head.
``Embeddings.depot`` is the same construction for ticket 06's synthetic depot
candidate — ``arc_embed`` over ``Tokens.depot_arc_tokens``, concatenated with
the **vehicle's own** context embedding (the depot's meaning is "return to
base", a fact about the vehicle — ADR-0007) — built here so the depot's arc
half goes through the identical pathway as every real candidate's, rather
than being hand-assembled by the Policy from a separate geometry read.
``Embeddings.cost``/``Embeddings.depot_cost`` (ticket 15) are the myopic base
``c`` for the same two candidate sets, built from the same two token tensors
but bypassing ``arc_embed`` entirely — see "The myopic base" below.

## The myopic base (this module's load-bearing part; ticket 15)

Ticket 05 built ``Q``'s initial value as a *warm start*: hand-set weights that
made the untrained network's output happen to equal a cheap myopic estimate,
trainable away from the very first gradient step. Ticket 08 then measured four
independently-arrived-at techniques — this module's "A dueling decomposition"
and "The level term" sections below, the Huber knee in
``transformer_policy.py``, and the ``"cost"`` warm start itself — and read them
as four separate results when they share one signature: each made the
optimizer more effective, and each only helped when the starting policy was
bad. That is one finding about where ``c(s, a)`` lived — in ordinary trainable
weights, so an optimizer with a bigger step unlearns it faster — not four
findings about three techniques. Ticket 15 removes the coupling structurally
instead of managing it with a bigger lever:

    Q(s, v, a) = c(s, v, a)  +  W · φ(s, v, a)
                 ↑ tokenizer     ↑ the only term with parameters, W = 0 at init

``c(s, v, a)`` — **the myopic base** — is the projected cost the tokenizer
already computes exactly for candidate ``a``: ``arc_tokens`` (or
``depot_arc_tokens`` for the synthetic depot row) dotted with
:data:`~stdvrp.policies.tokenizer.WARM_START_WEIGHTS`\\ ``["cost"]`` — the same
six-field vector ticket 08 introduced, now read directly off the token instead
of copied into a row of ``arc_embed``. :meth:`TokenEncoder.forward` computes
it alongside :class:`Embeddings`' other tensors (``Embeddings.cost``,
``Embeddings.depot_cost``) straight from ``arc_tokens``/``depot_arc_tokens`` —
plain ``torch.from_numpy`` tensors, never a ``Parameter`` — so ``c`` carries no
gradient by construction, not by convention: there is no weight anywhere on
its path for an optimizer to reach, and it is in no ``state_dict``. The depot
row additionally adds :data:`DEPOT_WARM_START_PENALTY` — see "The depot's
place in ``c``" below. Callers (``transformer_policy.py``'s ``_score``) add
``c`` to :class:`QHead`'s output themselves: "outside ``QHead``" is literal,
not a figure of speech — the class never receives ``c`` as an argument at all.

``QHead`` keeps the same forward signature and the same two-branch shape it
always had (``linear(x) + layer2(ReLU(layer1(x)))``); only ``_init_weights``
changes, and it gets simpler rather than more special-cased. Previously
``linear`` was zeroed *except* for two hand-set columns (the reconstruction of
``c`` and the depot penalty); now ``linear.weight``, ``linear.bias``,
``layer2.weight`` and ``layer2.bias`` are *all* zero, with no exception.
``layer1`` stays Xavier-random, for the same reason it always did — see "Why
``layer1`` gets real random weights, not zero" below, unchanged by this
ticket. The result is ``W · φ(s, v, a) == 0`` for every candidate, every
vehicle, every state, at construction: not approximately, and not because the
arc facts happen to be small, but because the two weights multiplying every
path into ``QHead``'s output are the zero tensor. ``Q(s, v, a) == c(s, v, a)``
at init follows for free, regardless of what ``arc_embed``, the transformer
layers, or ``layer1`` compute — which is what makes the equivalence to
ticket 14's frozen null exact rather than numerical coincidence (see
"Verified against ticket 14's frozen null" below).

``arc_embed`` row 0 is no longer special: ordinary Xavier-uniform, exactly
like ``client_base_embed``/``vehicle_embed``/``global_embed`` and every other
row of ``arc_embed`` itself. Nothing downstream depends any more on any
specific embedding dimension matching a raw fact, so ``TokenEncoder`` no
longer needs an identity-at-init encoder either — see "Real attention at
initialization" below.

``c(s, v, a)`` is a **myopic base**, not a warm start: a warm start is a point
optimization moves away from, and nothing here moves away from ``c`` — it is
structurally unreachable, not merely untouched so far. It is also **not**
Powell's post-decision state, despite ``CONTEXT.md`` following Powell's
vocabulary elsewhere: ``c`` is built from ``tau``, the time windows and
``EpisodeGeometry.average_minutes`` — a *static historical prior* baked into
the geometry at capture time, not an observation of this Episode (ADR-0006) —
so it is a *projection*, not the immediate cost the simulator will actually
charge for taking action ``a`` from state ``s``. ``Q = c + W·φ`` is a residual
VFA over a known myopic base: ``W·φ`` absorbs both the future cost-to-go *and*
the error in the projection, together, exactly as any residual value function
does.

## Real attention at initialization (ticket 15)

Every ``nn.TransformerEncoderLayer``'s ``self_attn.out_proj`` and FFN
``linear2`` used to be zeroed (weight **and** bias), which made each residual
branch evaluate to an exact-zero tensor regardless of what fed it, and hence
made the whole ``TransformerEncoder`` an exact identity map at init — the only
way to guarantee ``client_context`` equalled a plain, transformer-untouched
linear embedding of the client's three base facts, which the pre-ticket-15
warm start needed in order to be exact rather than approximate.

That guarantee is no longer needed: nothing downstream reads
``client_context``, or anything else the encoder computes, to reconstruct
``c`` any more — ``c`` is read straight off the raw arc token, before any
embedding happens at all (previous section). So ``out_proj``/``linear2`` are
now ordinary Xavier-random, like every other weight in this module, and the
encoder performs real, nontrivial attention from the very first forward
pass — a random feature map over the token set, not a linear embedding wearing
a transformer's clothes. This is what ticket 17's frozen-encoder arm needs to
be a meaningful arm at all: fixing the encoder at init only tests something
once "at init" is a real function of the input, not the identity.

## The depot's place in ``c`` (formerly "The depot is the last resort at init")

``QHead`` scores :class:`Embeddings`' ``clients`` tensor — one row per pending
Client — so the depot, never a Client, has no natural row. ``TokenEncoder``
builds one for it the same way it builds every real candidate's: ``c(s, v,
depot)`` is ``depot_arc_tokens[v]`` dotted with the same
``WARM_START_WEIGHTS["cost"]`` vector, **plus** :data:`DEPOT_WARM_START_PENALTY`
— one whole horizon, in the ``minutes / horizon_length`` currency every other
field of ``c`` already speaks. Every Client reachable within the shift scores
at most ``1`` in that currency, so the depot loses the pre-training argmin to
any feasible Client and only wins when nothing else survives ``_sweep``'s
candidate filter — "nearest feasible Client, home only when no Client is
feasible", the null model spec.md specifies.

Ticket 08's Gate A run measured what happens without this: the untrained
network at mean cost 81 701 against the linear baseline's 2 483, because every
vehicle starts parked at the depot and ``minutes_to_depot == 0`` there, so
every vehicle's argmin was the depot at decision epoch 1 and the whole fleet
retired on the first transition (``Model._reroute_for`` only reroutes a
*travelling* vehicle, so parking is irreversible).

``is_depot`` remains an ordinary ``QHead`` input (1.0 on the synthetic depot
row, 0.0 on every Client, concatenated into ``x`` beside ``claimed``) — the
penalty now lives entirely in ``c``, so ``is_depot`` no longer needs to reach
any hand-set weight to do its job. It is not a dead argument for all that:
exactly like ``claimed`` (``TestClaimedIsWired``,
``tests/unit/test_network.py``), ``layer1``'s Xavier-random rows read it from
the first forward pass, so a trained network is free to price going home
however the returns say it should — including below every Client near the
shift end — the moment training moves ``layer2``'s corresponding column off
zero. Nothing here masks the depot out of the action set: ADR-0007's
"feasible = every pending Client not already claimed, plus the depot" (as
amended by ADR-0011) is untouched.

## Verified against ticket 14's frozen null

Ticket 14 measured the untrained ``"cost"`` warm start (the pre-ticket-15
architecture, ``DEPOT_WARM_START_PENALTY = 1.0``) at ``m + 2`` over the 50
``evaluation_seeds``: mean cost **3365.09**
(``.scratch/neural-policy/results/action_set_m2_50.py``), the frozen null this
effort reports every trained number against. Because ``Q(s, v, a) ==
c(s, v, a)`` at init *by construction* (the two sections above), and ``c(s, v,
a)`` is defined as exactly the quantity the old ``"cost"`` warm start computed
(the same ``WARM_START_WEIGHTS["cost"]`` dot product, the same
``DEPOT_WARM_START_PENALTY`` addition on the depot row), the two architectures
compute the identical scalar for every candidate of every decision epoch —
independent of the random weights either one draws, since neither
architecture's random weights reach ``Q`` at init. Confirmed by re-running
``action_set_m2_50.py``'s arm 1 against this module on both the mini fixture
and the real dataset — see this ticket's Comments
(``.scratch/neural-policy/issues/15-the-myopic-base.md``) for the run.

## A dueling decomposition of ``Q`` — tried, measured, and rejected (ticket 08)

Recorded so it is not re-tried naively, because the *diagnosis* behind it is
sound and will suggest it again.

``Q(s, a) = V(s) + A(s, a)``, and **only ``A`` reaches the decision** — the
per-vehicle ``argmin`` reads differences between candidates of one sweep, so
any quantity shared by all of them is invisible to it. Training sees only the
sum: ``learn`` regresses ``sum_v Q(s, v, a_v)`` onto the Monte Carlo return
``U_t`` (``transformer_policy.py``, "One sample per decision epoch"), one
scalar per decision epoch. The gradient reaching *every* candidate term of
that epoch is the same scalar residual, so the loss is completely invariant to
how a given sum is split between candidates — and the split is the only thing
the ``argmin`` reads. At 595k parameters the encoder fits ``V(s)`` easily,
after which the residual is noise and the arc's cost weights are estimated
from noise.

The standard remedy for that symptom (Wang et al., "Dueling Network
Architectures", 2016) is to make the split structural:

    Q(s, v, a) = V(s, v) + [A(s, v, a) - mean over this sweep's candidates of A]

It was implemented exactly so — ``V`` an MLP over the candidate-set mean of the
same input rows, zero-initialised so the null model was provably untouched
(measured: identical null mean on the mini fixture, ``577.22`` with and
without) — and it made learning **much worse**. On the mini fixture, 200
episodes, paired against the same null: mean over 20 blocks ``-2.42%`` →
``+10.73%``, blocks worse than null 3/20 → 15/20, end of run ``-2.16%`` →
``+34.00%``. Scaling ``V``'s output by 10 (a per-step speedup on that branch,
testing whether ``V`` was simply too slow to absorb the level) recovered the
best single block of any arm measured, ``-7.28%`` at episode 10, but not the
run: mean ``+2.44%``, end ``+23.76%``.

**Why it backfires, which is the part worth keeping.** The ``argmin`` picks the
*minimum* candidate, so the chosen action's centred advantage is negative while
the target ``U_t`` is positive. The residual therefore pushes that advantage
*up, toward the candidate mean* — it actively un-learns that this action was
the best one, every step, until ``V`` catches up. Without the centring the same
pressure raises ``A`` for all candidates near-equally (they share parameters),
which is a level shift the ``argmin`` cannot see. **The centring removed a
benign escape valve and converted a level error into ranking damage.** The
level error is large for a structural reason: ``Q`` starts in "minutes /
horizon_length" units because of the warm start, and the return lives on an
unrelated scale, so the optimizer's first job is reconciling two arbitrary
scales — and it pays for it with the ranking. That is the same "fitted the mean
and threw the ranking away" signature ``transformer_policy.py`` records under
"Target scaling", one layer deeper.

Anything that revisits this has to fix the scale mismatch *first*, not add a
term that makes the mismatch land on the advantage.

**Why this section is moot now, and why the finding still stands (ticket
15).** The argmin's blindness to any quantity shared by every candidate is a
property of the argmin, not of where ``c`` lives, so it is unaffected by
ticket 15 and the "training sees only the joint sum" problem above is still
real (``transformer_policy.py``, "What the joint sum costs" — ticket 16's
problem to solve, not this one's). What ticket 15 removes is the specific
trigger that made centring backfire *here*: the "large, same-signed
correction" was large only because the pre-ticket-15 warm start pinned
``Q_joint`` at init to 0.3-0.9 against a ~0.03 target (next section) — an
artifact of the warm start sharing ``arc_embed``'s scale rather than the
return's. Under ``Q = c + W·φ``, ``Q_joint`` at init is ``Σ_v c(s, v, a_v)``, a
cost projection already living on roughly the scale of ``U_t`` (the ``"cost"``
warm start's own episode costs, ticket 08, were the same order of magnitude as
the trained linear baseline's). The specific failure mode measured above has
no obvious trigger left, which is a reason to *reconsider* a dueling split —
not a demonstration that it now works. This has not been re-measured.

## The level term, and why it needs a gain (``level_gain``, ticket 08)

That scale mismatch, measured. At initialization ``Q`` is a sum of
minute-normalised quantities: with ``m`` vehicles, ``Q_joint = sum_v Q(s, v,
a_v)`` lands around **0.3-0.9** under the ``"cost"`` warm start. The target
``learn`` regresses it onto is ``U_t / (number_clients * episode_length)``,
which for a Chengdu episode costing ~3800 is **0.03 at ``t = 0`` and decays to
0**. The two are an order of magnitude apart, and the first training episode's
logged loss says so exactly: ``0.5 * 0.8^2 = 0.32``, against a logged
``3.2e-01``.

So before the network can learn anything about *ranking*, it has to move
``Q_joint`` down by 0.3-0.9 — while the differences between candidates that
the ``argmin`` reads are ~0.03. The correction is 10-30x larger than the
signal **and it has the same sign at every step**, so Adam (whose step size is
~``lr`` regardless of gradient magnitude) walks in a straight line for as long
as it takes, dragging every weight it touches. Measured on the real dataset
with the ``"cost"`` warm start: 50 episodes took the policy from its untrained
3811 to 6169 on the ``test_seeds``, +62% *worse* than its own null, with
training-episode costs climbing monotonically 6-8k -> 22-50k.

There is exactly one weight in this head that is added to every candidate of
every sweep identically, and therefore the only one that can move ``Q``'s
magnitude without touching a single ``argmin``: ``linear``'s **bias**. It is
already trainable and already receives precisely this gradient — it is simply
too slow, because it moves at ~``lr`` per step like everything else. Travelling
the ~0.08 it needs at ``lr = 3e-5`` takes ~2700 steps, which at this fixture's
~27 steps per episode is ~100 episodes. **That is the exact window in which the
damage happens.**

:attr:`QHead.level_gain` multiplies that one weight's effective contribution,
so one optimizer step moves the level ``level_gain`` times as far. At 100 the
level converges inside the first episode instead of the hundredth. It is
written as an increment on top of ``self.linear(x)`` — ``(level_gain - 1.0) *
linear.bias`` — so that the default of ``1.0`` adds exactly ``0.0`` and is
bit-identical to this term not existing, and so that the parameter, its
meaning and the ``state_dict``'s keys are all unchanged.

This is what the dueling attempt above should have been. Both are "give the
level its own fast home"; the difference is that a *centred advantage* puts
the correction somewhere the ``argmin`` reads, and a *bias* puts it in the one
direction the ``argmin`` is blind to.

**Measured, and the result is conditional — read this before setting it.** On
the mini fixture (200 episodes, paired against the same untrained null), a gain
of 100 on the ``"minutes"`` warm start improves every column: best block
``-5.68% -> -7.44%``, mean over 20 blocks ``-2.42% -> -2.79%``, mean of the
last five ``-1.84% -> -3.03%``, blocks worse than the null ``3/20 -> 2/20``.
That is the best learning behaviour measured anywhere in this effort.

On the ``"cost"`` warm start the *same* gain makes things worse: mean over
blocks ``+2.57% -> +5.31%``, blocks worse than the null ``14/20 -> 16/20``.
The reading is uncomfortable but consistent with everything else here — the
gain does not make the ranking signal any less noisy, it just removes what was
throttling the optimizer. Starting from a poor ranking (nearest-neighbour)
that is an improvement, because there is something to find. Starting from a
good one (cost-greedy) it only lets the noise do damage sooner. **Training is
not adding value on top of a good initialization; it is spending it.**

So the default is ``1.0``, ``experiments/chengdu/config.yaml`` leaves it there
alongside ``"cost"``, and a Gate A run whose question is specifically "does it
learn" should use ``"minutes"`` with a gain of 100 — the configuration in
which the answer is most clearly yes.

**Why this section is moot now (ticket 15).** The 0.3-0.9-vs-0.03 mismatch
this whole section is built to correct was an artifact of the pre-ticket-15
warm start: ``Q_joint`` at init summed ``m`` copies of a *bare, unrelated*
travel-time fraction, on a scale the target return had no reason to share.
Under ``Q = c + W·φ``, ``Q_joint`` at init is ``Σ_v c(s, v, a_v)`` — a cost
*projection*, already on the scale of the return it approximates (the
``"cost"`` warm start's own untrained episode costs, measured throughout this
effort, were the same order of magnitude as a trained policy's, not 10-30x
off). There is no longer a large, same-signed gap for ``level_gain`` to
close, so the mechanism above has little left to do regardless of the value
chosen. **Ticket 16 removes what was left**: the per-episode Adam SGD loop
this section's "~100 episodes of same-signed steps" story depends on is gone
-- ``linear``/``layer2`` are now written in one step by
:meth:`QHead.load_w_vector` off a closed-form solve, never walked to, so
there is no longer any optimizer step for ``level_gain`` to scale the size
of. The field stays wired, at its measured bit-identical-to-absent default
(:attr:`QHead.forward`'s ``level`` term is `0.0` whenever it is `1.0`), purely
so a config file or a script that still sets it does not need to change; it
is dormant rather than deleted for the same citability reason as
``neural_warm_start``/``neural_huber_delta``.

## Why a dead ReLU can no longer touch ``c`` (formerly "the warm start is on a linear path")

The pre-ticket-15 warm start put its two hand-set weights on ``linear`` rather
than behind ``layer1``'s ReLU specifically to survive training: an earlier
draft put them behind the ReLU instead, and **it died in one training
episode.** ``hidden[0] = ReLU(w0 · x)`` with ``w0`` zeroed except for the two
warm-start weights meant the unit's pre-activation was ``minutes_from_vehicle
/ horizon_length`` — measured on the mini fixture, a number in ``[0.002,
0.061]``. The other 384 columns of ``w0`` start at exactly zero but are
ordinary parameters, and the ``[vehicle | client_context]`` prefix they
multiply has L2 norm ~2.9, so Adam's normalised steps (which move every weight
by ~``lr`` per step regardless of how small its gradient is) put a
perturbation on that pre-activation far larger than the signal it carries.
Measured, one episode of training moved it from ``[+0.003, +1.000]`` to
``[-0.546, -0.077]``: **dead for every candidate**. A dead ReLU has exactly
zero gradient, so ``w0`` could never come back — the myopic prior, the depot
penalty and the ranking they encoded were gone permanently after episode 1,
and the spread of ``Q`` across a vehicle's candidates collapsed from 0.036 to
0.004 with the argmin no longer picking the nearest Client. Putting the two
weights on ``linear`` instead (no kink) fixed it: the same weights stayed
trainable but degraded continuously and recovered, instead of being
annihilated by the first optimizer step that overshot.

Ticket 15 makes the whole class of failure structurally unreachable rather
than merely survivable. ``c`` is not behind an activation, linear or
otherwise — it is not inside ``QHead`` at all, added by the caller after
``QHead`` returns (see "The myopic base" above). No weight of any kind sits
between ``c`` and ``Q``, so there is nothing for a dead ReLU, an optimizer
overshoot, or any other training pathology to gate off. The finding above
stays true and citeable for ``W · φ`` itself — a dead ``layer1`` unit is still
a dead unit, and the deadlock note just below still matters for exactly that
reason — it simply no longer has anything to say about ``c``, which is the
finding worth keeping this section around for.

## Why ``layer1`` gets real random weights, not zero

An earlier draft zeroed the *entire* first layer, reasoning that units
contributing nothing at init can stay dormant. That is a genuine dead end, not
a transient one: with ``layer1.weight[row, :] == 0`` identically,
``hidden[row]`` is exactly ``0``
for *every* input, so ``d(loss)/d(layer2.weight[0, row]) = hidden[row] *
d(loss)/d(Q) == 0`` *always* — ``layer2``'s columns never move, so
``d(loss)/d(hidden[row])`` (which routes back through ``layer2.weight[0,
row]``) never moves either, and ``layer1``'s rows are frozen at zero forever.
Xavier-random ``layer1`` weights break this: ``hidden[row]`` is then a real,
input-dependent, generally nonzero value, so ``layer2``'s corresponding column
*does* receive a nonzero gradient from the first backward pass (its gradient
formula never depended on its own current value being nonzero — only on
``hidden[row]`` and the downstream loss gradient, both already nonzero) and
moves off zero immediately; ``layer1``'s row then unlocks starting the
following step, once ``layer2.weight[0, row]`` is no longer exactly zero.
(This is the same mechanism as zero-gamma residual-block initialisation in
ResNet-style networks: zeroing a block's *final* projection is safe and
standard; zeroing *everything feeding into it too* is not.) The exactly-zero
pieces in this module (``QHead.linear`` and ``QHead.layer2``, since ticket 15
— see "The myopic base" above) are of the "safe" kind — a real, nonzero,
input-dependent quantity feeds them from upstream, so their own gradient is
nonzero from step one even though their forward output starts at zero.

Note the contrast with the dead-ReLU failure above, which looks superficially
similar and is the opposite case. A zeroed *output* projection is safe because
its gradient depends on its input, not on itself. A zeroed *input* row behind a
ReLU whose pre-activation can be pushed negative is unsafe, because the ReLU
gates the gradient off entirely and nothing upstream can switch it back on.

## Reproducibility and determinism

Every learned weight is drawn from ``init_rng: np.random.Generator`` via
:func:`_xavier_uniform_` (computed with plain numpy, then copied into the
``nn.Parameter``) — never from torch's global default generator, which
``nn.Linear.reset_parameters()`` would otherwise silently consume. Biases are
always zero (a deliberate simplification, not a numerical requirement: a
bias's own gradient never depends on its current value, so zero-initialising
every bias costs nothing and needs no extra draws). Two ``TokenEncoder``s (or
``QHead``s) built from freshly-seeded generators with the same seed are
therefore bit-identical.

``dropout=0.0`` everywhere (no config knob for it exists yet, and none of
tickets 04-06 ask for one): a ``TransformerEncoderLayer`` with no dropout has
no stochastic op in its forward pass at all, so two forward calls on the same
instance — training mode or eval — agree bit for bit on CPU without needing
``torch.use_deterministic_algorithms``. CUDA's kernel-selection nondeterminism
(e.g. attention-backend choice) is a separate, real concern the ticket calls
out; :func:`~stdvrp.policies.torch_support.resolve_device` handles it once, for
every torch caller, rather than here.

## Device (ticket 12)

Both classes take an optional ``device: torch.device`` (default CPU, so every
existing direct-construction call site — the unit tests among them — keeps
working unchanged). Weight init (``_init_weights``) always draws through numpy
and copies onto CPU-resident parameters first, exactly as before; ``__init__``
then moves the whole module onto ``device`` in one ``self.to(device)`` call, so
init reproducibility (same seed -> bit-identical parameters) is unaffected by
which device the caller asked for. ``TokenEncoder.forward`` additionally moves
its three ``torch.from_numpy`` token tensors onto ``self.device`` explicitly —
the one place this module builds a tensor that ``self.to(device)`` does not
already cover, since it is created fresh on every call, not a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from stdvrp.policies.tokenizer import (
    ARC_TOKEN_WIDTH,
    CLIENT_TOKEN_BASE_WIDTH,
    GLOBAL_TOKEN_WIDTH,
    VEHICLE_TOKEN_BASE_WIDTH,
    WARM_START_WEIGHTS,
    Tokens,
)

__all__ = ["DEPOT_WARM_START_PENALTY", "Embeddings", "QHead", "TokenEncoder"]

#: How much the myopic base ``c`` adds to the depot candidate, in the units
#: every other field of ``c`` speaks (``minutes / horizon_length``): one whole
#: horizon. Any Client reachable within the shift scores below that, so the
#: untrained greedy policy is "nearest feasible Client, home only when no Client
#: is feasible" -- see this module's docstring, "The depot's place in `c`".
#: Read directly by ``TokenEncoder.forward`` when it builds ``Embeddings.
#: depot_cost`` (ticket 15) -- not a ``QHead`` weight any more, so an optimizer
#: can never move it; kept, at the value ticket 14 measured, rather than a
#: floor baked into the tokenizer.
DEPOT_WARM_START_PENALTY = 1.0

#: Per-candidate scalars ``QHead`` reads alongside the embeddings, in order:
#: ``claimed`` then ``is_depot``.
CANDIDATE_FLAG_WIDTH = 2


# Type-embedding indices (Embeddings' three token kinds).
_TYPE_CLIENT = 0
_TYPE_VEHICLE = 1
_TYPE_GLOBAL = 2
_N_TOKEN_TYPES = 3


def _xavier_uniform_(tensor: torch.Tensor, rng: np.random.Generator) -> None:
    """Xavier/Glorot-uniform init, drawn from ``rng`` (never torch's global generator).

    ``tensor`` must be 2D (``[fan_out, fan_in]``, the ``nn.Linear``/attention
    projection convention) or 1D (embedding-style, no fan-in/out — bounded like
    a fan_in-only Xavier row of width ``tensor.shape[0]``).
    """
    shape = tuple(tensor.shape)
    if tensor.dim() == 2:
        fan_out, fan_in = shape
    else:
        fan_in = fan_out = shape[0]
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    values = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(values))


def _zero_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.zero_()


@dataclass(frozen=True, slots=True)
class Embeddings:
    """One decision epoch's post-encoder embeddings — everything ``QHead`` needs."""

    clients: torch.Tensor
    """``[n_pending, m, 2*d_model]``: dims ``[:d_model]`` are the client's
    vehicle-independent context (transformer-refined), dims ``[d_model:]`` are
    the ``(client, vehicle)``-specific arc embedding. Sliced per vehicle by the
    caller: ``embeddings.clients[:, v, :]``."""

    vehicles: torch.Tensor
    """``[m, d_model]``, ``last_node_reached`` order (matches ``vehicle_tokens``
    and the vehicle axis of ``clients`` above)."""

    depot: torch.Tensor
    """``[m, 2*d_model]``: the synthetic depot candidate's row per vehicle —
    the vehicle's own context embedding concatenated with the arc embedding of
    ``Tokens.depot_arc_tokens[v]`` (module docstring; ADR-0007, "The depot's
    Q value"). Same layout as one row of a per-vehicle ``clients`` slice, so
    ``QHead`` scores it like any other candidate."""

    cost: torch.Tensor
    """``[n_pending, m]``: the myopic base ``c(s, v, client)`` (module
    docstring, "The myopic base") — ``arc_tokens`` dotted with
    ``WARM_START_WEIGHTS["cost"]``, bypassing ``arc_embed`` entirely. Plain
    arithmetic on a ``torch.from_numpy`` tensor: no gradient reaches it, it is
    not a ``Parameter``, and it is in no ``state_dict``. Added to ``QHead``'s
    output by the caller, never read by ``QHead`` itself."""

    depot_cost: torch.Tensor
    """``[m]``: ``c(s, v, depot)`` — the same dot product over
    ``Tokens.depot_arc_tokens``, plus :data:`DEPOT_WARM_START_PENALTY` (module
    docstring, "The depot's place in `c`"). Same non-gradient, non-``Parameter``
    discipline as ``cost`` above."""


class TokenEncoder(nn.Module):
    """Runs once per decision epoch: tokens -> :class:`Embeddings`.

    See the module docstring for the full shape rationale and the myopic-base
    construction (``Embeddings.cost``/``depot_cost``) this class computes
    alongside the trainable embeddings ``QHead`` scores.
    """

    # Declared here (not just assigned via register_buffer in __init__) so
    # mypy resolves attribute access in forward() to Tensor, not the
    # Tensor | Module union nn.Module.__getattr__ is typed with.
    _myopic_base_weights: torch.Tensor

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_observed_velocities: int,
        init_rng: np.random.Generator,
        dim_feedforward: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.device = device if device is not None else torch.device("cpu")
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model

        self.client_base_embed = nn.Linear(CLIENT_TOKEN_BASE_WIDTH, d_model)
        self.vehicle_embed = nn.Linear(VEHICLE_TOKEN_BASE_WIDTH + n_observed_velocities, d_model)
        self.global_embed = nn.Linear(GLOBAL_TOKEN_WIDTH, d_model)
        self.arc_embed = nn.Linear(ARC_TOKEN_WIDTH, d_model)
        self.type_embedding = nn.Parameter(torch.empty(_N_TOKEN_TYPES, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # The myopic base's weight vector (module docstring, "The myopic
        # base"): a plain buffer, not a Parameter, so it never appears in
        # .parameters()/state_dict() (persistent=False) and never receives a
        # gradient. Registered once here rather than rebuilt every forward
        # call, so it moves with .to(device) like everything else.
        self.register_buffer(
            "_myopic_base_weights",
            torch.tensor(WARM_START_WEIGHTS["cost"], dtype=torch.float32),
            persistent=False,
        )

        self._init_weights(init_rng)
        self.to(self.device)

    def _init_weights(self, rng: np.random.Generator) -> None:
        for embed in (self.client_base_embed, self.vehicle_embed, self.global_embed):
            _xavier_uniform_(embed.weight, rng)
            _zero_(embed.bias)

        # arc_embed: ordinary Xavier-random, every row -- ticket 15 retired the
        # hand-set row 0 that used to carry WARM_START_WEIGHTS (the myopic base
        # ``c`` is now read straight off ``arc_tokens`` in forward() below,
        # never through this embedding).
        _xavier_uniform_(self.arc_embed.weight, rng)
        _zero_(self.arc_embed.bias)

        _xavier_uniform_(self.type_embedding, rng)

        for encoder_layer in self.transformer.layers:
            attn = encoder_layer.self_attn
            _xavier_uniform_(attn.in_proj_weight, rng)
            _zero_(attn.in_proj_bias)
            # Ticket 15: ordinary Xavier-random, every projection -- the
            # identity-at-init construction this module used to need (module
            # docstring, "Real attention at initialization") retired along
            # with the warm start it existed to serve.
            _xavier_uniform_(attn.out_proj.weight, rng)
            _zero_(attn.out_proj.bias)
            _xavier_uniform_(encoder_layer.linear1.weight, rng)
            _zero_(encoder_layer.linear1.bias)
            _xavier_uniform_(encoder_layer.linear2.weight, rng)
            _zero_(encoder_layer.linear2.bias)
            # norm1/norm2: PyTorch's own default (weight=1, bias=0) is already
            # deterministic, not random -- nothing to draw from init_rng here.

    def forward(self, tokens: Tokens) -> Embeddings:
        client_tokens = torch.from_numpy(tokens.client_tokens).float().to(self.device)
        vehicle_tokens = torch.from_numpy(tokens.vehicle_tokens).float().to(self.device)
        global_token = torch.from_numpy(tokens.global_token).float().to(self.device)
        arc_tokens = torch.from_numpy(tokens.arc_tokens).float().to(self.device)
        depot_arc_tokens = torch.from_numpy(tokens.depot_arc_tokens).float().to(self.device)

        number_vehicles = vehicle_tokens.shape[0]

        client_in = self.client_base_embed(client_tokens) + self.type_embedding[_TYPE_CLIENT]
        vehicle_in = self.vehicle_embed(vehicle_tokens) + self.type_embedding[_TYPE_VEHICLE]
        global_in = self.global_embed(global_token) + self.type_embedding[_TYPE_GLOBAL]

        sequence = torch.cat([client_in, vehicle_in, global_in.unsqueeze(0)], dim=0)
        encoded = self.transformer(sequence.unsqueeze(0)).squeeze(0)

        n_pending = client_in.shape[0]
        client_context = encoded[:n_pending]
        vehicle_context = encoded[n_pending : n_pending + number_vehicles]

        arc = self.arc_embed(arc_tokens)  # [n_pending, number_vehicles, d_model]
        clients = torch.cat(
            [client_context.unsqueeze(1).expand(-1, number_vehicles, -1), arc], dim=-1
        )  # [n_pending, number_vehicles, 2*d_model]
        depot = torch.cat(
            [vehicle_context, self.arc_embed(depot_arc_tokens)], dim=-1
        )  # [number_vehicles, 2*d_model]

        # The myopic base (module docstring, "The myopic base"): read straight
        # off the raw arc tokens, never through arc_embed -- no Parameter, no
        # gradient, on either path. cost/depot_cost carry no gradient because
        # arc_tokens/depot_arc_tokens (plain torch.from_numpy tensors) and
        # _myopic_base_weights (a non-persistent buffer) are both leaves with
        # requires_grad=False; the product of two such leaves is too.
        cost = (arc_tokens * self._myopic_base_weights).sum(dim=-1)  # [n_pending, m]
        depot_cost = (depot_arc_tokens * self._myopic_base_weights).sum(
            dim=-1
        ) + DEPOT_WARM_START_PENALTY  # [m]

        return Embeddings(
            clients=clients,
            vehicles=vehicle_context,
            depot=depot,
            cost=cost,
            depot_cost=depot_cost,
        )


class QHead(nn.Module):
    """Scores one vehicle against every pending Client: the residual
    ``W · φ(s, v, a)``, **not** the final ``Q`` — a **cost to minimize**, added
    to (never a substitute for) the myopic base ``c(s, v, a)`` the caller reads
    off :class:`Embeddings`' ``cost``/``depot_cost`` (module docstring, "The
    myopic base"; ticket 15). ``QHead`` itself never sees ``c`` and has no way
    to reach it.

    Matches the baseline's ``argmin`` convention (spec.md) — the caller (ticket
    06) takes ``argmin`` over ``c + QHead(...)`` for every feasible candidate,
    exactly as ``MonteCarloPolicy`` does over its own scores. Never flip this
    sign.
    """

    def __init__(
        self,
        *,
        d_model: int,
        init_rng: np.random.Generator,
        hidden_dim: int | None = None,
        device: torch.device | None = None,
        level_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if level_gain <= 0.0:
            raise ValueError(f"level_gain must be positive, got {level_gain}")
        self.device = device if device is not None else torch.device("cpu")
        # See the module docstring, "The level term, and why it needs a gain"
        # (moot since ticket 15 -- kept wired at its measured default).
        self.level_gain = level_gain
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        client_dim = 2 * d_model  # Embeddings.clients' per-vehicle slice width
        input_dim = d_model + client_dim + CANDIDATE_FLAG_WIDTH  # + claimed, is_depot
        self.linear = nn.Linear(input_dim, 1)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        self._init_weights(init_rng)
        self.to(self.device)

    def _init_weights(self, rng: np.random.Generator) -> None:
        # W = 0 at init (module docstring, "The myopic base"): linear and
        # layer2 are now zero with no exception -- previously linear was zero
        # except two hand-set warm-start columns. layer1 stays Xavier-random;
        # see "Why layer1 gets real random weights, not zero" for why zeroing
        # it too would freeze layer2's columns at zero forever.
        _zero_(self.linear.weight)
        _zero_(self.linear.bias)
        _xavier_uniform_(self.layer1.weight, rng)
        _zero_(self.layer1.bias)
        _zero_(self.layer2.weight)
        _zero_(self.layer2.bias)

    def _candidate_inputs(
        self,
        vehicle_embedding: torch.Tensor,
        client_embeddings: torch.Tensor,
        claimed: torch.Tensor,
        is_depot: torch.Tensor,
    ) -> torch.Tensor:
        """``x``: ``[n_candidates, input_dim]`` -- the concatenation every
        candidate's score (:meth:`forward`) and feature vector (:meth:`features`)
        are both built from. Factored out once so the two paths cannot drift
        apart on how a candidate's raw inputs are assembled.
        """
        n_candidates = client_embeddings.shape[0]
        vehicle = vehicle_embedding.unsqueeze(0).expand(n_candidates, -1)
        return torch.cat(
            [vehicle, client_embeddings, claimed.unsqueeze(-1), is_depot.unsqueeze(-1)], dim=-1
        )

    def forward(
        self,
        vehicle_embedding: torch.Tensor,
        client_embeddings: torch.Tensor,
        claimed: torch.Tensor,
        is_depot: torch.Tensor,
    ) -> torch.Tensor:
        """``vehicle_embedding``: ``[d_model]``. ``client_embeddings``:
        ``[n_candidates, 2*d_model]`` (the caller's ``Embeddings.clients[:, v, :]``
        slice for this vehicle, plus whatever synthetic candidate rows it
        appends). ``claimed``: ``[n_candidates]`` (nonzero where another vehicle
        already claimed that Client this decision). ``is_depot``:
        ``[n_candidates]``, 1.0 on the synthetic depot row and 0.0 on every
        Client. Returns ``W · φ(s, v, a)`` — the trainable residual, ``0.0``
        for every candidate at init — over ``[n_candidates]``; the caller adds
        the myopic base ``c(s, v, a)`` to get ``Q`` (module docstring, "The
        myopic base").
        """
        x = self._candidate_inputs(vehicle_embedding, client_embeddings, claimed, is_depot)
        hidden = torch.relu(self.layer1(x))
        # ``linear``'s bias is the one weight added identically to every
        # candidate of every sweep, so scaling its contribution moves Q's
        # overall level without touching a single argmin -- module docstring,
        # "The level term, and why it needs a gain". Written as an increment so
        # the default gain of 1.0 adds exactly 0.0 and is bit-identical to this
        # term not being here at all. Ticket 16: with W solved directly (never
        # walked to by an optimizer), this term is inert regardless of
        # ``level_gain`` -- there is no longer a level to accelerate -- and is
        # kept wired only at its bit-identical-to-absent default (see
        # ``ridge_estimator.py``/``transformer_policy.py``, "the estimator").
        level = (self.level_gain - 1.0) * self.linear.bias
        output: torch.Tensor = (self.linear(x) + level + self.layer2(hidden)).squeeze(-1)
        return output

    @property
    def feature_dim(self) -> int:
        """Width of :meth:`features`' output: ``input_dim + hidden_dim + 1``
        (ticket 16: ``x``, ``ReLU(layer1(x))``, and one constant column for the
        combined intercept -- 386 + 128 + 1 = 515 at the shipped architecture).
        """
        return self.linear.in_features + self.layer1.out_features + 1

    def features(
        self,
        vehicle_embedding: torch.Tensor,
        client_embeddings: torch.Tensor,
        claimed: torch.Tensor,
        is_depot: torch.Tensor,
    ) -> torch.Tensor:
        """``phi(s, v, a)`` per candidate: ``[x ; ReLU(layer1(x)) ; 1]``, shape
        ``[n_candidates, feature_dim]`` (ticket 16, "It is exactly linear in the
        solvable parameters"). Same arguments as :meth:`forward`, and the two
        share :meth:`_candidate_inputs`/``layer1`` exactly, so ``forward(...)  ==
        features(...) @ w_vector()`` up to the (inert at its default) level term
        -- :class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator` regresses
        onto this, never onto ``forward``'s own autograd graph, since the
        estimator is solved in closed form, not by backpropagation.
        """
        x = self._candidate_inputs(vehicle_embedding, client_embeddings, claimed, is_depot)
        hidden = torch.relu(self.layer1(x))
        ones = x.new_ones((x.shape[0], 1))
        return torch.cat([x, hidden, ones], dim=-1)

    def w_vector(self) -> torch.Tensor:
        """The combined ``[linear.weight ; layer2.weight ; linear.bias +
        layer2.bias]`` row, in :meth:`features`' column order -- the only
        quantity :class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator`
        solves for. ``layer2``'s bias and ``linear``'s collapse into one
        intercept because they were never separately identifiable (ticket 16).
        Detached: this is a plain readout of current parameter values, not
        part of any autograd graph.
        """
        combined_bias = self.linear.bias.detach() + self.layer2.bias.detach()
        return torch.cat(
            [
                self.linear.weight.detach().reshape(-1),
                self.layer2.weight.detach().reshape(-1),
                combined_bias,
            ]
        )

    def load_w_vector(self, w: torch.Tensor) -> None:
        """Assign a solved combined weight vector back onto ``linear``/``layer2``
        in place (ticket 16: the ridge solve replaces gradient descent as the
        only way these four parameters ever move). The intercept is written
        entirely onto ``linear.bias``, with ``layer2.bias`` zeroed -- their
        *sum* is what :meth:`forward`/:meth:`w_vector` read, so the split
        between the two is arbitrary and this choice is as good as any other.
        """
        if w.shape != (self.feature_dim,):
            raise ValueError(f"w must have shape ({self.feature_dim},), got {tuple(w.shape)}")
        input_dim = self.linear.in_features
        hidden_dim = self.layer1.out_features
        with torch.no_grad():
            self.linear.weight.copy_(w[:input_dim].reshape(1, input_dim))
            self.layer2.weight.copy_(w[input_dim : input_dim + hidden_dim].reshape(1, hidden_dim))
            self.linear.bias.copy_(w[input_dim + hidden_dim :])
            self.layer2.bias.zero_()
