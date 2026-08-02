"""network: the transformer encoder over ticket-04 tokens, the Q head that scores
(vehicle, Client) pairs, and the myopic warm start (ticket 05, neural-policy).

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
keeping it un-mixed by cross-token attention is exactly what makes the warm
start below exact rather than approximate).

The arc pathway is where the cost features belong structurally: it is the
action-conditional path, and it is **linear**. ``QHead``'s own linear path
composed with ``arc_embed`` spans every linear function of the six raw arc
facts, so the advantage ``A(s, a)`` — the quantity the argmin reads, and the
one Gate A measured the raw-facts network failing to learn (``Q`` collapsing
to ``V(s)``) — is expressible as a near-linear readout of the projected costs
from the first gradient step, instead of something to be rediscovered from
noisy Monte Carlo returns.

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

## The warm start (this module's load-bearing part)

F2 (``docs/research/rl-methodology-for-stdvrp.md``) recommends a myopic warm
start: initialise so ``Q(i, j) ≈ minutes_from_vehicle_i_to_j`` rather than
leaving a ~150-Client argmin to an untrained network with no prior (the
farthest Client would be as likely a greedy choice as the nearest). This is an
**initialization, not a feature**: ``minutes_from_vehicle`` is not passed to
``QHead`` as an extra argument (its signature stays exactly
``(vehicle_embedding, client_embeddings, claimed)``); instead, specific weights
of the *ordinary, otherwise-trainable* layers below are hand-set so that the
network's output happens to equal that value at initialisation. Every one of
those weights remains a normal ``nn.Parameter`` — nothing here is a permanent,
non-trainable skip connection, and gradient descent is free to move every one
of them once training starts.

The construction, precisely:

1. **Every ``nn.TransformerEncoderLayer`` is an exact identity map at
   init.** With ``norm_first=True``, a layer computes
   ``x = x + attn(norm1(x))`` then ``x = x + ffn(norm2(x))``. Zero-initialising
   ``self_attn.out_proj`` and the FFN's ``linear2`` (weight **and** bias) makes
   ``attn(...)`` and ``ffn(...)`` evaluate to an exact-zero tensor *regardless*
   of what ``norm1``/``norm2``/``in_proj``/``linear1`` computed upstream (a
   ``Linear`` with zero weight and zero bias is a constant-zero function of any
   finite input) — so each residual sum degenerates to ``x + 0 = x``, bit for
   bit. Verified empirically before writing this module (a
   ``TransformerEncoder(num_layers=3)`` built this way returns its input
   unchanged, exactly). ``dropout=0.0`` throughout — see "Determinism" below.
   Consequently ``client_context`` (a client token's post-encoder embedding)
   equals its **pre-encoder linear embedding** at init: a plain, transformer-
   untouched function of the client's three base facts.
2. **The arc embedding's dimension 0 is a hand-set readout of the arc token.**
   ``arc_embed: nn.Linear(ARC_TOKEN_WIDTH, d_model)``'s row 0 is set to
   :data:`WARM_START_WEIGHTS`\\ ``[warm_start]``, bias ``0.0``, so
   ``arc_embed(arc)[0]`` is a chosen linear combination of the six arc facts
   (always ``>= 0``: every field is a duration or a cost). Every other output
   dimension of ``arc_embed`` (and every other weight in this module) is
   Xavier-uniform from ``init_rng``, giving the network real capacity to learn
   beyond the warm start.

   ``"minutes"`` — the default, and the initialization Gate A's frozen null
   model is written against — puts ``1.0`` on ``minutes`` and ``0.0``
   everywhere else, so the cost fields contribute **exactly nothing at init**
   while their gradient path is live from the very first backward pass
   (``linear``'s warm-start weight times ``arc_embed`` row 0's cost columns).
   ``"cost"`` additionally prices the leg by the three single-Client
   components of the simulator's own cost function. **Ticket 08 measured that
   difference at -28% of episode cost, against -20% for the same architecture
   after 650 training episodes** (eight ``evaluation_seeds``, real Chengdu
   data: 4754 for ``"minutes"``, 3421 for ``"cost"``, 3794 for the trained
   ``"minutes"`` network's best block). The cost components were already in
   the token and already reachable by a single weight; leaving that weight at
   zero was asking gradient descent to rediscover, from noisy Monte Carlo
   returns, an arithmetic identity the tokenizer had already computed.
3. **``QHead`` is a linear path plus an MLP branch, and the warm start lives
   on the linear path.** ``QHead``'s input row is ``[vehicle | client_context
   | arc | claimed | is_depot]`` (the layout ``TokenEncoder`` builds
   ``Embeddings.clients`` in, plus the two per-candidate scalars), and

       A = linear(x) + layer2(ReLU(layer1(x)))

   ``linear: nn.Linear(3*d_model + 2, 1)`` is zeroed except for a ``1.0`` at
   the column reading ``arc_embed``'s dimension 0 (global index ``2*d_model``)
   and :data:`DEPOT_WARM_START_PENALTY` at the column reading ``is_depot``
   (see "The depot is the last resort at init" below). Bias 0.
4. **The MLP branch contributes exactly zero at init.** ``layer2:
   nn.Linear(hidden, 1)`` is zeroed weight and bias, so whatever
   ``ReLU(layer1(x))`` computes is multiplied by zero. Therefore ``A ==
   linear(x) == minutes_from_vehicle_i_to_j`` exactly at construction, for
   every vehicle, every client, every state — not a statistical approximation.
   ``claimed`` is init-inert (by design — the warm start must not depend on
   it) but not a dead argument: ``layer1``'s rows are ordinary Xavier-random
   and do read it, so it starts affecting ``Q`` as soon as training moves
   ``layer2``'s columns off zero (see the deadlock note below for why
   ``layer1`` must *not* also be zeroed).

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

## Why the warm start is on a linear path

Because the previous version put it behind the ReLU, and **it died in one
training episode.** ``hidden[0] = ReLU(w0 · x)`` with ``w0`` zeroed except for
the two warm-start weights meant the unit's pre-activation was
``minutes_from_vehicle / horizon_length`` — measured on the mini fixture, a
number in ``[0.002, 0.061]``. The other 384 columns of ``w0`` start at exactly
zero but are ordinary parameters, and the ``[vehicle | client_context]``
prefix they multiply has L2 norm ~2.9, so Adam's normalised steps (which move
every weight by ~``lr`` per step regardless of how small its gradient is) put
a perturbation on that pre-activation far larger than the signal it carries.
Measured, one episode of training moved it from ``[+0.003, +1.000]`` to
``[-0.546, -0.077]``: **dead for every candidate**. A dead ReLU has exactly
zero gradient, so ``w0`` could never come back — the myopic prior, the depot
penalty and the ranking they encode were gone permanently after episode 1, and
the spread of ``Q`` across a vehicle's candidates collapsed from 0.036 to
0.004 with the argmin no longer picking the nearest Client.

On a linear path there is no kink to fall off. The same weights are just as
trainable — gradient descent can flatten, invert or replace the myopic prior
whenever the returns say so — but it degrades continuously and recovers,
instead of being annihilated by the first optimizer step that overshoots. The
MLP branch keeps the head's nonlinear capacity; it simply is not where the
initialization is stored.

## The depot is the last resort at init (the ``is_depot`` input)

Ticket 08's Gate A run measured the untrained network at mean cost **81 701**
against the linear baseline's **2 483** — not "a respectable nearest-neighbour
rival" but a policy that serves nobody. The cause was in this warm start.
``TransformerMonteCarloPolicy`` scores the depot as one more candidate whose
"arc" half is ``[minutes_to_depot, length_to_depot]``, so points 2-4 above gave
it ``Q(v, depot) == minutes_to_depot / horizon_length`` — which is **exactly
0.0 for a vehicle standing on the depot**, the smallest value the ReLU'd row 0
can produce. Every vehicle starts the Episode parked at the depot, so at
decision epoch 1 every vehicle's argmin was the depot; ``Model`` saw
``fleet.all_parked()`` and terminated the Episode after a single transition
with every Client unserved. Worse, ``Model._reroute_for`` only reroutes a
*travelling* vehicle, so parking is irreversible: even away from the depot, a
vehicle that happened to be nearer home than to any Client retired for good.

``QHead`` therefore takes a second per-candidate scalar next to ``claimed``:
``is_depot``, 1.0 on the synthetic depot row and 0.0 on every Client. Row 0
reads it with weight :data:`DEPOT_WARM_START_PENALTY` ``= 1.0`` — **one whole
horizon** in the units ``arc_embed``'s dimension 0 already speaks
(``minutes / horizon_length``), so at init

    Q(v, client) == minutes_to_client / horizon_length      (<= 1 for any
                                                             reachable Client)
    Q(v, depot)  == minutes_to_depot / horizon_length + 1

and the depot loses the argmin to *any* feasible Client, while still winning
when the mask leaves it as the only candidate. That is precisely the null model
spec.md specifies ("the untrained network already goes to the nearest feasible
Client"), and it is measurably so: on the mini fixture the untrained network
now reproduces an independently written nearest-neighbour policy's episode cost
to the last decimal (477.5 / 327.8 on seeds 100/101, against 8 350 / 10 293
before).

This is an **initialization, not a rule** — the same discipline as the rest of
this warm start. ``is_depot`` is an ordinary input column and
``DEPOT_WARM_START_PENALTY`` an ordinary weight in an ordinary trainable row, so
a trained network is free to price going home however the returns say it should,
including below every Client near the shift end. Nothing here masks the depot
out of the action set: ADR-0007's "feasible = every pending Client not already
claimed, plus the depot" is untouched, and the depot keeps both its meanings.

**Why ``layer1`` gets real random weights, not zero:** an earlier draft zeroed
the *entire* first layer, reasoning that units contributing nothing at init can
stay dormant. That is a genuine dead end, not a transient one: with
``layer1.weight[row, :] == 0`` identically, ``hidden[row]`` is exactly ``0``
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
pieces in this module (``out_proj``/``linear2`` for identity-at-init, and
``QHead.layer2``) are all of the "safe" kind — a real, nonzero,
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

#: How much the warm start adds to the depot candidate's ``Q``, in the units
#: ``arc_embed``'s dimension 0 speaks (``minutes / horizon_length``): one whole
#: horizon. Any Client reachable within the shift scores below that, so the
#: untrained greedy policy is "nearest feasible Client, home only when no Client
#: is feasible" -- see this module's docstring, "The depot is the last resort at
#: init". An ordinary weight in an ordinary trainable row, not a floor.
DEPOT_WARM_START_PENALTY = 1.0

#: Per-candidate scalars ``QHead`` reads alongside the embeddings, in order:
#: ``claimed`` then ``is_depot``.
CANDIDATE_FLAG_WIDTH = 2


# Type-embedding indices (Embeddings' three token kinds).
_TYPE_CLIENT = 0
_TYPE_VEHICLE = 1
_TYPE_GLOBAL = 2
_N_TOKEN_TYPES = 3


def _arc_dim0_index(d_model: int) -> int:
    """Global index of ``arc_embed``'s reconstruction dimension within a QHead
    input row (``[vehicle | client_context | arc | claimed | is_depot]``).

    The single source of truth for the layout ``TokenEncoder.forward`` builds
    (``Embeddings.clients`` = ``concat([client_context, arc], dim=-1)``, each
    ``d_model`` wide) and ``QHead`` depends on for the warm start — both sides
    call this instead of re-deriving the offset independently, and
    ``TestWarmStart`` (``tests/unit/test_network.py``) is the end-to-end check
    that would catch the two going out of sync (e.g. a future reordering of
    ``TokenEncoder``'s ``torch.cat``).
    """
    return d_model + d_model


def _is_depot_index(d_model: int) -> int:
    """Global index of the ``is_depot`` flag within a QHead input row.

    Last column of ``[vehicle | client_context | arc | claimed | is_depot]``:
    ``d_model`` + ``2 * d_model`` + the ``claimed`` scalar.
    """
    return d_model + 2 * d_model + 1


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


class TokenEncoder(nn.Module):
    """Runs once per decision epoch: tokens -> :class:`Embeddings`.

    See the module docstring for the full shape rationale and the warm-start
    construction this class is half of (the other half is :class:`QHead`).
    """

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
        warm_start: str = "minutes",
    ) -> None:
        super().__init__()
        if warm_start not in WARM_START_WEIGHTS:
            raise ValueError(
                f"unknown warm_start {warm_start!r}: expected one of "
                f"{sorted(WARM_START_WEIGHTS)}"
            )
        self.d_model = d_model
        self.warm_start = warm_start
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

        self._init_weights(init_rng)
        self.to(self.device)

    def _init_weights(self, rng: np.random.Generator) -> None:
        for embed in (self.client_base_embed, self.vehicle_embed, self.global_embed):
            _xavier_uniform_(embed.weight, rng)
            _zero_(embed.bias)

        # arc_embed: row 0 IS the warm start (its other load-bearing half lives
        # in QHead._init_weights below) -- one weight per arc-token field, from
        # WARM_START_WEIGHTS. Under "minutes" it reconstructs
        # minutes_from_vehicle exactly and the cost fields are init-inert;
        # under "cost" the three single-Client cost components join it in the
        # same 1/horizon_length currency. Every other output dimension is
        # ordinary Xavier-random capacity.
        _xavier_uniform_(self.arc_embed.weight, rng)
        _zero_(self.arc_embed.bias)
        with torch.no_grad():
            weights = WARM_START_WEIGHTS[self.warm_start]
            self.arc_embed.weight[0, :] = torch.tensor(weights, dtype=torch.float32)
            self.arc_embed.bias[0] = 0.0

        _xavier_uniform_(self.type_embedding, rng)

        for encoder_layer in self.transformer.layers:
            attn = encoder_layer.self_attn
            _xavier_uniform_(attn.in_proj_weight, rng)
            _zero_(attn.in_proj_bias)
            # Identity-at-init (see module docstring): a Linear with zero weight
            # AND zero bias is a constant-zero function of any finite input, so
            # zeroing out_proj/linear2 makes each residual branch contribute
            # exactly zero regardless of what in_proj/linear1 computed.
            _zero_(attn.out_proj.weight)
            _zero_(attn.out_proj.bias)
            _xavier_uniform_(encoder_layer.linear1.weight, rng)
            _zero_(encoder_layer.linear1.bias)
            _zero_(encoder_layer.linear2.weight)
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

        return Embeddings(clients=clients, vehicles=vehicle_context, depot=depot)


class QHead(nn.Module):
    """Scores one vehicle against every pending Client: a **cost to minimize**.

    Matches the baseline's ``argmin`` convention (spec.md) — the caller (ticket
    06) takes ``argmin`` over feasible clients, exactly as ``MonteCarloPolicy``
    does. Never flip this sign.
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
        # See the module docstring, "The level term, and why it needs a gain".
        self.level_gain = level_gain
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        client_dim = 2 * d_model  # Embeddings.clients' per-vehicle slice width
        input_dim = d_model + client_dim + CANDIDATE_FLAG_WIDTH  # + claimed, is_depot
        # The warm start lives on ``linear``, NOT behind ``layer1``'s ReLU --
        # see the module docstring, "Why the warm start is on a linear path".
        self.linear = nn.Linear(input_dim, 1)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self._arc_dim0_index = _arc_dim0_index(d_model)
        self._is_depot_index = _is_depot_index(d_model)

        self._init_weights(init_rng)
        self.to(self.device)

    def _init_weights(self, rng: np.random.Generator) -> None:
        # The linear path: zero everywhere except the two warm-start weights,
        # so at init Q is exactly minutes_from_vehicle / horizon_length for a
        # Client and one whole horizon more for the depot candidate. Ordinary
        # trainable parameters -- gradient reaches every one of them from the
        # first backward pass, because their input x is nonzero.
        _zero_(self.linear.weight)
        _zero_(self.linear.bias)
        with torch.no_grad():
            self.linear.weight[0, self._arc_dim0_index] = 1.0
            self.linear.weight[0, self._is_depot_index] = DEPOT_WARM_START_PENALTY

        # The MLP branch: Xavier-random first layer, zeroed output projection,
        # so it contributes exactly zero at init and the warm start above is
        # the whole of Q. layer1's rows must NOT also be zeroed -- see the
        # module docstring's deadlock note; with them random, layer2's columns
        # each see a nonzero hidden[r] and move off zero on the first step,
        # which unlocks layer1 from the step after.
        _xavier_uniform_(self.layer1.weight, rng)
        _zero_(self.layer1.bias)
        _zero_(self.layer2.weight)
        _zero_(self.layer2.bias)

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
        Client — the flag the warm start prices going home with (see this
        module's docstring). Returns ``[n_candidates]``.
        """
        n_candidates = client_embeddings.shape[0]
        vehicle = vehicle_embedding.unsqueeze(0).expand(n_candidates, -1)
        x = torch.cat(
            [vehicle, client_embeddings, claimed.unsqueeze(-1), is_depot.unsqueeze(-1)], dim=-1
        )
        hidden = torch.relu(self.layer1(x))
        # ``linear``'s bias is the one weight added identically to every
        # candidate of every sweep, so scaling its contribution moves Q's
        # overall level without touching a single argmin -- module docstring,
        # "The level term, and why it needs a gain". Written as an increment so
        # the default gain of 1.0 adds exactly 0.0 and is bit-identical to this
        # term not being here at all.
        level = (self.level_gain - 1.0) * self.linear.bias
        output: torch.Tensor = (self.linear(x) + level + self.layer2(hidden)).squeeze(-1)
        return output
