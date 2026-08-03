"""TransformerMonteCarloPolicy: decide, decide_train, learn (ticket 06, neural-policy).

The ``TrainablePolicy`` (ticket 02) that scores every pending Client with the
ticket-05 ``TokenEncoder``/``QHead`` and argmins over the linear baseline's own
``m + 2`` candidates (ticket 14, ADR-0011) rather than the whole scored row, and
learns from the same Monte Carlo return ``MonteCarloPolicy.learn`` targets.
Since ticket 16, ``learn`` fits that target by accumulating normal equations
across Episodes into a :class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator`
and re-solving on a cadence, rather than by an Adam step per Episode --
see "The estimator" below. Imports torch at module scope, like
``network.py``: this file's whole reason to exist is the network, so it must
never be imported from ``stdvrp.policies.__init__`` or any module reachable at
package-import time. Callers import it explicitly:
``from stdvrp.policies.transformer_policy import TransformerMonteCarloPolicy``.

## ADR-0011 — the action set is the baseline's own (reverses ADR-0007)

ADR-0007 gave this Policy a bespoke feasibility-only action set — "every
pending Client not already claimed, plus the depot" — on the argument that
``_select_vehicle_possible_actions`` is hand-engineered ranking built for the
linear baseline's small candidate pool, with nothing in it a network needs.
Ticket 14 measured that argument wrong: the candidate count is worth
**12.68%** to the linear baseline itself (``m+40`` 2168.39 vs ``m+2`` 2483.24
over 50 ``evaluation_seeds``, 36/50, Wilcoxon p = 8.24e-05 —
``results/baseline_null_50.py``); a ``k``-nearest shortlist is a regularizer
against long myopic hauls (``c(s, a)`` can send a vehicle far across the map to
a Client about to breach its window — myopically correct, globally ruinous),
not a crutch for a model that cannot see far. The second reason is fairness
rather than performance: spec.md decision 1's amendment already handed the
four projected costs to the network on the ground that the baseline's own
state-action features carry them, which levels the two Policies' *inputs*;
the action set was the last un-levelled axis, and a Gate B result could not be
attributed to the approximator while it stayed that way.

``_sweep`` now calls :func:`stdvrp.policies.action_set.select_vehicle_possible_actions`
— ticket 13's extraction of the linear baseline's own definition, unmodified —
at ``m + 2``, in both :meth:`decide` and :meth:`decide_train`.
``episode.py`` trains the baseline at ``m+2`` but evaluates it at ``m+40`` over
the swept action counts; this Policy is fixed at ``m+2`` throughout and does
not inherit that mismatch. ``self.action`` is threaded through as
``current_action``, mutated in place vehicle by vehicle within one ``_sweep``
call — mirroring ``MonteCarloPolicy.action`` (``Model.run_training_episode``'s
own comment already documents this aliasing generically as "``policy.action``").

Two things retire with this change, and one closes as a side effect rather
than as work of its own:

- **``_is_retired`` retires.** Branch 1 of ``select_vehicle_possible_actions``
  offers a vehicle parked at the depot past ``tau > 350`` only the depot, so it
  can no longer claim a Client it cannot serve — the same protection
  ``_is_retired`` gave, now expressed as one arm of the shared candidate
  computation instead of a Policy-private clock check. One behavior
  difference is deliberate, not a bug: ``_is_retired`` gated on
  ``horizon_start_minute`` (a config clock, chosen to avoid colliding with
  every vehicle's un-dispatched depot start at epoch 1); the shared branch
  gates on the literal ``350`` instead. Adopting the *identical* set means
  adopting that literal too, which is why this is a reversal of ADR-0007
  rather than an amendment to it.
- **``_depot_is_feasible`` retires.** Its condition 2 (the return leg already
  breaches the shift: ``tau + average_minutes(position, depot) >
  shift_end_minute``) *is* branch 3's own depot-append condition, literally
  the same formula; its condition 1 (no Client feasible) is subsumed by every
  branch's fallback to ``[depot]`` when nothing else survives.
- **The ``claimed_mask`` defect closes.** Ticket 08 left it open: the old
  ``claimed_mask`` was rebuilt fresh every ``_sweep`` call, so it only ever
  knew about vehicles already decided *this* pass — never a not-yet-processed
  vehicle's in-flight target from the previous decision epoch, the way
  ``MonteCarloPolicy`` seeds ``forbidden_actions`` from ``self.action`` for
  every other vehicle. Threading ``self.action`` through as ``current_action``
  adopts that behavior for free.

Two things ``action_set.py`` does not know about, because they are not part of
the linear baseline's own rule, stay Policy-side, layered on top of whatever
it returns:

1. **No double booking within this pass** (the B11 invariant) — the same
   guarantee as before ADR-0011, now arising from ``select_vehicle_possible_actions``
   excluding ``current_action``'s other entries rather than from a locally
   rebuilt mask.
2. **No self-node** (``simulator-correctness`` ticket 11, B20, ADR-0008): a
   pending Client the vehicle is already standing on is not a candidate —
   there is nothing to travel to. ADR-0008 leaves ``monte_carlo.py`` untouched
   because "its own candidate rules already exclude a vehicle's current node
   by construction"; that argument is about how the linear baseline happens to
   call the shared function, not a property of the function itself, so
   :meth:`TransformerMonteCarloPolicy._sweep` still filters it explicitly —
   in both the greedy and the ε-exploration branch — falling back to the
   depot if nothing survives the filter. The depot is never filtered by
   either rule.

``claimed`` is additionally fed to the network as an input (spec.md decision
6: "claimed enters at the head"), computed the same way it always has been —
a mask over every *pending* Client, not only the ``m + 2`` candidates — so a
trained network's *predictions* can still account for contention beyond the
shortlist. One consequence of the shortlist worth naming rather than
rediscovering by surprise: since ``select_vehicle_possible_actions`` already
excludes every other vehicle's current target from the candidates it returns,
``claimed`` is now **structurally ``False`` for every candidate the argmin
actually considers** — informative for the rows the argmin ignores, constant
for the rows it does not. The legality of an action never depended on what
the network output for it before ADR-0011 either; that discipline is
unchanged, only which candidates reach the argmin at all is different.

**``is_depot``/``DEPOT_WARM_START_PENALTY``: kept, decided by measurement.**
With the depot now entering the candidate list only where
``select_vehicle_possible_actions`` itself admits it (forced retirement, or a
shift-breach append competing with real Clients), the question this ticket
opened was whether the penalty still earns its keep. Measured directly
(``.scratch/neural-policy/results/action_set_m2_50.py``): the untrained
``cost`` warm start over the 50 ``evaluation_seeds`` reads 3365.09 at
``DEPOT_WARM_START_PENALTY = 1.0`` (as shipped) against 3364.52 at ``0.0`` —
**-0.02%, 1/50 seeds differ, Wilcoxon p = 0.317**. Not a close call the
penalty is winning; a null result. The structural prediction ("the depot
rarely competes any more") holds, and the penalty is left in place rather
than retired on it: it costs nothing where it no longer matters and is still
correct where it does (the shift-breach window, where the depot genuinely
does compete against real Clients in the argmin) — a null measurement is a
reason not to touch working code, not a reason to remove it.

## The depot's Q value

``QHead`` scores ``Embeddings.clients`` — one row per **pending Client** — so
the depot, which is never a Client, has no natural client row to score. The
tokenizer emits its arc facts as ``Tokens.depot_arc_tokens`` (the same six
fields every real candidate's arc vector carries, projected costs included —
see ``tokenizer.py``, "The cost fields") and the encoder builds the synthetic
candidate row as ``Embeddings.depot``:

    depot_row = concat([vehicle_context, arc_embed(depot_arc_tokens[v])])

and, since ticket 15, the synthetic candidate's myopic base alongside it:

    Embeddings.depot_cost[v] = c(s, v, depot)   (network.py, "The depot's place in `c`")

``_score`` appends both ``embeddings.depot[vehicle]`` (to the client
embedding rows, before calling ``QHead``) and ``embeddings.depot_cost[vehicle]``
(to the client myopic-base row, before adding it to ``QHead``'s output) — one
uniform pathway for each, rather than this module hand-building either from a
separate ``EpisodeGeometry`` read as it did before the decision-1 amendment.
The "context" half of a real Client's embedding row is that Client's
transformer-refined embedding; the depot has no such thing, so its row uses
the vehicle's own context embedding instead — a deliberate choice, not an
arbitrary filler: the depot's meaning is "return to base", which is a fact
about the *vehicle* (its remaining capacity, how deep into the shift it is),
not about the destination.

The embedding row additionally carries an ``is_depot`` flag into ``QHead``
beside ``claimed``: it is the only thing that distinguishes this synthetic
candidate's *embedding* row from a real Client's (the myopic-base gap between
them is carried separately, in ``c``, not through this flag — see
``network.py``, "The depot's place in `c`"). At construction ``Q(v, depot) ==
c(s, v, depot) == minutes_to_depot / horizon_length + 1`` (``QHead``'s own
output is exactly zero for every candidate at init — "The myopic base"),
while every Client scores ``Q(v, client) == c(s, v, client) ==
minutes_to_client / horizon_length <= 1``, so the untrained greedy policy is
**"go to the nearest feasible Client, home only when no Client is
feasible"** — the null model spec.md specifies.

An earlier version of this file left the flag out entirely, so the depot got
the same myopic estimate as a real candidate: ``Q(v, depot) ==
minutes_to_depot / horizon_length``, which is exactly ``0`` for a vehicle
standing on the depot. Every vehicle starts parked there, so every vehicle's
argmin was the depot at decision epoch 1, ``Model`` saw
``fleet.all_parked()``, and the Episode terminated after one transition with
every Client unserved (ticket 08 measured the resulting "null model" at mean
cost 81 701 against the linear baseline's 2 483). ``Model._reroute_for``
reroutes only *travelling* vehicles, so the same construction also retired
vehicles permanently whenever home happened to be nearer than any Client.

``c``'s one-horizon depot margin is, since ticket 15, structurally
permanent — no weight anywhere on its path for an optimizer to move (module
docstring, "The myopic base"). What training *can* still move is ``QHead``'s
residual: once ``layer2``'s columns move off zero, ``is_depot`` (read by
``layer1``'s Xavier-random rows from the first forward pass, exactly like
``claimed``) can drive the residual to favour or disfavour the depot row
independently of every Client's, which is enough on its own to put the depot
back under a Client's total ``Q`` regardless of ``c``'s fixed gap — a trained
network is free to price going home however the returns say it should,
including below every Client near the shift end (``network.py``, "The
depot's place in `c`"). Since ADR-0011, it is
``select_vehicle_possible_actions``'s own branches — not a Policy-private
``_depot_is_feasible`` — that keep the depot from being offered as a
candidate at all outside the window where heading home is legal; the myopic
base is what keeps the *untrained* policy honest inside that window, where
the depot competes with real Clients in the argmin like any other candidate.
See "ADR-0011" above for whether the penalty is still earning its keep now
that the depot enters far less often — decided by measurement, not argument.

## The estimator (ticket 16): accumulated ridge, not per-Episode Adam

``learn`` no longer runs any gradient step. ``QHead.linear``/``layer2`` (the
only parameters with a nonzero ``W`` at any point after init -- ``network.py``,
"The myopic base") are now fit exactly, in closed form, by
:class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator`, which this class
owns as ``self.ridge`` (injected, mutated in place, and carried by the caller
across Episodes exactly the way ``encoder``/``head`` already are -- see
``neural_episode.py``'s ``NeuralPolicyState``). ``encoder``/``head.layer1``
are not touched by this ticket at all: nothing in ``learn`` computes a
gradient any more, so they simply stay at whatever random weights ``__init__``
drew (the "frozen encoder" arm; ticket 17's "trained encoder" arm, which
resumes training them by SGD on the same residual, is a later ticket's work --
see spec.md's "Two timescales").

For every decision epoch of a (non-aborted) Episode, ``learn`` builds:

::

    Phi_t = sum_v phi(s, v, a_v)                    QHead.features(...), summed
    y_t   = targets[t] / _return_scale - sum_v c(s, v, a_v)

(:meth:`_replay_joint_features`, one encoder pass per epoch, mirroring
:meth:`_replay_joint_q`'s own replay) and folds the whole Episode's
``(Phi_t, y_t)`` rows into ``self.ridge`` in one
:meth:`~stdvrp.policies.ridge_estimator.RidgeAccumulator.observe_episode`
call. Every ``neural_solve_cadence`` Episodes (``self.solve_cadence``,
tracked by the accumulator's own ``episodes_since_solve``), the normal
equations are re-solved and the result is written straight onto
``QHead.linear``/``layer2`` via :meth:`~stdvrp.policies.network.QHead.load_w_vector`
-- the only way those four parameters ever move now. Before the first solve
(and whenever the accumulator has seen no usable data yet) the solve returns
the zero vector, which is exactly ``QHead``'s own zero-init -- ticket 15's
null is this ticket's start state, not merely its limit as training data
shrinks to nothing.

**Aborted Episodes are excluded, not merely down-weighted** (:meth:`_is_aborted`,
research note F10, ``ridge_estimator.py``'s own "Aborted Episodes are
excluded" section): ``learn``'s frozen ``TrainablePolicy`` signature — shared
structurally with ``MonteCarloPolicy.learn``, which this ticket does not touch
— carries no explicit "this Episode aborted" flag, so the detection reads it
off the one place ``ABORT_PENALTY - 200 * served`` is guaranteed to show up:
the final transition's reward. ``served <= self.number_clients`` always, so
``ABORT_PENALTY - ABORT_PENALTY_PER_SERVED_CLIENT * self.number_clients`` (a
constant computed once, at construction) is a guaranteed floor under any
abort's actual penalty for this Episode's demand, and far above any single
non-aborted transition's plausible cost. An excluded Episode still ages the
accumulator's memory (``self.ridge``'s own forgetting is applied on every
call, aborted or not) -- it contributes nothing *new*, but it does not freeze
the window in place either.

``self.last_loss`` (read by the Trainer's live per-episode report, not part of
the ``TrainablePolicy`` protocol) is repurposed from "the mean training loss
over this episode's minibatch passes" to "the mean squared residual this
Episode's samples show against the *entering* ``W``" — ``mean((y_t - Phi_t .
w_vector)**2)``, computed before this call's potential re-solve. It is not
touched on an aborted or empty Episode (matching the pre-ticket-16 behaviour
of leaving it at its last value when ``learn`` has nothing to fit).

## Why ``_already_acquired_cost`` is duplicated, not shared

``learn``'s target is exactly ``update_W``'s: backward Monte Carlo return
minus the same sunk-cost baseline (delay of already-late pending Clients,
overtime of vehicles already past the shift end at ``tau``). Sharing the
formula would mean this file importing from or editing ``monte_carlo.py`` —
the frozen opponent every ticket in this effort predicts a zero self-golden
diff against, and this ticket is no exception. The formula is ten lines of
plain arithmetic over two hardcoded legacy cost factors
(``delay_cost_factor=1``, ``overtime_cost_factor=5/6``); duplicating it here
keeps ``monte_carlo.py`` untouched, following the same precedent as
``scripts/measurement_bench.py``'s independent reimplementations elsewhere in
this codebase.

## Target scaling — ``y`` only, and why the first version scaled both sides

"Standardize ``y`` with fixed, config-derived scales, same discipline as
ticket 04's token normalization" (spec.md decision 9): ``learn`` divides the
Monte Carlo target by ``_return_scale = number_clients * episode_length`` — a
fixed, per-Episode, config-derived order-of-magnitude for the total accumulated
cost (a sum of per-Client delay/earliness/overtime terms, each roughly bounded
by the episode's duration) — before computing the Huber loss, instead of
feeding it the raw cost's three-to-four-digit scale.

That fixes the *range* but not the loss's shape: torch's ``delta=1.0`` is two
orders of magnitude above every residual this regression produces (targets and
predictions both land around ``1e-2``), so every sample falls in the quadratic
branch and ``huber_loss`` is exactly ``0.5 * MSE`` — the robustness the name
promises never engages, and a truncated episode's terminal penalty (research
note F10) is squared into every one of that episode's decision epochs. Ticket
08 made ``delta`` a config knob (``ExperimentConfig.neural_huber_delta``,
defaulting to torch's 1.0 so the knob alone changes nothing); setting it near
the residual scale is what turns those episodes into a bounded gradient.

The **prediction is not divided**, and that asymmetry is the whole point: the
prediction already lives in normalized units, because it is
``c(s, v, a) + QHead(...)`` (ticket 15, ``network.py``, "The myopic base") and
``c`` is built from the tokenizer's own minute-normalised arc facts. At init
``QHead(...) == 0`` exactly, so ``Q == c`` outright — even more directly
normalized than the pre-ticket-15 architecture this paragraph originally
described, where the network's *own* output (not an external term) was pinned
to that scale by the warm start. The two scales meet either way — a Chengdu
episode the linear baseline runs at cost ~2 500 gives ``y ≈ 2500 / (150 * 850)
≈ 0.020``, against an untrained ``Q`` of ``~5 min / 480 min ≈ 0.010``.

Ticket 08's Gate A run divided **both** sides, which is arithmetically the same
as regressing ``Q`` on the *raw* return: the network was asked to move its
output from ~1e-2 to ~1e+5 while the gradient reaching it was divided by
``_return_scale``. It did what least squares does when the intercept is off by
five orders of magnitude — it fitted the mean and threw the ranking away. The
logged loss collapsed to ``0.000`` within a handful of episodes while the eval
cost sat unchanged at the untrained level, which is the signature of a ``Q``
that has gone constant. Scaling ``y`` alone leaves the correction the network
has to make small enough for ``layer2.bias`` to absorb it, and a constant added
to every candidate is invisible to ``argmin`` — so the warm-start ordering
survives the fit of the mean instead of being spent on it.

Neither version changes what ``decide``'s argmin selects at a given set of
weights: scaling every candidate's ``Q`` by the same positive constant preserves
their relative order. What it changes is what *training* does to those weights.

**Ticket 16 keeps exactly this asymmetry, with no loss left to have a shape at
all.** There is no more Huber knee, no more ``delta`` (``neural_huber_delta``
retires -- ``config.py``) and nothing analogous to "the prediction is not
divided" to get wrong, because ``learn`` never computes ``Q`` as a scalar
prediction to compare against a target any more: the regression's own target
is ``y_t = targets[t] / _return_scale - sum_v c(s, v, a_v)`` (module docstring,
"The estimator") -- ``c`` is subtracted from the *already-scaled* return
directly, in the residual itself, rather than added back on the other side of
a loss. The reasoning above for why ``c``/``Q`` must never be divided by
``_return_scale`` is what justifies subtracting an *undivided* ``c`` from a
*divided* target here; it is no longer a loss-shape nicety, it is the formula.

## One sample per decision epoch, over the *joint* action

``learn`` regresses ``Q_joint(s, a) = sum_v Q(s, v, a_v)`` — the whole action
row's summed score — onto that epoch's Monte Carlo return. One sample per
decision epoch, not ``m``. Ticket 16 keeps this structure exactly: the
regression is over ``Phi_t = sum_v phi(s, v, a_v)`` against
``y_t = targets[t]/_return_scale - sum_v c(s, v, a_v)`` (module docstring,
"The estimator") rather than ``Q_joint`` against the raw target, but it is the
same joint-over-vehicles sum, for the same reason -- everything below still
explains why *that* structure is the right one; only the estimator fitting it
changed.

This is what the linear baseline already does, read off its own code rather
than invented here. ``MonteCarloPolicy.learn`` builds **one** ``X`` per epoch
via ``FeatureExtractor.action_features(features, actions[t])``, and that
vector's five action-dependent components — ``total_distance``,
``earliness_cost``, ``delay_cost``, ``future_delay``, ``overtime_cost`` — are
each summed over *every* vehicle's target, not just the one being decided. So
the baseline's ``Q = X · W`` is already ``W·general(s) + sum_v W·f(s, a_v)``:
a state term plus an additive-over-vehicles action term. Its greedy rule is
the matching coordinate-wise argmin — vary one vehicle's target, hold the rest
fixed — which is exactly :meth:`TransformerMonteCarloPolicy._sweep`. Summing
the per-vehicle heads reproduces that structure with a network in place of
``W``, which is the entire brief of this effort (spec.md decision 5: "Only
``X · W`` -> ``net(tokens)``").

The first version instead emitted ``m`` samples per epoch — one per vehicle,
each regressed **individually** onto the same ``U_t``. That is not the same
statistical object, and it is actively hostile to the decision rule. With one
shared target and separate losses, a vehicle whose ``Q`` sits above ``U_t`` is
pushed down while one below it is pushed up: the loss is minimised by making
``Q`` *identical across candidates*, and the ranking the argmin reads is the
first thing gradient descent spends. Measured on the mini fixture, one
training episode was enough to squeeze the spread of ``Q`` over a vehicle's
candidates from 0.036 at the warm start to 0.0007 — the network kept fitting
the state's cost-to-go and stopped distinguishing actions at all, which is the
"loss 0.000, eval cost unchanged" signature ticket 08's log ran for 1 338
episodes. Under the joint sum the gradient reaching every ``Q(s, v, a_v)`` of
an epoch is the *same* scalar, so the row moves together and the differences
between candidates are left to the data instead of being regressed away.

It is also cheaper: one encoder pass now serves a whole epoch's replay (the
same economy the acting path has always had), where the per-vehicle version
re-tokenized and re-encoded the identical snapshot ``m`` times.

**What the joint sum costs, and what has not paid for it (ticket 08).** One
scalar target per epoch means the gradient reaching every candidate term is
the same residual, so the loss cannot see *how* a sum is split between
candidates — and the split is the only thing :meth:`_sweep`'s ``argmin``
reads. Below ~20 parameters that is survivable (the linear baseline cannot fit
``V(s)`` well enough for the residual to vanish, so its action columns keep
receiving signal); at 595k it is not, because the encoder fits ``V(s)`` easily
and the arc's cost weights are then estimated from noise. **This is the
standing explanation for why training adds so little here**, and the obvious
structural remedy for it — a dueling ``V`` plus candidate-centred advantage —
was implemented, measured, and rejected: see ``network.py``, "A dueling
decomposition of ``Q`` — tried, measured, and rejected", including *why* it
made things worse rather than better.

**Ticket 16 pays for it, by changing the unit of estimation rather than the
decomposition.** The diagnosis above is about *one Episode's* ~400 samples:
within a single Episode ``U_t`` is a suffix sum, monotone in ``t``, and the
global token carries the clock directly, so 595k (now 515, ticket 15)
parameters fit it almost perfectly by reading ``t`` — the action carries no
incremental explanatory power *within* one Episode, only *across* them
(spec.md's redesign section, ticket 16's own issue). ``self.ridge``
accumulates across ~50 Episodes before its first solve at the default
cadence, at which point the sample count genuinely exceeds the parameter
count (515 parameters against ~20 000 samples, rather than 595k against
~400) and the joint sum's own signal is no longer drowned by one Episode's
own clock. The dueling remedy above stays rejected on its own terms (module
docstring there covers why); this is a different fix, aimed at the unit of
estimation rather than the split within one Episode's sum.

**A tried-and-rejected variant, so it is not re-tried naively:** ticket 08
measured ~24-33 % of these samples carrying an action the simulator discarded
(a mid-service vehicle — ``Model._reroute_for``'s first branch never reads
it; a retired one matches no branch), and a 2026-08-01 change filtered those
vehicles' ``Q`` terms out of the joint sum. Measured A/B on the mini fixture
(ticket 08's Comments, "the divergence attacked at its root"), the filter
made learning consistently *worse* on top of the cost features: dropping a
term removes one bias (regressing a counterfactual action) but introduces
another — the skipped vehicle's share of ``V(s)`` vanishes from that epoch's
sum while ``U_t`` still contains its future cost, so the per-term burden
varies by epoch and the additive decomposition stops being consistent. The
filter was reverted on that evidence; the full-row sum below is deliberate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.policies.action_set import select_vehicle_possible_actions
from stdvrp.policies.base import Policy
from stdvrp.policies.feature_extraction import FeatureExtractor, TimeWindows
from stdvrp.policies.network import Embeddings, QHead, TokenEncoder
from stdvrp.policies.ridge_estimator import RidgeAccumulator
from stdvrp.policies.tokenizer import tokenize
from stdvrp.simulation.cost_ledger import ABORT_PENALTY, ABORT_PENALTY_PER_SERVED_CLIENT
from stdvrp.simulation.state import is_parked_at_depot

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = ["CalibrationPair", "ResidualCalibrationPair", "TransformerMonteCarloPolicy"]


class CalibrationPair(NamedTuple):
    """One decision epoch/vehicle's calibration sample (ticket 08, Gate A).

    Named rather than a bare ``tuple[float, float]``: three modules
    (this one, ``neural_episode.py``, ``gate_a.py``) all pass this value
    around by position, and "which half is predicted vs. realised" is exactly
    the kind of thing a bare tuple lets a caller silently swap.
    """

    predicted_q: float
    realised_u: float


class ResidualCalibrationPair(NamedTuple):
    """One decision epoch's residual calibration sample (ticket 17, Gate A').

    Gate A' redefines the calibration check onto the residual the network is
    actually regressed onto: ``rho(Q, U_t)`` (:class:`CalibrationPair`, ticket
    08's original pairing) passes at ``W = 0`` with no parameter having moved
    -- ``Q`` still correlates with the return through ``c`` alone (spec.md,
    "Part 3 is redefined"), a guaranteed false PASS under the residual
    decomposition. Pairing ``predicted_residual`` (``W . phi(s, a)``, the
    *learned* term only) against ``residual_target`` (``y_t = targets[t] /
    return_scale - sum_v c(s, v, a_v)`` -- :meth:`TransformerMonteCarloPolicy.learn`'s
    own ridge target) is the check that cannot be faked that way.
    """

    predicted_residual: float
    residual_target: float


# Legacy cost factors MonteCarloPolicy hardcodes (see this module's docstring,
# "Why _already_acquired_cost is duplicated, not shared").
_DELAY_COST_FACTOR = 1.0
_OVERTIME_COST_FACTOR = 5 / 6

# Ticket 14: MonteCarloPolicy's other two hardcoded cost factors, needed only
# to satisfy FeatureExtractor's constructor -- state_features() (the only
# method this Policy calls) never reads them, only action_features() /
# candidate_features() do, and this Policy calls neither. Duplicated for the
# same reason as the two constants above: sharing them would mean importing
# from or editing monte_carlo.py.
_EARLINESS_COST_FACTOR = 0.1
_SERVICE_TIME = 5.0


class TransformerMonteCarloPolicy(Policy):
    """The transformer approximator: greedy/epsilon-greedy decide, Monte Carlo learn.

    Owns no trainable state of its own — ``encoder``/``head``/``ridge`` are
    injected, mutated in place by :meth:`learn`, and are the caller's to keep
    alive across Episodes (ticket 07's Trainer does exactly this: one Policy
    instance is built fresh per Episode around the same long-lived network and
    ridge accumulator, mirroring how ``MonteCarloPolicy``'s ``W`` flows in and
    out — except here the mutation is in-place on shared objects, not a fresh
    array). Since ticket 16, ``encoder``/``head.layer1`` are never mutated by
    this class at all on the frozen-encoder arm -- only ``head.linear``/
    ``head.layer2``, and only by :meth:`learn`'s ridge solve.

    ## The two arms (ticket 17)

    ``train_encoder=False`` (the default) is the frozen-encoder arm exactly as
    ticket 16 shipped it: ``learn`` only ever folds a decision epoch into
    ``self.ridge`` and (on cadence) writes the closed-form solve onto
    ``head.linear``/``head.layer2`` -- no gradient of any kind. ``train_encoder=
    True`` is the trained-encoder arm ("two timescales", spec.md decision 9's
    amendment): after the ridge fold/solve above, :meth:`_train_encoder_step`
    additionally runs ``neural_learn_passes`` shuffled minibatch passes of SGD
    over ``encoder``/``head.layer1`` — never ``head.linear``/``head.layer2``,
    which stay exclusively ridge-governed — minimizing the same residual
    target the ridge solve regresses onto, with the *current* solved ``W``
    held fixed (a plain readout, ``head.w_vector()``, already detached). This
    needs ``encoder_optimizer`` (scoped to exactly those two parameter groups —
    see :func:`~stdvrp.training.neural_episode.build_neural_policy_state`) and
    ``learn_rng`` (the per-episode minibatch shuffle); both are required
    whenever ``train_encoder`` is ``True`` and unused otherwise.
    """

    def __init__(
        self,
        number_vehicles: int,
        geometry: EpisodeGeometry,
        time_windows: TimeWindows,
        number_clients: int,
        epsilon: float,
        depot: int,
        shift_end_minute: int,
        episode_end_minute: int,
        horizon_start_minute: int,
        encoder: TokenEncoder,
        head: QHead,
        ridge: RidgeAccumulator,
        *,
        exploration_rng: np.random.Generator,
        solve_cadence: int = 1,
        device: torch.device | None = None,
        train_encoder: bool = False,
        encoder_optimizer: torch.optim.Optimizer | None = None,
        learn_rng: np.random.Generator | None = None,
        learn_passes: int = 1,
        batch_size: int = 32,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.number_vehicles = number_vehicles
        self.geometry = geometry
        self.time_windows = time_windows
        self.number_clients = number_clients
        self.epsilon = epsilon
        self.depot = depot
        self.shift_end_minute = shift_end_minute
        self.episode_end_minute = episode_end_minute
        self.horizon_start_minute = horizon_start_minute
        self.encoder = encoder
        self.head = head
        # Ticket 16: the accumulated-least-squares estimator (module
        # docstring, "The estimator") -- the caller's to keep alive across
        # Episodes, exactly like encoder/head.
        self.ridge = ridge
        # How many learn() calls (Episodes) pass between one ridge solve and
        # the next -- ExperimentConfig.neural_solve_cadence. 1 (the default)
        # re-solves every Episode; every direct-construction call site (the
        # unit tests among them) keeps working unchanged.
        self.solve_cadence = solve_cadence
        # Ticket 12: this class builds several ad hoc tensors of its own (the
        # infeasible mask, claimed, the depot arc pair, learn's target) that
        # are never part of encoder/head's parameters, so `.to(device)` at
        # their construction does not cover these -- they must be built on the
        # same device as encoder/head, or every op below errors on a device
        # mismatch. Defaults to CPU so every existing direct-construction call
        # site (the unit tests among them) keeps working unchanged.
        self.device = device if device is not None else torch.device("cpu")

        # Ticket 13 discipline (ADR-0001 phase 2): one injected Generator per
        # stochastic concern, never a global. ``exploration_rng`` is
        # decide_train's epsilon gate and exploratory pick. Ticket 16 retired
        # the second stream this class used to need (``learn_rng``, the
        # per-episode minibatch shuffle) -- the ridge solve has no shuffling of
        # anything to do; ticket 17's trained-encoder arm brings it back (below)
        # for its own SGD minibatch shuffle, which the frozen arm still has no
        # use for.
        self.exploration_rng = exploration_rng

        # Ticket 17: the trained-encoder arm ("two timescales", class
        # docstring). ``encoder_optimizer``/``learn_rng`` are required
        # whenever ``train_encoder`` is True -- there is no sensible default
        # optimizer or RNG to fall back on silently, and a caller that asked
        # for this arm without providing either almost certainly has a wiring
        # bug worth failing loudly on, not a frozen-arm run that happens to
        # ignore them.
        if train_encoder and (encoder_optimizer is None or learn_rng is None):
            raise ValueError("train_encoder=True needs both encoder_optimizer and learn_rng")
        self.train_encoder = train_encoder
        self.encoder_optimizer = encoder_optimizer
        self.learn_rng = learn_rng
        self.learn_passes = learn_passes
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm

        self._episode_length = float(episode_end_minute - horizon_start_minute)
        self._return_scale = float(number_clients) * self._episode_length
        # Ticket 16 (module docstring, "The estimator"): a guaranteed floor
        # under any aborted Episode's actual penalty for this Episode's
        # demand (``served <= number_clients`` always), used by
        # :meth:`_is_aborted` to detect one from the final transition's
        # reward alone -- the only signal available under `learn`'s frozen
        # ``TrainablePolicy`` protocol.
        self._abort_reward_floor = ABORT_PENALTY - ABORT_PENALTY_PER_SERVED_CLIENT * number_clients

        # Ticket 14 (ADR-0011): the shared action_set module, at the linear
        # baseline's own m + 2 -- see this module's docstring, "ADR-0011".
        self.feature_extractor = FeatureExtractor(
            geometry,
            time_windows,
            number_vehicles=number_vehicles,
            number_clients=number_clients,
            depot=depot,
            shift_end_minute=shift_end_minute,
            episode_end_minute=episode_end_minute,
            service_time=_SERVICE_TIME,
            delay_cost_factor=_DELAY_COST_FACTOR,
            earliness_cost_factor=_EARLINESS_COST_FACTOR,
            overtime_cost_factor=_OVERTIME_COST_FACTOR,
        )
        self._number_actions = number_vehicles + 2
        # select_vehicle_possible_actions' current_action -- mutated in place
        # vehicle by vehicle inside _sweep, exactly as MonteCarloPolicy.action
        # is (Model.run_training_episode's own comment already documents this
        # aliasing generically as "policy.action"). Starts at the depot for
        # every vehicle: depot is never a member of clients_not_visited, so an
        # unprocessed vehicle's sentinel excludes nothing from anyone else's
        # candidates until it has actually decided something.
        self.action: list[int] = [depot] * number_vehicles

        # Ticket 07: not part of the TrainablePolicy protocol (learn returns
        # None, matching MonteCarloPolicy), read separately by the Trainer's
        # live per-episode report. Ticket 16 repurposes this from "the mean
        # Huber loss over the last learn() call's minibatches" to "the mean
        # squared residual the most recent (non-aborted, non-empty) Episode's
        # samples show against the entering W" (module docstring, "The
        # estimator") -- left untouched on an aborted or empty Episode. 0.0
        # before the first contributing learn() call.
        self.last_loss = 0.0

        # Ticket 17: Gate A''s companion diagnostic, r = sd_candidates(W.phi) /
        # sd_candidates(c) (spec.md decision 10's amendment) -- not part of the
        # TrainablePolicy protocol, read separately by callers exactly like
        # ``last_loss``. One ``(sd_residual, sd_myopic_base)`` pair per
        # (decision epoch, vehicle) with >= 2 candidates, appended by
        # :meth:`_sweep`'s greedy branch -- "decide()-time candidate spread"
        # (ticket 16's own words), so this fills during both `decide` (always
        # greedy) and `decide_train` (greedy whenever the epsilon gate does not
        # fire). Starts empty every Episode: a fresh Policy wraps the same
        # long-lived encoder/head every Episode (this class's own docstring),
        # so this list is this Episode's own, never carried over from the last.
        self.spread_samples: list[tuple[float, float]] = []

    # --- Acting ------------------------------------------------------------

    def decide(self, state: State) -> list[int]:
        """Greedy per-vehicle argmin over every feasible action. No randomness."""
        with torch.no_grad():
            return self._sweep(state)

    def decide_train(self, state: State) -> list[int]:
        """Same sweep, epsilon-greedy per vehicle from the injected exploration_rng."""
        with torch.no_grad():
            return self._sweep(state, epsilon=self.epsilon, rng=self.exploration_rng)

    def _sweep(
        self,
        state: State,
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """One encoder pass, then ``m`` cheap per-vehicle head passes (spec.md decision 6).

        Ticket 14 (ADR-0011): candidates come from
        :func:`~stdvrp.policies.action_set.select_vehicle_possible_actions` —
        the linear baseline's own, unmodified, at ``m + 2`` — with
        ``self.action`` threaded through as ``current_action``. Two things are
        layered on top, Policy-side, because ``action_set.py`` does not know
        about either: the self-node exclusion (ADR-0008, B20 — the depot is
        never filtered by it) and a depot fallback for the resulting empty set.
        ``_is_retired``/``_depot_is_feasible`` retired with this change; see
        this module's docstring, "ADR-0011", for where their reasoning went.

        Narrower than ``decide``/``decide_train``'s shared ancestor type: only
        ``State`` has the mutable identity ``select_vehicle_possible_actions``
        expects, and ``_sweep`` is never called with a ``TrainingSnapshot`` —
        ``_replay_joint_q`` below is the snapshot-replay path, and it does not
        call this method.
        """
        tokens = tokenize(
            state,
            self.geometry,
            self.time_windows,
            horizon_start_minute=self.horizon_start_minute,
            shift_end_minute=self.shift_end_minute,
            episode_end_minute=self.episode_end_minute,
        )
        embeddings = self.encoder(tokens)
        features = self.feature_extractor.state_features(state)

        pending = list(state.clients_not_visited)
        n_pending = len(pending)
        pending_array = np.asarray(pending)
        pending_index = {client: index for index, client in enumerate(pending)}

        for vehicle in range(self.number_vehicles):
            vehicle_position = state.last_node_reached[vehicle]
            # Mirrors what select_vehicle_possible_actions computes internally
            # from current_action (it does not expose the list separately):
            # every other vehicle's target, freshly decided this pass or still
            # holding last epoch's in-flight one. Needed here only for the
            # `claimed` input feature below -- action_set already applies the
            # same exclusion to the candidate set itself.
            forbidden_ids = [self.action[v] for v in range(self.number_vehicles) if v != vehicle]

            candidates = select_vehicle_possible_actions(
                self._number_actions,
                vehicle,
                features,
                state,
                self.action,
                self.geometry,
                self.depot,
                self.number_vehicles,
                self.shift_end_minute,
            )
            candidates = [c for c in candidates if c == self.depot or c != vehicle_position]
            if not candidates:
                candidates = [self.depot]

            if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
                chosen_id = int(rng.choice(candidates))
            else:
                claimed_mask = np.isin(pending_array, forbidden_ids)
                q, residual, myopic_base = self._score_with_components(
                    embeddings, vehicle, pending, claimed_mask
                )
                allowed = torch.zeros(n_pending + 1, dtype=torch.bool, device=self.device)
                for candidate in candidates:
                    index = n_pending if candidate == self.depot else pending_index[candidate]
                    allowed[index] = True
                masked = q.clone()
                masked[~allowed] = float("inf")
                winner = int(torch.argmin(masked).item())
                chosen_id = self.depot if winner == n_pending else pending[winner]

                # Ticket 17's r diagnostic: the spread *among this decision's
                # actual candidates* (`allowed`), not every scored pending
                # Client -- the argmin only ever reads the former. Undefined
                # (and uninformative either way) with fewer than two
                # candidates, which every vehicle always has at least one of
                # (module docstring's depot fallback), so this only ever skips
                # the single-candidate case.
                if int(allowed.sum().item()) >= 2:
                    residual_sd = float(residual[allowed].std(unbiased=False).item())
                    cost_sd = float(myopic_base[allowed].std(unbiased=False).item())
                    self.spread_samples.append((residual_sd, cost_sd))

            self.action[vehicle] = chosen_id

        return self.action

    def _candidate_rows(
        self,
        embeddings: Embeddings,
        vehicle: int,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every pending Client plus the synthetic depot row, in that order --
        the four tensors :meth:`_score` (``QHead.forward``) and
        :meth:`_score_features` (``QHead.features``, ticket 16) both need and
        must never disagree about: the augmented candidate embedding, the
        myopic base ``c`` (ticket 15, ``network.py``, "The myopic base"),
        ``claimed`` and ``is_depot``.
        """
        n_pending = len(pending)
        client_embeddings = embeddings.clients[:, vehicle, :]
        augmented = torch.cat([client_embeddings, embeddings.depot[vehicle].unsqueeze(0)], dim=0)
        myopic_base = torch.cat(
            [embeddings.cost[:, vehicle], embeddings.depot_cost[vehicle].unsqueeze(0)], dim=0
        )

        claimed = torch.zeros(n_pending + 1, dtype=torch.float32, device=self.device)
        if n_pending:
            claimed[:n_pending] = torch.from_numpy(claimed_mask.astype(np.float32)).to(self.device)

        # The one structural fact separating the synthetic depot row from a real
        # Client's embedding row (the myopic-base gap between them is carried
        # separately, in `myopic_base` above -- network.py, "The depot's place
        # in `c`"). Without this flag the head cannot tell the two apart at
        # all -- the depot row's "context" half is the vehicle's own embedding,
        # which every Client row of this sweep also carries in its
        # `vehicle_embedding` argument.
        is_depot = torch.zeros(n_pending + 1, dtype=torch.float32, device=self.device)
        is_depot[n_pending] = 1.0

        return augmented, myopic_base, claimed, is_depot

    def _score(
        self,
        embeddings: Embeddings,
        vehicle: int,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> torch.Tensor:
        """``Q(vehicle, candidate) == c(s, v, candidate) + QHead(...)`` over every
        pending Client plus the depot, in that order (ticket 15, ``network.py``,
        "The myopic base") -- the addition ``QHead`` itself never performs.

        A thin wrapper over :meth:`_score_with_components` (ticket 17): kept as
        its own method, rather than folded away, because several direct
        callers only ever want the combined ``q`` — ``_sweep``'s own
        epsilon-exploration branch aside, ``test_transformer_policy.py``'s
        ``TestDepotMyopicBase``/``TestJointQIsAdditiveOverVehicles`` call
        ``policy._score(...)`` directly at this exact signature, and rewriting
        them to unpack a triple they do not need would add noise for no
        benefit.
        """
        q, _residual, _myopic_base = self._score_with_components(
            embeddings, vehicle, pending, claimed_mask
        )
        return q

    def _score_with_components(
        self,
        embeddings: Embeddings,
        vehicle: int,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q, residual, myopic_base)`` over every pending Client plus the
        depot -- :meth:`_score`'s own computation, split apart so
        :meth:`_sweep` can read the residual/myopic-base halves separately for
        ticket 17's ``r`` diagnostic (``spec.md`` decision 10's amendment:
        ``r = sd_candidates(W.phi) / sd_candidates(c)``) without recomputing
        either.
        """
        augmented, myopic_base, claimed, is_depot = self._candidate_rows(
            embeddings, vehicle, pending, claimed_mask
        )
        residual = self.head(embeddings.vehicles[vehicle], augmented, claimed, is_depot)
        q: torch.Tensor = myopic_base + residual
        return q, residual, myopic_base

    def _score_features(
        self,
        embeddings: Embeddings,
        vehicle: int,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(phi(s, v, candidate), c(s, v, candidate))`` over every pending
        Client plus the depot -- ticket 16's regression feature and myopic
        base, from the same candidate rows :meth:`_score` scores
        (``Q == c + phi @ head.w_vector()`` exactly matches
        ``_score``'s ``myopic_base + head(...)``; ``network.py``,
        ``QHead.features``). Never called with gradients enabled: the ridge
        estimator is solved in closed form, not by backpropagation.
        """
        augmented, myopic_base, claimed, is_depot = self._candidate_rows(
            embeddings, vehicle, pending, claimed_mask
        )
        phi = self.head.features(embeddings.vehicles[vehicle], augmented, claimed, is_depot)
        return phi, myopic_base

    # --- Learning ------------------------------------------------------------

    def learn(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> None:
        """Fold one Episode into the ridge accumulator; re-solve on cadence
        (ticket 16, module docstring "The estimator"). Excludes aborted
        Episodes (:meth:`_is_aborted`) from the accumulator, but not from its
        forgetting -- the accumulated memory still ages by one Episode either
        way.

        Ticket 17: on the trained-encoder arm (``self.train_encoder``),
        :meth:`_train_encoder_step` additionally runs after the ridge fold/
        solve below, using whatever ``W`` the ridge holds at that point --
        "held at its last solve" (spec.md's "Two timescales"). An aborted
        Episode is skipped by the SGD step too, for the identical reason the
        ridge accumulator excludes it (module docstring, "The heavy tail"):
        the terminal penalty carries no ranking information to buy a gradient
        step with either.
        """
        T = len(actions)
        if T == 0:
            return

        targets = self._backward_returns(snapshots, actions, rewards)
        aborted = self._is_aborted(rewards)

        if aborted:
            empty_phi = np.empty((0, self.ridge.dim), dtype=np.float64)
            empty_y = np.empty(0, dtype=np.float64)
            self.ridge.observe_episode(empty_phi, empty_y, aborted=True)
        else:
            phi_rows = np.empty((T, self.ridge.dim), dtype=np.float64)
            y_rows = np.empty(T, dtype=np.float64)
            with torch.no_grad():
                w_vector = self.head.w_vector()
                for t in range(T):
                    phi_sum, c_sum = self._replay_joint_features(snapshots[t], actions[t])
                    y_t = targets[t] / self._return_scale - c_sum
                    phi_rows[t] = phi_sum.cpu().numpy()
                    y_rows[t] = y_t
                residual = y_rows - phi_rows @ w_vector.cpu().numpy()
            self.last_loss = float(np.mean(np.square(residual)))
            self.ridge.observe_episode(phi_rows, y_rows, aborted=False)

        if self.ridge.episodes_since_solve >= self.solve_cadence:
            solved = self.ridge.solve()
            self.head.load_w_vector(torch.from_numpy(solved.astype(np.float32)).to(self.device))

        if self.train_encoder and not aborted:
            self._train_encoder_step(snapshots, actions, targets)

    def _train_encoder_step(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        targets: list[float],
    ) -> None:
        """Ticket 17's trained-encoder arm: ``self.learn_passes`` shuffled
        minibatch passes of SGD over ``encoder``/``head.layer1`` only --
        ``head.linear``/``head.layer2`` never receive a gradient here, since
        they are exclusively the ridge solve's to move (class docstring,
        "The two arms"; ``encoder_optimizer`` is scoped to exactly the two
        parameter groups that may move, see
        :func:`~stdvrp.training.neural_episode.build_neural_policy_state`).

        The loss is the *same* residual target the ridge accumulator regresses
        onto (``y_t = targets[t] / return_scale - c_sum``), evaluated against
        the *current* solved ``W`` held fixed (a plain, detached readout,
        ``head.w_vector()``) -- "held at its last solve" (spec.md's "Two
        timescales"). Each decision epoch costs one encoder forward pass
        (:meth:`_joint_features`, called *without* ``torch.no_grad()`` so a
        real graph reaches ``encoder``/``head.layer1``); a minibatch's samples
        share one ``backward()`` call, which accumulates their gradients onto
        the same leaf parameters exactly as a batch of any other variable-shape
        input would.
        """
        assert self.encoder_optimizer is not None
        assert self.learn_rng is not None
        T = len(actions)
        w = self.head.w_vector()  # detached (QHead.w_vector's own contract)
        indices = np.arange(T)
        for _ in range(self.learn_passes):
            self.learn_rng.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch = indices[start : start + self.batch_size]
                self.encoder_optimizer.zero_grad()
                losses = []
                for t in batch:
                    phi_sum, c_sum = self._joint_features(snapshots[t], actions[t])
                    y_t = targets[t] / self._return_scale - c_sum
                    predicted_residual = phi_sum @ w
                    losses.append((predicted_residual - y_t) ** 2)
                loss = torch.stack(losses).mean()
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [*self.encoder.parameters(), *self.head.layer1.parameters()],
                        self.grad_clip_norm,
                    )
                self.encoder_optimizer.step()

    def _is_aborted(self, rewards: list[float]) -> bool:
        """Whether this Episode ended by hitting ``CLOCK_CEILING`` (research
        note F10; module docstring, "The estimator") -- read off the final
        transition's reward, the only signal available under ``learn``'s
        frozen ``TrainablePolicy`` signature. ``rewards[-1]`` carries the
        abort penalty in full whenever it fired (plus whatever else that same
        transition charged), and ``self._abort_reward_floor`` is a guaranteed
        lower bound on that penalty for this Episode's demand -- see
        ``__init__``.
        """
        return rewards[-1] >= self._abort_reward_floor

    def _replay_joint_features(
        self, snapshot: TrainingSnapshot, action_row: list[int]
    ) -> tuple[torch.Tensor, float]:
        """``(Phi_t, sum_v c(s, v, a_v))`` for one decision epoch -- the ridge
        accumulator's own per-epoch sample (module docstring, "The
        estimator"). Mirrors :meth:`_replay_joint_q`'s replay exactly (one
        encoder pass, the same incremental ``claimed_mask`` seeded fresh per
        epoch) but sums :meth:`_score_features`'s ``phi`` instead of scoring
        through ``QHead.forward`` -- no gradient is ever built here, since the
        ridge estimator is solved in closed form. Every vehicle contributes a
        term, including one whose action the simulator will discard, for the
        same reason :meth:`_replay_joint_q` does (module docstring, "A
        tried-and-rejected variant").

        A thin ``torch.no_grad()`` wrapper over :meth:`_joint_features`
        (ticket 17): the trained-encoder arm's own SGD step
        (:meth:`_train_encoder_step`) calls that same core *without* the
        wrapper, so a real autograd graph reaches ``encoder``/``head.layer1``
        -- the one thing this method must never do, since every other caller
        (``learn``'s ridge fold, ``calibration_pairs``/``residual_calibration_pairs``)
        needs a plain readout, not a graph to hold onto.
        """
        with torch.no_grad():
            return self._joint_features(snapshot, action_row)

    def _joint_features(
        self, snapshot: TrainingSnapshot, action_row: list[int]
    ) -> tuple[torch.Tensor, float]:
        """Gradient-transparent core of :meth:`_replay_joint_features` (ticket
        17): identical computation, but never wrapped in ``torch.no_grad()``
        itself, so a caller outside any no-grad context gets a real autograd
        graph from ``phi_sum`` back through ``encoder``/``head.layer1`` --
        exactly what :meth:`_train_encoder_step` needs and everything else
        that reaches this method (through the wrapper above) must not get.
        """
        tokens = tokenize(
            snapshot,
            self.geometry,
            self.time_windows,
            horizon_start_minute=self.horizon_start_minute,
            shift_end_minute=self.shift_end_minute,
            episode_end_minute=self.episode_end_minute,
        )
        embeddings = self.encoder(tokens)

        pending = list(snapshot.clients_not_visited)
        n_pending = len(pending)
        index_of = {client: index for index, client in enumerate(pending)}
        claimed_mask = np.zeros(n_pending, dtype=np.bool_)

        phi_sum = torch.zeros(self.head.feature_dim, dtype=torch.float32, device=self.device)
        c_sum = 0.0
        for vehicle in range(self.number_vehicles):
            phi, myopic_base = self._score_features(embeddings, vehicle, pending, claimed_mask)
            chosen = action_row[vehicle]
            if chosen == self.depot:
                index = n_pending
            else:
                index = index_of[chosen]
                claimed_mask[index] = True
            phi_sum = phi_sum + phi[index]
            c_sum += float(myopic_base[index].item())
        return phi_sum, c_sum

    def calibration_pairs(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> list[CalibrationPair]:
        """``(Q_predicted, U_t)`` once per decision epoch (ticket 08, Gate A).

        ``Q_predicted`` is the joint ``Q`` the network assigns the action row
        actually taken (:meth:`_replay_joint_q`) -- the same quantity
        :meth:`learn` regresses; ``U_t`` is the backward Monte Carlo return it
        regresses onto (:meth:`_backward_returns`). Correlating the two is Gate
        A's calibration check (spec.md) -- whether the network learned the
        value function it was trained to learn, independent of whether the
        policy it induces improved. Pairing the *joint* prediction is what
        makes the check meaningful: emitting one pair per (epoch, vehicle), as
        this did before the joint decomposition below, repeated each epoch's
        single ``U_t`` ``m`` times against ``m`` different predictions, so the
        rank correlation was partly measuring an artefact of that duplication.
        Read-only: no gradient is built, no parameter or RNG stream is touched.
        """
        T = len(actions)
        if T == 0:
            return []

        targets = self._backward_returns(snapshots, actions, rewards)
        pairs: list[CalibrationPair] = []
        with torch.no_grad():
            for t in range(T):
                q_pred = float(self._replay_joint_q(snapshots[t], actions[t]).item())
                pairs.append(CalibrationPair(predicted_q=q_pred, realised_u=targets[t]))
        return pairs

    def residual_calibration_pairs(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> list[ResidualCalibrationPair]:
        """``(W . phi(s, a), y_t)`` once per decision epoch -- Gate A' 's (ticket
        17) calibration primitive, replacing :meth:`calibration_pairs`'
        ``(Q, U_t)`` pairing for the reason :class:`ResidualCalibrationPair`
        documents: ``rho(Q, U_t)`` passes at ``W = 0`` with no parameter having
        moved, since ``Q == c`` there regardless of the return. Pairing the
        learned term against the residual it is actually regressed onto
        (``learn``'s own ``y_t = targets[t] / return_scale - sum_v c(s, v,
        a_v)``) is the check that cannot be faked that way -- at ``W = 0`` the
        predicted half is identically zero while ``y_t`` still varies with the
        return, so the correlation reads ~0 exactly as it should.

        Read-only: no gradient is built, no parameter or RNG stream is
        touched -- the same discipline :meth:`calibration_pairs` follows.
        """
        T = len(actions)
        if T == 0:
            return []

        targets = self._backward_returns(snapshots, actions, rewards)
        pairs: list[ResidualCalibrationPair] = []
        with torch.no_grad():
            w = self.head.w_vector()
            for t in range(T):
                phi_sum, c_sum = self._replay_joint_features(snapshots[t], actions[t])
                y_t = targets[t] / self._return_scale - c_sum
                predicted_residual = float((phi_sum @ w).item())
                pairs.append(
                    ResidualCalibrationPair(
                        predicted_residual=predicted_residual, residual_target=y_t
                    )
                )
        return pairs

    def _replay_joint_q(self, snapshot: TrainingSnapshot, action_row: list[int]) -> torch.Tensor:
        """``sum_v Q(s, v, action_row[v])`` for one decision epoch, gradients attached.

        One encoder pass serves the whole row (the same economy ``_sweep`` gets
        at decide time), and ``claimed`` replays ``_sweep``'s index-order
        claiming: every earlier-indexed vehicle's realized target is marked
        before the next vehicle is scored. Every vehicle contributes a term,
        including one whose action the simulator will discard -- the
        executable-only variant was measured worse and reverted (module
        docstring, "A tried-and-rejected variant").
        """
        tokens = tokenize(
            snapshot,
            self.geometry,
            self.time_windows,
            horizon_start_minute=self.horizon_start_minute,
            shift_end_minute=self.shift_end_minute,
            episode_end_minute=self.episode_end_minute,
        )
        embeddings = self.encoder(tokens)

        pending = list(snapshot.clients_not_visited)
        n_pending = len(pending)
        index_of = {client: index for index, client in enumerate(pending)}
        claimed_mask = np.zeros(n_pending, dtype=np.bool_)

        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for vehicle in range(self.number_vehicles):
            q = self._score(embeddings, vehicle, pending, claimed_mask)
            chosen = action_row[vehicle]
            if chosen == self.depot:
                total = total + q[n_pending]
            else:
                index = index_of[chosen]
                total = total + q[index]
                claimed_mask[index] = True
        return total

    def _backward_returns(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> list[float]:
        """``U_t - acquired_cost`` per decision epoch — the same target ``update_W`` uses."""
        T = len(actions)
        targets = [0.0] * T
        u_t = 0.0
        for t in range(T - 1, -1, -1):
            u_t += rewards[t + 1]
            targets[t] = u_t - self._already_acquired_cost(snapshots[t])
        return targets

    def _already_acquired_cost(self, snapshot: TrainingSnapshot) -> float:
        """Ports ``MonteCarloPolicy._already_acquired_cost`` (see this module's docstring)."""
        total = 0.0
        for client in snapshot.clients_not_visited:
            due = self.time_windows[client][1]
            if due < snapshot.tau_episode:
                total += (snapshot.tau_episode - due) * _DELAY_COST_FACTOR
        for vehicle in range(self.number_vehicles):
            at_depot = is_parked_at_depot(
                snapshot.last_node_reached[vehicle], snapshot.vehicle_standing[vehicle], self.depot
            )
            if not at_depot and snapshot.tau_episode > self.shift_end_minute:
                total += (snapshot.tau_episode - self.shift_end_minute) * _OVERTIME_COST_FACTOR
        return total
