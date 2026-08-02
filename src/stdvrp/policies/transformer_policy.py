"""TransformerMonteCarloPolicy: decide, decide_train, learn (ticket 06, neural-policy).

The ``TrainablePolicy`` (ticket 02) that scores every pending Client with the
ticket-05 ``TokenEncoder``/``QHead`` and learns from the same Monte Carlo return
``MonteCarloPolicy.learn`` targets. Imports torch at module scope, like
``network.py``: this file's whole reason to exist is the network, so it must
never be imported from ``stdvrp.policies.__init__`` or any module reachable at
package-import time. Callers import it explicitly:
``from stdvrp.policies.transformer_policy import TransformerMonteCarloPolicy``.

## ADR-0007 — the action set is feasibility, not heuristic

``_select_vehicle_possible_actions``, ``_classify_shortest_distance_clients``,
``delayed_clients``, the ``350``/``310`` literals and ``number_actions_test``
are hand-engineered ranking built for the linear baseline's small candidate
pool. None of it applies here. What is left is three constraints, each one a
fact about what the simulator can execute rather than a guess about what is a
good idea:

1. **No double booking** (the B11 invariant): a Client another vehicle already
   claimed this decision epoch is not a candidate.
2. **No self-node** (ticket 11, B20, ADR-0008): a pending Client the vehicle is
   already standing on is not a candidate — there is nothing to travel to.
3. **Home is not a destination, it is an exit** — see
   :meth:`TransformerMonteCarloPolicy._depot_is_feasible`, which amends
   ADR-0007's original "plus the depot (always legal)" and documents the Gate A
   measurement that forced the amendment.

All three are enforced by :meth:`TransformerMonteCarloPolicy._sweep` directly,
never reconstructed from a heuristic, and none of them ranks the candidates
that survive: *which* feasible action wins is entirely the network's argmin.
``claimed`` is additionally fed to the network as an input (spec.md decision 6:
"claimed enters at the head"), so a trained network's *predictions* can account
for contention, but the legality of an action never depends on what the network
outputs for it.

## The depot's Q value

``QHead`` scores ``Embeddings.clients`` — one row per **pending Client** — so
the depot, which is never a Client, has no natural client row to score. The
tokenizer emits its arc facts as ``Tokens.depot_arc_tokens`` (the same six
fields every real candidate's arc vector carries, projected costs included —
see ``tokenizer.py``, "The cost fields") and the encoder builds the synthetic
candidate row as ``Embeddings.depot``:

    depot_row = concat([vehicle_context, arc_embed(depot_arc_tokens[v])])

``_score`` appends ``embeddings.depot[vehicle]`` to the client rows before
calling ``QHead`` — one uniform pathway, rather than this module hand-building
a 2-wide pair from a separate ``EpisodeGeometry`` read as it did before the
decision-1 amendment. The "context" half of a real Client's row is that
Client's transformer-refined embedding; the depot has no such thing, so its
row uses the vehicle's own context embedding instead — a deliberate choice,
not an arbitrary filler: the depot's meaning is "return to base", which is a
fact about the *vehicle* (its remaining capacity, how deep into the shift it
is), not about the destination.

The row additionally carries an ``is_depot`` flag into ``QHead`` beside
``claimed``: it is the only thing that distinguishes this synthetic candidate
from a real Client's row, and the warm start reads it (``network.py``,
"The depot is the last resort at init") to price going home at one whole
horizon. So at construction ``Q(v, depot) == minutes_to_depot / horizon_length
+ 1`` while every Client scores ``minutes_to_client / horizon_length <= 1``,
and the untrained greedy policy is **"go to the nearest feasible Client, home
only when no Client is feasible"** — the null model spec.md specifies.

An earlier version of this file left the flag out, so the depot got the same
myopic warm start as a real candidate: ``Q(v, depot) == minutes_to_depot /
horizon_length``, which is exactly ``0`` for a vehicle standing on the depot.
Every vehicle starts parked there, so every vehicle's argmin was the depot at
decision epoch 1, ``Model`` saw ``fleet.all_parked()``, and the Episode
terminated after one transition with every Client unserved (ticket 08 measured
the resulting "null model" at mean cost 81 701 against the linear baseline's
2 483). ``Model._reroute_for`` reroutes only *travelling* vehicles, so the same
warm start also retired vehicles permanently whenever home happened to be
nearer than any Client.

The flag alone is a **prior, not a guarantee** — measured on the mini fixture,
one episode of training is enough for ``QHead``'s Xavier-random background
units to overtake row 0 and put the depot back under every Client, at which
point the fleet retires again and the Episode stops producing training signal.
:meth:`TransformerMonteCarloPolicy._depot_is_feasible` is what makes that
unreachable; the flag is what keeps the *untrained* policy honest inside the
window where heading home is legal.

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
network's output already lives in normalized units, because ticket 05's warm
start pins it at ``minutes_from_vehicle / horizon_length`` (``network.py``).
The two scales meet — a Chengdu episode the linear baseline runs at cost ~2 500
gives ``y ≈ 2500 / (150 * 850) ≈ 0.020``, against a warm-started ``Q`` of
``~5 min / 480 min ≈ 0.010``.

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

## One sample per decision epoch, over the *joint* action

``learn`` regresses ``Q_joint(s, a) = sum_v Q(s, v, a_v)`` — the whole action
row's summed score — onto that epoch's Monte Carlo return. One sample per
decision epoch, not ``m``.

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

from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.nn.functional as functional

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.policies.base import Policy
from stdvrp.policies.feature_extraction import TimeWindows
from stdvrp.policies.network import Embeddings, QHead, TokenEncoder
from stdvrp.policies.tokenizer import tokenize
from stdvrp.simulation.state import is_parked_at_depot

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = ["CalibrationPair", "TransformerMonteCarloPolicy"]


class CalibrationPair(NamedTuple):
    """One decision epoch/vehicle's calibration sample (ticket 08, Gate A).

    Named rather than a bare ``tuple[float, float]``: three modules
    (this one, ``neural_episode.py``, ``gate_a.py``) all pass this value
    around by position, and "which half is predicted vs. realised" is exactly
    the kind of thing a bare tuple lets a caller silently swap.
    """

    predicted_q: float
    realised_u: float


# Legacy cost factors MonteCarloPolicy hardcodes (see this module's docstring,
# "Why _already_acquired_cost is duplicated, not shared").
_DELAY_COST_FACTOR = 1.0
_OVERTIME_COST_FACTOR = 5 / 6


class TransformerMonteCarloPolicy(Policy):
    """The transformer approximator: greedy/epsilon-greedy decide, Monte Carlo learn.

    Owns no trainable state of its own — ``encoder``/``head``/``optimizer`` are
    injected, mutated in place by :meth:`learn`, and are the caller's to keep
    alive across Episodes (ticket 07's Trainer does exactly this: one Policy
    instance is built fresh per Episode around the same long-lived network and
    optimizer, mirroring how ``MonteCarloPolicy``'s ``W`` flows in and out —
    except here the mutation is in-place on shared objects, not a fresh array).
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
        optimizer: torch.optim.Optimizer,
        *,
        exploration_rng: np.random.Generator,
        learn_rng: np.random.Generator,
        learn_passes: int,
        batch_size: int,
        device: torch.device | None = None,
        grad_clip_norm: float | None = None,
        huber_delta: float = 1.0,
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
        self.optimizer = optimizer
        self.learn_passes = learn_passes
        self.batch_size = batch_size
        # Ticket 08 (stability sweep): optional max L2 norm for one minibatch
        # step's gradient, over encoder+head jointly. None disables clipping.
        self.grad_clip_norm = grad_clip_norm
        # Ticket 08: where the Huber loss switches from quadratic to linear.
        # torch's default of 1.0 is far above every residual this regression
        # produces (see ``learn``), so it is exactly ``0.5 * MSE`` -- see
        # ``ExperimentConfig.neural_huber_delta``. Kept as the default here so
        # every direct-construction call site is unchanged.
        self.huber_delta = huber_delta
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
        # decide_train's epsilon gate and exploratory pick; ``learn_rng`` is
        # learn's minibatch shuffle.
        self.exploration_rng = exploration_rng
        self.learn_rng = learn_rng

        self._episode_length = float(episode_end_minute - horizon_start_minute)
        self._return_scale = float(number_clients) * self._episode_length

        # Ticket 07: the mean (standardized) Huber loss over every minibatch of
        # the most recent learn() call -- not part of the TrainablePolicy
        # protocol (learn returns None, matching MonteCarloPolicy), read
        # separately by the Trainer's live per-episode report. 0.0 before the
        # first learn() call.
        self.last_loss = 0.0

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
        state: State | TrainingSnapshot,
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """One encoder pass, then ``m`` cheap per-vehicle head passes (spec.md decision 6)."""
        tokens = tokenize(
            state,
            self.geometry,
            self.time_windows,
            horizon_start_minute=self.horizon_start_minute,
            shift_end_minute=self.shift_end_minute,
            episode_end_minute=self.episode_end_minute,
        )
        embeddings = self.encoder(tokens)

        tau = float(state.tau_episode)
        pending = list(state.clients_not_visited)
        n_pending = len(pending)
        pending_array = np.asarray(pending)
        claimed_mask = np.zeros(n_pending, dtype=np.bool_)
        action = [self.depot] * self.number_vehicles

        for vehicle in range(self.number_vehicles):
            vehicle_position = state.last_node_reached[vehicle]
            if self._is_retired(tau, vehicle_position, state.vehicle_standing[vehicle]):
                # Already home for good: the depot, and -- the part that matters
                # -- no claim. See :meth:`_is_retired`.
                continue

            # ADR-0008: a node the vehicle is already on is not a travel
            # destination -- there is nothing to travel to. Policy-side
            # feasibility, not a heuristic (unlike ``claimed_mask``, this
            # never applies to the depot, which keeps both its meanings).
            infeasible_mask = claimed_mask | (pending_array == vehicle_position)
            depot_feasible = self._depot_is_feasible(
                tau,
                vehicle_position,
                no_client_feasible=not bool(n_pending) or bool(infeasible_mask.all()),
            )

            if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
                feasible = [index for index in range(n_pending) if not infeasible_mask[index]]
                if depot_feasible:
                    feasible.append(n_pending)  # the depot sentinel
                chosen = int(rng.choice(feasible))
            else:
                q = self._score(embeddings, vehicle, pending, claimed_mask)
                masked = q.clone()
                if n_pending:
                    infeasible = torch.from_numpy(infeasible_mask).to(self.device)
                    masked[:n_pending][infeasible] = float("inf")
                if not depot_feasible:
                    masked[n_pending] = float("inf")
                chosen = int(torch.argmin(masked).item())

            if chosen == n_pending:
                action[vehicle] = self.depot
            else:
                action[vehicle] = pending[chosen]
                claimed_mask[chosen] = True

        return action

    def _is_retired(self, tau: float, vehicle_position: float, standing: bool) -> bool:
        """Has this vehicle already gone home for good? Then it gets no Client.

        A vehicle that reaches the depot is parked (``FleetRoutes.park``) and
        ``Model._reroute_for`` has **no branch that fires for a parked
        vehicle** — none of its three guards can be true once ``arrival_tau ==
        PARKED`` and ``departure_tau == 0``. Whatever this Policy names for
        that vehicle is silently discarded by the simulator.

        Silently discarded, but not harmless: :meth:`_sweep` claims whatever it
        assigns, and ``claimed_mask`` is what stops *another* vehicle taking
        the same Client. A retired vehicle handed the nearest pending Client
        therefore **starves** it — it cannot go itself, and it has locked out
        everyone who could — every decision epoch for the rest of the Episode,
        until that Client is charged unserved at termination. This never fired
        before the depot warm start was corrected (``network.py``) only because
        the bug it fixes got there first: a parked vehicle scored the depot at
        exactly ``0`` and so never claimed anything.

        "Home for good" is read off the two State fields ``is_parked_at_depot``
        already combines, plus one clock: **after the first decision epoch**, a
        vehicle standing on the depot has been there since it parked.
        ``horizon_start_minute`` is the exclusion because at ``tau ==
        horizon_start_minute`` the whole fleet is standing on the depot
        *un*-dispatched, indistinguishable from retired by State alone — and
        ``begin_arc`` clears ``vehicle_standing`` the instant a vehicle is
        dispatched, so from the second epoch on the two cases no longer
        collide. ``MonteCarloPolicy._select_vehicle_possible_actions`` draws
        the same distinction with ``is_parked_at_depot(...) and tau > 350``;
        this is that rule with the config clock the ``350`` literal was
        standing in for (``monte_carlo.py``'s own module docstring lists it
        under "Depot-idle cutoffs are inconsistent literals that ignore the
        configured horizon").
        """
        return tau > self.horizon_start_minute and is_parked_at_depot(
            vehicle_position, standing, self.depot
        )

    def _depot_is_feasible(
        self, tau: float, vehicle_position: float, *, no_client_feasible: bool
    ) -> bool:
        """May this vehicle head home this decision epoch? (ADR-0007, amended.)

        Going to the depot is not a travel action like any other: ``Model``
        parks the vehicle (``FleetRoutes.arrival_tau = PARKED``), and
        ``Model._reroute_for`` reroutes only *travelling* vehicles, so the
        vehicle never works again this Episode. When every vehicle has been
        retired, ``Model.transition_function`` ends the Episode and charges
        every pending Client as unserved. "Return to the depot" therefore does
        not mean "travel there", it means **"retire this vehicle"** — and
        retiring while there is still servable work is not a decision the
        Policy is allowed to explore its way into. Two conditions make it legal:

        1. **There is nothing left to travel to** — no pending Client is
           feasible for this vehicle (all served, all claimed by another
           vehicle, or the only one left is the node it already stands on).
           This is the pre-existing fallback, and it is what makes the depot
           the terminal action of a finished Episode.
        2. **The return leg already breaches the shift** — ``tau +
           average_minutes(position, depot) > shift_end_minute``. Past that
           point the vehicle is on overtime whatever it does, and heading home
           is the only way to stop the meter. This is the same condition
           ``MonteCarloPolicy._select_vehicle_possible_actions`` gates the
           depot behind, minus its ``350``/``310`` literals: a config clock and
           an ``EpisodeGeometry`` read, both already permitted (ADR-0006).

        **This amends ADR-0007's "the depot is always legal".** That clause
        rested on the depot being one more destination the network could learn
        to price, but it is an absorbing, irreversible exit whose warm-started
        score is ``0`` for a vehicle standing on it — so ticket 08's Gate A run
        retired the whole fleet at decision epoch 1 of every evaluation
        Episode, measured the "null model" at mean cost 81 701 against the
        linear baseline's 2 483, and never recovered: a one-transition Episode
        produces almost no training signal, so the network could not learn its
        way out of the trap it had fallen into. The warm-start penalty
        (``network.py``) alone was measured insufficient — one Episode of
        training is enough for ``QHead``'s background units to overtake row 0
        and put the depot back under every Client. B11-style masking is how
        this Policy already expresses a constraint the argmin must not be able
        to violate; this is one more of them, not a ranking heuristic: within
        the window where going home is legal, *which* action wins is still
        entirely the network's call.
        """
        if no_client_feasible:
            return True
        return tau + self.geometry.average_minutes(vehicle_position, self.depot) > (
            self.shift_end_minute
        )

    def _score(
        self,
        embeddings: Embeddings,
        vehicle: int,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> torch.Tensor:
        """``Q(vehicle, candidate)`` over every pending Client plus the depot, in that order."""
        n_pending = len(pending)
        client_embeddings = embeddings.clients[:, vehicle, :]
        augmented = torch.cat([client_embeddings, embeddings.depot[vehicle].unsqueeze(0)], dim=0)

        claimed = torch.zeros(n_pending + 1, dtype=torch.float32, device=self.device)
        if n_pending:
            claimed[:n_pending] = torch.from_numpy(claimed_mask.astype(np.float32)).to(self.device)

        # The one structural fact separating the synthetic depot row from a real
        # Client's, and what the warm start prices going home with (network.py,
        # "The depot is the last resort at init"). Without it the head cannot
        # tell the two apart at all -- the depot row's "context" half is the
        # vehicle's own embedding, which every Client row of this sweep also
        # carries in its `vehicle_embedding` argument.
        is_depot = torch.zeros(n_pending + 1, dtype=torch.float32, device=self.device)
        is_depot[n_pending] = 1.0

        q: torch.Tensor = self.head(embeddings.vehicles[vehicle], augmented, claimed, is_depot)
        return q

    # --- Learning ------------------------------------------------------------

    def learn(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> None:
        """One batch per Episode: K shuffled minibatch passes, then discard (spec.md decision 9)."""
        T = len(actions)
        if T == 0:
            return

        targets = self._backward_returns(snapshots, actions, rewards)

        minibatch_losses: list[float] = []
        for _ in range(self.learn_passes):
            order = self.learn_rng.permutation(T)
            for start in range(0, T, self.batch_size):
                batch = order[start : start + self.batch_size]
                self.optimizer.zero_grad()
                losses = []
                for index in batch:
                    t = int(index)
                    q_pred = self._replay_joint_q(snapshots[t], actions[t])
                    target = torch.tensor(
                        targets[t] / self._return_scale, dtype=torch.float32, device=self.device
                    )
                    losses.append(functional.huber_loss(q_pred, target, delta=self.huber_delta))
                loss = torch.stack(losses).mean()
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        chain(self.encoder.parameters(), self.head.parameters()),
                        self.grad_clip_norm,
                    )
                self.optimizer.step()
                minibatch_losses.append(float(loss.item()))

        if minibatch_losses:
            self.last_loss = sum(minibatch_losses) / len(minibatch_losses)

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
