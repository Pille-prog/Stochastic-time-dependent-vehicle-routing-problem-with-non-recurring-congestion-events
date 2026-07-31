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
pool. None of it applies here: **feasible = every pending Client not already
claimed by another vehicle this decision, plus the depot (always legal)**.
That is the whole action rule (spec.md, ticket 06). The no-double-booking rule
(the B11 invariant) survives as a **constraint** on the argmin's candidate set
— enforced by :meth:`TransformerMonteCarloPolicy._sweep` directly, never
reconstructed from a heuristic — not as a bias the network merely learns to
respect. ``claimed`` is still fed to the network as an input (spec.md decision
6: "claimed enters at the head"), so a trained network's *predictions* can
account for contention, but the actual legality of an action never depends on
what the network outputs for an already-claimed candidate.

## The depot's Q value

``QHead`` scores ``Embeddings.clients`` — one row per **pending Client** — so
the depot, which is never a Client and never a tokenizer row, has no natural
embedding to score. This module builds one: a synthetic "candidate" row
appended to the client embeddings before calling ``QHead``, built the same way
every real candidate's arc half is —

    depot_arc = encoder.arc_embed([minutes_to_depot, length_to_depot] / horizon_length)
    depot_row = concat([vehicle_context, depot_arc])

using ``EpisodeGeometry`` directly (permitted under the observability rule,
ADR-0006 — the same offline prior the linear baseline reads) rather than a new
tokenizer field, so ``tokenizer.py``'s frozen, structurally-pinned signature
(ADR-0006) needs no sixth argument. The "context" half of a real Client's row
is that Client's transformer-refined embedding; the depot has no such thing,
so this uses the vehicle's own context embedding instead — a deliberate
choice, not an arbitrary filler: the depot's meaning is "return to base",
which is a fact about the *vehicle* (its remaining capacity, how deep into the
shift it is), not about the destination.

This choice is invisible at initialization and only starts mattering once
training moves ``QHead.layer1``'s background rows off zero (see
``network.py``'s module docstring): row 0 — the only hidden unit
``layer2`` reads at init — is zeroed except for the column reading
``arc_embed``'s dimension 0, so at construction ``Q(v, depot) ==
minutes_to_depot / horizon_length`` exactly, the same myopic warm start every
real candidate gets (ticket 05). The untrained greedy policy is therefore
"go to whichever feasible target — Client or depot — is nearest by travel
time", not merely "nearest Client with the depot bolted on as an afterthought".

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

## Target scaling

"Standardize ``y`` with fixed, config-derived scales, same discipline as
ticket 04's token normalization" (spec.md decision 9): ``learn`` divides both
the predicted and target Q by ``number_clients * episode_length`` — a fixed,
per-Episode, config-derived order-of-magnitude for the total accumulated cost
(a sum of per-Client delay/earliness/overtime terms, each roughly bounded by
the episode's duration) — before computing the Huber loss, so its default
``delta=1.0`` sits in a sensible range instead of the raw cost's
three-to-four-digit scale. This does not change what ``decide``'s argmin
selects: dividing every candidate's Q by the same positive constant preserves
their relative order, whatever units the network's output nominally carries
before or after this scaling is introduced by training.

## Learning-time inefficiency, acknowledged

``learn``'s pseudocode (spec.md) does not describe caching the encoder pass
across the samples of one decision epoch during replay, unlike the acting
path's explicit "one encoder pass per decision epoch, not ``m``" requirement.
Shuffled minibatches routinely split one epoch's ``m`` vehicle-samples across
different minibatches (or different passes entirely), so this implementation
does not attempt that reuse: every training sample re-tokenizes and re-encodes
its snapshot from scratch. Correct, not maximally efficient — recorded as a
knob to measure later (mirroring spec.md decision 9's own treatment of the
replay buffer question), not the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

__all__ = ["TransformerMonteCarloPolicy"]

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

        # Ticket 13 discipline (ADR-0001 phase 2): one injected Generator per
        # stochastic concern, never a global. ``exploration_rng`` is
        # decide_train's epsilon gate and exploratory pick; ``learn_rng`` is
        # learn's minibatch shuffle.
        self.exploration_rng = exploration_rng
        self.learn_rng = learn_rng

        self._horizon_length = float(shift_end_minute - horizon_start_minute)
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

        pending = list(state.clients_not_visited)
        n_pending = len(pending)
        claimed_mask = np.zeros(n_pending, dtype=np.bool_)
        action = [self.depot] * self.number_vehicles

        for vehicle in range(self.number_vehicles):
            if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
                feasible = [index for index in range(n_pending) if not claimed_mask[index]]
                feasible.append(n_pending)  # the depot sentinel, always feasible
                chosen = int(rng.choice(feasible))
            else:
                q = self._score(
                    embeddings, vehicle, state.last_node_reached[vehicle], pending, claimed_mask
                )
                masked = q.clone()
                if n_pending:
                    masked[:n_pending][torch.from_numpy(claimed_mask)] = float("inf")
                chosen = int(torch.argmin(masked).item())

            if chosen == n_pending:
                action[vehicle] = self.depot
            else:
                action[vehicle] = pending[chosen]
                claimed_mask[chosen] = True

        return action

    def _score(
        self,
        embeddings: Embeddings,
        vehicle: int,
        vehicle_position: float,
        pending: list[int],
        claimed_mask: np.ndarray,
    ) -> torch.Tensor:
        """``Q(vehicle, candidate)`` over every pending Client plus the depot, in that order."""
        n_pending = len(pending)
        client_embeddings = embeddings.clients[:, vehicle, :]
        depot_row = self._depot_row(embeddings, vehicle, vehicle_position)
        augmented = torch.cat([client_embeddings, depot_row.unsqueeze(0)], dim=0)

        claimed = torch.zeros(n_pending + 1, dtype=torch.float32)
        if n_pending:
            claimed[:n_pending] = torch.from_numpy(claimed_mask.astype(np.float32))

        q: torch.Tensor = self.head(embeddings.vehicles[vehicle], augmented, claimed)
        return q

    def _depot_row(
        self, embeddings: Embeddings, vehicle: int, vehicle_position: float
    ) -> torch.Tensor:
        """The synthetic depot candidate's embedding — see this module's docstring."""
        minutes = self.geometry.average_minutes(vehicle_position, self.depot)
        length = self.geometry.length(vehicle_position, self.depot)
        pair = torch.tensor(
            [minutes / self._horizon_length, length / self._horizon_length], dtype=torch.float32
        )
        depot_arc = self.encoder.arc_embed(pair)
        return torch.cat([embeddings.vehicles[vehicle], depot_arc])

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
        samples = [(t, vehicle) for t in range(T) for vehicle in range(self.number_vehicles)]

        minibatch_losses: list[float] = []
        for _ in range(self.learn_passes):
            order = self.learn_rng.permutation(len(samples))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                self.optimizer.zero_grad()
                losses = []
                for index in batch:
                    t, vehicle = samples[int(index)]
                    q_pred = self._replay_q(snapshots[t], actions[t], vehicle)
                    target = torch.tensor(targets[t] / self._return_scale, dtype=torch.float32)
                    losses.append(functional.huber_loss(q_pred / self._return_scale, target))
                loss = torch.stack(losses).mean()
                loss.backward()
                self.optimizer.step()
                minibatch_losses.append(float(loss.item()))

        if minibatch_losses:
            self.last_loss = sum(minibatch_losses) / len(minibatch_losses)

    def _replay_q(
        self, snapshot: TrainingSnapshot, action_row: list[int], vehicle: int
    ) -> torch.Tensor:
        """``Q`` for the realized ``(vehicle, action_row[vehicle])`` pair, gradients attached.

        ``claimed`` replays the same index-order claiming :meth:`_sweep` uses at
        decide time: every earlier-indexed vehicle's realized target this epoch
        is marked claimed before scoring this vehicle's own realized pick.
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
        claimed_mask = np.zeros(n_pending, dtype=np.bool_)
        for other in range(vehicle):
            target = action_row[other]
            if target != self.depot and target in pending:
                claimed_mask[pending.index(target)] = True

        q = self._score(
            embeddings, vehicle, snapshot.last_node_reached[vehicle], pending, claimed_mask
        )
        chosen_action = action_row[vehicle]
        if chosen_action == self.depot:
            return q[n_pending]
        return q[pending.index(chosen_action)]

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
