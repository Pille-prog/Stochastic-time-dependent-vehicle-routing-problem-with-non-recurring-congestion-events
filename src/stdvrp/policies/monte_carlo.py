"""MonteCarloPolicy: linear function approximation over state and action features.

Phase-1 structural port (ADR-0001) of the legacy ``policy`` class, restricted to the
paths ``main()`` actually executes. Evaluation: ``monte_carlo_policy_test`` →
``select_epsilon_greedy_action_test`` → the single live ``select_vehicle_possible_actions``
definition (the per-vehicle one; every other definition in the monolith sits inside a
string literal) → ``generate_best_Q_pred_for_1_vehicle`` with the live feature
extractors ``extract_general_state_features`` / ``extract_state_action_features``.
Training (ticket 08): ``monte_carlo_policy_train`` → ``select_epsilon_greedy_action_train``
plus the Monte Carlo weight update ``actualize_W`` → ``update_W`` → ``learn`` (ticket 02,
the ``TrainablePolicy`` protocol's name for it; ``update_W`` lives on as an alias).

Ticket 13 (RNG modernization, ADR-0001 phase 2): the caller injects one
``exploration_rng: np.random.Generator`` — the policy's single stochastic
concern. Construction consumes one draw per vehicle for the initial action and
then runs one full greedy decision. Evaluation decisions themselves consume no
randomness. Training decisions draw from the same stream for all three legacy
draw sites: the exploration gate, the infeasible-carried-over-action repair, and
the exploratory action itself (the legacy split these across two private
UNSEEDED ``random.Random`` instances plus the global ``random`` stream — Phase 1,
ADR-0001; consolidated here since exact draw-order equality with the legacy is
no longer a goal). The weight update consumes no randomness.

Ticket 04 (simulation-performance, ADR-0003): every ``ShortestPathCache.path_between``
time/length read is replaced by an ``EpisodeGeometry`` array lookup — a pure
representation change, bit-identical to the dict lookups it replaces. The caller
builds one ``EpisodeGeometry`` per Episode (depot + that Episode's Clients as
columns) and injects it here; this Policy no longer touches the ShortestPathCache
directly. Path *node sequences* stay on the ShortestPathCache, read only by the
Model for routing.

Ticket 07 builds the first vectorized *reader* of those matrices:
``_closest_allowed_clients`` picks a vehicle's candidate actions with one row
slice and one ``np.lexsort`` instead of one lookup per remaining Client plus
``heapq.nsmallest`` — same ordering to the last tie. The endgame classifier
``_classify_shortest_distance_clients`` deliberately stays scalar: with one or
two Clients left its arrays are too short to pay numpy's per-call overhead
(measured 2x slower vectorized, ticket 07 Comments).

Ticket 13 (neural-policy, ADR-0011): ``_select_vehicle_possible_actions`` and
those two collaborators move to :mod:`~stdvrp.policies.action_set` — a
stateless module both this Policy and the transformer one (ticket 14) call, so
"identical candidate set" means one definition rather than two that can drift.
This method stays, as a thin delegate: the name is referenced by this
docstring's own change log, by ADR-0001's change log and by tickets 06/08/09.
No behaviour changed by the move — see ``action_set``'s module docstring for
the preserved quirks, which are unchanged from below.

Ticket 05 moves the feature arithmetic itself out to
:class:`~stdvrp.policies.feature_extraction.FeatureExtractor` (a concrete
collaborator, not a new seam — ADR-0002) and vectorizes it. What is left here is
the *decision* logic: which actions a vehicle may take, and which of them
minimizes Q. Both feature routines now flow through arguments and return values —
one :class:`~stdvrp.policies.feature_extraction.StateFeatures` per decision pass,
handed to every method that needs it — instead of the ``X_general_state`` /
``X_state_action`` / ``possible_actions`` / ``delayed_clients`` attributes the
port inherited from the legacy. A vehicle's whole candidate set is priced in one
``[candidates, 19]`` matrix and a single ``X @ W`` (:meth:`_best_q_action`).

Feature normalization constants (150, 850, 1150, 13, 60, 100, 180, 2500, the
earliness bins) are part of the feature definition and stay literal; only the values
the legacy read from argv or hardcoded as *experiment* knobs (horizon end, action
pool size, epsilon) are injected.

Preserved legacy quirks (do not fix before Phase 2; ADR-0001) — the feature-side
ones now live in ``feature_extraction`` and the candidate-set ones now live in
``action_set``, and are documented there:

- ``clients_left`` normalizes by a hardcoded 150 regardless of the episode's actual
  client count; ``late_count`` divides by 13.
- The candidate action set is deduplicated via ``list(set(...))`` — CPython set
  iteration order for these int node ids is deterministic and preserved in-process.
- Depot-idle cutoffs are inconsistent literals that ignore the configured horizon:
  ``tau > 350`` in ``select_vehicle_possible_actions`` vs ``tau > 310`` in the
  delayed/shortest-distance classifiers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.policies.action_set import select_vehicle_possible_actions
from stdvrp.policies.base import Policy
from stdvrp.policies.feature_extraction import (
    FeatureExtractor,
    StateFeatures,
    TimeWindows,
)
from stdvrp.simulation.state import is_parked_at_depot

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = ["MonteCarloPolicy", "TimeWindows"]


class MonteCarloPolicy(Policy):
    """Greedy argmin over Q predicted by a linear model ``W`` (evaluation mode)."""

    def __init__(
        self,
        number_vehicles: int,
        geometry: EpisodeGeometry,
        time_windows: TimeWindows,
        state: State,
        number_clients: int,
        epsilon: float,
        depot: int,
        number_actions_test: int,
        shift_end_minute: int,
        episode_end_minute: int,
        W: NDArray[np.float64] | None,
        *,
        exploration_rng: np.random.Generator,
        number_actions_train: int | None = None,
        learning_rate: float = 0.0,
    ) -> None:
        self.number_vehicles = number_vehicles
        self.geometry = geometry
        self.time_windows = time_windows
        self.state = state
        self.number_clients = number_clients
        self.epsilon = epsilon
        self.depot = depot
        self.number_actions_test = number_actions_test
        self.number_actions_train = number_actions_train
        self.learning_rate = learning_rate
        self.W = W

        # Ticket 13 (ADR-0001 phase 2): the policy's single stochastic concern —
        # the caller injects one per-Episode Generator, replacing the legacy's
        # two private UNSEEDED RNGs plus the global ``random`` stream.
        self.rng = exploration_rng

        # Cost factors as the legacy hardcodes them inside the policy.
        self.delay_cost_factor = 1
        self.earliness_cost_factor = 0.1
        self.overtime_cost = 5 / 6
        self.service_time = 5
        self.end_of_horizon = shift_end_minute

        # Ticket 05: the feature arithmetic, vectorized over the same geometry.
        self.feature_extractor = FeatureExtractor(
            geometry,
            time_windows,
            number_vehicles=number_vehicles,
            number_clients=number_clients,
            depot=depot,
            shift_end_minute=shift_end_minute,
            episode_end_minute=episode_end_minute,
            service_time=self.service_time,
            delay_cost_factor=self.delay_cost_factor,
            earliness_cost_factor=self.earliness_cost_factor,
            overtime_cost_factor=self.overtime_cost,
        )

        # Legacy constructor behavior: a random initial action (one
        # exploration_rng choice per vehicle), then one full greedy decision pass.
        self.action = [
            int(self.rng.choice(self.state.clients_not_visited)) for _ in range(number_vehicles)
        ]
        self.decide(state)

    def decide(self, state: State) -> list[int]:
        """Ports ``monte_carlo_policy_test`` → ``select_epsilon_greedy_action_test``.

        Greedy per-vehicle argmin, no randomness. The State's contribution to the
        features is computed once for the whole pass: it cannot change while the
        vehicles are being decided, and the legacy recomputed it anyway (twice,
        in fact — ``extract_general_state_features`` re-ran the delayed-Client
        classification the caller had just run).
        """
        self.state = state
        features = self.feature_extractor.state_features(state)
        for vehicle in range(self.number_vehicles):
            candidates = self._select_vehicle_possible_actions(
                self.number_actions_test, vehicle, features
            )
            self.action[vehicle] = self._best_q_action(features, vehicle, candidates)
        return self.action

    def decide_train(self, state: State) -> list[int]:
        """Ports ``monte_carlo_policy_train`` → ``select_epsilon_greedy_action_train``.

        A repair pass over every carried-over action that is no longer feasible,
        then the ε-greedy decision itself.
        """
        if self.number_actions_train is None:
            raise ValueError("number_actions_train is required for training decisions")
        self.state = state
        number_of_actions = self.number_actions_train
        features = self.feature_extractor.state_features(state)

        for vehicle in range(self.number_vehicles):
            candidates = self._select_vehicle_possible_actions(number_of_actions, vehicle, features)
            if self.action[vehicle] not in candidates:
                self.action[vehicle] = int(self.rng.choice(candidates))

        for vehicle in range(self.number_vehicles):
            candidates = self._select_vehicle_possible_actions(number_of_actions, vehicle, features)
            if self.rng.random() < self.epsilon:
                self.action[vehicle] = int(self.rng.choice(candidates))
            else:
                self.action[vehicle] = self._best_q_action(features, vehicle, candidates)
        return self.action

    def learn(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> None:
        """Ports ``actualize_W``: backward Monte Carlo return, one SGD step per epoch.

        Replays each saved decision epoch newest-first, accumulating the observed
        return ``U_t`` and stepping W against the already-acquired cost baseline.
        Consumes no randomness. Each epoch's ``TrainingSnapshot`` (ticket 06 — a
        purpose-built immutable capture of only the State fields this replay path
        reads) flows through as an argument; ticket 05 removed the rebinding of
        ``self.state`` that used to smuggle it in. The legacy's dead diagnostics
        (``self.rewards``, ``self.Q_preds``, ``self.error``) are not ported —
        nothing live reads them and they do not touch W.

        Ticket 02: this is ``TrainablePolicy.learn``, the protocol-facing name.
        ``update_W`` — the name ADR-0001's change log and tickets 06/08/09 use —
        is kept below as a plain alias to the same function, not a wrapper.
        """
        T = len(actions)
        U_t: float = 0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            snapshot = snapshots[t]
            acquired_cost = self._already_acquired_cost(snapshot)
            X = self.feature_extractor.action_features(
                self.feature_extractor.state_features(snapshot), actions[t]
            )
            assert self.W is not None

            Q_pred = np.dot(X, self.W)
            gradient = lr * ((U_t - acquired_cost - Q_pred) * X)
            self.W = self.W + gradient

    update_W = learn

    def _already_acquired_cost(self, state: State | TrainingSnapshot) -> float:
        """Ports ``calculate_already_acquired_cost``: sunk delay and overtime at tau."""
        total_cost_acquired = 0.0
        for client in state.clients_not_visited:
            delay_tw = self.time_windows[client][1]
            if delay_tw < state.tau_episode:
                total_cost_acquired += (state.tau_episode - delay_tw) * self.delay_cost_factor
        for vehicle in range(self.number_vehicles):
            at_depot = is_parked_at_depot(
                state.last_node_reached[vehicle], state.vehicle_standing[vehicle], self.depot
            )
            if not at_depot and state.tau_episode > self.end_of_horizon:
                total_cost_acquired += (
                    state.tau_episode - self.end_of_horizon
                ) * self.overtime_cost
        return total_cost_acquired

    def _best_q_action(self, features: StateFeatures, vehicle: int, candidates: list[int]) -> int:
        """Ports ``generate_best_Q_pred_for_1_vehicle``: strict argmin, ties keep first.

        Ticket 05: the whole candidate set is priced in one shot — a
        ``[candidates, 19]`` feature matrix and a single ``X @ W`` replace the
        legacy's per-candidate feature pass and dot product. ``np.argmin``
        returns the *first* minimum, which is exactly where the legacy's strict
        ``<`` against a running best left the winner.

        ``candidates`` is never empty: every branch of
        :meth:`_select_vehicle_possible_actions` falls back to the depot.
        """
        X = self.feature_extractor.candidate_features(features, self.action, vehicle, candidates)
        if self.W is None:
            self._create_W(X.shape[1])
        assert self.W is not None

        return candidates[int(np.argmin(X @ self.W))]

    def _create_W(self, number_features: int) -> None:
        """Ports ``create_W``: the weight vector starts at zero."""
        self.W = np.zeros(number_features)

    def _select_vehicle_possible_actions(
        self, number_of_actions: int, vehicle: int, features: StateFeatures
    ) -> list[int]:
        """Ports the live ``select_vehicle_possible_actions`` (per-vehicle) definition.

        Ticket 13 (neural-policy): delegates to the shared, stateless
        :func:`~stdvrp.policies.action_set.select_vehicle_possible_actions` —
        see that module's docstring for the full definition and its preserved
        quirks. This method stays only because its name is load-bearing
        elsewhere (this file's own module docstring, ADR-0001, tickets 06/08/09).
        """
        return select_vehicle_possible_actions(
            number_of_actions,
            vehicle,
            features,
            self.state,
            self.action,
            self.geometry,
            self.depot,
            self.number_vehicles,
            self.end_of_horizon,
        )
