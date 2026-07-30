"""MonteCarloPolicy: linear function approximation over state and action features.

Phase-1 structural port (ADR-0001) of the legacy ``policy`` class, restricted to the
paths ``main()`` actually executes. Evaluation: ``monte_carlo_policy_test`` →
``select_epsilon_greedy_action_test`` → the single live ``select_vehicle_possible_actions``
definition (the per-vehicle one; every other definition in the monolith sits inside a
string literal) → ``generate_best_Q_pred_for_1_vehicle`` with the live feature
extractors ``extract_general_state_features`` / ``extract_state_action_features``.
Training (ticket 08): ``monte_carlo_policy_train`` → ``select_epsilon_greedy_action_train``
plus the Monte Carlo weight update ``actualize_W`` → ``update_W``.

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
``heapq.nsmallest`` — same ordering to the last tie (see its docstring). The
endgame classifier ``_classify_shortest_distance_clients`` deliberately stays
scalar: with one or two Clients left its arrays are too short to pay numpy's
per-call overhead (measured 2x slower vectorized, ticket 07 Comments).

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
ones now live in ``feature_extraction`` and are documented there:

- ``clients_left`` normalizes by a hardcoded 150 regardless of the episode's actual
  client count; ``late_count`` divides by 13.
- The candidate action set is deduplicated via ``list(set(...))`` — CPython set
  iteration order for these int node ids is deterministic and preserved in-process.
- Depot-idle cutoffs are inconsistent literals that ignore the configured horizon:
  ``tau > 350`` in ``_select_vehicle_possible_actions`` vs ``tau > 310`` in the
  delayed/shortest-distance classifiers.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.policies.base import Policy
from stdvrp.policies.feature_extraction import (
    FeatureExtractor,
    StateFeatures,
    TimeWindows,
)

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = ["MonteCarloPolicy", "TimeWindows"]


class _RemainingClients(NamedTuple):
    """``clients_not_visited`` as the arrays candidate selection sorts over.

    All three are aligned index-for-index: ``clients`` is a *copy* of the State's
    list (which the Model mutates in place, and whose Python ints the selected
    actions are taken from, unchanged), ``client_ids`` is that list as an array
    for the tie-break sort key, and ``column_positions`` locates each Client in
    the EpisodeGeometry's columns.
    """

    clients: list[int]
    client_ids: NDArray[np.integer]
    column_positions: NDArray[np.intp]


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
        horizon_end_minute: int,
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
        self.end_of_horizon = horizon_end_minute

        # Ticket 05: the feature arithmetic, vectorized over the same geometry.
        self.feature_extractor = FeatureExtractor(
            geometry,
            time_windows,
            number_vehicles=number_vehicles,
            number_clients=number_clients,
            depot=depot,
            horizon_end_minute=horizon_end_minute,
            service_time=self.service_time,
            delay_cost_factor=self.delay_cost_factor,
            earliness_cost_factor=self.earliness_cost_factor,
            overtime_cost_factor=self.overtime_cost,
        )
        self._remaining_clients_cache: _RemainingClients | None = None

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

    def update_W(
        self, states: list[TrainingSnapshot], actions: list[list[int]], rewards: list[float]
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
        """
        T = len(actions)
        U_t: float = 0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            snapshot = states[t]
            acquired_cost = self._already_acquired_cost(snapshot)
            X = self.feature_extractor.action_features(
                self.feature_extractor.state_features(snapshot), actions[t]
            )
            assert self.W is not None

            Q_pred = np.dot(X, self.W)
            gradient = lr * ((U_t - acquired_cost - Q_pred) * X)
            self.W = self.W + gradient

    def _already_acquired_cost(self, state: State | TrainingSnapshot) -> float:
        """Ports ``calculate_already_acquired_cost``: sunk delay and overtime at tau."""
        total_cost_acquired = 0.0
        for client in state.clients_not_visited:
            delay_tw = self.time_windows[client][1]
            if delay_tw < state.tau_episode:
                total_cost_acquired += (state.tau_episode - delay_tw) * self.delay_cost_factor
        for vehicle in range(self.number_vehicles):
            if (
                state.vehicle_position[vehicle] != self.depot
                and state.tau_episode > self.end_of_horizon
            ):
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
        """Ports the live ``select_vehicle_possible_actions`` (per-vehicle) definition."""
        possible_actions: list[int] = []
        forbidden_actions = []

        for v in range(self.number_vehicles):
            if v == vehicle:
                continue
            else:
                forbidden_actions.append(self.action[v])

        # This 350 and the 310 in _classify_shortest_distance_clients below disagree
        # by 40 minutes (module docstring, "Depot-idle cutoffs"); ticket 05 leaves
        # both literals as-is (spec.md decision 3: fix what crashes or
        # misclassifies, never re-tune what is tuned) and only added the fallback
        # in the 310 branch for the window this gap otherwise leaves uncovered.
        if (
            self.state.vehicle_position[vehicle] == self.depot and self.state.tau_episode > 350
        ) or len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.depot)

        elif len(self.state.clients_not_visited) < 3:
            # B11: filter forbidden_actions here too, matching the normal branch
            # below — otherwise two vehicles can both be offered (and both pick)
            # the same Client this classifier assigns to more than one of them.
            shortest_distance_clients = self._classify_shortest_distance_clients()
            for _distance, client in shortest_distance_clients[vehicle]:
                if client not in forbidden_actions:
                    possible_actions.append(client)
            if not possible_actions:
                possible_actions.append(self.depot)

        else:
            possible_actions = self._closest_allowed_clients(
                self.state.vehicle_position[vehicle], number_of_actions, forbidden_actions
            )

            possible_actions = list(set(possible_actions))

            if (
                self.geometry.average_minutes(self.state.vehicle_position[vehicle], self.depot)
                + self.state.tau_episode
                > self.end_of_horizon
            ):
                possible_actions.append(self.depot)

            for delayed_client in features.delayed_clients[vehicle]:
                if (
                    delayed_client not in possible_actions
                    and delayed_client not in forbidden_actions
                ):
                    possible_actions.append(delayed_client)

            if len(possible_actions) == 0:
                possible_actions.append(self.depot)

        return possible_actions

    def _closest_allowed_clients(
        self, position: float, number_of_actions: int, forbidden_actions: list[int]
    ) -> list[int]:
        """The unvisited Clients closest to ``position``, nearest first, at most k of them.

        Ticket 07 (simulation-performance, ADR-0003): one geometry row slice plus
        one ``np.lexsort`` replace the per-Client ``average_minutes`` lookups and
        the ``heapq.nsmallest`` over ``(travel time, Client)`` tuples this ports.
        The ordering is identical, not merely equivalent: lexsort's primary key
        is the travel time and its secondary the Client id, exactly how tuple
        comparison broke float ties, and ``nsmallest(k, xs) == sorted(xs)[:k]``.

        Forbidden Clients (the other vehicles' current actions) are dropped
        *after* the sort: the ``k + len(forbidden)`` nearest contain at least
        ``k`` allowed ones whenever the filtered set has that many, so the first
        ``k`` survivors are the same list filtering first would give.

        The returned ids are the State's own Python ints, never numpy scalars:
        they flow into ``self.action`` and from there into the Model.
        """
        remaining = self._remaining_clients()
        travel_times = self.geometry.average_minutes_at(position, remaining.column_positions)
        nearest = np.lexsort((remaining.client_ids, travel_times))[
            : number_of_actions + len(forbidden_actions)
        ]

        clients = remaining.clients
        closest: list[int] = []
        for index in nearest:
            if len(closest) == number_of_actions:
                break
            client = clients[index]
            if client not in forbidden_actions:
                closest.append(client)
        return closest

    def _remaining_clients(self) -> _RemainingClients:
        """``clients_not_visited`` as sortable arrays, rebuilt only when it changes.

        Every vehicle of a decision pass selects candidates out of the same
        unvisited-Client list, which only the Model's transition function changes
        (in place — hence the copy in :class:`_RemainingClients`, and the
        content comparison here rather than an identity check).
        """
        clients = self.state.clients_not_visited
        cached = self._remaining_clients_cache
        if cached is not None and cached.clients == clients:
            return cached

        fresh = _RemainingClients(
            clients=list(clients),
            client_ids=np.asarray(clients),
            column_positions=self.geometry.column_positions(clients),
        )
        self._remaining_clients_cache = fresh
        return fresh

    def _classify_shortest_distance_clients(self) -> defaultdict[int, list[tuple[float, int]]]:
        """Ports ``clasify_shortest_distance_clients`` (endgame with < 3 Clients left).

        Stays scalar on purpose (ticket 07 Comments): with one or two Clients left
        its arrays are two elements wide, where numpy's per-call overhead measured
        2x slower than these loops.
        """
        shortest_distance_clients: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)

        clients_remaining = len(self.state.clients_not_visited)

        if clients_remaining == 2:
            # 310 here, 350 in _select_vehicle_possible_actions above — the
            # disagreement documented there. heapq.nsmallest(2, []) == [] below
            # if this filters out every vehicle, so this branch never hits B5's
            # empty-min() crash (that's the one-Client branch just below).
            vehicle_distances = []
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.depot and self.state.tau_episode > 310:
                    continue

                total_distance = sum(
                    self.geometry.average_minutes(vehicle_position, client)
                    for client in self.state.clients_not_visited
                )
                vehicle_distances.append((total_distance, vehicle_idx))

            closest_two_vehicles = heapq.nsmallest(2, vehicle_distances)

            for _, vehicle_idx in closest_two_vehicles:
                for client in self.state.clients_not_visited:
                    travel_time = self.geometry.average_minutes(
                        self.state.vehicle_position[vehicle_idx], client
                    )
                    shortest_distance_clients[vehicle_idx].append((travel_time, client))

        elif clients_remaining == 1:
            client = next(iter(self.state.clients_not_visited))
            distances = []
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.depot and self.state.tau_episode > 310:
                    continue

                travel_time = self.geometry.average_minutes(vehicle_position, client)
                distances.append((travel_time, vehicle_idx))

            # B5: every vehicle can read position == depot with tau in (310, 350]
            # (see the disagreement noted in _select_vehicle_possible_actions above),
            # leaving `distances` empty. Fall back to the depot, exactly as the
            # two-Client branch above already does via heapq.nsmallest(2, []) == [].
            if distances:
                closest_vehicle = min(distances)
                assigned_vehicle_idx = closest_vehicle[1]
                shortest_distance_clients[assigned_vehicle_idx].append((closest_vehicle[0], client))

        return shortest_distance_clients
