"""MonteCarloPolicy: linear function approximation over state and action features.

Phase-1 structural port (ADR-0001) of the legacy ``policy`` class, restricted to the
paths ``main()`` actually executes. Evaluation: ``monte_carlo_policy_test`` →
``select_epsilon_greedy_action_test`` → the single live ``select_vehicle_possible_actions``
definition (the per-vehicle one; every other definition in the monolith sits inside a
string literal) → ``generate_best_Q_pred_for_1_vehicle`` with the live feature
extractors ``extract_general_state_features`` / ``extract_state_action_features``.
Training (ticket 08): ``monte_carlo_policy_train`` → ``select_epsilon_greedy_action_train``
plus the Monte Carlo weight update ``actualize_W`` → ``update_W``.

Global-RNG order is behavior (ADR-0001): construction consumes one ``random.choice``
per vehicle for the initial action and then runs one full greedy decision, exactly
like the legacy constructor. Evaluation decisions themselves consume no randomness.
Training decisions draw from three streams exactly like the legacy: ``local_rng``
gates exploration and ``local_rng_2`` repairs infeasible carried-over actions (both
constructed UNSEEDED, so training is nondeterministic unless the caller seeds them —
the golden-master capture seeds them per Episode as offset + train seed), while the
exploratory action itself is drawn from the **global** ``random`` stream, interleaved
with the transition function's velocity draws. The weight update consumes no
randomness.

Feature normalization constants (150, 850, 1150, 13, 60, 100, 180, 2500, the
earliness bins) are part of the feature definition and stay literal; only the values
the legacy read from argv or hardcoded as *experiment* knobs (horizon end, action
pool size, epsilon) are injected.

Preserved legacy quirks (do not fix before Phase 2; ADR-0001):

- ``_classify_delayed_clients`` appends a (time, client) pair on *every* vehicle
  iteration after the first assignment, not once per client — closest-vehicle lists
  contain duplicates with evolving travel times.
- ``clients_left`` normalizes by a hardcoded 150 regardless of the episode's actual
  client count; ``late_count`` divides by 13.
- The candidate action set is deduplicated via ``list(set(...))`` — CPython set
  iteration order for these int node ids is deterministic and preserved in-process.
- Depot-idle cutoffs are inconsistent literals that ignore the configured horizon:
  ``tau > 350`` in ``_select_vehicle_possible_actions`` vs ``tau > 310`` in the
  delayed/shortest-distance classifiers.
- ``_extract_state_action_features`` emits a permanently-zero second feature; it
  pads W to its legacy 19 components.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from stdvrp.network.shortest_path_cache import ShortestPathCache
from stdvrp.policies.base import Policy
from stdvrp.simulation.state import State

TimeWindows = dict[int, tuple[int, int]]


class MonteCarloPolicy(Policy):
    """Greedy argmin over Q predicted by a linear model ``W`` (evaluation mode)."""

    def __init__(
        self,
        number_vehicles: int,
        shortest_path_cache: ShortestPathCache,
        time_windows: TimeWindows,
        state: State,
        number_clients: int,
        epsilon: float,
        depot: int,
        number_actions_test: int,
        horizon_end_minute: int,
        W: NDArray[np.float64] | None,
        *,
        number_actions_train: int | None = None,
        learning_rate: float = 0.0,
    ) -> None:
        self.number_vehicles = number_vehicles
        self.shortest_path_cache = shortest_path_cache
        self.time_windows = time_windows
        self.state = state
        self.number_clients = number_clients
        self.epsilon = epsilon
        self.depot = depot
        self.number_actions_test = number_actions_test
        self.number_actions_train = number_actions_train
        self.learning_rate = learning_rate
        self.W = W

        # Legacy quirk (ticket 04 finding 1): two UNSEEDED private RNGs drive the
        # training exploration gate (``local_rng``) and the infeasible-action
        # repair (``local_rng_2``). A caller wanting reproducible training must
        # seed them right after construction, exactly like the capture driver.
        self.local_rng = random.Random()
        self.local_rng_2 = random.Random()

        # Cost factors as the legacy hardcodes them inside the policy.
        self.delay_cost_factor = 1
        self.earliness_cost_factor = 0.1
        self.overtime_cost = 5 / 6
        self.service_time = 5
        self.end_of_horizon = horizon_end_minute

        self.number_of_actions = number_actions_test
        self.possible_actions: list[int] = []
        self.X_general_state: list[float] = []
        self.X_state_action: list[float] = []
        self.delayed_clients: list[list[int]] = []
        self.vehicle_to_clients: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
        self.shortest_distance_clients: defaultdict[int, list[tuple[float, int]]] = defaultdict(
            list
        )
        self.mean_velocities: list[float] = []

        # Legacy constructor behavior: a random initial action (one global-RNG
        # choice per vehicle), then one full greedy decision pass.
        self.action = [
            random.choice(self.state.clients_not_visited) for _ in range(number_vehicles)
        ]
        self.decide(state)

    def decide(self, state: State) -> list[int]:
        """Ports ``monte_carlo_policy_test``: greedy per-vehicle argmin, no randomness."""
        self.state = state
        self.number_of_actions = self.number_actions_test
        self._select_greedy_actions()
        return self.action

    def _select_greedy_actions(self) -> None:
        """Ports ``select_epsilon_greedy_action_test``."""
        self._classify_delayed_clients()
        self._extract_general_state_features()
        for vehicle in range(self.number_vehicles):
            self.possible_actions = list(
                self._select_vehicle_possible_actions(self.number_of_actions, vehicle)
            )
            self._select_best_q_action_for_vehicle(vehicle)

    def decide_train(self, state: State) -> list[int]:
        """Ports ``monte_carlo_policy_train``: ε-greedy per-vehicle decision."""
        if self.number_actions_train is None:
            raise ValueError("number_actions_train is required for training decisions")
        self.state = state
        self.number_of_actions = self.number_actions_train
        self._select_epsilon_greedy_actions_train()
        return self.action

    def _select_epsilon_greedy_actions_train(self) -> None:
        """Ports ``select_epsilon_greedy_action_train``: repair pass, then ε-greedy."""
        self._classify_delayed_clients()
        self._extract_general_state_features()

        # Repair pass: replace any carried-over action no longer feasible.
        for vehicle in range(self.number_vehicles):
            self.possible_actions = self._select_vehicle_possible_actions(
                self.number_of_actions, vehicle
            )
            if self.action[vehicle] not in self.possible_actions:
                self.action[vehicle] = self.local_rng_2.choice(self.possible_actions)

        for vehicle in range(self.number_vehicles):
            self.possible_actions = self._select_vehicle_possible_actions(
                self.number_of_actions, vehicle
            )
            if self.local_rng.random() < self.epsilon:
                # Exploration draws from the GLOBAL stream, interleaved with the
                # Model's velocity draws — order is behavior (ADR-0001).
                self.action[vehicle] = random.choice(self.possible_actions)
            else:
                self._select_best_q_action_for_vehicle(vehicle)

    def update_W(self, states: list[State], actions: list[list[int]], rewards: list[float]) -> None:
        """Ports ``actualize_W``: backward Monte Carlo return, one SGD step per epoch.

        Replays each saved decision epoch newest-first, accumulating the observed
        return ``U_t`` and stepping W against the already-acquired cost baseline.
        Consumes no randomness; rebinds ``self.state`` to each historical snapshot.
        The legacy's dead diagnostics (``self.rewards``, ``self.Q_preds``,
        ``self.error``) are not ported — nothing live reads them and they do not
        touch W.
        """
        T = len(actions)
        U_t: float = 0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            self.state = states[t]
            self._calculate_already_acquired_cost()
            self._extract_general_state_features()
            self._extract_state_action_features(actions[t])
            X = np.array(list(itertools.chain(self.X_general_state, self.X_state_action)))
            assert self.W is not None

            Q_pred = np.dot(X, self.W)
            gradient = lr * ((U_t - self.total_cost_acquired - Q_pred) * X)
            self.W = self.W + gradient

    def _calculate_already_acquired_cost(self) -> None:
        """Ports ``calculate_already_acquired_cost``: sunk delay and overtime at tau."""
        self.total_cost_acquired = 0.0
        for client in self.state.clients_not_visited:
            delay_tw = self.time_windows[client][1]
            if delay_tw < self.state.tau_episode:
                self.total_cost_acquired += (
                    self.state.tau_episode - delay_tw
                ) * self.delay_cost_factor
        for vehicle in range(self.number_vehicles):
            if (
                self.state.vehicle_position[vehicle] != self.depot
                and self.state.tau_episode > self.end_of_horizon
            ):
                self.total_cost_acquired += (
                    self.state.tau_episode - self.end_of_horizon
                ) * self.overtime_cost

    def _select_best_q_action_for_vehicle(self, vehicle: int) -> None:
        """Ports ``generate_best_Q_pred_for_1_vehicle``: strict argmin, ties keep first."""
        min_q_value = float("inf")
        best_client = 0
        for client in self.possible_actions:
            current_action = self.action.copy()
            current_action[vehicle] = client
            self._extract_state_action_features(current_action)
            X = list(itertools.chain(self.X_general_state, self.X_state_action))
            if self.W is None:
                self._create_W(len(X))
            assert self.W is not None

            q_value = np.dot(X, self.W)
            if q_value < min_q_value:
                min_q_value = float(q_value)
                best_client = client

        self.action[vehicle] = best_client

    def _create_W(self, number_features: int) -> None:
        """Ports ``create_W``: the weight vector starts at zero."""
        self.W = np.zeros(number_features)

    def _select_vehicle_possible_actions(self, number_of_actions: int, vehicle: int) -> list[int]:
        """Ports the live ``select_vehicle_possible_actions`` (per-vehicle) definition."""
        possible_actions: list[int] = []
        forbidden_actions = []

        for v in range(self.number_vehicles):
            if v == vehicle:
                continue
            else:
                forbidden_actions.append(self.action[v])

        if (
            self.state.vehicle_position[vehicle] == self.depot and self.state.tau_episode > 350
        ) or len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.depot)

        elif len(self.state.clients_not_visited) < 3:
            self._classify_shortest_distance_clients()
            if self.shortest_distance_clients[vehicle]:
                for i in range(len(self.shortest_distance_clients[vehicle])):
                    possible_actions.append(self.shortest_distance_clients[vehicle][i][1])
            else:
                possible_actions.append(self.depot)

        else:
            clients = self.state.clients_not_visited
            travel_times = [
                (
                    self.shortest_path_cache.path_between(
                        self.state.vehicle_position[vehicle], client
                    ).average_minutes,
                    client,
                )
                for client in clients
            ]
            top_vehicle_actions = [
                vehicle_action
                for vehicle_action in travel_times
                if vehicle_action[1] not in forbidden_actions
            ]

            possible_actions = [
                client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)
            ]

            possible_actions = list(set(possible_actions))

            if (
                self.shortest_path_cache.path_between(
                    self.state.vehicle_position[vehicle], self.depot
                ).average_minutes
                + self.state.tau_episode
                > self.end_of_horizon
            ):
                possible_actions.append(self.depot)

            for delayed_client in self.delayed_clients[vehicle]:
                if (
                    delayed_client not in possible_actions
                    and delayed_client not in forbidden_actions
                ):
                    possible_actions.append(delayed_client)

            if len(possible_actions) == 0:
                possible_actions.append(self.depot)

        return possible_actions

    def _classify_delayed_clients(self) -> None:
        """Ports ``clasify_delayed_clients`` — duplicate-append quirk included."""
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for client in self.state.clients_not_visited:
            assigned_vehicle = None
            min_travel_time = float("inf")
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.depot and self.state.tau_episode > 310:
                    continue

                travel_time = self.shortest_path_cache.path_between(
                    vehicle_position, client
                ).average_minutes
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx

                # Legacy quirk: inside the vehicle loop, so the pair is appended
                # once per remaining vehicle iteration, not once per client.
                if assigned_vehicle is not None:
                    self.vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))

        for vehicle_idx, client_list in self.vehicle_to_clients.items():
            client_list.sort()
            for travel_time, client in client_list:
                if len(self.delayed_clients[vehicle_idx]) >= 2:
                    break

                delay_tw = self.time_windows[client][1]
                if travel_time + self.state.tau_episode >= delay_tw:
                    self.delayed_clients[vehicle_idx].append(client)

    def _classify_shortest_distance_clients(self) -> None:
        """Ports ``clasify_shortest_distance_clients`` (endgame with < 3 Clients left)."""
        self.shortest_distance_clients = defaultdict(list)

        clients_remaining = len(self.state.clients_not_visited)

        if clients_remaining == 2:
            vehicle_distances = []
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.depot and self.state.tau_episode > 310:
                    continue

                total_distance = sum(
                    self.shortest_path_cache.path_between(vehicle_position, client).average_minutes
                    for client in self.state.clients_not_visited
                )
                vehicle_distances.append((total_distance, vehicle_idx))

            closest_two_vehicles = heapq.nsmallest(2, vehicle_distances)

            for _, vehicle_idx in closest_two_vehicles:
                for client in self.state.clients_not_visited:
                    travel_time = self.shortest_path_cache.path_between(
                        self.state.vehicle_position[vehicle_idx], client
                    ).average_minutes
                    self.shortest_distance_clients[vehicle_idx].append((travel_time, client))

        elif clients_remaining == 1:
            client = next(iter(self.state.clients_not_visited))
            distances = []
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.depot and self.state.tau_episode > 310:
                    continue

                travel_time = self.shortest_path_cache.path_between(
                    vehicle_position, client
                ).average_minutes
                distances.append((travel_time, vehicle_idx))

            closest_vehicle = min(distances)
            assigned_vehicle_idx = closest_vehicle[1]
            self.shortest_distance_clients[assigned_vehicle_idx].append(
                (closest_vehicle[0], client)
            )

    def _extract_general_state_features(self) -> None:
        """Ports the live ``extract_general_state_features`` (12 features)."""
        self.X_general_state = []

        clients_left = len(self.state.clients_not_visited) / 150

        if clients_left != 0:
            time_left = (1150 - self.state.tau_episode) / (850)
            time = (self.state.tau_episode - 300) / 850
        else:
            time_left = 0
            time = 0

        self.X_general_state.append(np.sqrt(clients_left))
        self.X_general_state.append(time_left)
        self.X_general_state.append(time_left**2)
        self.X_general_state.append(clients_left**2)
        self.X_general_state.append((clients_left**2) * time)
        self.X_general_state.append((time**2) * clients_left)
        self.X_general_state.append((time**2) * (clients_left**2))

        client_earliness_value = []
        client_delay_value = []
        for client in self.state.clients_not_visited:
            client_earliness, client_due_time = self.time_windows[client]
            client_earliness_value.append(client_earliness)
            client_delay_value.append(client_due_time)

        client_counts_earliness = [0 for _ in range(4)]

        for i in client_earliness_value:
            if i < 400 and self.state.tau_episode < 400:
                client_counts_earliness[0] += 1
            elif 400 <= i < 500 and self.state.tau_episode < 500:
                client_counts_earliness[1] += 1
            elif 500 <= i < 600 and self.state.tau_episode < 600:
                client_counts_earliness[2] += 1

        for count in client_counts_earliness:
            self.X_general_state.append(count / self.number_clients)

        time_left_for_earliness = (580 - self.state.tau_episode) / (280)
        mean_earliness_diff: float = 0
        if time_left_for_earliness > 0:
            mean_earliness: float = 0
            for i in client_earliness_value:
                mean_earliness += i

            if len(self.state.clients_not_visited) != 0:
                mean_earliness = mean_earliness / len(self.state.clients_not_visited)
                if mean_earliness > self.state.tau_episode:
                    mean_earliness_diff = (mean_earliness - self.state.tau_episode) / 120

        self.X_general_state.append(mean_earliness_diff)

        # Computed and stored but never appended as a feature — kept as in the legacy.
        self.mean_velocities = []
        for vehicle_velocities in self.state.observed_velocity:
            mean_velocity: float = 0
            for velocity in vehicle_velocities:
                mean_velocity += velocity
            mean_velocity = mean_velocity / len(vehicle_velocities)
            self.mean_velocities.append(mean_velocity)

        self._classify_delayed_clients()

    def _extract_state_action_features(self, action: list[int]) -> None:
        """Ports the live ``extract_state_action_features`` (7 features)."""
        cg_clients = self.time_windows
        paths = self.shortest_path_cache
        state = self.state
        tau = state.tau_episode
        clients_all = state.clients_not_visited

        depot = self.depot
        n_veh = self.number_vehicles
        service_time = self.service_time
        end_horizon = self.end_of_horizon
        earl_fact = self.earliness_cost_factor
        delay_fact = self.delay_cost_factor
        overtime_fact = self.overtime_cost

        selected = {a for a in action if a != depot}
        clients_left = [c for c in clients_all if c not in selected]

        features = []

        late_count = sum(1 for c in clients_left if tau > cg_clients[c][1])
        features.append(late_count / 13)
        # Preserved quirk: a permanently-zero feature. Removing it would shrink W
        # from 19 components and invalidate every stored weight vector.
        features.append(0)

        total_dist = sum(
            paths.path_between(state.vehicle_position[i], action[i]).length for i in range(n_veh)
        )
        features.append(total_dist / 100.0)

        earliness_cost = 0.0
        delay_cost = 0.0
        for i, a in enumerate(action):
            if a in clients_all and a != depot:
                travel_time = paths.path_between(state.vehicle_position[i], a).average_minutes
                est_arrival = tau + travel_time
                earl_tw, due_tw = cg_clients[a]

                if est_arrival < earl_tw:
                    earliness_cost += (earl_tw - est_arrival) * earl_fact
                elif est_arrival > due_tw:
                    delay_cost += (est_arrival - max(due_tw, tau)) * delay_fact

        features.append(earliness_cost / 60.0)
        features.append(delay_cost / 60.0)

        future_delay = 0.0
        for veh in range(n_veh):
            for _, client in self.vehicle_to_clients[veh]:
                if client not in action:
                    t1 = paths.path_between(
                        state.vehicle_position[veh], action[veh]
                    ).average_minutes
                    t2 = paths.path_between(action[veh], client).average_minutes
                    est = tau + t1 + t2 + service_time
                    _, due_tw = cg_clients[client]
                    if est > due_tw:
                        future_delay += (est - max(due_tw, tau)) * delay_fact

        features.append(future_delay / 2500.0)

        overtime_cost = 0.0
        for i, a in enumerate(action):
            if a != depot:
                t1 = paths.path_between(state.vehicle_position[i], a).average_minutes
                t2 = paths.path_between(a, depot).average_minutes
                est_ret = tau + t1 + t2 + service_time
            else:
                est_ret = tau + paths.path_between(state.vehicle_position[i], depot).average_minutes

            if est_ret > end_horizon:
                base = end_horizon if tau < end_horizon else tau
                overtime_cost += (est_ret - base) * overtime_fact

        features.append(overtime_cost / 180.0)

        self.X_state_action = features
