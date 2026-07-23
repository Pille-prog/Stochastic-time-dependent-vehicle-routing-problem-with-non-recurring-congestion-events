"""Model: the sequential decision model owning the transition function (Powell).

Phase-1 structural port (ADR-0001) of the legacy ``model`` class, restricted to the
path ``main()`` executes: ``transition_function`` with all its callees, the
evaluation Episode loop (``create_monte_carlo_episode_test``) and the training
Episode loop (``create_monte_carlo_episode_train``, ticket 08). Deliberately
concrete — no interface (ADR-0002); the CongestionGenerator seam is injected
instead of the legacy's hardcoded event method.

Stochastic velocities are sampled from the **global** ``random`` stream
(``random.gauss`` per arc-minute, memoized per Episode) and congestion events from
the global ``np.random`` stream via the CongestionGenerator; call order is behavior
(ADR-0001). The per-Episode mutable dicts the legacy kept on ``DataCalculations``
(``all_arc_velocity``, ``congested_arcs``) live on the Model, which is constructed
fresh per Episode — the legacy reset both at every episode boundary, so a fresh
Model starts from the identical state.

Preserved legacy quirks (ADR-0001):

- Hardcoded emergency-horizon constants 1150/1198 and the ``40000 - 200 * served``
  abort penalty (they ignore the configured horizon).
- Congestion is rolled only on epochs where ``(tau + 178) / 60`` is an exact float
  multiple of ``max_congestion_duration / 60`` hours.
- Congested-velocity samples are *not* memoized; normal samples are.

Phase-2 deliberate fixes (ticket 12, ADR-0001 change log):

- Both termination paths charge late Clients from the actual clock (``tau -
  due``); the legacy hardcoded ``1150 - due`` in the all-vehicles-back path.
- Training episodes keep their real ``total_distance_cost`` (the legacy zeroed
  the accumulator every step; reporting only).
"""

from __future__ import annotations

import copy
import math
import random

from stdvrp.congestion import CongestedArcs, CongestionGenerator
from stdvrp.network.shortest_path_cache import ShortestPathCache
from stdvrp.policies.base import Policy
from stdvrp.policies.monte_carlo import MonteCarloPolicy, TimeWindows
from stdvrp.simulation.state import State
from stdvrp.traffic.travel_time_model import TravelTimeModel

ArcMinuteKey = tuple[float, float, int]


class Model:
    """Advances the State given a decision and exogenous velocities and congestion."""

    def __init__(
        self,
        state: State,
        policy: Policy,
        travel_time_model: TravelTimeModel,
        shortest_path_cache: ShortestPathCache,
        time_windows: TimeWindows,
        number_vehicles: int,
        horizon_start_minute: int,
        horizon_end_minute: int,
        depot: int,
        congestion_generator: CongestionGenerator,
        max_congestion_duration: int,
    ) -> None:
        self.state = state
        self.policy = policy
        self.travel_time_model = travel_time_model
        self.shortest_path_cache = shortest_path_cache
        self.time_windows = time_windows
        self.number_vehicles = number_vehicles
        self.horizon_start_minute = horizon_start_minute
        self.horizon_end_minute = horizon_end_minute
        self.depot = depot
        self.congestion_generator = congestion_generator

        # Cost factors as the legacy hardcodes them inside the model.
        self.earliness_cost = 0.1
        self.distance_cost = 1
        self.delay_cost = 1
        self.overtime_cost = 5 / 6
        self.service_time = 5

        # Congestion epoch cadence (legacy ``hours_max_duration``).
        self.hours_max_duration = max_congestion_duration / 60

        # Decision epochs are two simulated minutes apart.
        self.tau_multiplicator_difference = 2
        self.tau_multiplicator: float = horizon_start_minute + self.tau_multiplicator_difference

        # Per-Episode stochastic state (legacy ``DataCalculations`` attributes).
        self.congested_arcs: CongestedArcs = {}
        self.sampled_arc_velocities: dict[ArcMinuteKey, list[float]] = {}

        self.action: list[int] = [0 for _ in range(number_vehicles)]
        self.vehicles_shortest_path: list[list[float]] = [[0, 0] for _ in range(number_vehicles)]
        self.node_time_arrival: list[float] = [horizon_start_minute for _ in range(number_vehicles)]
        # Per-vehicle departure time (legacy ``tau_salida``): the tau at which the
        # vehicle leaves its current node once service there is finished.
        self.departure_tau: list[float] = [horizon_start_minute for _ in range(number_vehicles)]
        self.tau_vehicle_horizon_change: list[float] = [
            horizon_start_minute for _ in range(number_vehicles)
        ]
        self.distance_arc_distance_travelled: list[float] = [0 for _ in range(number_vehicles)]
        self.visited_clients: list[float] = []

        self.end_transition_function = 1
        self.transition_cost: float = 0
        self.total_cost: float = 0
        self.total_distance_travelled: float = 0
        self.total_earliness_cost: float = 0
        self.total_delay_cost: float = 0
        self.total_distance_cost: float = 0
        self.total_overtime_cost: float = 0
        self.total_earliness_clients = 0
        self.total_delay_clients = 0
        self.total_overtime_vehicles = 0
        self.total_state_counter = 0

        # ``work_time`` in the legacy — always the horizon end.
        self.work_time = horizon_end_minute

    # --- Episode runner ---------------------------------------------------------

    def run_evaluation_episode(self) -> None:
        """Ports ``create_monte_carlo_episode_test``: greedy decisions until terminal."""
        self.congested_arcs = {}
        while not self.state.terminal:
            action = self.policy.decide(self.state)
            self.transition_function(action)
            self.total_state_counter += 1

        self.congested_arcs = {}
        self.sampled_arc_velocities = {}

    def run_training_episode(self) -> None:
        """Ports ``create_monte_carlo_episode_train``: ε-greedy Episode, then one W update.

        Snapshots the State and decision before every transition, replays them
        through ``MonteCarloPolicy.update_W`` when the Episode terminates.
        """
        policy = self.policy
        if not isinstance(policy, MonteCarloPolicy):
            raise TypeError("training episodes require a MonteCarloPolicy")

        self.episode_states: list[State] = []
        self.episode_actions: list[list[int]] = []
        self.episode_rewards: list[float] = [0]
        self.congested_arcs = {}
        while not self.state.terminal:
            action = policy.decide_train(self.state)
            # Snapshot BEFORE the transition — the weight update replays these epochs.
            self.episode_states.append(copy.deepcopy(self.state))
            self.episode_actions.append(copy.deepcopy(action))
            reward = self.transition_function(action)
            self.episode_rewards.append(reward)
            self.total_state_counter += 1

        policy.update_W(self.episode_states, self.episode_actions, self.episode_rewards)
        self.congested_arcs = {}
        self.sampled_arc_velocities = {}

    # --- Transition function and callees ----------------------------------------

    def transition_function(self, action: list[int]) -> float:
        """Advance simulated time until the next decision is needed; return its cost."""
        self.transition_cost = 0

        self.calculate_action_route(action)
        self.end_transition_function = 1

        while self.end_transition_function == 1:
            event_end, _ = self._next_congestion_end()
            min_travel_time_vehicle, min_travel_time = min(
                enumerate(self.node_time_arrival), key=lambda x: x[1]
            )

            t_next = min(self.tau_multiplicator, min_travel_time, event_end)

            # A congestion expires before anything else: re-sample velocities there.
            if t_next == event_end:
                self.vehicle_distance_transition_cost(t_next)
                for vehicle in range(self.number_vehicles):
                    self.time_horizon_actualization(vehicle)
                continue

            if (
                all(position == self.depot for position in self.state.vehicle_position)
                and len(self.state.clients_not_visited) == 0
            ):
                self.state.terminal = True
                self.end_transition_function = 2
                self.vehicle_distance_transition_cost(self.state.tau_episode)
                self.total_cost += self.transition_cost

            elif all(time == float("inf") for time in self.node_time_arrival):
                self.terminate_state_if_all_vehicles_come_back()

            elif self.tau_multiplicator > min_travel_time:
                # A vehicle finishes a Client's service time.
                if (
                    self.departure_tau[min_travel_time_vehicle] == min_travel_time
                    and len(self.vehicles_shortest_path[min_travel_time_vehicle]) <= 2
                ):
                    self.state.vehicle_completing_service[min_travel_time_vehicle] = 0
                    self.vehicle_distance_transition_cost(min_travel_time)
                    self.tau_vehicle_horizon_change[min_travel_time_vehicle] = (
                        self.state.tau_episode
                    )
                    self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0
                    self.state.tau_episode = min_travel_time
                    self.end_transition_function = 2
                    self.total_cost += self.transition_cost

                else:
                    next_node = self.vehicles_shortest_path[min_travel_time_vehicle][1]
                    self.state.vehicle_next_node[min_travel_time_vehicle] = next_node

                    # The vehicle reaches the depot for good.
                    if (
                        next_node == self.depot
                        and len(self.vehicles_shortest_path[min_travel_time_vehicle]) == 2
                    ):
                        self.vehicle_distance_transition_cost(min_travel_time)
                        self.state.vehicle_position[min_travel_time_vehicle] = self.depot

                        if self.state.tau_episode > self.work_time and self.node_time_arrival[
                            min_travel_time_vehicle
                        ] != float("inf"):
                            overtime = self.overtime_cost * (
                                self.state.tau_episode - self.work_time
                            )
                            self.transition_cost += overtime
                            self.total_overtime_cost += overtime
                            self.total_overtime_vehicles += 1

                        self.node_time_arrival[min_travel_time_vehicle] = float("inf")
                        self.departure_tau[min_travel_time_vehicle] = 0
                        self.total_cost += self.transition_cost
                        self.end_transition_function = 2

                    # The vehicle reaches the Client it was sent to.
                    elif (
                        next_node in self.state.clients_not_visited
                        and action[min_travel_time_vehicle] == next_node
                    ):
                        self.vehicle_reaches_client(min_travel_time_vehicle, min_travel_time)
                        self.end_transition_function = 2
                        self.total_cost += self.transition_cost

                    # The vehicle reaches an intermediate node.
                    else:
                        self.vehicle_reaches_node(min_travel_time, min_travel_time_vehicle)

            # A new decision epoch starts before any arrival.
            else:
                if self.tau_multiplicator >= 1198:
                    self.state.terminal = True
                    self.end_transition_function = 2
                    number_of_visited_clients = len(self.visited_clients)
                    self.transition_cost += 40000 - 200 * number_of_visited_clients
                    self.total_cost += self.transition_cost

                else:
                    self.vehicle_distance_transition_cost(self.tau_multiplicator)

                    time = (self.state.tau_episode + 180 - 2) / 60
                    if time % self.hours_max_duration == 0:
                        self.congestion_generator.generate(
                            self.state.tau_episode, self.congested_arcs
                        )

                    for vehicle in range(self.number_vehicles):
                        self.time_horizon_actualization(vehicle)

                    self.tau_multiplicator += self.tau_multiplicator_difference

                    # Phase-2 fix (ticket 12, ADR-0001 change log): the epoch-end
                    # gate below always fires at the emergency horizon, so the
                    # legacy fell through it after terminating and added the same
                    # transition_cost to total_cost a second time.
                    if self.tau_multiplicator >= 1150:
                        self.terminate_state_passing_horizon()
                    else:
                        # Preserved quirk: the transition only ends on epochs where
                        # the shifted clock (tau + 178) is a multiple of 6 — same
                        # arithmetic family as the congestion gate above.
                        time = self.state.tau_episode + 180 - 2
                        if time % 6 == 0:
                            self.end_transition_function = 2
                            self.total_cost += self.transition_cost

        return self.transition_cost

    def calculate_action_route(self, action: list[int]) -> None:
        """Ports ``calculate_action_route``: reroute vehicles whose decision changed."""
        for vehicle in range(len(action)):
            # Still serving a Client: only refresh the velocity bookkeeping.
            if self.departure_tau[vehicle] > self.state.tau_episode:
                self.create_and_actualize_state_velocity(vehicle)

            elif (
                action[vehicle] == self.depot and self.state.vehicle_position[vehicle] == self.depot
            ):
                self.node_time_arrival[vehicle] = float("inf")
                self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

            elif self.action[vehicle] != action[vehicle] and self.node_time_arrival[
                vehicle
            ] != float("inf"):
                vehicle_position = self.state.vehicle_position[vehicle]
                vehicle_destination = action[vehicle]
                if self.departure_tau[vehicle] == self.state.tau_episode:
                    # At a node: route straight from the current position.
                    shortest_path = list(
                        self.shortest_path_cache.path_between(
                            vehicle_position, vehicle_destination
                        ).nodes
                    )
                    self.vehicles_shortest_path[vehicle] = shortest_path[:]
                    self.create_and_actualize_state_velocity(vehicle)
                else:
                    # Mid-arc: finish the current arc, then follow the new route.
                    shortest_path = list(
                        self.shortest_path_cache.path_between(
                            self.vehicles_shortest_path[vehicle][1], vehicle_destination
                        ).nodes
                    )
                    shortest_path.insert(0, vehicle_position)
                    self.vehicles_shortest_path[vehicle] = shortest_path[:]

                self.action[vehicle] = action[vehicle]
                self.state.vehicles_direction[vehicle] = action[vehicle]

    def create_and_actualize_state_velocity(self, vehicle: int) -> None:
        """Ports ``create_and_actualize_state_velocity``."""
        if self.state.tau_episode > 1198:
            self.terminate_state_passing_horizon()

        elif self.departure_tau[vehicle] > self.state.tau_episode:
            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(0)
            self.node_time_arrival[vehicle] = self.departure_tau[vehicle]
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

        else:
            travel_time, velocity, _length = self.create_random_velocity(
                self.vehicles_shortest_path[vehicle][0],
                self.vehicles_shortest_path[vehicle][1],
                self.state.tau_episode,
            )

            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(velocity)
            self.node_time_arrival[vehicle] = self.state.tau_episode + travel_time
            self.departure_tau[vehicle] = self.state.tau_episode
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

    def vehicle_reaches_client(self, min_travel_time_vehicle: int, min_travel_time: float) -> None:
        """Ports ``vehicle_reaches_client``: serve, cost the time window, start service."""
        client = self.vehicles_shortest_path[min_travel_time_vehicle][1]
        self.visited_clients.append(client)
        # The float path-node id equals the int Client id (legacy parse quirk).
        self.state.clients_not_visited.remove(client)  # type: ignore[arg-type]

        earliness_time_window = self.time_windows[client][0]  # type: ignore[index]
        lateness_time_window = self.time_windows[client][1]  # type: ignore[index]

        self.state.vehicle_position[min_travel_time_vehicle] = client
        self.state.clients_arrival[client] = [min_travel_time, min_travel_time_vehicle]

        if min_travel_time < earliness_time_window:
            time_window_cost = (earliness_time_window - min_travel_time) * self.earliness_cost
            self.transition_cost += time_window_cost
            self.total_earliness_cost += time_window_cost
            self.total_earliness_clients += 1

        elif min_travel_time > lateness_time_window:
            time_window_cost = (min_travel_time - lateness_time_window) * self.delay_cost
            self.transition_cost += time_window_cost
            self.total_delay_cost += time_window_cost
            self.total_delay_clients += 1

        self.vehicle_distance_transition_cost(min_travel_time)

        self.departure_tau[min_travel_time_vehicle] = self.state.tau_episode + self.service_time
        self.state.vehicle_completing_service[min_travel_time_vehicle] = 1
        self.tau_vehicle_horizon_change[min_travel_time_vehicle] = self.state.tau_episode
        self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0

    def vehicle_distance_transition_cost(self, min_travel_time: float) -> None:
        """Ports ``vehicle_distance_transition_cost``: charge distance, advance the clock."""
        diff_tau = min_travel_time - self.state.tau_episode

        for vehicle in range(self.number_vehicles):
            if self.node_time_arrival[vehicle] != float("inf"):
                vehicle_velocity = self.state.observed_velocity[vehicle][self.state.n_arcs - 1]
                distance_travelled = vehicle_velocity * diff_tau
                self.total_distance_travelled += distance_travelled
                self.state.total_vehicle_distance_travelled[vehicle] += distance_travelled
                self.transition_cost += distance_travelled * self.distance_cost
                self.total_distance_cost += distance_travelled * self.distance_cost

        self.state.tau_episode = min_travel_time

    def vehicle_reaches_node(self, min_travel_time: float, min_travel_time_vehicle: int) -> None:
        """Ports ``vehicle_reaches_node``."""
        if min_travel_time > 1198 or self.state.tau_episode > 1198:
            self.terminate_state_passing_horizon()

        elif (
            len(self.vehicles_shortest_path[min_travel_time_vehicle]) == 2
            and self.vehicles_shortest_path[min_travel_time_vehicle][1] in self.visited_clients
        ):
            # Arrived at a Client already served by another vehicle: ask for a new decision.
            self.vehicle_distance_transition_cost(min_travel_time)
            self.state.vehicle_position[min_travel_time_vehicle] = self.vehicles_shortest_path[
                min_travel_time_vehicle
            ][1]
            self.departure_tau[min_travel_time_vehicle] = self.state.tau_episode
            self.tau_vehicle_horizon_change[min_travel_time_vehicle] = self.state.tau_episode
            self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0
            self.end_transition_function = 2

        else:
            self.state.vehicle_position[min_travel_time_vehicle] = self.vehicles_shortest_path[
                min_travel_time_vehicle
            ][1]
            self.vehicles_shortest_path[min_travel_time_vehicle].pop(0)
            self.vehicle_distance_transition_cost(min_travel_time)
            self.create_and_actualize_state_velocity(min_travel_time_vehicle)
            self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0

    def terminate_state_passing_horizon(self) -> None:
        """Ports ``terminate_state_passing_horizon``: charge unserved delays and overtime."""
        self.state.terminal = True
        self.end_transition_function = 2

        delay_costs: float = 0
        for client in self.state.clients_not_visited:
            client_due_time = self.time_windows[client][1]
            if self.state.tau_episode > client_due_time:
                delay_costs += (self.state.tau_episode - client_due_time) * self.delay_cost

        self.total_delay_cost += delay_costs
        self.transition_cost += delay_costs

        overtime_cost: float = 0
        for vehicle_position in self.state.vehicle_position:
            if vehicle_position != self.depot:
                overtime_cost += (self.state.tau_episode - self.work_time) * self.overtime_cost

        self.total_overtime_cost += overtime_cost
        self.transition_cost += overtime_cost

        self.total_cost += self.transition_cost

    def terminate_state_if_all_vehicles_come_back(self) -> None:
        """Ports ``terminate_state_if_all_vehicles_come_back``.

        Phase-2 fix (ticket 12, ADR-0001 change log): late unserved Clients are
        charged from the actual clock (``tau - due``) like the passing-horizon
        sibling — the legacy hardcoded its 1150 emergency horizon here.
        """
        self.state.terminal = True
        self.end_transition_function = 2
        delay_costs: float = 0
        for client in self.state.clients_not_visited:
            client_due_time = self.time_windows[client][1]
            if self.state.tau_episode > client_due_time:
                delay_costs += (self.state.tau_episode - client_due_time) * self.delay_cost

        self.total_delay_cost += delay_costs
        self.transition_cost += delay_costs
        self.total_cost += self.transition_cost

    def time_horizon_actualization(self, vehicle: int) -> None:
        """Ports ``time_horizon_actualization``: re-sample the current arc's velocity."""
        if self.departure_tau[vehicle] > self.state.tau_episode:
            random_velocity: float = 0
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode
            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(random_velocity)
            self.node_time_arrival[vehicle] = self.departure_tau[vehicle]

        elif self.node_time_arrival[vehicle] != float("inf"):
            _, random_velocity, arc_length = self.create_random_velocity(
                self.vehicles_shortest_path[vehicle][0],
                self.vehicles_shortest_path[vehicle][1],
                self.state.tau_episode,
            )

            time_in_arc = self.state.tau_episode - self.tau_vehicle_horizon_change[vehicle]
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

            distance_travelled = (
                self.state.observed_velocity[vehicle][self.state.n_arcs - 1] * time_in_arc
            )
            self.distance_arc_distance_travelled[vehicle] += distance_travelled

            distance_left_to_travel = arc_length - self.distance_arc_distance_travelled[vehicle]
            travel_time = distance_left_to_travel / random_velocity

            self.state.observed_velocity[vehicle].pop(0)
            self.state.observed_velocity[vehicle].append(random_velocity)
            self.node_time_arrival[vehicle] = self.state.tau_episode + travel_time

    def _next_congestion_end(self) -> tuple[float, int | None]:
        """Ports ``_next_congestion_end``: the soonest congestion expiry affecting a vehicle."""
        t_event = float("inf")
        veh_event = None
        for v in range(self.number_vehicles):
            if self.node_time_arrival[v] == float("inf"):
                continue
            start = self.vehicles_shortest_path[v][0]
            dest = self.vehicles_shortest_path[v][1]
            arc = (start, dest)
            if arc in self.congested_arcs:
                end = self.congested_arcs[arc][1]
                if self.state.tau_episode < end < self.node_time_arrival[v] and end < t_event:
                    t_event, veh_event = end, v
        return t_event, veh_event

    # --- Stochastic velocities (legacy ``DataCalculations`` methods) --------------

    def create_random_velocity(
        self, node_start: float, node_end: float, tau_episode: float
    ) -> tuple[float, float, float]:
        """Ports ``create_random_velocity``: (travel_time, velocity, length) for the arc.

        Normal samples are memoized per (arc, even minute); congested velocities are
        deterministic given the event and are not memoized, exactly as in the legacy.
        """
        minute_start = math.floor(tau_episode)
        if minute_start % 2 != 0:
            minute_start -= 1

        key_arc = (node_start, node_end)
        key_minute = (node_start, node_end, minute_start)

        if key_arc not in self.congested_arcs:
            return self._memoized_normal_velocity(key_minute)

        event_end = self.congested_arcs[key_arc][1]
        if tau_episode >= event_end:
            return self._memoized_normal_velocity(key_minute)

        length, speed = self.travel_time_model.travel_data[key_minute]  # type: ignore[index]
        congestion_multiplier = self.congested_arcs[key_arc][0]
        velocity = max(speed * congestion_multiplier, 0.0001)
        travel_time = length / velocity
        return travel_time, velocity, length

    def _memoized_normal_velocity(self, key: ArcMinuteKey) -> tuple[float, float, float]:
        """Return the memoized sample for the arc-minute, drawing a fresh one if absent."""
        if key not in self.sampled_arc_velocities:
            return self.generate_normal_velocity(*key)
        velocity, travel_time, length = self.sampled_arc_velocities[key]
        return travel_time, velocity, length

    def generate_normal_velocity(
        self, node_start: float, node_end: float, minute_start: int
    ) -> tuple[float, float, float]:
        """Ports ``generate_normal_velocity``: one global ``random.gauss`` per arc-minute.

        ``minute_start`` must already be the even-floored minute; the legacy re-floored
        it internally, but the only caller (``create_random_velocity``) does so first.
        """
        key = (node_start, node_end, minute_start)
        length, speed = self.travel_time_model.travel_data[key]  # type: ignore[index]
        std = self.travel_time_model.speed_std[key]  # type: ignore[index]

        velocity = random.gauss(speed, std)
        velocity = max(velocity, 0)

        # 60 km/h cap for ordinary streets (speeds are km/min), 120 km/h absolute cap.
        if speed < 1 and velocity > 1:
            velocity = 1
        if velocity > 2:
            velocity = 2
        if velocity <= 0:
            velocity = 0.001

        travel_time = length / velocity
        self.sampled_arc_velocities[key] = [velocity, travel_time, length]
        return travel_time, velocity, length
