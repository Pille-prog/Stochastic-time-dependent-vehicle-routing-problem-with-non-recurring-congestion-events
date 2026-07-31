"""FeatureExtractor parity against a verbatim transcription of the loop version.

Simulation-performance ticket 05 rebuilds the Policy's feature arithmetic as
vectorized numpy over the ticket-04 ``EpisodeGeometry`` matrices. The oracle here
is :class:`LoopReference` — the pre-ticket-05 ``MonteCarloPolicy`` bodies
(``_classify_delayed_clients``, ``_extract_general_state_features``,
``_extract_state_action_features``) copied verbatim, scalar lookups and all,
**including the duplicate-append quirk**. Every test drives both over the same
randomized worlds and States and asserts they agree.

Simulator-correctness ticket 06 (B10) is the one deliberate exception: the
loop's fourth earliness bin was never assigned, so ``LoopReference`` carries
the same one-line fix as :class:`~stdvrp.policies.feature_extraction.FeatureExtractor`
rather than reproducing that bug. Both sides must move together, or this file
would stop testing that the vectorization matches the (corrected) definition.

Two tolerances, deliberately:

* ``rtol=0`` (bit-exact) for the general-state features, the delayed-Client
  classification and the closest-Client multiplicities — those are integer or
  order-independent computations that the vectorization must not perturb.
* ``rtol=1e-12`` for the seven state-action features, whose per-vehicle and
  per-multiplicity sums are re-associated (Tier 2, see the ticket ``## Comments``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies.feature_extraction import FeatureExtractor
from stdvrp.simulation.state import State

DEPOT = 0
HORIZON_END = 780
# The oracle keeps this literal (it transcribes the pre-ticket-02 code
# unchanged); FeatureExtractor now takes it as episode_end_minute.
EPISODE_END = 1150


# --- the oracle: the loop bodies this ticket replaces, copied verbatim ----------


class LoopReference:
    """The pre-ticket-05 feature loops, transcribed unchanged."""

    def __init__(self, geometry, time_windows, number_vehicles, number_clients, depot):
        self.geometry = geometry
        self.time_windows = time_windows
        self.number_vehicles = number_vehicles
        self.number_clients = number_clients
        self.depot = depot
        self.delay_cost_factor = 1
        self.earliness_cost_factor = 0.1
        self.overtime_cost = 5 / 6
        self.service_time = 5
        self.end_of_horizon = HORIZON_END

    def classify_delayed_clients(self, state):
        self.state = state
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for client in self.state.clients_not_visited:
            assigned_vehicle = None
            min_travel_time = float("inf")
            for vehicle_idx, vehicle_position in enumerate(self.state.last_node_reached):
                # Ticket 04 (ADR-0005): moved in lockstep with the corrected
                # FeatureExtractor definition (same precedent as B10, ticket 06)
                # — a vehicle mid-arc past the depot is not idle.
                if (
                    vehicle_position == self.depot
                    and self.state.vehicle_standing[vehicle_idx]
                    and self.state.tau_episode > 310
                ):
                    continue

                travel_time = self.geometry.average_minutes(vehicle_position, client)
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx

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

        return self.delayed_clients, self.vehicle_to_clients

    def extract_general_state_features(self, state):
        self.state = state
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
        for client in self.state.clients_not_visited:
            client_earliness, _ = self.time_windows[client]
            client_earliness_value.append(client_earliness)

        client_counts_earliness = [0 for _ in range(4)]

        for i in client_earliness_value:
            if i < 400 and self.state.tau_episode < 400:
                client_counts_earliness[0] += 1
            elif 400 <= i < 500 and self.state.tau_episode < 500:
                client_counts_earliness[1] += 1
            elif 500 <= i < 600 and self.state.tau_episode < 600:
                client_counts_earliness[2] += 1

        # B10 fix: the fourth bin is whatever the first three left uncounted,
        # so the four bins always partition clients_not_visited.
        client_counts_earliness[3] = len(self.state.clients_not_visited) - sum(
            client_counts_earliness[:3]
        )

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
        return self.X_general_state

    def extract_state_action_features(self, state, action, vehicle_to_clients):
        self.state = state
        cg_clients = self.time_windows
        geometry = self.geometry
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
        features.append(0)

        total_dist = sum(
            geometry.length(state.last_node_reached[i], action[i]) for i in range(n_veh)
        )
        features.append(total_dist / 100.0)

        earliness_cost = 0.0
        delay_cost = 0.0
        for i, a in enumerate(action):
            if a in clients_all and a != depot:
                travel_time = geometry.average_minutes(state.last_node_reached[i], a)
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
            for _, client in vehicle_to_clients[veh]:
                if client not in action:
                    t1 = geometry.average_minutes(state.last_node_reached[veh], action[veh])
                    t2 = geometry.average_minutes(action[veh], client)
                    est = tau + t1 + t2 + service_time
                    _, due_tw = cg_clients[client]
                    if est > due_tw:
                        future_delay += (est - max(due_tw, tau)) * delay_fact

        features.append(future_delay / 2500.0)

        overtime_cost = 0.0
        for i, a in enumerate(action):
            if a != depot:
                t1 = geometry.average_minutes(state.last_node_reached[i], a)
                t2 = geometry.average_minutes(a, depot)
                est_ret = tau + t1 + t2 + service_time
            else:
                est_ret = tau + geometry.average_minutes(state.last_node_reached[i], depot)

            if est_ret > end_horizon:
                base = end_horizon if tau < end_horizon else tau
                overtime_cost += (est_ret - base) * overtime_fact

        features.append(overtime_cost / 180.0)
        return features


# --- randomized worlds ---------------------------------------------------------


class Scenario:
    """One randomized world plus a State to evaluate both implementations on."""

    def __init__(self, seed: int, *, number_vehicles: int = 4, number_clients: int = 9):
        rng = np.random.default_rng(seed)
        node_count = number_clients + 6
        self.clients = [
            int(c) for c in rng.choice(range(1, node_count), size=number_clients, replace=False)
        ]
        self.time_windows = {
            client: (start, start + 120)
            for client, start in zip(
                self.clients,
                (int(v) for v in rng.integers(300, 660, size=number_clients)),
                strict=True,
            )
        }
        entries = {}
        for node in range(node_count):
            for client in [DEPOT, *self.clients]:
                minutes = 0.0 if node == client else float(rng.uniform(1.0, 90.0))
                length = 0.0 if node == client else float(rng.uniform(0.5, 40.0))
                entries[(node, client)] = ShortestPath(
                    [float(node), float(client)], minutes, length
                )
        self.geometry = EpisodeGeometry.build(ShortestPathCache(entries), self.clients, DEPOT)

        self.number_vehicles = number_vehicles
        self.number_clients = number_clients
        self._rng = rng
        self._node_count = node_count

    def state(self, *, tau: float, remaining: list[int], depot_vehicles: int) -> State:
        state = State(self.number_vehicles, list(remaining), 3, 300, DEPOT)
        state.tau_episode = tau
        positions = [DEPOT] * depot_vehicles + [
            int(self._rng.integers(0, self._node_count))
            for _ in range(self.number_vehicles - depot_vehicles)
        ]
        state.last_node_reached = [float(p) for p in positions]
        return state

    def extractor(self) -> FeatureExtractor:
        return FeatureExtractor(
            self.geometry,
            self.time_windows,
            number_vehicles=self.number_vehicles,
            number_clients=self.number_clients,
            depot=DEPOT,
            shift_end_minute=HORIZON_END,
            episode_end_minute=EPISODE_END,
            service_time=5,
            delay_cost_factor=1,
            earliness_cost_factor=0.1,
            overtime_cost_factor=5 / 6,
        )

    def reference(self) -> LoopReference:
        return LoopReference(
            self.geometry, self.time_windows, self.number_vehicles, self.number_clients, DEPOT
        )


# tau values straddle every hardcoded cutoff the feature definition contains:
# the 310 depot-idle cutoff, the 400/500/600 earliness bins, 580, and the horizon.
TAUS = (300.0, 305.0, 311.0, 399.0, 450.0, 550.0, 590.0, 700.0, 800.0)


def scenarios():
    """Worlds x taus x remaining-Client sets x how many vehicles idle at the depot."""
    for seed in range(4):
        scenario = Scenario(seed)
        for tau in TAUS:
            for kept in (len(scenario.clients), 5, 2, 1, 0):
                remaining = scenario.clients[:kept]
                for depot_vehicles in (0, 1, scenario.number_vehicles):
                    yield (
                        scenario,
                        scenario.state(tau=tau, remaining=remaining, depot_vehicles=depot_vehicles),
                    )


ALL_SCENARIOS = list(scenarios())


def action_variants(scenario: Scenario, state: State) -> list[list[int]]:
    """Action vectors worth checking: all-depot, all-remaining, and a mix."""
    n = scenario.number_vehicles
    remaining = list(state.clients_not_visited)
    pool = remaining or [DEPOT]
    return [
        [DEPOT] * n,
        [pool[i % len(pool)] for i in range(n)],
        [DEPOT if i % 2 else pool[i % len(pool)] for i in range(n)],
        [scenario.clients[i % len(scenario.clients)] for i in range(n)],
    ]


# --- parity --------------------------------------------------------------------


class TestClassificationParity:
    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_delayed_clients_match_bit_for_bit(self, scenario, state):
        expected, _ = scenario.reference().classify_delayed_clients(state)
        produced = scenario.extractor().state_features(state).delayed_clients

        assert [list(v) for v in produced] == expected

    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_closest_client_multiplicities_reproduce_the_duplicate_appends(self, scenario, state):
        _, vehicle_to_clients = scenario.reference().classify_delayed_clients(state)
        features = scenario.extractor().state_features(state)

        columns = scenario.geometry.columns
        for vehicle in range(scenario.number_vehicles):
            expected: defaultdict[int, int] = defaultdict(int)
            for _, client in vehicle_to_clients[vehicle]:
                expected[client] += 1
            produced = {
                columns[column]: int(count)
                for column, count in enumerate(features.closest_client_counts[vehicle])
                if count
            }
            assert produced == dict(expected)


class TestGeneralStateFeatureParity:
    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_general_features_are_bit_exact(self, scenario, state):
        expected = scenario.reference().extract_general_state_features(state)
        produced = scenario.extractor().state_features(state).general

        assert produced.shape == (12,)
        np.testing.assert_array_equal(produced, np.array(expected, dtype=np.float64))


class TestEarlinessBinPartition:
    """B10: the four earliness bins partition pending demand, at every tau.

    ``general[7:11]`` are ``count / number_clients`` for the four earliness
    bins; their sum must equal ``len(clients_not_visited) / number_clients``
    regardless of which of the tau-gated bins is open, so that no pending
    Client is ever counted by zero bins.
    """

    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_the_four_bins_sum_to_pending_clients(self, scenario, state):
        features = scenario.extractor().state_features(state)

        produced = float(np.sum(features.general[7:11]))
        expected = len(state.clients_not_visited) / scenario.number_clients

        assert produced == pytest.approx(expected, abs=1e-9)


class TestStateActionFeatureParity:
    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_action_features_match_the_loops(self, scenario, state):
        reference = scenario.reference()
        _, vehicle_to_clients = reference.classify_delayed_clients(state)
        features = scenario.extractor().state_features(state)

        for action in action_variants(scenario, state):
            expected = reference.extract_state_action_features(state, action, vehicle_to_clients)
            produced = scenario.extractor().action_features(features, action)

            assert produced.shape == (19,)
            np.testing.assert_allclose(
                produced[12:], np.array(expected, dtype=np.float64), rtol=1e-12, atol=1e-12
            )

    @pytest.mark.parametrize("scenario, state", ALL_SCENARIOS)
    def test_candidate_batch_matches_one_row_per_candidate(self, scenario, state):
        reference = scenario.reference()
        _, vehicle_to_clients = reference.classify_delayed_clients(state)
        extractor = scenario.extractor()
        features = extractor.state_features(state)

        action = action_variants(scenario, state)[2]
        candidates = [DEPOT, *state.clients_not_visited][:5]
        vehicle = 1

        batched = extractor.candidate_features(features, action, vehicle, candidates)

        assert batched.shape == (len(candidates), 19)
        for row, candidate in enumerate(candidates):
            current_action = list(action)
            current_action[vehicle] = candidate
            expected = reference.extract_state_action_features(
                state, current_action, vehicle_to_clients
            )
            np.testing.assert_allclose(
                batched[row, 12:], np.array(expected, dtype=np.float64), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_array_equal(batched[row, :12], features.general)


class TestFeatureExtractorShape:
    def test_the_permanently_zero_feature_keeps_w_at_nineteen_components(self):
        scenario, state = ALL_SCENARIOS[0]
        extractor = scenario.extractor()
        features = extractor.state_features(state)

        row = extractor.action_features(features, [DEPOT] * scenario.number_vehicles)

        assert row.shape == (19,)
        assert row[13] == 0.0
