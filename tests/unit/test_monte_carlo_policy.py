"""Unit tests for MonteCarloPolicy's W update and epsilon-greedy selection (ticket 11).

These pin the pure computations on a world small enough to verify by hand: the
19-component feature/weight dimensions, one hand-computed SGD step, the backward
accumulation of the Monte Carlo return, and the two epsilon extremes (0 = pure
greedy, 1 = pure random from the injected exploration Generator, ticket 13).
"""

import math
from typing import NamedTuple

import numpy as np
import pytest

from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies.monte_carlo import MonteCarloPolicy, TimeWindows
from stdvrp.simulation.state import State

DEPOT = 0
HORIZON_END = 780


class World(NamedTuple):
    """The three inputs a policy needs, built together by the make_*_world helpers."""

    cache: ShortestPathCache
    time_windows: TimeWindows
    state: State


def make_cache(arcs: dict) -> ShortestPathCache:
    """arcs: (node, node) -> (average_minutes, length_km)."""
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


class ScriptedRng:
    """A minimal exploration-Generator double: exact, ordered ``choice``/``random``."""

    def __init__(self, *, choices: list = (), randoms: list = ()) -> None:
        self._choices = list(choices)
        self._randoms = list(randoms)
        self.choice_calls = 0
        self.random_calls = 0

    def choice(self, seq):
        self.choice_calls += 1
        return self._choices.pop(0)

    def random(self) -> float:
        self.random_calls += 1
        return self._randoms.pop(0)


def make_policy(world: World, *, epsilon=0.0, W=None, lr=0.0, seed=0, rng=None):
    # The constructor draws one exploration_rng choice per vehicle (ticket 13).
    if rng is None:
        rng = np.random.default_rng(seed)
    return MonteCarloPolicy(
        number_vehicles=1,
        shortest_path_cache=world.cache,
        time_windows=world.time_windows,
        state=world.state,
        number_clients=len(world.time_windows),
        epsilon=epsilon,
        depot=DEPOT,
        number_actions_test=2,
        horizon_end_minute=HORIZON_END,
        W=W,
        exploration_rng=rng,
        number_actions_train=2,
        learning_rate=lr,
    )


# --- W update ------------------------------------------------------------------


def make_update_world():
    """One vehicle at the depot at tau=400 with two unserved Clients."""
    cache = make_cache(
        {
            (0, 0): (0.0, 0.0),
            (0, 1): (10.0, 5.0),
            (1, 0): (10.0, 5.0),
            (0, 2): (20.0, 8.0),
            (2, 0): (20.0, 8.0),
            (1, 2): (15.0, 6.0),
            (2, 1): (15.0, 6.0),
        }
    )
    time_windows = {1: (350, 400), 2: (450, 500)}
    state = State(
        number_vehicles=1, clients=[1, 2], n_arcs=3, horizon_start_minute=300, depot=DEPOT
    )
    state.tau_episode = 400
    return World(cache, time_windows, state)


def hand_computed_features() -> np.ndarray:
    """The 19 features of the update world for action [1], derived on paper.

    General state (12): polynomial terms of clients-left and normalized time; the
    earliness bins hold only Client 2 (window start 450 in [400, 500), tau < 500);
    the mean-earliness diff is 0 because mean earliness == tau == 400.
    State-action (7) for sending the vehicle from the depot to Client 1: distance
    5 km; arrival 400 + 10 breaches Client 1's due time 400 by 10 minutes; no
    earliness, future delay or overtime; the second feature is the preserved
    always-zero quirk.
    """
    clients_left = 2 / 150
    time_left = (1150 - 400) / 850
    time = (400 - 300) / 850
    general = [
        math.sqrt(clients_left),
        time_left,
        time_left**2,
        clients_left**2,
        clients_left**2 * time,
        time**2 * clients_left,
        time**2 * clients_left**2,
        0.0,
        1 / 2,
        0.0,
        0.0,
        0.0,
    ]
    state_action = [0.0, 0.0, 5 / 100, 0.0, 10 / 60, 0.0, 0.0]
    return np.array(general + state_action)


class TestWUpdate:
    def test_feature_vector_and_created_w_have_19_components(self):
        world = make_update_world()
        policy = make_policy(world, W=None)

        assert policy.W is not None  # created by the constructor's greedy pass
        assert policy.W.shape == (19,)
        assert len(policy.X_general_state) == 12
        assert len(policy.X_state_action) == 7

    def test_update_preserves_w_dimensions(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        policy.update_W([world.state], [[1]], [0.0, 20.0])

        assert policy.W.shape == (19,)

    def test_single_step_matches_hand_computation(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        policy.update_W([world.state], [[1]], [0.0, 20.0])

        # U_t = rewards[1] = 20; acquired-cost baseline and Q_pred are both 0,
        # so W steps from zero to lr * U_t * X = 10 * X.
        np.testing.assert_allclose(policy.W, 10 * hand_computed_features(), rtol=1e-12)

    def test_zero_learning_rate_is_a_no_op(self):
        world = make_update_world()
        initial = np.full(19, 0.3)
        policy = make_policy(world, W=initial.copy(), lr=0.0)

        policy.update_W([world.state], [[1]], [0.0, 20.0])

        np.testing.assert_array_equal(policy.W, initial)

    def test_return_accumulates_rewards_newest_first(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        policy.update_W([world.state, world.state], [[1], [1]], [0.0, 5.0, 7.0])

        # Epochs replay newest-first: U_t is 7 for t=1, then 7 + 5 = 12 for t=0,
        # each stepping W against the Q predicted by the weights so far.
        x = hand_computed_features()
        expected = np.zeros(19)
        for u_t in (7.0, 12.0):
            expected = expected + 0.5 * ((u_t - np.dot(x, expected)) * x)
        np.testing.assert_allclose(policy.W, expected, rtol=1e-12)


# --- Epsilon-greedy selection --------------------------------------------------


def make_selection_world():
    """One vehicle at the depot at tau=300 with four unserved Clients.

    With two candidate actions, the possible set is the two closest Clients
    [1, 2]; a weight vector that prices only the distance feature makes Client 1
    (5 km) the unique greedy argmin over Client 2 (8 km).
    """
    client_arcs = {}
    for client, (minutes, length) in {
        1: (10.0, 5.0),
        2: (20.0, 8.0),
        3: (30.0, 12.0),
        4: (40.0, 16.0),
    }.items():
        client_arcs[(0, client)] = (minutes, length)
        client_arcs[(client, 0)] = (minutes, length)
    for a in range(1, 5):
        for b in range(1, 5):
            if a != b:
                client_arcs[(a, b)] = (10.0, 4.0)
    client_arcs[(0, 0)] = (0.0, 0.0)
    cache = make_cache(client_arcs)

    time_windows = {1: (400, 500), 2: (420, 520), 3: (440, 540), 4: (460, 560)}
    state = State(
        number_vehicles=1, clients=[1, 2, 3, 4], n_arcs=3, horizon_start_minute=300, depot=DEPOT
    )
    return World(cache, time_windows, state)


def distance_only_w() -> np.ndarray:
    w = np.zeros(19)
    w[14] = 1.0  # the total-distance feature
    return w


class TestEpsilonGreedy:
    def test_epsilon_zero_is_pure_greedy(self):
        world = make_selection_world()
        # constructor's initial-action draw (feasible: in [1, 2], no repair draw),
        # then the gate draw in decide_train (epsilon=0.0 always rejects it).
        rng = ScriptedRng(choices=[1], randoms=[0.5])
        policy = make_policy(world, epsilon=0.0, W=distance_only_w(), rng=rng)

        action = policy.decide_train(world.state)

        assert action == [1]  # the closest Client, never the exploratory draw
        assert rng.choice_calls == 1  # only the constructor's initial-action draw
        assert rng.random_calls == 1  # the gate draw itself, unconditionally made

    def test_epsilon_zero_matches_the_evaluation_greedy_decision(self):
        world = make_selection_world()
        policy = make_policy(
            world, epsilon=0.0, W=distance_only_w(), rng=ScriptedRng(choices=[1], randoms=[0.5])
        )

        twin_world = make_selection_world()
        twin_rng = ScriptedRng(choices=[1])
        twin = make_policy(twin_world, epsilon=0.0, W=distance_only_w(), rng=twin_rng)

        assert policy.decide_train(world.state) == twin.decide(twin_world.state)

    def test_epsilon_one_always_explores_via_the_injected_generator(self):
        world = make_selection_world()
        # constructor's initial-action draw, then the gate draw (epsilon=1.0
        # always explores) and the exploratory choice itself.
        rng = ScriptedRng(choices=[1, 2], randoms=[0.5])
        policy = make_policy(world, epsilon=1.0, W=distance_only_w(), rng=rng)
        possible = policy._select_vehicle_possible_actions(2, 0)
        assert possible == [1, 2]  # precondition: the two closest Clients

        assert policy.decide_train(world.state) == [2]
        assert rng.choice_calls == 2  # constructor's action + the exploratory pick
        assert rng.random_calls == 1  # the gate draw

    def test_epsilon_one_reproduces_with_equal_seeds(self):
        actions = []
        for _ in range(2):
            world = make_selection_world()
            policy = make_policy(world, epsilon=1.0, W=distance_only_w(), seed=11)
            actions.append(policy.decide_train(world.state))

        assert actions[0] == actions[1]
        assert actions[0][0] in {1, 2}

    def test_decide_train_requires_number_actions_train(self):
        world = make_update_world()
        policy = MonteCarloPolicy(
            number_vehicles=1,
            shortest_path_cache=world.cache,
            time_windows=world.time_windows,
            state=world.state,
            number_clients=len(world.time_windows),
            epsilon=0.0,
            depot=DEPOT,
            number_actions_test=2,
            horizon_end_minute=HORIZON_END,
            W=np.zeros(19),
            exploration_rng=np.random.default_rng(0),
        )

        with pytest.raises(ValueError, match="number_actions_train"):
            policy.decide_train(world.state)
