"""Unit tests for MonteCarloPolicy's W update, epsilon-greedy and candidate selection.

These pin the pure computations on a world small enough to verify by hand: the
19-component feature/weight dimensions, one hand-computed SGD step, the backward
accumulation of the Monte Carlo return, and the two epsilon extremes (0 = pure
greedy, 1 = pure random from the injected exploration Generator, ticket 13).

``TestCandidateSelection`` (simulation-performance ticket 07) pins the vectorized
per-vehicle candidate selection against :func:`reference_possible_actions`, the
pre-vectorization implementation kept here verbatim as a differential oracle.

Since ticket 05 the per-decision State features are an argument rather than an
attribute, so these tests build one with :func:`state_features` and hand it in —
exactly as ``decide``/``decide_train`` do. Feature *arithmetic* is pinned in
``tests/unit/test_feature_extraction.py``, not here.
"""

import heapq
import math
from typing import Any, NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies.feature_extraction import StateFeatures
from stdvrp.policies.monte_carlo import MonteCarloPolicy, TimeWindows
from stdvrp.simulation.state import State, TrainingSnapshot

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


def make_geometry(world: World) -> EpisodeGeometry:
    return EpisodeGeometry.build(world.cache, clients=list(world.time_windows), depot=DEPOT)


def state_features(policy: MonteCarloPolicy) -> StateFeatures:
    """The per-decision State features for the policy's current State."""
    return policy.feature_extractor.state_features(policy.state)


def make_policy(world: World, *, epsilon=0.0, W=None, lr=0.0, seed=0, rng=None, vehicles=1):
    # The constructor draws one exploration_rng choice per vehicle (ticket 13).
    if rng is None:
        rng = np.random.default_rng(seed)
    return MonteCarloPolicy(
        number_vehicles=vehicles,
        geometry=make_geometry(world),
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
        number_vehicles=1,
        clients=[1, 2],
        n_observed_velocities=3,
        horizon_start_minute=300,
        depot=DEPOT,
    )
    state.tau_episode = 400
    return World(cache, time_windows, state)


def hand_computed_features() -> np.ndarray:
    """The 19 features of the update world for action [1], derived on paper.

    General state (12): polynomial terms of clients-left and normalized time.
    Bin 1 holds Client 2 (window start 450 in [400, 500), tau < 500). Bin 0
    would hold Client 1 (window start 350 < 400), but tau == 400 already gates
    it shut (``tau < 400`` is false) — so Client 1 falls into bin 3 instead
    (B10: the fourth bin is whatever the first three leave uncounted, keeping
    the four bins a partition of the two pending Clients). The mean-earliness
    diff is 0 because mean earliness == tau == 400.
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
        1 / 2,
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
        features = state_features(policy)
        assert features.general.shape == (12,)
        assert policy.feature_extractor.action_features(features, policy.action).shape == (19,)

    def test_update_preserves_w_dimensions(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        policy.update_W([TrainingSnapshot.capture(world.state)], [[1]], [0.0, 20.0])

        assert policy.W.shape == (19,)

    def test_single_step_matches_hand_computation(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        policy.update_W([TrainingSnapshot.capture(world.state)], [[1]], [0.0, 20.0])

        # U_t = rewards[1] = 20; acquired-cost baseline and Q_pred are both 0,
        # so W steps from zero to lr * U_t * X = 10 * X.
        np.testing.assert_allclose(policy.W, 10 * hand_computed_features(), rtol=1e-12)

    def test_zero_learning_rate_is_a_no_op(self):
        world = make_update_world()
        initial = np.full(19, 0.3)
        policy = make_policy(world, W=initial.copy(), lr=0.0)

        policy.update_W([TrainingSnapshot.capture(world.state)], [[1]], [0.0, 20.0])

        np.testing.assert_array_equal(policy.W, initial)

    def test_return_accumulates_rewards_newest_first(self):
        world = make_update_world()
        policy = make_policy(world, W=np.zeros(19), lr=0.5)

        snapshot = TrainingSnapshot.capture(world.state)
        policy.update_W([snapshot, snapshot], [[1], [1]], [0.0, 5.0, 7.0])

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
        number_vehicles=1,
        clients=[1, 2, 3, 4],
        n_observed_velocities=3,
        horizon_start_minute=300,
        depot=DEPOT,
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
        possible = policy._select_vehicle_possible_actions(2, 0, state_features(policy))
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
            geometry=make_geometry(world),
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


# --- Batched candidate evaluation (ticket 05, simulation-performance) ----------


class TestBatchedQEvaluation:
    """``_best_q_action`` scores a vehicle's whole candidate set in one ``X @ W``."""

    def test_a_tie_keeps_the_first_candidate_in_iteration_order(self):
        # Every (node, column) pair of this world costs the same, so all
        # candidates score identically. The legacy scanned them keeping a strict
        # ``<`` against a running best, which leaves the *first* winner standing;
        # np.argmin over the batch must break the tie the same way.
        world = make_fleet_world({}, clients=[7, 3, 5], vehicles=1)
        policy = make_policy(world, W=np.ones(19))
        features = state_features(policy)

        for candidates in ([7, 3, 5], [5, 3, 7], [3, 7]):
            assert policy._best_q_action(features, 0, candidates) == candidates[0]

    def test_the_batch_picks_the_same_winner_as_scoring_candidates_one_by_one(self):
        # The legacy scored one candidate per pass: build the feature row and dot
        # it with W. Batching must not move the winner — here the depot, which the
        # distance-only W prices at zero.
        world = make_selection_world()
        policy = make_policy(world, W=distance_only_w())
        features = state_features(policy)
        candidates = [1, 2, 3, 4, DEPOT]

        one_by_one = [
            float(np.dot(policy.feature_extractor.action_features(features, [c]), policy.W))
            for c in candidates
        ]

        winner = int(np.argmin(one_by_one))
        assert one_by_one.count(one_by_one[winner]) == 1  # precondition: no tie to break
        assert policy._best_q_action(features, 0, candidates) == candidates[winner]

    def test_a_missing_w_is_created_at_the_feature_width(self):
        world = make_selection_world()
        policy = make_policy(world, W=None)
        policy.W = None

        assert policy._best_q_action(state_features(policy), 0, [1, 2]) == 1
        assert policy.W is not None
        np.testing.assert_array_equal(policy.W, np.zeros(19))


# --- Candidate action selection (ticket 07, simulation-performance) ------------


def reference_possible_actions(
    policy: MonteCarloPolicy, number_of_actions: int, vehicle: int, features: StateFeatures
) -> list[int]:
    """``_select_vehicle_possible_actions`` exactly as it read before ticket 07.

    The differential oracle for the vectorized selection: one
    ``geometry.average_minutes`` lookup per remaining Client into a list of
    ``(travel time, Client)`` tuples, ``heapq.nsmallest`` over it, then the
    ``list(set(...))`` dedup, the depot appends and the delayed-client injection.
    Kept verbatim — if the two ever disagree on any state, the optimization
    changed behavior. (Only the two collaborators ticket 05 turned from
    attributes into values — the endgame classifier's return and
    ``features.delayed_clients`` — read differently here than they did then.)
    """
    forbidden_actions = [policy.action[v] for v in range(policy.number_vehicles) if v != vehicle]
    state = policy.state
    position = state.vehicle_position[vehicle]
    possible_actions: list[int] = []

    if (position == policy.depot and state.tau_episode > 350) or len(
        state.clients_not_visited
    ) == 0:
        possible_actions.append(policy.depot)

    elif len(state.clients_not_visited) < 3:
        # ticket 05 (B11): filtered here too, to stay in lockstep with the fix in
        # _select_vehicle_possible_actions — this oracle pins ticket 07's
        # vectorization, not the pre-ticket-05 duplicate-booking bug.
        shortest_distance_clients = policy._classify_shortest_distance_clients()
        for _distance, client in shortest_distance_clients[vehicle]:
            if client not in forbidden_actions:
                possible_actions.append(client)
        if not possible_actions:
            possible_actions.append(policy.depot)

    else:
        travel_times = [
            (policy.geometry.average_minutes(position, client), client)
            for client in state.clients_not_visited
        ]
        allowed = [pair for pair in travel_times if pair[1] not in forbidden_actions]
        possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, allowed)]
        possible_actions = list(set(possible_actions))

        if (
            policy.geometry.average_minutes(position, policy.depot) + state.tau_episode
            > policy.end_of_horizon
        ):
            possible_actions.append(policy.depot)

        for delayed_client in features.delayed_clients[vehicle]:
            if delayed_client not in possible_actions and delayed_client not in forbidden_actions:
                possible_actions.append(delayed_client)

        if len(possible_actions) == 0:
            possible_actions.append(policy.depot)

    return possible_actions


def make_dense_cache(minutes: dict[tuple[int, int], float], nodes, columns) -> ShortestPathCache:
    """A cache dense over ``nodes x columns``; ``minutes`` overrides the default 100.0."""
    return make_cache(
        {
            (node, column): (minutes.get((node, column), 100.0), 1.0)
            for node in nodes
            for column in columns
        }
    )


def make_fleet_world(minutes, *, clients, vehicles, nodes=None, tau=300.0) -> World:
    """``vehicles`` vehicles at the depot at ``tau`` with every Client unvisited."""
    nodes = list(nodes if nodes is not None else [DEPOT, *clients])
    cache = make_dense_cache(minutes, nodes, [DEPOT, *clients])
    time_windows = {client: (400, 900) for client in clients}
    state = State(
        number_vehicles=vehicles,
        clients=list(clients),
        n_observed_velocities=3,
        horizon_start_minute=300,
        depot=DEPOT,
    )
    state.tau_episode = tau
    return World(cache, time_windows, state)


class TestCandidateSelection:
    def test_top_k_breaks_travel_time_ties_by_client_id(self):
        # Clients 5 and 3 sit at the same travel time, 5 first in the State's
        # list: the tuple sort the vectorized top-k replaces broke the tie on the
        # id, so a merely *stable* sort on the times is not enough.
        world = make_fleet_world({(0, 3): 7.0, (0, 5): 7.0}, clients=[5, 3, 7, 9], vehicles=1)
        policy = make_policy(world, W=np.zeros(19))
        features = state_features(policy)

        assert policy._select_vehicle_possible_actions(1, 0, features) == [3]
        assert policy._select_vehicle_possible_actions(2, 0, features) == (
            reference_possible_actions(policy, 2, 0, features)
        )

    def test_a_pair_the_cache_never_priced_still_raises(self):
        # The scalar lookups this replaced raised KeyError for an unpriced
        # (position, Client) pair; a raw row read would instead hand back the 0.0
        # such a cell holds and make the unreachable Client the nearest candidate.
        clients = [1, 2, 3]
        # Node 9 is an intermediate path node, never a Client: the cache prices
        # it toward every Client but 3.
        cache = make_cache(
            {
                (node, column): (10.0 + node + column, 1.0)
                for node in [DEPOT, *clients, 9]
                for column in [DEPOT, *clients]
                if (node, column) != (9, 3)
            }
        )
        state = State(
            number_vehicles=1,
            clients=list(clients),
            n_observed_velocities=3,
            horizon_start_minute=300,
            depot=DEPOT,
        )
        world = World(cache, {client: (400, 900) for client in clients}, state)
        policy = make_policy(world, W=np.zeros(19))

        state.vehicle_position[0] = 9

        with pytest.raises(KeyError):
            policy.geometry.average_minutes(9, 3)
        with pytest.raises(KeyError):
            policy._select_vehicle_possible_actions(2, 0, state_features(policy))

    def test_forbidden_clients_are_replaced_by_the_next_nearest(self):
        world = make_fleet_world(
            {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 4.0},
            clients=[1, 2, 3, 4],
            vehicles=3,
        )
        policy = make_policy(world, W=np.zeros(19), vehicles=3)
        policy.action = [4, 1, 2]  # vehicle 0's two nearest Clients are taken

        # The k nearest are 1 and 2; both forbidden, so selection reaches deeper.
        assert policy._select_vehicle_possible_actions(2, 0, state_features(policy)) == [3, 4]

    def test_fewer_allowed_clients_than_requested_returns_all_of_them(self):
        world = make_fleet_world(
            {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0}, clients=[1, 2, 3], vehicles=3
        )
        policy = make_policy(world, W=np.zeros(19), vehicles=3)
        policy.action = [0, 1, 2]

        assert policy._select_vehicle_possible_actions(5, 0, state_features(policy)) == [3]

    def test_selected_ids_are_the_states_own_python_ints(self):
        # They land in self.action and flow on into the Model, which indexes
        # dicts with them — a numpy scalar leaking in here would be invisible
        # to == comparisons and change hashing downstream.
        world = make_fleet_world(
            {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0}, clients=[1, 2, 3, 4], vehicles=1
        )
        policy = make_policy(world, W=np.zeros(19))

        actions = policy._select_vehicle_possible_actions(3, 0, state_features(policy))

        assert actions
        assert all(type(action) is int for action in actions)

    def test_selection_follows_clients_being_served(self):
        # The Model removes served Clients from the very list the State holds;
        # the cached remaining-Client arrays must not outlive that mutation.
        world = make_fleet_world(
            {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 4.0},
            clients=[1, 2, 3, 4],
            vehicles=1,
        )
        policy = make_policy(world, W=np.zeros(19))
        assert policy._select_vehicle_possible_actions(2, 0, state_features(policy)) == [1, 2]

        world.state.clients_not_visited.remove(1)

        assert policy._select_vehicle_possible_actions(2, 0, state_features(policy)) == [2, 3]

    def test_matches_the_pre_vectorization_reference_over_random_states(self):
        """Differential test: every branch, over states no hand-written case would reach.

        Travel times are quantized to half-minutes so float ties between Clients
        are common — the case ``np.argpartition`` alone would not order the way
        the ``(time, client)`` tuple sort did.
        """
        rng = np.random.default_rng(20260724)
        clients = list(range(1, 9))
        nodes = list(range(0, 13))
        columns = [DEPOT, *clients]
        minutes = {
            (node, column): float(rng.integers(2, 12)) / 2 for node in nodes for column in columns
        }
        world = make_fleet_world(minutes, clients=clients, vehicles=3, nodes=nodes)
        policy = make_policy(world, W=np.zeros(19), vehicles=3)

        for _ in range(200):
            remaining = rng.choice(clients, size=rng.integers(0, len(clients) + 1), replace=False)
            # In place, exactly as Model.vehicle_reaches_client mutates the list.
            world.state.clients_not_visited[:] = [int(client) for client in remaining]
            world.state.vehicle_position[:] = [int(node) for node in rng.choice(nodes, size=3)]
            world.state.tau_episode = float(rng.integers(300, 800))
            policy.action = [int(action) for action in rng.choice(columns, size=3)]
            features = state_features(policy)
            number_of_actions = int(rng.integers(1, 5))

            for vehicle in range(3):
                expected = reference_possible_actions(policy, number_of_actions, vehicle, features)
                assert (
                    policy._select_vehicle_possible_actions(number_of_actions, vehicle, features)
                    == expected
                )


# --- Endgame invariants (ticket 05, simulator-correctness: B5's crash, B11's --
# --- duplicate booking) ---------------------------------------------------------


@st.composite
def endgame_configs(draw: st.DrawFn) -> dict[str, Any]:
    """Small pending-Client counts and small fleets, tau swept across 310/350.

    The regime the review flags as most exposed for both findings: B5's crash
    needs exactly one pending Client with every vehicle reading
    ``position == depot`` in the uncovered ``(310, 350]`` window; B11's
    duplicate booking needs the two-Client classifier branch and >=2 vehicles.
    Sweeping 0/1/2 remaining Clients keeps 0 and 2 as controls alongside 1, the
    branch B5 actually crashes in.
    """
    vehicle_count = draw(st.integers(1, 4))
    return {
        "tau": draw(st.floats(300, 360, allow_nan=False)),
        "remaining": draw(st.integers(0, 2)),
        "vehicle_count": vehicle_count,
        "at_depot": draw(st.lists(st.booleans(), min_size=vehicle_count, max_size=vehicle_count)),
    }


def make_endgame_world(vehicle_count: int) -> World:
    """Two Clients, a fleet of ``vehicle_count``, plus one node away from the depot.

    The test body overwrites ``clients_not_visited``/``tau_episode``/
    ``vehicle_position`` per Hypothesis example; construction just needs a
    world the Policy can build a greedy initial decision from (ticket 13's
    constructor draw needs a non-empty ``clients_not_visited``).
    """
    return make_fleet_world({}, clients=[1, 2], vehicles=vehicle_count, nodes=[DEPOT, 1, 2, 9])


def _apply_endgame_config(world: World, config: dict[str, Any]) -> None:
    world.state.clients_not_visited[:] = [1, 2][: config["remaining"]]
    world.state.tau_episode = config["tau"]
    world.state.vehicle_position[:] = [
        DEPOT if at_depot else 9 for at_depot in config["at_depot"]
    ]


class TestEndgameInvariants:
    """Invariant catalogue (spec.md), ticket 05's two rows.

    "The Policy returns a legal action for every vehicle, at every tau, under
    every valid config" (B5) and "Two vehicles never receive the same
    non-depot Client in one decision" (B11). Both endgame branches (fewer than
    3 Clients left) are exercised directly, since that is where both findings
    live (``_select_vehicle_possible_actions``'s ``elif`` branch and
    ``_classify_shortest_distance_clients``).
    """

    @settings(max_examples=300, deadline=None, derandomize=True)
    @given(config=endgame_configs())
    def test_b5_never_raises_and_always_returns_a_legal_action(self, config: dict[str, Any]):
        vehicle_count = config["vehicle_count"]
        world = make_endgame_world(vehicle_count)
        policy = make_policy(world, W=np.zeros(19), vehicles=vehicle_count)

        _apply_endgame_config(world, config)
        features = state_features(policy)

        for vehicle in range(vehicle_count):
            actions = policy._select_vehicle_possible_actions(2, vehicle, features)
            assert actions, "the Policy must return at least one legal action"

    @settings(max_examples=300, deadline=None, derandomize=True)
    @given(config=endgame_configs())
    def test_b11_two_vehicles_never_get_the_same_non_depot_client(self, config: dict[str, Any]):
        vehicle_count = config["vehicle_count"]
        if vehicle_count < 2:
            return  # a duplicate booking needs at least two vehicles to collide
        world = make_endgame_world(vehicle_count)
        policy = make_policy(world, W=np.zeros(19), vehicles=vehicle_count)

        _apply_endgame_config(world, config)
        actions = policy.decide(world.state)

        non_depot = [action for action in actions if action != DEPOT]
        assert len(non_depot) == len(set(non_depot)), f"duplicate non-depot assignment: {actions}"
