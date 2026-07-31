"""Unit tests for :mod:`stdvrp.policies.transformer_policy` (ticket 06, neural-policy).

``TestFeasibility`` is the ticket's real deliverable (ADR-0007): B11 (no
double booking) and B5 (a legal action for every vehicle, every ``tau`` —
including zero pending Clients) hold by construction, checked directly rather
than trusted. ``TestOneEncoderPassPerDecisionEpoch`` pins the acceptance
criterion "one encoder pass per decision epoch, asserted — not ``m``".
``TestDepotWarmStart`` extends ticket 05's warm-start claim to the depot
candidate this module synthesizes (ADR-0007, "The depot's Q value").
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Same circular-import landmine as test_tokenizer.py/test_network.py (ticket 03's
# Comments): stdvrp.simulation must finish initializing first.
import stdvrp.simulation  # noqa: F401
from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.simulation.state import State, TrainingSnapshot

torch = pytest.importorskip("torch")

from stdvrp.policies import transformer_policy as transformer_policy_module  # noqa: E402
from stdvrp.policies.network import QHead, TokenEncoder  # noqa: E402
from stdvrp.policies.transformer_policy import TransformerMonteCarloPolicy  # noqa: E402

pytestmark = pytest.mark.neural

DEPOT = 0
HORIZON_START = 300
SHIFT_END = 780
EPISODE_END = 1150
N_OBS = 3
D_MODEL = 16


def make_cache(arcs: dict) -> ShortestPathCache:
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


def make_world(n_clients: int, seed: int) -> tuple[EpisodeGeometry, dict, list[int]]:
    """A dense geometry over ``n_clients`` Clients plus the depot, all arcs priced."""
    rng = np.random.default_rng(seed)
    clients = list(range(1, n_clients + 1))
    rows = [DEPOT, *clients]
    minutes = rng.uniform(0.5, 100.0, size=(len(rows), len(rows)))
    lengths = rng.uniform(0.1, 50.0, size=(len(rows), len(rows)))
    arcs = {
        (row, column): (float(minutes[r, c]), float(lengths[r, c]))
        for r, row in enumerate(rows)
        for c, column in enumerate(rows)
    }
    geometry = EpisodeGeometry.build(make_cache(arcs), clients=clients, depot=DEPOT)

    starts = rng.integers(300, 700, size=n_clients)
    spreads = rng.integers(10, 200, size=n_clients)
    time_windows = {
        client: (int(starts[i]), int(starts[i] + spreads[i])) for i, client in enumerate(clients)
    }
    return geometry, time_windows, clients


def build_policy(
    geometry: EpisodeGeometry,
    time_windows: dict,
    clients: list[int],
    number_vehicles: int,
    *,
    init_seed: int = 0,
    epsilon: float = 0.0,
    exploration_seed: int = 1,
    learn_seed: int = 2,
    learn_passes: int = 1,
    batch_size: int = 4,
) -> TransformerMonteCarloPolicy:
    rng = np.random.default_rng(init_seed)
    encoder = TokenEncoder(
        d_model=D_MODEL, n_layers=2, n_heads=4, n_observed_velocities=N_OBS, init_rng=rng
    )
    head = QHead(d_model=D_MODEL, init_rng=rng)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    return TransformerMonteCarloPolicy(
        number_vehicles=number_vehicles,
        geometry=geometry,
        time_windows=time_windows,
        number_clients=len(clients),
        epsilon=epsilon,
        depot=DEPOT,
        shift_end_minute=SHIFT_END,
        episode_end_minute=EPISODE_END,
        horizon_start_minute=HORIZON_START,
        encoder=encoder,
        head=head,
        optimizer=optimizer,
        exploration_rng=np.random.default_rng(exploration_seed),
        learn_rng=np.random.default_rng(learn_seed),
        learn_passes=learn_passes,
        batch_size=batch_size,
    )


def make_state(
    number_vehicles: int,
    pending: list[int],
    n_obs: int = N_OBS,
    positions: list[float] | None = None,
) -> State:
    state = State(number_vehicles, list(pending), n_obs, HORIZON_START, DEPOT)
    if positions is not None:
        state.last_node_reached = list(positions)
    return state


# --- Feasibility: B11 (no double booking) and B5 (always a legal action) ------


@st.composite
def sweep_worlds(draw: st.DrawFn) -> tuple[EpisodeGeometry, dict, list[int], int, list[int]]:
    n_clients = draw(st.integers(0, 8))
    n_vehicles = draw(st.integers(1, 5))
    seed = draw(st.integers(0, 1_000_000))
    geometry, time_windows, clients = make_world(max(n_clients, 1), seed)
    pending = clients[:n_clients]  # exercise 0..n_clients pending, including empty
    return geometry, time_windows, clients, n_vehicles, pending


class TestFeasibility:
    """ADR-0007: feasibility is a hard constraint on the argmin, not a heuristic bias."""

    @settings(max_examples=60, deadline=None, derandomize=True)
    @given(world=sweep_worlds())
    def test_b11_no_two_vehicles_get_the_same_non_depot_client(self, world) -> None:
        geometry, time_windows, clients, n_vehicles, pending = world
        policy = build_policy(geometry, time_windows, clients, n_vehicles)
        state = make_state(n_vehicles, pending)

        action = policy.decide(state)

        non_depot = [a for a in action if a != DEPOT]
        assert len(non_depot) == len(set(non_depot)), "B11 violated: a Client was double-booked"
        assert len(action) == n_vehicles

    @settings(max_examples=60, deadline=None, derandomize=True)
    @given(world=sweep_worlds())
    def test_b5_every_vehicle_gets_a_legal_action(self, world) -> None:
        geometry, time_windows, clients, n_vehicles, pending = world
        policy = build_policy(geometry, time_windows, clients, n_vehicles)
        state = make_state(n_vehicles, pending)

        action = policy.decide(state)

        legal = set(pending) | {DEPOT}
        assert all(a in legal for a in action)

    def test_zero_pending_clients_falls_back_to_depot_for_every_vehicle(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=5)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=[])

        action = policy.decide(state)

        assert action == [DEPOT, DEPOT, DEPOT]

    def test_decide_train_also_respects_b11(self) -> None:
        geometry, time_windows, clients = make_world(3, seed=9)
        # epsilon=1.0: every vehicle explores, the hardest case for the mask.
        policy = build_policy(geometry, time_windows, clients, number_vehicles=4, epsilon=1.0)
        state = make_state(4, pending=clients)

        for trial in range(20):
            policy.exploration_rng = np.random.default_rng(trial)
            action = policy.decide_train(state)
            non_depot = [a for a in action if a != DEPOT]
            assert len(non_depot) == len(set(non_depot))


# --- One encoder pass per decision epoch --------------------------------------


class TestOneEncoderPassPerDecisionEpoch:
    def test_decide_calls_the_transformer_exactly_once(self) -> None:
        geometry, time_windows, clients = make_world(6, seed=11)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=5)
        state = make_state(5, pending=clients)

        calls = {"count": 0}
        original_forward = policy.encoder.transformer.forward

        def counting_forward(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            return original_forward(*args, **kwargs)

        policy.encoder.transformer.forward = counting_forward
        policy.decide(state)

        assert calls["count"] == 1

    def test_decide_train_calls_the_transformer_exactly_once(self) -> None:
        geometry, time_windows, clients = make_world(6, seed=12)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=5, epsilon=0.3)
        state = make_state(5, pending=clients)

        calls = {"count": 0}
        original_forward = policy.encoder.transformer.forward

        def counting_forward(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            return original_forward(*args, **kwargs)

        policy.encoder.transformer.forward = counting_forward
        policy.decide_train(state)

        assert calls["count"] == 1


# --- The depot's warm start ----------------------------------------------------


class TestDepotWarmStart:
    """ADR-0007: at init, Q(v, depot) == minutes_to_depot / horizon_length exactly."""

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(
        n_clients=st.integers(1, 6),
        n_vehicles=st.integers(1, 3),
        seed=st.integers(0, 1_000_000),
    )
    def test_depot_q_matches_minutes_to_depot(
        self, n_clients: int, n_vehicles: int, seed: int
    ) -> None:
        geometry, time_windows, clients = make_world(n_clients, seed)
        policy = build_policy(geometry, time_windows, clients, n_vehicles, init_seed=seed)
        state = make_state(n_vehicles, pending=clients)

        horizon_length = float(SHIFT_END - HORIZON_START)
        with torch.no_grad():
            from stdvrp.policies.tokenizer import tokenize

            tokens = tokenize(
                state,
                geometry,
                time_windows,
                horizon_start_minute=HORIZON_START,
                shift_end_minute=SHIFT_END,
                episode_end_minute=EPISODE_END,
            )
            embeddings = policy.encoder(tokens)
            for vehicle in range(n_vehicles):
                pending = list(state.clients_not_visited)
                claimed = np.zeros(len(pending), dtype=bool)
                q = policy._score(
                    embeddings, vehicle, state.last_node_reached[vehicle], pending, claimed
                )
                expected = (
                    geometry.average_minutes(state.last_node_reached[vehicle], DEPOT)
                    / horizon_length
                )
                np.testing.assert_allclose(q[len(pending)].item(), expected, atol=1e-4, rtol=1e-4)


# --- learn() -------------------------------------------------------------------


def make_episode(policy: TransformerMonteCarloPolicy, state: State, length: int):
    snapshots = []
    actions = []
    rewards = [0.0]
    for _ in range(length):
        snapshots.append(TrainingSnapshot.capture(state))
        action = policy.decide_train(state)
        actions.append(action)
        rewards.append(-1.0)
    return snapshots, actions, rewards


class TestLearn:
    def test_learn_moves_the_weights(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=21)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        before = [p.clone() for p in policy.encoder.parameters()] + [
            p.clone() for p in policy.head.parameters()
        ]
        assert policy.last_loss == 0.0
        policy.learn(snapshots, actions, rewards)
        after = list(policy.encoder.parameters()) + list(policy.head.parameters())

        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
        assert policy.last_loss > 0.0
        for param in after:
            assert torch.isfinite(param).all()

    def test_learn_on_empty_episode_is_a_no_op(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=22)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)

        before = [p.clone() for p in policy.encoder.parameters()]
        policy.learn([], [], [0.0])
        after = list(policy.encoder.parameters())

        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_learn_is_deterministic_given_the_same_learn_rng_seed(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=23)

        def run():
            policy = build_policy(
                geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, learn_seed=99
            )
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=5)
            policy.learn(snapshots, actions, rewards)
            return [p.clone() for p in policy.encoder.parameters()]

        first = run()
        second = run()
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


# --- calibration_pairs() (ticket 08, Gate A) ------------------------------------


class TestCalibrationPairs:
    """(Q_predicted, U_t) per (epoch, vehicle) -- Gate A's calibration primitive."""

    def test_pairs_length_matches_epochs_times_vehicles(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=31)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        pairs = policy.calibration_pairs(snapshots, actions, rewards)

        assert len(pairs) == 6 * 2

    def test_empty_episode_returns_no_pairs(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=32)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)

        assert policy.calibration_pairs([], [], [0.0]) == []

    def test_realised_half_matches_backward_returns(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=33)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        targets = policy._backward_returns(snapshots, actions, rewards)

        pairs = policy.calibration_pairs(snapshots, actions, rewards)

        for t in range(4):
            for vehicle in range(2):
                _, u = pairs[t * 2 + vehicle]
                assert u == pytest.approx(targets[t])

    def test_predicted_half_matches_replay_q(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=34)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=3)

        pairs = policy.calibration_pairs(snapshots, actions, rewards)

        for t in range(3):
            for vehicle in range(2):
                q_pred, _ = pairs[t * 2 + vehicle]
                with torch.no_grad():
                    expected = policy._replay_q(snapshots[t], actions[t], vehicle).item()
                assert q_pred == pytest.approx(expected)

    def test_does_not_mutate_the_network(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=35)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        before = [p.clone() for p in policy.encoder.parameters()] + [
            p.clone() for p in policy.head.parameters()
        ]

        policy.calibration_pairs(snapshots, actions, rewards)

        after = list(policy.encoder.parameters()) + list(policy.head.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))
        assert policy.last_loss == 0.0


# --- monte_carlo.py stays untouched ---------------------------------------------


class TestDoesNotDependOnMonteCarlo:
    """ADR-0007: this Policy must not import the frozen linear baseline.

    Checks the module's actual ``import``/``from ... import`` statements, not
    its prose — the module docstring names ``monte_carlo.py`` freely to
    explain *why* it isn't imported.
    """

    def test_module_does_not_import_monte_carlo(self) -> None:
        source_path = inspect.getsourcefile(transformer_policy_module)
        assert source_path is not None
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any("monte_carlo" in module for module in imported_modules)
