"""Unit tests for :mod:`stdvrp.policies.tokenizer` (ticket 04, neural-policy).

``TestObservabilityRule`` is the ticket's real deliverable (spec.md, "The
observability rule, precisely"; ADR-0006): a structural pin, not a docstring,
that :func:`tokenize`'s signature and this module's own imports admit nothing
beyond ``State``/``TrainingSnapshot``, ``TimeWindows``, ``EpisodeGeometry`` and
the three config clocks — never ``EpisodeVelocities``, ``congested_arcs``,
``TravelTimeModel`` or ``FleetRoutes``. The remaining classes pin the token
arithmetic itself: one hand-computed example (catches a wrong field order or a
wrong scale immediately), then the three property tests the ticket asks for
(purity, permutation equivariance, variable set sizes).
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# stdvrp.policies.tokenizer imports stdvrp.policies.feature_extraction, and
# importing the stdvrp.policies package first triggers stdvrp.simulation to
# initialize mid-import — a pre-existing circular import (ticket 03, neural-policy
# Comments) that only resolves if stdvrp.simulation finishes initializing first.
# This is the first test module to import stdvrp.policies.tokenizer directly, so
# it must import stdvrp.simulation explicitly first, exactly like
# test_torch_support.py does for the same reason.
import stdvrp.simulation  # noqa: F401
from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies import tokenizer as tokenizer_module
from stdvrp.policies.tokenizer import tokenize
from stdvrp.simulation.state import TrainingSnapshot

DEPOT = 0
HORIZON_START = 300
SHIFT_END = 780
EPISODE_END = 1150


def make_cache(arcs: dict) -> ShortestPathCache:
    """arcs: (node, client) -> (average_minutes, length_km)."""
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


def make_snapshot(
    *,
    tau: float,
    pending: tuple[int, ...],
    positions: tuple[float, ...],
    standing: tuple[bool, ...],
    completing_service: tuple[float, ...],
    observed_velocity: tuple[tuple[float, ...], ...],
) -> TrainingSnapshot:
    return TrainingSnapshot(
        tau_episode=tau,
        clients_not_visited=pending,
        last_node_reached=positions,
        vehicle_standing=standing,
        observed_velocity=observed_velocity,
        vehicle_completing_service=completing_service,
    )


def call_tokenize(snapshot: TrainingSnapshot, geometry: EpisodeGeometry, time_windows: dict):
    return tokenize(
        snapshot,
        geometry,
        time_windows,
        horizon_start_minute=HORIZON_START,
        shift_end_minute=SHIFT_END,
        episode_end_minute=EPISODE_END,
    )


class TestObservabilityRule:
    """Pins spec.md's observability rule structurally (ADR-0006)."""

    FORBIDDEN_NAMES = frozenset(
        {
            "EpisodeVelocities",
            "congested_arcs",
            "TravelTimeModel",
            "FleetRoutes",
        }
    )
    FORBIDDEN_MODULE_PREFIXES = (
        "stdvrp.simulation.episode_velocities",
        "stdvrp.traffic.travel_time_model",
        "stdvrp.simulation.fleet_routes",
    )
    ALLOWED_PARAMETERS = frozenset(
        {
            "snapshot",
            "geometry",
            "time_windows",
            "horizon_start_minute",
            "shift_end_minute",
            "episode_end_minute",
        }
    )

    def _module_imports(self) -> tuple[set[str], set[str]]:
        """(imported names, imported module paths) from the tokenizer module's own source."""
        source_path = inspect.getsourcefile(tokenizer_module)
        assert source_path is not None
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
        names: set[str] = set()
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module)
                names.update(alias.name for alias in node.names)
        return names, modules

    def test_module_imports_nothing_forbidden(self) -> None:
        names, modules = self._module_imports()
        assert names.isdisjoint(self.FORBIDDEN_NAMES), names & self.FORBIDDEN_NAMES
        offending = {
            module
            for module in modules
            if any(module.startswith(prefix) for prefix in self.FORBIDDEN_MODULE_PREFIXES)
        }
        assert not offending, offending

    def test_signature_admits_only_the_permitted_inputs(self) -> None:
        parameters = set(inspect.signature(tokenize).parameters)
        # Pinned to the exact allow-list, not just "nothing forbidden": a future
        # ticket that "just adds" a congestion field must edit this test, not
        # slip past it unnoticed (the ticket's own stated failure mode).
        assert parameters == self.ALLOWED_PARAMETERS

    def test_no_parameter_or_annotation_names_a_forbidden_type(self) -> None:
        signature = inspect.signature(tokenize)
        for parameter in signature.parameters.values():
            annotation = str(parameter.annotation)
            for forbidden in self.FORBIDDEN_NAMES:
                assert forbidden not in annotation, (parameter.name, annotation)


class TestHandComputedExample:
    """One fully worked example: pins field order and the normalization scales.

    horizon_length = 780 - 300 = 480; episode_length = 1150 - 300 = 850. One
    vehicle at node 3 (away from the depot and both Clients, so every distance
    is nonzero and distinguishable), two pending Clients. The cache also prices
    every leg out of each Client and out of the depot, because the arc tokens'
    ``future_delay``/``overtime_cost`` fields price follow-up and homecoming
    legs (the 2026-08-01 amendment to spec.md decision 1).
    """

    def make_world(self) -> tuple[EpisodeGeometry, dict]:
        cache = make_cache(
            {
                (0, 0): (0.0, 0.0),
                (0, 1): (9.0, 4.0),
                (0, 2): (18.0, 7.0),
                (1, 0): (8.0, 4.0),
                (1, 1): (0.0, 0.0),
                (1, 2): (12.0, 6.0),
                (2, 0): (19.0, 7.5),
                (2, 1): (12.5, 6.0),
                (2, 2): (0.0, 0.0),
                (3, 0): (7.0, 3.5),
                (3, 1): (10.0, 5.0),
                (3, 2): (20.0, 8.0),
            }
        )
        geometry = EpisodeGeometry.build(cache, clients=[1, 2], depot=DEPOT)
        time_windows = {1: (350, 400), 2: (500, 600)}
        return geometry, time_windows

    def test_client_tokens(self) -> None:
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=320.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        expected = np.array(
            [
                [
                    (350 - 300) / 480,
                    (400 - 300) / 480,
                    (400 - 320) / 480,
                ],
                [
                    (500 - 300) / 480,
                    (600 - 300) / 480,
                    (600 - 320) / 480,
                ],
            ]
        )
        np.testing.assert_array_equal(tokens.client_tokens, expected)

    def test_arc_tokens_before_the_windows_open(self) -> None:
        """tau=320: both arrivals are early, nothing is late yet, no overtime."""
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=320.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        # Client 1: arrival 320+10=330 < 350 -> early by 20; follow-up to
        # Client 2 at 330+5+12=347 <= 600 -> no future delay; homecoming
        # 330+5+8=343 <= 780 -> no overtime. Client 2: arrival 340 < 500 ->
        # early by 160; follow-up to Client 1 at 340+5+12.5=357.5 <= 400;
        # homecoming 340+5+19=364.
        expected = np.array(
            [
                [[10.0 / 480, 5.0 / 480, (350 - 330.0) * 0.1 / 480, 0.0, 0.0, 0.0]],
                [[20.0 / 480, 8.0 / 480, (500 - 340.0) * 0.1 / 480, 0.0, 0.0, 0.0]],
            ]
        )
        np.testing.assert_array_equal(tokens.arc_tokens, expected)

    def test_arc_tokens_past_the_windows(self) -> None:
        """tau=760: both arrivals are late, follow-ups breach, one leg overtimes."""
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=760.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        # Both Clients are already past due (400/600 < 760), so delay is the
        # marginal past tau (FeatureExtractor's due_or_tau), never the sunk
        # part. Client 1: arrival 770 -> delay 10; follow-up to Client 2 at
        # 770+5+12=787 > 600 -> 787-760=27; homecoming 770+5+8=783 > 780 ->
        # (783-780)*5/6. Client 2: arrival 780 -> delay 20; follow-up to
        # Client 1 at 780+5+12.5=797.5 -> 37.5; homecoming 780+5+19=804 ->
        # (804-780)*5/6.
        expected = np.array(
            [
                [
                    [
                        10.0 / 480,
                        5.0 / 480,
                        0.0,
                        (770.0 - 760.0) * 1.0 / 480,
                        (787.0 - 760.0) * 1.0 / (480 * 2),
                        (783.0 - 780) * (5 / 6) / 480,
                    ]
                ],
                [
                    [
                        20.0 / 480,
                        8.0 / 480,
                        0.0,
                        (780.0 - 760.0) * 1.0 / 480,
                        (797.5 - 760.0) * 1.0 / (480 * 2),
                        (804.0 - 780) * (5 / 6) / 480,
                    ]
                ],
            ]
        )
        np.testing.assert_array_equal(tokens.arc_tokens, expected)

    def test_depot_arc_tokens(self) -> None:
        """The synthetic depot candidate: no window costs, follow-up legs out of
        the depot, overtime on the *direct* return leg (no service stop)."""
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=760.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        # Arrival home 760+7=767 <= 780 -> no overtime. Follow-ups out of the
        # depot: to Client 1 at 767+5+9=781 > 400 -> 781-760=21; to Client 2 at
        # 767+5+18=790 > 600 -> 790-760=30; summed 51.
        expected = np.array(
            [[7.0 / 480, 3.5 / 480, 0.0, 0.0, (21.0 + 30.0) / (480 * 2), 0.0]]
        )
        np.testing.assert_array_equal(tokens.depot_arc_tokens, expected)

    def test_depot_overtime_prices_the_direct_return_leg(self) -> None:
        """Past the shift end the reference clock is tau, and the depot's
        homecoming is ``tau + minutes_to_depot`` with no service stop."""
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=790.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        assert tokens.depot_arc_tokens[0, 5] == (797.0 - 790.0) * (5 / 6) / 480

    def test_vehicle_tokens(self) -> None:
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=320.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        expected = np.array([[1.0, 0.0, 7.0 / 480, 0.5, 0.6]])
        np.testing.assert_array_equal(tokens.vehicle_tokens, expected)

    def test_global_token(self) -> None:
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=320.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(True,),
            completing_service=(0.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        expected = np.array(
            [
                (320 - 300) / 480,
                (780 - 320) / 480,
                (1150 - 320) / 850,
                2 / 2,
                1.0,
            ]
        )
        np.testing.assert_array_equal(tokens.global_token, expected)

    def test_not_standing_and_completing_service_flip_their_columns(self) -> None:
        geometry, time_windows = self.make_world()
        snapshot = make_snapshot(
            tau=320.0,
            pending=(1, 2),
            positions=(3.0,),
            standing=(False,),
            completing_service=(1.0,),
            observed_velocity=((0.5, 0.6),),
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        assert tokens.vehicle_tokens[0, 0] == 0.0
        assert tokens.vehicle_tokens[0, 1] == 1.0


# --- property tests (spec.md's three, ticket 04) --------------------------------


@st.composite
def worlds(draw: st.DrawFn) -> tuple[EpisodeGeometry, dict, TrainingSnapshot]:
    """A small, randomized (geometry, time_windows, snapshot) triple.

    Dense over {depot} + this world's Clients: every (node, client) pair a
    vehicle could be tokenized against is priced, so row/column accessors never
    hit an absent cell (EpisodeGeometry's own assumption for those accessors).
    """
    n_clients = draw(st.integers(1, 6))
    n_vehicles = draw(st.integers(1, 3))
    n_obs = draw(st.integers(1, 3))
    clients = list(range(1, n_clients + 1))
    rows = [DEPOT, *clients]

    arcs = {}
    for row in rows:
        for client in [DEPOT, *clients]:
            arcs[(row, client)] = (
                draw(st.floats(0.5, 100.0, allow_nan=False, allow_infinity=False)),
                draw(st.floats(0.1, 50.0, allow_nan=False, allow_infinity=False)),
            )
    geometry = EpisodeGeometry.build(make_cache(arcs), clients=clients, depot=DEPOT)

    time_windows = {}
    for client in clients:
        start = draw(st.integers(300, 700))
        spread = draw(st.integers(10, 200))
        time_windows[client] = (start, start + spread)

    positions = tuple(float(draw(st.sampled_from(rows))) for _ in range(n_vehicles))
    standing = tuple(draw(st.booleans()) for _ in range(n_vehicles))
    completing_service = tuple(draw(st.sampled_from([0.0, 1.0])) for _ in range(n_vehicles))
    velocity = st.floats(0.0, 1.5, allow_nan=False, allow_infinity=False)
    observed_velocity = tuple(
        tuple(draw(velocity) for _ in range(n_obs)) for _ in range(n_vehicles)
    )
    tau = draw(st.floats(0.0, 2000.0, allow_nan=False, allow_infinity=False))
    pending = tuple(draw(st.permutations(clients)))

    snapshot = make_snapshot(
        tau=tau,
        pending=pending,
        positions=positions,
        standing=standing,
        completing_service=completing_service,
        observed_velocity=observed_velocity,
    )
    return geometry, time_windows, snapshot


class TestPureFunction:
    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(world=worlds())
    def test_same_snapshot_twice_is_bit_identical(self, world) -> None:
        geometry, time_windows, snapshot = world
        first = call_tokenize(snapshot, geometry, time_windows)
        second = call_tokenize(snapshot, geometry, time_windows)

        np.testing.assert_array_equal(first.client_tokens, second.client_tokens)
        np.testing.assert_array_equal(first.vehicle_tokens, second.vehicle_tokens)
        np.testing.assert_array_equal(first.global_token, second.global_token)
        np.testing.assert_array_equal(first.arc_tokens, second.arc_tokens)
        np.testing.assert_array_equal(first.depot_arc_tokens, second.depot_arc_tokens)


class TestPermutationEquivariance:
    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(world=worlds(), permutation_seed=st.integers(0, 1_000_000))
    def test_reordering_pending_clients_reorders_client_tokens_identically(
        self, world, permutation_seed: int
    ) -> None:
        geometry, time_windows, snapshot = world
        base = call_tokenize(snapshot, geometry, time_windows)

        pending = list(snapshot.clients_not_visited)
        permutation = np.random.default_rng(permutation_seed).permutation(len(pending))
        reordered_snapshot = replace(
            snapshot, clients_not_visited=tuple(pending[i] for i in permutation)
        )
        reordered = call_tokenize(reordered_snapshot, geometry, time_windows)

        np.testing.assert_array_equal(reordered.client_tokens, base.client_tokens[permutation])
        # Bit-exact even though future_delay sums over the *other* pending
        # Clients -- the tokenizer sums in sorted order precisely so this holds
        # (module docstring, last section).
        np.testing.assert_array_equal(reordered.arc_tokens, base.arc_tokens[permutation])
        np.testing.assert_array_equal(reordered.vehicle_tokens, base.vehicle_tokens)
        np.testing.assert_array_equal(reordered.global_token, base.global_token)
        np.testing.assert_array_equal(reordered.depot_arc_tokens, base.depot_arc_tokens)


def make_dense_world(
    n_clients: int, n_vehicles: int, n_obs: int, seed: int
) -> tuple[EpisodeGeometry, dict, TrainingSnapshot]:
    """A fully dense, vectorized world at real training scale (60/180 Clients)."""
    rng = np.random.default_rng(seed)
    clients = list(range(1, n_clients + 1))
    rows = [DEPOT, *clients]
    columns = [DEPOT, *clients]

    minutes = rng.uniform(0.5, 100.0, size=(len(rows), len(columns)))
    lengths = rng.uniform(0.1, 50.0, size=(len(rows), len(columns)))
    arcs = {
        (row, column): (float(minutes[r, c]), float(lengths[r, c]))
        for r, row in enumerate(rows)
        for c, column in enumerate(columns)
    }
    geometry = EpisodeGeometry.build(make_cache(arcs), clients=clients, depot=DEPOT)

    starts = rng.integers(300, 700, size=n_clients)
    spreads = rng.integers(10, 200, size=n_clients)
    time_windows = {
        client: (int(starts[i]), int(starts[i] + spreads[i])) for i, client in enumerate(clients)
    }

    positions = tuple(float(node) for node in rng.choice(rows, size=n_vehicles))
    standing = tuple(bool(flag) for flag in rng.integers(0, 2, size=n_vehicles))
    completing_service = tuple(float(flag) for flag in rng.integers(0, 2, size=n_vehicles))
    observed_velocity = tuple(
        tuple(float(v) for v in rng.uniform(0.0, 1.5, size=n_obs)) for _ in range(n_vehicles)
    )
    snapshot = make_snapshot(
        tau=float(rng.uniform(300.0, 900.0)),
        pending=tuple(clients),
        positions=positions,
        standing=standing,
        completing_service=completing_service,
        observed_velocity=observed_velocity,
    )
    return geometry, time_windows, snapshot


class TestVariableSetSizes:
    """spec.md's third property test: 60/180 Clients, 4/8 vehicles, well-formed tensors."""

    @pytest.mark.parametrize("n_clients", [60, 180])
    @pytest.mark.parametrize("n_vehicles", [4, 8])
    def test_well_formed_token_tensors(self, n_clients: int, n_vehicles: int) -> None:
        n_obs = 3
        geometry, time_windows, snapshot = make_dense_world(
            n_clients, n_vehicles, n_obs, seed=n_clients * 100 + n_vehicles
        )
        tokens = call_tokenize(snapshot, geometry, time_windows)

        assert tokens.client_tokens.shape == (n_clients, 3)
        assert tokens.vehicle_tokens.shape == (n_vehicles, 3 + n_obs)
        assert tokens.global_token.shape == (5,)
        assert tokens.arc_tokens.shape == (n_clients, n_vehicles, 6)
        assert tokens.depot_arc_tokens.shape == (n_vehicles, 6)
        assert np.isfinite(tokens.client_tokens).all()
        assert np.isfinite(tokens.vehicle_tokens).all()
        assert np.isfinite(tokens.global_token).all()
        assert np.isfinite(tokens.arc_tokens).all()
        assert np.isfinite(tokens.depot_arc_tokens).all()
        assert (tokens.arc_tokens[:, :, 2:] >= 0.0).all(), "projected costs are never negative"
        assert (tokens.depot_arc_tokens[:, 2:] >= 0.0).all()
