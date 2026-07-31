"""Unit tests for :mod:`stdvrp.policies.network` (ticket 05, neural-policy).

``TestWarmStart`` is the ticket's real deliverable: the untrained network's
greedy policy must be nearest-feasible-Client, checked directly against the
geometry (spec.md, "The warm start"). ``TestReproducibility``/``TestDeterminism``
pin the injected-generator discipline and the bit-for-bit forward-pass claim.
``TestIdentityAtInit`` and ``TestQHeadBackgroundUnitsAreTrainable`` pin the two
non-obvious mechanisms the module docstring documents at length: a
zero-initialised residual branch is an exact identity, and QHead's un-warm-
started hidden units must not be frozen at zero forever.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Same circular-import landmine as test_tokenizer.py/test_torch_support.py
# (ticket 03's Comments): stdvrp.simulation must finish initializing first.
import stdvrp.simulation  # noqa: F401
from stdvrp.network.episode_geometry import EpisodeGeometry
from stdvrp.network.shortest_path_cache import ShortestPath, ShortestPathCache
from stdvrp.policies.tokenizer import tokenize
from stdvrp.simulation.state import TrainingSnapshot

torch = pytest.importorskip("torch")

from stdvrp.policies.network import QHead, TokenEncoder  # noqa: E402

pytestmark = pytest.mark.neural

DEPOT = 0
HORIZON_START = 300
SHIFT_END = 780
EPISODE_END = 1150


def make_cache(arcs: dict) -> ShortestPathCache:
    return ShortestPathCache(
        {
            key: ShortestPath([float(key[0]), float(key[1])], minutes, length)
            for key, (minutes, length) in arcs.items()
        }
    )


def make_dense_world(
    n_clients: int, n_vehicles: int, n_obs: int, seed: int
) -> tuple[EpisodeGeometry, dict, TrainingSnapshot]:
    """Same shape as test_tokenizer.py's helper of the same name."""
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

    positions = tuple(float(node) for node in rng.choice(rows, size=n_vehicles))
    standing = tuple(bool(flag) for flag in rng.integers(0, 2, size=n_vehicles))
    completing_service = tuple(float(flag) for flag in rng.integers(0, 2, size=n_vehicles))
    observed_velocity = tuple(
        tuple(float(v) for v in rng.uniform(0.0, 1.5, size=n_obs)) for _ in range(n_vehicles)
    )
    snapshot = TrainingSnapshot(
        tau_episode=float(rng.uniform(300.0, 900.0)),
        clients_not_visited=tuple(clients),
        last_node_reached=positions,
        vehicle_standing=standing,
        observed_velocity=observed_velocity,
        vehicle_completing_service=completing_service,
    )
    return geometry, time_windows, snapshot


def call_tokenize(geometry: EpisodeGeometry, time_windows: dict, snapshot: TrainingSnapshot):
    return tokenize(
        snapshot,
        geometry,
        time_windows,
        horizon_start_minute=HORIZON_START,
        shift_end_minute=SHIFT_END,
        episode_end_minute=EPISODE_END,
    )


def build_network(
    seed: int,
    *,
    d_model: int = 16,
    n_layers: int = 3,
    n_heads: int = 4,
    n_obs: int = 3,
):
    rng = np.random.default_rng(seed)
    encoder = TokenEncoder(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_observed_velocities=n_obs,
        init_rng=rng,
    )
    head = QHead(d_model=d_model, init_rng=rng)
    return encoder, head


N_OBS = 3


@st.composite
def worlds(draw: st.DrawFn) -> tuple[EpisodeGeometry, dict, TrainingSnapshot]:
    n_clients = draw(st.integers(1, 6))
    n_vehicles = draw(st.integers(1, 3))
    seed = draw(st.integers(0, 1_000_000))
    geometry, time_windows, snapshot = make_dense_world(n_clients, n_vehicles, N_OBS, seed)
    return geometry, time_windows, snapshot


# Built once (fixed init seed): the warm-start property is about the network
# holding for many random *states*, not about re-initialising per example.
_ENCODER, _HEAD = build_network(seed=1234, n_obs=N_OBS)


class TestWarmStart:
    """spec.md's load-bearing requirement: untrained Q ~= minutes_from_vehicle,
    so the untrained greedy policy is nearest-feasible-Client -- checked
    directly against ``EpisodeGeometry``, independently of the tokenizer's own
    output (a bug shared between ``tokenizer.py`` and ``network.py``, e.g. both
    transposing the same field, would not be caught by comparing against
    ``tokens.client_tokens`` alone)."""

    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(world=worlds())
    def test_q_matches_minutes_from_vehicle_and_argmin_is_nearest_client(self, world) -> None:
        geometry, time_windows, snapshot = world
        tokens = call_tokenize(geometry, time_windows, snapshot)
        embeddings = _ENCODER(tokens)

        # Independently recomputed from EpisodeGeometry -- not read off tokens.
        pending = list(snapshot.clients_not_visited)
        vehicle_minutes = geometry.average_minutes_rows(snapshot.last_node_reached)
        columns = geometry.column_positions(pending)
        minutes_from_vehicle = vehicle_minutes[:, columns]  # [number_vehicles, n_pending]

        n_pending = tokens.client_tokens.shape[0]
        number_vehicles = tokens.vehicle_tokens.shape[0]
        claimed = torch.zeros(n_pending)

        for v in range(number_vehicles):
            with torch.no_grad():
                q = _HEAD(embeddings.vehicles[v], embeddings.clients[:, v, :], claimed)
            # Q is in the tokenizer's normalized units (raw minutes / horizon_length);
            # a positive constant scale, so exactness and argmin both still hold.
            horizon_length = float(SHIFT_END - HORIZON_START)
            expected = minutes_from_vehicle[v] / horizon_length

            np.testing.assert_allclose(q.numpy(), expected, atol=1e-5, rtol=1e-5)
            assert int(torch.argmin(q).item()) == int(np.argmin(minutes_from_vehicle[v]))


class TestReproducibility:
    """Ticket 05: init draws from an injected generator, never a global."""

    def test_same_seed_gives_bit_identical_parameters(self) -> None:
        encoder_a, head_a = build_network(seed=42, n_obs=N_OBS)
        encoder_b, head_b = build_network(seed=42, n_obs=N_OBS)

        for (name_a, param_a), (name_b, param_b) in zip(
            encoder_a.named_parameters(), encoder_b.named_parameters(), strict=True
        ):
            assert name_a == name_b
            torch.testing.assert_close(param_a, param_b, atol=0.0, rtol=0.0)
        for (name_a, param_a), (name_b, param_b) in zip(
            head_a.named_parameters(), head_b.named_parameters(), strict=True
        ):
            assert name_a == name_b
            torch.testing.assert_close(param_a, param_b, atol=0.0, rtol=0.0)

    def test_same_seed_gives_bit_identical_output(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(5, 3, N_OBS, seed=99)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        encoder_a, head_a = build_network(seed=7, n_obs=N_OBS)
        encoder_b, head_b = build_network(seed=7, n_obs=N_OBS)
        with torch.no_grad():
            emb_a = encoder_a(tokens)
            emb_b = encoder_b(tokens)
            claimed = torch.zeros(tokens.client_tokens.shape[0])
            q_a = head_a(emb_a.vehicles[0], emb_a.clients[:, 0, :], claimed)
            q_b = head_b(emb_b.vehicles[0], emb_b.clients[:, 0, :], claimed)
        torch.testing.assert_close(q_a, q_b, atol=0.0, rtol=0.0)

    def test_different_seeds_give_different_parameters(self) -> None:
        encoder_a, _ = build_network(seed=1, n_obs=N_OBS)
        encoder_b, _ = build_network(seed=2, n_obs=N_OBS)

        # arc_embed's row 0 and every zero-initialised weight are identical by
        # construction regardless of seed; client_base_embed's Xavier-random
        # weight is not, so it is the discriminating parameter.
        assert not torch.equal(
            encoder_a.client_base_embed.weight, encoder_b.client_base_embed.weight
        )


class TestDeterminism:
    """Two forward passes of the same instance agree bit-for-bit (no dropout)."""

    def test_two_forward_passes_are_bit_identical(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=17)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        with torch.no_grad():
            emb_1 = _ENCODER(tokens)
            emb_2 = _ENCODER(tokens)
        torch.testing.assert_close(emb_1.clients, emb_2.clients, atol=0.0, rtol=0.0)
        torch.testing.assert_close(emb_1.vehicles, emb_2.vehicles, atol=0.0, rtol=0.0)

        claimed = torch.zeros(tokens.client_tokens.shape[0])
        with torch.no_grad():
            q_1 = _HEAD(emb_1.vehicles[0], emb_1.clients[:, 0, :], claimed)
            q_2 = _HEAD(emb_2.vehicles[0], emb_2.clients[:, 0, :], claimed)
        torch.testing.assert_close(q_1, q_2, atol=0.0, rtol=0.0)


class TestShapes:
    @pytest.mark.parametrize("n_clients", [1, 60, 180])
    @pytest.mark.parametrize("n_vehicles", [1, 4, 8])
    def test_embeddings_shapes(self, n_clients: int, n_vehicles: int) -> None:
        geometry, time_windows, snapshot = make_dense_world(
            n_clients, n_vehicles, N_OBS, seed=n_clients * 100 + n_vehicles
        )
        tokens = call_tokenize(geometry, time_windows, snapshot)
        with torch.no_grad():
            embeddings = _ENCODER(tokens)

        assert embeddings.clients.shape == (n_clients, n_vehicles, 2 * _ENCODER.d_model)
        assert embeddings.vehicles.shape == (n_vehicles, _ENCODER.d_model)
        assert torch.isfinite(embeddings.clients).all()
        assert torch.isfinite(embeddings.vehicles).all()


class TestIdentityAtInit:
    """Pins the mechanism the module docstring relies on: a zero-initialised
    residual branch makes nn.TransformerEncoder an exact identity at init."""

    def test_transformer_is_identity_for_arbitrary_input(self) -> None:
        d_model = _ENCODER.d_model
        sequence = torch.randn(1, 9, d_model)
        with torch.no_grad():
            output = _ENCODER.transformer(sequence)
        torch.testing.assert_close(output, sequence, atol=0.0, rtol=0.0)


class TestQHeadBackgroundUnitsAreTrainable:
    """Regression guard for the deadlock the module docstring calls out: if a
    future edit zeroes QHead.layer1's background rows (not just row 0), those
    units would receive zero gradient forever. Verified over several seeds
    since a single dead ReLU unit on one batch is expected/normal -- the
    invariant is that background gradient exists *somewhere*, not everywhere
    on every batch."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_layer2_background_columns_receive_gradient(self, seed: int) -> None:
        _, head = build_network(seed=seed, n_obs=N_OBS, d_model=8)
        n_pending = 20
        x = torch.randn(n_pending, head.layer1.in_features, requires_grad=False)
        hidden = torch.relu(head.layer1(x))
        q = head.layer2(hidden).squeeze(-1)
        loss = (q - torch.rand(n_pending)).pow(2).sum()
        loss.backward()

        assert head.layer2.weight.grad is not None
        background_grad = head.layer2.weight.grad[0, 1:]
        assert (background_grad.abs() > 0).any(), (
            "no background hidden unit received gradient -- QHead's background "
            "rows may have been accidentally zero-initialised (see module docstring)"
        )


class TestClaimedIsWired:
    """``claimed`` is deliberately init-inert for the warm start (row 0 -- the
    only hidden unit read by ``layer2`` at init -- has a zero weight on the
    ``claimed`` column, same as ``vehicle_embedding``/``client_context``, so Q
    exactly equals ``minutes_from_vehicle`` regardless of ``claimed``'s value).
    That must not mean ``claimed`` is a dead argument: the background rows
    (Xavier-random, per ``QHead._init_weights``) do read it -- but, exactly
    like ``layer1``'s background-row *weight* gradient (see the module
    docstring's deadlock note), the *input*-side gradient through those rows
    is also zero on the very first backward pass, since it is scaled by
    ``layer2``'s still-exactly-zero background columns. One optimizer step
    unlocks those columns; only from the second backward pass does gradient
    reach ``claimed``."""

    def test_claimed_gradient_is_zero_at_init_and_nonzero_after_one_step(self) -> None:
        _, head = build_network(seed=3, n_obs=N_OBS, d_model=8)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        vehicle_embedding = torch.randn(8)
        client_embeddings = torch.randn(10, 16)

        claimed_1 = torch.zeros(10, requires_grad=True)
        q_1 = head(vehicle_embedding, client_embeddings, claimed_1)
        q_1.sum().backward()
        assert claimed_1.grad is not None
        assert torch.equal(claimed_1.grad, torch.zeros_like(claimed_1.grad)), (
            "claimed's gradient should be exactly zero on the first backward pass "
            "at init (see this class's docstring) -- if this now fails, the warm "
            "start's background-column zero-init may have changed"
        )
        optimizer.step()

        claimed_2 = torch.zeros(10, requires_grad=True)
        q_2 = head(vehicle_embedding, client_embeddings, claimed_2)
        q_2.sum().backward()
        assert claimed_2.grad is not None
        assert (claimed_2.grad.abs() > 0).any(), (
            "claimed never affects Q's gradient even after an optimizer step -- "
            "it may have been dropped from QHead's forward pass"
        )
