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

from stdvrp.policies.network import QHead, TokenEncoder, _arc_dim0_index  # noqa: E402

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
    device: torch.device | None = None,
):
    rng = np.random.default_rng(seed)
    encoder = TokenEncoder(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_observed_velocities=n_obs,
        init_rng=rng,
        device=device,
    )
    head = QHead(d_model=d_model, init_rng=rng, device=device)
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
        is_depot = torch.zeros(n_pending)

        for v in range(number_vehicles):
            with torch.no_grad():
                q = _HEAD(embeddings.vehicles[v], embeddings.clients[:, v, :], claimed, is_depot)
            # Q is in the tokenizer's normalized units (raw minutes / horizon_length);
            # a positive constant scale, so exactness and argmin both still hold.
            horizon_length = float(SHIFT_END - HORIZON_START)
            expected = minutes_from_vehicle[v] / horizon_length

            np.testing.assert_allclose(q.numpy(), expected, atol=1e-5, rtol=1e-5)
            assert int(torch.argmin(q).item()) == int(np.argmin(minutes_from_vehicle[v]))


class TestWarmStartWeights:
    """Ticket 08: ``arc_embed`` row 0 is one of ``WARM_START_WEIGHTS``."""

    def test_cost_warm_start_prices_the_projected_cost_components(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=17)
        rng = np.random.default_rng(21)
        encoder = TokenEncoder(
            d_model=16,
            n_layers=3,
            n_heads=4,
            n_observed_velocities=N_OBS,
            init_rng=rng,
            warm_start="cost",
        )
        head = QHead(d_model=16, init_rng=rng)
        tokens = call_tokenize(geometry, time_windows, snapshot)
        embeddings = encoder(tokens)
        n_pending = tokens.client_tokens.shape[0]

        with torch.no_grad():
            q = head(
                embeddings.vehicles[0],
                embeddings.clients[:, 0, :],
                torch.zeros(n_pending),
                torch.zeros(n_pending),
            )

        # Read straight off the arc token: minutes + earliness + delay +
        # overtime (fields 0, 2, 3, 5), all already in 1/horizon_length units.
        # future_delay (field 4) is deliberately not priced.
        arc = tokens.arc_tokens[:, 0, :]
        expected = arc[:, 0] + arc[:, 2] + arc[:, 3] + arc[:, 5]
        np.testing.assert_allclose(q.numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_unknown_warm_start_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown warm_start"):
            TokenEncoder(
                d_model=16,
                n_layers=3,
                n_heads=4,
                n_observed_velocities=N_OBS,
                init_rng=np.random.default_rng(0),
                warm_start="distance",
            )


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
            is_depot = torch.zeros(tokens.client_tokens.shape[0])
            q_a = head_a(emb_a.vehicles[0], emb_a.clients[:, 0, :], claimed, is_depot)
            q_b = head_b(emb_b.vehicles[0], emb_b.clients[:, 0, :], claimed, is_depot)
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
        is_depot = torch.zeros(tokens.client_tokens.shape[0])
        with torch.no_grad():
            q_1 = _HEAD(emb_1.vehicles[0], emb_1.clients[:, 0, :], claimed, is_depot)
            q_2 = _HEAD(emb_2.vehicles[0], emb_2.clients[:, 0, :], claimed, is_depot)
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
        assert embeddings.depot.shape == (n_vehicles, 2 * _ENCODER.d_model)
        assert torch.isfinite(embeddings.clients).all()
        assert torch.isfinite(embeddings.vehicles).all()
        assert torch.isfinite(embeddings.depot).all()


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
    future edit zeroes QHead.layer1 too (not just layer2), those units would
    receive zero gradient forever. Verified over several seeds since a single
    dead ReLU unit on one batch is expected/normal -- the invariant is that
    gradient exists *somewhere*, not everywhere on every batch."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_layer2_columns_receive_gradient(self, seed: int) -> None:
        _, head = build_network(seed=seed, n_obs=N_OBS, d_model=8)
        n_pending = 20
        x = torch.randn(n_pending, head.layer1.in_features, requires_grad=False)
        hidden = torch.relu(head.layer1(x))
        q = head.layer2(hidden).squeeze(-1)
        loss = (q - torch.rand(n_pending)).pow(2).sum()
        loss.backward()

        assert head.layer2.weight.grad is not None
        assert (head.layer2.weight.grad.abs() > 0).any(), (
            "no hidden unit received gradient -- QHead's layer1 may have been "
            "accidentally zero-initialised (see module docstring)"
        )


class TestWarmStartIsNotBehindAnActivation:
    """The regression that cost ticket 08 its Gate A run: with the warm start
    living on row 0 of ``layer1``, one training episode drove that unit's
    pre-activation from ``[+0.003, +1.000]`` to ``[-0.546, -0.077]`` -- dead
    for every candidate. A dead ReLU has exactly zero gradient, so the myopic
    prior could never come back, ``Q``'s spread across candidates collapsed
    from 0.036 to 0.0007, and the argmin stopped picking the nearest Client.

    The invariant that makes that unreachable is not "the warm start survives
    training" (it is *meant* to be trainable away) but "no activation sits
    between the warm-start weights and ``Q``". Saturating the ReLU branch
    entirely is the sharpest way to state it: with the MLP branch contributing
    a hard zero and back-propagating nothing, the warm start must still be
    driving ``Q`` and must still be receiving gradient."""

    def test_the_warm_start_still_drives_q_when_the_relu_branch_is_dead(self) -> None:
        _, head = build_network(seed=11, n_obs=N_OBS, d_model=8)
        arc_index = _arc_dim0_index(8)
        with torch.no_grad():
            head.layer1.bias.fill_(-1.0e3)  # every hidden unit off, for any input

        x = torch.randn(16, head.layer1.in_features)
        x[:, arc_index] = torch.linspace(0.002, 0.06, 16)
        hidden = torch.relu(head.layer1(x))
        assert float(hidden.detach().abs().max()) == 0.0, "the branch is not saturated off"

        q = (head.linear(x) + head.layer2(hidden)).squeeze(-1)
        assert float(q.max() - q.min()) > 0.0, (
            "Q went constant once the ReLU branch died -- the warm start is behind "
            "an activation that can gate it off, which is what killed ticket 08's run"
        )

        q.sum().backward()
        assert head.linear.weight.grad is not None
        assert abs(float(head.linear.weight.grad[0, arc_index])) > 0.0, (
            "the warm-start weight receives no gradient with the ReLU branch dead -- "
            "it could never recover from an overshoot"
        )


class TestCostFeaturesAreWired:
    """The 2026-08-01 amendment's mechanism (spec.md decision 1; tokenizer.py,
    "The cost fields"): the four projected cost inputs are init-inert on Q
    itself (``TestWarmStart`` proves Q equals bare minutes at init, whatever
    the cost fields hold) but their gradient path must be live from the very
    first backward pass -- ``QHead.linear``'s warm-start weight (1.0 on arc
    dimension 0) times ``arc_embed`` row 0's cost columns. If this fails, the
    cost function was wired somewhere the warm start gates off, and A(s, a)
    is back to being rediscovered from noisy returns."""

    def test_arc_embed_cost_columns_receive_gradient_on_the_first_backward(self) -> None:
        encoder, head = build_network(seed=13, n_obs=N_OBS)
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=901)
        tokens = call_tokenize(geometry, time_windows, snapshot)
        assert (tokens.arc_tokens[:, 0, 2:] != 0.0).any(), (
            "precondition: this world must produce nonzero projected costs "
            "for vehicle 0 -- pick another seed"
        )

        embeddings = encoder(tokens)
        claimed = torch.zeros(tokens.client_tokens.shape[0])
        is_depot = torch.zeros(tokens.client_tokens.shape[0])
        q = head(embeddings.vehicles[0], embeddings.clients[:, 0, :], claimed, is_depot)
        q.sum().backward()

        grad = encoder.arc_embed.weight.grad
        assert grad is not None
        assert (grad[0, 2:].abs() > 0).any(), (
            "the projected cost inputs receive no gradient at init -- the warm "
            "start's row 0 should carry them into Q's gradient from step one"
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

        is_depot = torch.zeros(10)

        claimed_1 = torch.zeros(10, requires_grad=True)
        q_1 = head(vehicle_embedding, client_embeddings, claimed_1, is_depot)
        q_1.sum().backward()
        assert claimed_1.grad is not None
        assert torch.equal(claimed_1.grad, torch.zeros_like(claimed_1.grad)), (
            "claimed's gradient should be exactly zero on the first backward pass "
            "at init (see this class's docstring) -- if this now fails, the warm "
            "start's background-column zero-init may have changed"
        )
        optimizer.step()

        claimed_2 = torch.zeros(10, requires_grad=True)
        q_2 = head(vehicle_embedding, client_embeddings, claimed_2, is_depot)
        q_2.sum().backward()
        assert claimed_2.grad is not None
        assert (claimed_2.grad.abs() > 0).any(), (
            "claimed never affects Q's gradient even after an optimizer step -- "
            "it may have been dropped from QHead's forward pass"
        )


class TestDeviceParity:
    """Ticket 12: one forward, two devices -- the only cross-device equivalence
    this effort asserts. Full-trajectory equivalence is explicitly NOT asserted
    (a 1e-7 rounding difference flips the discrete argmin the decision rule
    takes elsewhere, and episodes diverge from there -- see
    transformer_policy.py's module docstring); this pins only that identical
    weights and identical input still produce the same numbers on CPU and CUDA
    for a single, isolated forward pass."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_identical_weights_and_input_give_allclose_q_on_cpu_and_cuda(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(5, 3, N_OBS, seed=77)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        cpu_encoder, cpu_head = build_network(seed=55, n_obs=N_OBS, device=torch.device("cpu"))
        cuda_encoder, cuda_head = build_network(seed=55, n_obs=N_OBS, device=torch.device("cuda"))

        with torch.no_grad():
            cpu_embeddings = cpu_encoder(tokens)
            cuda_embeddings = cuda_encoder(tokens)
            claimed = torch.zeros(tokens.client_tokens.shape[0])
            is_depot = torch.zeros(tokens.client_tokens.shape[0])
            q_cpu = cpu_head(
                cpu_embeddings.vehicles[0], cpu_embeddings.clients[:, 0, :], claimed, is_depot
            )
            q_cuda = cuda_head(
                cuda_embeddings.vehicles[0],
                cuda_embeddings.clients[:, 0, :],
                claimed.to("cuda"),
                is_depot.to("cuda"),
            )

        torch.testing.assert_close(q_cpu, q_cuda.cpu(), atol=1e-4, rtol=1e-4)
