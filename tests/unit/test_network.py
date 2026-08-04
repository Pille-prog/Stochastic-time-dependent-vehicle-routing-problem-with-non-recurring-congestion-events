"""Unit tests for :mod:`stdvrp.policies.network` (ticket 05, extended by
ticket 15, neural-policy).

``TestUntrainedQEqualsMyopicBase`` is ticket 15's real deliverable: the
untrained network's ``Q`` must equal the myopic base ``c`` exactly, for every
candidate including the depot, checked against ``tokens.arc_tokens`` directly
(module docstring, "The myopic base"). ``TestQHeadResidualIsZeroAtInit`` pins
the more fundamental claim underneath it: ``QHead``'s own output —
independent of any embeddings, any world, any geometry — is exactly zero at
construction. ``TestReproducibility``/``TestDeterminism`` pin the
injected-generator discipline and the bit-for-bit forward-pass claim.
``TestRealAttentionAtInit`` and ``TestQHeadBackgroundUnitsAreTrainable`` pin
two non-obvious mechanisms the module docstring documents at length: the
encoder is *not* an identity map any more (ticket 15 retired that
construction), and QHead's Xavier-random hidden units must not be frozen at
zero forever.
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
from stdvrp.policies.tokenizer import WARM_START_WEIGHTS, tokenize
from stdvrp.simulation.state import TrainingSnapshot

torch = pytest.importorskip("torch")

from stdvrp.policies.network import DEPOT_WARM_START_PENALTY, QHead, TokenEncoder  # noqa: E402

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
        # Irrelevant to tokenize()/the network (neither reads this field) --
        # same value as last_node_reached is a shape-correct, neutral filler.
        vehicle_destination=positions,
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


class TestQHeadResidualIsZeroAtInit:
    """Ticket 15's structural guarantee, at its most fundamental: ``QHead``'s
    own output -- ``W · φ(s, v, a)``, never the final ``Q`` any more -- is
    exactly zero at construction, for *any* input, not just ones built from a
    real tokenized world (``network.py``, "The myopic base"). Independent of
    ``TestUntrainedQEqualsMyopicBase`` below, which checks the sum this
    combines with; this class checks ``QHead`` in isolation."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_residual_is_exactly_zero_for_arbitrary_input(self, seed: int) -> None:
        _, head = build_network(seed=seed, n_obs=N_OBS, d_model=8)
        n_candidates = 12
        vehicle_embedding = torch.randn(8)
        client_embeddings = torch.randn(n_candidates, 16)
        claimed = torch.rand(n_candidates)
        is_depot = torch.zeros(n_candidates)
        is_depot[0] = 1.0

        with torch.no_grad():
            residual = head(vehicle_embedding, client_embeddings, claimed, is_depot)

        torch.testing.assert_close(residual, torch.zeros(n_candidates), atol=0.0, rtol=0.0)


class TestUntrainedQEqualsMyopicBase:
    """Ticket 15's real deliverable: at init, ``Q(s, v, a) == c(s, v, a)``
    exactly, for every pending Client and the synthetic depot candidate --
    checked against ``tokens.arc_tokens``/``depot_arc_tokens`` directly
    (module docstring, "The myopic base"), combining ``Embeddings.cost``/
    ``depot_cost`` with ``QHead``'s (zero) residual exactly as
    ``transformer_policy.py``'s ``_score`` does."""

    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(world=worlds())
    def test_q_equals_c_and_argmin_is_cost_greedy(self, world) -> None:
        geometry, time_windows, snapshot = world
        tokens = call_tokenize(geometry, time_windows, snapshot)
        embeddings = _ENCODER(tokens)

        n_pending = tokens.client_tokens.shape[0]
        number_vehicles = tokens.vehicle_tokens.shape[0]
        claimed = torch.zeros(n_pending)
        is_depot = torch.zeros(n_pending)

        weights = np.array(WARM_START_WEIGHTS["cost"])
        expected_client_cost = tokens.arc_tokens @ weights  # [n_pending, number_vehicles]
        expected_depot_cost = (
            tokens.depot_arc_tokens @ weights + DEPOT_WARM_START_PENALTY
        )  # [number_vehicles]

        for v in range(number_vehicles):
            with torch.no_grad():
                residual = _HEAD(
                    embeddings.vehicles[v], embeddings.clients[:, v, :], claimed, is_depot
                )
                q = embeddings.cost[:, v] + residual
                depot_q = embeddings.depot_cost[v] + _HEAD(
                    embeddings.vehicles[v],
                    embeddings.depot[v].unsqueeze(0),
                    torch.zeros(1),
                    torch.ones(1),
                )

            np.testing.assert_allclose(q.numpy(), expected_client_cost[:, v], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(
                depot_q.numpy(), expected_depot_cost[v : v + 1], atol=1e-5, rtol=1e-5
            )

            augmented = torch.cat([q, depot_q])
            expected_augmented = np.concatenate(
                [expected_client_cost[:, v], expected_depot_cost[v : v + 1]]
            )
            assert int(torch.argmin(augmented).item()) == int(np.argmin(expected_augmented))


class TestLevelGain:
    """Ticket 08: ``level_gain`` scales how far one optimizer step moves ``Q``'s
    overall level (``network.py``, "The level term, and why it needs a gain").

    Two properties carry the whole design. It must be **invisible to every
    argmin** — otherwise it is the dueling mistake again, a level correction
    landing on the ranking — and the default must be **bit-identical** to the
    term not existing, so the frozen null model and every prior measurement
    stand unchanged."""

    def head_with(self, gain: float):
        return QHead(d_model=16, init_rng=np.random.default_rng(4), level_gain=gain)

    def score(self, head, world):
        geometry, time_windows, snapshot = world
        encoder, _ = build_network(seed=4, n_obs=N_OBS)
        tokens = call_tokenize(geometry, time_windows, snapshot)
        embeddings = encoder(tokens)
        n_pending = tokens.client_tokens.shape[0]
        return head(
            embeddings.vehicles[0],
            embeddings.clients[:, 0, :],
            torch.zeros(n_pending),
            torch.zeros(n_pending),
        )

    def test_the_default_gain_is_bit_identical_to_no_level_term(self) -> None:
        world = make_dense_world(6, 2, N_OBS, seed=44)
        with torch.no_grad():
            plain = self.score(self.head_with(1.0), world)
            head = self.head_with(1.0)
            head.linear.bias.copy_(torch.full_like(head.linear.bias, 0.25))
            biased = self.score(head, world)
        # A nonzero bias still lands exactly once, not twice.
        torch.testing.assert_close(biased, plain + 0.25, atol=0.0, rtol=0.0)

    @settings(max_examples=25, deadline=None, derandomize=True)
    @given(world=worlds())
    def test_the_gain_shifts_every_candidate_equally(self, world) -> None:
        head = self.head_with(100.0)
        with torch.no_grad():
            before = self.score(head, world).clone()
            head.linear.bias.copy_(torch.full_like(head.linear.bias, -0.01))
            after = self.score(head, world)

        shift = (after - before).numpy()
        np.testing.assert_allclose(shift, np.full_like(shift, shift[0]), atol=1e-6, rtol=0)
        assert int(torch.argmin(after).item()) == int(torch.argmin(before).item())

    def test_the_gain_multiplies_the_bias_gradient(self) -> None:
        """The mechanism: one step moves the level ``gain`` times as far,
        because the bias's gradient is multiplied by exactly ``gain``."""
        world = make_dense_world(6, 2, N_OBS, seed=44)
        grads = []
        for gain in (1.0, 100.0):
            head = self.head_with(gain)
            self.score(head, world).sum().backward()
            assert head.linear.bias.grad is not None
            grads.append(float(head.linear.bias.grad.item()))
        assert grads[1] == pytest.approx(100.0 * grads[0], rel=1e-5)

    def test_a_nonpositive_gain_is_refused(self) -> None:
        with pytest.raises(ValueError, match="level_gain must be positive"):
            QHead(d_model=16, init_rng=np.random.default_rng(0), level_gain=0.0)


class TestMyopicBaseMatchesTheToken:
    """Ticket 15: ``Embeddings.cost``/``depot_cost`` are read straight off the
    arc token, bypassing ``arc_embed`` -- pinned in isolation from ``QHead``
    here (``TestUntrainedQEqualsMyopicBase`` above pins the combined ``Q``)."""

    def test_cost_prices_the_projected_cost_components(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=17)
        encoder, _ = build_network(seed=21, n_obs=N_OBS)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        with torch.no_grad():
            embeddings = encoder(tokens)

        # minutes + earliness + delay + overtime (fields 0, 2, 3, 5), all
        # already in 1/horizon_length units. future_delay (field 4) is
        # deliberately not priced.
        arc = tokens.arc_tokens
        expected_cost = arc[:, :, 0] + arc[:, :, 2] + arc[:, :, 3] + arc[:, :, 5]
        np.testing.assert_allclose(embeddings.cost.numpy(), expected_cost, atol=1e-5, rtol=1e-5)

        depot_arc = tokens.depot_arc_tokens
        expected_depot_cost = (
            depot_arc[:, 0] + depot_arc[:, 2] + depot_arc[:, 3] + depot_arc[:, 5]
        ) + DEPOT_WARM_START_PENALTY
        np.testing.assert_allclose(
            embeddings.depot_cost.numpy(), expected_depot_cost, atol=1e-5, rtol=1e-5
        )

    def test_cost_carries_no_gradient(self) -> None:
        """Not a ``Parameter``, no gradient path -- the structural claim
        ``network.py``'s "The myopic base" makes, checked directly rather
        than inferred from a backward pass never reaching it."""
        geometry, time_windows, snapshot = make_dense_world(4, 2, N_OBS, seed=18)
        encoder, _ = build_network(seed=22, n_obs=N_OBS)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        embeddings = encoder(tokens)

        assert embeddings.cost.requires_grad is False
        assert embeddings.depot_cost.requires_grad is False


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

        # Every zero-initialised weight (QHead.linear/layer2) is identical by
        # construction regardless of seed; client_base_embed's Xavier-random
        # weight is not (ticket 15: neither is arc_embed's, but this parameter
        # alone is enough to discriminate), so it is the discriminating one.
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
        assert embeddings.cost.shape == (n_clients, n_vehicles)
        assert embeddings.depot_cost.shape == (n_vehicles,)
        assert torch.isfinite(embeddings.clients).all()
        assert torch.isfinite(embeddings.vehicles).all()
        assert torch.isfinite(embeddings.depot).all()
        assert torch.isfinite(embeddings.cost).all()
        assert torch.isfinite(embeddings.depot_cost).all()


class TestRealAttentionAtInit:
    """Ticket 15 retired the identity-at-init construction ``TestIdentityAtInit``
    used to pin (module docstring, "Real attention at initialization"): nothing
    downstream needs the encoder to be an identity map any more, so
    ``out_proj``/``linear2`` are ordinary Xavier-random like every other
    weight, and the transformer performs real, nontrivial attention from the
    first forward pass."""

    def test_transformer_is_not_identity_for_arbitrary_input(self) -> None:
        d_model = _ENCODER.d_model
        sequence = torch.randn(1, 9, d_model)
        with torch.no_grad():
            output = _ENCODER.transformer(sequence)
        assert not torch.allclose(output, sequence), (
            "the transformer reproduced its input exactly -- has the "
            "identity-at-init construction ticket 15 retired come back?"
        )


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


class TestEncoderGradientAtInit:
    """Ticket 15's version of the old "is the myopic prior reachable" concern
    (formerly ``TestWarmStartIsNotBehindAnActivation``/``TestCostFeaturesAreWired``
    -- both pinned mechanisms specific to a warm start that no longer exists).
    ``c`` cannot be gated off by a dead ReLU any more because it is not routed
    through ``QHead`` at all (module docstring, "Why a dead ReLU can no longer
    touch `c`"); what is worth pinning instead is the residual's own gradient
    shape: with ``linear``/``layer2`` both exactly zero at init, gradient
    cannot flow *back* through them into the encoder on the first backward
    pass (their own gradient depends on their input, not the reverse -- the
    same "safe zero" pattern as ``layer1``/``layer2``'s deadlock, now covering
    every encoder parameter, not just ``layer1``'s rows) -- but it does, once
    one optimizer step moves them off zero."""

    def _one_vehicle_q(self, encoder, head, tokens):
        embeddings = encoder(tokens)
        claimed = torch.zeros(tokens.client_tokens.shape[0])
        is_depot = torch.zeros(tokens.client_tokens.shape[0])
        residual = head(embeddings.vehicles[0], embeddings.clients[:, 0, :], claimed, is_depot)
        return embeddings.cost[:, 0] + residual

    def test_no_gradient_reaches_the_encoder_on_the_first_backward_pass(self) -> None:
        encoder, head = build_network(seed=13, n_obs=N_OBS)
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=901)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        q = self._one_vehicle_q(encoder, head, tokens)
        q.sum().backward()

        grad = encoder.arc_embed.weight.grad
        assert grad is not None
        assert float(grad.abs().max()) == 0.0, (
            "the encoder received nonzero gradient on the first backward pass -- "
            "has QHead.linear/layer2 stopped being exactly zero at init?"
        )

    def test_gradient_reaches_the_encoder_after_one_optimizer_step(self) -> None:
        encoder, head = build_network(seed=13, n_obs=N_OBS)
        geometry, time_windows, snapshot = make_dense_world(6, 2, N_OBS, seed=901)
        tokens = call_tokenize(geometry, time_windows, snapshot)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)

        optimizer.zero_grad()
        self._one_vehicle_q(encoder, head, tokens).sum().backward()
        optimizer.step()

        encoder.zero_grad()
        self._one_vehicle_q(encoder, head, tokens).sum().backward()

        grad = encoder.arc_embed.weight.grad
        assert grad is not None
        assert (grad.abs() > 0).any(), (
            "the encoder still receives no gradient after an optimizer step -- "
            "layer2's columns may not have moved off zero"
        )


class TestClaimedIsWired:
    """``claimed`` is deliberately init-inert: ``QHead``'s residual is exactly
    zero at init regardless of any input (``TestQHeadResidualIsZeroAtInit``),
    ``claimed`` included. That must not mean ``claimed`` is a dead argument:
    the background rows (Xavier-random, per ``QHead._init_weights``) do read
    it -- but, exactly
    like ``layer1``'s background-row *weight* gradient (see the module
    docstring's deadlock note), the *input*-side gradient through those rows
    is also zero on the very first backward pass, since it is scaled by
    ``layer2``'s still-exactly-zero background columns (and, since ticket 15,
    ``linear``'s too -- see ``TestEncoderGradientAtInit`` above for the same
    mechanism pinned on the encoder side). One optimizer step unlocks those
    columns; only from the second backward pass does gradient reach
    ``claimed``."""

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
            "at init (see this class's docstring) -- if this now fails, QHead's "
            "zero-init (linear/layer2, ticket 15) may have changed"
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


class TestFeaturesAndWVector:
    """Ticket 16: ``QHead.features``/``w_vector``/``load_w_vector`` -- the
    decomposition :class:`~stdvrp.policies.ridge_estimator.RidgeAccumulator`
    regresses onto. The load-bearing claim is
    ``forward(...) == features(...) @ w_vector()``; everything else here is in
    service of that identity actually holding, both at init and after the
    weights it reads have moved.
    """

    def test_feature_dim_matches_the_head_shape(self) -> None:
        _, head = build_network(seed=1, n_obs=N_OBS, d_model=16)
        assert head.feature_dim == head.linear.in_features + head.layer1.out_features + 1

    def test_forward_equals_features_dot_w_vector_at_init(self) -> None:
        _, head = build_network(seed=2, n_obs=N_OBS, d_model=8)
        vehicle_embedding = torch.randn(8)
        client_embeddings = torch.randn(5, 16)
        claimed = torch.rand(5)
        is_depot = torch.zeros(5)

        with torch.no_grad():
            forward_output = head(vehicle_embedding, client_embeddings, claimed, is_depot)
            phi = head.features(vehicle_embedding, client_embeddings, claimed, is_depot)
            w = head.w_vector()

        assert phi.shape == (5, head.feature_dim)
        torch.testing.assert_close(forward_output, phi @ w, atol=1e-6, rtol=1e-6)

    def test_forward_equals_features_dot_w_vector_after_load_w_vector(self) -> None:
        _, head = build_network(seed=3, n_obs=N_OBS, d_model=8)
        rng = np.random.default_rng(3)
        w = torch.from_numpy(rng.normal(size=head.feature_dim).astype(np.float32))
        head.load_w_vector(w)

        vehicle_embedding = torch.randn(8)
        client_embeddings = torch.randn(6, 16)
        claimed = torch.rand(6)
        is_depot = torch.zeros(6)
        is_depot[0] = 1.0

        with torch.no_grad():
            forward_output = head(vehicle_embedding, client_embeddings, claimed, is_depot)
            phi = head.features(vehicle_embedding, client_embeddings, claimed, is_depot)

        torch.testing.assert_close(forward_output, phi @ w, atol=1e-5, rtol=1e-5)

    def test_load_w_vector_round_trips_through_w_vector(self) -> None:
        _, head = build_network(seed=4, n_obs=N_OBS, d_model=8)
        rng = np.random.default_rng(4)
        w = torch.from_numpy(rng.normal(size=head.feature_dim).astype(np.float32))

        head.load_w_vector(w)

        torch.testing.assert_close(head.w_vector(), w, atol=1e-6, rtol=1e-6)

    def test_load_w_vector_rejects_the_wrong_shape(self) -> None:
        _, head = build_network(seed=5, n_obs=N_OBS, d_model=8)
        with pytest.raises(ValueError, match="must have shape"):
            head.load_w_vector(torch.zeros(head.feature_dim + 1))

    def test_w_vector_is_zero_at_init(self) -> None:
        """``linear``/``layer2`` are exactly zero at construction (ticket 15) --
        the combined vector this reads off them must be too."""
        _, head = build_network(seed=6, n_obs=N_OBS, d_model=8)
        torch.testing.assert_close(
            head.w_vector(), torch.zeros(head.feature_dim), atol=0.0, rtol=0.0
        )

    def test_features_carries_no_gradient_requirement(self) -> None:
        """The ridge estimator never backpropagates through this -- a plain
        numeric readout, not part of an autograd graph the caller must
        remember to detach."""
        _, head = build_network(seed=7, n_obs=N_OBS, d_model=8)
        vehicle_embedding = torch.randn(8)
        client_embeddings = torch.randn(4, 16)
        claimed = torch.zeros(4)
        is_depot = torch.zeros(4)

        with torch.no_grad():
            phi = head.features(vehicle_embedding, client_embeddings, claimed, is_depot)

        assert phi.shape == (4, head.feature_dim)
        assert torch.isfinite(phi).all()


class TestDeviceParity:
    """Ticket 12: one forward, two devices -- the only cross-device equivalence
    this effort asserts. Full-trajectory equivalence is explicitly NOT asserted
    (a 1e-7 rounding difference flips the discrete argmin the decision rule
    takes elsewhere, and episodes diverge from there -- see
    transformer_policy.py's module docstring); this pins only that identical
    weights and identical input still produce the same numbers on CPU and CUDA
    for a single, isolated forward pass.

    Ticket 15: ``QHead``'s own output is exactly zero at init on every device
    trivially (nothing to compare), so ``embeddings.clients``/``vehicles``/
    ``depot`` -- now the output of *real* attention, not an identity map --
    carry the numerically meaningful cross-device comparison; ``q_cpu``/
    ``q_cuda`` (``embeddings.cost``/``depot_cost`` plus the residual) stays as
    the end-to-end check ``_score`` actually performs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_identical_weights_and_input_give_allclose_q_on_cpu_and_cuda(self) -> None:
        geometry, time_windows, snapshot = make_dense_world(5, 3, N_OBS, seed=77)
        tokens = call_tokenize(geometry, time_windows, snapshot)

        cpu_encoder, cpu_head = build_network(seed=55, n_obs=N_OBS, device=torch.device("cpu"))
        cuda_encoder, cuda_head = build_network(seed=55, n_obs=N_OBS, device=torch.device("cuda"))

        with torch.no_grad():
            cpu_embeddings = cpu_encoder(tokens)
            cuda_embeddings = cuda_encoder(tokens)

            torch.testing.assert_close(
                cpu_embeddings.clients, cuda_embeddings.clients.cpu(), atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                cpu_embeddings.vehicles, cuda_embeddings.vehicles.cpu(), atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                cpu_embeddings.depot, cuda_embeddings.depot.cpu(), atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                cpu_embeddings.cost, cuda_embeddings.cost.cpu(), atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                cpu_embeddings.depot_cost, cuda_embeddings.depot_cost.cpu(), atol=1e-4, rtol=1e-4
            )

            claimed = torch.zeros(tokens.client_tokens.shape[0])
            is_depot = torch.zeros(tokens.client_tokens.shape[0])
            q_cpu = cpu_embeddings.cost[:, 0] + cpu_head(
                cpu_embeddings.vehicles[0], cpu_embeddings.clients[:, 0, :], claimed, is_depot
            )
            q_cuda = cuda_embeddings.cost[:, 0] + cuda_head(
                cuda_embeddings.vehicles[0],
                cuda_embeddings.clients[:, 0, :],
                claimed.to("cuda"),
                is_depot.to("cuda"),
            )

        torch.testing.assert_close(q_cpu, q_cuda.cpu(), atol=1e-4, rtol=1e-4)
