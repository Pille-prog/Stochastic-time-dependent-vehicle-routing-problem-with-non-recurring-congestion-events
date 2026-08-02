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
from stdvrp.policies.network import (  # noqa: E402
    DEPOT_WARM_START_PENALTY,
    QHead,
    TokenEncoder,
)
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
    device: torch.device | None = None,
    grad_clip_norm: float | None = None,
) -> TransformerMonteCarloPolicy:
    rng = np.random.default_rng(init_seed)
    encoder = TokenEncoder(
        d_model=D_MODEL,
        n_layers=2,
        n_heads=4,
        n_observed_velocities=N_OBS,
        init_rng=rng,
        device=device,
    )
    head = QHead(d_model=D_MODEL, init_rng=rng, device=device)
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
        device=device,
        grad_clip_norm=grad_clip_norm,
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


# --- A node the vehicle is on is not a travel destination (ticket 11, B20, ADR-0008) --


class TestSelfNodeNotACandidate:
    """ADR-0008: a node the vehicle is already on is not a travel destination.

    Only the depot has two meanings the simulator can tell apart (park vs.
    travel); for a pending Client there is nothing to travel to, so this is a
    Policy-side feasibility rule enforced by ``_sweep`` directly, in both the
    greedy argmin branch and the epsilon-exploration branch -- the depot is
    never filtered.
    """

    def test_greedy_excludes_the_vehicle_s_own_node_even_when_it_scores_lowest(self) -> None:
        """A hard mask, not a hope that the network never prefers it.

        ``_score`` is rigged to make the self-node candidate the global
        minimum -- if the exclusion were merely incidental (the network
        happening not to prefer it), this would expose that immediately.
        """
        geometry, time_windows, clients = make_world(4, seed=44)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1)
        own_node = clients[0]
        state = make_state(1, pending=clients, positions=[own_node])

        original_score = policy._score

        def rigged_score(embeddings, vehicle, pending, claimed_mask):
            q = original_score(embeddings, vehicle, pending, claimed_mask)
            q = q.clone()
            q[pending.index(own_node)] = -1e9
            return q

        policy._score = rigged_score

        action = policy.decide(state)

        assert action[0] != own_node

    def test_epsilon_exploration_excludes_the_vehicle_s_own_node(self) -> None:
        """The only pending Client is where the vehicle stands: the depot must win every time."""
        geometry, time_windows, clients = make_world(3, seed=45)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1, epsilon=1.0)
        own_node = clients[0]
        state = make_state(1, pending=[own_node], positions=[own_node])

        for trial in range(30):
            policy.exploration_rng = np.random.default_rng(trial)
            action = policy.decide_train(state)
            assert action[0] == DEPOT


class TestHomeIsAnExitNotADestination:
    """Ticket 14 (ADR-0011): the depot's place in the candidate set is now
    entirely ``select_vehicle_possible_actions``'s own branches, not a
    Policy-private ``_depot_is_feasible``/``_is_retired`` (both retired).
    ``tests/unit/test_action_set.py`` already pins the shared function's own
    branch arithmetic directly (the 350 cutoff, the shift-breach depot
    append, the no-Clients-left fallback); these tests pin that ``_sweep``
    actually wires it in end to end."""

    def test_epsilon_exploration_never_retires_a_vehicle_with_work_left(self) -> None:
        """The whole ε budget spent on one vehicle that has servable Clients: the
        depot must never come out of it, or exploration alone can park a fleet."""
        geometry, time_windows, clients = make_world(3, seed=46)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1, epsilon=1.0)
        state = make_state(1, pending=clients, positions=[DEPOT])

        seen = set()
        for trial in range(50):
            policy.exploration_rng = np.random.default_rng(trial)
            seen.add(policy.decide_train(state)[0])

        assert DEPOT not in seen
        assert seen == set(clients), "exploration should still reach every feasible Client"

    def test_a_parked_vehicle_past_the_350_cutoff_gets_the_depot_and_claims_nothing(
        self,
    ) -> None:
        """Branch 1 (``is_parked_at_depot and tau > 350``) offers only the depot
        -- no claim on a Client it cannot serve, so vehicle 1 must still be
        offered the Client nearest to it, not the runner-up."""
        geometry, time_windows, clients = make_world(4, seed=49)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients, positions=[DEPOT, clients[0]])
        state.vehicle_standing = [True, True]
        state.tau_episode = 351.0  # the literal branch 1 gates on

        action = policy.decide(state)

        assert action[0] == DEPOT
        nearest_to_one = min(
            (client for client in clients if client != clients[0]),
            key=lambda client: geometry.average_minutes(clients[0], client),
        )
        assert action[1] == nearest_to_one, "a retired vehicle's claim starved a Client"

    def test_a_parked_vehicle_before_the_350_cutoff_still_gets_a_real_candidate(
        self,
    ) -> None:
        """The retired ``_is_retired`` gated on ``horizon_start_minute``; the
        shared module gates on the literal ``350`` instead -- a vehicle parked
        at the depot between the two is now eligible for a real Client rather
        than forced home. Adopting the identical action set means adopting
        that literal too (this is why ticket 14 is a reversal of ADR-0007,
        not an amendment to it)."""
        geometry, time_windows, clients = make_world(4, seed=49)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1)
        state = make_state(1, pending=clients, positions=[DEPOT])
        state.vehicle_standing = [True]
        state.tau_episode = HORIZON_START + 2.0  # past the first epoch, still <= 350

        action = policy.decide(state)

        assert action[0] != DEPOT

    def test_the_whole_fleet_is_dispatchable_on_the_first_decision_epoch(self) -> None:
        """At ``tau == horizon_start_minute`` every vehicle stands on the depot
        un-dispatched; branch 1's literal ``350`` cutoff already keeps every
        vehicle out of the retired case without needing a first-epoch carve-out."""
        geometry, time_windows, clients = make_world(4, seed=50)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=clients, positions=[DEPOT, DEPOT, DEPOT])

        assert state.tau_episode == HORIZON_START
        assert DEPOT not in policy.decide(state)


class TestCurrentActionIsSelfAction:
    """Ticket 14 (ADR-0011): ``self.action`` *is* ``current_action`` -- the
    mechanism that closes ticket 08's ``claimed_mask`` defect. The old
    ``claimed_mask`` was rebuilt fresh every ``_sweep`` call, so a
    not-yet-processed vehicle's in-flight target from the *previous* decision
    epoch never excluded anything from an earlier vehicle's candidates.
    ``MonteCarloPolicy.action`` already avoids that (``Model.py``'s own
    comment: "``policy.action`` is mutated on every later decision"); this
    Policy now shares the same mechanism because it shares the same function.
    """

    def test_self_action_starts_at_the_depot_for_every_vehicle(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=60)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)

        assert policy.action == [DEPOT, DEPOT, DEPOT]

    def test_current_action_passed_to_the_shared_module_is_self_action(self) -> None:
        """A stale in-flight target set directly on ``policy.action`` (as if
        left there by the previous decision epoch) reaches
        ``select_vehicle_possible_actions`` as ``current_action`` for a vehicle
        processed *before* the one holding it -- proof the exclusion is not
        limited to vehicles already decided this pass, at the same ``m + 2``
        this ticket pins."""
        geometry, time_windows, clients = make_world(3, seed=61)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        stale_target = clients[0]
        policy.action = [DEPOT, stale_target]

        captured: dict[str, object] = {}
        original = transformer_policy_module.select_vehicle_possible_actions

        def spy(number_of_actions, vehicle, features, state_arg, current_action, *rest):
            if vehicle == 0:
                # Identity, not just equality: a callee that copied self.action
                # instead of aliasing it would still pass a value comparison.
                captured["is_self_action"] = current_action is policy.action
                captured["number_of_actions"] = number_of_actions
                captured["current_action"] = list(current_action)
            return original(number_of_actions, vehicle, features, state_arg, current_action, *rest)

        transformer_policy_module.select_vehicle_possible_actions = spy
        try:
            policy.decide(state)
        finally:
            transformer_policy_module.select_vehicle_possible_actions = original

        assert captured["is_self_action"] is True
        assert captured["current_action"] == [DEPOT, stale_target]
        assert captured["number_of_actions"] == policy.number_vehicles + 2


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
    """At init, ``Q(v, depot) == minutes_to_depot / horizon_length +
    DEPOT_WARM_START_PENALTY`` — one whole horizon above every Client, so the
    untrained greedy policy is "nearest feasible Client, home only when no
    Client is feasible" (network.py, "The depot is the last resort at init")."""

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(
        n_clients=st.integers(1, 6),
        n_vehicles=st.integers(1, 3),
        seed=st.integers(0, 1_000_000),
    )
    def test_depot_q_matches_minutes_to_depot_plus_one_horizon(
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
                q = policy._score(embeddings, vehicle, pending, claimed)
                expected = (
                    geometry.average_minutes(state.last_node_reached[vehicle], DEPOT)
                    / horizon_length
                ) + DEPOT_WARM_START_PENALTY
                np.testing.assert_allclose(q[len(pending)].item(), expected, atol=1e-4, rtol=1e-4)

    def test_a_vehicle_standing_on_the_depot_still_goes_to_the_nearest_client(self) -> None:
        """The Gate A regression: ``minutes(depot, depot) == 0`` used to make the
        depot the global argmin for every vehicle at decision epoch 1, which
        parked the whole fleet and ended the Episode after one transition."""
        geometry, time_windows, clients = make_world(5, seed=808)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3, epsilon=0.0)
        state = make_state(3, pending=clients, positions=[DEPOT, DEPOT, DEPOT])

        action = policy.decide(state)

        assert DEPOT not in action, "the untrained fleet retired at the depot instead of working"
        nearest = min(clients, key=lambda client: geometry.average_minutes(DEPOT, client))
        assert action[0] == nearest


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_learn_is_deterministic_on_cuda_with_the_same_seed(self) -> None:
        """Ticket 12: same-device determinism through a real backward pass on
        real CUDA hardware -- ticket 05's `use_deterministic_algorithms(True)`
        requirement (torch_support.resolve_device enables it for any CUDA use),
        tested against actual hardware for the first time (ticket 05 had none
        available). Enables it directly here since this test builds the network
        without going through resolve_device."""
        geometry, time_windows, clients = make_world(5, seed=23)
        was_enabled = torch.are_deterministic_algorithms_enabled()

        def run() -> list[torch.Tensor]:
            policy = build_policy(
                geometry,
                time_windows,
                clients,
                number_vehicles=2,
                epsilon=0.0,
                learn_seed=99,
                device=torch.device("cuda"),
            )
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=5)
            policy.learn(snapshots, actions, rewards)
            return [p.clone() for p in policy.encoder.parameters()]

        try:
            torch.use_deterministic_algorithms(True)
            first = run()
            second = run()
        finally:
            torch.use_deterministic_algorithms(was_enabled)

        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


class TestGradClip:
    """``neural_grad_clip_norm`` (ticket 08's stability-sweep lever): ``None``
    and a never-binding bound are bit-identical to the pre-knob behavior; a
    tight bound changes the step and stays deterministic."""

    def run_learn(self, clip: float | None) -> list[torch.Tensor]:
        geometry, time_windows, clients = make_world(5, seed=71)
        policy = build_policy(
            geometry,
            time_windows,
            clients,
            number_vehicles=2,
            epsilon=0.0,
            learn_seed=7,
            grad_clip_norm=clip,
        )
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=5)
        policy.learn(snapshots, actions, rewards)
        return [p.clone() for p in policy.encoder.parameters()] + [
            p.clone() for p in policy.head.parameters()
        ]

    def test_a_never_binding_bound_is_bit_identical_to_none(self) -> None:
        unclipped = self.run_learn(None)
        loose = self.run_learn(1.0e9)
        for a, b in zip(unclipped, loose, strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)

    def test_a_tight_bound_changes_the_step_and_is_deterministic(self) -> None:
        unclipped = self.run_learn(None)
        tight_1 = self.run_learn(1.0e-6)
        tight_2 = self.run_learn(1.0e-6)
        assert any(
            not torch.equal(a, b) for a, b in zip(unclipped, tight_1, strict=True)
        ), "a 1e-6 clip should visibly change the optimizer step"
        for a, b in zip(tight_1, tight_2, strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


class TestHuberDelta:
    """``neural_huber_delta`` (ticket 08): the loss's quadratic/linear knee.

    The regression's residuals live around ``1e-2``, so torch's default
    ``delta=1.0`` puts every one of them in the quadratic branch — the loss is
    exactly ``0.5 * MSE`` and a truncated episode's terminal penalty enters
    *squared*. These pin that the knob is actually plumbed through to the loss
    and that it does what a Huber knee does: bound the gradient of a large
    residual instead of scaling it."""

    def mean_loss(self, delta: float, target_scale: float) -> float:
        geometry, time_windows, clients = make_world(5, seed=71)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, learn_seed=7
        )
        policy.huber_delta = delta
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        # One big outlier return, the shape research note F10 describes.
        rewards = [reward * target_scale for reward in rewards]
        policy.learn(snapshots, actions, rewards)
        return float(policy.last_loss)

    def test_a_delta_below_the_residual_bounds_the_loss(self) -> None:
        big = 5000.0
        assert self.mean_loss(0.01, big) < self.mean_loss(1.0, big), (
            "a delta under the residual scale must put the outlier in the "
            "linear branch, which is strictly below the quadratic one there"
        )

    def test_the_default_delta_is_indistinguishable_from_having_no_knee(self) -> None:
        """The measurement behind the knob: at ``delta=1.0`` the loss is the
        same as at ``delta=1e6`` — *bit for bit*, on a real episode's targets.
        The default is not a mild Huber, it is plain ``0.5 * MSE``."""
        geometry, time_windows, clients = make_world(5, seed=71)

        def run(delta: float) -> list[torch.Tensor]:
            policy = build_policy(
                geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, learn_seed=7
            )
            policy.huber_delta = delta
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=4)
            policy.learn(snapshots, actions, rewards)
            return [p.clone() for p in policy.head.parameters()]

        for a, b in zip(run(1.0), run(1.0e6), strict=True):
            torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


# --- calibration_pairs() (ticket 08, Gate A) ------------------------------------


class TestCalibrationPairs:
    """(Q_joint, U_t) once per decision epoch -- Gate A's calibration primitive."""

    def test_one_pair_per_decision_epoch(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=31)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        pairs = policy.calibration_pairs(snapshots, actions, rewards)

        assert len(pairs) == 6

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
            assert pairs[t].realised_u == pytest.approx(targets[t])

    def test_predicted_half_matches_the_joint_replay(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=34)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=3)

        pairs = policy.calibration_pairs(snapshots, actions, rewards)

        for t in range(3):
            with torch.no_grad():
                expected = policy._replay_joint_q(snapshots[t], actions[t]).item()
            assert pairs[t].predicted_q == pytest.approx(expected)


class TestJointQIsAdditiveOverVehicles:
    """``learn`` regresses ``sum_v Q(s, v, a_v)`` — the same additive-over-vehicles
    shape ``MonteCarloPolicy``'s single joint feature vector already has, and what
    makes the coordinate-wise argmin in ``_sweep`` the matching decision rule."""

    def test_joint_q_is_the_sum_of_the_realized_per_vehicle_scores(self) -> None:
        geometry, time_windows, clients = make_world(6, seed=36)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=clients)
        snapshot = TrainingSnapshot.capture(state)
        action_row = policy.decide(state)

        with torch.no_grad():
            joint = policy._replay_joint_q(snapshot, action_row).item()

            from stdvrp.policies.tokenizer import tokenize

            tokens = tokenize(
                snapshot,
                geometry,
                time_windows,
                horizon_start_minute=HORIZON_START,
                shift_end_minute=SHIFT_END,
                episode_end_minute=EPISODE_END,
            )
            embeddings = policy.encoder(tokens)
            pending = list(snapshot.clients_not_visited)
            claimed = np.zeros(len(pending), dtype=bool)
            expected = 0.0
            for vehicle in range(3):
                q = policy._score(embeddings, vehicle, pending, claimed)
                chosen = action_row[vehicle]
                index = len(pending) if chosen == DEPOT else pending.index(chosen)
                expected += float(q[index].item())
                if chosen != DEPOT:
                    claimed[index] = True

        assert joint == pytest.approx(expected, rel=1e-5)

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
