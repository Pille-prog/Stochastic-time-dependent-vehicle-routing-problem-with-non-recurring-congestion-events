"""Unit tests for :mod:`stdvrp.policies.transformer_policy` (ticket 06, neural-policy).

``TestFeasibility`` is the ticket's real deliverable (ADR-0007): B11 (no
double booking) and B5 (a legal action for every vehicle, every ``tau`` —
including zero pending Clients) hold by construction, checked directly rather
than trusted. ``TestOneEncoderPassPerDecisionEpoch`` pins the acceptance
criterion "one encoder pass per decision epoch, asserted — not ``m``".
``TestDepotMyopicBase`` extends ticket 15's myopic-base claim to the depot
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
from stdvrp.policies.ridge_estimator import RidgeAccumulator  # noqa: E402
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
    solve_cadence: int = 1,
    forgetting: float = 1.0,
    ridge: float = 1.0,
    device: torch.device | None = None,
    train_encoder: bool = False,
    learn_seed: int = 2,
    learn_passes: int = 1,
    batch_size: int = 32,
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
    accumulator = RidgeAccumulator.zeros(head.feature_dim, forgetting=forgetting, ridge=ridge)
    encoder_optimizer = (
        torch.optim.Adam([*encoder.parameters(), *head.layer1.parameters()], lr=1e-3)
        if train_encoder
        else None
    )
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
        ridge=accumulator,
        exploration_rng=np.random.default_rng(exploration_seed),
        solve_cadence=solve_cadence,
        device=device,
        train_encoder=train_encoder,
        encoder_optimizer=encoder_optimizer,
        learn_rng=np.random.default_rng(learn_seed) if train_encoder else None,
        learn_passes=learn_passes,
        batch_size=batch_size,
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

        ``_sweep`` (via ``_score_with_components``, ticket 17) is rigged to
        make the self-node candidate the global minimum -- if the exclusion
        were merely incidental (the network happening not to prefer it),
        this would expose that immediately. Rigs ``_score_with_components``
        directly (what ``_sweep`` actually calls) rather than ``_score`` --
        the thin ``(embeddings, vehicle, pending, claimed_mask) -> q`` wrapper
        ``_sweep`` no longer calls internally, kept only for direct callers
        elsewhere in this file.
        """
        geometry, time_windows, clients = make_world(4, seed=44)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1)
        own_node = clients[0]
        state = make_state(1, pending=clients, positions=[own_node])

        original_score_with_components = policy._score_with_components

        def rigged_score_with_components(embeddings, vehicle, pending, claimed_mask):
            q, residual, myopic_base = original_score_with_components(
                embeddings, vehicle, pending, claimed_mask
            )
            q = q.clone()
            q[pending.index(own_node)] = -1e9
            return q, residual, myopic_base

        policy._score_with_components = rigged_score_with_components

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


# --- The depot's myopic base ----------------------------------------------------


class TestDepotMyopicBase:
    """At init, ``Q(v, depot) == c(s, v, depot)`` exactly (ticket 15,
    ``network.py``, "The depot's place in `c`") -- the tokenizer's projected
    cost for the return leg, plus :data:`DEPOT_WARM_START_PENALTY` (one whole
    horizon), so the untrained greedy policy is "nearest feasible Client, home
    only when no Client is feasible"."""

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(
        n_clients=st.integers(1, 6),
        n_vehicles=st.integers(1, 3),
        seed=st.integers(0, 1_000_000),
    )
    def test_depot_q_matches_the_myopic_base_plus_one_horizon(
        self, n_clients: int, n_vehicles: int, seed: int
    ) -> None:
        geometry, time_windows, clients = make_world(n_clients, seed)
        policy = build_policy(geometry, time_windows, clients, n_vehicles, init_seed=seed)
        state = make_state(n_vehicles, pending=clients)

        with torch.no_grad():
            from stdvrp.policies.tokenizer import WARM_START_WEIGHTS, tokenize

            tokens = tokenize(
                state,
                geometry,
                time_windows,
                horizon_start_minute=HORIZON_START,
                shift_end_minute=SHIFT_END,
                episode_end_minute=EPISODE_END,
            )
            embeddings = policy.encoder(tokens)
            weights = np.array(WARM_START_WEIGHTS["cost"])
            expected_depot_cost = tokens.depot_arc_tokens @ weights + DEPOT_WARM_START_PENALTY
            for vehicle in range(n_vehicles):
                pending = list(state.clients_not_visited)
                claimed = np.zeros(len(pending), dtype=bool)
                q = policy._score(embeddings, vehicle, pending, claimed)
                np.testing.assert_allclose(
                    q[len(pending)].item(), expected_depot_cost[vehicle], atol=1e-4, rtol=1e-4
                )

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


# --- _score_with_components() / spread_samples (ticket 17, Gate A') -------------


class TestScoreWithComponents:
    """``_score`` is a thin wrapper over ``_score_with_components`` (ticket 17):
    same signature, same return, just split apart so ``_sweep`` can read the
    residual/myopic-base halves for the ``r`` diagnostic without recomputing
    either."""

    def test_score_equals_the_q_component_of_score_with_components(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=70)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)

        from stdvrp.policies.tokenizer import tokenize

        tokens = tokenize(
            state,
            geometry,
            time_windows,
            horizon_start_minute=HORIZON_START,
            shift_end_minute=SHIFT_END,
            episode_end_minute=EPISODE_END,
        )
        with torch.no_grad():
            embeddings = policy.encoder(tokens)
            pending = list(state.clients_not_visited)
            claimed = np.zeros(len(pending), dtype=bool)

            q = policy._score(embeddings, 0, pending, claimed)
            q_component, residual, myopic_base = policy._score_with_components(
                embeddings, 0, pending, claimed
            )

        torch.testing.assert_close(q, q_component)
        torch.testing.assert_close(q_component, myopic_base + residual)


class TestSpreadSamples:
    """Ticket 17's ``r`` diagnostic raw material: ``(sd_candidates(W.phi),
    sd_candidates(c))``, accumulated by every greedy ``_sweep`` call
    (``decide()``, always greedy; ``decide_train()`` whenever the epsilon gate
    does not fire), restricted to the vehicle's actual candidate set."""

    def test_decide_appends_one_sample_per_vehicle(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=71)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=clients)

        assert policy.spread_samples == []
        policy.decide(state)

        assert len(policy.spread_samples) == 3
        for residual_sd, cost_sd in policy.spread_samples:
            assert residual_sd >= 0.0
            assert cost_sd >= 0.0

    def test_spread_samples_reset_every_fresh_policy_instance(self) -> None:
        """A fresh ``TransformerMonteCarloPolicy`` wraps the same long-lived
        network every Episode (class docstring) -- ``spread_samples`` must
        never leak across that boundary."""
        geometry, time_windows, clients = make_world(4, seed=72)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        policy.decide(state)
        assert policy.spread_samples

        fresh_policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        assert fresh_policy.spread_samples == []

    def test_no_sample_when_only_one_candidate_survives(self) -> None:
        """Undefined (and uninformative) with fewer than two candidates --
        this vehicle's only legal move is the depot (past the 350 cutoff,
        parked, module docstring "ADR-0011")."""
        geometry, time_windows, clients = make_world(4, seed=73)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=1)
        state = make_state(1, pending=clients, positions=[DEPOT])
        state.vehicle_standing = [True]
        state.tau_episode = 351.0  # branch 1's literal cutoff: depot-only

        policy.decide(state)

        assert policy.spread_samples == []

    def test_decide_train_greedy_branch_also_appends_samples(self) -> None:
        """epsilon=0 forces every vehicle through the greedy branch of
        ``_sweep``, exactly like ``decide`` -- "decide()-time candidate
        spread" (ticket 16's own words) is not literally limited to the
        ``decide`` method, only to the greedy code path."""
        geometry, time_windows, clients = make_world(5, seed=74)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        state = make_state(2, pending=clients)

        policy.decide_train(state)

        assert len(policy.spread_samples) == 2


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
    """Ticket 16: ``learn`` folds one Episode into the ridge accumulator and
    re-solves on cadence -- no optimizer, no loss, no minibatch shuffle, and
    (on the frozen-encoder arm this ticket ships) no movement of ``encoder``
    or ``head.layer1`` at all."""

    def test_learn_solves_and_moves_only_head_linear_and_layer2(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=21)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        encoder_before = [p.clone() for p in policy.encoder.parameters()]
        layer1_before = [p.clone() for p in policy.head.layer1.parameters()]
        assert policy.last_loss == 0.0

        policy.learn(snapshots, actions, rewards)

        assert all(
            torch.equal(before, after)
            for before, after in zip(encoder_before, policy.encoder.parameters(), strict=True)
        ), "the frozen-encoder arm must never move the encoder"
        assert all(
            torch.equal(before, after)
            for before, after in zip(layer1_before, policy.head.layer1.parameters(), strict=True)
        ), "the frozen-encoder arm must never move layer1"
        w = policy.head.w_vector()
        assert not torch.equal(w, torch.zeros_like(w)), (
            "the ridge solve should have moved W off zero"
        )
        assert torch.isfinite(w).all()
        assert policy.last_loss > 0.0

    def test_learn_on_empty_episode_is_a_no_op(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=22)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)

        before = [p.clone() for p in policy.encoder.parameters()]
        policy.learn([], [], [0.0])
        after = list(policy.encoder.parameters())

        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))
        assert policy.ridge.episodes_included == 0
        assert policy.ridge.episodes_excluded == 0

    def test_w_matches_the_ridge_accumulators_own_solve(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=24)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=5)

        policy.learn(snapshots, actions, rewards)

        expected = policy.ridge.solve()
        np.testing.assert_allclose(policy.head.w_vector().numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_learn_is_deterministic_given_the_same_inputs(self) -> None:
        """No RNG anywhere in ``learn`` any more (ticket 16): two runs from
        the same init seed over the same episode must agree bit for bit --
        there is no ``learn_rng`` seed left to hold fixed."""
        geometry, time_windows, clients = make_world(5, seed=23)

        def run() -> torch.Tensor:
            policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=5)
            policy.learn(snapshots, actions, rewards)
            return policy.head.w_vector().clone()

        first = run()
        second = run()
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
    def test_learn_is_deterministic_on_cuda(self) -> None:
        """Ticket 12's ``use_deterministic_algorithms(True)`` still matters
        here even with no backward pass anywhere in ``learn`` any more:
        :meth:`TransformerMonteCarloPolicy._replay_joint_features` runs a real
        forward pass through the encoder on CUDA, and CUDA kernel
        nondeterminism is not specific to backward passes."""
        geometry, time_windows, clients = make_world(5, seed=23)
        was_enabled = torch.are_deterministic_algorithms_enabled()

        def run() -> torch.Tensor:
            policy = build_policy(
                geometry,
                time_windows,
                clients,
                number_vehicles=2,
                epsilon=0.0,
                device=torch.device("cuda"),
            )
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=5)
            policy.learn(snapshots, actions, rewards)
            return policy.head.w_vector().clone()

        try:
            torch.use_deterministic_algorithms(True)
            first = run()
            second = run()
        finally:
            torch.use_deterministic_algorithms(was_enabled)

        torch.testing.assert_close(first.cpu(), second.cpu(), atol=0.0, rtol=0.0)


class TestWZeroReproducesTicket15sNull:
    """Acceptance: "W = 0 before the first solve reproduces ticket 15's null
    exactly." A ``solve_cadence`` the Episode count never reaches means
    ``learn`` accumulates but never solves, so accumulated data alone must
    never move a single decision.
    """

    def test_w_stays_zero_and_decide_is_unchanged_before_the_first_solve(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=40)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, solve_cadence=1000
        )
        state = make_state(2, pending=clients)

        policy.action = [DEPOT, DEPOT]
        before_action = policy.decide(state)

        snapshots, actions, rewards = make_episode(policy, state, length=6)
        policy.learn(snapshots, actions, rewards)

        assert policy.ridge.episodes_included == 1
        assert policy.ridge.episodes_since_solve == 1
        w = policy.head.w_vector()
        torch.testing.assert_close(w, torch.zeros_like(w), atol=0.0, rtol=0.0)

        policy.action = [DEPOT, DEPOT]
        after_action = policy.decide(state)
        assert after_action == before_action


class TestSolveCadence:
    """``ExperimentConfig.neural_solve_cadence`` (``self.solve_cadence``): the
    re-solve fires only once ``episodes_since_solve`` reaches it, then resets.
    """

    def test_the_solve_only_fires_every_cadence_episodes(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=41)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, solve_cadence=2
        )
        state = make_state(2, pending=clients)

        snapshots, actions, rewards = make_episode(policy, state, length=5)
        policy.learn(snapshots, actions, rewards)
        w_after_one = policy.head.w_vector().clone()
        torch.testing.assert_close(w_after_one, torch.zeros_like(w_after_one), atol=0.0, rtol=0.0)

        policy.learn(snapshots, actions, rewards)
        w_after_two = policy.head.w_vector()
        assert not torch.equal(w_after_two, w_after_one), (
            "the second episode should trigger a solve"
        )
        assert policy.ridge.episodes_since_solve == 0


class TestAbortedEpisodesAreExcluded:
    """Ticket 16: an Episode whose final reward carries the abort penalty
    (research note F10) is excluded from the accumulator outright."""

    def test_an_aborted_episode_leaves_the_accumulator_untouched_beyond_forgetting(
        self,
    ) -> None:
        geometry, time_windows, clients = make_world(5, seed=42)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, solve_cadence=1000
        )
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)
        # Guaranteed over the abort floor for any number of served Clients.
        rewards[-1] = policy._abort_reward_floor + 1.0

        policy.learn(snapshots, actions, rewards)

        assert policy.ridge.episodes_excluded == 1
        assert policy.ridge.episodes_included == 0
        np.testing.assert_array_equal(policy.ridge.A, np.zeros_like(policy.ridge.A))
        np.testing.assert_array_equal(policy.ridge.b, np.zeros_like(policy.ridge.b))

    def test_a_normal_episode_is_not_misclassified_as_aborted(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=43)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        assert not policy._is_aborted(rewards)

        policy.learn(snapshots, actions, rewards)

        assert policy.ridge.episodes_included == 1
        assert policy.ridge.episodes_excluded == 0


# --- the trained-encoder arm (ticket 17, "Two timescales") ---------------------


class TestTrainedEncoderArm:
    """``train_encoder=True``: after the ridge fold/solve every arm shares,
    ``learn`` additionally runs SGD over ``encoder``/``head.layer1`` --
    ``head.linear``/``head.layer2`` stay exclusively the ridge's to move."""

    def test_constructing_directly_without_optimizer_raises(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=80)
        rng = np.random.default_rng(0)
        encoder = TokenEncoder(
            d_model=D_MODEL, n_layers=2, n_heads=4, n_observed_velocities=N_OBS, init_rng=rng
        )
        head = QHead(d_model=D_MODEL, init_rng=rng)
        accumulator = RidgeAccumulator.zeros(head.feature_dim, forgetting=1.0, ridge=1.0)
        with pytest.raises(ValueError, match="train_encoder"):
            TransformerMonteCarloPolicy(
                number_vehicles=2,
                geometry=geometry,
                time_windows=time_windows,
                number_clients=len(clients),
                epsilon=0.0,
                depot=DEPOT,
                shift_end_minute=SHIFT_END,
                episode_end_minute=EPISODE_END,
                horizon_start_minute=HORIZON_START,
                encoder=encoder,
                head=head,
                ridge=accumulator,
                exploration_rng=np.random.default_rng(1),
                train_encoder=True,
                encoder_optimizer=None,
                learn_rng=np.random.default_rng(2),
            )

    def test_constructing_directly_without_learn_rng_raises(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=80)
        rng = np.random.default_rng(0)
        encoder = TokenEncoder(
            d_model=D_MODEL, n_layers=2, n_heads=4, n_observed_velocities=N_OBS, init_rng=rng
        )
        head = QHead(d_model=D_MODEL, init_rng=rng)
        accumulator = RidgeAccumulator.zeros(head.feature_dim, forgetting=1.0, ridge=1.0)
        optimizer = torch.optim.Adam([*encoder.parameters(), *head.layer1.parameters()], lr=1e-3)
        with pytest.raises(ValueError, match="train_encoder"):
            TransformerMonteCarloPolicy(
                number_vehicles=2,
                geometry=geometry,
                time_windows=time_windows,
                number_clients=len(clients),
                epsilon=0.0,
                depot=DEPOT,
                shift_end_minute=SHIFT_END,
                episode_end_minute=EPISODE_END,
                horizon_start_minute=HORIZON_START,
                encoder=encoder,
                head=head,
                ridge=accumulator,
                exploration_rng=np.random.default_rng(1),
                train_encoder=True,
                encoder_optimizer=optimizer,
                learn_rng=None,
            )

    def test_learn_moves_encoder_and_layer1_but_w_still_matches_the_ridge_solve(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=81)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, train_encoder=True
        )
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        encoder_before = [p.clone() for p in policy.encoder.parameters()]
        layer1_before = [p.clone() for p in policy.head.layer1.parameters()]

        policy.learn(snapshots, actions, rewards)

        assert any(
            not torch.equal(before, after)
            for before, after in zip(encoder_before, policy.encoder.parameters(), strict=True)
        ), "the trained-encoder arm must move the encoder"
        assert any(
            not torch.equal(before, after)
            for before, after in zip(layer1_before, policy.head.layer1.parameters(), strict=True)
        ), "the trained-encoder arm must move layer1"
        # head.linear/head.layer2 are still exclusively the ridge's: whatever
        # the SGD step above did to the encoder/layer1, the *combined* weight
        # vector must still equal a fresh solve of the (now-updated) ridge
        # accumulator -- no optimizer step ever touches linear/layer2 directly.
        expected = policy.ridge.solve()
        np.testing.assert_allclose(policy.head.w_vector().numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_frozen_arm_never_builds_an_optimizer_step_even_if_one_is_injected(self) -> None:
        """``train_encoder=False`` (the default): even a caller that hands in
        an optimizer/learn_rng anyway must see it go unused -- only
        ``self.train_encoder`` gates the SGD step, not "an optimizer happens
        to be present"."""
        geometry, time_windows, clients = make_world(5, seed=82)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2, epsilon=0.0)
        assert policy.train_encoder is False
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)
        encoder_before = [p.clone() for p in policy.encoder.parameters()]

        policy.learn(snapshots, actions, rewards)

        assert all(
            torch.equal(before, after)
            for before, after in zip(encoder_before, policy.encoder.parameters(), strict=True)
        )

    def test_aborted_episode_skips_the_sgd_step_too(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=83)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, epsilon=0.0, train_encoder=True
        )
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)
        rewards[-1] = policy._abort_reward_floor + 1.0
        encoder_before = [p.clone() for p in policy.encoder.parameters()]

        policy.learn(snapshots, actions, rewards)

        assert all(
            torch.equal(before, after)
            for before, after in zip(encoder_before, policy.encoder.parameters(), strict=True)
        ), "an aborted Episode carries no ranking information to buy a gradient step with"

    def test_learn_is_deterministic_given_the_same_learn_rng_seed(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=84)

        def run() -> torch.Tensor:
            policy = build_policy(
                geometry,
                time_windows,
                clients,
                number_vehicles=2,
                epsilon=0.0,
                train_encoder=True,
                learn_seed=7,
                learn_passes=2,
                batch_size=2,
            )
            state = make_state(2, pending=clients)
            snapshots, actions, rewards = make_episode(policy, state, length=6)
            policy.learn(snapshots, actions, rewards)
            return next(policy.encoder.parameters()).clone()

        first = run()
        second = run()
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


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


class TestResidualCalibrationPairs:
    """(W.phi, y~) once per decision epoch -- Gate A' 's (ticket 17) calibration
    primitive, a sibling of ``TestCalibrationPairs`` above pairing the learned
    term against the residual it is actually regressed onto, not Q against
    the raw return (which passes at W=0 with nothing learned)."""

    def test_one_pair_per_decision_epoch(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=31)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=6)

        pairs = policy.residual_calibration_pairs(snapshots, actions, rewards)

        assert len(pairs) == 6

    def test_empty_episode_returns_no_pairs(self) -> None:
        geometry, time_windows, clients = make_world(4, seed=32)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)

        assert policy.residual_calibration_pairs([], [], [0.0]) == []

    def test_at_w_zero_the_predicted_residual_is_identically_zero(self) -> None:
        """The guaranteed non-PASS spec.md's redefinition needs: an untrained
        network (W = 0, before any ridge solve) predicts a residual of
        exactly 0 for every candidate, so ``predicted_residual`` never varies
        with the return -- the correlation this class exists to make honest."""
        geometry, time_windows, clients = make_world(5, seed=33)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)

        pairs = policy.residual_calibration_pairs(snapshots, actions, rewards)

        assert len(pairs) == 4
        assert all(pair.predicted_residual == 0.0 for pair in pairs)

    def test_residual_target_matches_the_ridge_s_own_y_formula(self) -> None:
        """``y_t = targets[t] / return_scale - sum_v c(s, v, a_v)`` -- exactly
        ``learn``'s own regression target (module docstring, "The estimator"),
        recomputed here directly from ``_replay_joint_features`` rather than
        trusted to agree by construction."""
        geometry, time_windows, clients = make_world(5, seed=35)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        targets = policy._backward_returns(snapshots, actions, rewards)

        pairs = policy.residual_calibration_pairs(snapshots, actions, rewards)

        for t in range(4):
            with torch.no_grad():
                _phi_sum, c_sum = policy._replay_joint_features(snapshots[t], actions[t])
            expected_y = targets[t] / policy._return_scale - c_sum
            assert pairs[t].residual_target == pytest.approx(expected_y)

    def test_predicted_half_matches_phi_dot_w(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=36)
        policy = build_policy(
            geometry, time_windows, clients, number_vehicles=2, solve_cadence=1000
        )
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        # Force a nonzero W so the predicted half is not trivially zero.
        policy.head.load_w_vector(torch.ones(policy.head.feature_dim) * 0.01)

        pairs = policy.residual_calibration_pairs(snapshots, actions, rewards)

        for t in range(4):
            with torch.no_grad():
                phi_sum, _c_sum = policy._replay_joint_features(snapshots[t], actions[t])
                expected = float((phi_sum @ policy.head.w_vector()).item())
            assert pairs[t].predicted_residual == pytest.approx(expected)

    def test_does_not_mutate_the_network(self) -> None:
        geometry, time_windows, clients = make_world(5, seed=37)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=2)
        state = make_state(2, pending=clients)
        snapshots, actions, rewards = make_episode(policy, state, length=4)
        before = [p.clone() for p in policy.encoder.parameters()] + [
            p.clone() for p in policy.head.parameters()
        ]

        policy.residual_calibration_pairs(snapshots, actions, rewards)

        after = list(policy.encoder.parameters()) + list(policy.head.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


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


class TestReplayJointFeaturesMatchesReplayJointQ:
    """Ticket 16: :meth:`_replay_joint_features` must be the same additive
    decomposition :meth:`_replay_joint_q` already pins (``TestJointQIsAdditiveOverVehicles``
    above) -- ``c_sum + phi_sum @ w_vector() == _replay_joint_q(...)`` ties the
    new feature-accumulation replay to the already-tested Q replay directly,
    rather than trusting the two implementations to agree by construction."""

    def test_phi_sum_and_c_sum_reconstruct_the_joint_q(self) -> None:
        geometry, time_windows, clients = make_world(6, seed=37)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=clients)
        snapshot = TrainingSnapshot.capture(state)
        action_row = policy.decide(state)

        with torch.no_grad():
            joint_q = policy._replay_joint_q(snapshot, action_row).item()
            phi_sum, c_sum = policy._replay_joint_features(snapshot, action_row)
            reconstructed = c_sum + float(phi_sum @ policy.head.w_vector())

        assert phi_sum.shape == (policy.head.feature_dim,)
        assert reconstructed == pytest.approx(joint_q, rel=1e-5)

    def test_phi_sum_matches_a_manual_per_vehicle_sum(self) -> None:
        geometry, time_windows, clients = make_world(6, seed=38)
        policy = build_policy(geometry, time_windows, clients, number_vehicles=3)
        state = make_state(3, pending=clients)
        snapshot = TrainingSnapshot.capture(state)
        action_row = policy.decide(state)

        with torch.no_grad():
            phi_sum, c_sum = policy._replay_joint_features(snapshot, action_row)

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
            expected_phi = torch.zeros(policy.head.feature_dim)
            expected_c = 0.0
            for vehicle in range(3):
                phi, myopic_base = policy._score_features(embeddings, vehicle, pending, claimed)
                chosen = action_row[vehicle]
                index = len(pending) if chosen == DEPOT else pending.index(chosen)
                expected_phi = expected_phi + phi[index]
                expected_c += float(myopic_base[index].item())
                if chosen != DEPOT:
                    claimed[index] = True

        torch.testing.assert_close(phi_sum, expected_phi, atol=1e-6, rtol=1e-6)
        assert c_sum == pytest.approx(expected_c, rel=1e-5)


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
