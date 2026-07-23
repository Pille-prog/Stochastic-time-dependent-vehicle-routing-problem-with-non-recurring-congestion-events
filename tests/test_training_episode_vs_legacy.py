"""Exact-equality characterization: training Episodes and the W update vs the legacy.

Ticket 08: ``decide_train`` (repair pass + ε-greedy exploration) and ``update_W``
must reproduce the legacy ``monte_carlo_policy_train`` / ``actualize_W`` bit-for-bit
across a chained training run — including the warm-up learning-rate quirk (the
first Episode trains with the hardcoded 1e-6 before the configured rate) and the
golden-capture convention of seeding the legacy's otherwise-unseeded exploration
RNGs per Episode (offset + train seed, ticket 04 finding 1).

Venue: the shared 44-copy fixture world (``characterization_world`` + conftest
fixtures); every Episode must leave the global ``random``/``np.random`` streams at
the same position on both sides (ADR-0001).
"""

import random
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from characterization_world import (
    CONGESTION_LOWER,
    CONGESTION_UPPER,
    HORIZON_END,
    HORIZON_START,
    MAX_CONGESTION_DURATION,
    N_OBSERVED_ARCS,
    count_horizon_terminations,
)
from stdvrp.demand import ClientGenerator
from stdvrp.simulation import run_training_episode

EPSILON = 0.3  # high enough that exploration and repair genuinely fire

# The chained training run: W starts as None, the first Episode trains with the
# legacy warm-up learning rate, later ones with the configured rate.
TRAIN_SEEDS = (1000, 1001, 1002)
WARMUP_LEARNING_RATE = 0.000001
LEARNING_RATE = 0.001

# Golden-capture convention (ticket 04): per-Episode seeds for the legacy policy's
# otherwise-unseeded exploration/repair RNGs.
EXPLORATION_SEED_OFFSET = 10_000_000
REPAIR_SEED_OFFSET = 20_000_000


def run_legacy_training_episode(
    legacy_module: ModuleType,
    legacy_calc: Any,
    legacy_spm: Any,
    client_generator: ClientGenerator,
    seed: int,
    w: Any,
    learning_rate: float,
) -> dict[str, Any]:
    demand = client_generator.generate(seed)
    np.random.seed(seed)

    number_vehicles = demand.vehicle_count
    clients = [client.node for client in demand.clients]
    time_windows = {
        client.node: [client.time_window_start, client.time_window_end] for client in demand.clients
    }
    cg = SimpleNamespace(clients=time_windows, random_depot=0)

    state = legacy_module.state(number_vehicles, clients, N_OBSERVED_ARCS, HORIZON_START, 0)
    policy = legacy_module.policy(
        number_vehicles,
        [[]],
        legacy_spm,
        cg,
        legacy_calc,
        state,
        len(clients),
        EPSILON,
        0,
        CONGESTION_LOWER,
        CONGESTION_UPPER,
        number_vehicles + 2,
        number_vehicles + 2,
        learning_rate,
        w,
    )
    model = legacy_module.model(
        state,
        policy,
        legacy_calc,
        legacy_spm,
        cg,
        number_vehicles,
        HORIZON_START,
        HORIZON_END,
        0,
        CONGESTION_LOWER,
        CONGESTION_UPPER,
        MAX_CONGESTION_DURATION,
    )
    # The capture convention: seed the unseeded exploration RNGs right after
    # construction, before any training decision runs.
    policy.local_rng.seed(EXPLORATION_SEED_OFFSET + seed)
    policy.local_rng_2.seed(REPAIR_SEED_OFFSET + seed)
    with count_horizon_terminations(legacy_module) as horizon_calls:
        model.create_monte_carlo_episode_train()

    return {
        "w": np.array(model.policy.W, copy=True),
        "w_forward": model.policy.W,
        "total_cost": model.total_cost,
        # Ticket 12 fix 6: the legacy double-adds the terminating transition's
        # cost past the horizon; the comparison compensates the exact delta.
        "double_added_cost": model.transition_cost if horizon_calls else 0.0,
        "distance_cost": model.total_distance_cost,
        "delay_cost": model.total_delay_cost,
        "earliness_cost": model.total_earliness_cost,
        "overtime_cost": model.total_overtime_cost,
        "tau": model.state.tau_episode,
        "state_count": model.total_state_counter,
        "delay_clients": model.total_delay_clients,
        "earliness_clients": model.total_earliness_clients,
        "exploration_rng_state": policy.local_rng.getstate(),
        "repair_rng_state": policy.local_rng_2.getstate(),
        "stream": (random.random(), float(np.random.uniform(0, 1))),
    }


def test_chained_training_episodes_are_bit_identical(
    legacy_module: ModuleType,
    legacy_calc: Any,
    legacy_spm: Any,
    ported_world: dict[str, Any],
) -> None:
    """Three chained training Episodes: every W component and cost matches exactly."""
    legacy_w: Any = None
    ported_w: Any = None
    learning_rate = WARMUP_LEARNING_RATE  # legacy warm-up quirk: first Episode only

    for seed in TRAIN_SEEDS:
        legacy = run_legacy_training_episode(
            legacy_module,
            legacy_calc,
            legacy_spm,
            ported_world["client_generator"],
            seed,
            legacy_w,
            learning_rate,
        )

        result = run_training_episode(
            seed=seed,
            client_generator=ported_world["client_generator"],
            travel_time_model=ported_world["travel_time_model"],
            shortest_path_cache=ported_world["cache"],
            congestion_generator=ported_world["congestion_generator"],
            W=ported_w,
            learning_rate=learning_rate,
            epsilon=EPSILON,
            max_congestion_duration=MAX_CONGESTION_DURATION,
            horizon_start_minute=HORIZON_START,
            horizon_end_minute=HORIZON_END,
            n_observed_arcs=N_OBSERVED_ARCS,
            exploration_seed=EXPLORATION_SEED_OFFSET + seed,
            repair_seed=REPAIR_SEED_OFFSET + seed,
        )
        ported_stream = (random.random(), float(np.random.uniform(0, 1)))

        assert list(result.w) == list(legacy["w"]), f"W diverged at seed {seed}"
        # Ticket 12 fix 6: adding the legacy's double-added terminating cost
        # back reproduces its sum bit-for-bit (same operands, same order).
        assert result.episode.total_cost + legacy["double_added_cost"] == legacy["total_cost"]
        assert result.episode.delay_cost == legacy["delay_cost"]
        assert result.episode.earliness_cost == legacy["earliness_cost"]
        assert result.episode.overtime_cost == legacy["overtime_cost"]
        assert result.episode.tau == legacy["tau"]
        assert result.episode.state_count == legacy["state_count"]
        assert result.episode.delay_clients == legacy["delay_clients"]
        assert result.episode.earliness_clients == legacy["earliness_clients"]
        # Both sides must consume the global streams identically (ADR-0001).
        assert ported_stream == legacy["stream"]
        # Phase-2 fix (ticket 12): the legacy zeroed the distance accumulator
        # after every training step (reporting only — rewards are captured
        # before the zeroing), so it reports 0 where the port now reports the
        # Episode's real distance component.
        assert legacy["distance_cost"] == 0
        assert result.episode.distance_cost > 0

        # The Episode must genuinely exercise the training RNG paths: the
        # exploration gate draws once per vehicle per epoch (always), and these
        # seeds/ε are chosen so the infeasible-action repair fires too.
        assert (
            legacy["exploration_rng_state"]
            != random.Random(EXPLORATION_SEED_OFFSET + seed).getstate()
        ), f"exploration gate never drew at seed {seed}"
        assert legacy["repair_rng_state"] != random.Random(REPAIR_SEED_OFFSET + seed).getstate(), (
            f"repair pass never drew at seed {seed}"
        )

        legacy_w = legacy["w_forward"]
        ported_w = result.w
        learning_rate = LEARNING_RATE

    # Training genuinely moved the weights off the initial zero vector.
    assert ported_w is not None and any(component != 0 for component in ported_w)
