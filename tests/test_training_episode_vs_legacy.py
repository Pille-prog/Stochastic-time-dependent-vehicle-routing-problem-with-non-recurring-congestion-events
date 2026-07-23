"""Exact-equality characterization: training Episodes and the W update vs the legacy.

Ticket 08: ``decide_train`` (repair pass + ε-greedy exploration) and ``update_W``
must reproduce the legacy ``monte_carlo_policy_train`` / ``actualize_W`` bit-for-bit
across a chained training run — including the warm-up learning-rate quirk (the
first Episode trains with the hardcoded 1e-6 before the configured rate) and the
golden-capture convention of seeding the legacy's otherwise-unseeded exploration
RNGs per Episode (offset + train seed, ticket 04 finding 1).

Venue: identical to test_evaluation_episode_vs_legacy.py — the unmodified legacy
classes run on a temporary world of 44 fixture-day copies (single-day std is NaN,
44 copies make it exactly 0.0), demand comes from the ported ClientGenerator for
both sides, and every Episode must leave the global ``random``/``np.random``
streams at the same position on both sides (ADR-0001).
"""

import builtins
import importlib.util
import os
import random
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_training_episode
from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
LEGACY_SCRIPT = REPO_ROOT / "Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))

HORIZON_START, HORIZON_END = 300, 780
N_OBSERVED_ARCS = 3
EPSILON = 0.3  # high enough that exploration and repair genuinely fire
CONGESTION_LOWER, CONGESTION_UPPER = 0.3, 0.4
MAX_CONGESTION_DURATION = 120

# The chained training run: W starts as None, the first Episode trains with the
# legacy warm-up learning rate, later ones with the configured rate.
TRAIN_SEEDS = (1000, 1001, 1002)
WARMUP_LEARNING_RATE = 0.000001
LEARNING_RATE = 0.001

# Golden-capture convention (ticket 04): per-Episode seeds for the legacy policy's
# otherwise-unseeded exploration/repair RNGs.
EXPLORATION_SEED_OFFSET = 10_000_000
REPAIR_SEED_OFFSET = 20_000_000

FIXTURE_DEMAND = dict(
    mean_number_clients=20,
    client_count_stddev=4.0,
    min_number_clients=8,
    client_universe_node_range=(1, 45),
    clients_per_vehicle=4,
    time_window_spread=60,
    horizon_start_minute=HORIZON_START,
    horizon_end_minute=HORIZON_END,
)


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    world = tmp_path_factory.mktemp("legacy_world")
    shutil.copyfile(FIXTURE_DIR / "link.csv", world / "link.csv")
    for day in LEGACY_DAYS:
        for half in (0, 1):
            shutil.copyfile(
                FIXTURE_DIR / f"speed[601]_[{half}].csv",
                world / f"speed[{day}]_[{half}].csv",
            )
    return world


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("legacy_monolith", LEGACY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Imported modules get a dict __builtins__; the legacy transition_function
    # calls the literal ``__builtins__.min`` (script-style), so restore the module.
    module.__dict__["__builtins__"] = builtins
    return module


@pytest.fixture(scope="module")
def legacy_calc(legacy_module: ModuleType, legacy_world: Path) -> Any:
    previous_cwd = os.getcwd()
    os.chdir(legacy_world)
    try:
        env = legacy_module.environment(
            "link.csv", "speed[601]_[0].csv", "speed[601]_[1].csv", [0], HORIZON_START, HORIZON_END
        )
        env.preprocess_data_average()
        calc = legacy_module.DataCalculations(env, MAX_CONGESTION_DURATION)
    finally:
        os.chdir(previous_cwd)
    return calc


@pytest.fixture(scope="module")
def legacy_spm(legacy_module: ModuleType) -> Any:
    previous_cwd = os.getcwd()
    os.chdir(FIXTURE_DIR)  # the legacy loads "all_shortest_paths.csv" from the CWD
    try:
        return legacy_module.shortest_path_memory(SimpleNamespace(node_list=[]))
    finally:
        os.chdir(previous_cwd)


@pytest.fixture(scope="module")
def ported_world(legacy_world: Path) -> dict[str, Any]:
    source = CsvDataSource(legacy_world, "link.csv", 601, LEGACY_DAYS, "all_shortest_paths.csv")
    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        MAX_CONGESTION_DURATION,
        horizon_start_minute=HORIZON_START,
    )
    return {
        "travel_time_model": travel_time_model,
        "cache": ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv"),
        "client_generator": ClientGenerator(**FIXTURE_DEMAND),
        "congestion_generator": ArcProbabilityCongestionGenerator(
            event_probability=travel_time_model.event_probability,
            successors=travel_time_model.successors,
            congestion_lower_bound=CONGESTION_LOWER,
            congestion_upper_bound=CONGESTION_UPPER,
            max_congestion_duration=MAX_CONGESTION_DURATION,
        ),
    }


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
    model.create_monte_carlo_episode_train()

    return {
        "w": np.array(model.policy.W, copy=True),
        "w_forward": model.policy.W,
        "total_cost": model.total_cost,
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
        assert result.episode.total_cost == legacy["total_cost"]
        assert result.episode.distance_cost == legacy["distance_cost"]
        assert result.episode.delay_cost == legacy["delay_cost"]
        assert result.episode.earliness_cost == legacy["earliness_cost"]
        assert result.episode.overtime_cost == legacy["overtime_cost"]
        assert result.episode.tau == legacy["tau"]
        assert result.episode.state_count == legacy["state_count"]
        assert result.episode.delay_clients == legacy["delay_clients"]
        assert result.episode.earliness_clients == legacy["earliness_clients"]
        # Both sides must consume the global streams identically (ADR-0001).
        assert ported_stream == legacy["stream"]
        # The preserved quirk: training zeroes the distance accumulator per step.
        assert result.episode.distance_cost == 0

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
