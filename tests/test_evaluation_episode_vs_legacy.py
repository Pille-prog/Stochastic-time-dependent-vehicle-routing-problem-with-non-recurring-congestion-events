"""Exact-equality characterization: a full evaluation Episode vs the unmodified legacy.

The tracer bullet of ticket 07: State, MonteCarloPolicy (evaluation path), the live
ArcProbabilityCongestionGenerator and Model.transition_function must reproduce the
legacy episode bit-for-bit, including consuming the global ``random``/``np.random``
streams in the same order (ADR-0001).

Venue (the ticket 05/06 pattern): the legacy classes run byte-for-byte unmodified on
a temporary world of 44 fixture-day copies. The legacy ``ClientGenerator`` hardcodes
a 1,900-node universe and a 60-client floor, so demand comes from the ported
``ClientGenerator`` with fixture-sized parameters for BOTH sides — ticket 06 proved
its stream consumption is bit-identical to the legacy's, so the episodes that follow
start from identical RNG state. Each side re-seeds before running, and both must
leave both global streams at the same position afterwards.
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
from stdvrp.simulation import run_evaluation_episode
from stdvrp.traffic import CsvDataSource, TravelTimeModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
LEGACY_SCRIPT = REPO_ROOT / "Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))

HORIZON_START, HORIZON_END = 300, 780
N_OBSERVED_ARCS = 3
EPSILON = 0.05
CONGESTION_LOWER, CONGESTION_UPPER = 0.3, 0.4
MAX_CONGESTION_DURATION = 120
SEEDS = (0, 7, 1000, 100000)

# 12 general-state + 7 state-action features; a non-trivial W exercises real
# greedy decisions (np.dot raises loudly if the feature count ever drifts).
W_FIXED = np.linspace(-0.4, 0.6, 19)

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


def run_legacy_episode(
    legacy_module: ModuleType,
    legacy_calc: Any,
    legacy_spm: Any,
    client_generator: ClientGenerator,
    seed: int,
) -> tuple[dict[str, Any], tuple[float, float]]:
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
        0.001,
        W_FIXED,
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
    model.create_monte_carlo_episode_test()

    outcome = {
        "total_cost": model.total_cost,
        "distance_cost": model.total_distance_cost,
        "delay_cost": model.total_delay_cost,
        "earliness_cost": model.total_earliness_cost,
        "overtime_cost": model.total_overtime_cost,
        "tau": model.state.tau_episode,
        "state_count": model.total_state_counter,
        "delay_clients": model.total_delay_clients,
        "earliness_clients": model.total_earliness_clients,
    }
    stream_position = (random.random(), float(np.random.uniform(0, 1)))
    return outcome, stream_position


@pytest.mark.parametrize("seed", SEEDS)
def test_evaluation_episode_is_bit_identical(
    legacy_module: ModuleType,
    legacy_calc: Any,
    legacy_spm: Any,
    ported_world: dict[str, Any],
    seed: int,
) -> None:
    legacy_outcome, legacy_stream = run_legacy_episode(
        legacy_module, legacy_calc, legacy_spm, ported_world["client_generator"], seed
    )

    result = run_evaluation_episode(
        seed=seed,
        client_generator=ported_world["client_generator"],
        travel_time_model=ported_world["travel_time_model"],
        shortest_path_cache=ported_world["cache"],
        congestion_generator=ported_world["congestion_generator"],
        W=W_FIXED,
        epsilon=EPSILON,
        max_congestion_duration=MAX_CONGESTION_DURATION,
        horizon_start_minute=HORIZON_START,
        horizon_end_minute=HORIZON_END,
        n_observed_arcs=N_OBSERVED_ARCS,
    )
    ported_stream = (random.random(), float(np.random.uniform(0, 1)))

    assert result.total_cost == legacy_outcome["total_cost"]
    assert result.distance_cost == legacy_outcome["distance_cost"]
    assert result.delay_cost == legacy_outcome["delay_cost"]
    assert result.earliness_cost == legacy_outcome["earliness_cost"]
    assert result.overtime_cost == legacy_outcome["overtime_cost"]
    assert result.tau == legacy_outcome["tau"]
    assert result.state_count == legacy_outcome["state_count"]
    assert result.delay_clients == legacy_outcome["delay_clients"]
    assert result.earliness_clients == legacy_outcome["earliness_clients"]
    # Both sides must consume the global streams identically: the next draw from
    # each stream matches only if every intermediate draw matched (ADR-0001).
    assert ported_stream == legacy_stream
