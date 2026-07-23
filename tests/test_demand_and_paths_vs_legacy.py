"""Exact-equality characterization: ClientGenerator and ShortestPathCache vs the
unmodified legacy classes (read-only import, ADR-0001).

Client generation needs no data files at all — the legacy ``ClientGenerator`` is
pure global-RNG consumption — so the comparison sweeps training seeds (1000...),
evaluation seeds (100000...) and both rows of the legacy vehicle-ratio table.
The path cache comparison loads the same fixture CSV through both loaders and
requires the resulting mappings to be bit-identical, float node ids included.
"""

import os
import random
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"

HORIZON_START, HORIZON_END = 300, 780
TIME_WINDOW_SPREAD = 60
SEEDS = (0, *range(1000, 1010), *range(100000, 100010))
LEGACY_RATIO_TABLE = {150: 28, 250: 29}


@pytest.mark.parametrize("mean_number_clients", sorted(LEGACY_RATIO_TABLE))
@pytest.mark.parametrize("seed", SEEDS)
def test_client_generation_is_bit_identical(
    legacy_module: ModuleType, mean_number_clients: int, seed: int
) -> None:
    legacy = legacy_module.ClientGenerator(0)
    legacy.client_generator_function(
        seed, mean_number_clients, TIME_WINDOW_SPREAD, HORIZON_START, HORIZON_END
    )
    stream_after_legacy = random.random()

    generator = ClientGenerator(
        mean_number_clients=mean_number_clients,
        client_count_stddev=30.0,
        min_number_clients=60,
        client_universe_node_range=(1, 1900),
        clients_per_vehicle=LEGACY_RATIO_TABLE[mean_number_clients],
        time_window_spread=TIME_WINDOW_SPREAD,
        horizon_start_minute=HORIZON_START,
        horizon_end_minute=HORIZON_END,
    )
    demand = generator.generate(seed)

    assert [client.node for client in demand.clients] == legacy.client_list
    assert {
        client.node: [client.time_window_start, client.time_window_end] for client in demand.clients
    } == legacy.clients
    assert demand.vehicle_count == int(legacy.number_vehicles)
    # Both must leave the shared global stream at the same position: the episode
    # that follows consumes it, so an extra or missing draw would shift everything.
    assert random.random() == stream_after_legacy


def test_shortest_path_cache_is_bit_identical(legacy_module: ModuleType) -> None:
    previous_cwd = os.getcwd()
    os.chdir(FIXTURE_DIR)  # the legacy loads "all_shortest_paths.csv" from the CWD
    try:
        legacy = legacy_module.shortest_path_memory(SimpleNamespace(node_list=[]))
    finally:
        os.chdir(previous_cwd)

    cache = ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv")

    assert list(cache.as_dict()) == list(legacy.shortest_paths)
    assert cache.as_dict() == legacy.shortest_paths
