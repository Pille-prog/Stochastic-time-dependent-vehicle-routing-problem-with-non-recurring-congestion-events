"""ClientGenerator: config-driven demand generation (ticket 13: private per-call RNG)."""

from pathlib import Path

import pytest

from stdvrp.config import ExperimentConfig
from stdvrp.demand import Client, ClientGenerator, EpisodeDemand

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def fixture_generator() -> ClientGenerator:
    return ClientGenerator.from_config(ExperimentConfig.from_yaml(FIXTURE_CONFIG))


def test_from_config_wires_every_demand_knob() -> None:
    generator = fixture_generator()
    assert generator.mean_number_clients == 20
    assert generator.client_count_stddev == 4.0
    assert generator.min_number_clients == 8
    assert generator.clients_per_vehicle == 4
    assert generator.client_universe_node_range == (1, 45)
    assert generator.time_window_spread == 60
    assert generator.horizon_start_minute == 300
    assert generator.horizon_end_minute == 780


def test_same_seed_reproduces_the_same_demand() -> None:
    generator = fixture_generator()
    first = generator.generate(7)
    assert generator.generate(7) == first


def test_different_seeds_reproduce_independently() -> None:
    # Ticket 13: each call builds its own Generator from the seed, so calls for
    # different seeds cannot leak state into one another either.
    generator = fixture_generator()
    a1, b1, a2 = generator.generate(1), generator.generate(2), generator.generate(1)
    assert a1 == a2
    assert a1 != b1


def test_clients_are_unique_nodes_within_the_universe() -> None:
    demand = fixture_generator().generate(3)
    nodes = [client.node for client in demand.clients]
    assert len(nodes) == len(set(nodes))
    assert all(1 <= node < 45 for node in nodes)


def test_time_windows_have_the_configured_spread_inside_the_horizon() -> None:
    demand = fixture_generator().generate(3)
    for client in demand.clients:
        assert client.time_window_end - client.time_window_start == 60
        assert client.time_window_start >= 300
        assert client.time_window_end <= 780


def test_client_count_floor_applies() -> None:
    generator = ClientGenerator(
        mean_number_clients=2,
        client_count_stddev=0.5,
        min_number_clients=8,
        client_universe_node_range=(1, 45),
        clients_per_vehicle=4,
        time_window_spread=60,
        horizon_start_minute=300,
        horizon_end_minute=780,
    )
    assert all(len(generator.generate(seed).clients) == 8 for seed in range(20))


@pytest.mark.parametrize("seed", range(10))
def test_vehicle_count_is_clients_divided_by_ratio_rounded_up(seed: int) -> None:
    demand = fixture_generator().generate(seed)
    count = len(demand.clients)
    assert demand.vehicle_count == count // 4 + (1 if count % 4 else 0)


def test_demand_values_are_frozen() -> None:
    demand = fixture_generator().generate(0)
    with pytest.raises(AttributeError):
        demand.vehicle_count = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        demand.clients[0].node = 99  # type: ignore[misc]


def test_client_and_demand_are_plain_value_types() -> None:
    client = Client(node=4, time_window_start=310, time_window_end=370)
    demand = EpisodeDemand(clients=(client,), vehicle_count=1)
    assert demand.clients[0] == Client(4, 310, 370)
