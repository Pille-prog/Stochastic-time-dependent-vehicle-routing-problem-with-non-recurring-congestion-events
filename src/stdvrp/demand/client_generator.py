"""ClientGenerator: draws the Clients and fleet size of one Episode.

Phase-1 port of the legacy ``ClientGenerator.client_generator_function`` (ADR-0001):
``generate`` reseeds the **global** ``random`` module and consumes its stream in the
exact legacy order — one ``gauss`` for the client count, one ``sample`` over the
client node range, then one ``randint`` per client for the time-window start — so
episodes reproduce the legacy bit-for-bit for equal seeds. The values the legacy
hardcoded (stddev 30, 60-client floor, the {150: 28, 250: 29} vehicle-ratio table,
``range(1, 1900)``) come from ``ExperimentConfig`` instead.

Like the legacy, a gauss draw above the size of the node universe makes ``sample``
raise ``ValueError``; configs keep the mean far enough below the universe size.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from stdvrp.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class Client:
    """A demand point: a RoadNetwork node that must be served within its time window."""

    node: int
    time_window_start: int
    time_window_end: int


@dataclass(frozen=True, slots=True)
class EpisodeDemand:
    """What one Episode must serve: the drawn Clients (in draw order) and the fleet size."""

    clients: tuple[Client, ...]
    vehicle_count: int


@dataclass(frozen=True, slots=True)
class ClientGenerator:
    """Generates the EpisodeDemand of an Episode from its seed."""

    mean_number_clients: int
    client_count_stddev: float
    min_number_clients: int
    client_universe_node_range: tuple[int, int]
    clients_per_vehicle: int
    time_window_spread: int
    horizon_start_minute: int
    horizon_end_minute: int

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> ClientGenerator:
        return cls(
            mean_number_clients=config.mean_number_clients,
            client_count_stddev=config.client_count_stddev,
            min_number_clients=config.min_number_clients,
            client_universe_node_range=config.client_universe_node_range,
            clients_per_vehicle=config.clients_per_vehicle,
            time_window_spread=config.time_window_spread,
            horizon_start_minute=config.horizon_start_minute,
            horizon_end_minute=config.horizon_end_minute,
        )

    def generate(self, seed: int) -> EpisodeDemand:
        random.seed(seed)
        count = int(random.gauss(self.mean_number_clients, self.client_count_stddev))
        count = max(count, self.min_number_clients)
        low, high = self.client_universe_node_range
        nodes = random.sample(range(low, high), count)

        # Legacy rounding kept verbatim: float division, truncate, +1 on a remainder.
        if count % self.clients_per_vehicle == 0:
            vehicle_count = int(count / self.clients_per_vehicle)
        else:
            vehicle_count = int(count / self.clients_per_vehicle) + 1

        latest_start = self.horizon_end_minute - self.time_window_spread
        clients = []
        for node in nodes:
            start = random.randint(self.horizon_start_minute, latest_start)
            clients.append(Client(node, start, start + self.time_window_spread))
        return EpisodeDemand(clients=tuple(clients), vehicle_count=vehicle_count)
