"""CongestionGenerator: non-recurring congestion events (variation axis 2; ADR-0002).

The live implementation is a Phase-1 structural port (ADR-0001) of the legacy
``DataCalculations.create_random_unexpected_event_with_probability_and_2_nodes`` —
the only congestion model ``model.transition_function`` actually invokes. It draws
from the **global** ``np.random`` stream in the exact legacy order: one uniform per
arc of ``event_probability`` in dict insertion order, then, per triggered event,
one uniform for the velocity penalization and one for the duration.

Congested arcs are recorded as ``(float(node_start), float(node_end)) ->
[velocity_multiplier, end_minute]`` — the float keys are a legacy quirk that must
stay aligned with the float node ids of cached shortest paths (they hash and
compare equal to the int arc keys used elsewhere).

Phase-2 deliberate fixes (ticket 12, ADR-0001 change log): the spread walks the
full ``max_depth`` (the legacy passed ``max_depth - 1``, leaving the depth-3
damping dead), and spread multipliers saturate at ``congestion_upper_bound``
(damping divides by a factor < 1, which let them exceed the configured bound).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

ArcKey = tuple[float, float]
# velocity multiplier applied while congested, minute the event ends.
CongestedArcs = dict[ArcKey, list[float]]


class CongestionGenerator(ABC):
    """Generates non-recurring congestion events during an Episode."""

    @abstractmethod
    def generate(self, minute_start: float, congested_arcs: CongestedArcs) -> None:
        """Add this decision epoch's new events (if any) to ``congested_arcs``."""


@dataclass
class ArcProbabilityCongestionGenerator(CongestionGenerator):
    """One event roll per arc and epoch, spreading to neighbors with damped intensity.

    ``event_probability`` and ``successors`` come from the TravelTimeModel; both
    iteration orders are behavior (ADR-0001): one ``np.random.uniform`` is consumed
    per ``event_probability`` key whether or not an event triggers, and the spread
    walks ``successors`` lists in arc-table order.
    """

    event_probability: dict[tuple[int, int], float]
    successors: dict[int, list[int]]
    congestion_lower_bound: float
    congestion_upper_bound: float
    max_congestion_duration: int
    # The legacy model hardcoded max_depth=3; the damping table below covers
    # depths 0-3 only. probability_input mirrors the legacy enablement argument,
    # which was the event-probability dict itself — always ``!= 0``, so
    # congestion was never actually disabled; a nonzero float keeps that
    # behavior.
    max_depth: int = 3
    probability_input: float = 1.0

    def generate(self, minute_start: float, congested_arcs: CongestedArcs) -> None:
        if self.probability_input != 0:
            for key in self.event_probability:
                probability_for_congestion = np.random.uniform(0, 1)
                if probability_for_congestion < self.event_probability[key]:
                    node_start_congestion = key[0]
                    node_end_congestion = key[1]
                    congestion_road = [node_start_congestion, node_end_congestion]
                    velocity_penalization = np.random.uniform(
                        self.congestion_lower_bound, self.congestion_upper_bound
                    )
                    state_time_elimination = np.random.uniform(30, self.max_congestion_duration)

                    congested_arcs[(float(node_start_congestion), float(node_end_congestion))] = [
                        float(velocity_penalization),
                        float(minute_start + state_time_elimination),
                    ]

                    for node in congestion_road:
                        # Phase-2 fix (ticket 12, ADR-0001 change log): the legacy
                        # passed ``max_depth - 1`` here, so the depth-3 damping
                        # branch was dead code and the spread stopped one hop short.
                        node_starts, depth = self._reachable_nodes(node, 0, self.max_depth)
                        for node_start in node_starts:
                            connected_nodes = self.successors.get(node_start, [])

                            for affected_node in connected_nodes:
                                if node_start == affected_node:
                                    continue

                                if depth[node_start] == 0:
                                    factor = 1.0
                                elif depth[node_start] == 1:
                                    factor = 0.83
                                elif depth[node_start] == 2:
                                    factor = 0.78
                                elif depth[node_start] == 3:
                                    factor = 0.73
                                else:  # unreachable: recursion depth is capped at 3
                                    raise AssertionError(f"depth {depth[node_start]} > 3")

                                # Phase-2 fix (ticket 12): damping divides by a
                                # factor < 1 (milder congestion farther out), which
                                # let spread multipliers exceed the configured
                                # upper bound; they now saturate at it.
                                velocity_penalization_for_depth = min(
                                    velocity_penalization / factor,
                                    self.congestion_upper_bound,
                                )
                                if (node_start, affected_node) in congested_arcs and (
                                    congested_arcs[(node_start, affected_node)][1] > minute_start
                                    and congested_arcs[(node_start, affected_node)][0]
                                    <= velocity_penalization_for_depth
                                ):
                                    continue

                                congested_arcs[(float(node_start), float(affected_node))] = [
                                    float(velocity_penalization_for_depth),
                                    float(minute_start + state_time_elimination),
                                ]

    def _reachable_nodes(
        self,
        node_start: int,
        depth: int,
        max_depth: int,
        visited: set[int] | None = None,
        node_depth: dict[int, int] | None = None,
    ) -> tuple[set[int], dict[int, int]]:
        """Ports ``get_all_node_starts``: BFS-by-recursion collecting nodes and depths.

        Returns the *set* of reached nodes; the caller iterates it, so the set's
        insertion history (and therefore iteration order) is part of the preserved
        behavior (ADR-0001).
        """
        if visited is None:
            visited = set()
            node_depth = {}
        assert node_depth is not None

        visited.add(node_start)
        node_depth[node_start] = depth

        if depth < max_depth:
            connected_nodes = self.successors.get(node_start, [])
            for node in connected_nodes:
                if node not in visited:
                    self._reachable_nodes(node, depth + 1, max_depth, visited, node_depth)

        return visited, node_depth
