"""Unit tests for ArcProbabilityCongestionGenerator on a tiny hand-built network.

The bit-exact behavior is guarded by tests/test_evaluation_episode_vs_legacy.py;
these pin the seam's contract on inputs small enough to reason about: one uniform
consumed per arc regardless of triggering, congested values within bounds, spread
with damped intensity, and the keep-stronger-existing-event rule.
"""

import numpy as np
import pytest

from stdvrp.congestion import ArcProbabilityCongestionGenerator

# 1 -> 2 -> 3 -> 4 chain with a 2 -> 5 branch.
SUCCESSORS = {1: [2], 2: [3, 5], 3: [4], 5: []}


def make_generator(
    event_probability: dict[tuple[int, int], float],
    lower: float = 0.1,
    upper: float = 0.3,
) -> ArcProbabilityCongestionGenerator:
    return ArcProbabilityCongestionGenerator(
        event_probability=event_probability,
        successors=SUCCESSORS,
        congestion_lower_bound=lower,
        congestion_upper_bound=upper,
        max_congestion_duration=60,
    )


class TestRngConsumption:
    def test_one_uniform_per_arc_when_nothing_triggers(self):
        generator = make_generator({(1, 2): 0.0, (2, 3): 0.0, (3, 4): 0.0})
        congested: dict = {}
        np.random.seed(0)
        generator.generate(300, congested)
        after_generate = np.random.uniform(0, 1)

        np.random.seed(0)
        for _ in range(3):  # exactly one uniform consumed per arc, hit or miss
            np.random.uniform(0, 1)
        assert np.random.uniform(0, 1) == after_generate
        assert congested == {}

    def test_probability_input_zero_disables_generation_and_rng(self):
        generator = make_generator({(1, 2): 1.0})
        generator.probability_input = 0
        congested: dict = {}
        np.random.seed(123)
        untouched_first_draw = np.random.uniform(0, 1)

        np.random.seed(123)
        generator.generate(300, congested)
        assert congested == {}
        assert np.random.uniform(0, 1) == untouched_first_draw


class TestEventValues:
    def test_triggered_event_has_bounded_multiplier_and_duration(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.2, upper=0.4)
        congested: dict = {}
        np.random.seed(7)
        generator.generate(300, congested)

        multiplier, end_minute = congested[(1.0, 2.0)]
        assert 0.2 <= multiplier <= 0.4
        assert 330 <= end_minute <= 360  # 300 + uniform(30, 60)
        assert all(isinstance(k[0], float) and isinstance(k[1], float) for k in congested)

    def test_event_spreads_to_neighbors_with_damped_intensity(self):
        generator = make_generator({(1, 2): 1.0})
        congested: dict = {}
        # Seed 2 draws a multiplier near the lower bound, so no damped value
        # reaches the upper-bound clamp and the damping stays observable.
        np.random.seed(2)
        generator.generate(300, congested)

        # Spread reaches arcs out of both endpoint nodes up to depth 2.
        assert (2.0, 3.0) in congested
        assert (2.0, 5.0) in congested
        assert (3.0, 4.0) in congested
        # Deeper arcs are *less* congested: divided by a factor < 1 means a
        # higher multiplier, i.e. a milder slowdown.
        assert congested[(3.0, 4.0)][0] == pytest.approx(congested[(2.0, 3.0)][0] / 0.83)

    def test_stronger_existing_event_is_kept(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.5, upper=0.5)
        congested: dict = {(2.0, 3.0): [0.1, 400.0]}  # stronger (lower multiplier), active
        np.random.seed(7)
        generator.generate(300, congested)

        assert congested[(2.0, 3.0)] == [0.1, 400.0]

    def test_weaker_existing_event_is_overwritten(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.2, upper=0.2)
        congested: dict = {(2.0, 3.0): [0.9, 400.0]}  # weaker (higher multiplier)
        np.random.seed(7)
        generator.generate(300, congested)

        assert congested[(2.0, 3.0)][0] != 0.9


class TestPhase2Fixes:
    """Ticket 12 fix 7 (ADR-0001 change log): bounded spread, live depth 3."""

    def test_spread_multipliers_never_exceed_the_upper_bound(self):
        # lower == upper pins the drawn multiplier at 0.3; undamped spread would
        # store 0.3/0.83 and 0.3/0.78, both above the configured upper bound.
        generator = make_generator({(1, 2): 1.0}, lower=0.3, upper=0.3)
        congested: dict = {}
        np.random.seed(7)
        generator.generate(300, congested)

        assert len(congested) > 1, "the event never spread; the check would be vacuous"
        for multiplier, _end in congested.values():
            assert 0.3 <= multiplier <= 0.3

    def test_spread_reaches_depth_three_with_its_own_damping(self):
        # 1 -> 2 -> 3 -> 4 -> 5 -> 6 chain: node 5 sits at depth 3 from node 2,
        # so arc (5, 6) gets the 0.73 damping — dead code before the fix passed
        # the full max_depth to the BFS.
        generator = ArcProbabilityCongestionGenerator(
            event_probability={(1, 2): 1.0},
            successors={1: [2], 2: [3], 3: [4], 4: [5], 5: [6]},
            congestion_lower_bound=0.1,
            congestion_upper_bound=0.9,
            max_congestion_duration=60,
        )
        congested: dict = {}
        # Seed 2 draws a multiplier near the lower bound: 0.73-damped it stays
        # below the upper bound, keeping the depth-3 factor observable.
        np.random.seed(2)
        generator.generate(300, congested)

        epicenter = congested[(1.0, 2.0)][0]
        assert (5.0, 6.0) in congested
        assert congested[(5.0, 6.0)][0] == pytest.approx(epicenter / 0.73)
