"""Unit tests for ArcProbabilityCongestionGenerator on a tiny hand-built network.

Ticket 13 (RNG modernization): the generator now draws from an injected
``rng: np.random.Generator`` instead of the global ``np.random`` stream. These
tests use ``ScriptedRng``, a tiny test double that returns a scripted sequence of
``uniform`` draws, so scenarios (an event triggers, a spread stays observable
below the upper bound, ...) are expressed directly instead of by hunting for a
numpy seed that happens to land in the right region.
"""

import pytest

from stdvrp.congestion import ArcProbabilityCongestionGenerator

# 1 -> 2 -> 3 -> 4 chain with a 2 -> 5 branch.
SUCCESSORS = {1: [2], 2: [3, 5], 3: [4], 5: []}


class ScriptedRng:
    """A minimal ``np.random.Generator`` double: ``uniform`` returns a fixed script."""

    def __init__(self, draws: list[float]) -> None:
        self._draws = list(draws)
        self.calls = 0

    def uniform(self, low: float, high: float) -> float:
        self.calls += 1
        value = self._draws.pop(0)
        assert low <= value <= high, f"scripted draw {value} outside [{low}, {high}]"
        return value


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
        rng = ScriptedRng([0.5, 0.5, 0.5])  # never below probability 0.0

        generator.generate(300, congested, rng)

        assert rng.calls == 3  # exactly one uniform consumed per arc, hit or miss
        assert congested == {}

    def test_probability_input_zero_disables_generation_and_rng(self):
        generator = make_generator({(1, 2): 1.0})
        generator.probability_input = 0
        congested: dict = {}
        rng = ScriptedRng([])  # any draw would raise IndexError

        generator.generate(300, congested, rng)

        assert congested == {}
        assert rng.calls == 0


class TestEventValues:
    def test_triggered_event_has_bounded_multiplier_and_duration(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.2, upper=0.4)
        congested: dict = {}
        # probability draw < 1.0 always triggers; multiplier and duration are
        # scripted at fixed points inside their ranges.
        rng = ScriptedRng([0.0, 0.3, 45.0])

        generator.generate(300, congested, rng)

        assert congested[(1.0, 2.0)] == [0.3, 345.0]
        assert all(isinstance(k[0], float) and isinstance(k[1], float) for k in congested)

    def test_event_spreads_to_neighbors_with_damped_intensity(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.1, upper=0.9)
        congested: dict = {}
        # A low multiplier (0.2) stays well under the upper bound even after the
        # 0.83 depth-1 damping divides it up, keeping the damping observable.
        rng = ScriptedRng([0.0, 0.2, 30.0])

        generator.generate(300, congested, rng)

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
        rng = ScriptedRng([0.0, 0.5, 30.0])

        generator.generate(300, congested, rng)

        assert congested[(2.0, 3.0)] == [0.1, 400.0]

    def test_weaker_existing_event_is_overwritten(self):
        generator = make_generator({(1, 2): 1.0}, lower=0.2, upper=0.2)
        congested: dict = {(2.0, 3.0): [0.9, 400.0]}  # weaker (higher multiplier)
        rng = ScriptedRng([0.0, 0.2, 30.0])

        generator.generate(300, congested, rng)

        assert congested[(2.0, 3.0)][0] != 0.9


class TestPhase2Fixes:
    """Ticket 12 fix 7 (ADR-0001 change log): bounded spread, live depth 3."""

    def test_spread_multipliers_never_exceed_the_upper_bound(self):
        # lower == upper pins the drawn multiplier at 0.3; undamped spread would
        # store 0.3/0.83 and 0.3/0.78, both above the configured upper bound.
        generator = make_generator({(1, 2): 1.0}, lower=0.3, upper=0.3)
        congested: dict = {}
        rng = ScriptedRng([0.0, 0.3, 30.0])

        generator.generate(300, congested, rng)

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
        # A low multiplier stays below the upper bound even after 0.73 damping,
        # keeping the depth-3 factor observable.
        rng = ScriptedRng([0.0, 0.2, 30.0])

        generator.generate(300, congested, rng)

        epicenter = congested[(1.0, 2.0)][0]
        assert (5.0, 6.0) in congested
        assert congested[(5.0, 6.0)][0] == pytest.approx(epicenter / 0.73)
