"""Unit tests for EpisodeVelocities (ticket 09, simulation-performance).

The velocity sampler the Model used to be. These pin the memoization asymmetry
(normal samples remembered, congested ones not), the even-minute bucketing, the
speed caps and the congestion interaction — the behaviors the Tier-1 bit-exact
gate would otherwise only catch as an unexplained drift in an Episode total.

The rng is scripted rather than seeded: every assertion here is about what the
sampler does with a draw, not about the draw itself, and a scripted stream also
makes "was this arc-minute drawn again?" directly observable.
"""

from typing import Any

from stdvrp.simulation.episode_velocities import EpisodeVelocities

ARC = (1.0, 2.0)
LENGTH, SPEED, STD = 4.0, 0.5, 0.1


class ScriptedRng:
    """A ``Generator`` stand-in yielding the given draws in order."""

    def __init__(self, *draws: float) -> None:
        self.draws = list(draws)
        self.calls: list[tuple[float, float]] = []

    def normal(self, loc: float, scale: float) -> float:
        self.calls.append((loc, scale))
        return self.draws[len(self.calls) - 1]


class FakeTravelTimeModel:
    """Only the two lookups the sampler reads, for every even minute of interest."""

    def __init__(self, speed: float = SPEED, std: float = STD) -> None:
        self.travel_data: dict[Any, tuple[float, float]] = {}
        self.speed_std: dict[Any, float] = {}
        for minute in range(298, 320, 2):
            self.travel_data[(*ARC, minute)] = (LENGTH, speed)
            self.speed_std[(*ARC, minute)] = std


def make_velocities(
    *draws: float, speed: float = SPEED, std: float = STD
) -> tuple[EpisodeVelocities, ScriptedRng]:
    rng = ScriptedRng(*draws)
    velocities = EpisodeVelocities(
        FakeTravelTimeModel(speed, std),  # type: ignore[arg-type]
        rng,  # type: ignore[arg-type]
    )
    return velocities, rng


class TestSampling:
    def test_travel_time_is_the_length_over_the_drawn_velocity(self):
        velocities, _ = make_velocities(0.4)
        sampled = velocities.sample(*ARC, 300.0)

        assert sampled.velocity == 0.4
        assert sampled.length == LENGTH
        assert sampled.travel_time == LENGTH / 0.4

    def test_the_draw_is_centred_on_the_arc_minutes_mean_speed_and_deviation(self):
        velocities, rng = make_velocities(0.4)
        velocities.sample(*ARC, 300.0)
        assert rng.calls == [(SPEED, STD)]

    def test_a_slow_arc_is_capped_at_the_sixty_kmh_street_limit(self):
        velocities, _ = make_velocities(1.5, speed=0.9)
        assert velocities.sample(*ARC, 300.0).velocity == 1

    def test_a_fast_arc_keeps_a_draw_over_the_street_limit(self):
        """The 60 km/h cap applies only where the arc's own mean is below it."""
        velocities, _ = make_velocities(1.5, speed=1.2)
        assert velocities.sample(*ARC, 300.0).velocity == 1.5

    def test_every_arc_is_capped_at_the_absolute_limit(self):
        velocities, _ = make_velocities(9.0, speed=1.2)
        assert velocities.sample(*ARC, 300.0).velocity == 2

    def test_a_negative_draw_becomes_the_positive_floor(self):
        velocities, _ = make_velocities(-3.0)
        sampled = velocities.sample(*ARC, 300.0)

        assert sampled.velocity == 0.001
        assert sampled.travel_time == LENGTH / 0.001


class TestMemoization:
    def test_the_same_arc_minute_is_drawn_once(self):
        velocities, rng = make_velocities(0.4, 0.7)

        first = velocities.sample(*ARC, 300.0)

        assert velocities.sample(*ARC, 301.9) == first
        assert len(rng.calls) == 1

    def test_odd_minutes_bucket_back_to_the_even_one_below(self):
        velocities, rng = make_velocities(0.4, 0.7)

        assert velocities.sample(*ARC, 301.0) == velocities.sample(*ARC, 300.0)
        assert len(rng.calls) == 1

    def test_the_next_even_minute_is_drawn_afresh(self):
        velocities, _ = make_velocities(0.4, 0.7)

        assert velocities.sample(*ARC, 300.0).velocity == 0.4
        assert velocities.sample(*ARC, 302.0).velocity == 0.7


class TestCongestion:
    def test_a_congested_arc_is_slowed_by_its_event_multiplier(self):
        velocities, rng = make_velocities()
        velocities.congested_arcs[ARC] = [0.5, 400.0]

        sampled = velocities.sample(*ARC, 300.0)

        assert sampled.velocity == SPEED * 0.5
        assert sampled.travel_time == LENGTH / (SPEED * 0.5)
        assert rng.calls == [], "a congested velocity is derived, never drawn"

    def test_congested_samples_are_not_memoized(self):
        """Preserved quirk: only the normal draws are remembered."""
        velocities, _ = make_velocities(0.4)
        velocities.congested_arcs[ARC] = [0.5, 400.0]
        velocities.sample(*ARC, 300.0)

        del velocities.congested_arcs[ARC]

        assert velocities.sample(*ARC, 300.0).velocity == 0.4

    def test_an_expired_event_leaves_the_arc_uncongested(self):
        velocities, _ = make_velocities(0.4)
        velocities.congested_arcs[ARC] = [0.5, 300.0]
        assert velocities.sample(*ARC, 300.0).velocity == 0.4

    def test_a_congested_arc_never_stops_dead(self):
        velocities, _ = make_velocities()
        velocities.congested_arcs[ARC] = [0.0, 400.0]
        assert velocities.sample(*ARC, 300.0).velocity == 0.0001

    def test_nothing_is_congested_to_begin_with(self):
        velocities, _ = make_velocities()
        assert not velocities.any_congestion

    def test_one_event_makes_the_episode_congested(self):
        velocities, _ = make_velocities()
        velocities.congested_arcs[ARC] = [0.5, 400.0]
        assert velocities.any_congestion

    def test_the_expiry_of_an_uncongested_arc_is_none(self):
        velocities, _ = make_velocities()
        assert velocities.congestion_expiry(ARC) is None

    def test_the_expiry_of_a_congested_arc_is_its_end_minute(self):
        velocities, _ = make_velocities()
        velocities.congested_arcs[ARC] = [0.5, 400.0]
        assert velocities.congestion_expiry(ARC) == 400.0


class TestPurgeExpired:
    def test_purge_removes_an_expired_entry(self):
        velocities, _ = make_velocities()
        velocities.congested_arcs[ARC] = [0.5, 400.0]

        velocities.purge_expired(400.0)

        assert velocities.congested_arcs == {}

    def test_purge_keeps_a_still_active_entry(self):
        velocities, _ = make_velocities()
        velocities.congested_arcs[ARC] = [0.5, 400.0]

        velocities.purge_expired(399.0)

        assert velocities.congested_arcs == {ARC: [0.5, 400.0]}

    def test_purge_removes_only_the_expired_arcs(self):
        velocities, _ = make_velocities()
        active = (2.0, 3.0)
        velocities.congested_arcs[ARC] = [0.5, 400.0]
        velocities.congested_arcs[active] = [0.5, 500.0]

        velocities.purge_expired(400.0)

        assert velocities.congested_arcs == {active: [0.5, 500.0]}

    def test_purge_is_a_no_op_on_an_empty_book(self):
        velocities, _ = make_velocities()
        velocities.purge_expired(300.0)
        assert velocities.congested_arcs == {}


class TestRelease:
    def test_release_forgets_the_memoized_samples(self):
        velocities, _ = make_velocities(0.4, 0.7)
        assert velocities.sample(*ARC, 300.0).velocity == 0.4

        velocities.release()

        assert velocities.sample(*ARC, 300.0).velocity == 0.7

    def test_release_forgets_the_events(self):
        velocities, _ = make_velocities(0.4)
        velocities.congested_arcs[ARC] = [0.5, 400.0]

        velocities.release()

        assert velocities.congested_arcs == {}
        assert velocities.sample(*ARC, 300.0).velocity == 0.4

    def test_release_clears_the_event_book_in_place(self):
        """The Model holds a reference to it; rebinding would strand that reference."""
        velocities, _ = make_velocities()
        events = velocities.congested_arcs

        velocities.release()

        assert velocities.congested_arcs is events
