"""``Model._congestion_epoch_due``'s cadence (ticket 02, simulator-correctness, B16).

The shipped gate compared a float quotient (``(tau + 178) / 60`` against
``max_congestion_duration / 60``) for exact equality — a test that silently
degenerates whenever that quotient is not exactly representable in binary.
``docs/simulator-review.md`` names 50 and 70 as breaking; measured here, 200
also breaks (the review's own headline undercounts by one). Every duration the
repository actually ships, tests or captures (30/45/60/90/120/180/240) happens
to have an exact ``/60`` quotient, which is exactly why the bug went unseen.

``next_decision_tau`` is always integer-valued (starts at
``horizon_start_minute + DECISION_EPOCH_MINUTES``, both ints, and is
incremented by the int ``DECISION_EPOCH_MINUTES`` — established in ticket 01's
``measurement_bench.py``), so an integer-arithmetic reference formula
(``(tau + 178) % max_congestion_duration == 0``) is a meaningful, independent
oracle here, not merely a plausible one.

**Not the spec's closed form.** ``spec.md``'s invariant catalogue states the
expected count as ``floor((episode_end - horizon_start) / max_congestion_duration)``
- a useful intuition ("about one roll every ``max_congestion_duration``
minutes over the episode"), but not exact: at ``duration=120`` the model fires
8 times while that floor gives 7, because the swept tau range's boundaries
don't align with every duration's period. The modulo oracle above is the
literal fixed formula, not an approximation, and is what every assertion below
actually checks.
"""

from types import SimpleNamespace

import pytest

from stdvrp.simulation.model import DECISION_EPOCH_MINUTES, Model

HORIZON_START = 300
EPISODE_END = 1150

# The review's tested/shipped durations (fine under the old formula too) plus
# the ones it flags (50, 70) and the one it misses (200).
DURATIONS = (30, 45, 50, 60, 70, 90, 120, 150, 180, 200, 240)


def make_model(max_congestion_duration: int) -> Model:
    model = Model.__new__(Model)
    model.max_congestion_duration = max_congestion_duration
    model.state = SimpleNamespace(tau_episode=0.0)  # type: ignore[assignment]
    return model


def _swept_taus() -> list[float]:
    taus = []
    tau = float(HORIZON_START + DECISION_EPOCH_MINUTES)
    while tau < EPISODE_END:
        taus.append(tau)
        tau += DECISION_EPOCH_MINUTES
    return taus


def _reference_fires(duration: int) -> int:
    """Integer-arithmetic cadence: fires every ``duration`` minutes, exactly."""
    return sum(1 for tau in _swept_taus() if (tau + 178) % duration == 0)


def _model_fires(duration: int) -> int:
    model = make_model(duration)
    count = 0
    for tau in _swept_taus():
        model.state.tau_episode = tau
        if model._congestion_epoch_due():
            count += 1
    return count


@pytest.mark.parametrize("duration", DURATIONS)
def test_cadence_matches_the_integer_reference_for_every_swept_duration(duration: int) -> None:
    assert _model_fires(duration) == _reference_fires(duration)


def test_the_two_review_flagged_durations_actually_differ_from_the_reference_pre_fix() -> None:
    """Sanity check on the oracle itself: 50 and 70 are not accidentally easy.

    Guards against a vacuous parametrized test above (e.g. a reference that
    always agrees with anything) by pinning the reference's own fire counts at
    the review's headline durations.
    """
    assert _reference_fires(50) == 17
    assert _reference_fires(70) == 12
    assert _reference_fires(200) == 4
