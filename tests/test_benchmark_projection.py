"""Unit tests for the full-run projection math in ``scripts/benchmark_episodes.py``.

The projection is ticket 01's success denominator: it turns the scaled
real-dataset measurements (per-phase per-episode means) into the full-run
wall-clock the effort is measured against, *without ever running the full
experiment*. The arithmetic is pure and lives behind ``project_full_run`` /
``interpolate_test_time`` so it can be pinned here bit-for-bit; the timing I/O
around it is exercised by ``test_benchmark_smoke``.

The benchmark module is loaded via the ``benchmark_module`` fixture (conftest).
"""

from types import ModuleType

import pytest


def test_interpolate_returns_measured_value_at_the_endpoints(benchmark_module: ModuleType) -> None:
    measured = {2: 1.0, 50: 5.0}
    assert benchmark_module.interpolate_test_time(2, measured) == 1.0
    assert benchmark_module.interpolate_test_time(50, measured) == 5.0


def test_interpolate_is_linear_between_endpoints(benchmark_module: ModuleType) -> None:
    measured = {2: 1.0, 50: 5.0}
    # Halfway in action count (26) is halfway in time (3.0); the span is 48.
    assert benchmark_module.interpolate_test_time(26, measured) == pytest.approx(3.0)
    # A quarter of the way (14) -> 1.0 + 0.25 * 4.0.
    assert benchmark_module.interpolate_test_time(14, measured) == pytest.approx(2.0)


def test_interpolate_with_a_single_measured_action_count(benchmark_module: ModuleType) -> None:
    # No span to interpolate against: return the one measured value, not a divide-by-zero.
    assert benchmark_module.interpolate_test_time(30, {2: 4.0}) == 4.0


def test_projection_sums_load_train_eval_and_interpolated_test(
    benchmark_module: ModuleType,
) -> None:
    times = benchmark_module.PhaseTimes(
        world_load_seconds=100.0,
        train_seconds=2.0,
        eval_seconds=3.0,
        test_seconds={2: 4.0, 50: 4.0},  # flat -> every action count costs 4.0
    )
    shape = benchmark_module.FullRunShape(
        train_iterations=100,
        eval_episodes=500,
        test_action_counts=(2, 10, 20, 30, 40, 50),
        test_seed_count=50,
        test_episodes_per_cell=50,
    )
    projection = benchmark_module.project_full_run(times, shape)

    # load + 100*2 + 500*3 + 50*50*(6 action counts * 4.0)
    expected_test = 50 * 50 * (6 * 4.0)
    assert projection["test_seconds"] == pytest.approx(expected_test)
    assert projection["train_seconds"] == pytest.approx(200.0)
    assert projection["eval_seconds"] == pytest.approx(1500.0)
    assert projection["world_load_seconds"] == pytest.approx(100.0)
    assert projection["total_seconds"] == pytest.approx(100.0 + 200.0 + 1500.0 + expected_test)


def test_projection_interpolates_the_middle_action_counts(benchmark_module: ModuleType) -> None:
    times = benchmark_module.PhaseTimes(
        world_load_seconds=0.0,
        train_seconds=0.0,
        eval_seconds=0.0,
        test_seconds={2: 1.0, 50: 5.0},
    )
    shape = benchmark_module.FullRunShape(
        train_iterations=0,
        eval_episodes=0,
        test_action_counts=(2, 50),
        test_seed_count=1,
        test_episodes_per_cell=1,
    )
    projection = benchmark_module.project_full_run(times, shape)
    # Only the two endpoints: 1.0 + 5.0 = 6.0.
    assert projection["total_seconds"] == pytest.approx(6.0)
