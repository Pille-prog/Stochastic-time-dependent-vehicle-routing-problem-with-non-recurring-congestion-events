"""CI smoke for the episode benchmark (ticket 01): it runs, no timing assertions.

The benchmark (``scripts/benchmark_episodes.py``) is a measurement tool, not a
correctness gate, so CI only checks that the fixture path executes end to end and
produces finite, positive per-episode times — never a wall-clock threshold (those
are machine-dependent and would make CI flaky). The projection arithmetic is
pinned separately in ``test_benchmark_projection.py``. The benchmark module is
loaded via the ``benchmark_module`` fixture (conftest).
"""

from pathlib import Path
from types import ModuleType

import pytest


def test_fixture_benchmark_runs_and_times_are_finite(
    benchmark_module: ModuleType, tmp_path: Path
) -> None:
    data_dir = benchmark_module.build_fixture_data_dir(tmp_path / "world")
    config = benchmark_module.fixture_config(data_dir)
    world, load_seconds = benchmark_module.load_world(config)
    assert load_seconds > 0

    train_seeds = [config.first_train_seed + i for i in range(1)]
    eval_seeds = list(config.evaluation_seeds[:1])
    training, trained_w = benchmark_module.time_training(world, train_seeds)
    evaluation = benchmark_module.time_evaluation(world, eval_seeds, w=None)

    assert trained_w is not None
    assert training.episodes == 1 and evaluation.episodes == 1
    assert training.per_episode > 0
    assert evaluation.per_episode > 0
    assert training.episodes_per_second > 0


def test_main_dispatches_default_args_to_the_fixture_benchmark(
    benchmark_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --config routes to the fixture benchmark with the parsed episode counts.

    Dispatch only (the real 20s world build is covered once above) — monkeypatch
    the heavy runner so this stays a fast CLI-wiring check.
    """
    calls: dict[str, int] = {}
    monkeypatch.setattr(
        benchmark_module,
        "run_fixture_benchmark",
        lambda n_train, n_eval: calls.update(n_train=n_train, n_eval=n_eval),
    )
    benchmark_module.main(["--train", "3", "--eval", "7"])
    assert calls == {"n_train": 3, "n_eval": 7}
