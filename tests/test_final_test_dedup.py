"""Ticket 02 (simulation-performance): pins the final-test dedup's numeric effect.

``Trainer.final_test`` used to repeat every (action count, seed) episode
``test_episodes`` times and average; per-seed Generators (ticket 13) make every
repeat bit-identical, so the mean equals the single episode's value up to
float summation order. This test reimplements the retired legacy sum/k loop
locally (it is no longer in production code) against the real mini-fixture
world and documents exactly how the two compare: identical on most metrics,
at most one ULP off on the rest — never worse. The benchmark module is loaded
via the ``benchmark_module`` fixture (conftest).
"""

import dataclasses
import math
from pathlib import Path
from types import ModuleType
from typing import Any

from stdvrp.simulation import run_evaluation_episode
from stdvrp.training import Trainer
from stdvrp.training.trainer import EPISODE_METRICS


def _legacy_mean(
    world: Any, w: Any, seed: int, vehicle_count: int, action_count: int, test_episodes: int
) -> dict[str, float]:
    """The retired ``Trainer.final_test`` inner loop: sum ``test_episodes``
    bit-identical episodes, then divide — reimplemented here only to document
    its last-bit float behavior against the new single-episode computation."""
    totals = dict.fromkeys(EPISODE_METRICS, 0.0)
    for _ in range(test_episodes):
        episode = run_evaluation_episode(
            seed=seed,
            W=w,
            vehicle_count=vehicle_count,
            number_actions_test=vehicle_count + action_count,
            **world.episode_kwargs(),
        )
        for name in EPISODE_METRICS:
            totals[name] += float(getattr(episode, name))
    return {name: value / test_episodes for name, value in totals.items()}


def test_deduplicated_metrics_match_the_legacy_mean_within_one_ulp(
    benchmark_module: ModuleType, tmp_path: Path
) -> None:
    data_dir = benchmark_module.build_fixture_data_dir(tmp_path / "world")
    config = dataclasses.replace(
        benchmark_module.fixture_config(data_dir),
        total_train_iterations=2,
        test_frequency=2,
        test_episodes=7,  # deliberately not a power of two, to surface sum/k rounding
        test_action_counts=(2,),
        test_seeds=(100, 101),
        test_vehicle_counts=(4, 4),
    )
    world, _ = benchmark_module.load_world(config)
    trainer = Trainer(
        config,
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
    )
    training = trainer.train()
    w = training.best_w if training.best_w is not None else training.w_trajectory[-1]

    (report,) = trainer.final_test(w)

    exact_matches = 0
    ulp_matches = 0
    for entry in report.per_seed:
        legacy = _legacy_mean(
            world, w, entry.seed, entry.vehicle_count, report.action_count, config.test_episodes
        )
        for name in EPISODE_METRICS:
            deduped, mean = entry.metrics[name], legacy[name]
            if deduped == mean:
                exact_matches += 1
                continue
            ulp_matches += 1
            assert abs(deduped - mean) <= math.ulp(mean), (
                f"seed {entry.seed} {name}: deduplicated value {deduped!r} drifted "
                f"from the legacy sum/{config.test_episodes} mean {mean!r} by more than "
                "one ULP"
            )

    # Document that the drift is real (not every metric happens to divide
    # exactly) but always within the documented one-ULP bound.
    assert exact_matches > 0
    assert ulp_matches > 0
