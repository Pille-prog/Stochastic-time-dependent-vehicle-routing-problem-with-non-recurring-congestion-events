"""Trainer loop logic (ticket 09), isolated from real episodes via stubbed runners.

The Trainer's contract with the episode runners is the seam: these tests replace
``run_training_episode`` / ``run_evaluation_episode`` in the trainer module with
recording stubs and verify the legacy-faithful orchestration — seed sequences, the
warm-up learning-rate quirk, evaluation scheduling with best-W tracking, the final
test over the configured tables, and the per-run output files.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import stdvrp.training.trainer as trainer_module
from stdvrp.config import ExperimentConfig
from stdvrp.simulation import EpisodeResult, TrainingEpisodeResult
from stdvrp.training import Trainer


def make_config(**overrides: Any) -> ExperimentConfig:
    values: dict[str, Any] = {
        "data_dir": Path("."),
        "links_file": "link.csv",
        "shortest_paths_file": "all_shortest_paths.csv",
        "instance_day": 601,
        "traffic_days": (601,),
        "horizon_start_minute": 300,
        "horizon_end_minute": 780,
        "mean_number_clients": 20,
        "client_count_stddev": 4.0,
        "min_number_clients": 8,
        "clients_per_vehicle": 4,
        "time_window_spread": 60,
        "client_universe_seed": 0,
        "client_universe_size": 10,
        "client_universe_node_range": (1, 45),
        "congestion_lower_bound": 0.3,
        "congestion_upper_bound": 0.4,
        "max_congestion_duration": 120,
        "total_train_iterations": 3,
        "test_frequency": 5,
        "learning_rate": 1.0e-5,
        "warmup_learning_rate": 1.0e-6,
        "epsilon": 0.1,
        "n_observed_arcs": 3,
        "first_train_seed": 1000,
        "evaluation_seed_start": 100000,
        "evaluation_seed_count": 2,
        "test_episodes": 1,
        "test_action_counts": (2,),
        "test_seeds": (100, 101),
        "test_vehicle_counts": (6, 5),
        "train_exploration_seed_offset": 10_000_000,
        "train_repair_seed_offset": 20_000_000,
        "static_policy_mean_cost": None,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def make_trainer(config: ExperimentConfig) -> Trainer:
    """World components are opaque to the Trainer's loop logic; sentinels suffice."""
    return Trainer(
        config,
        client_generator=object(),
        travel_time_model=object(),
        shortest_path_cache=object(),
        congestion_generator=object(),
    )


def episode_result(total_cost: float = 100.0) -> EpisodeResult:
    return EpisodeResult(
        total_cost=total_cost,
        distance_cost=total_cost / 2,
        delay_cost=total_cost / 2,
        earliness_cost=0.0,
        overtime_cost=0.0,
        tau=700.0,
        state_count=10,
        delay_clients=3,
        earliness_clients=2,
    )


class TrainingStub:
    """Records every call and returns W vectors [1], [2], [3], ..."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> TrainingEpisodeResult:
        self.calls.append(kwargs)
        w = np.array([float(len(self.calls))])
        return TrainingEpisodeResult(w=w, episode=episode_result())


class EvaluationStub:
    """Records every call and returns queued (or constant) total costs."""

    def __init__(self, costs: list[float] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.costs = costs

    def __call__(self, **kwargs: Any) -> EpisodeResult:
        self.calls.append(kwargs)
        cost = self.costs.pop(0) if self.costs else 100.0
        return episode_result(cost)


@pytest.fixture
def training_stub(monkeypatch: pytest.MonkeyPatch) -> TrainingStub:
    stub = TrainingStub()
    monkeypatch.setattr(trainer_module, "run_training_episode", stub)
    return stub


@pytest.fixture
def evaluation_stub(monkeypatch: pytest.MonkeyPatch) -> EvaluationStub:
    stub = EvaluationStub()
    monkeypatch.setattr(trainer_module, "run_evaluation_episode", stub)
    return stub


def test_train_seeds_warmup_lr_and_w_chaining(
    training_stub: TrainingStub, evaluation_stub: EvaluationStub
) -> None:
    trainer = make_trainer(make_config())
    result = trainer.train()

    assert [call["seed"] for call in training_stub.calls] == [1000, 1001, 1002]
    # Legacy warm-up quirk: the FIRST Episode uses the hardcoded tiny rate.
    assert [call["learning_rate"] for call in training_stub.calls] == [1.0e-6, 1.0e-5, 1.0e-5]
    # W chains from episode to episode, starting lazily from None.
    assert training_stub.calls[0]["W"] is None
    assert list(training_stub.calls[1]["W"]) == [1.0]
    assert list(training_stub.calls[2]["W"]) == [2.0]
    # Capture-convention exploration seeding: offset + train seed.
    assert [call["exploration_seed"] for call in training_stub.calls] == [
        10_001_000,
        10_001_001,
        10_001_002,
    ]
    assert [call["repair_seed"] for call in training_stub.calls] == [
        20_001_000,
        20_001_001,
        20_001_002,
    ]
    assert [list(w) for w in result.w_trajectory] == [[1.0], [2.0], [3.0]]
    # test_frequency 5 > 3 episodes: no evaluation block ran.
    assert result.evaluations == ()
    assert result.newest_w is None and result.best_w is None
    assert evaluation_stub.calls == []


def test_null_offsets_leave_exploration_unseeded(
    training_stub: TrainingStub, evaluation_stub: EvaluationStub
) -> None:
    config = make_config(train_exploration_seed_offset=None, train_repair_seed_offset=None)
    make_trainer(config).train()
    assert all(call["exploration_seed"] is None for call in training_stub.calls)
    assert all(call["repair_seed"] is None for call in training_stub.calls)


def test_evaluation_blocks_and_best_w_tracking(
    training_stub: TrainingStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two blocks (after episodes 2 and 4); the second evaluates WORSE than the first.
    evaluation_stub = EvaluationStub(costs=[10.0, 20.0, 50.0, 60.0])
    monkeypatch.setattr(trainer_module, "run_evaluation_episode", evaluation_stub)

    config = make_config(total_train_iterations=4, test_frequency=2)
    result = make_trainer(config).train()

    assert [block.episodes_completed for block in result.evaluations] == [2, 4]
    assert result.evaluations[0].seed_costs == (10.0, 20.0)
    assert result.evaluations[0].mean_cost == 15.0
    assert result.evaluations[1].mean_cost == 55.0
    # Every evaluation Episode ran with the block's newest W and eval seeds.
    assert [call["seed"] for call in evaluation_stub.calls] == [100000, 100001] * 2
    assert all(list(call["W"]) == [2.0] for call in evaluation_stub.calls[:2])
    assert all(list(call["W"]) == [4.0] for call in evaluation_stub.calls[2:])
    # The evaluation runs the generated fleet with the default action pool.
    assert all(
        call["vehicle_count"] is None and call["number_actions_test"] is None
        for call in evaluation_stub.calls
    )
    # Best W is the FIRST block's (lower mean); newest is the LAST block's.
    assert result.best_w is not None and list(result.best_w) == [2.0]
    assert result.newest_w is not None and list(result.newest_w) == [4.0]
    # Recorded Ws are copies, not aliases of the live vector.
    assert not np.shares_memory(result.best_w, result.w_trajectory[1])


def test_final_test_walks_the_configured_tables(
    training_stub: TrainingStub, evaluation_stub: EvaluationStub
) -> None:
    config = make_config(test_action_counts=(2, 5), test_episodes=2)
    trainer = make_trainer(config)
    best_w = np.array([7.0])
    reports = trainer.final_test(best_w)

    # 2 action counts x 2 seeds x 2 episodes.
    assert len(evaluation_stub.calls) == 8
    assert [call["seed"] for call in evaluation_stub.calls] == [100, 100, 101, 101] * 2
    assert [call["vehicle_count"] for call in evaluation_stub.calls] == [6, 6, 5, 5] * 2
    # The action pool widens by the action count on top of the fleet size.
    assert [call["number_actions_test"] for call in evaluation_stub.calls] == [
        8,
        8,
        7,
        7,
        11,
        11,
        10,
        10,
    ]
    assert all(list(call["W"]) == [7.0] for call in evaluation_stub.calls)

    assert [report.action_count for report in reports] == [2, 5]
    per_seed = reports[0].per_seed
    assert [(entry.seed, entry.vehicle_count) for entry in per_seed] == [(100, 6), (101, 5)]
    # Metrics are means over test_episodes identical runs.
    assert per_seed[0].metrics["total_cost"] == 100.0
    assert per_seed[0].metrics["state_count"] == 10.0
    # Summary is mean/std across seeds.
    assert reports[0].summary["total_cost"] == (100.0, 0.0)


def test_run_writes_results_and_plot(
    training_stub: TrainingStub, evaluation_stub: EvaluationStub, tmp_path: Path
) -> None:
    config = make_config(total_train_iterations=2, test_frequency=2)
    result = make_trainer(config).run(tmp_path / "run")

    results_path = tmp_path / "run" / "results.json"
    plot_path = tmp_path / "run" / "training_plot.png"
    assert results_path.is_file() and plot_path.stat().st_size > 0

    document = json.loads(results_path.read_text(encoding="utf-8"))
    assert document["config"]["first_train_seed"] == 1000
    assert document["training"]["w_trajectory"] == [[1.0], [2.0]]
    assert document["training"]["evaluations"] == [
        {"episodes_completed": 2, "seed_costs": [100.0, 100.0], "mean_cost": 100.0}
    ]
    assert document["training"]["best_w"] == [2.0]
    assert document["tested_w"] == [2.0]
    assert document["test"]["2"]["per_seed"][0]["seed"] == 100
    assert document["test"]["2"]["summary"]["total_cost"] == {"mean": 100.0, "std": 0.0}
    assert list(result.tested_w) == [2.0]
    assert result.training.best_mean_cost == 100.0


def test_run_falls_back_to_final_w_when_no_evaluation_ran(
    training_stub: TrainingStub, evaluation_stub: EvaluationStub, tmp_path: Path
) -> None:
    # 3 episodes, frequency 5: the legacy would test with Best_W = [] and crash;
    # the Trainer documents and uses the final trained W instead.
    result = make_trainer(make_config()).run(tmp_path / "run")
    assert list(result.tested_w) == [3.0]
    final_test_calls = [call for call in evaluation_stub.calls if call["vehicle_count"] is not None]
    assert all(list(call["W"]) == [3.0] for call in final_test_calls)

    document = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert document["training"]["best_w"] is None


def test_config_serialization_round_trips_paths_and_tuples(tmp_path: Path) -> None:
    config = make_config(data_dir=tmp_path)
    document = trainer_module.config_as_json(config)
    assert document["data_dir"] == str(tmp_path)
    assert document["test_seeds"] == [100, 101]
    assert json.loads(json.dumps(document)) == document


def test_config_replace_still_validates() -> None:
    with pytest.raises(ValueError, match="test_seeds"):
        dataclasses.replace(make_config(), test_seeds=())
