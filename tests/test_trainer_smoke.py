"""Ticket 09 CI smoke: the Trainer end-to-end on the committed mini fixture.

A tiny experiment (2 train episodes, one evaluation block, one test seed) runs
the complete config-driven path — ``ExperimentConfig`` -> ``Trainer.from_config``
(CsvDataSource, TravelTimeModel, ShortestPathCache, ClientGenerator, congestion
generator) -> ``run()`` — and must produce finite costs plus the per-run output
files. The world is the 44-day copy of the fixture (single-day speed stds are
NaN, which would poison every travel time; see ``characterization_world``).
"""

import dataclasses
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import characterization_world
from stdvrp.config import ExperimentConfig
from stdvrp.training import (
    ActionCountReport,
    EvaluationBlock,
    ExperimentResult,
    Trainer,
    TrainingResult,
)

FIXTURE_DIR = characterization_world.FIXTURE_DIR
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory: pytest.TempPathFactory) -> ExperimentConfig:
    world = characterization_world.build_legacy_world(tmp_path_factory.mktemp("trainer_world"))
    shutil.copyfile(FIXTURE_DIR / "all_shortest_paths.csv", world / "all_shortest_paths.csv")
    return dataclasses.replace(
        ExperimentConfig.from_yaml(FIXTURE_DIR / "config.yaml"),
        data_dir=world,
        traffic_days=characterization_world.LEGACY_DAYS,
        total_train_iterations=2,
        test_frequency=2,
        evaluation_seed_count=2,
        test_episodes=2,
        test_action_counts=(2,),
        test_seeds=(100,),
        test_vehicle_counts=(4,),
    )


@pytest.fixture(scope="module")
def smoke_run(
    smoke_config: ExperimentConfig, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentResult, Path]:
    output_dir = tmp_path_factory.mktemp("trainer_out") / "run"
    result = Trainer.from_config(smoke_config).run(output_dir)
    return result, output_dir


def test_training_produces_finite_w_and_one_evaluation_block(
    smoke_run: tuple[ExperimentResult, Path],
) -> None:
    result, _ = smoke_run
    training = result.training
    assert len(training.w_trajectory) == 2
    assert all(all(math.isfinite(x) for x in w) for w in training.w_trajectory)
    assert [block.episodes_completed for block in training.evaluations] == [2]
    block = training.evaluations[0]
    assert len(block.seed_costs) == 2
    assert all(math.isfinite(cost) and cost > 0 for cost in block.seed_costs)
    # A single evaluation block is both the newest and the best W.
    assert training.best_w is not None and training.newest_w is not None
    assert list(training.best_w) == list(training.newest_w) == list(result.tested_w)


def test_final_test_metrics_are_finite_and_consistent(
    smoke_run: tuple[ExperimentResult, Path],
) -> None:
    result, _ = smoke_run
    (report,) = result.test
    assert report.action_count == 2
    (entry,) = report.per_seed
    assert (entry.seed, entry.vehicle_count) == (100, 4)
    assert all(math.isfinite(value) for value in entry.metrics.values())
    components = (
        entry.metrics["distance_cost"]
        + entry.metrics["delay_cost"]
        + entry.metrics["earliness_cost"]
        + entry.metrics["overtime_cost"]
    )
    assert entry.metrics["total_cost"] == pytest.approx(components)
    assert report.summary["total_cost"][0] == pytest.approx(entry.metrics["total_cost"])


def test_output_files_are_written(smoke_run: tuple[ExperimentResult, Path]) -> None:
    _, output_dir = smoke_run
    plot = output_dir / "training_plot.png"
    assert plot.stat().st_size > 0

    document: dict[str, Any] = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert document["config"]["total_train_iterations"] == 2
    assert len(document["training"]["w_trajectory"]) == 2
    assert document["training"]["best_w"] == document["tested_w"]
    assert document["test"]["2"]["per_seed"][0]["seed"] == 100


def test_entry_script_wires_config_to_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented command parses the config, runs the Trainer, prints the summary."""
    spec = importlib.util.spec_from_file_location(
        "chengdu_run", REPO_ROOT / "experiments" / "chengdu" / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    w = np.array([1.0])
    fake_result = ExperimentResult(
        training=TrainingResult(
            w_trajectory=(w,),
            evaluations=(EvaluationBlock(episodes_completed=2, seed_costs=(10.0, 20.0)),),
            newest_w=w,
            best_w=w,
        ),
        test=(ActionCountReport(action_count=2, per_seed=(), summary={"total_cost": (15.0, 5.0)}),),
        tested_w=w,
    )
    calls: dict[str, Any] = {}

    class TrainerStub:
        def __init__(self, config: ExperimentConfig) -> None:
            self.config = config

        @classmethod
        def from_config(cls, config: ExperimentConfig, log: Any = None) -> "TrainerStub":
            calls["config"] = config
            return cls(config)

        def run(self, output_dir: Path) -> ExperimentResult:
            calls["output_dir"] = output_dir
            return fake_result

    monkeypatch.setattr(module, "Trainer", TrainerStub)
    module.main(
        ["--config", str(FIXTURE_DIR / "config.yaml"), "--output-dir", str(tmp_path / "out")]
    )

    assert calls["config"].instance_day == 601
    assert calls["output_dir"] == tmp_path / "out"
    output = capsys.readouterr().out
    assert "best evaluation mean cost: 15.0000" in output
    assert "final test actions=2: mean cost 15.0000 (std 5.0000)" in output
