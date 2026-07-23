"""Trainer: the complete experiment as one config-driven run (ticket 09).

Ports ``training_and_testing.training_model`` / ``test_model`` (ADR-0001): the
training loop with periodic evaluation and best-W tracking, then the final test
over the configured seed/vehicle tables, writing ``results.json`` and the
training plot to a per-run output directory. Every value the legacy hardcoded —
horizon, ``n_arcs``, the warm-up learning rate, the evaluation seed range, the
test seed/fleet tables and the ``mean_static_policy`` plot baseline — comes from
``ExperimentConfig``.

Legacy fidelity notes:

* **Warm-up learning rate** (pending ticket 12 triage): the first training
  Episode always updates W with ``warmup_learning_rate``; every later Episode
  uses ``learning_rate`` — exactly the legacy's ``lr`` reassignment quirk.
* **Evaluation blocks**: after every ``test_frequency`` episodes, the newest W
  is evaluated greedily over ``evaluation_seeds`` (generated fleet, default
  ``vehicles + 2`` action pool); the block with the lowest mean cost pins
  ``best_w``, mirroring ``Q_pred`` / ``Best_W``. The legacy's ``Q_pred`` starts
  at 1e11; ``math.inf`` here is behavior-equivalent for any real cost.
* **Best-W fallback**: with fewer episodes than ``test_frequency`` the legacy
  would run its test with ``Best_W = []`` and crash; the Trainer falls back to
  the final trained W instead (documented deviation, same information).
* **Final test**: each (action count, seed) pair reruns ``test_episodes``
  episodes and averages, verbatim from ``test_model`` — every episode fully
  reseeds both RNG streams, so the iterations are identical and the mean equals
  a single episode's value.
* **Exploration seeding** (golden-capture convention, ticket 04 finding 1): the
  configured offsets seed the otherwise-unseeded training exploration RNGs per
  episode; ``None`` restores the legacy's nondeterministic training.
* **Reported metrics**: the nine golden-pinned Episode metrics. The legacy
  report's three mean-time metrics (``mean_delay_time``, ``mean_earliness_time``,
  ``mean_overtime``) were not ported with the ticket 07 Model and are not pinned
  by the golden master (ADR-0001 ticket 09 addendum).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure
from numpy.typing import NDArray

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator, CongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.traffic import CsvDataSource, TravelTimeModel

W = NDArray[np.float64]

EPISODE_METRICS = (
    "total_cost",
    "distance_cost",
    "delay_cost",
    "earliness_cost",
    "overtime_cost",
    "tau",
    "state_count",
    "delay_clients",
    "earliness_clients",
)


@dataclass(frozen=True, slots=True)
class EvaluationBlock:
    """One periodic evaluation: greedy episode costs with the newest W."""

    episodes_completed: int
    seed_costs: tuple[float, ...]

    @property
    def mean_cost(self) -> float:
        return sum(self.seed_costs) / len(self.seed_costs)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The training loop's outcome: W after every Episode plus the evaluations."""

    w_trajectory: tuple[W, ...]
    evaluations: tuple[EvaluationBlock, ...]
    # The last evaluated W and the best-evaluated W (legacy Newest_W / Best_W);
    # None when no evaluation block ran.
    newest_w: W | None
    best_w: W | None

    @property
    def best_mean_cost(self) -> float | None:
        """The best evaluation-block mean (the one that pinned ``best_w``)."""
        if not self.evaluations:
            return None
        return min(block.mean_cost for block in self.evaluations)


@dataclass(frozen=True, slots=True)
class SeedTestResult:
    """Final-test metrics for one seed: means over ``test_episodes`` runs."""

    seed: int
    vehicle_count: int
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ActionCountReport:
    """The final test at one action-pool width: per-seed metrics and their spread."""

    action_count: int
    per_seed: tuple[SeedTestResult, ...]
    # metric name -> (mean, population std) across seeds, as the legacy reports.
    summary: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything one experiment run produced."""

    training: TrainingResult
    test: tuple[ActionCountReport, ...]
    tested_w: W


class Trainer:
    """Runs the experiment an ExperimentConfig describes over a loaded world."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        client_generator: ClientGenerator,
        travel_time_model: TravelTimeModel,
        shortest_path_cache: ShortestPathCache,
        congestion_generator: CongestionGenerator,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.client_generator = client_generator
        self.travel_time_model = travel_time_model
        self.shortest_path_cache = shortest_path_cache
        self.congestion_generator = congestion_generator
        self._log = log if log is not None else lambda message: None

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, *, log: Callable[[str], None] | None = None
    ) -> Trainer:
        """Load the world from the config's DataSource and wire the Trainer."""
        source = CsvDataSource.from_config(config)
        travel_time_model = TravelTimeModel(
            source.load_road_network(),
            source.load_traffic_history(),
            config.max_congestion_duration,
            horizon_start_minute=config.horizon_start_minute,
        )
        return cls(
            config,
            client_generator=ClientGenerator.from_config(config),
            travel_time_model=travel_time_model,
            shortest_path_cache=source.load_shortest_path_cache(),
            congestion_generator=ArcProbabilityCongestionGenerator(
                event_probability=travel_time_model.event_probability,
                successors=travel_time_model.successors,
                congestion_lower_bound=config.congestion_lower_bound,
                congestion_upper_bound=config.congestion_upper_bound,
                max_congestion_duration=config.max_congestion_duration,
            ),
            log=log,
        )

    def run(self, output_dir: Path) -> ExperimentResult:
        """Train, run the final test, and write results + plot into ``output_dir``."""
        config = self.config
        training = self.train()
        tested_w = training.best_w if training.best_w is not None else training.w_trajectory[-1]
        test = self.final_test(tested_w)
        result = ExperimentResult(training=training, test=test, tested_w=tested_w)

        output_dir.mkdir(parents=True, exist_ok=True)
        write_results(output_dir / "results.json", config, result)
        write_training_plot(
            output_dir / "training_plot.png", training.evaluations, config.static_policy_mean_cost
        )
        self._log(f"results written to {output_dir}")
        return result

    def train(self) -> TrainingResult:
        config = self.config
        w: W | None = None
        # Legacy warm-up quirk: the first Episode trains with the tiny rate.
        learning_rate = config.warmup_learning_rate
        w_trajectory: list[W] = []
        evaluations: list[EvaluationBlock] = []
        newest_w: W | None = None
        best_w: W | None = None
        best_mean_cost = math.inf

        for index in range(config.total_train_iterations):
            seed = config.first_train_seed + index
            result = run_training_episode(
                seed=seed,
                W=w,
                learning_rate=learning_rate,
                exploration_seed=_offset_seed(config.train_exploration_seed_offset, seed),
                repair_seed=_offset_seed(config.train_repair_seed_offset, seed),
                **self._episode_kwargs(),
            )
            learning_rate = config.learning_rate
            w = result.w
            w_trajectory.append(_copy_w(result.w))
            episodes_completed = index + 1
            self._log(f"train episode {episodes_completed} (seed {seed}) done")

            if episodes_completed % config.test_frequency == 0:
                newest_w = _copy_w(w)
                block = EvaluationBlock(
                    episodes_completed=episodes_completed,
                    seed_costs=tuple(
                        self._evaluation_cost(eval_seed, newest_w)
                        for eval_seed in config.evaluation_seeds
                    ),
                )
                evaluations.append(block)
                self._log(
                    f"evaluation after {episodes_completed} episodes: "
                    f"mean cost {block.mean_cost:.4f}"
                )
                if block.mean_cost < best_mean_cost:
                    best_mean_cost = block.mean_cost
                    best_w = newest_w

        return TrainingResult(
            w_trajectory=tuple(w_trajectory),
            evaluations=tuple(evaluations),
            newest_w=newest_w,
            best_w=best_w,
        )

    def final_test(self, w: W) -> tuple[ActionCountReport, ...]:
        """The legacy ``test_model``: fixed seed/fleet tables at widening action pools."""
        config = self.config
        reports = []
        for action_count in config.test_action_counts:
            per_seed = []
            for seed, vehicle_count in zip(
                config.test_seeds, config.test_vehicle_counts, strict=True
            ):
                totals = dict.fromkeys(EPISODE_METRICS, 0.0)
                for _ in range(config.test_episodes):
                    episode = run_evaluation_episode(
                        seed=seed,
                        W=w,
                        vehicle_count=vehicle_count,
                        number_actions_test=vehicle_count + action_count,
                        **self._episode_kwargs(),
                    )
                    for name in EPISODE_METRICS:
                        totals[name] += float(getattr(episode, name))
                metrics = {name: value / config.test_episodes for name, value in totals.items()}
                per_seed.append(SeedTestResult(seed, vehicle_count, metrics))
            summary = {
                name: _mean_and_std([entry.metrics[name] for entry in per_seed])
                for name in EPISODE_METRICS
            }
            reports.append(ActionCountReport(action_count, tuple(per_seed), summary))
            self._log(
                f"final test actions={action_count}: mean cost {summary['total_cost'][0]:.4f}"
            )
        return tuple(reports)

    def _evaluation_cost(self, seed: int, w: W) -> float:
        """One greedy evaluation Episode: generated fleet, default action pool."""
        episode = run_evaluation_episode(
            seed=seed, W=w, vehicle_count=None, number_actions_test=None, **self._episode_kwargs()
        )
        return episode.total_cost

    def _episode_kwargs(self) -> dict[str, Any]:
        """The world and config arguments every episode runner shares."""
        config = self.config
        return {
            "client_generator": self.client_generator,
            "travel_time_model": self.travel_time_model,
            "shortest_path_cache": self.shortest_path_cache,
            "congestion_generator": self.congestion_generator,
            "epsilon": config.epsilon,
            "max_congestion_duration": config.max_congestion_duration,
            "horizon_start_minute": config.horizon_start_minute,
            "horizon_end_minute": config.horizon_end_minute,
            "n_observed_arcs": config.n_observed_arcs,
        }


def _copy_w(w: W) -> W:
    return np.array(w, dtype=np.float64, copy=True)


def _offset_seed(offset: int | None, seed: int) -> int | None:
    return None if offset is None else offset + seed


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def config_as_json(config: ExperimentConfig) -> dict[str, Any]:
    """The config as JSON-serializable values (Paths to str, tuples to lists)."""
    document = dataclasses.asdict(config)
    document["data_dir"] = str(config.data_dir)
    return {
        name: list(value) if isinstance(value, tuple) else value for name, value in document.items()
    }


def write_results(path: Path, config: ExperimentConfig, result: ExperimentResult) -> None:
    """Write results.json: the config snapshot plus everything the run produced."""
    document = {
        "config": config_as_json(config),
        "training": {
            "w_trajectory": [_w_as_json(w) for w in result.training.w_trajectory],
            "evaluations": [
                {
                    "episodes_completed": block.episodes_completed,
                    "seed_costs": list(block.seed_costs),
                    "mean_cost": block.mean_cost,
                }
                for block in result.training.evaluations
            ],
            "newest_w": _w_as_json(result.training.newest_w),
            "best_w": _w_as_json(result.training.best_w),
        },
        "tested_w": _w_as_json(result.tested_w),
        "test": {
            str(report.action_count): {
                "per_seed": [
                    {"seed": entry.seed, "vehicles": entry.vehicle_count, **entry.metrics}
                    for entry in report.per_seed
                ],
                "summary": {
                    name: {"mean": mean, "std": std} for name, (mean, std) in report.summary.items()
                },
            }
            for report in result.test
        },
    }
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")


def _w_as_json(w: W | None) -> list[float] | None:
    return None if w is None else [float(x) for x in w]


def write_training_plot(
    path: Path, evaluations: tuple[EvaluationBlock, ...], static_policy_mean_cost: float | None
) -> None:
    """The legacy training plot: evaluation means with the static-policy baseline.

    Rendered through a directly constructed Figure (no pyplot): backend-independent,
    headless-safe, and free of pyplot's global figure registry.
    """
    figure = Figure(figsize=(20, 5))
    axes = figure.subplots()
    axes.plot(
        [block.episodes_completed for block in evaluations],
        [block.mean_cost for block in evaluations],
        marker="o",
        linestyle="-",
        label="Cost",
    )
    if static_policy_mean_cost is not None:
        axes.axhline(
            y=static_policy_mean_cost, color="red", linestyle=":", label="Mean Static Policy"
        )
    axes.set_title("Objective Function under Greedy Policy during Training")
    axes.set_xlabel("Number of Episodes")
    axes.set_ylabel("Objective Function")
    axes.legend()
    # The legacy forces scientific notation at 10^3 on both axes.
    for axis in (axes.xaxis, axes.yaxis):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((3, 3))
        axis.set_major_formatter(formatter)
    figure.savefig(path)
