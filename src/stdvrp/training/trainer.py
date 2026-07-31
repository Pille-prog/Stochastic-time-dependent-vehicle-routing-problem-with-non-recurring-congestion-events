"""Trainer: the complete experiment as one config-driven run (ticket 09).

Ports ``training_and_testing.training_model`` / ``test_model`` (ADR-0001): the
training loop with periodic evaluation and best-W tracking, then the final test
over the configured seed/vehicle tables, writing ``results.json`` and the
training plot to a per-run output directory. Every value the legacy hardcoded —
horizon, ``n_arcs``, the warm-up learning rate, the evaluation seed range, the
test seed/fleet tables — comes from ``ExperimentConfig``. The legacy's
hardcoded ``mean_static_policy`` plot baseline is a ``ReferenceCard``
(``stdvrp.training.reference_card``, ticket 01) passed to :meth:`Trainer.run`
instead — a frozen per-seed cost vector, not a config field.

Legacy fidelity notes:

* **Warm-up learning rate** (made optional in ticket 12): when
  ``warmup_learning_rate`` is set, the first training Episode updates W with it
  and every later Episode uses ``learning_rate`` — exactly the legacy's ``lr``
  reassignment quirk; ``null`` applies ``learning_rate`` from episode 1.
* **Evaluation blocks**: after every ``test_frequency`` episodes, the newest W
  is evaluated greedily over ``evaluation_seeds`` (generated fleet, default
  ``vehicles + 2`` action pool); the block with the lowest mean cost pins
  ``best_w``, mirroring ``Q_pred`` / ``Best_W``. The legacy's ``Q_pred`` starts
  at 1e11; ``math.inf`` here is behavior-equivalent for any real cost.
* **Best-W fallback**: with fewer episodes than ``test_frequency`` the legacy
  would run its test with ``Best_W = []`` and crash; the Trainer falls back to
  the final trained W instead (documented deviation, same information).
* **Final test** (deduplicated, ticket 02, simulation-performance): each
  (action count, seed) pair runs a **single** evaluation episode. Every
  episode draws its own per-Episode Generators from its seed (ticket 13,
  ADR-0001 phase 2), so the legacy ``test_episodes`` repeats of the same
  episode were bit-identical and their mean equaled one episode's value —
  computed directly now instead of summed and divided (avoiding the sum/k
  division-order float noise that quirk could otherwise leak into the
  report). ``test_episodes`` stays in ``ExperimentConfig`` for config-file
  compatibility but is no longer read.
* **Reported metrics**: the ten golden-pinned Episode metrics (``unserved_clients``
  added by ticket 03, simulator-correctness, ADR-0004). The legacy report's
  three mean-time metrics (``mean_delay_time``, ``mean_earliness_time``,
  ``mean_overtime``) were not ported with the ticket 07 Model and are not pinned
  by the golden master (ADR-0001 ticket 09 addendum).

Parallelism (ticket 08, simulation-performance): the evaluation blocks and the
final test are batches of independent Episodes, so they go through
``stdvrp.training.episode_pool`` — a persistent pool of spawned workers, each
holding one world loaded from the ticket-03 binary cache. Results come back in
request order, so every reduction below sees the serial loops' seed order and
``results.json`` stays bit-identical. ``worker_count=1`` (the default) keeps
everything in this process. Training itself stays sequential: W is a serial
dependency.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure

from stdvrp.config import ExperimentConfig
from stdvrp.simulation import EpisodeResult, run_training_episode
from stdvrp.training.episode_pool import EpisodePool, EpisodeRequest, EpisodeWorld, W
from stdvrp.training.reference_card import ReferenceCard

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
    "unserved_clients",
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
    """Final-test metrics for one seed: one evaluation episode's raw values.

    (Ticket 02, simulation-performance) The legacy ``test_episodes`` repeats of
    this episode were bit-identical, so their mean equaled this value anyway.
    """

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
        world: EpisodeWorld,
        *,
        episode_pool: EpisodePool | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.world = world
        self.config = world.config
        # None (the default) evaluates in this process; a pool spreads the
        # evaluation blocks and the final test over worker processes (ticket 08).
        self.episode_pool = episode_pool
        self._log = log if log is not None else lambda message: None

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        cache_dir: Path | None = None,
        worker_count: int = 1,
        log: Callable[[str], None] | None = None,
    ) -> Trainer:
        """Load the world from the config's DataSource and wire the Trainer.

        ``cache_dir`` (ticket 03, simulation-performance) opts into the binary
        world cache: ``None`` (the default) parses the CSVs fresh every call,
        exactly as before; a directory reuses a matching prior snapshot instead
        of re-parsing, and writes one on a miss. See ``stdvrp.traffic.world_cache``.

        ``worker_count`` (ticket 08) is how many worker processes evaluate on: 1
        (the default) keeps every episode in this process, more than 1 needs
        ``cache_dir`` because every worker loads its own world through it. The
        results are identical either way — see ``stdvrp.training.episode_pool``.
        """
        # Before the (expensive) world load, so a bad worker/cache combination
        # fails immediately rather than after parsing the CSVs.
        episode_pool = EpisodePool.for_worker_count(
            config, cache_dir=cache_dir, worker_count=worker_count
        )
        world = EpisodeWorld.load(config, cache_dir=cache_dir)
        return cls(world, episode_pool=episode_pool, log=log)

    def close(self) -> None:
        """Shut the evaluation worker pool down; a no-op when running serially."""
        if self.episode_pool is not None:
            self.episode_pool.close()

    def __enter__(self) -> Trainer:
        """``run()`` closes its own pool; this is for driving ``train``/``final_test``
        directly, where nothing else would (benchmarks, tests, notebooks)."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def run(
        self, output_dir: Path, *, reference_card: ReferenceCard | None = None
    ) -> ExperimentResult:
        """Train, run the final test, and write results + plot into ``output_dir``.

        ``reference_card`` (ticket 01) is the linear baseline's frozen per-seed
        cost vector; when given, the training plot draws it as the comparison
        line ``config.static_policy_mean_cost`` used to be. ``None`` (the
        default) omits the line — what a run producing the card itself uses.
        """
        config = self.config
        try:
            training = self.train()
            tested_w = training.best_w if training.best_w is not None else training.w_trajectory[-1]
            test = self.final_test(tested_w)
        finally:
            # Every episode this run needs has been dispatched by now.
            self.close()
        result = ExperimentResult(training=training, test=test, tested_w=tested_w)

        output_dir.mkdir(parents=True, exist_ok=True)
        write_results(output_dir / "results.json", config, result)
        write_training_plot(output_dir / "training_plot.png", training.evaluations, reference_card)
        self._log(f"results written to {output_dir}")
        return result

    def train(self) -> TrainingResult:
        config = self.config
        w: W | None = None
        # Legacy warm-up quirk, now opt-in (ticket 12): the first Episode trains
        # with the warm-up rate when one is configured.
        learning_rate = (
            config.warmup_learning_rate
            if config.warmup_learning_rate is not None
            else config.learning_rate
        )
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
                **self.world.episode_kwargs(),
            )
            learning_rate = config.learning_rate
            w = result.w
            w_trajectory.append(_copy_w(result.w))
            episodes_completed = index + 1
            self._log(f"train episode {episodes_completed} (seed {seed}) done")

            if episodes_completed % config.test_frequency == 0:
                newest_w = _copy_w(w)
                # One greedy Episode per evaluation seed: generated fleet, default
                # action pool. Batched so a worker pool can spread them (ticket 08);
                # results come back in ``evaluation_seeds`` order either way.
                episodes = self._run_evaluation_batch(
                    newest_w, tuple(EpisodeRequest(seed) for seed in config.evaluation_seeds)
                )
                block = EvaluationBlock(
                    episodes_completed=episodes_completed,
                    seed_costs=tuple(episode.total_cost for episode in episodes),
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
        """The legacy ``test_model``: fixed seed/fleet tables at widening action pools.

        Ticket 02 (simulation-performance): runs each (action count, seed) pair
        **once**. With per-seed Generators (ticket 13), every one of the legacy
        ``test_episodes`` repeats was bit-identical, so the mean equaled a single
        episode's value — computed directly here rather than via a legacy
        sum/``test_episodes`` division that could round differently in its last
        bit. ``config.test_episodes`` is not read.

        Ticket 08: the whole table — every action count times every seed — is one
        batch, so a worker pool has the full 300-episode job to spread rather than
        one action count at a time. The results are sliced back apart below in
        exactly the nested order they were requested in.
        """
        config = self.config
        fleet = self._test_fleet()
        requests = self.final_test_requests()
        self._log(f"final test: {len(requests)} episodes")
        # One batch can take hours on the full dataset, and it is dispatched in
        # one go, so it reports every tenth of the way rather than going silent
        # until the per-action-count means below.
        step = max(1, len(requests) // 10)
        episodes = self._run_evaluation_batch(
            w,
            requests,
            on_progress=lambda done: (
                self._log(f"final test: {done}/{len(requests)} episodes")
                if done % step == 0
                else None
            ),
        )

        reports = []
        for index, action_count in enumerate(config.test_action_counts):
            row = episodes[index * len(fleet) : (index + 1) * len(fleet)]
            per_seed = tuple(
                SeedTestResult(
                    seed,
                    vehicle_count,
                    {name: float(getattr(episode, name)) for name in EPISODE_METRICS},
                )
                for (seed, vehicle_count), episode in zip(fleet, row, strict=True)
            )
            summary = {
                name: _mean_and_std([entry.metrics[name] for entry in per_seed])
                for name in EPISODE_METRICS
            }
            reports.append(ActionCountReport(action_count, per_seed, summary))
            self._log(
                f"final test actions={action_count}: mean cost {summary['total_cost'][0]:.4f}"
            )
        return tuple(reports)

    def final_test_requests(self) -> tuple[EpisodeRequest, ...]:
        """Every (action count, seed) cell of the final test, in submission order.

        The order is the contract :meth:`final_test` slices the results back apart
        with — action count outermost, the seed/fleet table inside — and the shape
        a benchmark needs to time the phase the way the experiment runs it.
        """
        config = self.config
        return tuple(
            EpisodeRequest(
                seed,
                vehicle_count=vehicle_count,
                number_actions_test=vehicle_count + action_count,
            )
            for action_count in config.test_action_counts
            for seed, vehicle_count in self._test_fleet()
        )

    def _test_fleet(self) -> tuple[tuple[int, int], ...]:
        """The final test's (seed, fleet size) table, paired once for both users."""
        return tuple(zip(self.config.test_seeds, self.config.test_vehicle_counts, strict=True))

    def _run_evaluation_batch(
        self,
        w: W,
        requests: tuple[EpisodeRequest, ...],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[EpisodeResult, ...]:
        """Run one batch of evaluation Episodes: on the worker pool, or right here.

        Both paths run the same ``EpisodeWorld.run_episodes`` and both return the
        results in *request* order, so callers reduce the serial loops' seed order
        whatever order the workers happen to finish in (ticket 08's Tier 1 gate).
        """
        if self.episode_pool is not None:
            return self.episode_pool.run(w, requests, on_progress=on_progress)
        return self.world.run_episodes(w, requests, on_progress=on_progress)


def _copy_w(w: W) -> W:
    return np.array(w, dtype=np.float64, copy=True)


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
    path: Path, evaluations: tuple[EvaluationBlock, ...], reference_card: ReferenceCard | None
) -> None:
    """The legacy training plot: evaluation means with the linear-baseline comparison.

    Rendered through a directly constructed Figure (no pyplot): backend-independent,
    headless-safe, and free of pyplot's global figure registry.

    ``reference_card`` (ticket 01) replaces the legacy's hardcoded
    ``static_policy_mean_cost`` scalar: the red line is now the frozen linear
    ``MonteCarloPolicy``'s mean cost over ``evaluation_seeds`` — the same seeds
    this plot's own evaluation blocks use — rather than an unpaired constant.
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
    if reference_card is not None:
        axes.axhline(
            y=reference_card.evaluation_mean_cost,
            color="red",
            linestyle=":",
            label="Linear baseline (reference card)",
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
