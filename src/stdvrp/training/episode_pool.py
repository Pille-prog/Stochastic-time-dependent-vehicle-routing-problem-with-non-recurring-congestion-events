"""Batches of evaluation Episodes, in-process or on a persistent worker pool (ticket 08).

The Trainer's two embarrassingly parallel phases — the periodic evaluation
blocks and the final test — are expressed here as a *batch of requests* rather
than a loop: :class:`EpisodeRequest` says which Episode to run,
:class:`EpisodeWorld` knows how to run one, and :class:`EpisodePool` runs a whole
batch on worker processes. Training stays sequential in the Trainer: W is a
serial dependency, and parallel training would be new science, not optimization.

**Seed-ordered aggregation.** ``EpisodePool.run`` returns results in *request*
order, never completion order (``Executor.map`` yields in submission order), so
every mean, std and ``results.json`` entry the Trainer reduces is bit-identical
to the serial loops'. That is the ticket's Tier 1 gate, pinned by
``tests/test_parallel_evaluation.py``.

**Workers.** The pool always uses the ``spawn`` start method — required on
Windows, and chosen explicitly everywhere so nothing can quietly come to depend
on ``fork`` inheriting parent state. A spawned worker therefore starts empty and
loads the world itself, exactly once, in the pool's initializer: that load reads
the ticket-03 binary world cache, which is why ``cache_dir`` is mandatory here
(without it 16 workers would each re-parse 88 speed CSVs and the 907 MB path
file — minutes apiece). Workers are persistent for the pool's lifetime, so the
cost is paid once per worker, not once per episode.

**Memory is the ceiling, not cores.** Each worker holds its own copy of the
world, and the full Chengdu world measures **8.0 GB resident** (ticket 08
benchmark), so peak memory is about ``worker_count`` times that: 16 workers
would ask for ~136 GB. Worker count is therefore a *memory* budget, which is
why nothing here defaults to one worker per core — see
:func:`default_worker_count`. CPython also spawns a worker only when a task
arrives and no idle worker is free, so a batch smaller than ``worker_count``
never costs more worlds than it has Episodes.

**Serial fallback.** ``EpisodePool.for_worker_count(..., worker_count=1)``
returns ``None`` and the Trainer keeps every episode in this process, running the
very same :meth:`EpisodeWorld.run_episodes` the workers run. That is the default,
the debugging path and the CI path; :func:`default_worker_count` exposes the
``STDVRP_WORKERS`` environment switch that opts out of it.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator, CongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import EpisodeResult, run_evaluation_episode
from stdvrp.traffic import TravelTimeModel, world_cache

W = NDArray[np.float64]

WORKER_COUNT_ENV = "STDVRP_WORKERS"


def default_worker_count() -> int:
    """Workers to evaluate with: ``STDVRP_WORKERS`` if set, else ``1`` (serial).

    Parallel evaluation is deliberately **opt-in** rather than "one worker per
    core": a worker is a whole world, and the full Chengdu world measures 8.0 GB
    resident (ticket 08 benchmark), so the ceiling is the machine's memory, not
    its cores — 16 cores would ask for ~136 GB. Only the operator knows how much
    their machine has, so they say: ``STDVRP_WORKERS`` once per machine, or
    ``--workers`` per run. ``1`` keeps every Episode in this process, which is
    also what debugging and CI want.
    """
    raw = os.environ.get(WORKER_COUNT_ENV)
    if raw is None or not raw.strip():
        return 1
    try:
        count = int(raw)
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError(f"{WORKER_COUNT_ENV} must be a positive integer, got {raw!r}")
    return count


def _validated_worker_count(worker_count: int) -> int:
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    return worker_count


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    """One evaluation Episode to run: its seed plus the two per-episode overrides.

    ``None`` overrides mean "as the Episode runner defaults": the generated fleet
    size and a ``vehicles + 2`` action pool, which is what the evaluation blocks
    use. The final test sets both from its seed/fleet tables.
    """

    seed: int
    vehicle_count: int | None = None
    number_actions_test: int | None = None


@dataclass(frozen=True, slots=True)
class EpisodeWorld:
    """The loaded world an Episode runs in, with the config scalars it needs.

    A superset of ``world_cache.LoadedWorld``: that value object carries what the
    binary cache stores (travel times and shortest paths), this one adds the two
    generators built around it and the config, which together are exactly the
    keyword arguments both Episode runners take.
    """

    config: ExperimentConfig
    client_generator: ClientGenerator
    travel_time_model: TravelTimeModel
    shortest_path_cache: ShortestPathCache
    congestion_generator: CongestionGenerator

    @classmethod
    def load(cls, config: ExperimentConfig, *, cache_dir: Path | None = None) -> EpisodeWorld:
        """Load the world from ``config``'s DataSource and wire its generators.

        ``cache_dir`` (ticket 03) opts into the binary world cache; ``None``
        parses the CSVs fresh.
        """
        loaded = world_cache.load_world(config, cache_dir=cache_dir)
        travel_time_model = loaded.travel_time_model
        return cls(
            config=config,
            client_generator=ClientGenerator.from_config(config),
            travel_time_model=travel_time_model,
            shortest_path_cache=loaded.shortest_path_cache,
            congestion_generator=ArcProbabilityCongestionGenerator(
                event_probability=travel_time_model.event_probability,
                successors=travel_time_model.successors,
                congestion_lower_bound=config.congestion_lower_bound,
                congestion_upper_bound=config.congestion_upper_bound,
                max_congestion_duration=config.max_congestion_duration,
            ),
        )

    def episode_kwargs(self) -> dict[str, Any]:
        """The world and config arguments every Episode runner shares."""
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
            "n_observed_velocities": config.n_observed_velocities,
        }

    def run_episode(self, w: W | None, request: EpisodeRequest) -> EpisodeResult:
        """Run one requested Episode here — what a worker does with each task."""
        return run_evaluation_episode(
            seed=request.seed,
            W=w,
            vehicle_count=request.vehicle_count,
            number_actions_test=request.number_actions_test,
            **self.episode_kwargs(),
        )

    def run_episodes(
        self,
        w: W | None,
        requests: Sequence[EpisodeRequest],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[EpisodeResult, ...]:
        """Run ``requests`` here, in order — the serial path the Trainer takes.

        ``on_progress`` is called with the running completed count, so a caller
        logs the same way whichever path a batch takes.
        """
        results = []
        for request in requests:
            results.append(self.run_episode(w, request))
            if on_progress is not None:
                on_progress(len(results))
        return tuple(results)


class EpisodePool:
    """A persistent pool of spawned workers that each hold one loaded world."""

    def __init__(self, config: ExperimentConfig, *, cache_dir: Path, worker_count: int) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.worker_count = _validated_worker_count(worker_count)
        self._executor: ProcessPoolExecutor | None = None

    @classmethod
    def for_worker_count(
        cls,
        config: ExperimentConfig,
        *,
        cache_dir: Path | None,
        worker_count: int,
    ) -> EpisodePool | None:
        """The pool for ``worker_count`` workers, or ``None`` for the serial fallback."""
        if _validated_worker_count(worker_count) == 1:
            return None
        if cache_dir is None:
            raise ValueError(
                f"worker_count {worker_count} needs a cache_dir: every spawned worker "
                "loads the world from the binary world cache, and without one they "
                "would each re-parse the CSVs. Pass cache_dir=... or worker_count=1."
            )
        return cls(config, cache_dir=cache_dir, worker_count=worker_count)

    @property
    def is_started(self) -> bool:
        """Whether the worker processes have been started (they start on first use)."""
        return self._executor is not None

    def run(
        self,
        w: W | None,
        requests: Sequence[EpisodeRequest],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[EpisodeResult, ...]:
        """Run ``requests`` on the workers and return their results in request order.

        ``on_progress`` is called with the running completed count — in request
        order, not completion order, so a batch that takes an hour still reports
        as it goes without any claim about which worker finished what.
        """
        tasks = [(w, request) for request in requests]
        if not tasks:
            return ()
        # Executor.map yields in submission order however the workers interleave,
        # so callers reduce the serial loops' seed order (the Tier 1 gate).
        results = []
        for result in self._started_executor().map(_run_episode, tasks):
            results.append(result)
            if on_progress is not None:
                on_progress(len(results))
        return tuple(results)

    def close(self) -> None:
        """Shut the workers down; a later ``run`` starts a fresh pool."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _started_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_load_worker_world,
                initargs=(self.config, self.cache_dir),
            )
        return self._executor

    def __enter__(self) -> EpisodePool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


# --- Worker process -----------------------------------------------------------
#
# A spawned worker imports this module fresh and runs the initializer once; the
# world it loads then lives in this module global for every task that worker
# takes. Nothing else is inherited from the parent — that is the point of spawn.

_WORKER_WORLD: EpisodeWorld | None = None


def _load_worker_world(config: ExperimentConfig, cache_dir: Path) -> None:
    """Pool initializer: load this worker's one world from the binary cache."""
    global _WORKER_WORLD
    _WORKER_WORLD = EpisodeWorld.load(config, cache_dir=cache_dir)


def _run_episode(task: tuple[W | None, EpisodeRequest]) -> EpisodeResult:
    """Run one requested Episode in this worker's world."""
    w, request = task
    if _WORKER_WORLD is None:  # pragma: no cover - the initializer always runs first
        raise RuntimeError("episode pool worker was asked for an episode before its world")
    return _WORKER_WORLD.run_episode(w, request)
