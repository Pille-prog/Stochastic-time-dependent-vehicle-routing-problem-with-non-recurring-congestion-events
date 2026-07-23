"""Episode throughput benchmark and full-run projection (ticket 01).

The measurement foundation every simulation-performance ticket gates on
(``.scratch/simulation-performance/``). Two jobs, one tool:

**Fixture benchmark (default).** Builds the committed mini-fixture world once and
times world load plus N training and N evaluation episodes, printing a compact
table (wall-clock per episode and episodes/sec). Deterministic seeds; no timing
assertions, so CI runs it as smoke through ``tests/test_benchmark_smoke.py``::

    uv run python scripts/benchmark_episodes.py            # default N
    uv run python scripts/benchmark_episodes.py --train 10 --eval 10

The fixture world copies the single committed day into the 44 legacy traffic days
(exactly ``tests/characterization_world.build_legacy_world``): a single day's
speed std is NaN, and 44 identical copies make it 0.0, so travel times stay
finite.

**Scaled real-dataset baseline + projection (``--config`` / ``--project``).**
Runs *scaled* real phases — never the full experiment, which is hours — and
projects the full-run wall-clock from the per-phase per-episode means. The
projection is the effort's success denominator::

    total = load
          + train_iterations * t_train
          + eval_episodes    * t_eval
          + test_seeds * Σ_actions interp(t_test; measured 2..50)

with the middle action counts linearly interpolated between the two measured
endpoints (``test_episodes`` no longer multiplies the test term: ticket 02,
simulation-performance, deduplicated ``Trainer.final_test`` to one episode per
action-count/seed cell, so ``full_run_shape()`` hardcodes that factor to 1).
See the ticket's ``## Comments`` for the recorded numbers::

    uv run python scripts/benchmark_episodes.py \
        --config experiments/chengdu/baseline_scaled.yaml \
        --project experiments/chengdu/config.yaml

The projection arithmetic (``project_full_run`` / ``interpolate_test_time``) is
pure and unit-tested in ``tests/test_benchmark_projection.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.traffic import TravelTimeModel, world_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
# The 44 traffic days the legacy hardcoded (601-630 and 701-714). Copying the one
# committed fixture day across all of them makes the per-day speed std 0.0 rather
# than NaN (see module docstring).
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))


# --- The world every episode runner shares -----------------------------------


@dataclass(frozen=True)
class World:
    """The loaded, immutable-per-run objects an Episode is built from."""

    config: ExperimentConfig
    client_generator: ClientGenerator
    travel_time_model: TravelTimeModel
    shortest_path_cache: ShortestPathCache
    congestion_generator: ArcProbabilityCongestionGenerator

    def episode_kwargs(self) -> dict[str, Any]:
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


def build_fixture_data_dir(destination: Path) -> Path:
    """Copy link.csv, the 44 day-copies and the path cache into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_DIR / "link.csv", destination / "link.csv")
    shutil.copyfile(FIXTURE_DIR / "all_shortest_paths.csv", destination / "all_shortest_paths.csv")
    for day in LEGACY_DAYS:
        for half in (0, 1):
            shutil.copyfile(
                FIXTURE_DIR / f"speed[601]_[{half}].csv",
                destination / f"speed[{day}]_[{half}].csv",
            )
    return destination


def fixture_config(data_dir: Path) -> ExperimentConfig:
    """The committed fixture config repointed at a 44-day ``data_dir``."""
    return dataclasses.replace(
        ExperimentConfig.from_yaml(FIXTURE_DIR / "config.yaml"),
        data_dir=data_dir,
        traffic_days=LEGACY_DAYS,
    )


def load_world(config: ExperimentConfig, *, cache_dir: Path | None = None) -> tuple[World, float]:
    """Build the world from ``config``; return it with the world-load wall-clock.

    ``cache_dir`` (ticket 03) opts into the binary world cache: ``None`` parses
    the CSVs fresh, exactly as before; a directory reuses a matching snapshot.
    """
    started = time.perf_counter()
    loaded = world_cache.load_world(config, cache_dir=cache_dir)
    travel_time_model = loaded.travel_time_model
    shortest_path_cache = loaded.shortest_path_cache
    congestion_generator = ArcProbabilityCongestionGenerator(
        event_probability=travel_time_model.event_probability,
        successors=travel_time_model.successors,
        congestion_lower_bound=config.congestion_lower_bound,
        congestion_upper_bound=config.congestion_upper_bound,
        max_congestion_duration=config.max_congestion_duration,
    )
    world = World(
        config=config,
        client_generator=ClientGenerator.from_config(config),
        travel_time_model=travel_time_model,
        shortest_path_cache=shortest_path_cache,
        congestion_generator=congestion_generator,
    )
    return world, time.perf_counter() - started


# --- Timing the three phases --------------------------------------------------


@dataclass(frozen=True)
class PhaseTiming:
    """Per-episode wall-clock for one phase."""

    episodes: int
    total_seconds: float

    @property
    def per_episode(self) -> float:
        return self.total_seconds / self.episodes if self.episodes else 0.0

    @property
    def episodes_per_second(self) -> float:
        return self.episodes / self.total_seconds if self.total_seconds else 0.0


def time_training(world: World, seeds: list[int]) -> tuple[PhaseTiming, np.ndarray | None]:
    """Sequential training episodes carrying W, warm-up on the first (legacy quirk).

    Returns the phase timing and the final trained W (so callers can evaluate with
    a real policy without re-running training). W is ``None`` only for empty seeds.
    """
    config = world.config
    learning_rate = (
        config.warmup_learning_rate
        if config.warmup_learning_rate is not None
        else config.learning_rate
    )
    w: np.ndarray | None = None
    started = time.perf_counter()
    for seed in seeds:
        result = run_training_episode(
            seed=seed,
            W=w,
            learning_rate=learning_rate,
            depot=0,
            **world.episode_kwargs(),
        )
        learning_rate = config.learning_rate
        w = result.w
    return PhaseTiming(len(seeds), time.perf_counter() - started), w


def time_evaluation(world: World, seeds: list[int], w: np.ndarray | None) -> PhaseTiming:
    """Greedy evaluation episodes with a fixed W, generated fleet and action pool."""
    started = time.perf_counter()
    for seed in seeds:
        run_evaluation_episode(seed=seed, W=w, **world.episode_kwargs())
    return PhaseTiming(len(seeds), time.perf_counter() - started)


def time_test_at_action_count(world: World, action_count: int, w: np.ndarray | None) -> PhaseTiming:
    """Final-test episodes at one action-pool width over the config's seed/fleet table."""
    config = world.config
    started = time.perf_counter()
    for seed, vehicles in zip(config.test_seeds, config.test_vehicle_counts, strict=True):
        run_evaluation_episode(
            seed=seed,
            W=w,
            vehicle_count=vehicles,
            number_actions_test=vehicles + action_count,
            **world.episode_kwargs(),
        )
    return PhaseTiming(len(config.test_seeds), time.perf_counter() - started)


# --- Full-run projection (pure; unit-tested) ----------------------------------


@dataclass(frozen=True)
class PhaseTimes:
    """The scaled measurements the full-run projection is built from."""

    world_load_seconds: float
    train_seconds: float
    eval_seconds: float
    # measured mean per test episode, keyed by action count (must include the two
    # endpoints the middle action counts are interpolated between).
    test_seconds: dict[int, float]


@dataclass(frozen=True)
class FullRunShape:
    """The full experiment's episode counts (read off the full config)."""

    train_iterations: int
    eval_episodes: int
    test_action_counts: tuple[int, ...]
    test_seed_count: int
    test_episodes_per_cell: int


def interpolate_test_time(action_count: int, measured: dict[int, float]) -> float:
    """Per-test-episode time at ``action_count``, linear between the measured ends."""
    if action_count in measured:
        return measured[action_count]
    low = min(measured)
    high = max(measured)
    if high == low:  # a single measured action count: nothing to interpolate against
        return measured[low]
    fraction = (action_count - low) / (high - low)
    return measured[low] + (measured[high] - measured[low]) * fraction


def project_full_run(times: PhaseTimes, shape: FullRunShape) -> dict[str, float]:
    """Project the full-run wall-clock from scaled per-phase per-episode means."""
    train_seconds = shape.train_iterations * times.train_seconds
    eval_seconds = shape.eval_episodes * times.eval_seconds
    test_seconds = (
        shape.test_seed_count
        * shape.test_episodes_per_cell
        * sum(interpolate_test_time(a, times.test_seconds) for a in shape.test_action_counts)
    )
    total = times.world_load_seconds + train_seconds + eval_seconds + test_seconds
    return {
        "world_load_seconds": times.world_load_seconds,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "test_seconds": test_seconds,
        "total_seconds": total,
    }


def full_run_shape(config: ExperimentConfig) -> FullRunShape:
    """The full experiment's episode counts, derived from its config.

    Ticket 02 (simulation-performance) deduplicated ``Trainer.final_test``: it
    now runs each (action count, seed) pair once regardless of
    ``config.test_episodes`` (inert field, kept for YAML compatibility), so the
    projection hardcodes 1 rather than reading it.
    """
    evaluation_blocks = config.total_train_iterations // config.test_frequency
    return FullRunShape(
        train_iterations=config.total_train_iterations,
        eval_episodes=evaluation_blocks * config.evaluation_seed_count,
        test_action_counts=config.test_action_counts,
        test_seed_count=len(config.test_seeds),
        test_episodes_per_cell=1,
    )


# --- CLI ----------------------------------------------------------------------


def _print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label.ljust(width)}  {value}")


def run_fixture_benchmark(n_train: int, n_eval: int, cache_dir: Path | None = None) -> None:
    """The committed mini-fixture benchmark: world load + N train + N eval."""
    with tempfile.TemporaryDirectory(prefix="benchmark_world_") as tmp:
        data_dir = build_fixture_data_dir(Path(tmp))
        config = fixture_config(data_dir)
        print(f"building the mini-fixture world ({len(LEGACY_DAYS)} day-copies) ...", flush=True)
        world, load_seconds = load_world(config, cache_dir=cache_dir)

        train_seeds = [config.first_train_seed + i for i in range(n_train)]
        eval_seeds = list(config.evaluation_seeds[:n_eval])
        training, _ = time_training(world, train_seeds)
        # Evaluate greedily from the zero vector (W=None): a fixed, deterministic W
        # that needs no prior training run, so the eval timing stands alone.
        evaluation = time_evaluation(world, eval_seeds, w=None)

    print("\nmini-fixture episode benchmark")
    _print_table(
        [
            ("world load (s)", f"{load_seconds:8.3f}"),
            (
                f"training  ({training.episodes} ep)",
                f"{training.per_episode:8.3f} s/ep   {training.episodes_per_second:6.2f} ep/s",
            ),
            (
                f"evaluation ({evaluation.episodes} ep)",
                f"{evaluation.per_episode:8.3f} s/ep   {evaluation.episodes_per_second:6.2f} ep/s",
            ),
        ]
    )


def run_config_baseline(
    config_path: Path,
    full_config_path: Path | None,
    with_test: bool,
    cache_dir: Path | None = None,
) -> None:
    """The scaled real-dataset baseline: time each phase, optionally project the full run."""
    config = ExperimentConfig.from_yaml(config_path)
    print(f"config: {config_path}")
    print(
        "loading world (the full Chengdu archive takes minutes, or seconds warm-cached) ...",
        flush=True,
    )
    world, load_seconds = load_world(config, cache_dir=cache_dir)
    print(f"world loaded in {load_seconds:.1f}s", flush=True)

    train_seeds = [config.first_train_seed + i for i in range(config.total_train_iterations)]
    eval_seeds = list(config.evaluation_seeds)
    training, trained_w = time_training(world, train_seeds)
    print(f"training: {training.per_episode:.3f} s/ep over {training.episodes} ep", flush=True)
    # Evaluate with the freshly trained W so the estimate reflects a real policy.
    evaluation = time_evaluation(world, eval_seeds, w=trained_w)
    print(
        f"evaluation: {evaluation.per_episode:.3f} s/ep over {evaluation.episodes} ep", flush=True
    )

    test_means: dict[int, float] = {}
    if with_test:
        for action_count in config.test_action_counts:
            timing = time_test_at_action_count(world, action_count, trained_w)
            test_means[action_count] = timing.per_episode
            print(
                f"test actions={action_count}: {timing.per_episode:.3f} s/ep "
                f"over {timing.episodes} ep",
                flush=True,
            )

    print("\nscaled real-dataset baseline")
    rows = [
        ("world load (s)", f"{load_seconds:10.3f}"),
        ("train  s/ep", f"{training.per_episode:10.3f}"),
        ("eval   s/ep", f"{evaluation.per_episode:10.3f}"),
    ]
    for action_count, mean in test_means.items():
        rows.append((f"test s/ep @{action_count}", f"{mean:10.3f}"))
    _print_table(rows)

    if full_config_path is not None and test_means:
        full = ExperimentConfig.from_yaml(full_config_path)
        shape = full_run_shape(full)
        projection = project_full_run(
            PhaseTimes(
                world_load_seconds=load_seconds,
                train_seconds=training.per_episode,
                eval_seconds=evaluation.per_episode,
                test_seconds=test_means,
            ),
            shape,
        )
        print(f"\nprojected full run ({full_config_path.name})")
        _print_table(
            [
                ("world load", _hms(projection["world_load_seconds"])),
                (f"training ({shape.train_iterations} ep)", _hms(projection["train_seconds"])),
                (f"evaluation ({shape.eval_episodes} ep)", _hms(projection["eval_seconds"])),
                (
                    f"final test ({_test_episode_count(shape)} ep)",
                    _hms(projection["test_seconds"]),
                ),
                ("TOTAL", _hms(projection["total_seconds"])),
            ]
        )


def _test_episode_count(shape: FullRunShape) -> int:
    return len(shape.test_action_counts) * shape.test_seed_count * shape.test_episodes_per_cell


def _hms(seconds: float) -> str:
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{seconds:12.1f}s  ({hours:d}h{minutes:02d}m{secs:02d}s)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="real-dataset config to baseline (default: the committed mini fixture)",
    )
    parser.add_argument(
        "--train", type=int, default=5, help="fixture training episodes (default 5)"
    )
    parser.add_argument(
        "--eval", type=int, default=5, help="fixture evaluation episodes (default 5)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="also time the final-test phase (real-config baseline only)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="full config to project the full-run wall-clock from the scaled measurements",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=world_cache.default_cache_dir(),
        help=(
            "binary world cache directory, ticket 03 (default: %(default)s, "
            "or STDVRP_WORLD_CACHE_DIR)"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="always rebuild the world from the CSVs and do not write the cache",
    )
    args = parser.parse_args(argv)
    cache_dir = None if args.no_cache else args.cache_dir

    if args.config is None:
        run_fixture_benchmark(args.train, args.eval, cache_dir=cache_dir)
    else:
        run_config_baseline(
            args.config,
            args.project,
            with_test=args.test or args.project is not None,
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    main()
