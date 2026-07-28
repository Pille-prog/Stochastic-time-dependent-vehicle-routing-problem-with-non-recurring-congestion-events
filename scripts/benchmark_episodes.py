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

**Worker sweep (``--workers``, ticket 08).** Times the two phases that ticket 08
parallelizes — an evaluation block and the final test — at each worker count, on
one world loaded once, and projects the full run for each. The per-episode means
it feeds the projection with are already divided by whatever parallelism the run
achieved, so the same arithmetic answers "how long at N workers?"::

    uv run python scripts/benchmark_episodes.py \
        --config experiments/chengdu/parallel_scaled.yaml \
        --project experiments/chengdu/config.yaml --workers 1,2,3

Every count above 1 needs the world cache: each worker loads its own world
through it (and holds it — see ticket 08's Comments for the memory ceiling).
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stdvrp.config import ExperimentConfig
from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.traffic import world_cache
from stdvrp.training import EpisodePool, EpisodeRequest, EpisodeWorld, Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
# The 44 traffic days the legacy hardcoded (601-630 and 701-714). Copying the one
# committed fixture day across all of them makes the per-day speed std 0.0 rather
# than NaN (see module docstring).
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))


# --- The world every episode runner shares -----------------------------------
#
# ``EpisodeWorld`` (stdvrp.training.episode_pool) is exactly this: the loaded
# world plus the config scalars, and one ``episode_kwargs()`` both Episode
# runners take. It is what the Trainer and its worker processes run on, so the
# benchmark times the same object the experiment uses rather than a lookalike.


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


def load_world(
    config: ExperimentConfig, *, cache_dir: Path | None = None
) -> tuple[EpisodeWorld, float]:
    """Build the world from ``config``; return it with the world-load wall-clock.

    ``cache_dir`` (ticket 03) opts into the binary world cache: ``None`` parses
    the CSVs fresh, exactly as before; a directory reuses a matching snapshot.
    """
    started = time.perf_counter()
    world = EpisodeWorld.load(config, cache_dir=cache_dir)
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


def time_training(world: EpisodeWorld, seeds: list[int]) -> tuple[PhaseTiming, np.ndarray | None]:
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


def time_evaluation(world: EpisodeWorld, seeds: list[int], w: np.ndarray | None) -> PhaseTiming:
    """Greedy evaluation episodes with a fixed W, generated fleet and action pool."""
    started = time.perf_counter()
    for seed in seeds:
        run_evaluation_episode(seed=seed, W=w, **world.episode_kwargs())
    return PhaseTiming(len(seeds), time.perf_counter() - started)


def time_test_at_action_count(
    world: EpisodeWorld, action_count: int, w: np.ndarray | None
) -> PhaseTiming:
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


# --- Worker sweep (ticket 08) --------------------------------------------------


def time_worker_count(
    world: EpisodeWorld, cache_dir: Path | None, worker_count: int, w: np.ndarray | None
) -> tuple[float, PhaseTiming, dict[int, PhaseTiming]]:
    """Time the two parallelizable phases at one worker count, on one loaded world.

    Returns the pool's startup (one Episode per worker, so every worker has loaded
    its world), the evaluation block, and the final test *per action count* — the
    per-episode means the projection needs, already divided by whatever
    parallelism the run achieved. The final test goes through ``Trainer.final_test``
    itself, one action count at a time, so the phase being timed is the phase the
    experiment runs rather than a lookalike.
    """
    config = world.config
    pool = EpisodePool.for_worker_count(config, cache_dir=cache_dir, worker_count=worker_count)
    requests = tuple(EpisodeRequest(seed) for seed in config.evaluation_seeds)
    try:
        prime_seconds = 0.0
        if pool is not None:
            started = time.perf_counter()
            pool.run(w, tuple(requests[0] for _ in range(worker_count)))
            prime_seconds = time.perf_counter() - started

        started = time.perf_counter()
        # Exactly what Trainer._run_evaluation_batch dispatches an evaluation
        # block to, on each side of its one branch.
        if pool is None:
            world.run_episodes(w, requests)
        else:
            pool.run(w, requests)
        evaluation = PhaseTiming(len(requests), time.perf_counter() - started)

        test: dict[int, PhaseTiming] = {}
        for action_count in config.test_action_counts:
            scoped = dataclasses.replace(
                world, config=dataclasses.replace(config, test_action_counts=(action_count,))
            )
            started = time.perf_counter()
            Trainer(scoped, episode_pool=pool).final_test(w)
            test[action_count] = PhaseTiming(len(config.test_seeds), time.perf_counter() - started)
    finally:
        if pool is not None:
            pool.close()
    return prime_seconds, evaluation, test


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


def run_worker_sweep(
    config_path: Path,
    full_config_path: Path | None,
    worker_counts: list[int],
    cache_dir: Path | None,
) -> None:
    """Ticket 08: the parallelizable phases at each worker count, and what they project to.

    One world is loaded and one W trained here; every worker count then replays the
    same two phases on it, so the counts differ only in how many processes ran them.
    """
    config = ExperimentConfig.from_yaml(config_path)
    print(f"config: {config_path}")
    print("loading world (seconds warm-cached, minutes cold) ...", flush=True)
    world, load_seconds = load_world(config, cache_dir=cache_dir)
    print(f"world loaded in {load_seconds:.1f}s", flush=True)

    train_seeds = [config.first_train_seed + i for i in range(config.total_train_iterations)]
    training, trained_w = time_training(world, train_seeds)
    print(f"training (stays sequential): {training.per_episode:.3f} s/ep", flush=True)

    rows = []
    for worker_count in worker_counts:
        print(f"\nworkers = {worker_count}", flush=True)
        prime, evaluation, test = time_worker_count(world, cache_dir, worker_count, trained_w)
        print(f"  pool start        {prime:8.1f}s", flush=True)
        print(
            f"  evaluation        {evaluation.total_seconds:8.1f}s "
            f"({evaluation.per_episode:.3f} s/ep over {evaluation.episodes})",
            flush=True,
        )
        for action_count, timing in test.items():
            print(
                f"  final test @{action_count:<3}   {timing.total_seconds:8.1f}s "
                f"({timing.per_episode:.3f} s/ep over {timing.episodes})",
                flush=True,
            )
        rows.append((worker_count, prime, evaluation, test))

    print("\nworker sweep")
    baseline = rows[0][2].total_seconds + sum(t.total_seconds for t in rows[0][3].values())
    for worker_count, prime, evaluation, test in rows:
        phases = evaluation.total_seconds + sum(t.total_seconds for t in test.values())
        _print_table(
            [
                (
                    f"{worker_count} worker(s)",
                    f"phases {phases:8.1f}s   speedup {baseline / phases:5.2f}x"
                    f"   pool start {prime:6.1f}s",
                )
            ]
        )
        if full_config_path is not None:
            shape = full_run_shape(ExperimentConfig.from_yaml(full_config_path))
            projection = project_full_run(
                PhaseTimes(
                    world_load_seconds=load_seconds,
                    train_seconds=training.per_episode,
                    eval_seconds=evaluation.per_episode,
                    test_seconds={a: timing.per_episode for a, timing in test.items()},
                ),
                shape,
            )
            _print_table([("  projected full run", _hms(projection["total_seconds"] + prime))])


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
    parser.add_argument(
        "--workers",
        default=None,
        help=(
            "comma-separated worker counts to sweep the parallelizable phases over, "
            "ticket 08 (e.g. 1,2,3); needs --config, and every count above 1 needs the "
            "world cache because each worker loads its own world through it"
        ),
    )
    args = parser.parse_args(argv)
    cache_dir = None if args.no_cache else args.cache_dir

    if args.workers is not None:
        if args.config is None:
            parser.error("--workers sweeps a real-dataset config: pass --config too")
        run_worker_sweep(
            args.config,
            args.project,
            [int(count) for count in args.workers.split(",")],
            cache_dir=cache_dir,
        )
    elif args.config is None:
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
