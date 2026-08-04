"""Capture this repo's W after every training episode, in the legacy capture's schema.

The other half of ``.scratch/linear-policy-learning/issues/01-legacy-w-trajectory-diff.md``:
:mod:`scripts.capture_legacy_w_trajectory` produces the reference monolith's
trajectory, this one produces ours, and :mod:`scripts.compare_w_trajectories`
diffs the 24 components.

Deliberately *not* built on ``scripts/legacy_fidelity_bench.py``: that bench
exists to price candidate legacy reverts, and every one of its toggles
monkeypatches ``src``. This driver runs the shipped code with nothing patched —
the clip included, since ``UPDATE_FEATURE_CEILING`` is in ``MonteCarloPolicy``
now — so the trajectory it writes is the one the repo actually produces.

No evaluation blocks: the ticket asks about W, not cost, and the evaluation
numbers at these parameters are already in ``spec.md``. The per-episode training
cost is recorded anyway, since it is free.

**Same seed integers, different worlds.** Training seeds start at 1000 on both
sides, but ticket 13 (ADR-0001 phase 2) replaced the legacy's two shared global
Mersenne streams with four independent per-Episode PCG64 streams. Seed 1000 draws
a different client set, congestion and velocity field here than it does there.
Compare the *shape* of each component's trajectory, never its values.

Usage::

    .venv/Scripts/python.exe -u scripts/capture_repo_w_trajectory.py \\
        experiments/chengdu/config_linear_congestion_10k.yaml --episodes 200
"""

from __future__ import annotations

import argparse
import dataclasses
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ``stdvrp.simulation`` must resolve before ``stdvrp.policies``: importing
# policies first goes monte_carlo -> action_set -> simulation.state, which pulls
# in simulation/__init__, which imports episode, which imports monte_carlo while
# it is still half-initialised (ImportError). A plain ``import`` sorts ahead of
# every ``from`` import in this block, so this line both breaks the cycle and
# keeps ruff's isort happy. The cycle is pre-existing in the package.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.policies.feature_extraction import FEATURE_COUNT
from stdvrp.policies.monte_carlo import UPDATE_FEATURE_CEILING
from stdvrp.simulation import run_training_episode
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_w_trajectories import TrajectoryLog  # noqa: E402


def log(message: str) -> None:
    print(f"[capture] {message}", file=sys.stderr, flush=True)


def capture_trajectory(
    config: ExperimentConfig, episodes: int, checkpoint: Path | None = None
) -> TrajectoryLog:
    """Run ``episodes`` training episodes, returning W and the cost after each."""
    world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
    episode_kwargs = world.episode_kwargs()

    w = None
    learning_rate = (
        config.warmup_learning_rate
        if config.warmup_learning_rate is not None
        else config.learning_rate
    )
    trajectory = TrajectoryLog()

    for index in range(episodes):
        seed = config.first_train_seed + index
        t_ep = time.perf_counter()
        demand = world.client_generator.generate(seed)
        result = run_training_episode(seed=seed, W=w, learning_rate=learning_rate, **episode_kwargs)
        learning_rate = config.learning_rate
        w = result.w
        trajectory.record(w, result.episode.total_cost, demand.vehicle_count, len(demand.clients))
        log(
            f"episode {index + 1:4d} (seed {seed}): cost={result.episode.total_cost:.2f} "
            f"|W|={float(np.linalg.norm(w)):.2f} ({time.perf_counter() - t_ep:.1f}s)"
        )
        if checkpoint is not None:
            trajectory.checkpoint(checkpoint)

    return trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="override the config's data_dir (it is machine-specific)",
    )
    parser.add_argument(
        "--time-window-spread",
        type=int,
        default=None,
        help=(
            "override the config's time_window_spread (the legacy's diff_TW), so a "
            "comparison at another width never has to edit a committed config. "
            "The configs are 150; pass 60 to reproduce the superseded measurements."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / ".scratch" / "linear-policy-learning" / "repo_w_trajectory.json",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.data_dir is not None:
        config = dataclasses.replace(config, data_dir=args.data_dir)
    if args.time_window_spread is not None:
        config = dataclasses.replace(config, time_window_spread=args.time_window_spread)

    t0 = time.perf_counter()
    trajectory = capture_trajectory(
        config, args.episodes, checkpoint=args.out.with_suffix(".partial.json")
    )
    trajectory.write(
        args.out,
        meta={
            "side": "repo",
            "config": str(args.config),
            "feature_count": FEATURE_COUNT,
            "update_feature_ceiling": UPDATE_FEATURE_CEILING,
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
        },
        protocol={
            "learning_rate": config.learning_rate,
            "warmup_learning_rate": config.warmup_learning_rate,
            "epsilon": config.epsilon,
            "congestion_lower_bound": config.congestion_lower_bound,
            "congestion_upper_bound": config.congestion_upper_bound,
            "max_congestion_duration": config.max_congestion_duration,
            "mean_number_clients": config.mean_number_clients,
            "diff_TW": config.time_window_spread,
            "first_train_seed": config.first_train_seed,
        },
    )
    log(f"W trajectory written to {args.out} ({time.perf_counter() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
