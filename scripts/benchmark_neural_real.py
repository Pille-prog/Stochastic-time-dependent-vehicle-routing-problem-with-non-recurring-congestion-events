"""Real-network per-episode cost measurement (ticket 12, neural-policy).

Replaces spec.md's stub-based compute budget table (ticket 03,
``scripts/benchmark_neural_stub.py``): that table timed a stub encoder in
isolation, never the tokenizer, never a real Episode. This script times the
real thing -- ``run_neural_training_episode`` / ``run_neural_evaluation_episode``
(tickets 04-07) against the real committed architecture (spec.md's "Starting
architecture": d=128, 3 layers, 4 heads) and the real Chengdu dataset -- the
combined network + simulator cost a training run actually pays, on both
devices.

Usage (full dataset, both devices when CUDA is available; needs a warm world
cache -- ``uv run python scripts/run_gate_a.py --max-episodes 1`` primes one)::

    uv run python scripts/benchmark_neural_real.py

Mini fixture, fast, for local iteration::

    uv run python scripts/benchmark_neural_real.py \\
        --config tests/fixtures/chengdu_mini/config.yaml --episodes 3
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

# Same circular-import landmine every neural script in this repo guards
# against (see benchmark_neural_stub.py's own comment): stdvrp.simulation must
# finish initializing before anything reaches stdvrp.policies.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.policies.torch_support import TORCH_INSTALL_HINT
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld

try:
    import torch
except ImportError:
    print(f"scripts/benchmark_neural_real.py needs torch: {TORCH_INSTALL_HINT}", file=sys.stderr)
    raise SystemExit(1) from None

from stdvrp.training.neural_episode import (
    build_neural_policy_state,
    run_neural_evaluation_episode,
    run_neural_training_episode,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclasses.dataclass(frozen=True)
class Sample:
    seconds: float
    decisions: int  # EpisodeResult.state_count -- decision epochs this episode ran


def _time_training(run_one: object, seeds: range, device: torch.device) -> list[Sample]:
    samples = []
    for seed in seeds:
        started = time.perf_counter()
        result, _loss = run_one(seed)  # type: ignore[operator]
        _sync(device)
        samples.append(Sample(time.perf_counter() - started, result.state_count))
    return samples


def _time_evaluation(run_one: object, seeds: range, device: torch.device) -> list[Sample]:
    samples = []
    for seed in seeds:
        started = time.perf_counter()
        result = run_one(seed)  # type: ignore[operator]
        _sync(device)
        samples.append(Sample(time.perf_counter() - started, result.state_count))
    return samples


def _report(label: str, samples: list[Sample]) -> None:
    seconds = [s.seconds for s in samples]
    mean = sum(seconds) / len(seconds)
    ms_per_decision = [1000 * s.seconds / s.decisions for s in samples if s.decisions]
    print(f"  {label}: mean {mean:6.3f} s/ep  (min {min(seconds):.2f}, max {max(seconds):.2f})")
    if ms_per_decision:
        print(
            f"    ms/decision: mean {sum(ms_per_decision) / len(ms_per_decision):.2f}  "
            f"(min {min(ms_per_decision):.2f}, max {max(ms_per_decision):.2f})"
        )
    print(f"    per-episode (s, decisions): {[(f'{s.seconds:.2f}', s.decisions) for s in samples]}")


def measure(
    device_name: str,
    config: ExperimentConfig,
    world: EpisodeWorld,
    *,
    episodes: int,
    first_train_seed: int,
    eval_seed_start: int,
) -> None:
    device_config = dataclasses.replace(config, device=device_name)
    policy_state = build_neural_policy_state(device_config, np.random.default_rng(0))
    device = policy_state.device

    def run_training(seed: int) -> object:
        return run_neural_training_episode(
            seed=seed,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=policy_state,
            config=device_config,
        )

    def run_evaluation(seed: int) -> object:
        return run_neural_evaluation_episode(
            seed=seed,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=policy_state,
            config=device_config,
        )

    # One untimed warmup episode: first-call overhead (CUDA context init,
    # attention-backend algorithm selection under use_deterministic_algorithms)
    # would otherwise land inside episode 1's timed measurement.
    run_training(first_train_seed - 1)
    _sync(device)

    train_samples = _time_training(
        run_training, range(first_train_seed, first_train_seed + episodes), device
    )
    eval_samples = _time_evaluation(
        run_evaluation, range(eval_seed_start, eval_seed_start + episodes), device
    )

    print(f"\n{device} (config device={device_name!r}, n={episodes} episodes/path)")
    _report("training episode (acting + learning)", train_samples)
    _report("evaluation episode (acting only)     ", eval_samples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "chengdu" / "config.yaml",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="timed episodes per path per device"
    )
    parser.add_argument(
        "--first-train-seed",
        type=int,
        default=900000,
        help="offset well away from any real run's own training seeds",
    )
    parser.add_argument("--eval-seed-start", type=int, default=900100)
    parser.add_argument("--cache-dir", type=Path, default=world_cache.default_cache_dir())
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cpu-only", action="store_true", help="skip CUDA even if torch reports it available"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "override the config's data_dir with an absolute path -- needed when running "
            "from a git worktree, where the config's own relative '../../..' resolves "
            "against the worktree's nesting depth rather than the repo root"
        ),
    )
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    if args.data_dir is not None:
        config = dataclasses.replace(config, data_dir=args.data_dir)
    cache_dir = None if args.no_cache else args.cache_dir

    print(f"config: {args.config}")
    print("loading world...", flush=True)
    started = time.perf_counter()
    world = EpisodeWorld.load(config, cache_dir=cache_dir)
    print(f"world loaded in {time.perf_counter() - started:.1f}s")

    devices = ["cpu"]
    if not args.cpu_only and torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        measure(
            device_name,
            config,
            world,
            episodes=args.episodes,
            first_train_seed=args.first_train_seed,
            eval_seed_start=args.eval_seed_start,
        )

    if "cuda" not in devices and not args.cpu_only:
        print("\n(CUDA not available on this machine; ran CPU only)")


if __name__ == "__main__":
    main()
