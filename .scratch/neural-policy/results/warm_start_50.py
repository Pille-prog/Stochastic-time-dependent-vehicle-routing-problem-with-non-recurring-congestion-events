"""The two warm starts over the full evaluation_seeds set (50), paired.

The 8-seed probe that motivated the change is not a number to quote in a
config comment. This is: same seeds, same init_seed, greedy, per-seed pairing
so the Wilcoxon is meaningful. evaluation_seeds only -- never test_seeds.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy import stats

import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.training.episode_pool import EpisodeWorld
from stdvrp.training.neural_episode import build_neural_policy_state, run_neural_evaluation_episode
from stdvrp.traffic import world_cache


def main() -> None:
    base = ExperimentConfig.from_yaml("experiments/chengdu/config.yaml")
    base = dataclasses.replace(
        base, data_dir=Path("C:/Users/ferna/OneDrive/Documentos/Mega city"), device="cpu"
    )
    world = EpisodeWorld.load(base, cache_dir=world_cache.default_cache_dir())
    seeds = list(base.evaluation_seeds)
    kwargs = dict(
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
    )
    print(f"WORLD READY, {len(seeds)} seeds", flush=True)

    results = {}
    for name in ("minutes", "cost"):
        config = dataclasses.replace(base, neural_warm_start=name)
        state = build_neural_policy_state(config, np.random.default_rng(0))
        costs = []
        for index, seed in enumerate(seeds, start=1):
            result = run_neural_evaluation_episode(
                seed=seed, policy_state=state, config=config, **kwargs
            )
            costs.append(result.total_cost)
            if index % 10 == 0:
                print(f"  {name} {index}/50 running mean {np.mean(costs):.1f}", flush=True)
        results[name] = np.array(costs)
        print(f"{name:8s} mean {np.mean(costs):9.2f}  median {np.median(costs):9.2f}", flush=True)

    minutes, cost = results["minutes"], results["cost"]
    delta = 100.0 * (cost.mean() - minutes.mean()) / minutes.mean()
    statistic, p = stats.wilcoxon(cost, minutes)
    print(
        f"\nWARM START, 50 evaluation_seeds:"
        f"\n  minutes {minutes.mean():.2f}   cost {cost.mean():.2f}   delta {delta:+.2f}%"
        f"\n  cost wins on {(cost < minutes).sum()}/50 seeds"
        f"   Wilcoxon statistic={statistic:.1f} p={p:.2e}",
        flush=True,
    )


main()
