"""A/B measurement on the mini fixture: trained vs. the untrained null, paired.

Reproduces the methodology of issue 08's mini-fixture table (train N episodes,
evaluate greedily every K episodes over ten held-out seeds, report the paired
delta against the same-architecture untrained null). Run once per code state
(control = current working tree, treatment = cost-feature enrichment) and
compare best block / band / end-of-run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import stdvrp.simulation  # noqa: F401  (circular-import guard, as every neural script)
from stdvrp.config import ExperimentConfig
from stdvrp.training.episode_pool import EpisodeWorld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--eval-seeds", default="100,101,102,103,104,105,106,107,108,109"
    )
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from stdvrp.training.neural_episode import (
        build_neural_policy_state,
        run_neural_evaluation_episode,
        run_neural_training_episode,
    )

    config = ExperimentConfig.from_yaml(args.config)
    eval_seeds = [int(seed) for seed in args.eval_seeds.split(",")]
    world = EpisodeWorld.load(config, cache_dir=None)
    kwargs = dict(
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
        config=config,
    )

    null_state = build_neural_policy_state(config, np.random.default_rng(args.init_seed))
    null_costs = [
        run_neural_evaluation_episode(seed=seed, policy_state=null_state, **kwargs).total_cost
        for seed in eval_seeds
    ]
    null_mean = sum(null_costs) / len(null_costs)
    print(f"[{args.label}] null mean {null_mean:.2f}", flush=True)

    state = build_neural_policy_state(config, np.random.default_rng(args.init_seed))
    blocks = []
    start = time.monotonic()
    for episode in range(1, args.episodes + 1):
        seed = config.first_train_seed + episode - 1
        run_neural_training_episode(seed=seed, policy_state=state, **kwargs)
        if episode % args.eval_every == 0:
            costs = [
                run_neural_evaluation_episode(seed=seed, policy_state=state, **kwargs).total_cost
                for seed in eval_seeds
            ]
            mean = sum(costs) / len(costs)
            delta = 100.0 * (mean - null_mean) / null_mean
            wins = sum(1 for cost, null in zip(costs, null_costs) if cost < null)
            blocks.append(
                {"episode": episode, "mean": mean, "delta_pct": delta, "wins": wins, "costs": costs}
            )
            print(
                f"[{args.label}] ep {episode:4d}  mean {mean:9.2f}  vs null {delta:+7.2f}%  "
                f"wins {wins}/{len(eval_seeds)}  ({time.monotonic() - start:.0f}s)",
                flush=True,
            )
            args.out.write_text(
                json.dumps(
                    {
                        "label": args.label,
                        "null_mean": null_mean,
                        "null_costs": null_costs,
                        "blocks": blocks,
                    },
                    indent=1,
                )
            )

    best = min(blocks, key=lambda block: block["mean"])
    print(
        f"[{args.label}] BEST ep {best['episode']} ({best['delta_pct']:+.2f}%, "
        f"wins {best['wins']}/{len(eval_seeds)})   END {blocks[-1]['delta_pct']:+.2f}%",
        flush=True,
    )


main()
