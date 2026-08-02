"""mini_ab.py with a QHead.forward ablation, to isolate the dueling change.

Three shapes, patched in before any network is built so the null model and the
trained network agree on which one is in force:

- ``none``          — the working tree as-is: Q = V + (A - mean A).
- ``no-dueling``    — the pre-ticket-08 head: Q = A, no centring, no V. Lets a
                      run measure the cost warm start without the architecture
                      change riding along.
- ``value-scale=N`` — Q = N*V + (A - mean A). Tests the hypothesis for why
                      dueling degrades: the centring projects out the uniform
                      component the level error used to be absorbed by, so
                      until V reaches the return's magnitude the whole level
                      error lands as *differential* pressure on the ranking.
                      Scaling V's output is a per-step speedup of exactly N on
                      that branch, with no optimizer surgery.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.policies.network import QHead
from stdvrp.training.episode_pool import EpisodeWorld


def patch(ablate: str) -> None:
    if ablate == "none":
        return

    def rows(self, vehicle_embedding, client_embeddings, claimed, is_depot):
        n = client_embeddings.shape[0]
        vehicle = vehicle_embedding.unsqueeze(0).expand(n, -1)
        return torch.cat(
            [vehicle, client_embeddings, claimed.unsqueeze(-1), is_depot.unsqueeze(-1)], dim=-1
        )

    if ablate == "no-dueling":

        def forward(self, vehicle_embedding, client_embeddings, claimed, is_depot):
            x = rows(self, vehicle_embedding, client_embeddings, claimed, is_depot)
            return (self.linear(x) + self.layer2(torch.relu(self.layer1(x)))).squeeze(-1)

    elif ablate.startswith("value-scale="):
        scale = float(ablate.split("=", 1)[1])

        def forward(self, vehicle_embedding, client_embeddings, claimed, is_depot):
            x = rows(self, vehicle_embedding, client_embeddings, claimed, is_depot)
            advantage = (self.linear(x) + self.layer2(torch.relu(self.layer1(x)))).squeeze(-1)
            value = self.value2(torch.relu(self.value1(x.mean(dim=0)))).squeeze(-1)
            return (advantage - advantage.mean()) + scale * value

    else:
        raise SystemExit(f"unknown --ablate {ablate!r}")

    QHead.forward = forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-seeds", default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--ablate", default="none")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    patch(args.ablate)

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
    print(f"[{args.label}] ablate={args.ablate} null mean {null_mean:.2f}", flush=True)

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
                        "ablate": args.ablate,
                        "null_mean": null_mean,
                        "null_costs": null_costs,
                        "blocks": blocks,
                    },
                    indent=1,
                )
            )

    best = min(blocks, key=lambda block: block["mean"])
    print(
        f"[{args.label}] BEST ep {best['episode']} ({best['delta_pct']:+.2f}%)   "
        f"END {blocks[-1]['delta_pct']:+.2f}%",
        flush=True,
    )


main()
