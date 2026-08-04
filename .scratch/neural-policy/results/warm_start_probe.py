"""Two diagnostics for ticket 08, on the real dataset's evaluation_seeds.

1. **Cost decomposition** of the untrained null vs. the best checkpoint of the
   live Gate A run: is the trained network's regression (5484 -> 4150 -> 9517)
   a ranking that got worse, or vehicles retiring early and leaving Clients
   unserved?

2. **Warm-start variants.** ``arc_embed`` row 0 is the whole of ``Q`` at init
   (network.py). Today it is ``[1,0,0,0,0,0]`` -> ``Q = minutes/H``, the
   nearest-feasible-Client null. The arc token now carries the four projected
   cost components, so other rows of that same vector are other myopic
   policies, at zero training cost. Which one is cheapest?

   Token fields: [minutes, length, earliness, delay, future_delay, overtime],
   the first four scaled by 1/H and future_delay by 1/(H*total_clients), so a
   weight of ``total_clients`` on field 4 puts it in the same units as the rest.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.training.episode_pool import EpisodeWorld
from stdvrp.training.neural_episode import build_neural_policy_state, run_neural_evaluation_episode
from stdvrp.traffic import world_cache

TOTAL_CLIENTS = 150.0

VARIANTS: dict[str, list[float]] = {
    "null: minutes                 ": [1.0, 0, 0, 0, 0, 0],
    "cost only                     ": [0.0, 0, 1, 1, 0, 1],
    "cost + minutes                ": [1.0, 0, 1, 1, 0, 1],
    "cost + minutes + future_delay ": [1.0, 0, 1, 1, TOTAL_CLIENTS, 1],
    "future_delay only             ": [0.0, 0, 0, 0, TOTAL_CLIENTS, 0],
    "cost + min + 0.1*future_delay ": [1.0, 0, 1, 1, 0.1 * TOTAL_CLIENTS, 1],
}


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    config = ExperimentConfig.from_yaml("experiments/chengdu/config.yaml")
    config = dataclasses.replace(
        config, data_dir=Path("C:/Users/ferna/OneDrive/Documentos/Mega city"), device="cpu"
    )
    world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
    seeds = list(config.evaluation_seeds)[:n_seeds]
    kwargs = dict(
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
        config=config,
    )
    print(f"WORLD READY, {len(seeds)} seeds", flush=True)

    def sweep(state, label: str) -> None:
        results = [run_neural_evaluation_episode(seed=s, policy_state=state, **kwargs) for s in seeds]
        mean = lambda f: sum(f(r) for r in results) / len(results)  # noqa: E731
        print(
            f"{label}  total {mean(lambda r: r.total_cost):9.1f}"
            f" | delay {mean(lambda r: r.delay_cost):8.1f}"
            f" early {mean(lambda r: r.earliness_cost):7.1f}"
            f" over {mean(lambda r: r.overtime_cost):7.1f}"
            f" | unserved {mean(lambda r: r.unserved_clients):5.1f}"
            f" late {mean(lambda r: r.delay_clients):5.1f}"
            f" | tau {mean(lambda r: r.tau):6.1f}",
            flush=True,
        )

    # --- 1. the live run's best checkpoint vs. its own untrained null ---------
    state = build_neural_policy_state(config, np.random.default_rng(0))
    sweep(state, "untrained null                ")

    checkpoint_path = Path("runs/gate_a_v2/gate_a_init0.pt")
    if checkpoint_path.exists():
        document = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best = document.get("best_weights")
        for name, source in (("latest", document), ("best-block", best)):
            if source is None:
                continue
            trained = build_neural_policy_state(config, np.random.default_rng(0))
            trained.encoder.load_state_dict(source["encoder_state"])
            trained.head.load_state_dict(source["head_state"])
            sweep(trained, f"gate_a checkpoint ({name:10s})")

    # --- 2. warm-start variants ----------------------------------------------
    for label, weights in VARIANTS.items():
        variant = build_neural_policy_state(config, np.random.default_rng(0))
        with torch.no_grad():
            variant.encoder.arc_embed.weight[0, :] = torch.tensor(weights, dtype=torch.float32)
        sweep(variant, label)


main()
