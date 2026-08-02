"""Ticket 14's missing cell (1): cost-greedy ranking rule, m+2 candidate set.

Zero training throughout. Same 50 ``evaluation_seeds`` as ``baseline_null_50.py``
and ``warm_start_50.py``, so the three files are read together as one table
(``results/README.md``). Two arms, both post-ticket-14 (the ``m+2``
``action_set.py`` shortlist is now live in ``_sweep``):

    1. cost warm start @ m+2, DEPOT_WARM_START_PENALTY as shipped (1.0)
       -- this IS the missing cell: holds the ranking rule fixed at
       cost-greedy and moves only the candidate set, against the 3693.23
       already measured at ~151 candidates (warm_start_50.py).
    2. cost warm start @ m+2, DEPOT_WARM_START_PENALTY = 0.0
       -- decides whether the penalty is still earning its keep once the
       depot enters the candidate list only where action_set.py admits it
       (ticket 14's own open question, "is_depot / DEPOT_WARM_START_PENALTY
       need a decision inside this ticket... decide by measurement").

Monkeypatches ``stdvrp.policies.network.DEPOT_WARM_START_PENALTY`` for arm 2
rather than adding a production knob -- QHead reads the module-level name at
construction time (``_init_weights``), so this is a real, isolated toggle, not
a config the shipped code carries.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy import stats

import stdvrp.policies.network as network_module
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld
from stdvrp.training.neural_episode import (
    build_neural_policy_state,
    run_neural_evaluation_episode,
)


def run_arm(world, seeds, config, kwargs, label, *, depot_penalty: float) -> np.ndarray:
    original_penalty = network_module.DEPOT_WARM_START_PENALTY
    network_module.DEPOT_WARM_START_PENALTY = depot_penalty
    try:
        state = build_neural_policy_state(config, np.random.default_rng(0))
    finally:
        network_module.DEPOT_WARM_START_PENALTY = original_penalty

    costs = []
    for index, seed in enumerate(seeds, start=1):
        result = run_neural_evaluation_episode(
            seed=seed, policy_state=state, config=config, **kwargs
        )
        costs.append(result.total_cost)
        if index % 10 == 0:
            print(f"  {label} {index}/{len(seeds)} running mean {np.mean(costs):.1f}", flush=True)
    array = np.array(costs)
    print(f"{label:28s} mean {array.mean():9.2f}   median {np.median(array):9.2f}", flush=True)
    return array


def main() -> None:
    base = ExperimentConfig.from_yaml("experiments/chengdu/config.yaml")
    base = dataclasses.replace(
        base,
        data_dir=Path("C:/Users/ferna/OneDrive/Documentos/Mega city"),
        device="cpu",
        neural_warm_start="cost",
    )
    world = EpisodeWorld.load(base, cache_dir=world_cache.default_cache_dir())
    seeds = list(base.evaluation_seeds)
    kwargs = dict(
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
    )
    print(f"WORLD READY, {len(seeds)} evaluation_seeds", flush=True)

    with_penalty = run_arm(world, seeds, base, kwargs, "cost @ m+2, penalty=1.0", depot_penalty=1.0)
    without_penalty = run_arm(
        world, seeds, base, kwargs, "cost @ m+2, penalty=0.0", depot_penalty=0.0
    )

    print("\n=== TICKET 14, CELL (1): cost-greedy @ m+2, 50 evaluation_seeds ===", flush=True)
    print(f"  cost @ m+2, penalty=1.0   {with_penalty.mean():9.2f}", flush=True)
    print(f"  cost @ m+2, penalty=0.0   {without_penalty.mean():9.2f}", flush=True)
    print("  cost @ ~151 (warm_start_50.py, already measured)   3693.23", flush=True)

    delta = 100.0 * (without_penalty.mean() - with_penalty.mean()) / with_penalty.mean()
    _statistic, p = stats.wilcoxon(with_penalty, without_penalty)
    print(
        f"\nDEPOT_WARM_START_PENALTY at m+2: penalty=0.0 vs penalty=1.0  {delta:+.2f}%"
        f"   wins(0.0 < 1.0) {(without_penalty < with_penalty).sum()}/{len(seeds)}"
        f"   p={p:.2e}",
        flush=True,
    )

    out = Path(".scratch/neural-policy/results/action_set_m2_50.json")
    import json

    out.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "cost_m2_penalty1": with_penalty.tolist(),
                "cost_m2_penalty0": without_penalty.tolist(),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


main()
