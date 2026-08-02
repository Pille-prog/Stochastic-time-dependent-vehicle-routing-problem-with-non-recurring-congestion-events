"""The linear baseline's own null and its action-count axis, 50 evaluation_seeds.

Three arms, all through ``EpisodeWorld.run_episode`` -- the identical path
``Trainer._run_evaluation_block`` takes, so these are comparable to the
reference card's own ``evaluation_seed_costs``:

    1. W = 0        at m+2   (the trainer's evaluation default)
    2. best_w       at m+2   (what ticket 14 pins the network to)
    3. best_w       at m+40  (the reference card's winning cell)

evaluation_seeds only -- **never** test_seeds.

## What it found (2026-08-02)

    W=0    @ m+2    30791.43
    best_w @ m+2     2483.24     <- reproduces ticket 01's episode-50 figure
    best_w @ m+40    2168.39        to the cent: the harness is right

**1. The action count is worth 12.68%** (m+40 vs m+2, 36/50 seeds, Wilcoxon
p = 8.24e-05). On test_seeds the same axis reads 2.1%, so the effect
reproduces and its magnitude does not transfer. This is the measured basis
for ticket 14 reversing ADR-0007.

**2. W = 0 is NOT a myopic null, which is what this script set out to assume.**
The hypothesis was: ``_create_W`` is ``np.zeros(19)``, so ``X @ W == 0`` for
every candidate, ``np.argmin`` returns index 0, and ``_closest_allowed_clients``
orders nearest-first -- therefore W = 0 is "go to the nearest allowed Client",
a cheap myopic null that would decompose the baseline's advantage into
candidate heuristic versus learned weights.

30791.43 says otherwise. ``_select_vehicle_possible_actions``'s branch 3 runs
``possible_actions = list(set(possible_actions))`` *after* the nearest-first
sort, and node ids are arbitrary ints, so the dedup returns hash-table order:
**W = 0 picks an arbitrary feasible Client.** The quirk ADR-0001 preserves eats
the tie-break. The linear baseline has no cheap myopic null, and the
decomposition has to wait for ticket 14's own 2x2.

**3. F12's winner's curse is policy-dependent.** Read beside ``warm_start_50.py``
(same 50 seeds): ``best_w`` reads 2168.39 here against 3384.82 on test_seeds
(x1.56), while cost-greedy reads 3693.23 against 3811.28 (x1.03). A Gate A'
threshold defined as "the gap to the baseline" was drafted and reverted on
exactly this -- it computes to 41.3% on selection data and 11.2% on verdict
data, and neither is "the gap".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy import stats

from stdvrp.config import ExperimentConfig
from stdvrp.training.episode_pool import EpisodeRequest, EpisodeWorld
from stdvrp.traffic import world_cache

DATA_DIR = Path("C:/Users/ferna/OneDrive/Documentos/Mega city")
CARD = Path("experiments/chengdu/reference_card.json")


def run_arm(world, seeds, w, action_count, label):
    """One arm: greedy over every evaluation seed, generated fleet."""
    costs = []
    for index, seed in enumerate(seeds, start=1):
        if action_count is None:
            # As the evaluation blocks default: generated fleet, m + 2.
            request = EpisodeRequest(seed)
        else:
            demand = world.client_generator.generate(seed)
            request = EpisodeRequest(
                seed, number_actions_test=demand.vehicle_count + action_count
            )
        result = world.run_episode(w, request)
        costs.append(result.total_cost)
        if index % 10 == 0:
            print(f"  {label} {index}/{len(seeds)} running mean {np.mean(costs):.1f}", flush=True)
    array = np.array(costs)
    print(f"{label:24s} mean {array.mean():9.2f}   median {np.median(array):9.2f}", flush=True)
    return array


def main() -> None:
    config = ExperimentConfig.from_yaml("experiments/chengdu/config.yaml")
    config = dataclasses.replace(config, data_dir=DATA_DIR)
    card = json.loads(CARD.read_text())
    best_w = np.array(card["best_w"], dtype=np.float64)
    print(f"card: budget {card['winning_budget']} action_count {card['winning_test_action_count']}")

    world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
    seeds = list(config.evaluation_seeds)
    print(f"WORLD READY, {len(seeds)} evaluation_seeds", flush=True)

    zero = np.zeros_like(best_w)
    null_m2 = run_arm(world, seeds, zero, None, "W=0 @ m+2")
    trained_m2 = run_arm(world, seeds, best_w, None, "best_w @ m+2")
    trained_m40 = run_arm(world, seeds, best_w, 40, "best_w @ m+40")

    def compare(name, better, worse):
        delta = 100.0 * (better.mean() - worse.mean()) / worse.mean()
        _statistic, p = stats.wilcoxon(better, worse)
        print(
            f"  {name:34s} {delta:+7.2f}%   wins {(better < worse).sum():2d}/{len(seeds)}"
            f"   p={p:.2e}",
            flush=True,
        )

    print("\n=== BASELINE NULL, 50 evaluation_seeds ===", flush=True)
    print(f"  W=0    @ m+2   {null_m2.mean():9.2f}", flush=True)
    print(f"  best_w @ m+2   {trained_m2.mean():9.2f}", flush=True)
    print(f"  best_w @ m+40  {trained_m40.mean():9.2f}", flush=True)
    print("\nWhat learning is worth, and what the action count is worth:", flush=True)
    compare("learning @ m+2 (best_w vs W=0)", trained_m2, null_m2)
    compare("action count (m+40 vs m+2)", trained_m40, trained_m2)

    out = Path(".scratch/neural-policy/results/baseline_null_50.json")
    out.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "null_m2": null_m2.tolist(),
                "trained_m2": trained_m2.tolist(),
                "trained_m40": trained_m40.tolist(),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


main()
