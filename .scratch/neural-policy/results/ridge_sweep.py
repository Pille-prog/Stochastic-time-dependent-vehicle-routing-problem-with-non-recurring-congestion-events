"""Ticket 16's acceptance check: choose `neural_ridge_gamma` / `neural_ridge_lambda`
/ `neural_solve_cadence` on `evaluation_seeds`, sweep recorded.

Real Chengdu dataset (`experiments/chengdu/config.yaml`, `data_dir` overridden
to this machine's copy -- the repo's own relative `data_dir` no longer resolves,
see the memory note "Dataset path override"). A world-cache directory is used
so every combination in the grid reuses the same parsed CSVs.

A first pilot run (this file's own development, recorded in ticket 16's
Comments) found `neural_solve_cadence: 1` -- solving after a single Episode,
against 515 parameters -- produces a wildly unstable early fit: two training
Episodes contributed ~20-40 samples total, the frozen column scale was
estimated from that same tiny sample, and the resulting policy scored *worse*
than an aborting one on the very next evaluation seed. That motivates this
sweep varying `neural_solve_cadence` explicitly, not just gamma/lambda at a
fixed (and, it turns out, too-short) cadence.

For each (gamma, lambda, N) combination: train `--episodes` Episodes from a
fixed init seed, then evaluate once over all 50 `evaluation_seeds`. Reports
mean cost, wins/Wilcoxon-p against the untrained null (`W = 0`, ticket 15's
frozen 3365.09 on these same seeds -- not re-measured, since `decide()` is
untouched by ticket 16 and W stays exactly zero before the first solve), and
the exclusion count.

Run across four rounds, editing GRID_GAMMA/GRID_LAMBDA/GRID_CADENCE/
TRAIN_EPISODES between each and renaming the output file (this file always
reflects the *last* round's grid, round 4's -- see ticket 16's Comments for
the full four-round table): round 1 (lambda in {1, 10, 100}, N=50) ->
ridge_sweep_round1.{json,log}; round 2 (lambda in {1e3, 1e4, 1e5}, N=50) ->
round2; round 3 (lambda=1, N=150) -> round3; round 4 (gamma in
{0.90, 0.95, 0.99}, lambda=1, N=50) -> round4. Round 4 is the only one run
after the raw_sum_sq forgetting bug (ticket 16's Comments, found by code
review) was fixed -- rounds 1-3's absolute numbers are approximate.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from pathlib import Path

import numpy as np

import stdvrp.simulation  # noqa: F401  (circular-import landmine, ticket 03)
from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld
from stdvrp.training.neural_episode import (
    build_neural_policy_state,
    run_neural_evaluation_episode,
    run_neural_training_episode,
)
from stdvrp.training.neural_report import paired_wilcoxon_p

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA_DIR = Path("C:/Users/ferna/OneDrive/Documentos/Mega city")
EVALUATION_SEEDS = tuple(range(100000, 100050))

# Ticket 14/15's frozen untrained (W = 0) null on these exact 50 seeds --
# not re-measured here: decide()'s code path is untouched by ticket 16, and
# W is exactly zero before the first solve (RidgeAccumulator.zeros' own start
# state), so this number is unchanged by construction, not by re-derivation.
FROZEN_NULL_MEAN = 3365.092529


# Round 4: rounds 1-3 held gamma=0.98 fixed throughout and swept only lambda
# and N -- the ticket asks for both gamma and lambda swept. This round holds
# the best cell found so far (lambda=1, N=50) and varies gamma instead, to
# check whether the effective window (~1/(1-gamma) episodes) matters as much
# as lambda did.
GRID_GAMMA = (0.90, 0.95, 0.99)
GRID_LAMBDA = (1.0,)
GRID_CADENCE = (50,)
TRAIN_EPISODES = 60
FIRST_TRAIN_SEED = 1000
INIT_SEED = 0


def run_one(world: EpisodeWorld, config: ExperimentConfig) -> dict:
    kwargs = dict(
        client_generator=world.client_generator,
        travel_time_model=world.travel_time_model,
        shortest_path_cache=world.shortest_path_cache,
        congestion_generator=world.congestion_generator,
    )
    state = build_neural_policy_state(config, np.random.default_rng(INIT_SEED))

    train_start = time.time()
    for index in range(TRAIN_EPISODES):
        seed = FIRST_TRAIN_SEED + index
        run_neural_training_episode(seed=seed, policy_state=state, config=config, **kwargs)
    train_seconds = time.time() - train_start

    eval_start = time.time()
    costs = [
        run_neural_evaluation_episode(
            seed=seed, policy_state=state, config=config, **kwargs
        ).total_cost
        for seed in EVALUATION_SEEDS
    ]
    eval_seconds = time.time() - eval_start

    costs_array = np.array(costs)
    null_array = np.full_like(costs_array, FROZEN_NULL_MEAN)
    wins = int((costs_array < null_array).sum())
    p_value = paired_wilcoxon_p(tuple(costs), tuple(null_array.tolist()))

    return {
        "gamma": config.neural_ridge_gamma,
        "lambda": config.neural_ridge_lambda,
        "cadence": config.neural_solve_cadence,
        "effective_n": state.ridge.effective_n,
        "mean_cost": float(costs_array.mean()),
        "wins_vs_null": wins,
        "wilcoxon_p_vs_null": p_value,
        "episodes_included": state.ridge.episodes_included,
        "episodes_excluded": state.ridge.episodes_excluded,
        "w_norm": float(np.linalg.norm(state.head.w_vector().numpy())),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "seed_costs": costs,
    }


def main() -> None:
    config = ExperimentConfig.from_yaml(str(REPO_ROOT / "experiments" / "chengdu" / "config.yaml"))
    config = dataclasses.replace(config, data_dir=REAL_DATA_DIR, device="cpu")
    world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())

    print(f"untrained null (frozen, ticket 14/15): {FROZEN_NULL_MEAN:.4f}", flush=True)

    results = []
    combos = list(itertools.product(GRID_GAMMA, GRID_LAMBDA, GRID_CADENCE))
    for index, (gamma, lam, cadence) in enumerate(combos, start=1):
        run_config = dataclasses.replace(
            config,
            neural_ridge_gamma=gamma,
            neural_ridge_lambda=lam,
            neural_solve_cadence=cadence,
        )
        print(
            f"\n=== [{index}/{len(combos)}] gamma={gamma} lambda={lam} cadence={cadence} ===",
            flush=True,
        )
        result = run_one(world, run_config)
        print(
            f"  mean {result['mean_cost']:.2f}  wins {result['wins_vs_null']}/50  "
            f"p={result['wilcoxon_p_vs_null']:.4f}  |W|={result['w_norm']:.4f}  "
            f"effective_n={result['effective_n']:.1f}  "
            f"excluded {result['episodes_excluded']}/{TRAIN_EPISODES}  "
            f"train {result['train_seconds']:.1f}s eval {result['eval_seconds']:.1f}s",
            flush=True,
        )
        results.append(result)

    out = Path(__file__).resolve().parent / "ridge_sweep_round4.json"
    out.write_text(json.dumps({"frozen_null_mean": FROZEN_NULL_MEAN, "results": results}, indent=2))
    print(f"\nwrote {out}", flush=True)

    best = min(results, key=lambda r: r["mean_cost"])
    delta_pct = 100 * (best["mean_cost"] - FROZEN_NULL_MEAN) / FROZEN_NULL_MEAN
    print(
        f"\nBEST: gamma={best['gamma']} lambda={best['lambda']} cadence={best['cadence']} "
        f"mean={best['mean_cost']:.2f} ({delta_pct:+.1f}% vs null)",
        flush=True,
    )


main()
