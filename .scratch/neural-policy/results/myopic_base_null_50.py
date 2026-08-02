"""Ticket 15's acceptance check: ``W = 0`` under the new architecture must
reproduce the pre-ticket-15 ``"cost"`` warm start's untrained ``Q`` exactly --
"to the cent" on both the mini fixture and the real dataset.

Two independent comparisons, both zero training, both against a **recorded
constant** so this script stays re-runnable without any extra setup:

1. **Mini fixture** (``tests/fixtures/chengdu_mini``): this architecture's
   mean cost against **461.287099** -- the number a sibling git worktree
   checked out at ticket 15's parent commit (25c4ea6, the pre-ticket-15
   architecture) produced over the same seeds when this check was first run.
   Every one of the 50 per-seed costs agreed with the old architecture to the
   last printed digit (``max |old - new| = 0.0000000000``), not merely the
   mean -- see this file's git history / ticket 15's Comments for that raw
   comparison. No number was previously frozen for the mini fixture by any
   earlier ticket; this one is this ticket's own.
2. **Real dataset**, 50 ``evaluation_seeds``, ``DEPOT_WARM_START_PENALTY =
   1.0`` (as shipped): reproduces ``action_set_m2_50.py``'s arm 1 verbatim,
   against the frozen number ticket 14 recorded there, 3365.09
   (``.scratch/neural-policy/results/action_set_m2_50.py``).

Pass ``--old-src <path>`` (a checkout of any pre-ticket-15 commit's ``src/``,
e.g. from ``git worktree add <path> 25c4ea6``) to additionally re-derive the
461.287099 mini-fixture constant from scratch instead of trusting it --
that mode imports ``stdvrp`` twice, once from each ``sys.path`` entry, in one
process (a subprocess-per-codebase approach would need two venvs). Without
the flag, only this repository's current code runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA_DIR = Path("C:/Users/ferna/OneDrive/Documentos/Mega city")
SEEDS = range(100000, 100050)

# Recorded the first time this check ran (ticket 15's Comments): a sibling
# worktree at the pre-ticket-15 commit produced this exact mean, matching
# this architecture's to the last printed digit on every one of 50 seeds.
FROZEN_MINI_FIXTURE_NULL = 461.287099
TICKET_14_FROZEN_NULL = 3365.09


def _purge_stdvrp() -> None:
    for name in list(sys.modules):
        if name == "stdvrp" or name.startswith("stdvrp."):
            del sys.modules[name]


def _run_block(*, src_dir: Path, config_path: str, data_dir: Path | None, label: str):
    """Import ``stdvrp`` fresh from ``src_dir`` and run one greedy,
    untrained, m+2 block over ``SEEDS``. Returns the per-seed cost array."""
    _purge_stdvrp()
    original_path = list(sys.path)
    sys.path.insert(0, str(src_dir))
    try:
        importlib.invalidate_caches()
        import stdvrp.simulation  # noqa: F401  (circular-import landmine, ticket 03)
        from stdvrp.config import ExperimentConfig
        from stdvrp.traffic import world_cache
        from stdvrp.training.episode_pool import EpisodeWorld
        from stdvrp.training.neural_episode import (
            build_neural_policy_state,
            run_neural_evaluation_episode,
        )

        config = ExperimentConfig.from_yaml(config_path)
        if data_dir is not None:
            config = dataclasses.replace(config, data_dir=data_dir)
        config = dataclasses.replace(config, device="cpu", neural_warm_start="cost")

        world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
        state = build_neural_policy_state(config, np.random.default_rng(0))
        kwargs = dict(
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
        )

        costs = []
        for index, seed in enumerate(SEEDS, start=1):
            result = run_neural_evaluation_episode(
                seed=seed, policy_state=state, config=config, **kwargs
            )
            costs.append(result.total_cost)
            if index % 10 == 0:
                running_mean = np.mean(costs)
                print(f"  {label} {index}/{len(SEEDS)} running mean {running_mean:.4f}", flush=True)
        array = np.array(costs)
        print(f"{label:32s} mean {array.mean():.6f}  n={len(array)}", flush=True)
        return array
    finally:
        sys.path[:] = original_path
        _purge_stdvrp()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-src",
        type=Path,
        default=None,
        help="src/ of a pre-ticket-15 checkout, to re-derive the mini-fixture "
        "constant from scratch instead of trusting it (see module docstring).",
    )
    args = parser.parse_args()

    mini_config = str(REPO_ROOT / "tests" / "fixtures" / "chengdu_mini" / "config.yaml")
    real_config = str(REPO_ROOT / "experiments" / "chengdu" / "config.yaml")

    print("=== Mini fixture: this architecture vs the recorded constant ===", flush=True)
    new_mini = _run_block(
        src_dir=REPO_ROOT / "src", config_path=mini_config, data_dir=None, label="new, mini fixture"
    )
    mini_delta = float(new_mini.mean() - FROZEN_MINI_FIXTURE_NULL)
    print(f"recorded constant (pre-ticket-15): {FROZEN_MINI_FIXTURE_NULL:.6f}", flush=True)
    print(f"this run:                          {new_mini.mean():.6f}", flush=True)
    print(f"delta:                             {mini_delta:+.6f}", flush=True)

    old_mini = None
    if args.old_src is not None:
        print("\n=== Mini fixture: re-deriving the constant from --old-src ===", flush=True)
        old_mini = _run_block(
            src_dir=args.old_src, config_path=mini_config, data_dir=None, label="old, mini fixture"
        )
        max_abs_diff = float(np.max(np.abs(old_mini - new_mini)))
        print(f"max |old - new| per seed: {max_abs_diff:.10f}", flush=True)

    print("\n=== Real dataset: this architecture vs ticket 14's frozen 3365.09 ===", flush=True)
    new_real = _run_block(
        src_dir=REPO_ROOT / "src",
        config_path=real_config,
        data_dir=REAL_DATA_DIR,
        label="new, real dataset",
    )
    real_delta = float(new_real.mean() - TICKET_14_FROZEN_NULL)
    print(f"ticket 14 frozen null (old architecture): {TICKET_14_FROZEN_NULL:.6f}", flush=True)
    print(f"this run (new architecture):               {new_real.mean():.6f}", flush=True)
    print(f"delta:                                     {real_delta:+.6f}", flush=True)

    out = Path(".scratch/neural-policy/results/myopic_base_null_50.json")
    out.write_text(
        json.dumps(
            {
                "mini_fixture": {
                    "seeds": list(SEEDS),
                    "new": new_mini.tolist(),
                    "frozen_constant": FROZEN_MINI_FIXTURE_NULL,
                    "delta_from_frozen_constant": mini_delta,
                    "old": old_mini.tolist() if old_mini is not None else None,
                },
                "real_dataset": {
                    "seeds": list(SEEDS),
                    "new": new_real.tolist(),
                    "new_mean": float(new_real.mean()),
                    "ticket_14_frozen_null": TICKET_14_FROZEN_NULL,
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


main()
