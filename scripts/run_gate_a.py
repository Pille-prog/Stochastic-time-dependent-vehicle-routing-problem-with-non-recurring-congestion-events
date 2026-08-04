"""Gate A -- does the transformer Policy learn at all (ticket 08, neural-policy).

The hard landing gate: trains ``--init-seeds`` independent networks to
convergence and measures each against one shared untrained (null) network,
paired per seed over ``test_seeds`` -- never ``evaluation_seeds``. See
``stdvrp.training.gate_a``'s module docstring for the three parts (null
model, reproducibility, calibration) and spec.md's "Frozen parameters" table
for the thresholds.

Develop and debug against the mini fixture first (fast, no 8 GB world load;
its own ``test_seeds`` is only 2 entries -- enough to exercise the plumbing,
not to draw a conclusion from)::

    uv run python scripts/run_gate_a.py \\
        --config tests/fixtures/chengdu_mini/config.yaml \\
        --reference-card /tmp/dummy_card.json \\
        --init-seeds 0,1 --max-episodes 4 --eval-cadence-minimum 4

Run the actual gate against the real dataset (hours per arm -- this is a
"train to convergence" run, not a fixed budget; the safety cap is 10 000
episodes or 24h *per arm*, spec.md decision 12)::

    uv run python scripts/run_gate_a.py \\
        --config experiments/chengdu/config.yaml \\
        --reference-card experiments/chengdu/reference_card.json

A dummy reference card is fine for the mini fixture (Gate A's own statistics
never read it -- it only feeds ``train_neural``'s live per-episode report);
the real run should point at the real, already-committed
``experiments/chengdu/reference_card.json`` (ticket 01).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Same circular-import landmine every neural script/test in this repo guards
# against (see benchmark_neural_stub.py's own comment): stdvrp.simulation must
# finish initializing before anything reaches stdvrp.policies.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.policies.torch_support import TORCH_INSTALL_HINT
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld
from stdvrp.training.reference_card import ReferenceCard

try:
    import torch  # noqa: F401
except ImportError:
    print(f"scripts/run_gate_a.py needs torch: {TORCH_INSTALL_HINT}", file=sys.stderr)
    raise SystemExit(1) from None

from stdvrp.training.gate_a import format_gate_a_report, run_gate_a


def result_as_json(result: object) -> dict[str, object]:
    """Every arm's numbers, so a real run's evidence survives the terminal it ran in."""
    from stdvrp.training.gate_a import GateAResult  # local: only needed for the isinstance check

    assert isinstance(result, GateAResult)
    return {
        "passes": result.passes,
        "null_model_passes": result.null_model_passes,
        "reproducibility_passes": result.reproducibility_passes,
        "reproducibility_mean_pct": result.reproducibility_mean_pct,
        "reproducibility_sd_pct": result.reproducibility_sd_pct,
        "reproducibility_spread_straddles_zero": result.reproducibility_spread_straddles_zero,
        "calibration_passes": result.calibration_passes,
        "arms": [
            {
                "init_seed": arm.init_seed,
                "episodes_completed": arm.episodes_completed,
                "converged": arm.converged,
                "mean_reduction_pct": arm.mean_reduction_pct,
                "median_reduction_pct": arm.median_reduction_pct,
                "wilcoxon_p": arm.wilcoxon_p,
                "null_model_passes": arm.null_model_passes,
                "trained_calibration_spearman": arm.trained_calibration_spearman,
                "untrained_calibration_spearman": arm.untrained_calibration_spearman,
                "calibration_passes": arm.calibration_passes,
                "trained_seed_costs": list(arm.trained_seed_costs),
                "untrained_seed_costs": list(arm.untrained_seed_costs),
            }
            for arm in result.arms
        ],
    }


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
        "--reference-card",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments"
        / "chengdu"
        / "reference_card.json",
    )
    parser.add_argument(
        "--init-seeds",
        default="0,1,2",
        help="comma-separated network-init seeds (>= 3 for a real gate)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="one checkpoint per init seed (default: a runs/ sibling, gitignored)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="override the 10 000-episode safety cap"
    )
    parser.add_argument("--max-hours", type=float, default=None, help="override the 24h safety cap")
    parser.add_argument(
        "--eval-cadence-minimum",
        type=int,
        default=None,
        help="override the ~every-50-episodes floor",
    )
    parser.add_argument("--cache-dir", type=Path, default=world_cache.default_cache_dir())
    parser.add_argument("--no-cache", action="store_true")
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
    parser.add_argument(
        "--results-path", type=Path, default=None, help="write the full report as JSON here"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="override the config's device (ticket 12: 'auto' resolves once per run and is "
        "not guaranteed to be the faster choice on every machine -- see ticket 12's Comments)",
    )
    args = parser.parse_args(argv)

    init_seeds = tuple(int(value) for value in args.init_seeds.split(","))
    cache_dir = None if args.no_cache else args.cache_dir
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else Path(__file__).resolve().parents[1] / "runs" / "gate_a"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.from_yaml(args.config)
    if args.data_dir is not None:
        config = dataclasses.replace(config, data_dir=args.data_dir)
    if args.device is not None:
        config = dataclasses.replace(config, device=args.device)
    reference_card = ReferenceCard.load(args.reference_card)

    print(f"config: {args.config}")
    print(f"data_dir: {config.data_dir}")
    print(f"test_seeds: {len(config.test_seeds)} seeds")
    print(f"init_seeds: {init_seeds}")
    print(f"checkpoint_dir: {checkpoint_dir}")

    world = EpisodeWorld.load(config, cache_dir=cache_dir)
    result = run_gate_a(
        world,
        reference_card=reference_card,
        checkpoint_dir=checkpoint_dir,
        init_seeds=init_seeds,
        max_episodes=args.max_episodes,
        max_hours=args.max_hours,
        evaluation_cadence_minimum=args.eval_cadence_minimum,
        log=print,
    )

    print()
    print(format_gate_a_report(result))

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        args.results_path.write_text(
            json.dumps(result_as_json(result), indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"\nfull results written to {args.results_path}")


if __name__ == "__main__":
    main()
