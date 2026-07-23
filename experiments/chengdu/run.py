"""Run the complete Chengdu experiment from its ExperimentConfig (ticket 09).

Loads the world through the config's DataSource, trains with periodic evaluation
and best-W tracking, runs the final test over the configured seed/vehicle tables,
and writes ``results.json`` plus the training plot to a per-run output directory:

    uv run python experiments/chengdu/run.py [--config config.yaml] [--output-dir DIR]

The default output directory is ``runs/<timestamp>`` next to the config file
(gitignored). Loading the full Chengdu data cold takes ~15 minutes; the binary
world cache (ticket 03, simulation-performance) makes a repeat run with the same
data and world-shaping config load in seconds instead — on by default, see
``--cache-dir``/``--no-cache``. The training run itself depends on
``total_train_iterations``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache
from stdvrp.training import Trainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="experiment config YAML (default: config.yaml next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="per-run output directory (default: runs/<timestamp> next to the config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=world_cache.default_cache_dir(),
        help=(
            "binary world cache directory, ticket 03 (default: %(default)s, "
            "or STDVRP_WORLD_CACHE_DIR)"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="always rebuild the world from the CSVs and do not write the cache",
    )
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.config.parent / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_dir = None if args.no_cache else args.cache_dir

    print(f"config: {args.config}")
    print("loading world data (the full Chengdu archive takes ~15 minutes cold)...")
    trainer = Trainer.from_config(config, cache_dir=cache_dir, log=print)
    result = trainer.run(output_dir)

    best = result.training.best_mean_cost
    if best is not None:
        print(f"best evaluation mean cost: {best:.4f}")
    for report in result.test:
        mean, std = report.summary["total_cost"]
        print(f"final test actions={report.action_count}: mean cost {mean:.4f} (std {std:.4f})")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
