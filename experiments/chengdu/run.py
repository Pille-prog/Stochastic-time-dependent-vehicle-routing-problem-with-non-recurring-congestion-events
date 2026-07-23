"""Run the complete Chengdu experiment from its ExperimentConfig (ticket 09).

Loads the world through the config's DataSource, trains with periodic evaluation
and best-W tracking, runs the final test over the configured seed/vehicle tables,
and writes ``results.json`` plus the training plot to a per-run output directory:

    uv run python experiments/chengdu/run.py [--config config.yaml] [--output-dir DIR]

The default output directory is ``runs/<timestamp>`` next to the config file
(gitignored). Loading the full Chengdu data takes ~15 minutes; the training run
itself depends on ``total_train_iterations``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stdvrp.config import ExperimentConfig
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
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.config.parent / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"config: {args.config}")
    print("loading world data (the full Chengdu archive takes ~15 minutes)...")
    trainer = Trainer.from_config(config, log=print)
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
