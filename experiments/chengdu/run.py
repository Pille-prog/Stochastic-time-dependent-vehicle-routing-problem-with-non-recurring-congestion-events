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

The evaluation blocks and the final test can run on a worker pool (ticket 08,
simulation-performance): ``--workers N`` (or ``STDVRP_WORKERS``) spreads them
over N processes, and the results are identical either way. It is opt-in, and
the ceiling is memory rather than cores: each worker holds its own copy of the
world, 8.0 GB of it on the full Chengdu data, so a 32 GB machine fits two
workers beside the training process. See the ticket's benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache
from stdvrp.training import Trainer, default_worker_count


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "worker processes for the evaluation blocks and the final test, ticket 08 "
            "(default: STDVRP_WORKERS, else 1 — everything in this process). Each "
            "worker holds its own copy of the world: 8.0 GB on the full Chengdu data"
        ),
    )
    args = parser.parse_args(argv)
    # Resolved after parsing, never as an argparse default: reading the
    # environment eagerly would make a malformed STDVRP_WORKERS break --help too.
    if args.workers is None:
        args.workers = default_worker_count()
    if args.no_cache and args.workers > 1:
        parser.error(
            "--no-cache needs --workers 1: each worker loads its world through the "
            "binary cache, so without it every worker would re-parse the CSVs"
        )

    config = ExperimentConfig.from_yaml(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.config.parent / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_dir = None if args.no_cache else args.cache_dir

    print(f"config: {args.config}")
    print(f"evaluating on {args.workers} worker process(es)")
    print("loading world data (the full Chengdu archive takes ~15 minutes cold)...")
    trainer = Trainer.from_config(config, cache_dir=cache_dir, worker_count=args.workers, log=print)
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
