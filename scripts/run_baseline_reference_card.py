"""The linear MonteCarloPolicy's frozen reference card (ticket 01, neural-policy).

.. warning::

   **The committed ``experiments/chengdu/reference_card.json`` is stale.** It
   records ``config.time_window_spread: 60``; the Chengdu configs now say 150,
   which is what the reference runs actually used
   (``capture_golden_master.py``'s ``diff_TW``, and the ``Tw150`` in the
   legacy's own result filenames). The mismatch was found by
   ``.scratch/linear-policy-learning/issues/01-legacy-w-trajectory-diff.md``;
   at TW 150 the linear Policy's ``|W|`` after 200 episodes is 15 754 against
   7 502 at TW 60, so the two are not comparable.

   The card is left untouched on purpose: it is a *measurement record*, and
   editing the config it embeds would misreport what was measured. (Its
   ``best_w`` is 19 components wide against today's 24 for the same reason —
   it predates the restored features.) Re-run this script to replace it, and
   note that every number pinned against it — the neural effort's Gate A and
   Gate A' results, and everything under ``runs/`` — was measured at TW 60.

The first full training run of the repaired simulator: there is no baseline
number yet (``.scratch/simulator-correctness/spec.md`` deliberately left it
out — "the first experiment of the repaired lab, not part of repairing it").
This script runs the existing linear ``MonteCarloPolicy`` at three training
budgets, each with the full ``test_action_counts`` sweep, and freezes the
winning cell's per-seed costs to ``ReferenceCard`` — the fixed opponent every
later Policy in this effort is measured against (spec.md, "Why the paired
comparison is valid")::

    uv run python scripts/run_baseline_reference_card.py \
        --config experiments/chengdu/config.yaml \
        --card-path experiments/chengdu/reference_card.json

Evaluation cadence is ``--eval-cadence`` (default 50, per spec.md's frozen
parameters), **not** the config's own ``test_frequency`` — at 2000 training
episodes, ``test_frequency: 10`` would cost 10 000 evaluation episodes, more
than training itself. Each budget's own ``results.json`` and training plot
land under ``--output-dir`` (default a ``runs/`` sibling, gitignored, for
inspection); the ``ReferenceCard`` at ``--card-path`` is the ticket's
committed artifact.

Wall-clock is per phase, per budget: world load, ``train()`` (which interleaves
the periodic evaluation blocks — the Trainer does not time them separately),
and ``final_test()``. The winning budget's own numbers are the ones the
``ReferenceCard`` carries forward, so ticket 03's transformer measurement has
something to be a multiple of.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache
from stdvrp.training import ActionCountReport, ReferenceCard, Trainer, TrainingResult
from stdvrp.training.trainer import (
    ExperimentResult,
    config_as_json,
    write_results,
    write_training_plot,
)


@dataclasses.dataclass(frozen=True)
class BudgetRun:
    """One training budget's complete outcome: results plus phase wall-clock."""

    budget: int
    training: TrainingResult
    test: tuple[ActionCountReport, ...]
    world_load_seconds: float
    train_seconds: float
    final_test_seconds: float


def run_budget(
    base_config: ExperimentConfig,
    budget: int,
    *,
    eval_cadence: int,
    cache_dir: Path | None,
    worker_count: int,
    output_dir: Path | None,
) -> BudgetRun:
    """Train and final-test one budget; write its own results.json/plot if asked."""
    config = dataclasses.replace(
        base_config, total_train_iterations=budget, test_frequency=eval_cadence
    )

    started = time.perf_counter()
    trainer = Trainer.from_config(config, cache_dir=cache_dir, worker_count=worker_count, log=print)
    world_load_seconds = time.perf_counter() - started

    try:
        started = time.perf_counter()
        training = trainer.train()
        train_seconds = time.perf_counter() - started

        tested_w = training.best_w if training.best_w is not None else training.w_trajectory[-1]

        started = time.perf_counter()
        test = trainer.final_test(tested_w)
        final_test_seconds = time.perf_counter() - started
    finally:
        trainer.close()

    if output_dir is not None:
        budget_dir = output_dir / f"budget_{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        result = ExperimentResult(training=training, test=test, tested_w=tested_w)
        write_results(budget_dir / "results.json", config, result)
        write_training_plot(budget_dir / "training_plot.png", training.evaluations, None)

    return BudgetRun(
        budget=budget,
        training=training,
        test=test,
        world_load_seconds=world_load_seconds,
        train_seconds=train_seconds,
        final_test_seconds=final_test_seconds,
    )


def winning_cell(runs: list[BudgetRun]) -> tuple[BudgetRun, ActionCountReport]:
    """The (budget, test_action_count) cell with the lowest mean total_cost."""
    candidates = [
        (run, report, report.summary["total_cost"][0]) for run in runs for report in run.test
    ]
    best_run, best_report, _ = min(candidates, key=lambda candidate: candidate[2])
    return best_run, best_report


def build_reference_card(
    run: BudgetRun, report: ActionCountReport, config: ExperimentConfig
) -> ReferenceCard:
    """Freeze the winning cell: its per-seed test vector and its best evaluation block."""
    best_block = min(run.training.evaluations, key=lambda block: block.mean_cost)
    tested_w = (
        run.training.best_w if run.training.best_w is not None else run.training.w_trajectory[-1]
    )
    return ReferenceCard(
        winning_budget=run.budget,
        winning_test_action_count=report.action_count,
        test_seeds=tuple(entry.seed for entry in report.per_seed),
        test_seed_costs=tuple(entry.metrics["total_cost"] for entry in report.per_seed),
        evaluation_seeds=config.evaluation_seeds,
        evaluation_seed_costs=best_block.seed_costs,
        best_w=tuple(float(x) for x in tested_w),
        config=config_as_json(config),
        wall_clock_seconds={
            "world_load": run.world_load_seconds,
            "train": run.train_seconds,
            "train_episodes": run.budget,
            "final_test": run.final_test_seconds,
            "final_test_episodes": sum(len(r.per_seed) for r in run.test),
        },
    )


def print_side_by_side(runs: list[BudgetRun]) -> None:
    """Every (budget, action_count) cell's mean +/- std, so a longer budget getting
    worse is as visible as one getting better (acceptance: report side by side)."""
    print("\nbudget x test_action_count  (mean total_cost +/- std, over test_seeds)")
    header = "action_count".rjust(14) + "".join(f"{run.budget:>16}" for run in runs)
    print(header)
    for index, report in enumerate(runs[0].test):
        row = f"{report.action_count:>14}"
        for run in runs:
            mean, std = run.test[index].summary["total_cost"]
            row += f"{mean:>10.1f}+-{std:<4.0f}"
        print(row)

    print("\nwall-clock per phase (seconds)")
    for run in runs:
        print(
            f"  budget {run.budget:>5}: world_load {run.world_load_seconds:8.1f}s   "
            f"train {run.train_seconds:9.1f}s ({run.train_seconds / run.budget:.3f} s/ep)   "
            f"final_test {run.final_test_seconds:8.1f}s"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "chengdu" / "config.yaml",
    )
    parser.add_argument(
        "--budgets", default="100,500,2000", help="comma-separated training budgets"
    )
    parser.add_argument(
        "--eval-cadence", type=int, default=50, help="evaluation every N training episodes"
    )
    parser.add_argument(
        "--card-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments"
        / "chengdu"
        / "reference_card.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="per-budget results.json/plot (default: none written)",
    )
    parser.add_argument("--cache-dir", type=Path, default=world_cache.default_cache_dir())
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "override the config's data_dir with an absolute path — needed when running "
            "from a git worktree, where the config's own relative '../../..' resolves "
            "against the worktree's nesting depth rather than the repo root"
        ),
    )
    args = parser.parse_args(argv)
    cache_dir = None if args.no_cache else args.cache_dir
    budgets = [int(value) for value in args.budgets.split(",")]

    # ``portable_config`` keeps the YAML's own (relative) data_dir for the card's
    # config snapshot; ``base_config`` may override it to an absolute path for
    # this run's own world loading (a worktree-nesting workaround, not something
    # that belongs baked into a committed, machine-independent artifact).
    portable_config = ExperimentConfig.from_yaml(args.config)
    base_config = portable_config
    if args.data_dir is not None:
        base_config = dataclasses.replace(base_config, data_dir=args.data_dir)
    print(f"config: {args.config}")
    print(f"data_dir: {base_config.data_dir}")
    print(f"budgets: {budgets}   eval cadence: every {args.eval_cadence} episodes")

    runs: list[BudgetRun] = []
    for budget in budgets:
        print(f"\n=== budget {budget} ===", flush=True)
        run = run_budget(
            base_config,
            budget,
            eval_cadence=args.eval_cadence,
            cache_dir=cache_dir,
            worker_count=args.workers,
            output_dir=args.output_dir,
        )
        runs.append(run)
        best = run.training.best_mean_cost
        print(
            f"budget {budget} done: world_load {run.world_load_seconds:.1f}s  "
            f"train {run.train_seconds:.1f}s  final_test {run.final_test_seconds:.1f}s  "
            f"best eval mean cost {best}"
        )

    print_side_by_side(runs)

    winning_run, winning_report = winning_cell(runs)
    mean_costs = sorted(
        {round(report.summary["total_cost"][0], 6) for run in runs for report in run.test}
    )
    margin = mean_costs[1] - mean_costs[0] if len(mean_costs) > 1 else 0.0
    print(
        f"\nwinning cell: budget={winning_run.budget} "
        f"test_action_count={winning_report.action_count} "
        f"mean_cost={winning_report.summary['total_cost'][0]:.4f} "
        f"(margin over next-best cell: {margin:.4f})"
    )

    winning_config = dataclasses.replace(
        portable_config, total_train_iterations=winning_run.budget, test_frequency=args.eval_cadence
    )
    card = build_reference_card(winning_run, winning_report, winning_config)
    args.card_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(args.card_path)
    print(f"\nreference card written to {args.card_path}")
    print(f"  test_mean_cost:       {card.test_mean_cost:.4f}  ({len(card.test_seeds)} seeds)")
    print(
        f"  evaluation_mean_cost: {card.evaluation_mean_cost:.4f}  "
        f"({len(card.evaluation_seeds)} seeds)"
    )


if __name__ == "__main__":
    main()
