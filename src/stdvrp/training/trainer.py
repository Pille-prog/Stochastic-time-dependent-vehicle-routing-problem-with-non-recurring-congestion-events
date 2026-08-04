"""Trainer: the complete experiment as one config-driven run (ticket 09).

Ports ``training_and_testing.training_model`` / ``test_model`` (ADR-0001): the
training loop with periodic evaluation and best-W tracking, then the final test
over the configured seed/vehicle tables, writing ``results.json`` and the
training plot to a per-run output directory. Every value the legacy hardcoded —
horizon, ``n_arcs``, the warm-up learning rate, the evaluation seed range, the
test seed/fleet tables — comes from ``ExperimentConfig``. The legacy's
hardcoded ``mean_static_policy`` plot baseline is a ``ReferenceCard``
(``stdvrp.training.reference_card``, ticket 01) passed to :meth:`Trainer.run`
instead — a frozen per-seed cost vector, not a config field.

Legacy fidelity notes:

* **Warm-up learning rate** (made optional in ticket 12): when
  ``warmup_learning_rate`` is set, the first training Episode updates W with it
  and every later Episode uses ``learning_rate`` — exactly the legacy's ``lr``
  reassignment quirk; ``null`` applies ``learning_rate`` from episode 1.
* **Evaluation blocks**: after every ``test_frequency`` episodes, the newest W
  is evaluated greedily over ``evaluation_seeds`` (generated fleet, default
  ``vehicles + 2`` action pool); the block with the lowest mean cost pins
  ``best_w``, mirroring ``Q_pred`` / ``Best_W``. The legacy's ``Q_pred`` starts
  at 1e11; ``math.inf`` here is behavior-equivalent for any real cost.
* **Best-W fallback**: with fewer episodes than ``test_frequency`` the legacy
  would run its test with ``Best_W = []`` and crash; the Trainer falls back to
  the final trained W instead (documented deviation, same information).
* **Final test** (deduplicated, ticket 02, simulation-performance): each
  (action count, seed) pair runs a **single** evaluation episode. Every
  episode draws its own per-Episode Generators from its seed (ticket 13,
  ADR-0001 phase 2), so the legacy ``test_episodes`` repeats of the same
  episode were bit-identical and their mean equaled one episode's value —
  computed directly now instead of summed and divided (avoiding the sum/k
  division-order float noise that quirk could otherwise leak into the
  report). ``test_episodes`` stays in ``ExperimentConfig`` for config-file
  compatibility but is no longer read.
* **Reported metrics**: the ten golden-pinned Episode metrics (``unserved_clients``
  added by ticket 03, simulator-correctness, ADR-0004). The legacy report's
  three mean-time metrics (``mean_delay_time``, ``mean_earliness_time``,
  ``mean_overtime``) were not ported with the ticket 07 Model and are not pinned
  by the golden master (ADR-0001 ticket 09 addendum).

Parallelism (ticket 08, simulation-performance): the evaluation blocks and the
final test are batches of independent Episodes, so they go through
``stdvrp.training.episode_pool`` — a persistent pool of spawned workers, each
holding one world loaded from the ticket-03 binary cache. Results come back in
request order, so every reduction below sees the serial loops' seed order and
``results.json`` stays bit-identical. ``worker_count=1`` (the default) keeps
everything in this process. Training itself stays sequential: W is a serial
dependency.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure

from stdvrp.config import ExperimentConfig
from stdvrp.simulation import EpisodeResult, run_training_episode
from stdvrp.training.episode_pool import EpisodePool, EpisodeRequest, EpisodeWorld, W
from stdvrp.training.reference_card import ReferenceCard

if TYPE_CHECKING:
    # Ticket 07: torch is an optional extra (ticket 03) -- Trainer itself must
    # stay importable without it, so every neural-policy import that reaches
    # torch (transitively, via network.py) is deferred into the method bodies
    # that actually need it (train_neural, _run_neural_evaluation_block). Only
    # type hints reach these names, resolved lazily by ``from __future__ import
    # annotations`` above, never at import time.
    from stdvrp.training.neural_checkpoint import BestWeights
    from stdvrp.training.neural_episode import NeuralPolicyState
    from stdvrp.training.neural_report import EvaluationReport

EPISODE_METRICS = (
    "total_cost",
    "distance_cost",
    "delay_cost",
    "earliness_cost",
    "overtime_cost",
    "tau",
    "state_count",
    "delay_clients",
    "earliness_clients",
    "unserved_clients",
)


@dataclass(frozen=True, slots=True)
class EvaluationBlock:
    """One periodic evaluation: greedy episode costs with the newest W."""

    episodes_completed: int
    seed_costs: tuple[float, ...]

    @property
    def mean_cost(self) -> float:
        return sum(self.seed_costs) / len(self.seed_costs)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The training loop's outcome: W after every Episode plus the evaluations."""

    w_trajectory: tuple[W, ...]
    evaluations: tuple[EvaluationBlock, ...]
    # The last evaluated W and the best-evaluated W (legacy Newest_W / Best_W);
    # None when no evaluation block ran.
    newest_w: W | None
    best_w: W | None

    @property
    def best_mean_cost(self) -> float | None:
        """The best evaluation-block mean (the one that pinned ``best_w``)."""
        if not self.evaluations:
            return None
        return min(block.mean_cost for block in self.evaluations)


@dataclass(frozen=True, slots=True)
class SeedTestResult:
    """Final-test metrics for one seed: one evaluation episode's raw values.

    (Ticket 02, simulation-performance) The legacy ``test_episodes`` repeats of
    this episode were bit-identical, so their mean equaled this value anyway.
    """

    seed: int
    vehicle_count: int
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ActionCountReport:
    """The final test at one action-pool width: per-seed metrics and their spread."""

    action_count: int
    per_seed: tuple[SeedTestResult, ...]
    # metric name -> (mean, population std) across seeds, as the legacy reports.
    summary: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything one experiment run produced."""

    training: TrainingResult
    test: tuple[ActionCountReport, ...]
    tested_w: W


@dataclass(frozen=True, slots=True)
class NeuralTrainingResult:
    """Ticket 07: the neural training loop's outcome.

    Unlike :class:`ExperimentResult`, there is no final test here: the blocks
    below are paired against ``evaluation_seeds`` only, which select
    checkpoints and hyperparameters and are therefore contaminated for a
    verdict by construction (spec.md). ``test_seeds`` are touched only by
    tickets 08/09 — reading a verdict off ``evaluations`` here would be
    exactly the p-hacking spec.md's anti-p-hacking clause forbids.
    """

    episodes_completed: int
    converged: bool
    evaluations: tuple[EvaluationReport, ...]
    elapsed_seconds: float
    policy_state: NeuralPolicyState
    # The device this run's weights live on and every episode ran on (ticket
    # 12) -- resolved once (policy_state.device) and carried into the results
    # record, since "auto" means the config alone no longer pins it.
    device: str
    # ``None`` when the default salted first_train_seed derivation was used
    # (see train_neural's own docstring); the explicit value the caller passed
    # otherwise. Ticket 08 (Gate A) reads this back to confirm which of its
    # three reproducibility runs produced which trained network.
    init_seed: int | None = None
    # Ticket 16: the run stopped because a block's exclusion rate reached
    # EXCLUSION_RATE_STOP_THRESHOLD, distinct from both "converged" (True) and
    # "hit the safety cap" (both False) -- a caller needs to tell the three
    # apart without parsing the log. Never True at the same time as
    # ``converged``: the loop breaks on whichever of the two fires first.
    stopped_for_exclusion_rate: bool = False


class Trainer:
    """Runs the experiment an ExperimentConfig describes over a loaded world."""

    def __init__(
        self,
        world: EpisodeWorld,
        *,
        episode_pool: EpisodePool | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.world = world
        self.config = world.config
        # None (the default) evaluates in this process; a pool spreads the
        # evaluation blocks and the final test over worker processes (ticket 08).
        self.episode_pool = episode_pool
        self._log = log if log is not None else lambda message: None

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        cache_dir: Path | None = None,
        worker_count: int = 1,
        log: Callable[[str], None] | None = None,
    ) -> Trainer:
        """Load the world from the config's DataSource and wire the Trainer.

        ``cache_dir`` (ticket 03, simulation-performance) opts into the binary
        world cache: ``None`` (the default) parses the CSVs fresh every call,
        exactly as before; a directory reuses a matching prior snapshot instead
        of re-parsing, and writes one on a miss. See ``stdvrp.traffic.world_cache``.

        ``worker_count`` (ticket 08) is how many worker processes evaluate on: 1
        (the default) keeps every episode in this process, more than 1 needs
        ``cache_dir`` because every worker loads its own world through it. The
        results are identical either way — see ``stdvrp.training.episode_pool``.
        """
        # Before the (expensive) world load, so a bad worker/cache combination
        # fails immediately rather than after parsing the CSVs.
        episode_pool = EpisodePool.for_worker_count(
            config, cache_dir=cache_dir, worker_count=worker_count
        )
        world = EpisodeWorld.load(config, cache_dir=cache_dir)
        return cls(world, episode_pool=episode_pool, log=log)

    def close(self) -> None:
        """Shut the evaluation worker pool down; a no-op when running serially."""
        if self.episode_pool is not None:
            self.episode_pool.close()

    def __enter__(self) -> Trainer:
        """``run()`` closes its own pool; this is for driving ``train``/``final_test``
        directly, where nothing else would (benchmarks, tests, notebooks)."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def run(
        self, output_dir: Path, *, reference_card: ReferenceCard | None = None
    ) -> ExperimentResult:
        """Train, run the final test, and write results + plot into ``output_dir``.

        ``reference_card`` (ticket 01) is the linear baseline's frozen per-seed
        cost vector; when given, the training plot draws it as the comparison
        line ``config.static_policy_mean_cost`` used to be. ``None`` (the
        default) omits the line — what a run producing the card itself uses.
        """
        config = self.config
        try:
            training = self.train()
            tested_w = training.best_w if training.best_w is not None else training.w_trajectory[-1]
            test = self.final_test(tested_w)
        finally:
            # Every episode this run needs has been dispatched by now.
            self.close()
        result = ExperimentResult(training=training, test=test, tested_w=tested_w)

        output_dir.mkdir(parents=True, exist_ok=True)
        write_results(output_dir / "results.json", config, result)
        write_training_plot(output_dir / "training_plot.png", training.evaluations, reference_card)
        self._log(f"results written to {output_dir}")
        return result

    def train(self) -> TrainingResult:
        config = self.config
        w: W | None = None
        # Legacy warm-up quirk, now opt-in (ticket 12): the first Episode trains
        # with the warm-up rate when one is configured.
        learning_rate = (
            config.warmup_learning_rate
            if config.warmup_learning_rate is not None
            else config.learning_rate
        )
        w_trajectory: list[W] = []
        evaluations: list[EvaluationBlock] = []
        newest_w: W | None = None
        best_w: W | None = None
        best_mean_cost = math.inf

        for index in range(config.total_train_iterations):
            seed = config.first_train_seed + index
            result = run_training_episode(
                seed=seed,
                W=w,
                learning_rate=learning_rate,
                **self.world.episode_kwargs(),
            )
            learning_rate = config.learning_rate
            w = result.w
            w_trajectory.append(_copy_w(result.w))
            episodes_completed = index + 1
            self._log(
                f"train episode {episodes_completed} (seed {seed}) done   "
                f"cost {result.episode.total_cost:.4f}"
            )

            if episodes_completed % config.test_frequency == 0:
                newest_w = _copy_w(w)
                # One greedy Episode per evaluation seed: generated fleet, default
                # action pool. Batched so a worker pool can spread them (ticket 08);
                # results come back in ``evaluation_seeds`` order either way.
                episodes = self._run_evaluation_batch(
                    newest_w, tuple(EpisodeRequest(seed) for seed in config.evaluation_seeds)
                )
                block = EvaluationBlock(
                    episodes_completed=episodes_completed,
                    seed_costs=tuple(episode.total_cost for episode in episodes),
                )
                evaluations.append(block)
                is_new_best = block.mean_cost < best_mean_cost
                if is_new_best:
                    best_mean_cost = block.mean_cost
                    best_w = newest_w
                self._log(
                    f"evaluation after {episodes_completed} episodes: "
                    f"mean cost {block.mean_cost:.4f}"
                    f"{'  (new best)' if is_new_best else ''}"
                )
                self._log(f"  theta (this evaluation): {_format_w(newest_w)}")
                self._log(
                    f"  theta (best so far, mean cost {best_mean_cost:.4f}): "
                    f"{_format_w(best_w)}"
                )

        return TrainingResult(
            w_trajectory=tuple(w_trajectory),
            evaluations=tuple(evaluations),
            newest_w=newest_w,
            best_w=best_w,
        )

    def final_test(self, w: W) -> tuple[ActionCountReport, ...]:
        """The legacy ``test_model``: fixed seed/fleet tables at widening action pools.

        Ticket 02 (simulation-performance): runs each (action count, seed) pair
        **once**. With per-seed Generators (ticket 13), every one of the legacy
        ``test_episodes`` repeats was bit-identical, so the mean equaled a single
        episode's value — computed directly here rather than via a legacy
        sum/``test_episodes`` division that could round differently in its last
        bit. ``config.test_episodes`` is not read.

        Ticket 08: the whole table — every action count times every seed — is one
        batch, so a worker pool has the full 300-episode job to spread rather than
        one action count at a time. The results are sliced back apart below in
        exactly the nested order they were requested in.
        """
        config = self.config
        fleet = self._test_fleet()
        requests = self.final_test_requests()
        self._log(f"final test: {len(requests)} episodes")
        # One batch can take hours on the full dataset, and it is dispatched in
        # one go, so it reports every tenth of the way rather than going silent
        # until the per-action-count means below.
        step = max(1, len(requests) // 10)
        episodes = self._run_evaluation_batch(
            w,
            requests,
            on_progress=lambda done: (
                self._log(f"final test: {done}/{len(requests)} episodes")
                if done % step == 0
                else None
            ),
        )

        reports = []
        for index, action_count in enumerate(config.test_action_counts):
            row = episodes[index * len(fleet) : (index + 1) * len(fleet)]
            per_seed = tuple(
                SeedTestResult(
                    seed,
                    vehicle_count,
                    {name: float(getattr(episode, name)) for name in EPISODE_METRICS},
                )
                for (seed, vehicle_count), episode in zip(fleet, row, strict=True)
            )
            summary = {
                name: _mean_and_std([entry.metrics[name] for entry in per_seed])
                for name in EPISODE_METRICS
            }
            reports.append(ActionCountReport(action_count, per_seed, summary))
            self._log(
                f"final test actions={action_count}: mean cost {summary['total_cost'][0]:.4f}"
            )
        return tuple(reports)

    def final_test_requests(self) -> tuple[EpisodeRequest, ...]:
        """Every (action count, seed) cell of the final test, in submission order.

        The order is the contract :meth:`final_test` slices the results back apart
        with — action count outermost, the seed/fleet table inside — and the shape
        a benchmark needs to time the phase the way the experiment runs it.
        """
        config = self.config
        return tuple(
            EpisodeRequest(
                seed,
                vehicle_count=vehicle_count,
                number_actions_test=vehicle_count + action_count,
            )
            for action_count in config.test_action_counts
            for seed, vehicle_count in self._test_fleet()
        )

    def _test_fleet(self) -> tuple[tuple[int, int], ...]:
        """The final test's (seed, fleet size) table, paired once for both users."""
        return tuple(zip(self.config.test_seeds, self.config.test_vehicle_counts, strict=True))

    def _run_evaluation_batch(
        self,
        w: W,
        requests: tuple[EpisodeRequest, ...],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[EpisodeResult, ...]:
        """Run one batch of evaluation Episodes: on the worker pool, or right here.

        Both paths run the same ``EpisodeWorld.run_episodes`` and both return the
        results in *request* order, so callers reduce the serial loops' seed order
        whatever order the workers happen to finish in (ticket 08's Tier 1 gate).
        """
        if self.episode_pool is not None:
            return self.episode_pool.run(w, requests, on_progress=on_progress)
        return self.world.run_episodes(w, requests, on_progress=on_progress)

    # --- Ticket 07: the neural Policy's training loop ----------------------------
    #
    # Deliberately not built on ``self.episode_pool``/``_run_evaluation_batch``
    # above: those batch a fixed ``W`` array across worker processes (ticket 08,
    # simulation-performance), and the transformer's weights change every
    # episode. Broadcasting fresh weights to a worker pool every evaluation
    # block is a real, unsolved parallelism question of its own -- out of this
    # ticket's scope, so evaluation blocks run serially here, exactly like the
    # ``worker_count=1`` default everywhere else in this codebase.

    def train_neural(
        self,
        *,
        reference_card: ReferenceCard,
        checkpoint_path: Path,
        resume: bool = False,
        max_episodes: int | None = None,
        max_hours: float | None = None,
        evaluation_cadence_minimum: int | None = None,
        init_seed: int | None = None,
        train_encoder: bool = False,
    ) -> NeuralTrainingResult:
        """Ticket 07: train the transformer Policy to convergence, watchably and resumably.

        Every evaluation block prints a **paired** report against
        ``reference_card`` (seed-by-seed, never a difference of means) and is
        checkpointed atomically. Stops on convergence (patience -> lr cuts ->
        three cuts with no further improvement) or the hard safety cap (10 000
        episodes or 24 h — "not the budget"; if it fires, the run did not
        converge). ``resume=True`` continues an interrupted run from
        ``checkpoint_path`` — same trajectory, because every stochastic stream
        is re-derived from each episode's own seed (see
        ``neural_episode.py``'s module docstring), so nothing but the episode
        count needs restoring.

        The blocks read only ``evaluation_seeds`` (via ``reference_card`` and
        this run's own evaluation episodes) — never ``test_seeds``, which
        select nothing here and are reserved for tickets 08/09's verdict.

        ``max_episodes``/``max_hours``/``evaluation_cadence_minimum`` default
        to spec.md's frozen values (``None``) — the only reason to pass them
        is to make a test's safety cap or evaluation cadence reachable in a
        few episodes instead of thousands; a real training run should never
        set them.

        ``init_seed`` (ticket 08, Gate A): ``None`` (the default) keeps the
        existing salted derivation from ``config.first_train_seed`` below —
        every other call site's unchanged behavior. An explicit value seeds
        *only* the network's initial weights, holding ``first_train_seed``
        (and therefore the whole training episode sequence) fixed — the lever
        Gate A's reproducibility check needs to vary just the one thing "three
        independent network-init seeds" names, without also reshuffling which
        Episodes get trained on between runs. Reported back on the result
        (``NeuralTrainingResult.init_seed``) so a caller running several arms
        can tell which trained network came from which seed.

        ``train_encoder`` (ticket 17): ``False`` (the default) is the
        frozen-encoder arm ticket 16 shipped -- every other call site's
        unchanged behavior. ``True`` is the trained-encoder arm (spec.md's
        "Two timescales"): ``encoder``/``head.layer1`` additionally move by
        SGD every ``learn()`` call, on top of the ridge solve both arms share.
        Forwarded to :func:`~stdvrp.training.neural_episode.build_neural_policy_state`,
        which is the only place this actually changes anything (which
        parameters ``policy_state.optimizer`` covers).
        """
        # Deferred imports: torch is an optional extra (ticket 03); only
        # calling this method should ever require it (see the TYPE_CHECKING
        # block at the top of this module).
        from stdvrp.training.neural_checkpoint import (
            TrainingCheckpoint,
            load_checkpoint,
            save_checkpoint,
        )
        from stdvrp.training.neural_episode import build_neural_policy_state
        from stdvrp.training.neural_report import (
            MAX_EPISODES,
            MAX_HOURS,
            MIN_EVALUATION_CADENCE,
            ConvergenceAction,
            ConvergenceState,
            EvaluationReport,
            evaluation_cadence,
            exclusion_rate_stop_signal,
            format_episode_line,
            format_evaluation_block,
            format_lr,
            hit_safety_cap,
            update_convergence,
        )

        max_episodes = max_episodes if max_episodes is not None else MAX_EPISODES
        max_hours = max_hours if max_hours is not None else MAX_HOURS
        cadence_minimum = (
            evaluation_cadence_minimum
            if evaluation_cadence_minimum is not None
            else MIN_EVALUATION_CADENCE
        )

        config = self.config
        reference_by_seed = reference_card.evaluation_cost_by_seed()
        reference_seed_costs = tuple(reference_by_seed[seed] for seed in config.evaluation_seeds)

        # A fixed, documented, reproducible seed for the network's own weight
        # init -- distinct from every per-episode stream, which spawns from
        # *that episode's own* seed, never from this one (ticket 13 discipline).
        # Entropy is a two-element sequence, not the bare ``first_train_seed``
        # int every episode seed is (``first_train_seed + index``): a bare-int
        # SeedSequence's first spawned child is identical regardless of how
        # many children the call asks for, so ``SeedSequence(first_train_seed
        # + 0).spawn(4)[0]`` (episode 0's congestion stream) would otherwise be
        # bit-identical to this seed -- confirmed by measurement, not assumed.
        # The salt makes this entropy structurally unreachable from any plain
        # per-episode seed (verified over the safety cap's full 10 000-episode
        # range, all four spawned streams, zero collisions).
        _INIT_SEED_SALT = 0x4E4554494E49  # arbitrary, distinguishing only
        if init_seed is None:
            (default_init_seed,) = np.random.SeedSequence(
                [config.first_train_seed, _INIT_SEED_SALT]
            ).spawn(1)
            policy_state = build_neural_policy_state(
                config, np.random.default_rng(default_init_seed), train_encoder=train_encoder
            )
        else:
            policy_state = build_neural_policy_state(
                config, np.random.default_rng(init_seed), train_encoder=train_encoder
            )
        # Ticket 12: config.device may be "auto", so the config alone no
        # longer pins which device this run actually used -- print the
        # resolved device once, up front, so the run's own log records it.
        self._log(f"device: {policy_state.device}")
        convergence = ConvergenceState(current_lr=config.neural_learning_rate)
        evaluations: list[EvaluationReport] = []
        episodes_completed = 0
        elapsed_seconds = 0.0
        # The weights of the best evaluation block, kept aside so the network
        # this returns is the one that measured best -- not whichever one the
        # last training episode left behind. ``update_convergence`` already
        # tracks *that* a block was the best; nothing used to snapshot the
        # network at it, so ticket 08's Gate A measured a final network its own
        # patience machinery had spent 750 episodes declining to improve on.
        best_weights: BestWeights | None = None

        if resume and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, policy_state)
            episodes_completed = checkpoint.episodes_completed
            elapsed_seconds = checkpoint.elapsed_seconds
            convergence = checkpoint.convergence
            evaluations = list(checkpoint.evaluations)
            best_weights = checkpoint.best_weights
            self._log(
                f"resumed from {checkpoint_path}: {episodes_completed} episodes completed, "
                f"lr={format_lr(convergence.current_lr)}"
            )

        session_start = time.monotonic()

        def total_elapsed() -> float:
            return elapsed_seconds + (time.monotonic() - session_start)

        next_eval_at = episodes_completed + evaluation_cadence(
            episodes_completed, minimum=cadence_minimum
        )
        converged = False
        # Ticket 16: the ridge estimator's per-block exclusion count/rate is a
        # *delta* against the previous block (spec.md's live report is about
        # this Episode window, not the whole run so far), so both cumulative
        # counters are snapshotted here -- at the run's start, or wherever a
        # resumed run's policy_state.ridge already stands -- and re-snapshotted
        # after every block below.
        episodes_at_last_block = episodes_completed
        excluded_at_last_block = policy_state.ridge.episodes_excluded
        stopped_for_exclusion_rate = False

        try:
            while not hit_safety_cap(
                episodes_completed=episodes_completed,
                elapsed_seconds=total_elapsed(),
                max_episodes=max_episodes,
                max_hours=max_hours,
            ):
                seed = config.first_train_seed + episodes_completed
                episode_start = time.monotonic()
                result, loss = self._run_neural_training_episode(seed, policy_state)
                episode_wall = time.monotonic() - episode_start
                episodes_completed += 1
                self._log(
                    format_episode_line(
                        episode=episodes_completed,
                        seed=seed,
                        cost=result.total_cost,
                        loss=loss,
                        lr=policy_state.current_lr,
                        wall_clock_seconds=episode_wall,
                    )
                )

                if episodes_completed < next_eval_at:
                    continue

                seed_costs, spread_samples = self._run_neural_evaluation_block(policy_state)
                report = EvaluationReport(
                    episodes_completed=episodes_completed,
                    seed_costs=seed_costs,
                    reference_seed_costs=reference_seed_costs,
                    training_episodes=episodes_completed - episodes_at_last_block,
                    excluded_episodes=policy_state.ridge.episodes_excluded - excluded_at_last_block,
                    spread_samples=spread_samples,
                )
                episodes_at_last_block = episodes_completed
                excluded_at_last_block = policy_state.ridge.episodes_excluded
                evaluations.append(report)
                improved = report.mean_cost < convergence.best_mean_cost
                action = update_convergence(convergence, report)
                self._log(format_evaluation_block(report, convergence))

                if improved:
                    best_weights = {
                        "encoder_state": copy.deepcopy(policy_state.encoder.state_dict()),
                        "head_state": copy.deepcopy(policy_state.head.state_dict()),
                    }

                if action == ConvergenceAction.REDUCE_LR:
                    policy_state.current_lr = convergence.current_lr
                    self._log(
                        f"PATIENCE EXCEEDED at episode {episodes_completed}: lr -> "
                        f"{format_lr(convergence.current_lr)} (reduction "
                        f"{convergence.reductions_applied}/3)"
                    )
                elif action == ConvergenceAction.CONVERGED:
                    converged = True
                    self._log(
                        f"CONVERGED at episode {episodes_completed}: no improvement since "
                        f"episode {convergence.best_episode} ({convergence.best_delta_pct:+.1f}%) "
                        "after 3 lr reductions"
                    )

                if exclusion_rate_stop_signal(report):
                    stopped_for_exclusion_rate = True
                    self._log(
                        f"EXCLUSION RATE {report.exclusion_rate * 100:.1f}% AT episode "
                        f"{episodes_completed}: stopping -- the ridge estimator excludes "
                        "aborted Episodes from its accumulator, so a policy that starts "
                        "aborting degrades silently (ticket 16). Not reported as converged."
                    )

                elapsed_seconds = total_elapsed()
                session_start = time.monotonic()
                save_checkpoint(
                    checkpoint_path,
                    TrainingCheckpoint(
                        episodes_completed=episodes_completed,
                        elapsed_seconds=elapsed_seconds,
                        convergence=convergence,
                        evaluations=tuple(evaluations),
                        best_weights=best_weights,
                    ),
                    policy_state,
                )

                if converged or stopped_for_exclusion_rate:
                    break
                next_eval_at = episodes_completed + evaluation_cadence(
                    episodes_completed, minimum=cadence_minimum
                )
            else:
                self._log(
                    f"SAFETY CAP REACHED at episode {episodes_completed} "
                    f"({total_elapsed() / 3600:.1f}h) -- run did NOT converge"
                )
        except KeyboardInterrupt:
            last_checkpoint_episode = evaluations[-1].episodes_completed if evaluations else 0
            self._log(
                f"INTERRUPTED at episode {episodes_completed}; {checkpoint_path} holds the "
                f"last complete checkpoint (episode {last_checkpoint_episode}) and is safe "
                "to resume"
            )
            raise

        # Hand back the best-measured network, not the last one trained. The
        # trajectory's final weights stay in the checkpoint for a resume; what
        # ticket 08/09 measure is this.
        if best_weights is not None:
            policy_state.encoder.load_state_dict(best_weights["encoder_state"])
            policy_state.head.load_state_dict(best_weights["head_state"])
            self._log(
                f"restored the best evaluated network (episode {convergence.best_episode}, "
                f"{convergence.best_delta_pct:+.1f}% vs the reference card) for measurement"
            )

        return NeuralTrainingResult(
            episodes_completed=episodes_completed,
            converged=converged,
            evaluations=tuple(evaluations),
            elapsed_seconds=total_elapsed(),
            policy_state=policy_state,
            device=str(policy_state.device),
            init_seed=init_seed,
            stopped_for_exclusion_rate=stopped_for_exclusion_rate,
        )

    def _run_neural_training_episode(
        self, seed: int, policy_state: NeuralPolicyState
    ) -> tuple[EpisodeResult, float]:
        from stdvrp.training.neural_episode import run_neural_training_episode

        return run_neural_training_episode(
            seed=seed,
            client_generator=self.world.client_generator,
            travel_time_model=self.world.travel_time_model,
            shortest_path_cache=self.world.shortest_path_cache,
            congestion_generator=self.world.congestion_generator,
            policy_state=policy_state,
            config=self.config,
        )

    def _run_neural_evaluation_block(
        self, policy_state: NeuralPolicyState
    ) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
        """Greedy evaluation over every ``evaluation_seeds`` seed, in that order.

        Also pools every seed's candidate-spread samples (ticket 17) for the
        block report's ``r`` diagnostic -- see
        :func:`~stdvrp.training.neural_episode.run_neural_evaluation_episode`.
        """
        from stdvrp.training.neural_episode import run_neural_evaluation_episode

        costs = []
        spread_samples: list[tuple[float, float]] = []
        for seed in self.config.evaluation_seeds:
            result, seed_spread_samples = run_neural_evaluation_episode(
                seed=seed,
                client_generator=self.world.client_generator,
                travel_time_model=self.world.travel_time_model,
                shortest_path_cache=self.world.shortest_path_cache,
                congestion_generator=self.world.congestion_generator,
                policy_state=policy_state,
                config=self.config,
            )
            costs.append(result.total_cost)
            spread_samples.extend(seed_spread_samples)
        return tuple(costs), tuple(spread_samples)


def _copy_w(w: W) -> W:
    return np.array(w, dtype=np.float64, copy=True)


def _format_w(w: W | None) -> str:
    """Render a weight vector for the log: one line, fixed precision, no truncation."""
    if w is None:
        return "unset"
    return "[" + ", ".join(f"{value:.6g}" for value in w) + "]"


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def config_as_json(config: ExperimentConfig) -> dict[str, Any]:
    """The config as JSON-serializable values (Paths to str, tuples to lists)."""
    document = dataclasses.asdict(config)
    document["data_dir"] = str(config.data_dir)
    return {
        name: list(value) if isinstance(value, tuple) else value for name, value in document.items()
    }


def write_results(path: Path, config: ExperimentConfig, result: ExperimentResult) -> None:
    """Write results.json: the config snapshot plus everything the run produced."""
    document = {
        "config": config_as_json(config),
        "training": {
            "w_trajectory": [_w_as_json(w) for w in result.training.w_trajectory],
            "evaluations": [
                {
                    "episodes_completed": block.episodes_completed,
                    "seed_costs": list(block.seed_costs),
                    "mean_cost": block.mean_cost,
                }
                for block in result.training.evaluations
            ],
            "newest_w": _w_as_json(result.training.newest_w),
            "best_w": _w_as_json(result.training.best_w),
        },
        "tested_w": _w_as_json(result.tested_w),
        "test": {
            str(report.action_count): {
                "per_seed": [
                    {"seed": entry.seed, "vehicles": entry.vehicle_count, **entry.metrics}
                    for entry in report.per_seed
                ],
                "summary": {
                    name: {"mean": mean, "std": std} for name, (mean, std) in report.summary.items()
                },
            }
            for report in result.test
        },
    }
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")


def _w_as_json(w: W | None) -> list[float] | None:
    return None if w is None else [float(x) for x in w]


def write_training_plot(
    path: Path, evaluations: tuple[EvaluationBlock, ...], reference_card: ReferenceCard | None
) -> None:
    """The legacy training plot: evaluation means with the linear-baseline comparison.

    Rendered through a directly constructed Figure (no pyplot): backend-independent,
    headless-safe, and free of pyplot's global figure registry.

    ``reference_card`` (ticket 01) replaces the legacy's hardcoded
    ``static_policy_mean_cost`` scalar: the red line is now the frozen linear
    ``MonteCarloPolicy``'s mean cost over ``evaluation_seeds`` — the same seeds
    this plot's own evaluation blocks use — rather than an unpaired constant.
    A shaded band one population standard deviation either side of that mean
    (ticket 07) draws the reference's per-seed **spread**, not only its mean —
    this cost distribution's variance is wide enough that a bare mean line
    understates how much a block's own mean can move by chance alone.
    """
    figure = Figure(figsize=(20, 5))
    axes = figure.subplots()
    axes.plot(
        [block.episodes_completed for block in evaluations],
        [block.mean_cost for block in evaluations],
        marker="o",
        linestyle="-",
        label="Cost",
    )
    if reference_card is not None:
        reference_mean = reference_card.evaluation_mean_cost
        reference_std = float(np.std(reference_card.evaluation_seed_costs))
        axes.axhline(
            y=reference_mean,
            color="red",
            linestyle=":",
            label="Linear baseline (reference card)",
        )
        if reference_std > 0:
            axes.axhspan(
                reference_mean - reference_std,
                reference_mean + reference_std,
                color="red",
                alpha=0.1,
                label="_nolegend_",
            )
    axes.set_title("Objective Function under Greedy Policy during Training")
    axes.set_xlabel("Number of Episodes")
    axes.set_ylabel("Objective Function")
    axes.legend()
    # The legacy forces scientific notation at 10^3 on both axes.
    for axis in (axes.xaxis, axes.yaxis):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((3, 3))
        axis.set_major_formatter(formatter)
    figure.savefig(path)
