"""Gate A -- does the transformer Policy learn at all (ticket 08, neural-policy).

The hard landing gate spec.md and the ticket define, independent of whether
the transformer beats the linear baseline (Gate B, tickets 09/10): trained vs.
**the same architecture untrained** (ticket 05's myopic warm start -- a
nearest-feasible-Client-or-depot rival, not noise), on the held-out
``test_seeds``, never on ``evaluation_seeds`` (those pick checkpoints and
hyperparameters and are contaminated by construction).

Three parts, all required:

1. **Null model** -- paired Wilcoxon signed-rank test, trained vs. untrained
   ``total_cost`` over every ``test_seeds`` entry: p < 0.05 *and* >= 5% mean
   cost reduction (:data:`GATE_A_SIGNIFICANCE_THRESHOLD`,
   :data:`GATE_A_EFFECT_THRESHOLD_PCT`).
2. **Reproducibility** -- >= 3 independent network-init seeds
   (:data:`GATE_A_MIN_INIT_SEEDS`); the per-arm mean reduction is reported as
   mean +/- sd, and the gate additionally fails if that spread straddles zero
   even when every individual arm cleared its own bar (ticket 08: "if the
   spread straddles zero, it did not learn, whatever the best run says").
3. **Calibration** -- Spearman rho(predicted Q, realised ``U_t``) over every
   ``(t, vehicle)`` decision sample pooled across every ``test_seeds`` episode:
   >= 0.5 (:data:`GATE_A_CALIBRATION_THRESHOLD`). Spearman, not Pearson: the
   cost distribution's right tail is brutal (research note F10).

Torch-free at module scope, exactly like ``trainer.py``: every import that
reaches torch (``neural_episode.py``, transitively ``network.py``) is deferred
into the functions that actually run a network, so this module (and its
statistics) stay importable and unit-testable without the ``neural`` extra.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from stdvrp.training.neural_report import paired_wilcoxon_p
from stdvrp.training.reference_card import ReferenceCard
from stdvrp.training.trainer import Trainer

if TYPE_CHECKING:
    from stdvrp.training.episode_pool import EpisodeWorld
    from stdvrp.training.neural_episode import NeuralPolicyState

__all__ = [
    "GATE_A_CALIBRATION_THRESHOLD",
    "GATE_A_EFFECT_THRESHOLD_PCT",
    "GATE_A_MIN_INIT_SEEDS",
    "GATE_A_SIGNIFICANCE_THRESHOLD",
    "ArmResult",
    "GateAResult",
    "format_gate_a_report",
    "run_gate_a",
]

#: Null-model effect size Gate A requires: mean cost reduction, trained vs.
#: untrained, over test_seeds (spec.md's "Frozen parameters" table).
GATE_A_EFFECT_THRESHOLD_PCT = 5.0

#: Wilcoxon signed-rank significance threshold for the null-model test.
GATE_A_SIGNIFICANCE_THRESHOLD = 0.05

#: Minimum acceptable Spearman rho(predicted Q, realised U_t) on a trained network.
GATE_A_CALIBRATION_THRESHOLD = 0.5

#: Reproducibility needs at least this many independent network-init seeds.
GATE_A_MIN_INIT_SEEDS = 3


def _spearman(pairs: tuple[tuple[float, float], ...]) -> float:
    """Spearman rho over ``(predicted, realised)`` pairs; NaN below 2 samples.

    Also NaN when either side is constant (``scipy`` warns and returns NaN
    itself in that case) -- reported as such, matching this module's own
    Wilcoxon helper's "NaN rather than crash" discipline.
    """
    if len(pairs) < 2:
        return float("nan")
    predicted = [p for p, _ in pairs]
    realised = [u for _, u in pairs]
    result = stats.spearmanr(predicted, realised)
    return float(result.statistic)


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One network-init seed's complete Gate A measurement.

    ``trained_calibration``/``untrained_calibration`` are ``(Q_predicted,
    U_t)`` pairs pooled across every seed's decision epochs and vehicles
    (:meth:`~stdvrp.policies.transformer_policy.TransformerMonteCarloPolicy.calibration_pairs`).
    """

    init_seed: int
    trained_seed_costs: tuple[float, ...]
    untrained_seed_costs: tuple[float, ...]
    trained_calibration: tuple[tuple[float, float], ...]
    untrained_calibration: tuple[tuple[float, float], ...]
    episodes_completed: int
    converged: bool

    def __post_init__(self) -> None:
        if len(self.trained_seed_costs) != len(self.untrained_seed_costs):
            raise ValueError(
                "trained_seed_costs and untrained_seed_costs must be the same paired length"
            )
        if not self.trained_seed_costs:
            raise ValueError("an arm must measure at least one seed")

    @property
    def seed_reductions_pct(self) -> tuple[float, ...]:
        """Per-seed percent cost reduction, trained vs. untrained; positive improves."""
        return tuple(
            100.0 * (untrained - trained) / untrained
            for trained, untrained in zip(
                self.trained_seed_costs, self.untrained_seed_costs, strict=True
            )
        )

    @property
    def mean_reduction_pct(self) -> float:
        return float(np.mean(self.seed_reductions_pct))

    @property
    def median_reduction_pct(self) -> float:
        """Report alongside :attr:`mean_reduction_pct` -- the right-tailed cost
        distribution (F10) makes the two genuinely differ, and the difference
        is itself informative (ticket 08's own work item)."""
        return float(np.median(self.seed_reductions_pct))

    @property
    def wilcoxon_p(self) -> float:
        """Two-sided paired Wilcoxon signed-rank p-value, trained vs. untrained."""
        return paired_wilcoxon_p(self.trained_seed_costs, self.untrained_seed_costs)

    @property
    def null_model_passes(self) -> bool:
        """This arm alone: significant *and* at least the required effect size."""
        p = self.wilcoxon_p
        return (
            p == p  # not NaN
            and p < GATE_A_SIGNIFICANCE_THRESHOLD
            and self.mean_reduction_pct >= GATE_A_EFFECT_THRESHOLD_PCT
        )

    @property
    def trained_calibration_spearman(self) -> float:
        return _spearman(self.trained_calibration)

    @property
    def untrained_calibration_spearman(self) -> float:
        """Expected to be ~0 (spec.md) -- the untrained network's Q never
        varies with the realised return, only with static travel time."""
        return _spearman(self.untrained_calibration)

    @property
    def calibration_passes(self) -> bool:
        rho = self.trained_calibration_spearman
        return rho == rho and rho >= GATE_A_CALIBRATION_THRESHOLD  # not NaN


@dataclass(frozen=True, slots=True)
class GateAResult:
    """The full reproducibility protocol: every arm, plus the combined verdict."""

    arms: tuple[ArmResult, ...]

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("Gate A needs at least one arm")

    @property
    def mean_reductions_pct(self) -> tuple[float, ...]:
        """One number per arm -- the distribution reproducibility is measured over."""
        return tuple(arm.mean_reduction_pct for arm in self.arms)

    @property
    def reproducibility_mean_pct(self) -> float:
        return float(np.mean(self.mean_reductions_pct))

    @property
    def reproducibility_sd_pct(self) -> float:
        return float(np.std(self.mean_reductions_pct))

    @property
    def reproducibility_spread_straddles_zero(self) -> bool:
        """``mean +/- sd`` crossing zero means "did not learn", even if every
        individual arm cleared its own bar (ticket 08's own wording)."""
        mean = self.reproducibility_mean_pct
        sd = self.reproducibility_sd_pct
        return (mean - sd) <= 0.0 <= (mean + sd)

    @property
    def null_model_passes(self) -> bool:
        """Every arm individually passes, and there are enough of them to call
        it reproducibility rather than one lucky init."""
        return len(self.arms) >= GATE_A_MIN_INIT_SEEDS and all(
            arm.null_model_passes for arm in self.arms
        )

    @property
    def reproducibility_passes(self) -> bool:
        return self.null_model_passes and not self.reproducibility_spread_straddles_zero

    @property
    def calibration_passes(self) -> bool:
        return all(arm.calibration_passes for arm in self.arms)

    @property
    def passes(self) -> bool:
        """All three parts of the gate (spec.md's acceptance table), combined."""
        return self.reproducibility_passes and self.calibration_passes


def _evaluate_over_seeds(
    world: EpisodeWorld, policy_state: NeuralPolicyState, seeds: tuple[int, ...]
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    """Greedy cost + pooled calibration pairs for one network, over ``seeds``."""
    from stdvrp.training.neural_episode import run_neural_calibration_episode

    costs: list[float] = []
    pairs: list[tuple[float, float]] = []
    for seed in seeds:
        result, calibration = run_neural_calibration_episode(
            seed=seed,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=policy_state,
            config=world.config,
        )
        costs.append(result.total_cost)
        pairs.extend(calibration)
    return tuple(costs), tuple(pairs)


def run_gate_a(
    world: EpisodeWorld,
    *,
    reference_card: ReferenceCard,
    checkpoint_dir: Path,
    init_seeds: tuple[int, ...] = (0, 1, 2),
    test_seeds: tuple[int, ...] | None = None,
    untrained_init_seed: int = 0,
    max_episodes: int | None = None,
    max_hours: float | None = None,
    evaluation_cadence_minimum: int | None = None,
    log: Callable[[str], None] | None = None,
) -> GateAResult:
    """Run the whole Gate A protocol: train ``len(init_seeds)`` networks to
    convergence, measure each against one shared untrained (null) network.

    ``test_seeds`` defaults to ``world.config.test_seeds`` -- the verdict set,
    never ``evaluation_seeds`` (spec.md's anti-contamination rule; those only
    ever feed ``reference_card``'s live report during training below).

    The untrained network is built **once**, not once per arm: ticket 05's
    myopic warm start makes its greedy decisions -- and therefore its cost and
    calibration numbers -- exactly independent of ``init_rng`` (the random
    background weights the warm start zeroes out of the forward pass at
    construction; ``test_network.py``/``test_transformer_policy.py`` pin this
    numerically). Recomputing it per arm would only repeat the same
    measurement at extra wall-clock cost, not add evidence.

    Each arm's own checkpoint lands at ``checkpoint_dir / f"gate_a_init{seed}.pt"``.
    ``max_episodes``/``max_hours``/``evaluation_cadence_minimum`` default to
    spec.md's frozen convergence parameters (``None``, forwarded to
    :meth:`~stdvrp.training.trainer.Trainer.train_neural`); pass them only to
    make a dev run's cap reachable quickly, never for the real gate.
    """
    from stdvrp.training.neural_episode import build_neural_policy_state

    config = world.config
    seeds = test_seeds if test_seeds is not None else config.test_seeds
    emit = log if log is not None else lambda _message: None

    emit(f"Gate A: building the shared untrained (null) network, {len(seeds)} test seeds")
    untrained_state = build_neural_policy_state(config, np.random.default_rng(untrained_init_seed))
    untrained_costs, untrained_pairs = _evaluate_over_seeds(world, untrained_state, seeds)
    emit(f"Gate A: untrained mean cost {float(np.mean(untrained_costs)):.4f}")

    arms = []
    for init_seed in init_seeds:
        emit(f"Gate A arm: init_seed={init_seed} -- training to convergence")
        trainer = Trainer(world, log=log)
        training = trainer.train_neural(
            reference_card=reference_card,
            checkpoint_path=checkpoint_dir / f"gate_a_init{init_seed}.pt",
            max_episodes=max_episodes,
            max_hours=max_hours,
            evaluation_cadence_minimum=evaluation_cadence_minimum,
            init_seed=init_seed,
        )
        trained_costs, trained_pairs = _evaluate_over_seeds(world, training.policy_state, seeds)
        arm = ArmResult(
            init_seed=init_seed,
            trained_seed_costs=trained_costs,
            untrained_seed_costs=untrained_costs,
            trained_calibration=trained_pairs,
            untrained_calibration=untrained_pairs,
            episodes_completed=training.episodes_completed,
            converged=training.converged,
        )
        arms.append(arm)
        emit(
            f"Gate A arm init_seed={init_seed}: {training.episodes_completed} episodes "
            f"({'converged' if training.converged else 'DID NOT CONVERGE'}), "
            f"mean reduction {arm.mean_reduction_pct:+.1f}% "
            f"(median {arm.median_reduction_pct:+.1f}%), "
            f"Wilcoxon p={arm.wilcoxon_p:.4g}, "
            f"calibration rho={arm.trained_calibration_spearman:.3f}"
        )

    return GateAResult(arms=tuple(arms))


def format_gate_a_report(result: GateAResult) -> str:
    """The three-part report ticket 08's acceptance asks to be recorded, whatever it says."""
    lines = ["Gate A -- does it learn?", "=" * 60]
    for arm in result.arms:
        lines.append(
            f"  init_seed={arm.init_seed}: {arm.episodes_completed} episodes "
            f"({'converged' if arm.converged else 'DID NOT CONVERGE'})"
        )
        lines.append(
            f"    reduction: mean {arm.mean_reduction_pct:+.2f}%  "
            f"median {arm.median_reduction_pct:+.2f}%  "
            f"Wilcoxon p={arm.wilcoxon_p:.4g}  "
            f"{'PASS' if arm.null_model_passes else 'FAIL'}"
        )
        lines.append(
            f"    calibration: trained rho={arm.trained_calibration_spearman:+.3f}  "
            f"untrained rho={arm.untrained_calibration_spearman:+.3f}  "
            f"{'PASS' if arm.calibration_passes else 'FAIL'}"
        )
    lines.append("-" * 60)
    lines.append(
        f"1. Null model:       {'PASS' if result.null_model_passes else 'FAIL'} "
        f"(>= {GATE_A_MIN_INIT_SEEDS} arms, each p<{GATE_A_SIGNIFICANCE_THRESHOLD} "
        f"and >= {GATE_A_EFFECT_THRESHOLD_PCT:.0f}% reduction)"
    )
    spread_note = (
        "straddles zero" if result.reproducibility_spread_straddles_zero else "consistent sign"
    )
    lines.append(
        f"2. Reproducibility:  {'PASS' if result.reproducibility_passes else 'FAIL'} "
        f"(mean {result.reproducibility_mean_pct:+.2f}% "
        f"+/- sd {result.reproducibility_sd_pct:.2f}%, {spread_note})"
    )
    lines.append(
        f"3. Calibration:      {'PASS' if result.calibration_passes else 'FAIL'} "
        f"(>= {GATE_A_CALIBRATION_THRESHOLD} required)"
    )
    lines.append("=" * 60)
    lines.append(f"GATE A: {'PASS' if result.passes else 'FAIL'}")
    return "\n".join(lines)
